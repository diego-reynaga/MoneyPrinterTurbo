"""Caché en disco de los resultados de búsqueda de material en línea."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

from loguru import logger

from app.models.schema import MaterialInfo, VideoAspect
from app.utils import utils


MATERIAL_SEARCH_CACHE_TTL_SECONDS = 24 * 60 * 60
_CACHE_FORMAT_VERSION = 2
_CACHE_CLEANUP_INTERVAL_SECONDS = 60 * 60
_CACHE_FILE_PATTERN = re.compile(r"^[0-9a-f]{64}\.json$")
# Al guardar la caché primero se escribe en NamedTemporaryFile(delete=False) y luego se
# publica con os.replace. Cuando el proceso se termina a la fuerza (Ctrl+C, parada del
# contenedor, corte de luz), el respaldo de excepciones de Python no llega a ejecutarse
# y queda un archivo intermedio sin sustituir; no coincide con el patrón de archivos de
# caché de arriba, así que hay que identificarlo aparte para recuperarlo, o se
# acumularía para siempre en el directorio de caché.
_CACHE_TEMP_FILE_PATTERN = re.compile(r"^\.[0-9a-f]{64}-[a-z0-9_]+\.tmp$")

# La API permite por defecto varias tareas de vídeo concurrentes. Un número fijo de
# fragmentos de bloqueo hace que las búsquedas con las mismas condiciones compartan un
# bloqueo, sin guardar para siempre un Lock por palabra clave, lo que haría crecer la
# memoria sin parar. Solo se encarga de agrupar las solicitudes concurrentes dentro del
# proceso actual; la integridad de las escrituras entre procesos la siguen garantizando
# los archivos temporales y os.replace.
_CACHE_LOCKS = tuple(threading.Lock() for _ in range(256))
_cleanup_state_lock = threading.Lock()
_last_cleanup_monotonic: float | None = None


def _safe_public_url(value) -> str | None:
    """Quita los parámetros de consulta y las credenciales de usuario de las URL de páginas
    públicas, para que la caché no guarde un token por accidente.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _cached_source_info(item: MaterialInfo) -> dict | None:
    """
    Construye con una lista blanca la información de origen que se puede guardar en
    disco.

    La palabra clave de búsqueda ya forma parte de la clave de caché, así que no se
    escribe en claro en el contenido; al leer se recupera de los parámetros de la
    llamada. La URL de descarga se guarda aparte en ``MaterialInfo.url``; aquí solo se
    permiten la página pública del material, la página pública del autor y los
    identificadores de negocio estables, para que campos de extensión arbitrarios no
    entren en la caché en disco.
    """
    source = item.source_info
    if not isinstance(source, dict) or not source:
        return None

    cached: dict = {
        "provider": str(source.get("provider") or item.provider),
    }
    asset_id = source.get("asset_id")
    source_page = _safe_public_url(source.get("source_page"))
    if asset_id not in (None, ""):
        cached["asset_id"] = str(asset_id)
    if source_page:
        cached["source_page"] = source_page

    raw_creator = source.get("creator")
    if isinstance(raw_creator, dict):
        creator = {}
        creator_id = raw_creator.get("id")
        creator_name = raw_creator.get("name")
        creator_page = _safe_public_url(raw_creator.get("profile_page"))
        if creator_id not in (None, ""):
            creator["id"] = str(creator_id)
        if creator_name not in (None, ""):
            creator["name"] = str(creator_name)
        if creator_page:
            creator["profile_page"] = creator_page
        if creator:
            cached["creator"] = creator

    raw_rendition = source.get("rendition")
    if isinstance(raw_rendition, dict):
        rendition = {}
        for field in ("id", "width", "height"):
            value = raw_rendition.get(field)
            if value not in (None, ""):
                rendition[field] = str(value) if field == "id" else value
        if rendition:
            cached["rendition"] = rendition
    return cached


def _cache_dir() -> Path:
    """
    Devuelve el directorio de caché de búsqueda de material que comparten todas las
    entradas de ejecución.

    La caché debe estar bajo ``storage`` y no en la sesión de la WebUI ni en la memoria
    del proceso, para que la WebUI, la API, la CLI y las tareas tras reiniciar Docker
    reutilicen el mismo resultado.
    """
    return Path(utils.storage_dir("cache_material_search", create=True))


def _cache_key(
    provider: str,
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect | str,
) -> str:
    """
    Genera un nombre de archivo estable a partir de los parámetros de negocio que
    afectan a los resultados de búsqueda.

    La clave API solo sirve para autenticar y no afecta a los resultados de búsqueda
    públicos, así que no puede escribirse en la clave ni en el contenido de la caché.
    Usar SHA-256 evita que la palabra clave aparezca en el nombre del archivo y mantiene
    fija la longitud de la ruta.
    """
    aspect_value = getattr(video_aspect, "value", video_aspect)
    cache_key = json.dumps(
        {
            "provider": str(provider).strip().lower(),
            "search_term": str(search_term).strip(),
            "minimum_duration": int(minimum_duration),
            "video_aspect": str(aspect_value),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()


def _cache_path(
    provider: str,
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect | str,
) -> Path:
    digest = _cache_key(
        provider=provider,
        search_term=search_term,
        minimum_duration=minimum_duration,
        video_aspect=video_aspect,
    )
    return _cache_dir() / f"{digest}.json"


def get_material_search_cache_lock(
    provider: str,
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect | str,
) -> threading.Lock:
    """Devuelve el fragmento de bloqueo del proceso que corresponde a las condiciones de
    búsqueda actuales.
    """
    digest = _cache_key(
        provider=provider,
        search_term=search_term,
        minimum_duration=minimum_duration,
        video_aspect=video_aspect,
    )
    return _CACHE_LOCKS[int(digest[:8], 16) % len(_CACHE_LOCKS)]


def _is_same_cache_file(cache_path: Path, expected: os.stat_result) -> bool:
    """Evita borrar un archivo que se haya sustituido desde la instantánea del
    análisis/lectura.
    """
    try:
        current = cache_path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    return stat.S_ISREG(current.st_mode) and (
        current.st_dev, current.st_ino, current.st_mtime_ns, current.st_size
    ) == (expected.st_dev, expected.st_ino, expected.st_mtime_ns, expected.st_size)


def _remove_invalid_cache(
    cache_path: Path, expected: os.stat_result | None = None
) -> None:
    """Borra un archivo de caché caducado o que no se puede analizar; si falla, no afecta
    al flujo principal de búsqueda de material.
    """
    try:
        if expected is not None and not _is_same_cache_file(cache_path, expected):
            return
        cache_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(
            f"no se pudo eliminar la caché de búsqueda de material no válida: "
            f"file={cache_path.name}, error={exc}"
        )


def load_material_search_cache(
    provider: str,
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect | str,
    *,
    now: float | None = None,
) -> list[MaterialInfo] | None:
    """
    Lee los resultados de búsqueda de material que siguen dentro de la validez de 24
    horas.

    ``None`` significa fallo de caché y hay que pedir a la API remota; una lista vacía
    no se devuelve como caché válida, para que un error de red o un fallo del proveedor
    guardados por error no sigan bloqueando las tareas posteriores.
    """
    if str(provider).strip().lower() == "coverr":
        # Las direcciones de descarga de Coverr contienen un JWT firmado vinculado a la
        # clave API. Solo sirve para la solicitud actual y no puede entrar en la caché
        # en disco; al consultar con las mismas condiciones se borra de paso la caché
        # que pudieran dejar versiones antiguas.
        try:
            _remove_invalid_cache(
                _cache_path(
                    provider=provider,
                    search_term=search_term,
                    minimum_duration=minimum_duration,
                    video_aspect=video_aspect,
                )
            )
        except Exception as exc:
            logger.warning(
                "no se pudo eliminar la caché de búsqueda de material de Coverr desactivada: "
                f"error={type(exc).__name__}, detail={exc}"
            )
        return None

    try:
        cache_path = _cache_path(
            provider=provider,
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )
    except Exception as exc:
        # Las excepciones al crear el directorio de caché, resolver rutas, etc. no
        # pueden bloquear la búsqueda de material remota. Aquí se conservan el tipo y el
        # mensaje completos de la excepción para localizar problemas de permisos o de
        # montaje, y se sigue el flujo principal como fallo de caché.
        logger.warning(
            "no se pudo preparar la caché de búsqueda de material: "
            f"operation=read, error={type(exc).__name__}, detail={exc}"
        )
        return None
    try:
        stat_result = cache_path.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning(
            f"no se pudo inspeccionar la caché de búsqueda de material: "
            f"file={cache_path.name}, error={exc}"
        )
        return None

    current_time = time.time() if now is None else now
    cache_age = current_time - stat_result.st_mtime
    # Si la hora del sistema retrocede o el archivo se copia desde otra máquina, el
    # mtime puede quedar en el futuro. En ese caso no se puede considerar la caché como
    # fresca durante mucho tiempo; es más fiable invalidarla y volver a pedir al
    # servidor.
    if cache_age < 0 or cache_age >= MATERIAL_SEARCH_CACHE_TTL_SECONDS:
        _remove_invalid_cache(cache_path, expected=stat_result)
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)

        if (
            not isinstance(payload, dict)
            or payload.get("version") != _CACHE_FORMAT_VERSION
            or not isinstance(payload.get("items"), list)
            or not payload["items"]
        ):
            raise ValueError("carga útil de caché no válida")

        items = []
        for raw_item in payload["items"]:
            if not isinstance(raw_item, dict):
                raise ValueError("elemento de material no válido")
            item_provider = raw_item.get("provider")
            item_url = raw_item.get("url")
            item_duration = raw_item.get("duration")
            source_info = raw_item.get("source_info")
            if (
                not isinstance(item_provider, str)
                or not item_provider
                or not isinstance(item_url, str)
                or not item_url
                or isinstance(item_duration, bool)
                or not isinstance(item_duration, (int, float))
                or (isinstance(item_duration, float) and not math.isfinite(item_duration))
                or item_duration < 1
                or not isinstance(source_info, dict)
                or not source_info
            ):
                raise ValueError("campos de material no válidos")
            source_info = dict(source_info)
            source_info["search_term"] = search_term
            items.append(
                MaterialInfo(
                    provider=item_provider,
                    url=item_url,
                    duration=int(item_duration),
                    source_info=source_info,
                )
            )
    except (OSError, ValueError, TypeError) as exc:
        logger.warning(
            f"no se pudo cargar la caché de búsqueda de material: file={cache_path.name}, error={exc}"
        )
        _remove_invalid_cache(cache_path, expected=stat_result)
        return None

    logger.info(
        f"acierto de la caché de búsqueda de material: provider={provider}, "
        f"term={search_term!r}, items={len(items)}"
    )
    return items


def save_material_search_cache(
    provider: str,
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect | str,
    items: Iterable[MaterialInfo],
) -> bool:
    """
    Guarda de forma atómica un resultado de búsqueda de material correcto y no vacío.

    Varias tareas pueden buscar a la vez la misma palabra clave. Escribir primero en un
    archivo temporal único del mismo directorio y publicarlo luego con ``os.replace``
    garantiza que un proceso lector solo vea el archivo antiguo completo o el nuevo
    completo; aunque dos procesos escritores terminen a la vez, el contenido final
    siempre es un resultado válido para la misma clave de caché.
    """
    if str(provider).strip().lower() == "coverr":
        return False

    temp_path = None
    try:
        serialized_items = []
        for item in items:
            source_info = _cached_source_info(item)
            if not item.url or item.duration <= 0 or not source_info:
                continue
            serialized_items.append(
                {
                    "provider": item.provider,
                    "url": item.url,
                    "duration": int(item.duration),
                    "source_info": source_info,
                }
            )
        if not serialized_items:
            return False

        cache_path = _cache_path(
            provider=provider,
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )
        cleanup_expired_material_search_cache()
        payload = {
            "version": _CACHE_FORMAT_VERSION,
            "items": serialized_items,
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(
                payload,
                temp_file,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(temp_path, cache_path)
        return True
    except Exception as exc:
        logger.warning(
            "no se pudo guardar la caché de búsqueda de material: "
            f"error={type(exc).__name__}, detail={exc}"
        )
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        return False


def cleanup_expired_material_search_cache(
    *,
    now: float | None = None,
    force: bool = False,
) -> int:
    """
    Limpia con poca frecuencia la caché de búsquedas caducadas que no se han vuelto a
    consultar y los archivos temporales que dejaron escrituras interrumpidas.

    La ruta de escritura normal analiza el directorio como mucho una vez por hora, para
    que cada búsqueda no genere un recorrido lineal del directorio; ``force`` solo es
    para tests o mantenimiento explícito. Solo se borran los archivos de caché JSON con
    nombre SHA-256 y los intermedios ``.tmp`` que genera este módulo, sin tocar otros
    archivos que el usuario haya puesto en el directorio. Ambos comparten el mismo
    criterio de caducidad, así que los temporales que se siguen escribiendo no los borra
    por error una limpieza concurrente.
    """
    global _last_cleanup_monotonic

    monotonic_now = time.monotonic()
    with _cleanup_state_lock:
        if (
            not force
            and _last_cleanup_monotonic is not None
            and monotonic_now - _last_cleanup_monotonic
            < _CACHE_CLEANUP_INTERVAL_SECONDS
        ):
            return 0
        _last_cleanup_monotonic = monotonic_now

    try:
        cache_dir = _cache_dir()
        entries = os.scandir(cache_dir)
    except Exception as exc:
        logger.warning(
            "no se pudo analizar la caché de búsqueda de material: "
            f"error={type(exc).__name__}, detail={exc}"
        )
        return 0

    current_time = time.time() if now is None else now
    deleted_count = 0
    failed_count = 0
    with entries:
        for entry in entries:
            if not (
                _CACHE_FILE_PATTERN.fullmatch(entry.name)
                or _CACHE_TEMP_FILE_PATTERN.fullmatch(entry.name)
            ):
                continue
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                # DirEntry.stat() devuelve st_dev/st_ino como 0 en Windows, lo que nunca
                # coincidiría con la comprobación de identidad de _is_same_cache_file().
                expected = os.stat(entry.path, follow_symlinks=False)
                cache_age = current_time - expected.st_mtime
                if _CACHE_TEMP_FILE_PATTERN.fullmatch(entry.name):
                    # Una escritura activa puede parecer del futuro tras un retroceso
                    # del reloj. Solo se eliminan archivos temporales que sean
                    # demostrablemente lo bastante antiguos.
                    if cache_age < MATERIAL_SEARCH_CACHE_TTL_SECONDS:
                        continue
                elif 0 <= cache_age < MATERIAL_SEARCH_CACHE_TTL_SECONDS:
                    continue
                if not _is_same_cache_file(Path(entry.path), expected):
                    continue
                os.unlink(entry.path)
                deleted_count += 1
            except OSError as exc:
                failed_count += 1
                logger.warning(
                    "no se pudo borrar el archivo de caché de búsqueda de material: "
                    f"file={entry.name}, error={exc}"
                )

    if deleted_count or failed_count:
        logger.info(
            "terminó la limpieza de la caché de búsqueda de material: "
            f"deleted={deleted_count}, failed={failed_count}"
        )
    return deleted_count
