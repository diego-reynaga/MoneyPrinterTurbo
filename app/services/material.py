import base64
from concurrent.futures import ThreadPoolExecutor
import io
import math
import os
import random
import tempfile
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable, List
from urllib.parse import quote_plus, urlencode, urlsplit, urlunsplit

import requests
from loguru import logger

from app.utils.request_logging import trace_http_call
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image, UnidentifiedImageError

from app.config import config
from app.models.schema import ImageMotion, MaterialInfo, VideoAspect, VideoConcatMode
from app.services import (
    infistar_video,
    material_cache,
    material_upload,
    metaso_minimax,
    muapi,
    ofox,
    task_artifacts,
    video,
    volcengine_seedance,
)
from app.utils import logging_utils, utils

# Contador seguro entre hilos para la rotación de claves API
_api_key_counter = 0
_api_key_lock = threading.Lock()

# Una URL del proveedor puede apuntar a un objeto inesperadamente grande o a un stream
# sin fin. Los clips cortos de stock y generados deberían quedar muy por debajo de este
# límite conservador.
MAX_VIDEO_DOWNLOAD_BYTES = 512 * 1024 * 1024

# Por defecto se mantiene en serie, igual que la versión antigua; si hace falta, se
# puede aumentar en la configuración la concurrencia del material de stock.
_DEFAULT_MATERIAL_CONCURRENCY = 1


def _get_material_concurrency() -> int:
    try:
        concurrency = int(config.app.get("material_concurrency", _DEFAULT_MATERIAL_CONCURRENCY))
    except (TypeError, ValueError):
        concurrency = _DEFAULT_MATERIAL_CONCURRENCY
    return max(1, min(8, concurrency))


class _OpenAIImageDecodeError(ValueError):
    """Indica que los bytes devueltos por la interfaz compatible no se pueden decodificar
    como imagen; no incluye fallos al escribir el archivo local.
    """


class OpenAIImagePaidResultError(RuntimeError):
    """Una solicitud de imagen de pago no puede proporcionar un material de vídeo local
    utilizable.
    """


class OpenAIImageUnconfirmedError(OpenAIImagePaidResultError):
    """Una solicitud de imagen de pago puede haber tenido éxito sin devolver respuesta."""


def _safe_public_url(value: Any) -> str | None:
    """
    Solo conserva direcciones de páginas HTTP(S) que se pueden mostrar públicamente,
    quitando los parámetros de consulta y las credenciales.

    Las direcciones de descarga del material pueden llevar una clave API, un JWT firmado
    o un token temporal. El manifiesto de la tarea solo necesita ayudar al usuario a
    volver a la página pública del material del proveedor, y no debe guardar parámetros
    de autenticación; las URL con información de usuario también se rechazan, para que
    contenidos como ``https://user:pass@example.com`` no se guarden en disco.
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


def _creator_info(value: Any) -> dict[str, str] | None:
    """Extrae campos públicos unificados de las distintas estructuras de autor de cada
    proveedor.
    """
    if isinstance(value, str) and value.strip():
        return {"name": value.strip()}
    if not isinstance(value, dict):
        return None

    creator: dict[str, str] = {}
    creator_id = value.get("id")
    creator_name = value.get("name") or value.get("username")
    creator_page = _safe_public_url(
        value.get("url") or value.get("profile_url") or value.get("profile_page")
    )
    if creator_id is not None:
        creator["id"] = str(creator_id)
    if creator_name:
        creator["name"] = str(creator_name)
    if creator_page:
        creator["profile_page"] = creator_page
    return creator or None


def _material_source_record(item: MaterialInfo, local_path: str) -> dict[str, Any]:
    """
    Genera un registro de origen ligero para el material descargado con éxito.

    ``source_info`` puede venir de la caché, o incluso de un ``MaterialInfo`` construido
    externamente, así que no se puede escribir tal cual. Aquí se reconstruye con una
    lista blanca, conservando solo la página pública, los identificadores de negocio y
    el tamaño, y registrando solo el nombre del archivo local, para que los directorios
    del usuario o las rutas de montaje de Docker no lleguen al archivo de la tarea.
    """
    source = item.source_info if isinstance(item.source_info, dict) else {}
    record: dict[str, Any] = {
        "provider": str(item.provider or source.get("provider") or ""),
        "local_file": Path(local_path).name,
        "duration": int(item.duration),
    }

    search_term = source.get("search_term")
    asset_id = source.get("asset_id")
    source_page = _safe_public_url(source.get("source_page"))
    if isinstance(search_term, str) and search_term.strip():
        record["search_term"] = search_term.strip()
    if asset_id not in (None, ""):
        record["asset_id"] = str(asset_id)
    if source_page:
        record["source_page"] = source_page

    creator = _creator_info(source.get("creator"))
    if creator:
        record["creator"] = creator

    raw_rendition = source.get("rendition")
    if isinstance(raw_rendition, dict):
        rendition = {}
        for field in ("id", "width", "height"):
            value = raw_rendition.get(field)
            if value not in (None, ""):
                rendition[field] = str(value) if field == "id" else value
        if rendition:
            record["rendition"] = rendition
    return record


def _persist_material_sources(
    task_id: str,
    material_sources: list[dict[str, Any]],
) -> None:
    """
    Añade al manifiesto de la tarea el origen del material que se ha descargado
    realmente.

    El registro de la tarea es una función auxiliar: no puede cambiar el valor de
    retorno de la función de descarga de vídeo ni interrumpir el flujo principal del
    vídeo final por un fallo al escribir en disco. ``patch_script_data`` se encarga de
    la sustitución atómica y del log de excepciones; aquí solo se registra la cantidad
    tras el éxito, para poder confirmar si la información de trazabilidad de la tarea se
    ha guardado en disco.
    """
    try:
        saved = task_artifacts.patch_script_data(
            task_id,
            material_sources=material_sources,
        )
        if saved:
            logger.info(
                f"registros de origen del material guardados: "
                f"task_id={task_id}, count={len(material_sources)}"
            )
    except Exception as exc:
        # task_artifacts ya está diseñado para degradar ante fallos; aquí se mantiene
        # aun así un último aislamiento, para que futuros cambios de implementación o
        # una excepción al resolver el directorio no afecten por accidente al valor de
        # retorno de la descarga de material.
        logger.warning(
            "no se pudieron guardar los registros de origen del material: "
            f"task_id={task_id}, error={type(exc).__name__}, detail={exc}"
        )


def _get_tls_verify() -> bool:
    # La validación de certificados TLS está activada por defecto, para que la búsqueda
    # y la descarga de material no puedan manipularse con un ataque de intermediario.
    # Solo en casos que lo requieran claramente, como proxies corporativos o
    # certificados autofirmados, se permite al usuario desactivarla temporalmente
    # configurando explícitamente `tls_verify = false` en `config.toml`.
    tls_verify = config.app.get("tls_verify", True)
    if isinstance(tls_verify, str):
        tls_verify = tls_verify.strip().lower() not in ("0", "false", "no", "off")

    if not tls_verify:
        logger.warning(
            "la verificación de certificados TLS está desactivada por config.app.tls_verify=false. "
            "Úsalo solo en entornos con proxy de confianza."
        )

    return bool(tls_verify)


def get_api_key(cfg_key: str):
    api_keys = config.app.get(cfg_key)
    if not api_keys:
        raise ValueError(
            f"\n\n##### {cfg_key} no está configurado #####\n\n"
            f"Configúralo en el archivo config.toml: {config.config_file}\n"
        )

    # si solo se proporciona una clave, se devuelve
    if isinstance(api_keys, str):
        return api_keys

    global _api_key_counter
    with _api_key_lock:
        _api_key_counter += 1
        return api_keys[_api_key_counter % len(api_keys)]


def _redact_secret(message: str, secret: str) -> str:
    """
    Censura de forma mínima el texto de una excepción que se va a escribir en el log.

    Las excepciones de conexión de requests pueden contener la URL completa de la
    solicitud, y la clave API de Pixabay se pasa como parámetro de consulta. Aquí se
    sustituyen tanto el valor original como el codificado en URL, conservando la
    información del error de red para diagnosticar y evitando que la clave llegue al
    archivo de log.
    """
    safe_message = str(message)
    if not secret:
        return safe_message

    safe_message = safe_message.replace(secret, "***")
    encoded_secret = quote_plus(secret)
    if encoded_secret != secret:
        safe_message = safe_message.replace(encoded_secret, "***")
    return safe_message


def _redact_request_error(error: Exception, *secrets: str) -> str:
    """
    Conserva la información útil para diagnosticar excepciones de red, eliminando la
    clave API y las credenciales del proxy.

    Registrar solo el tipo de excepción perdería contexto clave como DNS, certificados o
    tiempos de espera; registrar la excepción original podría repetir la URL completa de
    la solicitud. Una entrada común permite que los tres proveedores de material usen
    las mismas reglas de censura.
    """
    safe_message = str(error)
    for secret in secrets:
        safe_message = _redact_secret(safe_message, str(secret or ""))
    for proxy_url in config.proxy.values():
        safe_message = _redact_secret(safe_message, str(proxy_url))
    return safe_message


def _is_cloudflare_challenge(response: requests.Response) -> bool:
    """
    Identifica el Challenge HTML que devuelve Cloudflare, en lugar de tomarlo por JSON
    de Pixabay.

    Cloudflare suele poner `cf-mitigated: challenge`; algunos despliegues solo devuelven
    un HTML con "Just a moment" o challenge-platform, así que se mantienen las
    características del contenido como respaldo. El cuerpo de la respuesta solo se
    evalúa en memoria y no se escribe en el log, para no registrar grandes bloques de
    HTML sin valor.
    """
    headers = getattr(response, "headers", {}) or {}
    if str(headers.get("cf-mitigated", "")).lower() == "challenge":
        return True

    content_type = str(headers.get("content-type", "")).lower()
    if "text/html" not in content_type:
        return False

    body = str(getattr(response, "text", "")).lower()
    return "just a moment" in body or "/cdn-cgi/challenge-platform/" in body


def _matches_video_aspect(
    width: Any,
    height: Any,
    video_aspect: VideoAspect,
    *,
    is_vertical: Any = None,
) -> bool:
    """
    Decide si el material remoto coincide con la orientación de imagen de destino.

    Los campos de respuesta de Pexels, Pixabay y Coverr no son uniformes, así que
    primero se usa el ancho y alto para una decisión fiable; cuando a algunas respuestas
    históricas de Coverr les falta el tamaño, se usa como respaldo el booleano explícito
    ``is_vertical``. El material cuya orientación no se puede confirmar se omite
    directamente, para que una tarea vertical no mezcle material horizontal y acabe con
    bandas negras en el vídeo final.
    """
    aspect = VideoAspect(video_aspect)
    try:
        normalized_width = int(float(width))
        normalized_height = int(float(height))
    except (OverflowError, TypeError, ValueError):
        normalized_width = 0
        normalized_height = 0

    if normalized_width > 0 and normalized_height > 0:
        if aspect == VideoAspect.portrait:
            return normalized_height > normalized_width
        if aspect == VideoAspect.landscape:
            return normalized_width > normalized_height
        return normalized_width == normalized_height

    if isinstance(is_vertical, bool) and aspect != VideoAspect.square:
        return is_vertical == (aspect == VideoAspect.portrait)
    return False


def _filter_materials_by_aspect(
    items: List[MaterialInfo],
    video_aspect: VideoAspect,
) -> List[MaterialInfo]:
    """
    Vuelve a validar la orientación de los resultados en caché.

    La caché de búsqueda de material se conserva hasta 24 horas, y la caché escrita
    antes de actualizar puede contener material con una orientación que no coincide.
    Filtrar en la entrada común de caché hace que la corrección tenga efecto de
    inmediato y también protege ante proveedores de terceros o cachés antiguas que se
    saltaran el filtrado remoto. Las entradas antiguas cuyo tamaño de rendition no se
    puede leer se tratan como no verificadas y se omiten.
    """
    aspect = VideoAspect(video_aspect)
    if aspect == VideoAspect.square:
        # Pixabay y Coverr rara vez ofrecen material cuadrado nativo. La salida cuadrada
        # mantiene el comportamiento existente: acepta los candidatos utilizables y deja
        # que la fase de composición del vídeo los recorte, para que tras actualizar las
        # tareas 1:1 no se queden sin material.
        return list(items)

    filtered_items = []
    for item in items:
        source_info = item.source_info if isinstance(item.source_info, dict) else {}
        rendition = source_info.get("rendition")
        rendition = rendition if isinstance(rendition, dict) else {}
        if _matches_video_aspect(
            rendition.get("width"),
            rendition.get("height"),
            aspect,
        ):
            filtered_items.append(item)
    return filtered_items


def search_videos_pexels(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()
    api_key = get_api_key("pexels_api_keys")
    headers = {
        "Authorization": api_key,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    }
    # Construye la URL
    params = {"query": search_term, "per_page": 20}
    if aspect != VideoAspect.square:
        params["orientation"] = aspect.name
    query_url = f"https://api.pexels.com/v1/videos/search?{urlencode(params)}"
    logger.info(f"buscando vídeos en Pexels: term={search_term!r}")

    try:
        r = trace_http_call(
            "GET", requests.get,
            query_url,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )
        response = r.json()
        video_items = []
        if not isinstance(response, dict) or not isinstance(
            response.get("videos"), list
        ):
            logger.error("la búsqueda de vídeos de Pexels devolvió una respuesta no admitida")
            return video_items
        videos = response["videos"]
        # recorre cada vídeo del resultado
        for v in videos:
            if not isinstance(v, dict):
                continue
            duration = v.get("duration")
            # comprueba si el vídeo tiene la duración mínima deseada
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration < minimum_duration
            ):
                continue
            video_files = v.get("video_files")
            if not isinstance(video_files, list):
                continue
            # Prefiere la rendition más pequeña que pueda llenar el lienzo de salida. La
            # salida cuadrada puede recortar cualquier orientación, como hacen otros
            # proveedores.
            renditions = []
            for video in video_files:
                if not isinstance(video, dict):
                    continue
                try:
                    w = int(video.get("width"))
                    h = int(video.get("height"))
                except (OverflowError, TypeError, ValueError):
                    continue
                video_url = video.get("link")
                if not isinstance(video_url, str) or not video_url:
                    continue
                if (
                    (aspect == VideoAspect.square or _matches_video_aspect(w, h, aspect))
                    and w >= video_width
                    and h >= video_height
                ):
                    renditions.append((w * h, video, w, h, video_url))
            if renditions:
                _, video, w, h, video_url = min(renditions, key=lambda candidate: candidate[0])
                item = MaterialInfo()
                item.provider = "pexels"
                item.url = video_url
                item.duration = duration
                item.source_info = {
                    "provider": "pexels",
                    "search_term": search_term,
                    "asset_id": (
                        str(v.get("id")) if v.get("id") is not None else None
                    ),
                    "source_page": _safe_public_url(v.get("url")),
                    "creator": _creator_info(v.get("user")),
                    "rendition": {
                        "id": (
                            str(video.get("id"))
                            if video.get("id") is not None
                            else None
                        ),
                        "width": w,
                        "height": h,
                    },
                }
                video_items.append(item)
        return video_items
    except Exception as e:
        logger.error(
            "falló la búsqueda de vídeos de Pexels: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
        )

    return []


def search_videos_pixabay(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)

    video_width, video_height = aspect.to_resolution()

    api_key = get_api_key("pixabay_api_keys")
    # Construye la URL
    params = {
        "q": search_term,
        "video_type": "all",  # Valores aceptados: "all", "film", "animation"
        "per_page": 50,
        "key": api_key,
    }
    query_url = f"https://pixabay.com/api/videos/?{urlencode(params)}"
    logger.info(
        f"buscando vídeos en Pixabay: term={search_term!r}, "
        f"proxy_enabled={bool(config.proxy)}"
    )

    try:
        r = trace_http_call(
            "GET", requests.get,
            query_url, proxies=config.proxy, verify=_get_tls_verify(), timeout=(30, 60)
        )
        status_code = int(getattr(r, "status_code", 200))
        headers = getattr(r, "headers", {}) or {}
        content_type = str(headers.get("content-type", ""))
        retry_after = headers.get("retry-after")
        cf_ray = headers.get("cf-ray")

        if _is_cloudflare_challenge(r):
            logger.error(
                "la búsqueda en Pixabay fue bloqueada por un desafío de Cloudflare: "
                f"status={status_code}, cf_ray={cf_ray or 'unknown'}. "
                "Revisa la red o el proxy del servidor, o usa Pexels/Coverr en su lugar."
            )
            return []

        if status_code == 429:
            logger.error(
                "se superó el límite de frecuencia de la API de Pixabay: "
                f"status=429, retry_after={retry_after or 'unknown'}"
            )
            return []

        if status_code >= 400:
            logger.error(
                "falló la petición de búsqueda a Pixabay: "
                f"status={status_code}, content_type={content_type or 'unknown'}"
            )
            return []

        try:
            response = r.json()
        except ValueError:
            logger.error(
                "Pixabay devolvió una respuesta inesperada que no es JSON: "
                f"status={status_code}, content_type={content_type or 'unknown'}"
            )
            return []

        video_items = []
        if not isinstance(response, dict) or not isinstance(
            response.get("hits"), list
        ):
            logger.error("la búsqueda de vídeos de Pixabay devolvió una respuesta no admitida")
            return video_items
        videos = response["hits"]
        # recorre cada vídeo del resultado
        for v in videos:
            if not isinstance(v, dict):
                continue
            duration = v.get("duration")
            # comprueba si el vídeo tiene la duración mínima deseada
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration < minimum_duration
            ):
                continue
            video_files = v.get("videos")
            if not isinstance(video_files, dict):
                continue
            # recorre cada url para determinar la mejor calidad
            for video_type, video in video_files.items():
                if not isinstance(video, dict):
                    continue
                try:
                    w = int(video["width"])
                    h = int(video["height"])
                except (KeyError, OverflowError, TypeError, ValueError):
                    continue
                video_url = video.get("url")
                if not isinstance(video_url, str) or not video_url:
                    continue
                # Pixabay rara vez devuelve vídeos cuadrados nativos; la salida 1:1
                # sigue aceptando los candidatos con resolución suficiente y deja que la
                # fase de composición los recorte. Para horizontal y vertical, la
                # orientación de destino debe coincidir estrictamente.
                orientation_matches = aspect == VideoAspect.square or (
                    _matches_video_aspect(w, h, aspect)
                )
                if orientation_matches and w >= video_width and h >= video_height:
                    item = MaterialInfo()
                    item.provider = "pixabay"
                    item.url = video_url
                    item.duration = duration
                    item.source_info = {
                        "provider": "pixabay",
                        "search_term": search_term,
                        "asset_id": (
                            str(v.get("id")) if v.get("id") is not None else None
                        ),
                        "source_page": _safe_public_url(v.get("pageURL")),
                        "creator": _creator_info(
                            {
                                "id": v.get("user_id"),
                                "name": v.get("user"),
                            }
                        ),
                        "rendition": {
                            "id": video_type,
                            "width": w,
                            "height": video.get("height"),
                        },
                    }
                    video_items.append(item)
                    break
        return video_items
    except Exception as e:
        error_message = _redact_request_error(e, api_key)
        logger.error(
            "falló la petición de búsqueda a Pixabay: "
            f"error={type(e).__name__}, detail={error_message}"
        )

    return []


def search_videos_coverr(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    """
    Coverr (https://coverr.co): vídeos de stock HD/4K gratuitos, sujetos a los términos
    de la licencia de Coverr (https://coverr.co/license).

    Notas de la API de Coverr (según la documentación oficial en api.coverr.co/docs/):
      - Autenticación: Authorization: Bearer <api_key>
      - Endpoint de búsqueda: GET /videos?query=..., con estructura de respuesta
        {"hits": [...], ...}
      - añadir ?urls=true devuelve directamente en la respuesta de búsqueda los enlaces
        directos mp4
      - la URL es un JWT firmado (vinculado a la clave API, sin caducidad)
      - Coverr permite filtrar material horizontal/vertical con
        filter=is_vertical:true/false;
        tras la respuesta se sigue validando en local con max_width/max_height o
        is_vertical
      - el campo duration aparece tanto como number como string, y esta función acepta
        ambos

    Esta función usa el campo urls.mp4_download como dirección de descarga; según la
    documentación oficial de Coverr
    (https://api.coverr.co/docs/videos/#download-a-video), hacer GET a esa URL ya cuenta
    para Coverr como un evento de descarga legítimo en las estadísticas, sin necesidad
    de llamar además a PATCH /videos/:id/stats/downloads.
    """
    aspect = VideoAspect(video_aspect)
    api_key = get_api_key("coverr_api_keys")
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "query": search_term,
        "page_size": 20,
        "urls": "true",
        "sort": "popular",
    }
    # El filtrado de orientación en el servidor puede devolver el material de destino
    # directamente de los resultados de búsqueda completos, evitando que al tomar
    # primero los resultados populares y filtrar en local se queden vacíos los
    # candidatos verticales. El material cuadrado no tiene un booleano correspondiente,
    # así que sigue dependiendo de la validación local de ancho y alto.
    if aspect == VideoAspect.portrait:
        params["filter"] = "is_vertical:true"
    elif aspect == VideoAspect.landscape:
        params["filter"] = "is_vertical:false"
    query_url = f"https://api.coverr.co/videos?{urlencode(params)}"
    logger.info(f"buscando vídeos en Coverr: term={search_term!r}")

    try:
        r = trace_http_call(
            "GET", requests.get,
            query_url,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )
        response = r.json()
        video_items: List[MaterialInfo] = []

        if not isinstance(response, dict) or not isinstance(
            response.get("hits"), list
        ):
            logger.error("la búsqueda de vídeos de Coverr devolvió una respuesta no admitida")
            return video_items

        for v in response["hits"]:
            if not isinstance(v, dict):
                continue
            # duration puede ser number (11.625) o string ("10.500000") según la
            # respuesta
            try:
                duration = int(float(v.get("duration") or 0))
            except (OverflowError, TypeError, ValueError):
                continue
            if duration < minimum_duration:
                continue

            video_id = v.get("id")
            urls = v.get("urls")
            if not isinstance(urls, dict):
                continue
            mp4_download_url = urls.get("mp4_download")
            if (
                not video_id
                or not isinstance(mp4_download_url, str)
                or not mp4_download_url
            ):
                continue
            if aspect != VideoAspect.square and not _matches_video_aspect(
                v.get("max_width"),
                v.get("max_height"),
                aspect,
                is_vertical=v.get("is_vertical"),
            ):
                continue

            item = MaterialInfo()
            item.provider = "coverr"
            item.url = mp4_download_url
            item.duration = duration
            item.source_info = {
                "provider": "coverr",
                "search_term": search_term,
                "asset_id": str(video_id),
                "source_page": _safe_public_url(v.get("canonical_url") or v.get("url")),
                "creator": _creator_info(v.get("creator") or v.get("author")),
                "rendition": {
                    "id": "mp4_download",
                    "width": v.get("max_width"),
                    "height": v.get("max_height"),
                },
            }
            video_items.append(item)
        return video_items
    except Exception as e:
        logger.error(
            "falló la búsqueda de vídeos de Coverr: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
        )

    return []


# WaveSpeed AI (https://wavespeed.ai) genera el material directamente a partir de las
# palabras clave del guion con modelos de texto a vídeo, compartiendo con las tres
# fuentes de material de stock la estructura de resultado MaterialInfo y el flujo
# posterior de descarga y edición.
WAVESPEED_API_BASE_URL = "https://api.wavespeed.ai/api/v3"
WAVESPEED_DEFAULT_T2V_MODEL = "bytedance/seedance-2.0-fast/text-to-video"
WAVESPEED_POLL_INTERVAL_SECONDS = 2.0
WAVESPEED_RUN_TIMEOUT_SECONDS = 600.0
# El modelo por defecto bytedance/seedance-2.0-fast/text-to-video solo acepta de 4 a 15
# segundos; las solicitudes fuera de rango la API las rechaza directamente. La duración
# de clip por defecto de la WebUI es de 3 segundos, así que antes de enviar hay que
# ajustarla al rango que admite el modelo, y la duración sobrante la recorta el flujo de
# edición existente según la duración del clip.
WAVESPEED_MIN_DURATION_SECONDS = 4
WAVESPEED_MAX_DURATION_SECONDS = 15
# Los tres estados de fallo tienen significados distintos (error del modelo /
# cancelación del usuario / tiempo agotado en la plataforma), pero para el flujo de
# material todos significan que esta palabra clave no tiene producto, así que se tratan
# siempre como resultado vacío y la capa superior se salta ese clip y sigue generando.
WAVESPEED_FAILURE_STATUSES = frozenset({"failed", "cancelled", "timeout"})
# El mismo criterio que el SDK oficial de Python y el nodo de n8n de WaveSpeed: 429 y
# 5xx son fallos temporales que merecen un número limitado de reintentos con espera; 4xx
# son errores claros del cliente y fallan rápido.
WAVESPEED_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# Número de fallos temporales seguidos permitidos en un sondeo. Un GET con mala suerte
# no puede dejar ilocalizable una tarea ya facturada.
WAVESPEED_MAX_POLL_RETRIES = 5
# Base del retroceso lineal: el reintento n espera base * n segundos.
WAVESPEED_RETRY_BASE_SECONDS = 1.0
# Número de reintentos sobre la misma dirección firmada cuando falla la descarga del
# producto. El material ya se ha generado pagando, así que se reintenta primero la
# dirección original, sin volver a enviar una tarea de generación de pago por un fallo
# pasajero de descarga.
WAVESPEED_MAX_DOWNLOAD_RETRIES = 2


class WaveSpeedUnconfirmedTaskError(RuntimeError):
    """
    La tarea de generación de pago se ha enviado, pero su estado final no se puede
    confirmar en local.

    Este tipo de excepción nunca equivale a "la tarea ha fallado y se puede repetir": la
    tarea remota puede seguir en marcha o haber terminado y facturado. El flujo de
    material debe detenerse aquí, sin enviar nuevas tareas de pago para las palabras
    clave siguientes, y pasar el prediction id ya enviado al estado de la tarea para que
    se pueda recuperar a mano.
    """

    def __init__(self, message: str, prediction_id: str = ""):
        super().__init__(message)
        self.prediction_id = prediction_id


class WaveSpeedDownloadError(RuntimeError):
    """Una predicción de pago terminó, pero su vídeo no se pudo guardar en local."""

    def __init__(self, message: str, prediction_id: str = ""):
        super().__init__(message)
        self.prediction_id = prediction_id


def _wavespeed_status_code(response: Any) -> int:
    """Lee el código de estado de la respuesta; si a un doble de prueba o a un objeto de
    excepción le falta ese campo, se trata como 200.
    """
    try:
        return int(getattr(response, "status_code", 200))
    except (TypeError, ValueError):
        return 200


def _is_wavespeed_retryable_error(error: Exception) -> bool:
    """
    Decide si merece la pena reintentar una excepción de sondeo.

    Las excepciones de red como las de conexión o tiempo de espera no tienen código de
    estado y se tratan como fallos temporales; las respuestas con código de estado solo
    se reintentan con 429 y 5xx, igual que el conjunto de reintentos del SDK oficial.
    """
    if isinstance(
        error,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ),
    ):
        return True
    response = getattr(error, "response", None)
    if response is not None:
        return _wavespeed_status_code(response) in WAVESPEED_RETRYABLE_STATUS_CODES
    return False


def _wavespeed_duration_bounds() -> tuple[int, int]:
    """
    Devuelve el rango de duración de generación que admite el modelo actual (en
    segundos).

    El rango por defecto corresponde al modelo Seedance por defecto; cuando el usuario
    cambia a otro modelo de texto a vídeo, puede ajustar el rango en la configuración a
    la vez. Cualquier configuración anómala vuelve al valor por defecto, garantizando
    min <= max, para no convertir la entrada del usuario en una solicitud remota que
    fallará seguro.
    """

    def read_bound(key: str, fallback: int) -> int:
        try:
            value = int(config.app.get(key, fallback))
        except (TypeError, ValueError):
            return fallback
        return value if value >= 1 else fallback

    min_duration = read_bound("wavespeed_min_duration", WAVESPEED_MIN_DURATION_SECONDS)
    max_duration = read_bound("wavespeed_max_duration", WAVESPEED_MAX_DURATION_SECONDS)
    return min_duration, max(max_duration, min_duration)


def generate_videos_wavespeed(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    """
    Genera un fragmento de material para una palabra clave del guion con un modelo de
    texto a vídeo de WaveSpeed.

    Mantiene la misma firma y el mismo contrato de fallo con lista vacía que los
    search_videos_* de las fuentes de stock, para poder conectarse directamente al flujo
    común de descarga y cómputo de duración de ``download_videos``. En el contexto de
    generación, ``minimum_duration`` es la duración objetivo del clip (en segundos).
    """
    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()
    api_key = get_api_key("wavespeed_api_keys")
    model_id = (
        str(
            config.app.get("wavespeed_text_to_video_model", "")
            or WAVESPEED_DEFAULT_T2V_MODEL
        )
        .strip()
        .strip("/")
    )
    headers = {"Authorization": f"Bearer {api_key}"}
    requested_duration = max(int(minimum_duration), 1)
    min_duration, max_duration = _wavespeed_duration_bounds()
    duration = min(max(requested_duration, min_duration), max_duration)
    if duration != requested_duration:
        # Generar más de lo pedido no afecta al vídeo final: el montaje sigue recortando
        # según la duración del clip; generar menos de lo pedido solo ocurre cuando la
        # solicitud supera el máximo del modelo, y entonces solo se puede ajustar a ese
        # máximo.
        logger.info(
            f"duración del clip de WaveSpeed ajustada al rango admitido por el modelo: "
            f"requested={requested_duration}s, using={duration}s "
            f"(admitido {min_duration}-{max_duration}s)"
        )
    payload = {
        "prompt": search_term,
        "aspect_ratio": aspect.value,
        "duration": duration,
    }
    logger.info(
        f"generando vídeo en WaveSpeed: model={model_id}, "
        f"term={search_term!r}, duration={duration}s"
    )

    # El POST de envío nunca se reintenta automáticamente: la solicitud puede haber
    # creado ya una tarea de pago en remoto, y reenviarla generaría y cobraría dos veces
    # (igual que la estrategia de submission del SDK oficial).
    try:
        submit_response = trace_http_call(
            "POST", requests.post,
            f"{WAVESPEED_API_BASE_URL}/{model_id}",
            json=payload,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )
    except Exception as e:
        # No haber recibido respuesta no significa que no se haya creado la tarea. En
        # ese caso el estado es desconocido y hay que terminar todo el flujo de
        # generación, en lugar de seguir enviando nuevas tareas de pago para la
        # siguiente palabra clave.
        raise WaveSpeedUnconfirmedTaskError(
            "el envío a WaveSpeed no devolvió respuesta; puede que la tarea "
            "ya exista en remoto: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
        ) from e

    submit_status = _wavespeed_status_code(submit_response)
    if submit_status >= 500:
        # Un 5xx puede ocurrir después de crearse la tarea, y no se puede saber si ya se
        # ha facturado.
        raise WaveSpeedUnconfirmedTaskError(
            f"el envío a WaveSpeed falló con HTTP {submit_status}; "
            "puede que la tarea ya exista en remoto"
        )
    try:
        submit_body = submit_response.json()
    except Exception as e:
        if 400 <= submit_status < 500:
            # Los 4xx devueltos por el gateway o la capa de autenticación pueden ser
            # texto plano/HTML. Un 4xx sigue siendo un rechazo claro: el servidor no ha
            # creado ninguna tarea, así que no se puede informar por error de que "puede
            # haberse facturado" y terminar toda la tarea.
            logger.error(
                "petición de generación de vídeo de WaveSpeed rechazada: "
                f"http_status={submit_status}, cuerpo ilegible: {type(e).__name__}"
            )
            return []
        raise WaveSpeedUnconfirmedTaskError(
            "el envío a WaveSpeed devolvió una respuesta ilegible; puede que la "
            f"tarea ya exista en remoto: error={type(e).__name__}"
        ) from e

    submit_data = submit_body.get("data") if isinstance(submit_body, dict) else None
    if not isinstance(submit_body, dict) or submit_body.get("code") != 200:
        # Los 4xx y los códigos de error de negocio son rechazos claros: el servidor no
        # ha creado ninguna tarea, así que no hay riesgo de cobro doble, y según el
        # contrato de las fuentes de material se devuelve un resultado vacío y se sigue.
        logger.error(
            "petición de generación de vídeo de WaveSpeed rechazada: "
            f"http_status={submit_status}, "
            f"code={submit_body.get('code') if isinstance(submit_body, dict) else None}, "
            f"detail={_redact_secret(str((submit_body or {}).get('message') or ''), api_key)}"
        )
        return []
    prediction_id = (
        str(submit_data.get("id") or "") if isinstance(submit_data, dict) else ""
    )
    if not prediction_id:
        # El envío se aceptó pero no se obtuvo ID: la tarea puede existir sin poder
        # rastrearse, así que no se puede seguir haciendo pedidos.
        raise WaveSpeedUnconfirmedTaskError(
            "WaveSpeed aceptó el envío sin devolver un prediction id"
        )
    # En cuanto el envío de la tarea de generación tiene éxito se produce un efecto de
    # facturación remota, así que primero se registra el ID de la tarea en el log, para
    # que aunque falle el sondeo posterior el usuario pueda recuperar el producto en la
    # consola de WaveSpeed con ese ID.
    logger.info(f"predicción de WaveSpeed creada: id={prediction_id}")

    result_data = _wait_for_wavespeed_prediction(
        prediction_id=prediction_id,
        headers=headers,
        api_key=api_key,
    )
    if result_data is None:
        return []

    try:
        video_items = []
        outputs = result_data.get("outputs")
        for output in outputs if isinstance(outputs, list) else []:
            # La URL del producto es una dirección de descarga temporal firmada y debe
            # conservarse entera (sin quitar los parámetros de consulta), así que no se
            # escribe en source_info y solo se usa para la descarga inmediata posterior.
            if not isinstance(output, str) or not output.startswith(
                ("http://", "https://")
            ):
                continue
            item = MaterialInfo()
            item.provider = "wavespeed"
            item.url = output
            item.duration = duration
            item.source_info = {
                "provider": "wavespeed",
                "search_term": search_term,
                "asset_id": prediction_id,
                "rendition": {
                    "id": None,
                    "width": video_width,
                    "height": video_height,
                },
            }
            video_items.append(item)
        if not video_items:
            logger.error(
                "la predicción de WaveSpeed terminó sin resultados descargables: "
                f"id={prediction_id}"
            )
        return video_items
    except Exception as e:
        # El producto ya se ha generado y facturado, así que esta excepción solo puede
        # venir del análisis local. Se registra y se devuelve un resultado vacío para
        # que la capa superior se salte ese clip, pero el estado de la tarea en sí está
        # determinado y se puede seguir con los clips siguientes.
        logger.error(
            "falló el análisis de la salida de WaveSpeed: "
            f"id={prediction_id}, error={type(e).__name__}, "
            f"detail={_redact_request_error(e, api_key)}"
        )

    return []


def _wait_for_wavespeed_prediction(
    *,
    prediction_id: str,
    headers: dict,
    api_key: str,
) -> dict | None:
    """
    Sondea el mismo prediction id hasta obtener un resultado definitivo.

    Devuelve los data de ``completed``; ante un fallo remoto explícito (failed /
    cancelled / timeout) devuelve None, indicando que la tarea ha terminado y se puede
    seguir con seguridad con los clips siguientes. Los fallos temporales reintentan el
    mismo ID con retroceso lineal, sin volver nunca a enviar la tarea; si el estado no
    se puede confirmar en ningún momento lanza :class:`WaveSpeedUnconfirmedTaskError`, y
    quien llama termina todo el flujo de generación.
    """
    deadline = time.monotonic() + WAVESPEED_RUN_TIMEOUT_SECONDS
    consecutive_failures = 0
    while True:
        try:
            response = trace_http_call(
                "GET", requests.get,
                f"{WAVESPEED_API_BASE_URL}/predictions/{prediction_id}/result",
                headers=headers,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(30, 60),
            )
            status_code = _wavespeed_status_code(response)
            if status_code in WAVESPEED_RETRYABLE_STATUS_CODES:
                raise requests.exceptions.HTTPError(
                    f"HTTP {status_code}", response=response
                )
            result_body = response.json()
            result_data = (
                result_body.get("data") if isinstance(result_body, dict) else None
            )
            if not isinstance(result_body, dict) or result_body.get("code") != 200:
                # Cuando el sondeo se rechaza explícitamente (por ejemplo, con 4xx), el
                # estado de la tarea sigue siendo desconocido: la tarea ya se ha enviado
                # y solo es que no se puede consultar el resultado en local, así que
                # tampoco se pueden enviar nuevas tareas de pago.
                raise WaveSpeedUnconfirmedTaskError(
                    "el estado de la predicción de WaveSpeed es desconocido: "
                    f"http_status={status_code}, "
                    f"code={result_body.get('code') if isinstance(result_body, dict) else None}, "
                    f"detail={_redact_secret(str((result_body or {}).get('message') or ''), api_key)}",
                    prediction_id=prediction_id,
                )
            if not isinstance(result_data, dict):
                raise WaveSpeedUnconfirmedTaskError(
                    "la carga útil del resultado de la predicción de WaveSpeed está mal formada",
                    prediction_id=prediction_id,
                )
        except WaveSpeedUnconfirmedTaskError:
            raise
        except Exception as e:
            if not _is_wavespeed_retryable_error(e):
                raise WaveSpeedUnconfirmedTaskError(
                    "falló el sondeo de la predicción de WaveSpeed y el estado de la tarea "
                    f"es desconocido: error={type(e).__name__}, "
                    f"detail={_redact_request_error(e, api_key)}",
                    prediction_id=prediction_id,
                ) from e
            consecutive_failures += 1
            if consecutive_failures > WAVESPEED_MAX_POLL_RETRIES:
                raise WaveSpeedUnconfirmedTaskError(
                    "falló el sondeo de la predicción de WaveSpeed tras "
                    f"{WAVESPEED_MAX_POLL_RETRIES + 1} intentos; puede que la tarea "
                    "siga ejecutándose en remoto: "
                    f"error={type(e).__name__}, "
                    f"detail={_redact_request_error(e, api_key)}",
                    prediction_id=prediction_id,
                ) from e
            delay = WAVESPEED_RETRY_BASE_SECONDS * consecutive_failures
            logger.warning(
                "el sondeo de la predicción de WaveSpeed tuvo un error transitorio; se "
                f"reintenta la misma tarea: id={prediction_id}, "
                f"attempt={consecutive_failures}/{WAVESPEED_MAX_POLL_RETRIES}, "
                f"error={type(e).__name__}, retry_in={delay:.1f}s"
            )
            time.sleep(delay)
            continue

        # Al obtener una respuesta válida se reinicia el contador; solo los fallos
        # seguidos consumen el cupo de reintentos.
        consecutive_failures = 0
        status = str(result_data.get("status") or "")
        if status == "completed":
            return result_data
        if status in WAVESPEED_FAILURE_STATUSES:
            logger.error(
                "la predicción de WaveSpeed no produjo un vídeo: "
                f"id={prediction_id}, status={status}, "
                f"detail={_redact_secret(str(result_data.get('error') or ''), api_key)}"
            )
            return None
        if time.monotonic() > deadline:
            # La tarea remota sigue ejecutándose y no se puede confirmar su estado final
            # en local, así que hay que dejar de hacer pedidos.
            raise WaveSpeedUnconfirmedTaskError(
                f"la predicción de WaveSpeed sigue en {status or 'pending'} tras "
                f"{WAVESPEED_RUN_TIMEOUT_SECONDS:.0f}s de espera local",
                prediction_id=prediction_id,
            )
        time.sleep(WAVESPEED_POLL_INTERVAL_SECONDS)


def _save_generated_video_with_retry(
    video_url: str, save_dir: str, provider: str
) -> str:
    """
    Descarga el producto ya generado pagando; si falla, se reintenta primero la misma
    dirección.

    Volver a generar la tarea remota cuesta pagar otra vez, así que ante un fallo
    pasajero de descarga primero hay que reintentar un número limitado de veces con
    espera sobre la dirección original, y al agotar los reintentos quien llama informa
    de un fallo recuperable de tarea de pago.
    """
    for attempt in range(WAVESPEED_MAX_DOWNLOAD_RETRIES + 1):
        try:
            saved_video_path = save_video(video_url=video_url, save_dir=save_dir)
            if saved_video_path:
                return saved_video_path
            failure_detail = "resultado vacío"
        except Exception as e:
            failure_detail = (
                f"error={type(e).__name__}, "
                f"detail={_redact_request_error(e, video_url)}"
            )
        if attempt >= WAVESPEED_MAX_DOWNLOAD_RETRIES:
            break
        delay = WAVESPEED_RETRY_BASE_SECONDS * (attempt + 1)
        logger.warning(
            "no se pudo descargar el vídeo generado; se reintenta la misma URL: "
            f"provider={provider}, "
            f"attempt={attempt + 1}/{WAVESPEED_MAX_DOWNLOAD_RETRIES}, "
            f"{failure_detail}, retry_in={delay:.1f}s"
        )
        time.sleep(delay)
    logger.error(
        "no se pudo descargar el vídeo generado tras "
        f"{WAVESPEED_MAX_DOWNLOAD_RETRIES + 1} intentos: "
        f"provider={provider}, {failure_detail}"
    )
    return ""


def _get_downloaded_video_duration(video_path: str) -> float:
    """Lee la duración utilizable del archivo multimedia descargado."""
    clip = None
    try:
        clip = VideoFileClip(video_path)
        duration = float(clip.duration or 0)
    except Exception as exc:
        raise ValueError(
            f"no se pudo medir la duración del vídeo descargado: {video_path}"
        ) from exc
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception as close_error:
                logger.warning(
                    "no se pudo cerrar el vídeo descargado tras medir su duración: "
                    f"path={video_path}, error={type(close_error).__name__}, "
                    f"detail={close_error}"
                )
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(
            f"la duración del vídeo descargado no es positiva y finita: {video_path}"
        )
    return duration


# Durante una descarga en streaming de requests no se produce ningún log. Con una red
# lenta, un solo material 4K tarda minutos en descargarse y la tarea parece colgada. Si
# se supera este intervalo sin terminar, se registran el tamaño descargado y la
# velocidad; los archivos pequeños que terminan dentro del intervalo no producen logs
# adicionales, para no inundar el log.
_DOWNLOAD_HEARTBEAT_SECONDS = 15.0
_BYTES_PER_MEGABYTE = 1024 * 1024


def _describe_download_progress(
    downloaded_bytes: int, declared_size: int, elapsed_seconds: float
) -> str:
    """Convierte el progreso de la descarga en un texto de log; si el servidor no declara
    el tamaño no se da porcentaje.
    """
    downloaded_mb = downloaded_bytes / _BYTES_PER_MEGABYTE
    speed = downloaded_mb / elapsed_seconds if elapsed_seconds > 0 else 0.0
    if declared_size > 0:
        percent = min(100, round(downloaded_bytes * 100 / declared_size))
        return (
            f"{downloaded_mb:.1f} de {declared_size / _BYTES_PER_MEGABYTE:.1f} MB "
            f"({percent}%), {speed:.2f} MB/s"
        )
    return f"{downloaded_mb:.1f} MB, {speed:.2f} MB/s"


def save_video(video_url: str, save_dir: str = "") -> str:
    if not save_dir:
        save_dir = utils.storage_dir("cache_videos")

    os.makedirs(save_dir, exist_ok=True)

    # Los parámetros de consulta pueden identificar el propio recurso (por ejemplo,
    # /download?file_id=123). Quitar la query haría que vídeos de pago no relacionados
    # compartieran una entrada de caché y reutilizaran en silencio la primera escena.
    # Los fragmentos no se envían en las solicitudes HTTP, así que no afectan a los
    # bytes descargados.
    url_hash = utils.md5(video_url.split("#", 1)[0])
    video_id = f"vid-{url_hash}"
    video_path = f"{save_dir}/{video_id}.mp4"

    # si el vídeo ya existe, devuelve la ruta
    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        logger.info(f"el vídeo ya existe: {video_path}")
        return video_path

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    # Un archivo no vacío se trata arriba como acierto de caché. La ruta final se
    # mantiene sin publicar hasta que la descarga completa pasa la validación de medios,
    # para que una descarga fallida no envenene la caché de esta tarea ni la de otra
    # concurrente.
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{video_id}-",
            suffix=".mp4",
            dir=save_dir,
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
            with trace_http_call(
                "GET", requests.get,
                video_url,
                headers=headers,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(60, 240),
                stream=True,
            ) as response:
                response.raise_for_status()
                headers = getattr(response, "headers", {}) or {}
                try:
                    declared_size = int(headers.get("Content-Length", ""))
                except (TypeError, ValueError):
                    declared_size = 0
                if declared_size > MAX_VIDEO_DOWNLOAD_BYTES:
                    raise ValueError("la descarga del vídeo supera el límite de 512 MB")

                downloaded_bytes = 0
                started_at = time.monotonic()
                last_report_at = started_at
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        downloaded_bytes += len(chunk)
                        if downloaded_bytes > MAX_VIDEO_DOWNLOAD_BYTES:
                            raise ValueError("la descarga del vídeo supera el límite de 512 MB")
                        temp_file.write(chunk)
                        now = time.monotonic()
                        if now - last_report_at >= _DOWNLOAD_HEARTBEAT_SECONDS:
                            last_report_at = now
                            # Solo se registra el nombre del archivo de caché: la
                            # dirección de descarga puede llevar firma o clave.
                            logger.info(
                                f"descargando el vídeo {video_id}.mp4: "
                                + _describe_download_progress(
                                    downloaded_bytes,
                                    declared_size,
                                    now - started_at,
                                )
                            )

        if os.path.getsize(temp_path) == 0:
            return ""

        clip = None
        try:
            clip = VideoFileClip(temp_path)
            duration = clip.duration
            fps = clip.fps
            # La duración del contenedor y un fotograma inicial no demuestran que haya
            # llegado el resto de un MP4 faststart. Se validan todos los paquetes de
            # vídeo antes de que una entrada de caché no vacía se pueda reutilizar en
            # tareas posteriores.
            material_upload._validate_video(temp_path)
            if not (duration > 0 and fps > 0):
                logger.warning(f"archivo de vídeo no válido: {temp_path} => duración o fps no válidos")
                return ""
        except Exception as e:
            logger.warning(f"archivo de vídeo no válido: {temp_path} => {str(e)}")
            return ""
        finally:
            if clip is not None:
                try:
                    clip.close()
                except Exception as close_error:
                    logger.warning(
                        f"no se pudo cerrar el clip de vídeo: {temp_path}, error: {str(close_error)}"
                    )

        os.replace(temp_path, video_path)
        temp_path = ""
        return video_path
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass
            except OSError as remove_error:
                logger.warning(
                    f"no se pudo eliminar el archivo de vídeo temporal: {temp_path}, "
                    f"error: {str(remove_error)}"
                )


# El texto a imagen compatible con OpenAI (Issue #1274) genera material de imagen para
# las palabras clave del guion mediante el protocolo /images/generations, y puede
# apuntar tanto a un gateway ComfyUI/SD local como a cualquier servicio intermedio con
# protocolo OpenAI. La imagen generada se renderiza de inmediato como un clip mp4 con el
# mismo "zoom lento" que el material local, totalmente transparente para el flujo de
# edición posterior.
OPENAI_IMAGE_ENDPOINT_PATH = "images/generations"
# La interfaz de imágenes oficial de OpenAI solo acepta los tamaños que fija el modelo,
# y no se le puede pasar directamente la resolución del vídeo (como 1080x1920). Si
# openai_image_size se deja vacío, se toman los siguientes valores por defecto
# compatibles según la relación de aspecto; un gateway local puede sobrescribirlos
# explícitamente.
OPENAI_IMAGE_DEFAULT_SIZES = {
    VideoAspect.portrait: "1024x1536",
    VideoAspect.landscape: "1536x1024",
    VideoAspect.square: "1024x1024",
}
# El mismo criterio de reintentos que WaveSpeed: 429 y 5xx son fallos temporales y se
# reintentan un número limitado de veces con espera.
OPENAI_IMAGE_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# 401/403 significan que la clave actual se ha rechazado explícitamente. get_api_key
# rota la clave en cada llamada, así que con varias claves configuradas el reintento
# cambia de clave automáticamente; con una sola clave falla rápido, sin reintentos
# inútiles.
OPENAI_IMAGE_KEY_ERROR_STATUS_CODES = frozenset({401, 403})
OPENAI_IMAGE_MAX_ATTEMPTS = 3
OPENAI_IMAGE_MAX_BYTES = 25 * 1024 * 1024
OPENAI_IMAGE_MAX_PIXELS = 50_000_000
# Imágenes en serie + retroceso lineal, compatible con la ventana de recuperación de
# límites habitual de los servicios intermedios.
OPENAI_IMAGE_RETRY_BACKOFF_SECONDS = (5, 15, 30)
# La interfaz de generación síncrona puede tardar decenas de segundos en devolver la
# imagen, así que se da un margen holgado al tiempo de espera de lectura.
OPENAI_IMAGE_REQUEST_TIMEOUT = (30, 300)
# Reintentos de descarga después de que la imagen ya se haya cobrado por unidad: se
# reintenta primero la dirección original, en lugar de volver a generar la misma imagen.
OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS = 3
OPENAI_IMAGE_DOWNLOAD_BACKOFF_SECONDS = 2


def is_openai_image_enabled(app_config: dict | None = None) -> bool:
    """
    Decide si la fuente de material de texto a imagen compatible con OpenAI tiene la
    configuración mínima.

    La clave API puede estar vacía: un gateway ComfyUI/SD totalmente local normalmente
    no necesita autenticación, y si está vacía la solicitud va sin cabecera
    Authorization. Sirve para que la comprobación previa de tareas y la WebUI
    intercepten las tareas con configuración incompleta antes de gastar saldo del LLM y
    del TTS.
    """
    app_config = config.app if app_config is None else app_config
    return bool(
        str(app_config.get("openai_image_base_url", "") or "").strip()
        and str(app_config.get("openai_image_model", "") or "").strip()
    )


def _openai_image_endpoint() -> tuple[str, str]:
    """
    Lee el endpoint de texto a imagen y el nombre del modelo; si faltan, lanza un error
    con instrucciones de configuración.
    """
    base_url = (
        str(config.app.get("openai_image_base_url", "") or "").strip().rstrip("/")
    )
    model = str(config.app.get("openai_image_model", "") or "").strip()
    if not base_url:
        raise ValueError(
            "\n\n##### openai_image_base_url no está configurado #####\n\n"
            f"Configúralo en el archivo config.toml: {config.config_file}\n"
        )
    if not model:
        raise ValueError(
            "\n\n##### openai_image_model no está configurado #####\n\n"
            f"Configúralo en el archivo config.toml: {config.config_file}\n"
        )
    return f"{base_url}/{OPENAI_IMAGE_ENDPOINT_PATH}", model


def _openai_image_size(video_aspect: VideoAspect) -> str:
    """
    Resuelve el size de imagen de la solicitud.

    La interfaz oficial de OpenAI solo acepta los tamaños que fija el modelo (como
    1024x1536), y pasar directamente la resolución del vídeo (como 1080x1920) devuelve
    400. Por defecto se toma un tamaño compatible según la relación de aspecto;
    ``openai_image_size`` se puede configurar explícitamente para sobrescribirlo, para
    gateways locales que admiten cualquier resolución (como SD WebUI).
    """
    configured = str(config.app.get("openai_image_size", "") or "").strip()
    if configured:
        return configured
    return OPENAI_IMAGE_DEFAULT_SIZES.get(VideoAspect(video_aspect), "1024x1024")


def _openai_image_prompt(search_term: str) -> str:
    """
    Envuelve la palabra clave del guion en el prompt final.

    La opción ``openai_image_prompt_template`` admite el marcador ``{term}``, para
    añadir de forma común modificadores de estilo (como calidad, composición o lenguaje
    de cámara) que mejoran la correspondencia entre imagen y texto:

    .. code-block:: toml

        openai_image_prompt_template = "cinematic photo of {term}, photorealistic"

    Si se deja vacío o no contiene el marcador se vuelve a la palabra clave original,
    con exactamente el mismo comportamiento que la versión antigua. Si la sustitución
    del marcador falla (por ejemplo, si la plantilla tiene mal escrita la sintaxis de
    formato) también se vuelve al texto original, para que un error de configuración no
    interrumpa toda la tarea de generación.
    """
    template = str(config.app.get("openai_image_prompt_template", "") or "").strip()
    if not template or "{term}" not in template:
        return search_term
    try:
        return template.replace("{term}", search_term)
    except Exception:
        return search_term


def _response_json_safely(response: Any) -> Any:
    """Lee el JSON de la respuesta; si falla el análisis con un doble de prueba o una
    respuesta anómala, devuelve None.
    """
    try:
        return response.json()
    except Exception:
        return None


def _openai_image_response_message(body: Any) -> str:
    """
    Extrae una descripción de error legible de una respuesta compatible con OpenAI.

    El formato estándar es ``{"error": {"message": ...}}``, y los servicios intermedios
    a menudo lo degradan a ``{"message": ...}`` o dan directamente una cadena. Si no se
    obtiene ninguno devuelve una cadena vacía, y quien llama decide si vuelve al cuerpo
    de la respuesta.
    """
    if not isinstance(body, dict):
        return str(body or "")[:300]
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "")[:300]
    if error is not None:
        return str(error)[:300]
    return str(body.get("message") or "")[:300]


def _openai_image_http_failure(response: Any, status: int, api_key: str) -> str:
    """Convierte una respuesta de error HTTP en una descripción censurada y legible para el
    log.
    """
    message = _openai_image_response_message(_response_json_safely(response))
    if not message:
        message = str(getattr(response, "text", "") or "")[:300]
    return f"HTTP {status}: {_redact_secret(message, api_key)}"


def _openai_image_download_bytes(
    image_url: str,
    api_key: str,
) -> tuple[bytes | None, str]:
    """
    Descarga la URL temporal de la imagen generada.

    La imagen ya se ha cobrado por unidad, así que si falla la descarga se reintenta
    primero la dirección original en lugar de volver a generarla, para no pagar dos
    veces por la misma imagen. Si fallan todos los reintentos lanza un error, impidiendo
    que la capa exterior compre la siguiente imagen.
    """
    failure_detail = "no se hizo ningún intento de descarga"
    for attempt in range(1, OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS + 1):
        response = None
        try:
            response = trace_http_call(
                "GET", requests.get,
                image_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/115.0.0.0 Safari/537.36"
                },
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(30, 120),
                stream=True,
            )
            if response.status_code == 200:
                try:
                    declared_size = int(response.headers.get("Content-Length", 0))
                except (TypeError, ValueError):
                    declared_size = 0
                if declared_size > OPENAI_IMAGE_MAX_BYTES:
                    return None, "la imagen generada supera el límite de descarga de 25 MB"
                image_bytes = bytearray()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    if len(image_bytes) + len(chunk) > OPENAI_IMAGE_MAX_BYTES:
                        return None, "la imagen generada supera el límite de descarga de 25 MB"
                    image_bytes.extend(chunk)
                if image_bytes:
                    return bytes(image_bytes), ""
                failure_detail = "la descarga de la imagen generada estaba vacía"
            else:
                failure_detail = f"HTTP {response.status_code} al descargar la imagen"
        except Exception as e:
            failure_detail = (
                f"error={type(e).__name__}, "
                f"detail={_redact_request_error(e, api_key, image_url)}"
            )
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception as exc:
                    logger.warning(
                        "no se pudo cerrar la respuesta de descarga de la imagen generada: "
                        f"error={type(exc).__name__}"
                    )
        if attempt < OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS:
            logger.warning(
                "falló la descarga de la imagen generada; se reintenta la misma URL: "
                f"attempt={attempt}/{OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS}, "
                f"{failure_detail}"
            )
            time.sleep(OPENAI_IMAGE_DOWNLOAD_BACKOFF_SECONDS)
    raise OpenAIImagePaidResultError(
        "no se pudo descargar la imagen generada tras los reintentos"
    )


def _parse_openai_image_response(
    response: Any,
    api_key: str,
) -> tuple[bytes | None, str]:
    """
    Analiza la respuesta de /images/generations y obtiene los datos de imagen url o
    b64_json.

    Un fallo de análisis es un rechazo de negocio claro (por ejemplo, por la política de
    contenido) o un formato de respuesta anómalo, así que se devuelve directamente la
    descripción del error, sin reintentos con espera: reenviar la misma solicitud solo
    daría el mismo resultado.
    """
    body = _response_json_safely(response)
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or not data:
        return None, _redact_secret(_openai_image_response_message(body), api_key)

    entry = data[0]
    if not isinstance(entry, dict):
        return None, "entrada de datos de imagen no válida"

    b64_payload = entry.get("b64_json")
    if b64_payload:
        if (
            not isinstance(b64_payload, str)
            or len(b64_payload) > 4 * ((OPENAI_IMAGE_MAX_BYTES + 2) // 3)
        ):
            return None, "la imagen generada supera el límite de respuesta de 25 MB"
        try:
            image_bytes = base64.b64decode(b64_payload)
        except Exception as e:
            return None, f"carga útil b64_json no válida: {type(e).__name__}"
        if len(image_bytes) > OPENAI_IMAGE_MAX_BYTES:
            return None, "la imagen generada supera el límite de respuesta de 25 MB"
        return image_bytes, ""

    image_url = entry.get("url")
    if isinstance(image_url, str) and image_url.startswith(("http://", "https://")):
        return _openai_image_download_bytes(image_url, api_key)

    return None, "la respuesta de imagen no tiene ni url ni b64_json"


def _request_openai_image(endpoint: str, payload: dict) -> tuple[bytes | None, str]:
    """
    Llama a la interfaz /images/generations compatible con OpenAI, con reintentos con
    espera y rotación de claves.

    429/5xx se reintentan con espera como fallos temporales; 401/403 solo se reintentan
    cuando hay varias claves configuradas (cambiando de clave con la rotación de
    get_api_key); los demás 4xx son rechazos claros que fallan rápido y dejan que la
    capa superior se salte esa palabra clave.

    Seguridad de facturación: un tiempo de espera de lectura o un corte de conexión del
    POST se consideran estado "sin confirmar": el servidor puede haber generado y
    cobrado ya, aunque la respuesta no haya llegado, y volver a enviar automáticamente
    podría generar y cobrar dos veces, así que se lanza una excepción propia que termina
    la tarea local. Solo un tiempo de espera en la fase de conexión (ConnectTimeout,
    cuando es seguro que la solicitud no llegó al servidor) confirma que no se creó
    ninguna tarea de generación, y se puede reintentar con seguridad.

    La clave API puede estar vacía: un gateway ComfyUI/SD totalmente local normalmente
    no necesita autenticación, y si está vacía no se envía la cabecera Authorization.
    """
    api_keys = config.app.get("openai_image_api_keys")
    if isinstance(api_keys, (list, tuple)):
        configured_keys = [k for k in api_keys if str(k or "").strip()]
    elif str(api_keys or "").strip():
        configured_keys = [api_keys]
    else:
        configured_keys = []

    failure_detail = "no se hizo ningún intento de petición"
    for attempt in range(1, OPENAI_IMAGE_MAX_ATTEMPTS + 1):
        api_key = get_api_key("openai_image_api_keys") if configured_keys else ""
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        retryable = False
        try:
            response = trace_http_call(
                "POST", requests.post,
                endpoint,
                json=payload,
                headers=headers,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=OPENAI_IMAGE_REQUEST_TIMEOUT,
                allow_redirects=False,
            )
        except requests.exceptions.ConnectTimeout as e:
            # Tiempo de espera en la fase de conexión: es seguro que la solicitud no
            # llegó al servidor y no se creó ninguna tarea de generación, así que se
            # puede reintentar con seguridad.
            failure_detail = (
                f"tiempo de conexión agotado: detail={_redact_request_error(e, api_key)}"
            )
            retryable = True
        except Exception as e:
            # Los tiempos de espera de lectura, los cortes de conexión, etc., son
            # estados "sin confirmar": el servidor puede haber aceptado y cobrado ya, y
            # volver a enviar automáticamente o tratar las palabras clave siguientes
            # podría cobrar dos veces, así que hay que terminar la tarea local.
            raise OpenAIImageUnconfirmedError(
                "petición de imagen sin confirmar (sin reintento para evitar un cobro doble): "
                f"{type(e).__name__}, detail={_redact_request_error(e, api_key)}"
            ) from e
        else:
            status = int(getattr(response, "status_code", 200) or 200)
            if 300 <= status < 400:
                response.close()
                raise OpenAIImageUnconfirmedError(
                    "el envío de la imagen devolvió una redirección; no se envió ninguna "
                    "petición redirigida ni reintento porque el resultado de pago no está confirmado"
                )
            if status in OPENAI_IMAGE_KEY_ERROR_STATUS_CODES:
                failure_detail = _openai_image_http_failure(response, status, api_key)
                # Solo con varias claves configuradas puede el reintento rotar a una
                # clave utilizable.
                retryable = len(configured_keys) > 1
            elif status in OPENAI_IMAGE_RETRYABLE_STATUS_CODES:
                failure_detail = _openai_image_http_failure(response, status, api_key)
                retryable = True
            elif status >= 400:
                return None, _openai_image_http_failure(response, status, api_key)
            else:
                image_bytes, parse_error = _parse_openai_image_response(
                    response, api_key
                )
                if image_bytes is not None:
                    return image_bytes, ""
                failure_detail = parse_error

        if retryable and attempt < OPENAI_IMAGE_MAX_ATTEMPTS:
            backoff_seconds = OPENAI_IMAGE_RETRY_BACKOFF_SECONDS[
                min(attempt - 1, len(OPENAI_IMAGE_RETRY_BACKOFF_SECONDS) - 1)
            ]
            logger.warning(
                "falló la petición de imagen OpenAI, reintentando: "
                f"attempt={attempt}/{OPENAI_IMAGE_MAX_ATTEMPTS}, "
                f"next_retry_in={backoff_seconds}s, detail={failure_detail}"
            )
            time.sleep(backoff_seconds)
            continue
        return None, failure_detail

    return None, failure_detail


def _save_openai_image_file(
    image_bytes: bytes,
    save_dir: str,
) -> tuple[str, int, int]:
    """
    Normaliza el resultado generado a PNG y lo guarda en disco, devolviendo (ruta,
    ancho, alto).

    Convertirlo todo a PNG evita dos tipos de problemas: que un servicio intermedio
    devuelva WebP/JPEG sin una extensión fiable, y que una imagen con metadatos anómalos
    haga fallar el análisis de MoviePy (en línea con la lógica de saneado del material
    local, aquí la normalización se hace ya en la fase de guardado en disco).
    """
    if not save_dir:
        save_dir = utils.storage_dir("cache_images", create=True)
    elif not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    image_path = os.path.join(save_dir, f"openai-image-{uuid.uuid4().hex[:12]}.png")
    if len(image_bytes) > OPENAI_IMAGE_MAX_BYTES:
        raise _OpenAIImageDecodeError("la imagen generada supera el límite de 25 MB")

    # Un fallo al decodificar la imagen puede degradarse a "saltarse la palabra clave
    # actual", pero los fallos de permisos del directorio, de espacio en disco o de
    # escritura del archivo deben seguir lanzándose; si no, el bucle de generación bajo
    # demanda seguiría creando tareas de pago aunque no se puedan guardar archivos en
    # local. Image.open solo lee bytes en memoria, así que aquí un OSError es un fallo
    # al reconocer el formato; un OSError de image.load corresponde a datos de imagen
    # truncados o dañados.
    image = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(image_bytes))
            if image.width * image.height > OPENAI_IMAGE_MAX_PIXELS:
                raise ValueError("la imagen generada supera el límite de 50 millones de píxeles")
            image.load()
    except (
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        if image is not None:
            image.close()
        raise _OpenAIImageDecodeError(f"{type(exc).__name__}: {exc}") from exc

    with image:
        if image.mode not in ("RGB", "RGBA", "L", "LA", "P"):
            image = image.convert("RGB")
        # save no va dentro de la zona protegida de excepciones de decodificación: un
        # error de escritura indica que el entorno de ejecución no está disponible de
        # forma persistente, y hay que terminar de inmediato toda la tarea, para que las
        # palabras clave siguientes no sigan produciendo imágenes de pago que no se
        # pueden guardar.
        image.save(image_path, format="PNG")
        width, height = image.size
    return image_path, width, height


def generate_images_openai(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    save_dir: str = "",
) -> List[MaterialInfo]:
    """
    Genera una imagen para una palabra clave del guion con la interfaz de texto a imagen
    compatible con OpenAI y la guarda en local.

    Mantiene la misma firma y el mismo contrato de fallo con lista vacía que
    generate_videos_wavespeed. Las imágenes no tienen duración propia, así que
    ``duration`` registra la duración objetivo del clip (en segundos), para que el flujo
    de descarga bajo demanda calcule si ya se cubre la duración de la locución. El
    tamaño real que devuelve la API puede no coincidir con el pedido, así que aquí se
    escribe en la rendition el tamaño real de la imagen, sin depender de los parámetros
    de la solicitud.
    """
    aspect = VideoAspect(video_aspect)
    clip_duration = max(int(minimum_duration), 1)
    endpoint, model = _openai_image_endpoint()
    image_size = _openai_image_size(aspect)
    payload = {
        "model": model,
        "prompt": _openai_image_prompt(search_term),
        "n": 1,
        "size": image_size,
    }
    logger.info(
        f"generando imagen con un endpoint compatible con OpenAI: model={model}, "
        f"term={search_term!r}, size={image_size}"
    )
    image_bytes, failure_detail = _request_openai_image(endpoint, payload)
    if image_bytes is None:
        logger.error(
            f"falló la generación de imagen OpenAI: term={search_term!r}, "
            f"detail={failure_detail}"
        )
        return []

    try:
        image_path, width, height = _save_openai_image_file(image_bytes, save_dir)
    except _OpenAIImageDecodeError as e:
        # La capa compatible puede devolver 200 pero con un cuerpo que no es una imagen
        # (por ejemplo, una página de error HTML disfrazada de JSON o una página de
        # aviso de degradación del gateway). Que la imagen no se pueda decodificar
        # significa que "esta generación ha fallado", así que según el contrato de las
        # fuentes de material se devuelve una lista vacía para que la capa superior se
        # salte esa palabra clave y siga, en lugar de dejar que la excepción interrumpa
        # toda la tarea.
        logger.error(
            "la respuesta de imagen OpenAI no es una imagen decodificable; se omite el término: "
            f"term={search_term!r}, error={type(e).__name__}, detail={e}"
        )
        return []
    item = MaterialInfo()
    item.provider = "openai_image"
    item.url = image_path
    item.duration = clip_duration
    item.source_info = {
        "provider": "openai_image",
        "search_term": search_term,
        "rendition": {
            "id": None,
            "width": width,
            "height": height,
        },
    }
    return [item]


def _render_openai_image_video(
    image_path: str, clip_duration: int, image_motion: ImageMotion = ImageMotion.zoom
) -> str:
    """
    Renderiza la imagen generada como un clip mp4, reutilizando el flujo "imagen → clip
    dinámico" del material local.

    Si falla el renderizado devuelve una cadena vacía, y quien llama detiene las
    solicitudes de pago siguientes.
    """
    try:
        return video.render_image_zoom_video(
            image_path, clip_duration, image_motion=image_motion
        )
    except Exception as e:
        logger.error(
            "no se pudo convertir la imagen generada en un clip de vídeo: "
            f"image={image_path}, error={type(e).__name__}, detail={e}"
        )
        return ""


def _download_videos_openai_image_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
    image_motion: ImageMotion = ImageMotion.zoom,
) -> List[str]:
    """
    Genera imagen a imagen, en el orden de los fragmentos del guion, material de texto a
    imagen compatible con OpenAI, y se detiene en cuanto se cubre la duración total
    necesaria.

    La misma semántica de seguridad de pago que la generación bajo demanda de WaveSpeed:
    el texto a imagen se cobra por imagen, y generar primero todo para elegir después
    pagaría imágenes que no se usan. Cada imagen se renderiza nada más generarse como
    clip mp4 y se acumula su duración válida (igual que en el flujo de stock, limitada
    por la duración del clip); cuando la suma alcanza la duración de locución necesaria
    ya no se lanzan nuevas solicitudes de pago. Las imágenes rechazadas explícitamente
    se pueden saltar; si el resultado de una solicitud es incierto, falla la descarga de
    un resultado pagado o falla el renderizado local, hay que terminar la tarea, para
    que las palabras clave siguientes no vuelvan a cobrar.
    """
    # Aunque se llame directamente al servicio de material, hay que validar antes de la
    # solicitud de pago, para no descubrir después de generar que los parámetros del
    # efecto no son válidos.
    image_motion = ImageMotion(image_motion)
    if not material_directory:
        # Las imágenes generadas se cobran por tarea y no se pueden reutilizar, así que
        # por defecto se guardan en el directorio de la tarea para facilitar su
        # trazabilidad.
        material_directory = utils.task_dir(task_id)

    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []
    total_duration = 0.0

    # Una duración de locución no positiva haría que la comprobación de "alcanzar la
    # duración necesaria" perdiera sentido, así que se vuelve sin nada, para no seguir
    # pagando por imagen en una tarea que nunca se podría completar (igual que la
    # semántica de la comprobación previa de Seedance).
    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError):
        required_duration = 0.0
    if required_duration <= 0:
        logger.warning(
            "se omite la generación de imágenes OpenAI porque la duración de audio "
            f"necesaria no es positiva: duration={audio_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    for search_term in search_terms:
        try:
            items = generate_images_openai(
                search_term=search_term,
                minimum_duration=max_clip_duration,
                video_aspect=video_aspect,
                save_dir=material_directory,
            )
        except OpenAIImagePaidResultError:
            _persist_material_sources(task_id, material_sources)
            raise
        for item in items:
            video_file = _render_openai_image_video(
                item.url, max_clip_duration, image_motion=image_motion
            )
            if not video_file:
                _persist_material_sources(task_id, material_sources)
                raise OpenAIImagePaidResultError(
                    "no se pudo renderizar en local la imagen generada"
                )
            logger.info(f"material de imagen renderizado: {video_file}")
            video_paths.append(video_file)
            try:
                material_sources.append(_material_source_record(item, video_file))
            except Exception as source_error:
                # Igual que con las fuentes de stock: una excepción del registro de
                # origen no puede tratar como fallido un material ya generado pagando y
                # renderizado con éxito, ni mucho menos bloquear la generación del
                # vídeo.
                logger.warning(
                    "no se pudo preparar el registro de origen del material generado: "
                    f"provider=openai_image, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            total_duration += min(max_clip_duration, item.duration)
            # La misma comparación >= que WaveSpeed: generar otra imagen cuando ya se
            # cubre justo sería pagar una vez más. Las dos comprobaciones, interior y
            # exterior, deben mantener la misma semántica.
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "el material de imagen generado cubre la duración necesaria; no se "
                f"generan más imágenes: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break

    logger.success(f"se generaron y renderizaron {len(video_paths)} materiales de imagen")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _search_videos_with_cache(
    provider: str,
    search_videos: Callable[..., List[MaterialInfo]],
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect,
) -> List[MaterialInfo]:
    """
    Trata de forma común la caché de búsqueda de 24 horas de las tres fuentes de
    material en línea.

    La caché solo envuelve la API de búsqueda, sin cambiar la lógica posterior de
    descarga de vídeo ni la deduplicación. Cuando el servidor devuelve una lista vacía
    no se escribe la caché, porque la interfaz actual de proveedores usa la lista vacía
    tanto para "sin resultados" como para "la solicitud falló"; mientras no se separen
    en tipos de resultado explícitos, es preferible reintentar la próxima vez antes que
    guardar un fallo temporal durante un día.
    """
    cache_args = {
        "provider": provider,
        "search_term": search_term,
        "minimum_duration": minimum_duration,
        "video_aspect": video_aspect,
    }

    def load_cache_safely() -> List[MaterialInfo] | None:
        try:
            return material_cache.load_material_search_cache(**cache_args)
        except Exception as exc:
            # La caché es una optimización opcional; cualquier excepción de su
            # implementación debe tratarse como fallo de caché, sin bloquear la búsqueda
            # remota normal de Pexels, Pixabay o Coverr.
            logger.warning(
                "falló la lectura de la caché de búsqueda de material; se sigue con la búsqueda remota: "
                f"provider={provider}, error={type(exc).__name__}, detail={exc}"
            )
            return None

    def load_matching_cache() -> tuple[List[MaterialInfo] | None, int]:
        cached_items = load_cache_safely()
        if cached_items is None:
            return None, 0

        filtered_cached_items = _filter_materials_by_aspect(
            cached_items,
            video_aspect,
        )
        ignored_count = len(cached_items) - len(filtered_cached_items)
        if ignored_count:
            # La caché de versiones antiguas puede mezclar material de otra orientación.
            # Aunque queden unas pocas entradas utilizables, hay que refrescar el
            # conjunto completo de candidatos; si no, durante la validez de la caché se
            # usaría una y otra vez el mismo puñado de vídeos.
            return None, ignored_count
        return filtered_cached_items, 0

    cached_items, ignored_count = load_matching_cache()
    if cached_items is not None:
        return cached_items
    if ignored_count:
        logger.info(
            "la caché de búsqueda de material contiene orientaciones que no coinciden; "
            f"se refresca desde el proveedor: provider={provider}, term={search_term!r}, "
            f"ignored={ignored_count}"
        )

    cache_lock = material_cache.get_material_search_cache_lock(**cache_args)
    with cache_lock:
        # Espera a que termine el hilo con las mismas condiciones de búsqueda y vuelve a
        # leer, para que varias tareas de la API no pidan a la vez al servidor en el
        # primer fallo de caché, reduciendo la probabilidad de activar los límites y el
        # control de riesgos de la interfaz de terceros.
        cached_items, _ = load_matching_cache()
        if cached_items is not None:
            return cached_items

        items = search_videos(
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )
        # Normalmente el proveedor escribe la palabra clave actual, pero los dobles de
        # prueba, las extensiones de terceros o las implementaciones antiguas pueden
        # omitirla o traer un valor erróneo. La lectura de la caché recupera ese campo a
        # partir de la clave de caché, así que los resultados remotos también se
        # corrigen en la misma entrada, para que el registro de origen de las tareas sea
        # igual en la primera búsqueda y en los aciertos de caché.
        for item in items:
            if isinstance(item.source_info, dict):
                item.source_info = dict(item.source_info)
                item.source_info["search_term"] = search_term
        if items:
            try:
                material_cache.save_material_search_cache(
                    **cache_args,
                    items=items,
                )
            except Exception as exc:
                logger.warning(
                    "falló la escritura de la caché de búsqueda de material; se usan los resultados remotos: "
                    f"provider={provider}, error={type(exc).__name__}, detail={exc}"
                )
        return items


def _search_terms_in_parallel(
    search_terms: List[str],
    search_videos: Callable[[str, int, VideoAspect], List[MaterialInfo]],
    minimum_duration: int,
    video_aspect: VideoAspect,
) -> list[tuple[str, List[MaterialInfo]]]:
    """Busca las palabras clave en paralelo; las palabras clave que fallan devuelven un
    resultado vacío sin bloquear el resto del material.
    """
    if not search_terms:
        return []

    workers = min(_get_material_concurrency(), len(search_terms))
    if workers == 1:
        results = []
        for search_term in search_terms:
            try:
                items = search_videos(
                    search_term=search_term,
                    minimum_duration=minimum_duration,
                    video_aspect=video_aspect,
                )
            except Exception as exc:
                logger.error(
                    "no se pudieron buscar vídeos de material: "
                    f"search_term={search_term!r}, "
                    f"error={type(exc).__name__}"
                )
                items = []
            logger.info(f"se encontraron {len(items)} vídeos para '{search_term}'")
            results.append((search_term, items))
        return results

    futures = {}
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="material-search",
    ) as executor:
        for search_term in search_terms:
            futures[executor.submit(
                logging_utils.bind_log_scope(search_videos),
                search_term,
                minimum_duration,
                video_aspect,
            )] = search_term

        results = []
        for future in futures:
            search_term = futures[future]
            try:
                items = future.result()
            except Exception as exc:
                logger.error(
                    "no se pudieron buscar vídeos de material: "
                    f"search_term={search_term!r}, "
                    f"error={type(exc).__name__}, detail={exc}"
                )
                items = []
            logger.info(f"se encontraron {len(items)} vídeos para '{search_term}'")
            results.append((search_term, items))
        return results


def _report_material_downloaded(
    position: int,
    total: int,
    item: MaterialInfo,
    saved_video_path: str,
    on_downloaded: Callable[[MaterialInfo], None] | None,
) -> None:
    """
    Registra que un material ha terminado de descargarse y avisa a quien llama para que
    actualice el progreso.

    Siempre se llama en el hilo que lanzó la descarga, así que el log por archivo y la
    actualización del progreso no dependen de la concurrencia de descarga.
    """
    description = os.path.basename(saved_video_path)
    try:
        size_mb = os.path.getsize(saved_video_path) / _BYTES_PER_MEGABYTE
        description += f" ({size_mb:.1f} MB)"
    except OSError:
        # El tamaño del archivo es solo información auxiliar; si no se puede leer no
        # afecta al resultado de la descarga.
        pass
    logger.info(f"material descargado {position}/{total}: {description}")
    if on_downloaded is None:
        return
    try:
        on_downloaded(item)
    except Exception as exc:
        # El progreso es solo información para mostrar. Un fallo del callback (por
        # ejemplo, que el backend de estado no esté disponible temporalmente) no puede
        # invalidar un material ya descargado con éxito, ni mucho menos interrumpir toda
        # la tarea.
        logger.warning(
            "no se pudo informar del progreso de descarga del material: "
            f"error={type(exc).__name__}, detail={exc}"
        )


def _covered_duration_reporter(
    progress_callback: Callable[[float], None] | None,
    required_duration: float,
    max_clip_duration: int,
) -> Callable[[MaterialInfo], None] | None:
    """
    Genera el callback por material: convierte la "duración de locución ya cubierta" en
    un progreso de descarga de 0~1.

    El criterio coincide con la condición de parada del bucle de descarga: cada material
    aporta como mucho la duración de un clip, y al cubrir la duración necesaria se
    termina, así que lo que la supera se limita a 1,0. Si no hay callback de progreso
    devuelve None.
    """
    if progress_callback is None:
        return None
    covered_duration = 0.0

    def report(item: MaterialInfo) -> None:
        nonlocal covered_duration
        covered_duration += min(max_clip_duration, item.duration)
        fraction = 1.0
        if required_duration > 0:
            fraction = min(1.0, covered_duration / required_duration)
        progress_callback(fraction)

    return report


def _download_materials_in_parallel(
    materials: List[tuple[str, MaterialInfo]],
    material_directory: str,
    on_downloaded: Callable[[MaterialInfo], None] | None = None,
) -> list[tuple[str, MaterialInfo, str]]:
    """
    Descarga en paralelo una ronda de material, conservando el orden de candidatos que
    pasa quien llama.

    Tras descargar con éxito cada material se registra un log y se llama a
    ``on_downloaded``, con el que quien llama actualiza el progreso de la tarea. Tanto
    el callback como el log ocurren en el hilo que llama a esta función.
    """
    if not materials:
        return []

    total = len(materials)
    workers = min(_get_material_concurrency(), total)
    if workers == 1:
        downloaded = []
        for position, (search_term, item) in enumerate(materials, start=1):
            try:
                saved_video_path = save_video(
                    video_url=item.url,
                    save_dir=material_directory,
                )
            except Exception as exc:
                logger.error(
                    "no se pudo descargar el vídeo de material: "
                    f"provider={item.provider}, "
                    f"error={type(exc).__name__}, "
                    f"detail={_redact_request_error(exc, item.url)}"
                )
                continue
            if saved_video_path:
                downloaded.append((search_term, item, saved_video_path))
                _report_material_downloaded(
                    position, total, item, saved_video_path, on_downloaded
                )
        return downloaded

    futures = {}
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="material-download",
    ) as executor:
        for search_term, item in materials:
            futures[executor.submit(
                logging_utils.bind_log_scope(save_video),
                item.url,
                material_directory,
            )] = (search_term, item)

        downloaded = []
        for position, future in enumerate(futures, start=1):
            search_term, item = futures[future]
            try:
                saved_video_path = future.result()
            except Exception as exc:
                logger.error(
                    "no se pudo descargar el vídeo de material: "
                    f"provider={item.provider}, "
                    f"error={type(exc).__name__}, "
                    f"detail={_redact_request_error(exc, item.url)}"
                )
                continue
            if saved_video_path:
                downloaded.append((search_term, item, saved_video_path))
                _report_material_downloaded(
                    position, total, item, saved_video_path, on_downloaded
                )
        return downloaded


def _select_materials_until_duration(
    materials: List[tuple[str, MaterialInfo]],
    max_clip_duration: int,
    audio_duration: float,
    current_duration: float = 0.0,
) -> List[tuple[str, MaterialInfo]]:
    """
    Elige una ronda de material en el orden de los candidatos, para que con concurrencia
    no se descarguen candidatos que ya superan la duración de la locución.

    Devuelve un subconjunto prefijo de ``materials``: acumula desde el primer candidato
    y se detiene la primera vez que la duración acumulada supera la de la locución
    (incluyendo el que la supera). Quien llama usa esa propiedad de prefijo para saber
    qué candidatos se intentaron realmente en esta ronda.
    """
    selected = []
    total_duration = current_duration
    for material in materials:
        selected.append(material)
        total_duration += min(max_clip_duration, material[1].duration)
        if total_duration > audio_duration:
            break
    return selected


def download_videos(
    task_id: str,
    search_terms: List[str],
    source: str = "pexels",
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    audio_duration: float = 0.0,
    max_clip_duration: int = 5,
    match_script_order: bool = False,
    progress_callback: Callable[[float], None] | None = None,
    image_motion: ImageMotion = ImageMotion.zoom,
) -> List[str]:
    """
    Busca y descarga el material necesario para cubrir la duración de la locución, y
    devuelve las rutas de los archivos locales.

    ``progress_callback`` es opcional: el material de stock (Pexels, Pixabay, Coverr) lo
    llama una vez por cada archivo descargado con la proporción de avance 0~1, para que
    la capa de tareas haga avanzar la barra de progreso. Las fuentes de material de pago
    generadas bajo demanda no informan del progreso de descarga.
    """
    provider = "pexels"
    remote_search_videos = search_videos_pexels
    if source == "pixabay":
        provider = "pixabay"
        remote_search_videos = search_videos_pixabay
    elif source == "coverr":
        provider = "coverr"
        remote_search_videos = search_videos_coverr

    def search_videos(
        search_term: str,
        minimum_duration: int,
        video_aspect: VideoAspect,
    ) -> List[MaterialInfo]:
        return _search_videos_with_cache(
            provider=provider,
            search_videos=remote_search_videos,
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )

    material_directory = config.app.get("material_directory", "").strip()
    if material_directory == "task":
        material_directory = utils.task_dir(task_id)
    elif material_directory and not os.path.isdir(material_directory):
        material_directory = ""

    if source == "infistar":
        return _download_videos_infistar_on_demand(
            task_id=task_id, search_terms=search_terms, video_aspect=video_aspect,
            audio_duration=audio_duration, max_clip_duration=max_clip_duration,
        )
    if source == "wavespeed":
        # La generación con IA se cobra por unidad, así que no puede seguir el flujo de
        # las fuentes de stock de "obtener primero los candidatos de todas las palabras
        # clave y elegir después", o se pagarían clips que no se usan. Las fuentes de
        # generación pasan a generar segmento a segmento bajo demanda y se detienen en
        # cuanto se cubre la duración necesaria; tampoco entran en la caché de búsqueda
        # de 24 horas: la URL del producto es una dirección firmada que caduca, y
        # reutilizar la caché haría que tareas distintas obtuvieran una y otra vez el
        # mismo vídeo generado.
        return _download_videos_wavespeed_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source == "volcengine_seedance":
        # Igual que WaveSpeed, la interfaz oficial de Ark crea tareas de pago
        # asíncronas. Hay que generar segmento a segmento bajo demanda, comprando solo
        # el material que necesita de verdad la duración de la locución actual.
        return _download_videos_seedance_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source == "ofox":
        # La misma semántica de pago bajo demanda que WaveSpeed/Ark: /v1/videos del
        # gateway OFox crea tareas de pago asíncronas, así que hay que generar segmento
        # a segmento y detenerse en cuanto se cubre la duración necesaria; la dirección
        # del producto es un enlace directo temporal que caduca, así que tampoco entra
        # en la caché de búsqueda de 24 horas.
        return _download_videos_ofox_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source == "muapi":
        # Los endpoints de modelo de MuAPI crean tareas de pago asíncronas. Se envían
        # segmento a segmento y se detienen al cubrir la duración de la locución, para
        # no enviar al servidor palabras clave del guion que no se usan.
        return _download_videos_muapi_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source == "metaso_minimax":
        # Metaso MiniMax también factura por tarea asíncrona remota. Su cuerpo de
        # solicitud se parece al de Volcengine Ark, pero la ruta de consulta de tareas y
        # la estructura de respuesta son distintas, así que solo comparte la semántica
        # local de generación bajo demanda, sin reutilizar el cliente del proveedor,
        # para que las diferencias de protocolo no se filtren a la capa de orquestación
        # del material.
        return _download_videos_metaso_minimax_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source == "openai_image":
        # La misma semántica de pago bajo demanda que WaveSpeed: el texto a imagen se
        # cobra por imagen, así que se genera segmento a segmento y se detiene en cuanto
        # se cubre la duración necesaria. El resultado generado es un archivo de imagen
        # local de un solo uso, así que tampoco entra en la caché de búsqueda de 24
        # horas: la caché haría que tareas distintas obtuvieran una y otra vez la misma
        # imagen.
        return _download_videos_openai_image_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
            image_motion=image_motion,
        )

    if match_script_order:
        return _download_videos_by_script_order(
            task_id=task_id,
            search_terms=search_terms,
            search_videos=search_videos,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
            progress_callback=progress_callback,
        )

    valid_video_items = []
    valid_video_urls = []
    found_duration = 0.0
    for search_term, video_items in _search_terms_in_parallel(
        search_terms=search_terms,
        search_videos=search_videos,
        minimum_duration=max_clip_duration,
        video_aspect=video_aspect,
    ):
        for item in video_items:
            if item.url not in valid_video_urls:
                valid_video_items.append(item)
                valid_video_urls.append(item.url)
                found_duration += item.duration

    logger.info(
        f"vídeos encontrados en total: {len(valid_video_items)}, duración necesaria: {audio_duration} segundos, duración encontrada: {found_duration} segundos"
    )
    video_paths = []
    material_sources: list[dict[str, Any]] = []

    concat_mode_value = getattr(video_concat_mode, "value", video_concat_mode)
    if concat_mode_value == VideoConcatMode.random.value:
        random.shuffle(valid_video_items)

    total_duration = 0.0
    on_downloaded = _covered_duration_reporter(
        progress_callback, audio_duration, max_clip_duration
    )
    pending_items = list(valid_video_items)
    # La ruta por defecto/aleatoria también la controla la configuración de concurrencia
    # del material: cada ronda preselecciona los candidatos con la condición de parada
    # de la lógica en serie (acumular en orden y detenerse la primera vez que se supera
    # la duración de la locución). Si todas las descargas tienen éxito, el conjunto
    # descargado en cada ronda coincide exactamente con el de la lógica en serie; los
    # candidatos cuya descarga falla se completan en la ronda siguiente con candidatos
    # posteriores.
    while pending_items and total_duration <= audio_duration:
        batch = []
        projected_duration = total_duration
        for item in pending_items:
            batch.append(item)
            projected_duration += min(max_clip_duration, item.duration)
            if projected_duration > audio_duration:
                break
        for item in batch:
            source_info = item.source_info if isinstance(item.source_info, dict) else {}
            logger.info(
                f"descargando vídeo de {item.provider}: "
                f"asset_id={source_info.get('asset_id') or 'unknown'}"
            )
        downloaded_materials = _download_materials_in_parallel(
            materials=[("", item) for item in batch],
            material_directory=material_directory,
            on_downloaded=on_downloaded,
        )
        pending_items = pending_items[len(batch):]
        for _, item, saved_video_path in downloaded_materials:
            try:
                if saved_video_path:
                    logger.info(f"vídeo guardado: {saved_video_path}")
                    video_paths.append(saved_video_path)
                    try:
                        material_sources.append(
                            _material_source_record(item, saved_video_path)
                        )
                    except Exception as source_error:
                        # Una excepción del registro de origen no puede tratar un
                        # material ya descargado con éxito como fallo de descarga, ni
                        # mucho menos bloquear la generación del vídeo; se conservan el
                        # proveedor y el tipo de excepción para localizarlo después.
                        logger.warning(
                            "no se pudo preparar el registro de origen del material: "
                            f"provider={item.provider}, "
                            f"error={type(source_error).__name__}, detail={source_error}"
                        )
                    seconds = min(max_clip_duration, item.duration)
                    total_duration += seconds
                    if total_duration > audio_duration:
                        logger.info(
                            f"duración total de los vídeos descargados: {total_duration} segundos; no se descargan más"
                        )
                        break
            except Exception as e:
                logger.error(
                    "no se pudo descargar el vídeo de material: "
                    f"provider={item.provider}, error={type(e).__name__}, "
                    f"detail={_redact_request_error(e, item.url)}"
                )
    logger.success(f"se descargaron {len(video_paths)} vídeos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_wavespeed_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """
    Genera segmento a segmento, en el orden de los fragmentos del guion, material de
    WaveSpeed, y se detiene en cuanto se cubre la duración total necesaria.

    Cada palabra clave corresponde por naturaleza a un fragmento del guion, y generar es
    pagar: generar todo primero para elegir después pagaría fragmentos que no se usan.
    Aquí, cada segmento generado se descarga de inmediato y se acumula su duración
    válida (igual que en el flujo de stock, limitada por la duración del clip), y cuando
    la suma cubre la duración de locución necesaria ya no se lanzan nuevas solicitudes
    de generación. Los fragmentos que fallan explícitamente en remoto se pueden saltar;
    las tareas de pago con estado desconocido o con descarga fallida deben conservar su
    ID y terminar la tarea local.
    """
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []

    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("la duración del audio para WaveSpeed debe ser finita") from exc
    if not math.isfinite(required_duration):
        raise ValueError("la duración del audio para WaveSpeed debe ser finita")
    if required_duration <= 0:
        logger.warning(
            "se omite la generación de pago de WaveSpeed porque la duración de audio "
            f"necesaria no es positiva: duration={required_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("la duración del clip de WaveSpeed debe ser un entero positivo") from exc
    if clip_duration <= 0:
        raise ValueError("la duración del clip de WaveSpeed debe ser un entero positivo")

    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = generate_videos_wavespeed(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except WaveSpeedUnconfirmedTaskError as e:
            # El estado de una tarea de pago ya enviada es desconocido: puede seguir
            # ejecutándose en remoto o haber terminado y facturado. Seguir haciendo
            # pedidos para las palabras clave siguientes generaría y cobraría dos veces,
            # así que se detiene en el acto y se pasa el prediction id a la capa de
            # tareas para que lo escriba en el estado de fallo y el usuario pueda
            # recuperar el producto.
            logger.error(
                "se dejan de enviar tareas nuevas a WaveSpeed: la última tarea enviada "
                f"no está confirmada: prediction_id={e.prediction_id or 'unknown'}, "
                f"detail={e}"
            )
            _persist_material_sources(task_id, material_sources)
            raise
        for item in video_items:
            saved_video_path = _save_generated_video_with_retry(
                item.url, material_directory, "wavespeed"
            )
            if not saved_video_path:
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                prediction_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise WaveSpeedDownloadError(
                    "WaveSpeed generó un vídeo de pago pero no se pudo descargar el "
                    f"resultado: id={prediction_id or 'unknown'}",
                    prediction_id=prediction_id,
                )
            logger.info(f"vídeo guardado: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(_material_source_record(item, saved_video_path))
            except Exception as source_error:
                # Igual que con las fuentes de stock: una excepción del registro de
                # origen no puede tratar como fallido un material ya generado pagando y
                # descargado con éxito, ni mucho menos bloquear la generación del vídeo.
                logger.warning(
                    "no se pudo preparar el registro de origen del material: "
                    f"provider={item.provider}, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            total_duration += min(clip_duration, item.duration)
            # Se usa >=: cuando la duración acumulada es exactamente igual a la
            # necesaria ya basta, y generar otro costaría un pago más. Las dos
            # comprobaciones, interior y exterior, deben mantener la misma semántica.
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "el material generado cubre la duración necesaria; no se generan "
                f"más clips: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break
    logger.success(f"se generaron y descargaron {len(video_paths)} vídeos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_seedance_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Genera en orden material Seedance de Ark y deja de hacer pedidos de pago en cuanto
    cubre la duración de la locución.
    """
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []

    # El bucle de generación de pago debe validar primero las dos duraciones que
    # controlan el número de vueltas. NaN/Infinity harían que ``total_duration >=
    # audio_duration`` nunca se cumpliera, y una duración de clip no positiva haría que
    # la suma no creciera; en ambos casos se podrían crear tareas de pago inútiles para
    # todas las palabras clave.
    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError) as exc:
        raise volcengine_seedance.VolcEngineSeedanceError(
            "la duración del audio para Seedance debe ser un número finito"
        ) from exc
    if not math.isfinite(required_duration):
        raise volcengine_seedance.VolcEngineSeedanceError(
            "la duración del audio para Seedance debe ser un número finito"
        )
    if required_duration <= 0:
        logger.warning(
            "se omite la generación de pago de Seedance porque la duración de audio "
            f"necesaria no es positiva: duration={required_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise volcengine_seedance.VolcEngineSeedanceError(
            "la duración del clip de Seedance debe ser un entero positivo"
        ) from exc
    if clip_duration <= 0:
        raise volcengine_seedance.VolcEngineSeedanceError(
            "la duración del clip de Seedance debe ser un entero positivo"
        )

    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = volcengine_seedance.generate_videos(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except volcengine_seedance.VolcEngineSeedanceUnconfirmedTaskError as exc:
            # La tarea de pago remota aún puede tener éxito. Se deja de hacer pedidos de
            # inmediato y se conserva el ID de la tarea, para que el usuario pueda
            # confirmar o recuperar después el resultado en la consola de Ark.
            logger.error(
                "se dejan de enviar tareas nuevas a Seedance porque la última tarea de "
                f"pago no está confirmada: task_id={exc.task_id or 'unknown'}, detail={exc}"
            )
            _persist_material_sources(task_id, material_sources)
            raise
        except volcengine_seedance.VolcEngineSeedanceError as exc:
            logger.error(f"la generación con Seedance falló antes de terminar: {exc}")
            _persist_material_sources(task_id, material_sources)
            raise

        for item in video_items:
            saved_video_path = _save_generated_video_with_retry(
                item.url, material_directory, "volcengine_seedance"
            )
            if not saved_video_path:
                # La tarea remota ya ha terminado y ha generado coste; si falla la
                # descarga local, hay que devolver el ID de la tarea remota al estado de
                # la tarea, para que el usuario pueda recuperar el resultado en la
                # consola de Ark. Aquí se lanza directamente un error propio, impidiendo
                # a la vez que las palabras clave siguientes creen nuevas tareas de
                # pago.
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise volcengine_seedance.VolcEngineSeedanceDownloadError(
                    "Seedance generó un vídeo de pago pero no se pudo descargar el "
                    f"resultado: id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                )
            logger.info(f"vídeo guardado: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(_material_source_record(item, saved_video_path))
            except Exception as source_error:
                logger.warning(
                    "no se pudo preparar el registro de origen del material generado: "
                    f"provider=volcengine_seedance, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            total_duration += min(clip_duration, item.duration)
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "el material generado con Seedance cubre la duración necesaria; no se "
                f"envían más tareas de pago: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break

    logger.success(
        f"se generaron y descargaron {len(video_paths)} vídeos de Volcano Engine Seedance"
    )
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_ofox_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Genera en orden material de OFox y deja de hacer pedidos de pago en cuanto cubre la
    duración de la locución.
    """
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []

    # El bucle de generación de pago debe validar primero las dos duraciones que
    # controlan el número de vueltas. NaN/Infinity harían que ``total_duration >=
    # audio_duration`` nunca se cumpliera, y una duración de clip no positiva haría que
    # la suma no creciera; en ambos casos se podrían crear tareas de pago inútiles para
    # todas las palabras clave.
    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError) as exc:
        raise ofox.OFoxError("la duración del audio para OFox debe ser un número finito") from exc
    if not math.isfinite(required_duration):
        raise ofox.OFoxError("la duración del audio para OFox debe ser un número finito")
    if required_duration <= 0:
        logger.warning(
            "se omite la generación de pago de OFox porque la duración de audio "
            f"necesaria no es positiva: duration={required_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ofox.OFoxError("la duración del clip de OFox debe ser un entero positivo") from exc
    if clip_duration <= 0:
        raise ofox.OFoxError("la duración del clip de OFox debe ser un entero positivo")

    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = ofox.generate_videos(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except ofox.OFoxUnconfirmedTaskError as exc:
            # La tarea de pago remota aún puede tener éxito. Se deja de hacer pedidos de
            # inmediato y se conserva el ID de la tarea, para que el usuario pueda
            # confirmar o recuperar después el resultado en la consola de OFox.
            logger.error(
                "se dejan de enviar tareas nuevas a OFox porque la última tarea de "
                f"pago no está confirmada: task_id={exc.task_id or 'unknown'}, detail={exc}"
            )
            _persist_material_sources(task_id, material_sources)
            raise
        except ofox.OFoxError as exc:
            logger.error(f"la generación con OFox falló antes de terminar: {exc}")
            _persist_material_sources(task_id, material_sources)
            raise

        # Cuando el servidor da explícitamente por fallida una palabra clave (por
        # ejemplo, por moderación de contenido), se devuelve una lista vacía: la tarea
        # ha terminado, no hay dudas de facturación, y se salta ese clip para seguir
        # generando las palabras clave siguientes.
        for item in video_items:
            saved_video_path = _save_generated_video_with_retry(
                item.url, material_directory, "ofox"
            )
            if not saved_video_path:
                # La tarea remota ya ha terminado y ha generado coste; si falla la
                # descarga local, hay que devolver el ID de la tarea remota al estado de
                # la tarea, para que el usuario pueda recuperar el resultado en la
                # consola de OFox. Aquí se lanza directamente un error propio,
                # impidiendo a la vez que las palabras clave siguientes creen nuevas
                # tareas de pago.
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise ofox.OFoxDownloadError(
                    "OFox generó un vídeo de pago pero no se pudo descargar el "
                    f"resultado: id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                )
            logger.info(f"vídeo guardado: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(_material_source_record(item, saved_video_path))
            except Exception as source_error:
                logger.warning(
                    "no se pudo preparar el registro de origen del material generado: "
                    f"provider=ofox, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            total_duration += min(clip_duration, item.duration)
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "el material generado con OFox cubre la duración necesaria; no se "
                f"envían más tareas de pago: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break

    logger.success(f"se generaron y descargaron {len(video_paths)} vídeos de OFox")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_muapi_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Genera material de MuAPI hasta cubrir la duración de la narración."""
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []

    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError) as exc:
        raise muapi.MuAPIError(
            "la duración del audio para MuAPI debe ser un número finito"
        ) from exc
    if not math.isfinite(required_duration):
        raise muapi.MuAPIError("la duración del audio para MuAPI debe ser un número finito")
    if required_duration <= 0:
        logger.warning(
            "se omite la generación de pago de MuAPI porque la duración de audio "
            f"necesaria no es positiva: duration={required_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise muapi.MuAPIError(
            "la duración del clip de MuAPI debe ser un entero positivo"
        ) from exc
    if clip_duration <= 0:
        raise muapi.MuAPIError("la duración del clip de MuAPI debe ser un entero positivo")

    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = muapi.generate_videos(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except muapi.MuAPIUnconfirmedTaskError as exc:
            # Un envío de pago ambiguo debe detener el bucle: reintentar con la
            # siguiente palabra clave podría crear un cobro duplicado. El ID de la tarea
            # se conserva para la recuperación del servicio de tareas cuando el
            # proveedor devolvió uno.
            logger.error(
                "se dejan de enviar tareas nuevas a MuAPI porque la última tarea de "
                f"pago no está confirmada: task_id={exc.task_id or 'unknown'}, "
                f"detail={exc}"
            )
            _persist_material_sources(task_id, material_sources)
            raise
        except muapi.MuAPIError as exc:
            logger.error(f"la generación con MuAPI falló antes de terminar: {exc}")
            _persist_material_sources(task_id, material_sources)
            raise

        for item in video_items:
            saved_video_path = _save_generated_video_with_retry(
                item.url, material_directory, "muapi"
            )
            if not saved_video_path:
                # El trabajo remoto terminó y puede haber generado ya un cargo. No se
                # envía una tarea de sustitución tras un fallo de descarga; en su lugar
                # se devuelve el ID remoto para recuperarlo a mano.
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise muapi.MuAPIDownloadError(
                    "MuAPI generó un vídeo de pago pero no se pudo descargar el "
                    f"resultado: id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                )
            logger.info(f"vídeo guardado: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(_material_source_record(item, saved_video_path))
            except Exception as source_error:
                logger.warning(
                    "no se pudo preparar el registro de origen del material generado: "
                    f"provider=muapi, error={type(source_error).__name__}, "
                    f"detail={source_error}"
                )
            try:
                downloaded_duration = _get_downloaded_video_duration(saved_video_path)
            except Exception as duration_error:
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise muapi.MuAPIDownloadError(
                    "MuAPI generó un vídeo de pago pero no se pudo medir la duración "
                    "de la descarga: "
                    f"id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                ) from duration_error
            total_duration += min(clip_duration, downloaded_duration)
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "el material generado con MuAPI cubre la duración necesaria; no se "
                f"envían más tareas de pago: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break

    logger.success(f"se generaron y descargaron {len(video_paths)} vídeos de MuAPI")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_metaso_minimax_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Genera en orden material de Metaso MiniMax y deja de hacer pedidos de pago en cuanto
    cubre la duración de la locución.
    """
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []

    # El servidor genera como mínimo 4 segundos, pero en local se sigue recortando y
    # acumulando según la duración de clip del usuario. Se validan antes los parámetros
    # que controlan el bucle, para que NaN, Infinity o valores no positivos no hagan que
    # la condición de parada nunca se cumpla y se envíen todas las palabras clave como
    # tareas de pago.
    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError) as exc:
        raise metaso_minimax.MetasoMiniMaxError(
            "la duración del audio para Metaso MiniMax debe ser un número finito"
        ) from exc
    if not math.isfinite(required_duration):
        raise metaso_minimax.MetasoMiniMaxError(
            "la duración del audio para Metaso MiniMax debe ser un número finito"
        )
    if required_duration <= 0:
        logger.warning(
            "se omite la generación de pago de Metaso MiniMax porque la duración de "
            f"audio necesaria no es positiva: duration={required_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise metaso_minimax.MetasoMiniMaxError(
            "la duración del clip de Metaso MiniMax debe ser un entero positivo"
        ) from exc
    if clip_duration <= 0:
        raise metaso_minimax.MetasoMiniMaxError(
            "la duración del clip de Metaso MiniMax debe ser un entero positivo"
        )

    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = metaso_minimax.generate_videos(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except metaso_minimax.MetasoMiniMaxUnconfirmedTaskError as exc:
            # Cuando el estado de la solicitud o del sondeo es desconocido, la tarea
            # remota aún puede tener éxito y facturarse. Se detiene de inmediato todo el
            # bucle de generación, para que las palabras clave siguientes no hagan más
            # pedidos, y se pasa el ID de la tarea al servicio de tareas para guardarlo.
            logger.error(
                "se dejan de enviar tareas nuevas a Metaso MiniMax porque la última "
                f"tarea de pago no está confirmada: task_id={exc.task_id or 'unknown'}, "
                f"detail={exc}"
            )
            _persist_material_sources(task_id, material_sources)
            raise
        except metaso_minimax.MetasoMiniMaxError as exc:
            logger.error(f"la generación con Metaso MiniMax falló antes de terminar: {exc}")
            _persist_material_sources(task_id, material_sources)
            raise

        for item in video_items:
            saved_video_path = _save_generated_video_with_retry(
                item.url, material_directory, "metaso_minimax"
            )
            if not saved_video_path:
                # Una generación correcta ya ha generado coste, así que si falla la
                # descarga no se pueden crear tareas nuevas para sustituirla. Se lanza
                # un error propio con el ID remoto, para el estado de la tarea y la
                # recuperación manual.
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise metaso_minimax.MetasoMiniMaxDownloadError(
                    "Metaso MiniMax generó un vídeo de pago pero no se pudo "
                    f"descargar el resultado: id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                )
            logger.info(f"vídeo guardado: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(_material_source_record(item, saved_video_path))
            except Exception as source_error:
                logger.warning(
                    "no se pudo preparar el registro de origen del material generado: "
                    f"provider=metaso_minimax, error={type(source_error).__name__}, "
                    f"detail={source_error}"
                )

            # El vídeo final local solo usa la duración de clip que eligió el usuario;
            # aunque H3 genere material más largo por su restricción de duración mínima,
            # la parte no usada no puede contarse en la duración cubierta, porque se
            # generaría menos imagen de la necesaria.
            total_duration += min(clip_duration, item.duration)
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "el material generado con Metaso MiniMax cubre la duración necesaria; "
                f"no se envían más tareas de pago: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break

    logger.success(f"se generaron y descargaron {len(video_paths)} vídeos de Metaso MiniMax")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_by_script_order(
    task_id: str,
    search_terms: List[str],
    search_videos,
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
    progress_callback: Callable[[float], None] | None = None,
) -> List[str]:
    """
    Descarga el material en el orden del texto del guion.

    La lógica de descarga por defecto junta los candidatos de todas las palabras clave
    en una gran lista; si la primera palabra clave devuelve muchos resultados, la
    descarga final puede ir consumiendo solo el material de esa palabra clave, y los
    temas posteriores del guion no llegan a la línea de tiempo. Aquí se agrupan por
    palabra clave y se descargan por rondas:
    en la ronda 1 se toma el 1.er candidato de cada palabra clave, en la ronda 2 el 2.º
    de cada una. Así, sin reescribir el motor de composición de vídeo, se intenta que el
    orden del material siga lo más posible el orden del texto.
    """
    logger.info("descargando vídeos con el material emparejado en el orden del guion")
    candidate_groups = []
    valid_video_urls = set()
    found_duration = 0.0

    for search_term, video_items in _search_terms_in_parallel(
        search_terms=search_terms,
        search_videos=search_videos,
        minimum_duration=max_clip_duration,
        video_aspect=video_aspect,
    ):
        term_items = []
        for item in video_items:
            if item.url in valid_video_urls:
                continue
            term_items.append(item)
            valid_video_urls.add(item.url)
            found_duration += item.duration

        if term_items:
            candidate_groups.append((search_term, term_items))

    logger.info(
        f"candidatos de vídeo ordenados encontrados en total: {sum(len(items) for _, items in candidate_groups)}, "
        f"duración necesaria: {audio_duration} segundos, duración encontrada: {found_duration} segundos"
    )

    video_paths = []
    material_sources: list[dict[str, Any]] = []
    total_duration = 0.0
    on_downloaded = _covered_duration_reporter(
        progress_callback, audio_duration, max_clip_duration
    )
    # Cada palabra clave avanza su índice de candidatos de forma independiente: solo
    # avanzan los candidatos que se eligieron de verdad para descargar en esta ronda, y
    # los no elegidos se guardan para la siguiente, para que un índice común para toda
    # la ronda no se los salte.
    next_candidate_indices = [0] * len(candidate_groups)
    while candidate_groups and total_duration <= audio_duration:
        round_materials = [
            (
                group_position,
                search_term,
                term_items[next_candidate_indices[group_position]],
            )
            for group_position, (search_term, term_items) in enumerate(
                candidate_groups
            )
            if next_candidate_indices[group_position] < len(term_items)
        ]
        if not round_materials:
            break

        selected_materials = _select_materials_until_duration(
            materials=[
                (search_term, item) for _, search_term, item in round_materials
            ],
            max_clip_duration=max_clip_duration,
            audio_duration=audio_duration,
            current_duration=total_duration,
        )
        # _select_materials_until_duration devuelve un subconjunto prefijo de la lista
        # de candidatos, así que los primeros len(selected_materials) son los candidatos
        # que se intentaron realmente descargar en esta ronda, y solo avanzan los
        # índices de los grupos de esos candidatos.
        for group_position, _, _ in round_materials[: len(selected_materials)]:
            next_candidate_indices[group_position] += 1

        downloaded_materials = _download_materials_in_parallel(
            materials=selected_materials,
            material_directory=material_directory,
            on_downloaded=on_downloaded,
        )
        for search_term, item, saved_video_path in downloaded_materials:
            try:
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                logger.info(
                    f"descargando vídeo ordenado de {item.provider} para {search_term!r}: "
                    f"asset_id={source_info.get('asset_id') or 'unknown'}"
                )
                if saved_video_path:
                    logger.info(f"vídeo guardado: {saved_video_path}")
                    video_paths.append(saved_video_path)
                    try:
                        material_sources.append(
                            _material_source_record(item, saved_video_path)
                        )
                    except Exception as source_error:
                        logger.warning(
                            "no se pudo preparar el registro de origen del material ordenado: "
                            f"provider={item.provider}, "
                            f"error={type(source_error).__name__}, "
                            f"detail={source_error}"
                        )
                    total_duration += min(max_clip_duration, item.duration)
                    if total_duration > audio_duration:
                        logger.info(
                            f"duración total de los vídeos descargados: {total_duration} segundos; no se descargan más"
                        )
                        break
            except Exception as e:
                logger.error(
                    "no se pudo descargar el vídeo de material ordenado: "
                    f"provider={item.provider}, error={type(e).__name__}, "
                    f"detail={_redact_request_error(e, item.url)}"
                )

    logger.success(f"se descargaron {len(video_paths)} vídeos ordenados")
    _persist_material_sources(task_id, material_sources)
    return video_paths


if __name__ == "__main__":
    download_videos(
        "test123", ["Money Exchange Medium"], audio_duration=100, source="pixabay"
    )


def _download_videos_infistar_on_demand(
    *, task_id, search_terms, video_aspect, audio_duration, max_clip_duration,
):
    """Pide segmento a segmento; se detiene cuando el material basta, y con estado
    desconocido o fallo de descarga no se compra el siguiente segmento.
    """
    try:
        required = float(audio_duration)
        clip_duration = int(max_clip_duration)
    except (ValueError, TypeError, OverflowError) as exc:
        raise infistar_video.InfistarVideoError("la duración de Infistar debe ser finita y positiva") from exc
    if not math.isfinite(required) or clip_duration <= 0:
        raise infistar_video.InfistarVideoError("la duración de Infistar debe ser finita y positiva")
    paths = []
    sources = []
    covered = 0.0
    try:
        for term in search_terms:
            if covered >= required:
                break
            item = infistar_video.generate_video(
                search_term=term, video_aspect=video_aspect,
                output_dir=utils.task_dir(task_id),
            )
            paths.append(item.url)
            sources.append(_material_source_record(item, item.url))
            covered += min(clip_duration, item.duration)
    finally:
        _persist_material_sources(task_id, sources)
    return paths
