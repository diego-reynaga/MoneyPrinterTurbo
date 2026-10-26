"""Comprueba si hay una nueva versión oficial de MoneyPrinterTurbo disponible."""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import requests
from loguru import logger

from app.utils.request_logging import trace_http_call
from packaging.version import InvalidVersion, Version


LATEST_RELEASE_API_URL: Final = (
    "https://api.github.com/repos/harry0703/MoneyPrinterTurbo/releases/latest"
)
LATEST_RELEASE_PAGE_URL: Final = (
    "https://github.com/harry0703/MoneyPrinterTurbo/releases/latest"
)
# La comprobación de actualizaciones es una función auxiliar, y los problemas de red no
# deben ralentizar visiblemente la WebUI local. La conexión y la lectura tienen tiempos
# de espera separados, lo que permite que GitHub responda en una red normal y evita
# esperas largas sin conexión.
RELEASE_CHECK_TIMEOUT: Final = (1.0, 2.0)
RELEASE_CHECK_HEADERS: Final = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "MoneyPrinterTurbo-Version-Checker",
}
UPDATE_CHECK_CACHE_TTL_SECONDS: Final = 12 * 60 * 60


def _parse_version(value: str) -> Version:
    """Admite las etiquetas ``v1.2.3`` habituales de GitHub y las convierte en versiones
    comparables.
    """
    normalized = str(value or "").strip()
    if normalized.lower().startswith("v"):
        normalized = normalized[1:]
    return Version(normalized)


def get_available_update(current_version: str) -> str | None:
    """
    Devuelve la última versión oficial superior a la actual; si no hay actualización o
    falla la comprobación devuelve ``None``.

    La interfaz ``releases/latest`` de GitHub excluye automáticamente borradores y
    prelanzamientos, así que aquí no se repite el filtrado por estado de publicación. La
    WebUI llama a esta función en segundo plano mediante ``AsyncUpdateChecker``; ante
    problemas de red, de formato de respuesta o de etiqueta de versión solo se registra
    en el log y se degrada a "no mostrar aviso", sin afectar a funciones principales
    como la generación de vídeo.
    """
    try:
        installed_version = _parse_version(current_version)
    except InvalidVersion:
        logger.warning(
            f"se omite la comprobación de actualizaciones porque la versión actual no es válida: {current_version!r}"
        )
        return None

    try:
        response = trace_http_call(
            "GET", requests.get,
            LATEST_RELEASE_API_URL,
            headers=RELEASE_CHECK_HEADERS,
            timeout=RELEASE_CHECK_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        # Un fallo de la comprobación de actualizaciones es una excepción recuperable y
        # no esencial. Se conservan el tipo y el mensaje de la excepción para localizar
        # problemas de proxy, DNS, límites de GitHub o respuestas dañadas, sin molestar
        # a los usuarios normales en la WebUI.
        logger.debug(
            "falló la comprobación de versiones en GitHub: "
            f"error_type={type(exc).__name__}, error={exc}"
        )
        return None

    if not isinstance(payload, dict):
        logger.debug(
            "la comprobación de versiones en GitHub devolvió una carga útil no válida: "
            f"payload_type={type(payload).__name__}"
        )
        return None

    tag_name = payload.get("tag_name", "")
    try:
        latest_version = _parse_version(tag_name)
    except InvalidVersion:
        logger.warning(
            f"se omite el aviso de actualización porque la etiqueta de la versión no es válida: {tag_name!r}"
        )
        return None

    if latest_version <= installed_version:
        return None

    normalized_latest_version = str(latest_version)
    logger.info(
        "hay una actualización de MoneyPrinterTurbo disponible: "
        f"current={installed_version}, latest={normalized_latest_version}"
    )
    return normalized_latest_version


@dataclass(frozen=True)
class UpdateCheckSnapshot:
    """Estado instantáneo de la comprobación de versión en segundo plano, para que la WebUI
    lo lea sin bloquear.
    """

    complete: bool
    available_version: str | None = None


class AsyncUpdateChecker:
    """
    Ejecuta la comprobación de versión en un hilo en segundo plano y guarda en caché el
    último resultado.

    Streamlit vuelve a ejecutar el script de la página desde el principio tras cualquier
    interacción con un control. Si se accediera a GitHub directamente en la zona del
    título, la primera apertura o la caducidad de la caché bloquearían toda la página.
    Aquí la solicitud de red va en un hilo daemon y la página solo lee la instantánea
    actual; cuando termina la comprobación, un fragment de corta duración de la WebUI
    refresca el resultado una vez.

    Tanto si el resultado es "hay actualización" como "no hay actualización / fallo de
    red", se guarda en caché, para no volver a pedirlo en cada rerun cuando GitHub no es
    accesible. El bloqueo solo protege el estado en memoria, no envuelve la solicitud de
    red, así que no bloquea a otras sesiones que lean el estado de la comprobación.
    """

    def __init__(
        self,
        check: Callable[[str], str | None] = get_available_update,
        ttl_seconds: float = UPDATE_CHECK_CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._check = check
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._current_version: str | None = None
        self._available_version: str | None = None
        self._completed_at: float | None = None
        self._checking = False
        self._request_id = 0

    def poll(self, current_version: str) -> UpdateCheckSnapshot:
        """Devuelve de inmediato la instantánea de la comprobación; si la caché ha
        caducado, lanza una nueva comprobación en segundo plano.
        """
        normalized_current_version = str(current_version or "").strip()
        now = self._clock()

        with self._lock:
            cache_is_fresh = (
                self._current_version == normalized_current_version
                and self._completed_at is not None
                and now - self._completed_at < self._ttl_seconds
            )
            if cache_is_fresh:
                return UpdateCheckSnapshot(
                    complete=True,
                    available_version=self._available_version,
                )

            if (
                self._checking
                and self._current_version == normalized_current_version
            ):
                return UpdateCheckSnapshot(complete=False)

            # Si cambia la versión o caduca la caché, el resultado antiguo no debe
            # seguir mostrándose. Se vacía el estado antes de arrancar el nuevo hilo,
            # para que quien llama reciba una instantánea pending clara durante la
            # comprobación.
            self._current_version = normalized_current_version
            self._available_version = None
            self._completed_at = None
            self._checking = True
            self._request_id += 1

            try:
                worker = threading.Thread(
                    target=self._run_check,
                    args=(normalized_current_version, self._request_id),
                    name="mpt-version-check",
                    daemon=True,
                )
                worker.start()
            except RuntimeError as exc:
                # Quedarse sin hilos no debe dejar pendientes todos los sondeos
                # posteriores.
                logger.warning(f"no se pudo iniciar el hilo de comprobación de actualizaciones: {exc}")
                self._checking = False
                self._completed_at = now
                return UpdateCheckSnapshot(complete=True)

        return UpdateCheckSnapshot(complete=False)

    def _run_check(self, current_version: str, request_id: int) -> None:
        try:
            available_version = self._check(current_version)
        except Exception:
            # get_available_update ya trata las excepciones de red y de datos previstas.
            # Esto es el último límite de protección del hilo en segundo plano y debe
            # registrar la traza completa, para que una excepción inesperada no termine
            # en silencio y deje el estado pending para siempre.
            logger.exception(
                "error inesperado al comprobar si hay una actualización de MoneyPrinterTurbo"
            )
            available_version = None

        with self._lock:
            # A -> B -> A puede lanzar dos workers para la misma versión. Solo la
            # solicitud más reciente puede completar la instantánea actual.
            if self._request_id != request_id:
                return
            self._available_version = available_version
            self._completed_at = self._clock()
            self._checking = False


_ASYNC_UPDATE_CHECKER = AsyncUpdateChecker()


def poll_available_update(current_version: str) -> UpdateCheckSnapshot:
    """Lee el estado del comprobador global en segundo plano, para que distintas sesiones
    de Streamlit no pidan a GitHub varias veces.
    """
    return _ASYNC_UPDATE_CHECKER.poll(current_version)
