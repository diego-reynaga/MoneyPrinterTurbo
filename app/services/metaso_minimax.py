"""Cliente de texto a vídeo MiniMax H3 de Metaso.

Este módulo solo se encarga del protocolo MiniMax V2 del proxy de Metaso: enviar tareas
de pago, sondear esa misma tarea y analizar el resultado generado. La generación de
material bajo demanda, la descarga de archivos y el montaje del vídeo final siguen
siendo responsabilidad del servicio ``material``, para no acoplar el protocolo del
proveedor con el flujo de vídeo local.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, quote_plus, urlsplit

import requests
from loguru import logger

from app.utils.request_logging import trace_http_call

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect


DEFAULT_BASE_URL = "https://metaso.cn/api/minimax"
DEFAULT_MODEL_ID = "MiniMax-H3"
DEFAULT_RESOLUTION = "2K"
DEFAULT_MIN_DURATION_SECONDS = 4
DEFAULT_MAX_DURATION_SECONDS = 15
DEFAULT_POLL_INTERVAL_SECONDS = 10.0
DEFAULT_RUN_TIMEOUT_SECONDS = 1800.0
MAX_PROMPT_LENGTH = 7000
MAX_POLL_RETRIES = 5
RETRY_BASE_SECONDS = 1.0
MAX_ERROR_TEXT_LENGTH = 500
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
ACTIVE_STATUSES = frozenset({"queued", "running"})
TERMINAL_FAILURE_STATUSES = frozenset({"failed", "cancelled", "canceled"})
SUPPORTED_RESOLUTIONS = frozenset({"768P", "2K"})


class MetasoMiniMaxError(RuntimeError):
    """Error determinista de configuración, solicitud o respuesta de Metaso MiniMax."""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(message)
        # Una vez creada la tarea remota, todas las excepciones posteriores llevan el
        # mismo ID. El servicio de tareas puede guardar las pistas de recuperación de
        # forma uniforme, sin saber si el fallo fue en el sondeo, en el análisis del
        # resultado o en la descarga.
        self.task_id = task_id


class MetasoMiniMaxUnconfirmedTaskError(MetasoMiniMaxError):
    """Puede que se haya creado una tarea de pago remota, pero no se puede confirmar su
    estado final desde aquí.
    """


class MetasoMiniMaxDownloadError(MetasoMiniMaxError):
    """La tarea de pago remota terminó bien, pero el vídeo no pudo descargarse a este
    equipo.
    """


def get_api_key(settings: Mapping[str, Any] | None = None) -> str:
    """
    Lee las credenciales de Metaso con una prioridad fija.

    La clave ``mk-`` de Metaso y la clave oficial de MiniMax pertenecen a sistemas de
    cuentas distintos, así que no se puede reutilizar el ``minimax_api_key`` que ya
    tiene el proyecto. Una configuración y una variable de entorno propias también
    evitan que el usuario cambie sin querer las credenciales de generación de vídeo al
    cambiar de proveedor LLM.
    """
    settings = config.app if settings is None else settings
    configured = str(settings.get("metaso_minimax_api_key", "") or "").strip()
    environment_key = os.getenv("METASO_MINIMAX_API_KEY", "").strip()
    return configured or environment_key


def is_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    """Devuelve si la configuración actual tiene credenciales para llamar a la interfaz de
    vídeo de Metaso.
    """
    return bool(get_api_key(settings))


def _base_url() -> str:
    value = (
        str(
            config.app.get("metaso_minimax_base_url", DEFAULT_BASE_URL)
            or DEFAULT_BASE_URL
        )
        .strip()
        .rstrip("/")
    )
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise MetasoMiniMaxError(
            "metaso_minimax_base_url debe ser una URL HTTP(S) absoluta"
        )
    return value


def _resolution() -> str:
    configured = config.app.get("metaso_minimax_resolution", DEFAULT_RESOLUTION)
    value = str(configured).strip().upper()
    if value not in SUPPORTED_RESOLUTIONS:
        supported = ", ".join(sorted(SUPPORTED_RESOLUTIONS))
        # La resolución afecta directamente al coste de generación, así que si el
        # usuario la escribe mal no se puede volver en silencio a 2K. Solo se usa el
        # valor por defecto cuando la opción falta del todo, para no crear sin querer
        # tareas más caras de lo previsto.
        raise MetasoMiniMaxError(
            f"Resolución de Metaso MiniMax no admitida {value!r}; "
            f"se esperaba una de: {supported}"
        )
    return value


def _tls_verify() -> bool:
    value = config.app.get("tls_verify", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _bounded_float(key: str, default: float, minimum: float, maximum: float) -> float:
    """Lee una opción de coma flotante finita y la limita a un rango seguro que no sature
    ni el servidor remoto ni este equipo.
    """
    try:
        value = float(config.app.get(key, default))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return min(max(value, minimum), maximum)


def _status_code(response: Any) -> int:
    try:
        return int(getattr(response, "status_code", 200))
    except (TypeError, ValueError):
        return 200


def _redact_secret(value: Any, api_key: str) -> str:
    """Conserva el texto útil para diagnosticar, eliminando la clave API, la clave
    codificada en la URL y las credenciales del proxy.
    """
    text = str(value or "")
    if api_key:
        text = text.replace(api_key, "***")
        encoded = quote_plus(api_key)
        if encoded != api_key:
            text = text.replace(encoded, "***")
    for proxy_url in config.proxy.values():
        proxy_secret = str(proxy_url or "")
        if proxy_secret:
            text = text.replace(proxy_secret, "***")
    return text[:MAX_ERROR_TEXT_LENGTH]


def _response_error(response: Any, api_key: str) -> str:
    """Admite la estructura de error anidada de MiniMax V2 y limita la longitud de la
    respuesta en los logs.
    """
    try:
        payload = response.json()
    except Exception:
        return f"HTTP {_status_code(response)}"
    if not isinstance(payload, dict):
        return f"HTTP {_status_code(response)}"

    error = payload.get("error")
    if isinstance(error, dict):
        error_type = error.get("type")
        message = error.get("message")
        http_code = error.get("http_code")
    else:
        error_type = None
        message = payload.get("message") or error
        http_code = payload.get("code")
    detail = ": ".join(
        str(item) for item in (error_type, http_code, message) if item not in (None, "")
    )
    return _redact_secret(detail or f"HTTP {_status_code(response)}", api_key)


def _is_retryable_error(error: Exception) -> bool:
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
    return response is not None and _status_code(response) in RETRYABLE_STATUS_CODES


def _normalize_duration(minimum_duration: int) -> tuple[int, int]:
    """Devuelve "duración pedida por el usuario, duración enviada realmente", para explicar
    en los logs la restricción del mínimo de 4 segundos.
    """
    try:
        requested = int(minimum_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MetasoMiniMaxError(
            "La duración del clip de Metaso MiniMax debe ser un entero positivo"
        ) from exc
    if requested <= 0:
        raise MetasoMiniMaxError(
            "La duración del clip de Metaso MiniMax debe ser un entero positivo"
        )
    duration = min(
        max(requested, DEFAULT_MIN_DURATION_SECONDS),
        DEFAULT_MAX_DURATION_SECONDS,
    )
    return requested, duration


def generate_videos(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> list[MaterialInfo]:
    """Envía una tarea de texto a vídeo Metaso MiniMax H3 y espera a un resultado
    descargable.
    """
    api_key = get_api_key()
    if not api_key:
        raise MetasoMiniMaxError("Metaso MiniMax necesita una clave API")

    term = str(search_term or "").strip()
    if not term:
        # Un prompt vacío suele indicar que falló la división del guion en el paso
        # anterior. No se puede tantear una interfaz de pago con entradas no válidas,
        # porque aunque el servidor la acepte solo produciría material facturado e
        # inutilizable.
        raise MetasoMiniMaxError("El término de búsqueda de Metaso MiniMax no puede estar vacío")
    if len(term) > MAX_PROMPT_LENGTH:
        raise MetasoMiniMaxError(
            f"El término de búsqueda de Metaso MiniMax supera los {MAX_PROMPT_LENGTH} caracteres"
        )

    aspect = VideoAspect(video_aspect)
    requested_duration, duration = _normalize_duration(minimum_duration)
    if duration != requested_duration:
        logger.info(
            "Duración del clip de Metaso MiniMax ajustada a los límites de H3: "
            f"requested={requested_duration}s, using={duration}s, "
            f"supported={DEFAULT_MIN_DURATION_SECONDS}-{DEFAULT_MAX_DURATION_SECONDS}s"
        )
    resolution = _resolution()
    payload = {
        "model": DEFAULT_MODEL_ID,
        "content": [{"type": "text", "text": term}],
        "resolution": resolution,
        "duration": duration,
        "ratio": aspect.value,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    base_url = _base_url()
    create_url = f"{base_url}/v2/video_generation"
    logger.info(
        "generando vídeo con Metaso MiniMax H3: "
        f"resolution={resolution}, ratio={aspect.value}, duration={duration}s, "
        f"prompt_length={len(term)}"
    )

    # Si el POST agota el tiempo o devuelve 5xx, puede que el servidor ya haya creado y
    # cobrado la tarea. La interfaz no ofrece una clave de idempotencia del cliente, así
    # que aquí nunca se reenvía automáticamente; la capa superior detiene las palabras
    # clave siguientes para no cobrar dos veces.
    try:
        response = trace_http_call(
            "POST", requests.post,
            create_url,
            json=payload,
            headers=headers,
            proxies=config.proxy,
            verify=_tls_verify(),
            timeout=(30, 60),
            allow_redirects=False,
        )
    except Exception as exc:
        raise MetasoMiniMaxUnconfirmedTaskError(
            "El envío a Metaso MiniMax no devolvió respuesta; puede que ya exista "
            "una tarea de pago en remoto: "
            f"error={type(exc).__name__}, detail={_redact_secret(exc, api_key)}"
        ) from exc

    status_code = _status_code(response)
    if 300 <= status_code < 400:
        raise MetasoMiniMaxUnconfirmedTaskError(
            "El envío a Metaso MiniMax devolvió una redirección; el estado de la tarea de pago "
            "es desconocido y la solicitud no se repitió"
        )
    if status_code >= 500:
        raise MetasoMiniMaxUnconfirmedTaskError(
            f"El envío a Metaso MiniMax falló con HTTP {status_code}; puede que ya "
            "exista una tarea de pago en remoto"
        )
    if not 200 <= status_code < 300:
        raise MetasoMiniMaxError(
            "Solicitud de generación de vídeo de Metaso MiniMax rechazada: "
            f"HTTP {status_code}, {_response_error(response, api_key)}"
        )
    try:
        body = response.json()
    except Exception as exc:
        raise MetasoMiniMaxUnconfirmedTaskError(
            "El envío a Metaso MiniMax devolvió una respuesta ilegible; puede que ya "
            f"exista una tarea de pago en remoto: error={type(exc).__name__}"
        ) from exc

    task_id = str(body.get("task_id") or "").strip() if isinstance(body, dict) else ""
    if not task_id:
        raise MetasoMiniMaxUnconfirmedTaskError(
            "Metaso MiniMax aceptó el envío sin devolver un id de tarea"
        )
    logger.info(f"Tarea de pago de Metaso MiniMax creada: id={task_id}")

    task = _wait_for_task(
        task_id=task_id,
        base_url=base_url,
        headers=headers,
        api_key=api_key,
    )
    content = task.get("content")
    video_url = content.get("url") if isinstance(content, dict) else None
    if not isinstance(video_url, str) or not video_url.startswith(
        ("http://", "https://")
    ):
        raise MetasoMiniMaxError(
            f"La tarea de Metaso MiniMax terminó bien pero sin un vídeo descargable: id={task_id}",
            task_id=task_id,
        )

    actual_duration = task.get("duration", duration)
    try:
        actual_duration = int(actual_duration)
    except (TypeError, ValueError, OverflowError):
        actual_duration = duration
    if actual_duration <= 0:
        actual_duration = duration

    return [
        MaterialInfo(
            provider="metaso_minimax",
            url=video_url,
            duration=actual_duration,
            source_info={
                "provider": "metaso_minimax",
                "search_term": term,
                "asset_id": task_id,
                # 2K/768P de MiniMax son nombres de especificación; la interfaz no
                # garantiza un tamaño en píxeles fijo. No se adivinan width/height; si
                # más adelante hace falta el tamaño exacto, debe usarse el valor medido
                # en el archivo descargado.
                "rendition": {"id": task_id},
            },
        )
    ]


def _wait_for_task(
    *,
    task_id: str,
    base_url: str,
    headers: dict[str, str],
    api_key: str,
) -> dict[str, Any]:
    """Sondea la misma tarea de pago hasta que termina bien, falla explícitamente o ya no
    se puede confirmar su estado aquí.
    """
    deadline = time.monotonic() + _bounded_float(
        "metaso_minimax_run_timeout",
        DEFAULT_RUN_TIMEOUT_SECONDS,
        60.0,
        7200.0,
    )
    poll_interval = _bounded_float(
        "metaso_minimax_poll_interval",
        DEFAULT_POLL_INTERVAL_SECONDS,
        1.0,
        60.0,
    )
    query_url = f"{base_url}/v2/query/video_generation/{quote(task_id, safe='')}"
    consecutive_failures = 0

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MetasoMiniMaxUnconfirmedTaskError(
                "La tarea de Metaso MiniMax sigue en curso tras el tiempo de espera local "
                f"configurado: id={task_id}",
                task_id=task_id,
            )

        # Los tiempos de espera de conexión y lectura se cuentan por separado, y cada
        # uno usa la mitad del tiempo restante, para que un GET no supere a propósito el
        # plazo total de la tarea. Una vez vencido no se lanza otro sondeo.
        phase_timeout = max(min(remaining / 2.0, 30.0), 0.001)
        try:
            response = trace_http_call(
                "GET", requests.get,
                query_url,
                headers=headers,
                proxies=config.proxy,
                verify=_tls_verify(),
                timeout=(phase_timeout, phase_timeout),
                allow_redirects=False,
            )
            status_code = _status_code(response)
            if status_code in RETRYABLE_STATUS_CODES:
                raise requests.exceptions.HTTPError(
                    f"HTTP {status_code}", response=response
                )
            if not 200 <= status_code < 300:
                raise MetasoMiniMaxUnconfirmedTaskError(
                    "El estado de la tarea de Metaso MiniMax es desconocido: "
                    f"http_status={status_code}, "
                    f"detail={_response_error(response, api_key)}",
                    task_id=task_id,
                )
            body = response.json()
            task = body.get("task") if isinstance(body, dict) else None
            if not isinstance(task, dict):
                raise MetasoMiniMaxUnconfirmedTaskError(
                    "La respuesta de estado de la tarea de Metaso MiniMax está mal formada",
                    task_id=task_id,
                )
        except MetasoMiniMaxUnconfirmedTaskError:
            raise
        except Exception as exc:
            if not _is_retryable_error(exc):
                raise MetasoMiniMaxUnconfirmedTaskError(
                    "Falló el sondeo de Metaso MiniMax y el estado de la tarea de pago es "
                    f"desconocido: error={type(exc).__name__}, "
                    f"detail={_redact_secret(exc, api_key)}",
                    task_id=task_id,
                ) from exc

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MetasoMiniMaxUnconfirmedTaskError(
                    "La tarea de Metaso MiniMax sigue en curso tras el tiempo de espera "
                    f"local configurado: id={task_id}",
                    task_id=task_id,
                ) from exc
            consecutive_failures += 1
            if consecutive_failures > MAX_POLL_RETRIES:
                raise MetasoMiniMaxUnconfirmedTaskError(
                    "El sondeo de Metaso MiniMax falló tras los reintentos; la tarea de pago "
                    f"puede seguir en curso en remoto: id={task_id}",
                    task_id=task_id,
                ) from exc
            delay = min(RETRY_BASE_SECONDS * consecutive_failures, remaining)
            logger.warning(
                "El sondeo de Metaso MiniMax encontró un error pasajero; se reintenta la misma "
                f"tarea: id={task_id}, attempt={consecutive_failures}/"
                f"{MAX_POLL_RETRIES}, retry_in={delay:.1f}s"
            )
            time.sleep(delay)
            continue

        consecutive_failures = 0
        status = str(task.get("status") or "").strip().lower()
        logger.info(f"Estado de la tarea de Metaso MiniMax: id={task_id}, status={status}")
        if status == "succeeded":
            return task
        if status in TERMINAL_FAILURE_STATUSES:
            raise MetasoMiniMaxError(
                "La tarea de Metaso MiniMax no produjo ningún vídeo: "
                f"id={task_id}, status={status}, "
                f"detail={_redact_secret(task.get('error'), api_key)}",
                task_id=task_id,
            )
        if status not in ACTIVE_STATUSES:
            raise MetasoMiniMaxUnconfirmedTaskError(
                "Metaso MiniMax devolvió un estado de tarea desconocido: "
                f"id={task_id}, status={status!r}",
                task_id=task_id,
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MetasoMiniMaxUnconfirmedTaskError(
                "La tarea de Metaso MiniMax sigue en curso tras el tiempo de espera local "
                f"configurado: id={task_id}",
                task_id=task_id,
            )
        time.sleep(min(poll_interval, remaining))
