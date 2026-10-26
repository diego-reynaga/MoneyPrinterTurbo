import math
import os
import time
from typing import Any, Mapping
from urllib.parse import quote_plus

import requests
from loguru import logger

from app.utils.request_logging import trace_http_call

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect


DEFAULT_BASE_URL = "https://api.ofox.ai/v1"
DEFAULT_MODEL_ID = "bytedance/seedance-2.0-fast"
DEFAULT_RESOLUTION = "720p"
# Por defecto se fija el canal del proveedor internacional: con un público global la
# política de contenido es más coherente; si se configura explícitamente vacío, se
# devuelve al gateway el reparto por pesos entre los proveedores disponibles.
DEFAULT_PROVIDER_TYPE = "byteplus"
# El modelo por defecto bytedance/seedance-2.0-fast solo acepta de 4 a 15 segundos
# (valor validado en el servidor en pruebas reales). Otros modelos opcionales tienen
# rangos distintos (por ejemplo alibaba/wan-2.7 admite de 2 a 15 segundos); al cambiar
# de modelo hay que ajustar también el rango en la configuración. Las solicitudes fuera
# de rango la API las rechaza con un 400 claro y no se facturan.
DEFAULT_MIN_DURATION_SECONDS = 4
DEFAULT_MAX_DURATION_SECONDS = 15
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_RUN_TIMEOUT_SECONDS = 1800.0
MAX_POLL_RETRIES = 5
RETRY_BASE_SECONDS = 1.0
MAX_ERROR_TEXT_LENGTH = 500
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
TERMINAL_SUCCESS_STATUSES = frozenset({"completed", "succeeded"})
TERMINAL_FAILURE_STATUSES = frozenset(
    {"failed", "error", "cancelled", "canceled", "expired"}
)
# Ruta de éxito oficial: pending (recibido, pendiente de enviar al proveedor) → queued →
# in_progress → completed
ACTIVE_STATUSES = frozenset({"pending", "queued", "in_progress"})


class OFoxError(RuntimeError):
    """Error determinista de configuración, solicitud o respuesta."""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(message)
        # Una vez creada la tarea remota, todos los tipos de error llevan el ID de la
        # tarea. La capa superior no necesita mantener lógica de recuperación según la
        # subclase de excepción, y la WebUI/API pueden mostrar de forma estable los
        # datos para diagnosticar.
        self.task_id = task_id


class OFoxUnconfirmedTaskError(OFoxError):
    """Puede que se haya creado una tarea de pago remota, pero no se puede confirmar su
    estado final desde aquí.
    """

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(message, task_id=task_id)


class OFoxDownloadError(OFoxError):
    """La tarea de pago remota terminó bien, pero el vídeo generado no pudo descargarse a
    este equipo.
    """

    def __init__(self, message: str, task_id: str):
        super().__init__(message, task_id=task_id)


def get_api_key(settings: Mapping[str, Any] | None = None) -> str:
    """
    Lee las credenciales de OFox con una prioridad clara y única.

    La clave propia del archivo de configuración tiene la prioridad más alta; la única
    variable de entorno admitida en tiempo de ejecución es ``OFOX_API_KEY``, de
    significado claro.
    """
    settings = config.app if settings is None else settings
    configured = str(settings.get("ofox_api_key", "") or "").strip()
    environment_key = os.getenv("OFOX_API_KEY", "").strip()
    return configured or environment_key


def is_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    return bool(get_api_key(settings))


def _base_url() -> str:
    return str(
        config.app.get("ofox_base_url", DEFAULT_BASE_URL) or DEFAULT_BASE_URL
    ).rstrip("/")


def _model_id() -> str:
    return str(
        config.app.get("ofox_text_to_video_model", DEFAULT_MODEL_ID) or DEFAULT_MODEL_ID
    ).strip()


def _resolution() -> str:
    """
    Lee la resolución de generación.

    OFox valida la resolución en el servidor según el modelo (por ejemplo, el modelo
    Seedance fast por defecto solo acepta 480p/720p); un valor no válido recibe un
    rechazo 400 claro y no crea una tarea de pago, así que aquí solo se quitan los
    espacios, sin mantener una lista blanca local, que se quedaría obsoleta al cambiar
    el catálogo remoto de modelos. Un valor vacío vuelve al predeterminado, para no
    enviar una cadena vacía al servidor.
    """
    value = str(config.app.get("ofox_resolution", DEFAULT_RESOLUTION) or "").strip()
    return value or DEFAULT_RESOLUTION


def _provider_type() -> str:
    """
    Lee el proveedor fijado (provider routing).

    Algunos modelos de OFox los suministran varios proveedores (por ejemplo volcengine y
    byteplus para la serie Seedance), cada uno con su propia política de contenido y
    disponibilidad regional. Por defecto se fija byteplus (proveedor internacional, con
    una política de contenido más coherente para un público global y un enrutamiento
    previsible); si se configura otro nombre de proveedor se fija ese; si se configura
    explícitamente una cadena vacía no se fija nada y el gateway reparte por pesos. Un
    nombre de proveedor no válido la API lo rechaza con un 400 ``invalid_provider_type``
    específico sin crear tareas de pago, así que no se mantiene una lista blanca local
    de proveedores.
    """
    value = config.app.get("ofox_provider", DEFAULT_PROVIDER_TYPE)
    if value is None:
        return DEFAULT_PROVIDER_TYPE
    return str(value).strip()


def _config_bool(key: str, default: bool) -> bool:
    value = config.app.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _bounded_float(key: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(config.app.get(key, default))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return min(max(value, minimum), maximum)


def _duration_bounds() -> tuple[int, int]:
    def read(key: str, default: int) -> int:
        try:
            value = int(config.app.get(key, default))
        except (TypeError, ValueError, OverflowError):
            # TOML acepta `inf`; int(inf) lanza OverflowError, que no es un error del
            # proveedor y abortaría la tarea con una traza.
            return default
        return value if value >= 1 else default

    minimum = read("ofox_min_duration", DEFAULT_MIN_DURATION_SECONDS)
    maximum = read("ofox_max_duration", DEFAULT_MAX_DURATION_SECONDS)
    return minimum, max(minimum, maximum)


def _tls_verify() -> bool:
    return _config_bool("tls_verify", True)


def _status_code(response: Any) -> int:
    try:
        return int(getattr(response, "status_code", 200))
    except (TypeError, ValueError):
        return 200


def _redact_secret(value: Any, secret: str) -> str:
    text = str(value or "")
    if secret:
        text = text.replace(secret, "***")
        encoded = quote_plus(secret)
        if encoded != secret:
            text = text.replace(encoded, "***")
    for proxy_url in config.proxy.values():
        proxy_secret = str(proxy_url or "")
        if proxy_secret:
            text = text.replace(proxy_secret, "***")
    return text[:MAX_ERROR_TEXT_LENGTH]


def _response_error(response: Any, api_key: str) -> str:
    try:
        payload = response.json()
    except Exception:
        return f"HTTP {_status_code(response)}"
    if not isinstance(payload, dict):
        return f"HTTP {_status_code(response)}"
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code") or payload.get("code")
        message = error.get("message") or payload.get("message")
    else:
        code = payload.get("code")
        message = payload.get("message") or error
    detail = ": ".join(str(item) for item in (code, message) if item not in (None, ""))
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


def generate_videos(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> list[MaterialInfo]:
    """Envía una tarea de texto a vídeo de OFox y espera la dirección de un resultado
    descargable.
    """
    api_key = get_api_key()
    if not api_key:
        raise OFoxError("La generación de vídeo con OFox necesita una clave API de OFox")

    term = str(search_term or "").strip()
    if not term:
        # Un prompt vacío puede venir de un fallo al dividir el guion en el paso
        # anterior. Una fuente de generación de pago no puede enviarlo al servidor,
        # porque aunque la interfaz acepte la solicitud solo daría un vídeo inutilizable
        # y ya facturado.
        raise OFoxError("El término de búsqueda de OFox no puede estar vacío")

    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()
    try:
        requested_duration = max(int(minimum_duration), 1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OFoxError("La duración del clip de OFox debe ser un entero positivo") from exc
    minimum, maximum = _duration_bounds()
    duration = min(max(requested_duration, minimum), maximum)
    if duration != requested_duration:
        # Generar más de lo pedido no afecta al vídeo final: el montaje sigue recortando
        # según la duración del clip; generar menos de lo pedido solo ocurre cuando la
        # solicitud supera el máximo del modelo, y entonces solo se puede ajustar a ese
        # máximo.
        logger.info(
            f"duración del clip de ofox ajustada al rango del modelo configurado: "
            f"requested={requested_duration}s, using={duration}s "
            f"(configurado {minimum}-{maximum}s)"
        )
    payload = {
        "model": _model_id(),
        "prompt": term,
        "duration": duration,
        "resolution": _resolution(),
        "aspect_ratio": aspect.value,
    }
    provider_type = _provider_type()
    if provider_type:
        payload["provider"] = {"type": provider_type}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    videos_url = f"{_base_url()}/videos"
    logger.info(
        "generando vídeo con OFox: "
        f"model={payload['model']}, term={term!r}, duration={duration}s"
    )

    # La interfaz de envío no se reintenta automáticamente: un tiempo de espera o un 5xx
    # pueden ocurrir después de crearse la tarea de pago, y reintentar a ciegas cobraría
    # dos veces. Solo se considera fallo determinista cuando hay una respuesta de
    # rechazo explícita.
    try:
        response = trace_http_call(
            "POST", requests.post,
            videos_url,
            json=payload,
            headers=headers,
            proxies=config.proxy,
            verify=_tls_verify(),
            timeout=(30, 60),
            allow_redirects=False,
        )
    except Exception as exc:
        raise OFoxUnconfirmedTaskError(
            "El envío a OFox no devolvió respuesta; puede que ya exista "
            "una tarea de pago en remoto: "
            f"error={type(exc).__name__}, detail={_redact_secret(exc, api_key)}"
        ) from exc

    status_code = _status_code(response)
    if 300 <= status_code < 400:
        raise OFoxUnconfirmedTaskError(
            "El envío a OFox devolvió una redirección; el estado de la tarea de pago es "
            "desconocido y la solicitud no se repitió"
        )
    if status_code >= 500:
        raise OFoxUnconfirmedTaskError(
            f"El envío a OFox falló con HTTP {status_code}; puede que ya exista "
            "una tarea de pago en remoto"
        )
    if not 200 <= status_code < 300:
        # Un 4xx es un rechazo explícito (por ejemplo, duration/resolution fuera del
        # rango que admite el modelo): el servidor no ha creado ninguna tarea y no hay
        # riesgo de cobro doble; el mensaje de error incluye el rango de valores válidos
        # que da el servidor, y se pasa directamente al usuario para que corrija la
        # configuración.
        raise OFoxError(
            "Solicitud de generación de vídeo de OFox rechazada: "
            f"HTTP {status_code}, {_response_error(response, api_key)}"
        )
    try:
        body = response.json()
    except Exception as exc:
        raise OFoxUnconfirmedTaskError(
            "El envío a OFox devolvió una respuesta ilegible; puede que ya exista "
            f"una tarea de pago en remoto: error={type(exc).__name__}"
        ) from exc
    task_id = str(body.get("id") or "").strip() if isinstance(body, dict) else ""
    if not task_id:
        raise OFoxUnconfirmedTaskError(
            "OFox aceptó el envío sin devolver un id de tarea"
        )
    logger.info(f"Tarea de vídeo de OFox creada: id={task_id}")

    task = _wait_for_task(
        task_id=task_id,
        videos_url=videos_url,
        headers=headers,
        api_key=api_key,
    )
    if task is None:
        return []

    # La recomendación oficial es usar primero mirror_urls (direcciones firmadas
    # persistentes de la CDN de OFox, que solo se devuelven cuando el proveedor tiene
    # activado el espejo) y, si faltan, volver a unsigned_urls (enlaces directos
    # temporales del proveedor, que pueden caducar en 24 horas). Ambas deben conservarse
    # enteras y usarse de inmediato para descargar, sin escribirlas en el source_info de
    # larga duración.
    video_url = ""
    for field in ("mirror_urls", "unsigned_urls"):
        urls = task.get(field)
        for candidate in urls if isinstance(urls, list) else []:
            if isinstance(candidate, str) and candidate.startswith(
                ("http://", "https://")
            ):
                video_url = candidate
                break
        if video_url:
            break
    if not video_url:
        raise OFoxError(
            f"La tarea de OFox terminó sin un vídeo descargable: id={task_id}",
            task_id=task_id,
        )

    return [
        MaterialInfo(
            provider="ofox",
            url=video_url,
            duration=duration,
            source_info={
                "provider": "ofox",
                "search_term": term,
                "asset_id": task_id,
                "rendition": {
                    "id": task_id,
                    "width": video_width,
                    "height": video_height,
                },
            },
        )
    ]


def _wait_for_task(
    *,
    task_id: str,
    videos_url: str,
    headers: dict[str, str],
    api_key: str,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + _bounded_float(
        "ofox_run_timeout",
        DEFAULT_RUN_TIMEOUT_SECONDS,
        60.0,
        7200.0,
    )
    poll_interval = _bounded_float(
        "ofox_poll_interval",
        DEFAULT_POLL_INTERVAL_SECONDS,
        0.5,
        60.0,
    )
    consecutive_failures = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise OFoxUnconfirmedTaskError(
                "La tarea de OFox sigue en curso tras el tiempo de espera local "
                f"configurado: id={task_id}",
                task_id=task_id,
            )

        # Los tiempos de espera de conexión y lectura de requests se cuentan por
        # separado, así que cada uno usa la mitad del tiempo total restante. Aunque la
        # conexión y la lectura lleguen al máximo, una vuelta no supera a propósito el
        # plazo total; la librería de red aún puede tener un pequeño error de
        # planificación, y la siguiente comprobación del plazo impide otro reintento.
        phase_timeout = max(min(remaining / 2.0, 30.0), 0.001)
        try:
            response = trace_http_call(
                "GET", requests.get,
                f"{videos_url}/{task_id}",
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
                raise OFoxUnconfirmedTaskError(
                    "El estado de la tarea de OFox es desconocido: "
                    f"http_status={status_code}, "
                    f"detail={_response_error(response, api_key)}",
                    task_id=task_id,
                )
            body = response.json()
            if not isinstance(body, dict):
                raise OFoxUnconfirmedTaskError(
                    "La respuesta de estado de la tarea de OFox está mal formada", task_id=task_id
                )
        except OFoxUnconfirmedTaskError:
            raise
        except Exception as exc:
            if not _is_retryable_error(exc):
                raise OFoxUnconfirmedTaskError(
                    "Falló el sondeo de OFox y el estado de la tarea de pago es desconocido: "
                    f"error={type(exc).__name__}, "
                    f"detail={_redact_secret(exc, api_key)}",
                    task_id=task_id,
                ) from exc

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OFoxUnconfirmedTaskError(
                    "La tarea de OFox sigue en curso tras el tiempo de espera local "
                    f"configurado: id={task_id}",
                    task_id=task_id,
                ) from exc
            consecutive_failures += 1
            if consecutive_failures > MAX_POLL_RETRIES:
                raise OFoxUnconfirmedTaskError(
                    "El sondeo de OFox falló tras los reintentos; la tarea de pago puede seguir "
                    f"en curso en remoto: id={task_id}",
                    task_id=task_id,
                ) from exc
            delay = min(RETRY_BASE_SECONDS * consecutive_failures, remaining)
            logger.warning(
                "El sondeo de OFox encontró un error pasajero; se reintenta la misma tarea: "
                f"id={task_id}, attempt={consecutive_failures}/{MAX_POLL_RETRIES}, "
                f"retry_in={delay:.1f}s"
            )
            time.sleep(delay)
            continue

        consecutive_failures = 0
        status = str(body.get("status") or "").strip().lower()
        if status in TERMINAL_SUCCESS_STATUSES:
            return body
        if status in TERMINAL_FAILURE_STATUSES:
            error_detail = body.get("error")
            logger.error(
                "La tarea de OFox no produjo ningún vídeo: "
                f"id={task_id}, status={status}, "
                f"detail={_redact_secret(error_detail, api_key)}"
            )
            # Un fallo remoto explícito significa que la tarea ya ha terminado y se
            # puede seguir con los clips siguientes con seguridad; quien llama decide si
            # reintenta con otra palabra clave.
            return None
        if status not in ACTIVE_STATUSES:
            raise OFoxUnconfirmedTaskError(
                f"OFox devolvió un estado de tarea desconocido: id={task_id}, "
                f"status={status!r}",
                task_id=task_id,
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise OFoxUnconfirmedTaskError(
                "La tarea de OFox sigue en curso tras el tiempo de espera local "
                f"configurado: id={task_id}",
                task_id=task_id,
            )
        time.sleep(min(poll_interval, remaining))
