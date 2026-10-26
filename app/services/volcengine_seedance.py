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


DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL_ID = "doubao-seedance-1-0-pro-250528"
DEFAULT_RESOLUTION = "1080p"
DEFAULT_MIN_DURATION_SECONDS = 2
DEFAULT_MAX_DURATION_SECONDS = 12
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_RUN_TIMEOUT_SECONDS = 1800.0
MAX_POLL_RETRIES = 5
RETRY_BASE_SECONDS = 1.0
MAX_ERROR_TEXT_LENGTH = 500
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
TERMINAL_FAILURE_STATUSES = frozenset(
    {"failed", "cancelled", "canceled", "expired"}
)
ACTIVE_STATUSES = frozenset({"queued", "running"})
SUPPORTED_RESOLUTIONS = frozenset({"480p", "720p", "1080p"})


class VolcEngineSeedanceError(RuntimeError):
    """Error determinista de configuración, solicitud o respuesta."""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(message)
        # Una vez creada la tarea remota, todos los tipos de error llevan el ID de la
        # tarea. La capa superior no necesita mantener lógica de recuperación según la
        # subclase de excepción, y la WebUI/API pueden mostrar de forma estable los
        # datos para diagnosticar.
        self.task_id = task_id


class VolcEngineSeedanceUnconfirmedTaskError(VolcEngineSeedanceError):
    """Puede que se haya creado una tarea de pago remota, pero no se puede confirmar su
    estado final desde aquí.
    """

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(message, task_id=task_id)


class VolcEngineSeedanceDownloadError(VolcEngineSeedanceError):
    """La tarea de pago remota terminó bien, pero el vídeo generado no pudo descargarse a
    este equipo.
    """

    def __init__(self, message: str, task_id: str):
        super().__init__(message, task_id=task_id)


def get_api_key(settings: Mapping[str, Any] | None = None) -> str:
    """
    Lee las credenciales de Ark con una prioridad clara y única.

    La configuración propia de Seedance tiene la prioridad más alta; la única variable
    de entorno admitida en tiempo de ejecución es ``VOLCENGINE_ARK_API_KEY``, de
    significado claro. El histórico ``volcengine_api_key`` solo sirve como respaldo
    compartido, para que los usuarios que ya usan los modelos de Ark no tengan que
    volver a escribir la misma clave al actualizar.
    """
    settings = config.app if settings is None else settings
    configured = str(settings.get("volcengine_seedance_api_key", "") or "").strip()
    environment_key = os.getenv("VOLCENGINE_ARK_API_KEY", "").strip()
    shared_ark_key = str(settings.get("volcengine_api_key", "") or "").strip()
    return configured or environment_key or shared_ark_key


def is_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    return bool(get_api_key(settings))


def _base_url() -> str:
    return str(
        config.app.get("volcengine_seedance_base_url", DEFAULT_BASE_URL)
        or DEFAULT_BASE_URL
    ).rstrip("/")


def _model_id() -> str:
    return str(
        config.app.get("volcengine_seedance_model", DEFAULT_MODEL_ID)
        or DEFAULT_MODEL_ID
    ).strip()


def _resolution() -> str:
    configured = config.app.get("volcengine_seedance_resolution", DEFAULT_RESOLUTION)
    value = str(configured).strip().lower()
    if value not in SUPPORTED_RESOLUTIONS:
        # La resolución afecta directamente a la especificación de la tarea de pago. Un
        # valor no válido no puede volver en silencio a la resolución más alta por
        # defecto. Si la opción falta, get usa el valor por defecto; pero si el usuario
        # escribe explícitamente un vacío, None, 0 u otro valor no válido, debe dar
        # error, o podría generar un coste mayor del esperado.
        supported = ", ".join(sorted(SUPPORTED_RESOLUTIONS))
        raise VolcEngineSeedanceError(
            f"Resolución de Seedance no admitida {value!r}; se esperaba una de: {supported}"
        )
    return value


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

    minimum = read(
        "volcengine_seedance_min_duration", DEFAULT_MIN_DURATION_SECONDS
    )
    maximum = read(
        "volcengine_seedance_max_duration", DEFAULT_MAX_DURATION_SECONDS
    )
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


def _rendition_size(aspect: VideoAspect, resolution: str) -> tuple[int, int]:
    # En el vídeo 480p de Ark, el lado largo se alinea por codificación y la salida real
    # es de 864, no los 854 del cálculo matemático; 720p y 1080p dan 1280 y 1920 según
    # la proporción oficial. Una llamada real en este equipo verificó que el producto
    # vertical a 480p es de 480x864. El registro de origen debe describir el producto
    # real, o en auditorías o diagnósticos de material posteriores se verían tamaños que
    # no coinciden con el archivo.
    short_edge = {"480p": 480, "720p": 720, "1080p": 1080}[resolution]
    long_edge = {"480p": 864, "720p": 1280, "1080p": 1920}[resolution]
    if aspect == VideoAspect.portrait:
        return short_edge, long_edge
    if aspect == VideoAspect.square:
        return short_edge, short_edge
    return long_edge, short_edge


def generate_videos(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> list[MaterialInfo]:
    """Envía una tarea de texto a vídeo Seedance de Ark y espera la dirección de un
    resultado descargable.
    """
    api_key = get_api_key()
    if not api_key:
        raise VolcEngineSeedanceError(
            "Volcano Engine Seedance necesita una clave API de Ark"
        )

    term = str(search_term or "").strip()
    if not term:
        # Un prompt vacío puede venir de un fallo al dividir el guion en el paso
        # anterior. Una fuente de generación de pago no puede enviarlo al servidor,
        # porque aunque la interfaz acepte la solicitud solo daría un vídeo inutilizable
        # y ya facturado.
        raise VolcEngineSeedanceError("El término de búsqueda de Seedance no puede estar vacío")

    aspect = VideoAspect(video_aspect)
    try:
        requested_duration = max(int(minimum_duration), 1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise VolcEngineSeedanceError(
            "La duración del clip de Seedance debe ser un entero positivo"
        ) from exc
    minimum, maximum = _duration_bounds()
    duration = min(max(requested_duration, minimum), maximum)
    if duration != requested_duration:
        # Generar más de lo pedido no afecta al vídeo final: el montaje sigue recortando
        # según la duración del clip; generar menos de lo pedido solo ocurre cuando la
        # solicitud supera el máximo del modelo, y entonces solo se puede ajustar a ese
        # máximo. Las demás fuentes de vídeo de pago explican este ajuste; si no, el
        # usuario no podría entender por qué los clips del vídeo final son más cortos
        # que lo configurado.
        logger.info(
            "Duración del clip de Seedance ajustada al rango del modelo configurado: "
            f"requested={requested_duration}s, using={duration}s "
            f"(configurado {minimum}-{maximum}s)"
        )
    resolution = _resolution()
    payload = {
        "model": _model_id(),
        "content": [{"type": "text", "text": term}],
        "ratio": aspect.value,
        "duration": duration,
        "resolution": resolution,
        "watermark": _config_bool("volcengine_seedance_watermark", False),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    tasks_url = f"{_base_url()}/contents/generations/tasks"
    logger.info(
        "generando vídeo con Volcano Engine Seedance: "
        f"model={payload['model']}, term={term!r}, duration={duration}s"
    )

    # La interfaz de envío no se reintenta automáticamente: un tiempo de espera o un 5xx
    # pueden ocurrir después de crearse la tarea de pago, y reintentar a ciegas cobraría
    # dos veces. Solo se considera fallo determinista cuando hay una respuesta de
    # rechazo explícita.
    try:
        response = trace_http_call(
            "POST", requests.post,
            tasks_url,
            json=payload,
            headers=headers,
            proxies=config.proxy,
            verify=_tls_verify(),
            timeout=(30, 60),
            allow_redirects=False,
        )
    except Exception as exc:
        raise VolcEngineSeedanceUnconfirmedTaskError(
            "El envío a Seedance no devolvió respuesta; puede que ya exista "
            "una tarea de pago en remoto: "
            f"error={type(exc).__name__}, detail={_redact_secret(exc, api_key)}"
        ) from exc

    status_code = _status_code(response)
    if 300 <= status_code < 400:
        raise VolcEngineSeedanceUnconfirmedTaskError(
            "El envío a Seedance devolvió una redirección; el estado de la tarea de pago es "
            "desconocido y la solicitud no se repitió"
        )
    if status_code >= 500:
        raise VolcEngineSeedanceUnconfirmedTaskError(
            f"El envío a Seedance falló con HTTP {status_code}; puede que ya exista "
            "una tarea de pago en remoto"
        )
    if not 200 <= status_code < 300:
        raise VolcEngineSeedanceError(
            "Solicitud de generación de vídeo de Seedance rechazada: "
            f"HTTP {status_code}, {_response_error(response, api_key)}"
        )
    try:
        body = response.json()
    except Exception as exc:
        raise VolcEngineSeedanceUnconfirmedTaskError(
            "El envío a Seedance devolvió una respuesta ilegible; puede que ya exista "
            f"una tarea de pago en remoto: error={type(exc).__name__}"
        ) from exc
    task_id = str(body.get("id") or "").strip() if isinstance(body, dict) else ""
    if not task_id:
        raise VolcEngineSeedanceUnconfirmedTaskError(
            "Seedance aceptó el envío sin devolver un id de tarea"
        )
    logger.info(f"Tarea de Volcano Engine Seedance creada: id={task_id}")

    task = _wait_for_task(
        task_id=task_id,
        tasks_url=tasks_url,
        headers=headers,
        api_key=api_key,
    )
    if task is None:
        return []
    content = task.get("content")
    video_url = content.get("video_url") if isinstance(content, dict) else None
    if not isinstance(video_url, str) or not video_url.startswith(("http://", "https://")):
        raise VolcEngineSeedanceError(
            f"La tarea de Seedance terminó bien pero sin un vídeo descargable: id={task_id}",
            task_id=task_id,
        )

    width, height = _rendition_size(aspect, resolution)
    return [
        MaterialInfo(
            provider="volcengine_seedance",
            url=video_url,
            duration=duration,
            source_info={
                "provider": "volcengine_seedance",
                "search_term": term,
                "asset_id": task_id,
                "rendition": {
                    "id": task_id,
                    "width": width,
                    "height": height,
                },
            },
        )
    ]


def _wait_for_task(
    *,
    task_id: str,
    tasks_url: str,
    headers: dict[str, str],
    api_key: str,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + _bounded_float(
        "volcengine_seedance_run_timeout",
        DEFAULT_RUN_TIMEOUT_SECONDS,
        60.0,
        7200.0,
    )
    poll_interval = _bounded_float(
        "volcengine_seedance_poll_interval",
        DEFAULT_POLL_INTERVAL_SECONDS,
        0.5,
        60.0,
    )
    consecutive_failures = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise VolcEngineSeedanceUnconfirmedTaskError(
                "La tarea de Seedance sigue en curso tras el tiempo de espera local "
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
                f"{tasks_url}/{task_id}",
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
                raise VolcEngineSeedanceUnconfirmedTaskError(
                    "El estado de la tarea de Seedance es desconocido: "
                    f"http_status={status_code}, detail={_response_error(response, api_key)}",
                    task_id=task_id,
                )
            body = response.json()
            if not isinstance(body, dict):
                raise VolcEngineSeedanceUnconfirmedTaskError(
                    "La respuesta de estado de la tarea de Seedance está mal formada", task_id=task_id
                )
        except VolcEngineSeedanceUnconfirmedTaskError:
            raise
        except Exception as exc:
            if not _is_retryable_error(exc):
                raise VolcEngineSeedanceUnconfirmedTaskError(
                    "Falló el sondeo de Seedance y el estado de la tarea de pago es desconocido: "
                    f"error={type(exc).__name__}, detail={_redact_secret(exc, api_key)}",
                    task_id=task_id,
                ) from exc

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VolcEngineSeedanceUnconfirmedTaskError(
                    "La tarea de Seedance sigue en curso tras el tiempo de espera local "
                    f"configurado: id={task_id}",
                    task_id=task_id,
                ) from exc
            consecutive_failures += 1
            if consecutive_failures > MAX_POLL_RETRIES:
                raise VolcEngineSeedanceUnconfirmedTaskError(
                    "El sondeo de Seedance falló tras los reintentos; la tarea de pago puede seguir "
                    f"en curso en remoto: id={task_id}",
                    task_id=task_id,
                ) from exc
            delay = min(RETRY_BASE_SECONDS * consecutive_failures, remaining)
            logger.warning(
                "El sondeo de Seedance encontró un error pasajero; se reintenta la misma tarea: "
                f"id={task_id}, attempt={consecutive_failures}/{MAX_POLL_RETRIES}, "
                f"retry_in={delay:.1f}s"
            )
            time.sleep(delay)
            continue

        consecutive_failures = 0
        status = str(body.get("status") or "").strip().lower()
        if status == "succeeded":
            return body
        if status in TERMINAL_FAILURE_STATUSES:
            raise VolcEngineSeedanceError(
                "La tarea de Seedance no produjo ningún vídeo: "
                f"id={task_id}, status={status}, "
                f"detail={_redact_secret(body.get('error'), api_key)}",
                task_id=task_id,
            )
        if status not in ACTIVE_STATUSES:
            raise VolcEngineSeedanceUnconfirmedTaskError(
                f"Seedance devolvió un estado de tarea desconocido: id={task_id}, status={status!r}",
                task_id=task_id,
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise VolcEngineSeedanceUnconfirmedTaskError(
                "La tarea de Seedance sigue en curso tras el tiempo de espera local "
                f"configurado: id={task_id}",
                task_id=task_id,
            )
        time.sleep(min(poll_interval, remaining))
