import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import requests
from loguru import logger

from app.utils.request_logging import trace_http_call

from app.config import config
from app.services import bgm as bgm_service
from app.utils import utils


DEFAULT_BASE_URL = "https://api.elevenlabs.io"
VIDEO_TO_MUSIC_PATH = "/v1/music/video-to-music"
SUBSCRIPTION_PATH = "/v1/user/subscription"
DEFAULT_MODEL_ID = "music_v2"
SUPPORTED_MODEL_IDS = frozenset({"music_v1", "music_v2"})
MAX_VIDEO_DURATION_SECONDS = 600
MAX_PROMPT_LENGTH = 1000
MAX_PROXY_BYTES = 200 * 1024 * 1024
MAX_GENERATED_AUDIO_BYTES = 50 * 1024 * 1024
MAX_ERROR_BODY_BYTES = 500


class ElevenLabsMusicError(RuntimeError):
    """Indica que falló la solicitud de música de ElevenLabs, la generación del proxy o la
    validación del audio devuelto.
    """


class ElevenLabsPaidPlanRequiredError(ElevenLabsMusicError):
    """Indica que la clave es válida, pero el plan de la cuenta actual no incluye
    ElevenLabs Music API.
    """


class ElevenLabsAuthenticationError(ElevenLabsMusicError):
    """Indica que falta la clave API de ElevenLabs o que el servidor la ha rechazado."""


def get_api_key() -> str:
    """
    Lee la clave API compartida de ElevenLabs.

    La música y el TTS de ElevenLabs existente usan la misma configuración de cuenta,
    para que el usuario no tenga que mantener dos claves en la WebUI; la variable de
    entorno solo es un respaldo cuando la configuración local está vacía.
    """
    configured_key = str(config.elevenlabs.get("api_key", "") or "").strip()
    return configured_key or os.getenv("ELEVENLABS_API_KEY", "").strip()


def is_enabled() -> bool:
    return bool(get_api_key())


def _base_url() -> str:
    return str(
        config.elevenlabs.get("music_base_url", DEFAULT_BASE_URL)
        or DEFAULT_BASE_URL
    ).rstrip("/")


def _model_id() -> str:
    """Solo se permiten los modelos que publica actualmente el Video-to-Music oficial; ante
    una configuración errónea se vuelve con seguridad al valor por defecto.
    """
    model_id = str(
        config.elevenlabs.get("music_model_id", DEFAULT_MODEL_ID)
        or DEFAULT_MODEL_ID
    ).strip()
    return model_id if model_id in SUPPORTED_MODEL_IDS else DEFAULT_MODEL_ID


def _request_timeout() -> tuple[int, int]:
    """Limita el tiempo de espera de lectura de la música, equilibrando el tiempo de
    generación de vídeos largos con la recuperación ante configuraciones erróneas.
    """
    raw_timeout = config.elevenlabs.get("music_timeout", 600)
    try:
        read_timeout = float(raw_timeout)
    except (TypeError, ValueError):
        read_timeout = 600
    if not math.isfinite(read_timeout) or read_timeout <= 0:
        read_timeout = 600
    return 15, max(1, math.ceil(min(read_timeout, 1800)))


def _safe_response_error(response: requests.Response) -> str:
    """Solo lee una cantidad limitada del cuerpo de error del tercero, para que una
    respuesta anómala no agote la memoria ni ensucie el log de la tarea.
    """
    try:
        body_bytes = next(
            response.iter_content(chunk_size=MAX_ERROR_BODY_BYTES),
            b"",
        )
    except requests.RequestException:
        body_bytes = b""
    if isinstance(body_bytes, bytes):
        try:
            body = body_bytes.decode(
                response.encoding or "utf-8",
                errors="replace",
            )
        except LookupError:
            # response.encoding se toma directamente del charset que declara el
            # proveedor; un valor desconocido (por ejemplo charset=unknown-charset) hace
            # que codecs lance LookupError, mientras que errors solo afecta a
            # UnicodeDecodeError. El antiguo response.text volvía internamente a UTF-8;
            # aquí se mantiene el mismo comportamiento, para que esa excepción no se
            # salte la cadena de degradación de música de quien llama.
            body = body_bytes.decode("utf-8", errors="replace")
    else:
        body = str(body_bytes)
    body = body.strip().replace("\n", " ")[:MAX_ERROR_BODY_BYTES]
    return body or response.reason or "la petición falló"


def test_connection() -> dict[str, Any]:
    """
    Comprueba la clave API y el plan de la cuenta con la interfaz de suscripción, que no
    consume saldo de generación musical.

    Esta interfaz solo confirma que la clave puede acceder a la información de
    suscripción y que la cuenta no es de plan gratuito; no demuestra que la clave tenga
    permiso para el endpoint de Music. ElevenLabs permite limitar las claves por
    endpoint, saldo e IP, así que el aviso de éxito de la interfaz debe mantener ese
    matiz, y el permiso real lo confirma al final la solicitud de generación. Los
    detalles de facturación y uso de la respuesta no se escriben en el log, para no
    registrar datos privados de la cuenta.
    """
    api_key = get_api_key()
    if not api_key:
        raise ElevenLabsAuthenticationError("Hace falta la clave API de ElevenLabs")
    try:
        with trace_http_call(
            "GET", requests.get,
            f"{_base_url()}{SUBSCRIPTION_PATH}",
            headers={"xi-api-key": api_key},
            timeout=(15, 30),
            stream=True,
            allow_redirects=False,
        ) as response:
            if 300 <= response.status_code < 400:
                raise ElevenLabsMusicError(
                    "La comprobación de la cuenta de ElevenLabs devolvió una redirección; "
                    "no se reenvió la clave API"
                )
            if response.status_code == 401:
                raise ElevenLabsAuthenticationError(
                    "ElevenLabs rechazó la clave API (401): "
                    f"{_safe_response_error(response)}"
                )
            if not response.ok:
                raise ElevenLabsMusicError(
                    "Falló la comprobación de la cuenta de ElevenLabs "
                    f"({response.status_code}): "
                    f"{_safe_response_error(response)}"
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise ElevenLabsMusicError(
                    "ElevenLabs devolvió una respuesta de suscripción no válida"
                ) from exc
    except requests.RequestException as exc:
        raise ElevenLabsMusicError(
            f"no se pudo conectar con ElevenLabs: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ElevenLabsMusicError(
            "ElevenLabs devolvió una respuesta de suscripción inesperada"
        )
    tier = str(payload.get("tier") or "").strip().lower()
    if not tier:
        raise ElevenLabsMusicError(
            "La respuesta de suscripción de ElevenLabs no incluye el nivel de la cuenta"
        )
    if tier == "free":
        raise ElevenLabsPaidPlanRequiredError(
            "ElevenLabs Music API requiere un plan de pago; "
            "la cuenta actual está en el nivel gratuito"
        )
    logger.info(f"Comprobación de la cuenta y el plan de ElevenLabs correcta: tier={tier}")
    return payload


def validate_generation_access() -> None:
    """
    Descarta, antes de que empiece el costoso flujo de vídeo, las cuentas que seguro no
    pueden generar música.

    El plan gratuito y las claves no válidas son errores deterministas y deben detener
    la tarea de inmediato, para no gastar antes el saldo del LLM, del TTS y de los
    servicios de material. La interfaz de suscripción también puede ser inaccesible por
    un alcance de endpoint solo de Music, límites de IP o problemas de red temporales;
    esos resultados no demuestran que Music API no esté disponible, así que solo se
    registra un aviso y se deja que decida la solicitud de generación real, para no
    bloquear por error una clave limitada pero funcional.
    """
    try:
        test_connection()
    except (ElevenLabsPaidPlanRequiredError, ElevenLabsAuthenticationError):
        raise
    except ElevenLabsMusicError as exc:
        logger.warning(
            "La comprobación previa de la cuenta de ElevenLabs no fue concluyente; "
            f"la generación verificará el acceso a Music API: error={exc}"
        )


def _remove_file(file_path: str) -> None:
    """Intenta limpiar los archivos intermedios de ElevenLabs sin sobrescribir la excepción
    original que está tratando quien llama.
    """
    if not file_path or not os.path.exists(file_path):
        return
    try:
        os.remove(file_path)
    except OSError as exc:
        logger.warning(
            "no se pudo eliminar el archivo temporal de ElevenLabs: "
            f"path={file_path}, error={exc}"
        )


def _create_video_proxy(video_path: str) -> str:
    """
    Genera un vídeo proxy H.264 sin pista de audio y con el lado más largo de 1280
    píxeles.

    Video-to-Music solo analiza la imagen; subir el vídeo final original en alta
    definición no mejora la música y aumenta el tráfico y la espera. El proxy se limita
    estrictamente al máximo oficial de 200 MB y se borra al terminar la solicitud.
    """
    descriptor, proxy_path = tempfile.mkstemp(
        prefix=".elevenlabs-music-proxy-",
        suffix=".mp4",
        dir=os.path.dirname(os.path.abspath(video_path)),
    )
    os.close(descriptor)
    command = [
        utils.get_ffmpeg_binary(),
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        video_path,
        "-vf",
        (
            "scale=w=1280:h=1280:force_original_aspect_ratio=decrease:"
            "force_divisible_by=2"
        ),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "30",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-fs",
        str(MAX_PROXY_BYTES),
        proxy_path,
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _remove_file(proxy_path)
        raise ElevenLabsMusicError(
            "Se agotó el tiempo al generar el proxy de vídeo para ElevenLabs"
        ) from exc
    except OSError as exc:
        _remove_file(proxy_path)
        raise ElevenLabsMusicError(
            "no se pudo ejecutar FFmpeg para el proxy de vídeo de ElevenLabs"
        ) from exc
    if result.returncode != 0:
        _remove_file(proxy_path)
        detail = (result.stderr or "").strip().replace("\n", " ")[-500:]
        raise ElevenLabsMusicError(
            f"no se pudo generar el proxy de vídeo de ElevenLabs: {detail}"
        )
    proxy_size = os.path.getsize(proxy_path) if os.path.isfile(proxy_path) else 0
    if proxy_size <= 0 or proxy_size > MAX_PROXY_BYTES:
        _remove_file(proxy_path)
        raise ElevenLabsMusicError(
            "El proxy de vídeo de ElevenLabs está vacío o supera el límite de 200 MB"
        )
    logger.info(
        "Proxy de vídeo de ElevenLabs preparado: "
        f"source={video_path}, size={proxy_size} bytes"
    )
    return proxy_path


def _stream_audio(response: requests.Response, temp_audio_path: str) -> int:
    """Guarda el audio por bloques y limita su tamaño máximo, para que una respuesta
    anómala no agote el disco local.
    """
    total_bytes = 0
    with open(temp_audio_path, "wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            total_bytes += len(chunk)
            if total_bytes > MAX_GENERATED_AUDIO_BYTES:
                raise ElevenLabsMusicError(
                    "El audio de ElevenLabs supera el límite de 50 MB"
                )
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    if total_bytes <= 0:
        raise ElevenLabsMusicError("ElevenLabs no devolvió datos de audio")
    return total_bytes


def _request_bgm(video_path: str, output_path: str, prompt: str) -> str:
    """Pide música a ElevenLabs, la descarga entera y la valida con FFmpeg antes de
    publicarla de forma atómica.
    """
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    descriptor, temp_audio_path = tempfile.mkstemp(
        prefix=".elevenlabs-music-",
        suffix=Path(output_path).suffix or ".mp3",
        dir=output_dir,
    )
    os.close(descriptor)
    try:
        model_id = _model_id()
        logger.info(
            "pidiendo música de fondo a ElevenLabs: "
            f"video={video_path}, model={model_id}, "
            f"prompt_provided={bool(prompt)}"
        )
        request_data = {"model_id": model_id}
        if prompt:
            request_data["description"] = prompt
        try:
            with open(video_path, "rb") as video_file:
                response = trace_http_call(
                    "POST", requests.post,
                    f"{_base_url()}{VIDEO_TO_MUSIC_PATH}",
                    headers={"xi-api-key": get_api_key()},
                    params={"output_format": "mp3_44100_128"},
                    files=[
                        (
                            # La documentación oficial muestra el array del formulario
                            # como ``videos[]``, pero a fecha 2026-07-18 la interfaz de
                            # producción devuelve 422 con ese campo; el nombre real del
                            # parámetro de Starlette es ``videos``. Al subir varios,
                            # requests puede seguir añadiendo campos con el mismo
                            # nombre.
                            "videos",
                            (Path(video_path).name, video_file, "video/mp4"),
                        )
                    ],
                    data=request_data,
                    stream=True,
                    timeout=_request_timeout(),
                    allow_redirects=False,
                )
                with response:
                    if 300 <= response.status_code < 400:
                        raise ElevenLabsMusicError(
                            "La generación de ElevenLabs devolvió una redirección; "
                            "no se reenviaron el vídeo ni la clave API"
                        )
                    if not response.ok:
                        raise ElevenLabsMusicError(
                            "Falló la generación de ElevenLabs "
                            f"({response.status_code}): "
                            f"{_safe_response_error(response)}"
                        )
                    total_bytes = _stream_audio(response, temp_audio_path)
        except requests.RequestException as exc:
            # Un corte durante la descarga también es un fallo de la solicitud y debe
            # entrar en la lógica de degradación de la tarea; no puede dejar medio audio
            # ni hacer fallar por completo un vídeo ya generado por un problema de red
            # de un tercero.
            raise ElevenLabsMusicError(
                f"no se pudo pedir la música a ElevenLabs: {exc}"
            ) from exc

        try:
            bgm_service.validate_audio_file(temp_audio_path, timeout_seconds=120)
        except (bgm_service.BgmUploadError, bgm_service.BgmServiceError) as exc:
            raise ElevenLabsMusicError(
                "ElevenLabs devolvió un audio que FFmpeg no puede decodificar"
            ) from exc
        os.replace(temp_audio_path, output_path)
        temp_audio_path = ""
        logger.info(
            "Música de fondo de ElevenLabs generada: "
            f"output={output_path}, size={total_bytes} bytes"
        )
        return output_path
    finally:
        _remove_file(temp_audio_path)


def generate_bgm(
    video_path: str,
    output_path: str,
    video_duration: float,
    prompt: str = "",
) -> str:
    """Genera una música de fondo de ElevenLabs que encaje en duración e imagen con un
    vídeo ya montado.
    """
    if not get_api_key():
        raise ElevenLabsMusicError("Hace falta la clave API de ElevenLabs")
    if not os.path.isfile(video_path):
        raise ElevenLabsMusicError("El vídeo de entrada de ElevenLabs no existe")
    try:
        duration = float(video_duration)
    except (TypeError, ValueError) as exc:
        raise ElevenLabsMusicError(
            "La duración del vídeo para ElevenLabs no es válida"
        ) from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ElevenLabsMusicError("La duración del vídeo para ElevenLabs no es válida")
    if duration > MAX_VIDEO_DURATION_SECONDS:
        raise ElevenLabsMusicError(
            "ElevenLabs admite vídeos de hasta 600 segundos"
        )
    prompt = str(prompt or "").strip()
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ElevenLabsMusicError(
            "El prompt de música de ElevenLabs supera los 1000 caracteres"
        )

    proxy_path = ""
    try:
        proxy_path = _create_video_proxy(video_path)
        return _request_bgm(proxy_path, output_path, prompt)
    except ElevenLabsMusicError:
        raise
    except OSError as exc:
        raise ElevenLabsMusicError(
            f"Falló una operación con archivos locales de ElevenLabs: {exc}"
        ) from exc
    finally:
        _remove_file(proxy_path)
