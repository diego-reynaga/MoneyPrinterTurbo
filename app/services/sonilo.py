import base64
import binascii
import json
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


DEFAULT_BASE_URL = "https://api.sonilo.com"
VIDEO_TO_MUSIC_PATH = "/v1/video-to-music"
SERVICES_PATH = "/v1/account/services"
MAX_VIDEO_DURATION_SECONDS = 360
MAX_PROMPT_LENGTH = 2000
MAX_PROXY_BYTES = 300 * 1024 * 1024
MAX_GENERATED_AUDIO_BYTES = 30 * 1024 * 1024
MAX_ERROR_BODY_BYTES = 500
# Un evento puede contener todo el audio permitido en base64, más metadatos JSON.
MAX_STREAM_EVENT_BYTES = 4 * ((MAX_GENERATED_AUDIO_BYTES + 2) // 3) + 64 * 1024
_STREAM_READ_BYTES = 64 * 1024
VIDEO_TO_MUSIC_SERVICE_ID = "video_to_music"


class SoniloError(RuntimeError):
    """Indica que falló la solicitud a Sonilo, el protocolo de respuesta o la validación
    del audio generado.
    """


def get_api_key() -> str:
    """Lee primero la configuración guardada por la WebUI; si no está configurada, se
    permite usar la variable de entorno.
    """
    configured_key = str(config.app.get("sonilo_api_key", "") or "").strip()
    return configured_key or os.getenv("SONILO_API_KEY", "").strip()


def is_enabled() -> bool:
    return bool(get_api_key())


def _base_url() -> str:
    return str(
        config.app.get("sonilo_base_url", DEFAULT_BASE_URL) or DEFAULT_BASE_URL
    ).rstrip(
        "/"
    )


def _request_timeout() -> tuple[int, int]:
    """Limita el rango de los valores de configuración, para que un infinito o un negativo
    no dejen la solicitud colgada para siempre ni fallen al instante.
    """
    raw_timeout = config.app.get("sonilo_timeout", 600)
    try:
        read_timeout = float(raw_timeout)
    except (TypeError, ValueError):
        read_timeout = 600
    if not math.isfinite(read_timeout) or read_timeout <= 0:
        read_timeout = 600
    # Requests no acepta un tiempo de espera de lectura de 0 segundos. Redondear hacia
    # arriba conserva el sentido de las configuraciones decimales y evita que int()
    # trunque 0,1-0,9 a 0 y lance un ValueError que no entra en la cadena de degradación
    # de Sonilo.
    return 15, max(1, math.ceil(min(read_timeout, 1800)))


def _normalize_service_id(service_id: str) -> str:
    """
    Normaliza el identificador de servicio de Sonilo al formato con guion bajo que usa
    internamente el proyecto.

    El 2026-07-14 la interfaz real devolvía ``video_to_music``, pero el ejemplo de la
    documentación pública del mismo día usaba ``video-to-music``. La diferencia es solo
    el separador de palabras, así que se unifica el formato en el límite del protocolo
    del tercero, para que la prueba de conexión de la interfaz no dé un falso fallo
    porque la documentación y la respuesta de producción no coincidan por un tiempo.
    """
    return service_id.strip().lower().replace("-", "_")


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
            # salte la cadena de degradación de Sonilo de quien llama.
            body = body_bytes.decode("utf-8", errors="replace")
    else:
        body = str(body_bytes)
    body = body.strip().replace("\n", " ")[:MAX_ERROR_BODY_BYTES]
    return body or response.reason or "la petición falló"


def test_connection() -> dict[str, Any]:
    """
    Valida la clave API con la interfaz de lista de servicios, que no consume saldo de
    música.

    Devuelve el JSON original para que la interfaz pueda mostrar los servicios
    disponibles, pero nunca registra en el log la clave ni las cabeceras de la
    solicitud.
    """
    api_key = get_api_key()
    if not api_key:
        raise SoniloError("Hace falta la clave API de Sonilo")
    try:
        with trace_http_call(
            "GET", requests.get,
            f"{_base_url()}{SERVICES_PATH}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=(15, 30),
            stream=True,
            allow_redirects=False,
        ) as response:
            if 300 <= response.status_code < 400:
                raise SoniloError(
                    "La comprobación de conexión de Sonilo devolvió una redirección; "
                    "no se reenvió la clave API"
                )
            if not response.ok:
                raise SoniloError(
                    f"Falló la conexión con Sonilo ({response.status_code}): "
                    f"{_safe_response_error(response)}"
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise SoniloError(
                    "Sonilo devolvió una respuesta de servicio no válida"
                ) from exc
    except requests.RequestException as exc:
        raise SoniloError(f"no se pudo conectar con Sonilo: {exc}") from exc
    if not isinstance(payload, dict):
        raise SoniloError("Sonilo devolvió una respuesta de servicio inesperada")
    available_services = payload.get("available_services")
    if not isinstance(available_services, list) or not all(
        isinstance(service_id, str) for service_id in available_services
    ):
        raise SoniloError("Sonilo devolvió una lista de servicios no válida")
    normalized_services = {
        _normalize_service_id(service_id) for service_id in available_services
    }
    if VIDEO_TO_MUSIC_SERVICE_ID not in normalized_services:
        raise SoniloError("El servicio video-to-music de Sonilo no está disponible para esta clave")
    logger.info("La prueba de conexión con Sonilo fue correcta")
    return payload


def _remove_file(file_path: str) -> None:
    """Intenta limpiar los archivos intermedios de Sonilo sin sobrescribir la excepción
    original que está tratando quien llama.
    """
    if not file_path or not os.path.exists(file_path):
        return
    try:
        os.remove(file_path)
    except OSError as exc:
        logger.warning(
            f"no se pudo eliminar el archivo temporal de Sonilo: path={file_path}, error={exc}"
        )


def _create_video_proxy(video_path: str) -> str:
    """
    Genera un vídeo proxy H.264 sin pista de audio y con el lado más largo de 1280
    píxeles.

    Sonilo solo necesita analizar el ritmo y el contenido de la imagen; subir el vídeo
    final original en alta definición aumenta la espera y el tráfico sin mejorar la
    calidad de la generación. El archivo proxy se guarda en el mismo directorio que el
    de entrada y se limpia al terminar la tarea.
    """
    descriptor, proxy_path = tempfile.mkstemp(
        prefix=".sonilo-proxy-",
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
        raise SoniloError("Se agotó el tiempo al generar el proxy de vídeo para Sonilo") from exc
    except OSError as exc:
        _remove_file(proxy_path)
        raise SoniloError("no se pudo ejecutar FFmpeg para el proxy de vídeo de Sonilo") from exc
    if result.returncode != 0:
        _remove_file(proxy_path)
        detail = (result.stderr or "").strip().replace("\n", " ")[-500:]
        raise SoniloError(f"no se pudo generar el proxy de vídeo de Sonilo: {detail}")
    proxy_size = os.path.getsize(proxy_path) if os.path.isfile(proxy_path) else 0
    if proxy_size <= 0 or proxy_size > MAX_PROXY_BYTES:
        _remove_file(proxy_path)
        raise SoniloError("El proxy de vídeo de Sonilo está vacío o supera el límite de 300 MB")
    logger.info(
        f"Proxy de vídeo de Sonilo preparado: source={video_path}, size={proxy_size} bytes"
    )
    return proxy_path


def _iter_bounded_events(response: requests.Response):
    """Lee NDJSON sin el búfer ilimitado de líneas sin terminar de Requests."""
    pending = bytearray()
    for chunk in response.iter_content(chunk_size=_STREAM_READ_BYTES):
        if not chunk:
            continue
        # Solo divide el fragmento de transporte acotado; nunca se concatena una línea
        # demasiado grande antes de comprobar su tamaño acumulado.
        pieces = chunk.split(b"\n")
        for index, piece in enumerate(pieces):
            if len(pending) + len(piece) > MAX_STREAM_EVENT_BYTES:
                raise SoniloError("El evento de streaming de Sonilo supera el límite de tamaño")
            pending.extend(piece)
            if index < len(pieces) - 1:
                if pending:
                    yield bytes(pending).rstrip(b"\r")
                pending.clear()
    if pending:
        yield bytes(pending).rstrip(b"\r")


def _parse_event(raw_line: bytes) -> dict[str, Any]:
    """Analiza estrictamente una línea NDJSON, sin ignorar en silencio respuestas truncadas
    o que no sean objetos.
    """
    try:
        event = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SoniloError("Sonilo devolvió datos de streaming mal formados") from exc
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        raise SoniloError("Sonilo devolvió un evento de streaming no válido")
    return event


def _stream_audio(response: requests.Response, temp_audio_path: str) -> tuple[int, str]:
    """
    Escribe en un archivo temporal el primer flujo de música en el orden de los eventos,
    limitando su tamaño máximo.

    La API puede devolver varios flujos candidatos a la vez; el producto actual solo
    necesita una BGM, así que se elige siempre stream_index=0. Solo se publica tras
    recibir el evento complete y decodificarlo entero con FFmpeg.
    """
    total_bytes = 0
    title = ""
    completed = False
    with open(temp_audio_path, "wb") as output:
        for raw_line in _iter_bounded_events(response):
            if not raw_line:
                continue
            event = _parse_event(raw_line)
            event_type = event["type"]
            if event_type == "error":
                message = str(
                    event.get("message") or event.get("error") or "error desconocido"
                )
                raise SoniloError(f"Falló la generación de Sonilo: {message}")
            if event_type == "title":
                title = str(event.get("title") or event.get("data") or "")[:200]
                continue
            if event_type == "complete":
                completed = True
                break
            if event_type != "audio_chunk":
                logger.debug(f"se ignora un evento de Sonilo no admitido: type={event_type}")
                continue

            stream_index = event.get("stream_index", 0)
            if stream_index != 0:
                continue
            encoded_chunk = event.get("data") or event.get("audio")
            if not isinstance(encoded_chunk, str) or not encoded_chunk:
                raise SoniloError("Sonilo devolvió un fragmento de audio vacío")
            try:
                chunk = base64.b64decode(encoded_chunk, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise SoniloError("Sonilo devolvió un fragmento de audio no válido") from exc
            if not chunk:
                raise SoniloError("Sonilo devolvió un fragmento de audio vacío")
            total_bytes += len(chunk)
            if total_bytes > MAX_GENERATED_AUDIO_BYTES:
                raise SoniloError("El audio de Sonilo supera el límite de 30 MB")
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())

    if not completed:
        raise SoniloError("El stream de Sonilo terminó antes de completarse")
    if total_bytes <= 0:
        raise SoniloError("Sonilo no devolvió datos de audio")
    return total_bytes, title


def _request_bgm(video_path: str, output_path: str, prompt: str) -> str:
    """Pide la música y la guarda de forma atómica cuando pasan la validación completa del
    protocolo y del audio.
    """
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    descriptor, temp_audio_path = tempfile.mkstemp(
        prefix=".sonilo-audio-",
        suffix=Path(output_path).suffix or ".m4a",
        dir=output_dir,
    )
    os.close(descriptor)
    try:
        logger.info(
            f"pidiendo música de fondo a Sonilo: video={video_path}, "
            f"prompt_provided={bool(prompt)}"
        )
        try:
            with open(video_path, "rb") as video_file:
                response = trace_http_call(
                    "POST", requests.post,
                    f"{_base_url()}{VIDEO_TO_MUSIC_PATH}",
                    headers={"Authorization": f"Bearer {get_api_key()}"},
                    files={"video": (Path(video_path).name, video_file, "video/mp4")},
                    data={"prompt": prompt} if prompt else None,
                    stream=True,
                    timeout=_request_timeout(),
                    allow_redirects=False,
                )
                with response:
                    if 300 <= response.status_code < 400:
                        raise SoniloError(
                            "La generación de Sonilo devolvió una redirección; "
                            "no se reenviaron el vídeo ni la clave API"
                        )
                    if not response.ok:
                        raise SoniloError(
                            f"Falló la generación de Sonilo ({response.status_code}): "
                            f"{_safe_response_error(response)}"
                        )
                    total_bytes, title = _stream_audio(response, temp_audio_path)
        except requests.RequestException as exc:
            # Los cortes de red durante iter_lines() también son excepciones de
            # requests; no se puede capturar solo la fase de conexión, o medio audio
            # podría hacer que la tarea termine con una excepción sin poder degradar.
            raise SoniloError(f"no se pudo pedir la música a Sonilo: {exc}") from exc

        try:
            bgm_service.validate_audio_file(temp_audio_path, timeout_seconds=120)
        except (bgm_service.BgmUploadError, bgm_service.BgmServiceError) as exc:
            raise SoniloError("Sonilo devolvió un audio que FFmpeg no puede decodificar") from exc
        os.replace(temp_audio_path, output_path)
        temp_audio_path = ""
        logger.info(
            f"Música de fondo de Sonilo generada: output={output_path}, "
            f"size={total_bytes} bytes, title={title or '-'}"
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
    """Genera una música de fondo de Sonilo que encaje en duración con un vídeo ya montado."""
    if not get_api_key():
        raise SoniloError("Hace falta la clave API de Sonilo")
    if not os.path.isfile(video_path):
        raise SoniloError("El vídeo de entrada de Sonilo no existe")
    try:
        duration = float(video_duration)
    except (TypeError, ValueError) as exc:
        raise SoniloError("La duración del vídeo para Sonilo no es válida") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise SoniloError("La duración del vídeo para Sonilo no es válida")
    if duration > MAX_VIDEO_DURATION_SECONDS:
        raise SoniloError("Sonilo admite vídeos de hasta 360 segundos")
    prompt = str(prompt or "").strip()
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise SoniloError("El prompt de música de Sonilo supera los 2000 caracteres")

    proxy_path = ""
    try:
        proxy_path = _create_video_proxy(video_path)
        return _request_bgm(proxy_path, output_path, prompt)
    except SoniloError:
        raise
    except OSError as exc:
        # El directorio temporal, el archivo proxy y la sustitución atómica final pueden
        # sufrir errores del sistema de archivos. Se convierten todos en SoniloError
        # para que la capa de orquestación de tareas pueda degradar, según lo previsto,
        # a "sin música de fondo" y conservar el vídeo final.
        raise SoniloError(f"Falló una operación con archivos locales de Sonilo: {exc}") from exc
    finally:
        _remove_file(proxy_path)
