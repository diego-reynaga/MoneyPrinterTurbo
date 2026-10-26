"""Texto a vídeo asíncrono de Infistar: hace pedidos con especificaciones de modelo
verificadas y conserva las pistas para recuperar tareas de pago.
"""

import hashlib
import math
import os
from pathlib import Path
import tempfile
import time
from collections.abc import Mapping
from urllib.parse import quote, urljoin, urlsplit

import requests
from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.services import material_upload
from app.utils.request_logging import sanitize_payload, trace_http_call

DEFAULT_BASE_URL = "https://infistar.cc/v1"
DEFAULT_MODEL = "doubao-seedance-2-0-fast-260128"
# Usa las especificaciones de clip corto del ejemplo de la documentación y de pruebas
# reales, para no aplicar restricciones de duración/tamaño de otros modelos.
MODELS = {
    DEFAULT_MODEL: {"label": "Seedance 2.0 Fast", "seconds": 5, "size": "720p"},
    "MiniMax-H3": {"label": "MiniMax H3", "seconds": 6, "size": "768p"},
    "kling-v2-6": {"label": "Kling 2.6", "seconds": 5, "size": "720p"},
}
ACTIVE_STATUSES = {"pending", "queued", "in_progress", "running", "processing"}
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024


class InfistarVideoError(RuntimeError):
    def __init__(self, message, task_id=""):
        super().__init__(message)
        self.task_id = task_id


class InfistarVideoUnconfirmedError(InfistarVideoError):
    """Si el envío de pago o el estado remoto no están confirmados, no se puede volver a
    pedir automáticamente.
    """


def get_api_key(settings: Mapping | None = None) -> str:
    values = config.app if settings is None else settings
    return (
        str(values.get("infistar_video_api_key", "") or "").strip()
        or os.getenv("INFISTAR_VIDEO_API_KEY", "").strip()
    )


def is_enabled(settings: Mapping | None = None) -> bool:
    return bool(get_api_key(settings))


def model_id(settings: Mapping | None = None) -> str:
    values = config.app if settings is None else settings
    value = str(values.get("infistar_video_model", "") or DEFAULT_MODEL).strip()
    if value not in MODELS:
        raise InfistarVideoError(f"Modelo de vídeo de Infistar no admitido: {value}")
    return value


def _base_url() -> str:
    # La configuración de vídeo es independiente del LLM; cambiar de modelo o de su
    # pasarela no cambia las credenciales ni la dirección del vídeo.
    value = (
        str(config.app.get("infistar_video_base_url", "") or DEFAULT_BASE_URL)
        .strip()
        .rstrip("/")
    )
    parts = urlsplit(value)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.netloc
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise InfistarVideoError(
            "La URL base de Infistar debe ser una URL HTTP(S) sin credenciales ni parámetros de consulta"
        )
    return value


def _setting_seconds(key, default, minimum, maximum):
    try:
        value = float(config.app.get(key, default))
    except (ValueError, TypeError):
        return default
    return min(max(value, minimum), maximum) if math.isfinite(value) else default


def _request(method, url, **kwargs):
    verify = config.app.get("tls_verify", True)
    if isinstance(verify, str):
        verify = verify.lower().strip() not in {"false", "0", "off", "no", ""}
    return trace_http_call(
        method,
        getattr(requests, method.lower()),
        url,
        proxies=config.proxy,
        verify=bool(verify),
        allow_redirects=False,
        **kwargs,
    )


def _detail(response, secret):
    try:
        body = response.json()
        error = (body.get("error") or body) if isinstance(body, dict) else {}
        text = error.get("message", "") if isinstance(error, dict) else str(error)
    except (ValueError, TypeError):
        text = ""
    return str(sanitize_payload(str(text).replace(secret, "***")))[:500]


def _json(response, task_id=""):
    try:
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError()
        return body
    except (ValueError, TypeError) as exc:
        raise InfistarVideoUnconfirmedError(
            "Infistar devolvió una respuesta de tarea ilegible", task_id
        ) from exc


def _wait(task_id, base, headers, secret):
    deadline = time.monotonic() + _setting_seconds(
        "infistar_video_run_timeout", 1800, 60, 7200
    )
    interval = _setting_seconds("infistar_video_poll_interval", 5, 1, 60)
    failures = 0
    previous = ""
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        phase = max(min(remaining / 2, 30), 0.001)
        try:
            response = _request(
                "GET",
                f"{base}/videos/{quote(task_id, safe='')}",
                headers=headers,
                timeout=(phase, phase),
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"HTTP {response.status_code}")
            if response.status_code != 200:
                raise InfistarVideoUnconfirmedError(
                    f"El estado de la tarea de Infistar no está disponible: HTTP {response.status_code}, {_detail(response, secret)}",
                    task_id,
                )
            body = _json(response, task_id)
        except (requests.RequestException, OSError) as exc:
            failures += 1
            if failures > 5:
                raise InfistarVideoUnconfirmedError(
                    "Falló el sondeo de Infistar; conserva el ID de la tarea y revisa la consola",
                    task_id,
                ) from exc
            time.sleep(min(interval, max(deadline - time.monotonic(), 0)))
            continue
        failures = 0
        status = str(body.get("status", "")).strip().lower()
        if status != previous:
            logger.info(f"Tarea de vídeo de Infistar: id={task_id}, status={status}")
            previous = status
        if status in {"completed", "succeeded"}:
            return body
        if status in {"failed", "error", "cancelled", "canceled", "expired"}:
            detail = str(
                sanitize_payload(str(body.get("error", "")).replace(secret, "***"))
            )[:500]
            raise InfistarVideoError(f"Falló la tarea de vídeo de Infistar: {detail}", task_id)
        if status not in ACTIVE_STATUSES:
            raise InfistarVideoUnconfirmedError(
                f"Infistar devolvió un estado desconocido: {status!r}", task_id
            )
        time.sleep(min(interval, max(deadline - time.monotonic(), 0)))
    raise InfistarVideoUnconfirmedError(
        "El vídeo de Infistar sigue en curso tras el tiempo de espera local", task_id
    )


def _download(task_id, base, headers, output_dir):
    """La clave solo se envía al endpoint de content original; tras redirigir al
    almacenamiento de objetos no se envían credenciales.
    """
    url = f"{base}/videos/{quote(task_id, safe='')}/content"
    response = None
    staged = None
    try:
        for hop in range(6):
            response = _request(
                "GET",
                url,
                headers=headers if hop == 0 else {},
                timeout=(30, 120),
                stream=True,
            )
            if response.status_code not in {301, 302, 303, 307, 308}:
                break
            next_url = urljoin(url, response.headers.get("Location", ""))
            response.close()
            response = None
            if urlsplit(next_url).scheme != "https" or not urlsplit(next_url).hostname:
                raise ValueError("La descarga del vídeo se redirigió a una URL no válida")
            url = next_url
        else:
            raise ValueError("Demasiadas redirecciones en la descarga del vídeo")
        if response.status_code != 200:
            raise ValueError(f"Falló la descarga del vídeo: HTTP {response.status_code}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(task_id.encode()).hexdigest()[:24] + ".mp4"
        destination = Path(output_dir) / f"infistar-{name}"
        with tempfile.NamedTemporaryFile(
            dir=output_dir, suffix=".part", delete=False
        ) as stream:
            staged = Path(stream.name)
            count = 0
            for chunk in response.iter_content(1024 * 1024):
                count += len(chunk)
                if count > MAX_DOWNLOAD_BYTES:
                    raise ValueError("La descarga del vídeo supera los 512 MiB")
                stream.write(chunk)
            if count == 0:
                raise ValueError("La descarga del vídeo está vacía")
        # Reutiliza la validación de material: rechaza páginas de error, listas de
        # reproducción y vídeos truncados que solo tienen cabecera; solo después de
        # decodificarlo por completo se entrega el archivo, guardado de forma atómica, a
        # la composición de material posterior.
        material_upload._validate_video(str(staged))
        from moviepy import VideoFileClip

        with VideoFileClip(str(staged)) as clip:
            duration = float(clip.duration)
            width, height = clip.size
            if not math.isfinite(duration) or duration < 1:
                raise ValueError("El vídeo generado dura menos de un segundo")
        staged.replace(destination)
        return destination, duration, width, height
    except Exception as exc:
        raise InfistarVideoError(
            f"Infistar generó un vídeo de pago pero falló la descarga/validación ({type(exc).__name__}); conserva el ID de la tarea para recuperarlo",
            task_id,
        ) from exc
    finally:
        if response is not None:
            response.close()
        if staged is not None:
            staged.unlink(missing_ok=True)


def generate_video(*, search_term, video_aspect, output_dir) -> MaterialInfo:
    secret = get_api_key()
    if not secret:
        raise InfistarVideoError("La generación de vídeo de Infistar necesita una clave API")
    model = model_id()
    profile = MODELS[model]
    aspect = VideoAspect(video_aspect)
    if aspect not in {VideoAspect.landscape, VideoAspect.portrait}:
        raise InfistarVideoError("El vídeo de Infistar admite actualmente 16:9 y 9:16")
    term = str(search_term or "").strip()
    if not term or len(term) > 2000:
        raise InfistarVideoError("El prompt de vídeo de Infistar debe tener entre 1 y 2000 caracteres")
    base = _base_url()
    headers = {"Authorization": f"Bearer {secret}"}
    size = "1280x720" if aspect == VideoAspect.landscape else "720x1280"
    if model == "MiniMax-H3":
        size = "768p"
    payload = {
        "model": model,
        "prompt": term,
        "seconds": str(profile["seconds"]),
        "size": size,
    }
    # La ruta actual de MiniMax ignora aspect_ratio y size solo acepta niveles de
    # resolución. Se genera explícitamente material horizontal; el vídeo vertical se
    # recorta o se rellena con la lógica de ajuste de imagen existente, y la WebUI lo
    # indica.
    try:
        response = _request(
            "POST", f"{base}/videos", json=payload, headers=headers, timeout=(30, 90)
        )
    except requests.RequestException as exc:
        raise InfistarVideoUnconfirmedError(
            "El envío a Infistar no devolvió respuesta; puede que ya exista una tarea de pago. Revisa la consola antes de reintentar."
        ) from exc
    if response.status_code >= 500 or 300 <= response.status_code < 400:
        raise InfistarVideoUnconfirmedError(
            f"Envío a Infistar sin confirmar: HTTP {response.status_code}; no lo vuelvas a enviar automáticamente"
        )
    if not 200 <= response.status_code < 300:
        raise InfistarVideoError(
            f"Solicitud de vídeo de Infistar rechazada: HTTP {response.status_code}, {_detail(response, secret)}"
        )
    body = _json(response)
    task_id = str(body.get("id") or body.get("task_id") or "").strip()
    if not task_id:
        raise InfistarVideoUnconfirmedError(
            "Infistar aceptó el envío sin un ID de tarea; revisa la consola"
        )
    logger.info(f"Tarea de vídeo de pago de Infistar creada: id={task_id}, model={model}")
    _wait(task_id, base, headers, secret)
    path, duration, width, height = _download(task_id, base, headers, output_dir)
    return MaterialInfo(
        provider="infistar",
        url=str(path),
        duration=int(duration),
        source_info={
            "provider": "infistar",
            "search_term": term,
            "asset_id": task_id,
            "rendition": {"id": model, "width": width, "height": height},
        },
    )
