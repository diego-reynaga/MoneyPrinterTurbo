"""
Integración con la API de Upload-Post para publicar vídeos de forma cruzada en TikTok,
Instagram y YouTube Shorts.

Documentación: https://docs.upload-post.com
"""
import os
import time
from typing import Callable, Optional
from uuid import uuid4

import requests
from loguru import logger

from app.utils.request_logging import trace_http_call
from app.config import config


_UPLOAD_STATUS_POLL_INTERVAL_SECONDS = 10
_UPLOAD_STATUS_TIMEOUT_SECONDS = 3600
_MAX_CONSECUTIVE_STATUS_ERRORS = 3


class UploadPostService:
    API_BASE = "https://api.upload-post.com"

    def __init__(self, account: dict | None = None):
        # Una publicación en segundo plano debe conservar su cuenta durante la cola y el
        # sondeo. Solo el trabajo en memoria lleva esta instantánea; nunca son metadatos
        # de la tarea.
        self._account = dict(account) if account is not None else None

    def snapshot_account(self) -> dict:
        settings = dict(config.app if self._account is None else self._account)
        return {
            "upload_post_api_key": settings.get("upload_post_api_key", ""),
            "upload_post_username": settings.get("upload_post_username", ""),
            "upload_post_enabled": settings.get("upload_post_enabled", False),
        }

    def _account_setting(self, key: str, default):
        settings = config.app if self._account is None else self._account
        return settings.get(key, default)

    @staticmethod
    def _with_platform_outcome(result: dict, expected_platforms: list | None = None) -> dict:
        """Una solicitud a la API correcta puede contener publicaciones fallidas en alguna
        plataforma.
        """
        platform_results = result.get("results")
        if isinstance(platform_results, dict):
            entries = list(platform_results.items())
        elif isinstance(platform_results, list):
            entries = [
                (
                    entry.get("platform", "unknown")
                    if isinstance(entry, dict)
                    else "unknown",
                    entry,
                )
                for entry in platform_results
            ]
        else:
            if "results" in result:
                return {
                    **result,
                    "success": False,
                    "error": "Upload-Post devolvió resultados de plataforma no válidos",
                }
            return result

        failures = [
            str(platform)
            for platform, entry in entries
            if not isinstance(entry, dict)
            or entry.get("success") is not True
            or entry.get("skipped") is True
        ]
        reported_platforms = {platform for platform, _ in entries if isinstance(platform, str)}
        failures.extend(
            f"{platform} (falta el resultado)"
            for platform in dict.fromkeys(expected_platforms or [])
            if platform not in reported_platforms
        )
        if not platform_results:
            failures.append("sin resultados de plataforma")
        if failures:
            return {
                **result,
                "success": False,
                "error": "Upload-Post falló u omitió plataformas: "
                + ", ".join(failures),
            }
        return result

    def _wait_for_upload_completion(self, request_id: str, expected_platforms: list | None = None) -> dict:
        """Resuelve el paso automático de síncrono a segundo plano de Upload-Post."""
        consecutive_errors = 0
        deadline = time.monotonic() + _UPLOAD_STATUS_TIMEOUT_SECONDS
        while True:
            status_result = self.check_status(request_id)
            status = status_result.get("status")
            if status == "completed":
                if not isinstance(status_result.get("results"), (dict, list)):
                    return {
                        **status_result,
                        "request_id": request_id,
                        "success": False,
                        "error": "Upload-Post terminó sin resultados de plataforma",
                    }
                return self._with_platform_outcome(
                    {**status_result, "request_id": request_id, "success": True},
                    expected_platforms,
                )
            if status == "failed":
                return {
                    **status_result,
                    "request_id": request_id,
                    "success": False,
                    "error": status_result.get("message")
                    or "falló la subida en segundo plano de Upload-Post",
                }
            if status in {"pending", "queued", "processing", "in_progress"}:
                consecutive_errors = 0
            else:
                consecutive_errors += 1
                if consecutive_errors >= _MAX_CONSECUTIVE_STATUS_ERRORS:
                    return {
                        "success": False,
                        "request_id": request_id,
                        "error": "no se pudo confirmar el estado de Upload-Post; "
                        f"consulta el request_id {request_id}",
                    }
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(_UPLOAD_STATUS_POLL_INTERVAL_SECONDS, remaining))
        return {
            "success": False,
            "request_id": request_id,
            "error": "el estado de Upload-Post no se completó en 1 hora; "
            f"consulta el request_id {request_id}",
        }

    @property
    def api_key(self) -> str:
        return self._account_setting("upload_post_api_key", "")

    @property
    def username(self) -> str:
        return self._account_setting("upload_post_username", "")

    @property
    def enabled(self) -> bool:
        return self._account_setting("upload_post_enabled", False)

    @property
    def platforms(self) -> list:
        return config.app.get("upload_post_platforms", ["tiktok", "instagram"])

    @property
    def auto_upload(self) -> bool:
        return config.app.get("upload_post_auto_upload", False)

    @property
    def youtube_privacy_status(self) -> str:
        return config.app.get("upload_post_youtube_privacy_status", "public")

    @property
    def youtube_made_for_kids(self) -> bool:
        # Sin configurar, las siguientes subidas a YouTube se declaran explícitamente
        # como no dirigidas a niños; el contenido infantil debe configurarse como true.
        # Esto difiere de la versión antigua, que omitía el campo de declaración, pero
        # no modifica los vídeos ya subidos. Antes de subir se valida estrictamente el
        # tipo, para no tomar la cadena TOML "false" por un valor verdadero y declarar
        # mal el público.
        return config.app.get("upload_post_youtube_made_for_kids", False)

    def is_configured(self) -> bool:
        return bool(self.api_key and self.username and self.enabled)

    def upload_video(
        self,
        video_path: str,
        title: str,
        platforms: Optional[list] = None,
        privacy_level: str = "PUBLIC_TO_EVERYONE",
        youtube_extra: Optional[dict] = None,
        on_background_start: Callable[[str], None] | None = None,
    ) -> dict:
        if not self.is_configured():
            logger.warning("Upload-Post no está configurado. Se omite la publicación multiplataforma.")
            return {"success": False, "error": "Upload-Post no está configurado"}

        if platforms is None:
            platforms = self.platforms

        if not os.path.exists(video_path):
            logger.error(f"No se encontró el archivo de vídeo: {video_path}")
            return {"success": False, "error": f"No se encontró el archivo de vídeo: {video_path}"}

        has_youtube = any(p.startswith("youtube") for p in platforms)
        if has_youtube:
            # La tarea en segundo plano pasa la instantánea de la declaración tomada al
            # encolar; solo al llamar directamente al servicio se lee la configuración
            # actual.
            made_for_kids = (youtube_extra or {}).get(
                "selfDeclaredMadeForKids", self.youtube_made_for_kids
            )
            if not isinstance(made_for_kids, bool):
                error = "el ajuste de contenido para niños de YouTube debe ser un booleano"
                logger.error(error)
                return {"success": False, "error": error}

        logger.info(f"Publicando el vídeo en {', '.join(platforms)} mediante Upload-Post...")

        # Genera el identificador remoto antes del POST: una respuesta perdida no
        # demuestra que Upload-Post haya dejado de publicar el vídeo recibido.
        client_request_id = str(uuid4())
        def unconfirmed_response(message: str) -> dict:
            return {
                "success": False,
                "request_id": client_request_id,
                "error": (
                    f"{message}; el resultado de la subida no está confirmado; "
                    f"consulta el request_id {client_request_id} antes de volver a enviar"
                ),
            }

        try:
            with open(video_path, 'rb') as video_file:
                files = {'video': video_file}

                data = [
                    ('user', self.username),
                    ('request_id', client_request_id),
                    ('title', title[:2200]),
                    ('privacy_level', privacy_level),
                ]

                for platform in platforms:
                    data.append(('platform[]', platform))

                if has_youtube:
                    # El formulario multipart usa cadenas booleanas en minúscula y no
                    # puede depender de que existan metadatos del LLM; siempre que se
                    # publique en YouTube, se pasa explícitamente la declaración de
                    # público del usuario.
                    data.append(('selfDeclaredMadeForKids', str(made_for_kids).lower()))
                    # La privacidad es un ajuste de la cuenta/usuario, independiente de
                    # los títulos y descripciones generados opcionales. Se conservan las
                    # sobrescrituras encoladas.
                    data.append(('privacyStatus', (youtube_extra or {}).get(
                        "privacyStatus", self.youtube_privacy_status
                    )))
                    logger.info(f"Declaración de público de YouTube: made_for_kids={made_for_kids}")

                if youtube_extra and has_youtube:
                    if "youtube_title" in youtube_extra:
                        data.append(('youtube_title', youtube_extra["youtube_title"][:100]))
                    if "youtube_description" in youtube_extra:
                        data.append(('youtube_description', youtube_extra["youtube_description"]))
                    for tag in youtube_extra.get("tags", []):
                        data.append(('tags[]', tag))
                    data.append(('containsSyntheticMedia', "true"))

                headers = {'Authorization': f'Apikey {self.api_key}'}

                response = trace_http_call(
                    "POST", requests.post,
                    f"{self.API_BASE}/api/upload",
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=300,
                    allow_redirects=False,
                )

                if 300 <= response.status_code < 400:
                    logger.error(
                        "la subida a Upload-Post devolvió una redirección inesperada: "
                        f"status={response.status_code}"
                    )
                    return unconfirmed_response(
                        "la subida a Upload-Post devolvió una redirección inesperada"
                    )

                response.raise_for_status()
                try:
                    result = response.json()
                except ValueError:
                    logger.error("Upload-Post devolvió un JSON no válido a la subida")
                    return unconfirmed_response("Upload-Post devolvió JSON no válido")

            # Libera el archivo de origen antes de esperar una subida remota en segundo
            # plano, que puede tardar mucho más que el POST inicial.
            if not isinstance(result, dict) or not isinstance(
                result.get("success"), bool
            ):
                logger.error("Upload-Post devolvió una respuesta no válida a la subida")
                return unconfirmed_response("Upload-Post devolvió una respuesta no válida")

            if result["success"]:
                is_background = "results" not in result
                if is_background:
                    request_id = result.get("request_id")
                    if not isinstance(request_id, str) or not request_id.strip():
                        return unconfirmed_response(
                            "Upload-Post inició una subida en segundo plano sin request_id"
                        )
                    logger.info(
                        "Upload-Post aceptó la subida en segundo plano: "
                        f"request_id={request_id.strip()}"
                    )
                    if on_background_start is not None:
                        try:
                            on_background_start(request_id.strip())
                        except Exception as exc:
                            # La subida remota ya ha empezado. Un fallo local al
                            # escribir el estado no debe volver a enviar el vídeo.
                            logger.warning(
                                "no se pudo registrar el ID de solicitud de la subida en segundo plano: "
                                f"{exc}"
                            )
                    result = self._wait_for_upload_completion(request_id.strip(), platforms)
                else:
                    result = self._with_platform_outcome(result, platforms)

            if result.get("success"):
                logger.info(
                    f"¡Vídeo publicado correctamente! ID de solicitud: {result.get('request_id')}"
                )
            else:
                logger.warning(
                    f"Falló la publicación: {result.get('error') or result.get('message') or 'Error desconocido'}"
                )

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"No se pudo publicar el vídeo: {str(e)}")
            uncertain_outcome = isinstance(
                e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
            ) or (
                e.response is not None and e.response.status_code >= 500
            )
            error = str(e)
            if uncertain_outcome:
                error += (
                    "; el resultado de la subida no está confirmado; "
                    f"consulta el request_id {client_request_id} antes de volver a enviar"
                )
            return {
                "success": False,
                "request_id": client_request_id,
                "error": error,
            }

    def check_status(self, request_id: str) -> dict:
        """
        Comprueba el estado de una solicitud de subida.

        Args:
            request_id (str): El ID de solicitud devuelto por la subida

        Returns:
            dict: Información de estado
        """
        try:
            headers = {
                'Authorization': f'Apikey {self.api_key}'
            }

            response = trace_http_call(
                "GET", requests.get,
                f"{self.API_BASE}/api/uploadposts/status",
                params={'request_id': request_id},
                headers=headers,
                timeout=30
            )

            response.raise_for_status()
            try:
                result = response.json()
            except ValueError:
                logger.error("Upload-Post devolvió un JSON no válido a la consulta de estado")
                return {
                    "success": False,
                    "error": "Upload-Post devolvió un JSON de estado no válido",
                }
            if not isinstance(result, dict):
                logger.error("Upload-Post devolvió una respuesta no válida a la consulta de estado")
                return {
                    "success": False,
                    "error": "Upload-Post devolvió una respuesta no válida",
                }
            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"No se pudo comprobar el estado: {str(e)}")
            return {"success": False, "error": str(e)}


# Instancia única
upload_post_service = UploadPostService()


def cross_post_video(
    video_path: str,
    title: str,
    platforms: Optional[list] = None,
    youtube_extra: Optional[dict] = None,
    on_background_start: Callable[[str], None] | None = None,
    account: dict | None = None,
) -> dict:
    service = UploadPostService(account) if account is not None else upload_post_service
    return service.upload_video(
        video_path,
        title,
        platforms,
        youtube_extra=youtube_extra,
        on_background_start=on_background_start,
    )
