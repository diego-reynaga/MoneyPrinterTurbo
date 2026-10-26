import mimetypes
import os
import pathlib
import shutil
from typing import Union
from urllib.parse import quote

from fastapi import BackgroundTasks, Depends, Path, Query, Request, UploadFile
from fastapi.params import File
from fastapi.responses import FileResponse, Response, StreamingResponse
from loguru import logger
from starlette.background import BackgroundTask

from app.config import config
from app.controllers import base
from app.controllers.manager.base_manager import TaskQueueFullError
from app.controllers.manager.memory_manager import InMemoryTaskManager
from app.controllers.manager.redis_manager import RedisTaskManager
from app.controllers.v1.base import new_router
from app.models.exception import HttpException
from app.models.schema import (
    AudioRequest,
    BgmRetrieveResponse,
    BgmUploadResponse,
    SubtitleRequest,
    TaskDeletionResponse,
    TaskListResponse,
    TaskQueryRequest,
    TaskQueryResponse,
    TaskResponse,
    TaskVideoRequest,
    VideoMaterialUploadResponse,
    VideoMaterialRetrieveResponse
)
from app.services import bgm as bgm_service
from app.services import material_upload as material_upload_service
from app.services import state as sm
from app.services import task as tm
from app.utils import file_security, utils

# La autenticación se aplica de forma común en la entrada de las rutas de vídeo V1.
# verify_token mantiene el comportamiento actual sin autenticación cuando api_key está
# vacía, y solo afecta a los clientes cuando el administrador la configura
# explícitamente.
router = new_router(dependencies=[Depends(base.verify_token)])

_enable_redis = config.app.get("enable_redis", False)
_redis_host = config.app.get("redis_host", "localhost")
_redis_port = config.app.get("redis_port", 6379)
_redis_db = config.app.get("redis_db", 0)
_redis_password = config.app.get("redis_password", None)
_max_concurrent_tasks = config.app.get("max_concurrent_tasks", 5)
_max_queued_tasks = config.app.get("max_queued_tasks", 100)


def _build_redis_url(host: str, port: int, db: int, password: str | None) -> str:
    # Las contraseñas son userinfo de la URL. Se escapan los caracteres reservados para
    # que Redis reciba exactamente el secreto configurado en lugar de interpretar parte
    # de él como host, puerto, ruta, query o fragmento.
    auth = f":{quote(password, safe='')}@" if password else ""
    # Las autoridades de URL exigen corchetes alrededor de los literales IPv6.
    # RedisState también acepta un host IPv6 sin más, así que el cliente de la cola se
    # mantiene compatible con él.
    url_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"redis://{auth}{url_host}:{port}/{db}"


redis_url = _build_redis_url(_redis_host, _redis_port, _redis_db, _redis_password)
# Elige el gestor de tareas adecuado según la configuración
if _enable_redis:
    task_manager = RedisTaskManager(
        max_concurrent_tasks=_max_concurrent_tasks,
        redis_url=redis_url,
        max_queued_tasks=_max_queued_tasks,
    )
else:
    task_manager = InMemoryTaskManager(
        max_concurrent_tasks=_max_concurrent_tasks,
        max_queued_tasks=_max_queued_tasks,
    )


def _sanitize_upload_filename(filename: str, request_id: str) -> str:
    # El navegador o el cliente a veces adjuntan información de directorio, e incluso
    # fragmentos de recorrido como ../. Aquí solo se conserva el nombre de archivo puro,
    # para que la interfaz de subida no escriba el archivo fuera del directorio de
    # destino.
    normalized_name = (filename or "").replace("\\", "/").split("/")[-1].strip()
    if not normalized_name or normalized_name in {".", ".."}:
        raise HttpException(
            task_id=request_id,
            status_code=400,
            message=f"{request_id}: nombre de archivo no válido",
        )
    return normalized_name


def _resolve_path_within_directory(base_dir: str, unsafe_path: str, request_id: str) -> str:
    try:
        return file_security.resolve_path_within_directory(base_dir, unsafe_path)
    except ValueError as exc:
        logger.warning(
            f"se rechaza una ruta de archivo insegura, request_id: {request_id}, path: {unsafe_path}, "
            f"error: {str(exc)}"
        )
        raise HttpException(
            task_id=request_id,
            status_code=404 if str(exc) == "el archivo no existe" else 403,
            message=f"{request_id}: ruta de archivo no válida",
        )


def _public_task_data(task: dict) -> dict:
    """Copia el estado de la tarea quitando los campos internos que solo sirven para
    coordinar procesos del servidor.
    """
    public_task = dict(task)
    public_task.pop("cross_post_owner", None)
    return public_task


def _task_file_to_uri(file: str, endpoint: str, task_dir: str, request_id: str) -> str:
    if not isinstance(file, str):
        return file

    if file.startswith(("http://", "https://")):
        return file

    try:
        resolved_path = file_security.resolve_path_within_directory(task_dir, file)
    except ValueError as exc:
        # En teoría, el estado de la tarea solo debería guardar rutas de artefactos
        # dentro del directorio de la tarea. Aquí ya no se sigue construyendo la URL,
        # para no envolver una ruta anómala en un enlace accesible; a la vez se conserva
        # el valor original, para poder investigar datos históricos sucios.
        logger.warning(
            f"se omite una ruta de salida de tarea insegura, request_id: {request_id}, path: {file}, "
            f"error: {str(exc)}"
        )
        return file

    relative_path = os.path.relpath(
        resolved_path, os.path.realpath(task_dir)
    ).replace("\\", "/")
    uri_path = f"tasks/{quote(relative_path, safe='/')}"
    if endpoint:
        return f"{endpoint.rstrip('/')}/{uri_path}"
    return f"/{uri_path}"


def _task_response_data(task: dict, endpoint: str, task_dir: str, request_id: str) -> dict:
    response_task = _public_task_data(task)
    for key in ("videos", "combined_videos"):
        if key in task:
            response_task[key] = [
                _task_file_to_uri(file, endpoint, task_dir, request_id)
                for file in task[key]
            ]
    for key in ("audio_file", "subtitle_path"):
        if task.get(key):
            response_task[key] = _task_file_to_uri(
                task[key], endpoint, task_dir, request_id
            )
    return response_task


def _parse_byte_range(
    range_header: str | None, file_size: int, request_id: str
) -> tuple[int, int]:
    """Analiza un Range HTTP de un solo segmento y convierte de forma estable las
    solicitudes no válidas o fuera de rango en 416.
    """
    # 416 solo se usa para responder a solicitudes Range. Sin Range, un archivo vacío
    # también es una representación completa, y se devuelve un 200 de 0 bytes, igual que
    # /download.
    if not range_header:
        return 0, file_size - 1

    if file_size <= 0:
        raise HttpException(
            task_id=request_id,
            status_code=416,
            message=f"{request_id}: el rango solicitado no se puede satisfacer",
        )

    try:
        # El reproductor de vídeo solo necesita aquí un bytes range de un segmento.
        # Rechazar las solicitudes de varios segmentos evita que el cuerpo devuelto no
        # coincida con Content-Range, y que cadenas anómalas lleguen a int() y produzcan
        # un 500.
        unit, separator, positions = range_header.strip(" \t").partition("=")
        if unit.lower() != "bytes" or not separator or "," in positions:
            raise ValueError("formato de rango no admitido")
        start_text, end_text = positions.split("-", 1)
        if not start_text and not end_text:
            raise ValueError("rango vacío")
        # Las posiciones HTTP usan DIGIT ASCII, no la sintaxis de int más amplia de
        # Python.
        if any(value and not (value.isascii() and value.isdigit())
               for value in (start_text, end_text)):
            raise ValueError("posición de byte no válida")

        if not start_text:
            suffix_length = int(end_text)
            if suffix_length <= 0:
                raise ValueError("longitud de sufijo no válida")
            start = max(file_size - suffix_length, 0)
            end = file_size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
            if start < 0 or start >= file_size or end < start:
                raise ValueError("rango fuera del archivo")
            end = min(end, file_size - 1)
    except (TypeError, ValueError) as exc:
        logger.warning(
            f"se rechaza un rango de vídeo no válido, request_id: {request_id}, "
            f"range: {range_header}, file_size: {file_size}, error: {str(exc)}"
        )
        raise HttpException(
            task_id=request_id,
            status_code=416,
            message=f"{request_id}: el rango solicitado no se puede satisfacer",
        ) from exc

    return start, end


@router.post("/videos", response_model=TaskResponse, summary="Generar un vídeo corto")
def create_video(
    background_tasks: BackgroundTasks, request: Request, body: TaskVideoRequest
):
    return create_task(request, body, stop_at="video")


@router.post("/subtitle", response_model=TaskResponse, summary="Generar solo los subtítulos")
def create_subtitle(
    background_tasks: BackgroundTasks, request: Request, body: SubtitleRequest
):
    return create_task(request, body, stop_at="subtitle")


@router.post("/audio", response_model=TaskResponse, summary="Generar solo el audio")
def create_audio(
    background_tasks: BackgroundTasks, request: Request, body: AudioRequest
):
    return create_task(request, body, stop_at="audio")


def create_task(
    request: Request,
    body: Union[TaskVideoRequest, SubtitleRequest, AudioRequest],
    stop_at: str,
):
    task_id = utils.get_uuid()
    request_id = base.get_task_id(request)
    try:
        if (
            stop_at == "video"
            and isinstance(body, TaskVideoRequest)
            and body.subtitle_enabled
        ):
            tm.video.validate_subtitle_colors(body)
            # El nombre de la fuente lo puede enviar directamente el cliente de la API;
            # no se puede esperar al renderizado en segundo plano para descubrir que la
            # ruta sale de los límites. Aquí se comparte con la capa de renderizado la
            # validación de límites del directorio, para que las solicitudes no válidas
            # devuelvan 400 antes de crear tareas de pago.
            file_security.resolve_path_within_directory(
                utils.font_dir(), body.font_name or "STHeitiMedium.ttc"
            )
        task = {
            "task_id": task_id,
            "request_id": request_id,
            "params": body.model_dump(),
        }
        sm.state.update_task(task_id)
        try:
            task_manager.add_task(
                tm.start, task_id=task_id, params=body, stop_at=stop_at
            )
        except Exception:
            # El registro de estado se crea antes de planificar y por defecto se marca
            # como processing. Si el planificador no logra hacerse cargo de la tarea
            # (por ejemplo porque falla el arranque del hilo o la cola de Redis no está
            # disponible), hay que deshacer ese registro; si no, la API y la WebUI
            # mostrarían para siempre una tarea que en realidad nunca se ejecutó.
            sm.state.delete_task(task_id)
            raise
        logger.success(f"Tarea creada: {utils.to_json(task)}")
        return utils.get_response(200, task)
    except TaskQueueFullError as e:
        logger.warning(
            f"se rechaza la tarea porque la cola está llena, request_id: {request_id}, task_id: {task_id}"
        )
        raise HttpException(
            task_id=task_id, status_code=429, message=f"{request_id}: {str(e)}"
        )
    except ValueError as e:
        raise HttpException(
            task_id=task_id, status_code=400, message=f"{request_id}: {str(e)}"
        )

@router.get("/tasks", response_model=TaskListResponse, summary="Obtener todas las tareas")
def get_all_tasks(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=1000),
):
    tasks, total = sm.state.get_all_tasks(page, page_size)
    request_id = base.get_task_id(request)
    endpoint = config.app.get("endpoint", "").rstrip("/")
    task_dir = utils.task_dir()

    response = {
        "tasks": [
            _task_response_data(task, endpoint, task_dir, request_id) for task in tasks
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }
    return utils.get_response(200, response)



@router.get(
    "/tasks/{task_id}", response_model=TaskQueryResponse, summary="Consultar el estado de una tarea"
)
def get_task(
    request: Request,
    task_id: str = Path(..., description="ID de la tarea"),
    query: TaskQueryRequest = Depends(),
):
    request_id = base.get_task_id(request)
    endpoint = config.app.get("endpoint", "").rstrip("/")
    task = sm.state.get_task(task_id)
    if task:
        task_dir = utils.task_dir()
        response_task = _task_response_data(task, endpoint, task_dir, request_id)
        return utils.get_response(200, response_task)

    raise HttpException(
        task_id=task_id, status_code=404, message=f"{request_id}: tarea no encontrada"
    )


@router.delete(
    "/tasks/{task_id}",
    response_model=TaskDeletionResponse,
    summary="Borrar una tarea de vídeo corto generada",
)
def delete_video(request: Request, task_id: str = Path(..., description="ID de la tarea")):
    request_id = base.get_task_id(request)
    task = sm.state.get_task(task_id)
    if task:
        if tm.is_task_busy(task):
            logger.warning(
                f"se niega el borrado de una tarea ocupada, request_id: {request_id}, "
                f"task_id: {task_id}, state: {task.get('state')}, "
                f"cross_post_state: {task.get('cross_post_state')}"
            )
            raise HttpException(
                task_id=task_id,
                status_code=409,
                message=f"{request_id}: la tarea sigue en ejecución",
            )

        tasks_dir = utils.task_dir()
        current_task_dir = os.path.join(tasks_dir, task_id)
        if os.path.exists(current_task_dir):
            try:
                shutil.rmtree(current_task_dir)
            except FileNotFoundError:
                # Otro borrado de una tarea completada puede eliminar este directorio
                # después de nuestra comprobación de existencia. Solo se acepta una raíz
                # ausente; un borrado parcial y otros errores del sistema de archivos
                # deben conservar el estado.
                if os.path.lexists(current_task_dir):
                    raise

        sm.state.delete_task(task_id)
        logger.success(f"vídeo borrado: {utils.to_json(task)}")
        return utils.get_response(200)

    raise HttpException(
        task_id=task_id, status_code=404, message=f"{request_id}: tarea no encontrada"
    )


@router.get(
    "/musics", response_model=BgmRetrieveResponse, summary="Obtener los archivos de BGM locales"
)
def get_bgm_list(request: Request):
    bgm_list = []
    for file in bgm_service.list_bgm_files():
        filename = os.path.basename(file)
        try:
            size = os.path.getsize(file)
        except OSError as exc:
            logger.warning(f"se omite una música de fondo no disponible: name={filename}, error={exc}")
            continue
        bgm_list.append(
            {
                "name": filename,
                "size": size,
                # Solo se devuelve el nombre del archivo, para no exponer a quien llama
                # la ruta absoluta del servidor. El servidor la vuelve a resolver en los
                # dos directorios de la lista blanca, storage/bgm y resource/songs.
                "file": filename,
            }
        )
    response = {"files": bgm_list}
    return utils.get_response(200, response)


@router.post(
    "/musics",
    response_model=BgmUploadResponse,
    summary="Subir un archivo de música de fondo",
    description=(
        "Valida un archivo MP3, M4A, AAC, WAV, FLAC, OGG, OPUS o WMA de hasta "
        "30 MB y lo guarda con un nombre UUID inmutable en storage/bgm."
    ),
    responses={
        400: {"description": "El nombre, el formato, el tamaño o la pista de audio no son válidos"},
        500: {"description": "La validación con FFmpeg o el almacenamiento persistente no están disponibles"},
    },
)
def upload_bgm_file(request: Request, file: UploadFile = File(...)):
    request_id = base.get_task_id(request)
    try:
        safe_filename = bgm_service.save_bgm_upload(file.filename, file.file)
    except bgm_service.BgmUploadError as exc:
        # Un fallo de subida normalmente se puede resolver si el usuario cambia de
        # archivo, así que se registran el request_id y el motivo concreto, pero no el
        # contenido del archivo ni la ruta absoluta, para que los logs no filtren datos
        # del usuario.
        logger.warning(
            f"se rechaza la subida de música de fondo: request_id={request_id}, error={str(exc)}"
        )
        raise HttpException(
            task_id=request_id,
            status_code=400,
            message=f"{request_id}: {str(exc)}",
        )
    except bgm_service.BgmServiceError as exc:
        # Los fallos de la cadena de herramientas o del almacenamiento son problemas del
        # servidor y no pueden disfrazarse de error del archivo del usuario. El log
        # conserva el request_id y el motivo interno, y la respuesta HTTP solo devuelve
        # un texto estable, para no exponer rutas del servidor.
        logger.error(
            f"falló la subida de música de fondo: request_id={request_id}, error={str(exc)}"
        )
        raise HttpException(
            task_id=request_id,
            status_code=500,
            message=f"{request_id}: la validación de la música de fondo no está disponible",
        )

    response = {"file": safe_filename}
    return utils.get_response(200, response)

@router.get(
    "/video_materials", response_model=VideoMaterialRetrieveResponse, summary="Obtener el material de vídeo local"
)
def get_video_materials_list(request: Request):
    allowed_suffixes = material_upload_service.SUPPORTED_MATERIAL_EXTENSIONS
    local_videos_dir = utils.storage_dir("local_videos", create=True)
    video_materials_list = []
    with os.scandir(local_videos_dir) as entries:
        for entry in entries:
            if (
                entry.name.startswith(".")
                or pathlib.Path(entry.name).suffix.lower() not in allowed_suffixes
            ):
                continue
            try:
                # No se siguen enlaces fuera de local_videos ni se lista una subida que
                # desapareció mientras se analizaba el directorio.
                if not entry.is_file(follow_symlinks=False):
                    continue
                size = entry.stat(follow_symlinks=False).st_size
            except OSError as exc:
                logger.warning(
                    f"se omite un material local no disponible: name={entry.name}, error={exc}"
                )
                continue
            video_materials_list.append(
                {"name": entry.name, "size": size, "file": entry.name}
            )
    # Mantiene estable la selección ordenada de material entre sistemas de archivos y
    # ejecuciones.
    video_materials_list.sort(key=lambda item: (item["name"].casefold(), item["name"]))
    response = {"files": video_materials_list}
    return utils.get_response(200, response)


@router.post(
    "/video_materials",
    response_model=VideoMaterialUploadResponse,
    summary="Subir el archivo de material de vídeo al directorio de vídeos locales",
)
def upload_video_material_file(request: Request, file: UploadFile = File(...)):
    request_id = base.get_task_id(request)
    try:
        # Se siguen aceptando las rutas de cliente que manda el navegador, pero se
        # persiste una clave de almacenamiento UUID inmutable para que los nombres
        # repetidos no puedan sobrescribir entradas de tareas en cola.
        safe_filename = _sanitize_upload_filename(file.filename, request_id)
        stored_filename = material_upload_service.save_material_upload(
            safe_filename, file.file
        )
    except material_upload_service.MaterialUploadError as exc:
        logger.warning(
            f"se rechaza la subida de material local: request_id={request_id}, "
            f"error={str(exc)}"
        )
        raise HttpException(
            task_id=request_id,
            status_code=400,
            message=f"{request_id}: {str(exc)}",
        )
    except material_upload_service.MaterialServiceError as exc:
        logger.error(
            f"falló la subida de material local: request_id={request_id}, "
            f"error={str(exc)}"
        )
        raise HttpException(
            task_id=request_id,
            status_code=500,
            message=f"{request_id}: la validación del material local no está disponible",
        )

    response = {"file": stored_filename}
    return utils.get_response(200, response)

@router.get("/stream/{file_path:path}")
@router.head("/stream/{file_path:path}", summary="Inspeccionar los metadatos del flujo de vídeo")
async def stream_video(request: Request, file_path: str):
    request_id = base.get_task_id(request)
    tasks_dir = utils.task_dir()
    video_path = _resolve_path_within_directory(tasks_dir, file_path, request_id)
    range_header = request.headers.get("Range")
    if range_header and request.headers.get("If-Range") is not None:
        # Este stream no publica validadores ETag/Last-Modified fuertes. Un If-Range que
        # no se puede verificar debe volver a la representación completa, para que un
        # cliente no pueda combinar bytes antiguos de caché con el final de un vídeo
        # nuevo.
        range_header = None
    # El cuerpo se produce después de que vuelva este manejador. Se abre ya para que un
    # borrado o sustitución de la tarea entre las cabeceras y la iteración no pueda
    # hacer fallar el stream ni desajustar el tamaño usado en Content-Range.
    try:
        video_file = open(video_path, "rb")
    except FileNotFoundError as exc:
        raise HttpException(
            task_id=request_id,
            status_code=404,
            message=f"{request_id}: el archivo ya no existe",
        ) from exc
    try:
        video_size = os.fstat(video_file.fileno()).st_size
        if request.method == "HEAD":
            # Range solo se aplica a GET. Se cierra antes de devolver los metadatos; no
            # hace falta generador ni limpieza diferida en segundo plano para HEAD.
            video_file.close()
            return Response(
                media_type="video/mp4",
                headers={"Accept-Ranges": "bytes", "Content-Length": str(video_size)},
            )
        start, end = _parse_byte_range(range_header, video_size, request_id)
    except Exception:
        video_file.close()
        raise
    length = end - start + 1

    def file_iterator():
        try:
            video_file.seek(start, os.SEEK_SET)
            remaining = length
            while remaining > 0:
                data = video_file.read(min(4096, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data
        finally:
            video_file.close()

    response = StreamingResponse(
        file_iterator(),
        media_type="video/mp4",
        background=BackgroundTask(video_file.close),
    )
    response.headers["Accept-Ranges"] = "bytes"
    response.headers["Content-Length"] = str(length)
    if range_header:
        response.headers["Content-Range"] = f"bytes {start}-{end}/{video_size}"
        response.status_code = 206  # Contenido parcial

    return response


@router.get("/download/{file_path:path}")
@router.head("/download/{file_path:path}", summary="Inspeccionar los metadatos de descarga del artefacto")
async def download_video(request: Request, file_path: str):
    """
    descarga el vídeo :param request: solicitud Request :param file_path: ruta del
    archivo de vídeo, p. ej.: /cd1727ed-3473-42a2-a7da-4faafafec72b/final-1.mp4 :return:
    archivo de vídeo
    """
    request_id = base.get_task_id(request)
    tasks_dir = utils.task_dir()
    video_path = _resolve_path_within_directory(tasks_dir, file_path, request_id)
    file_path = pathlib.Path(video_path)
    filename = file_path.name
    media_type, _ = mimetypes.guess_type(filename)
    stat_result = None
    if request.method == "HEAD":
        try:
            stat_result = os.stat(video_path)
        except FileNotFoundError as exc:
            raise HttpException(
                task_id=request_id,
                status_code=404,
                message=f"{request_id}: el archivo ya no existe",
            ) from exc
    response = FileResponse(
        path=video_path,
        filename=filename,
        media_type=media_type or "application/octet-stream",
        stat_result=stat_result,
    )
    if request.method == "HEAD":
        # Reutiliza los metadatos de FileResponse sin leer su cuerpo ni aplicar a esta
        # solicitud HEAD el procesamiento de Range exclusivo de GET.
        return Response(headers=response.headers)
    return response
