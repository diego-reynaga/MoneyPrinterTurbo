import copy
import threading
from collections import deque

from loguru import logger

from app.config import config
from app.controllers.manager.memory_manager import InMemoryTaskManager
from app.models import const
from app.models.schema import VideoParams
from app.services import state as sm
from app.services import task as tm
from app.services.loomloom import LoomLoomConfirmedVideoRequest
from app.utils.logging_utils import (
    format_log_record,
    log_scope_thread_id,
    runtime_log_filter,
)


# La configuración de la WebUI se guarda en un diccionario global del proceso. La
# implementación síncrona original mantenía runtime_config_lock durante toda la
# generación, así que en la práctica las distintas sesiones del navegador también se
# ejecutaban en serie. Aquí la concurrencia se fija en 1, lo que mantiene la coherencia
# de configuración original y evita que varios hilos se limiten a esperar sin sentido
# fuera del bloqueo de configuración.
_task_manager = InMemoryTaskManager(
    max_concurrent_tasks=1,
    max_queued_tasks=max(1, int(config.app.get("max_queued_tasks", 100))),
)
_task_logs: dict[str, deque[str]] = {}
_task_logs_lock = threading.RLock()
_MAX_LOG_TASKS = 20
_MAX_LOG_RECORDS_PER_TASK = 1000
# Streamlit no permite que un hilo en segundo plano envíe directamente actualizaciones
# de componentes; solo se puede sondear con un Fragment. 0,5 segundos bastan para que el
# log de la WebUI se acerque a la salida en tiempo real del terminal, sin ocupar
# continuamente recursos del navegador como un refresco muy frecuente.
TASK_LOG_REFRESH_INTERVAL_SECONDS = 0.5


def _append_task_log(task_id: str, message: str) -> None:
    """Guarda una cantidad limitada de logs por tarea para que el Fragment de Streamlit los
    sondee de forma segura.
    """
    with _task_logs_lock:
        records = _task_logs.get(task_id)
        if records is None:
            # Solo conserva los logs de las tareas recientes, para que la WebUI no ocupe
            # cada vez más memoria tras mucho tiempo en marcha. dict mantiene el orden
            # de inserción; los logs de tareas solo sirven para diagnóstico en la
            # interfaz, así que descartar los más antiguos no afecta a las tareas.
            if len(_task_logs) >= _MAX_LOG_TASKS:
                oldest_task_id = next(iter(_task_logs))
                _task_logs.pop(oldest_task_id, None)
            records = deque(maxlen=_MAX_LOG_RECORDS_PER_TASK)
            _task_logs[task_id] = records
        records.append(message.rstrip())


def get_task_logs(task_id: str) -> list[str]:
    """Devuelve una instantánea de los logs, para no tener durante el renderizado de la
    página el bloqueo que usa el hilo en segundo plano.
    """
    with _task_logs_lock:
        return list(_task_logs.get(task_id, ()))


def _run_generation(
    task_id: str,
    params: VideoParams,
    capture_logs: bool,
    voice_preview: dict | None = None,
    loomloom_video_request: LoomLoomConfirmedVideoRequest | None = None,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
) -> dict:
    """
    Ejecuta el flujo de vídeo existente en un hilo en segundo plano.

    El sink de Loguru es un recurso del proceso, así que hay que filtrar por el hilo de
    trabajo actual. Si no, se mezclarían en la tarea actual los logs de tareas de la API
    o de otras páginas que se ejecuten a la vez. Los hilos auxiliares que la tarea
    arranca para descargas, codificación de clips y latidos se asignan al hilo de
    trabajo mediante bind_log_scope, y sus logs también deben recogerse; si no, estas
    fases largas no mostrarían nada en la WebUI. La página solo lee una instantánea de
    lista normal y nunca accede a session_state de Streamlit desde el hilo en segundo
    plano, lo que evita de raíz el desorden de rutas delta al refrescar.
    """
    log_handler_id = None
    worker_thread_id = threading.get_ident()
    try:
        if capture_logs:
            log_handler_id = logger.add(
                lambda message: _append_task_log(task_id, str(message)),
                level=0,
                format=format_log_record,
                colorize=False,
                filter=lambda record: (
                    log_scope_thread_id(record["thread"].id) == worker_thread_id
                    and runtime_log_filter(record)
                ),
            )

        # La tarea completa sigue usando el bloqueo de configuración original, para que
        # otra sesión de la WebUI no cambie a mitad de la generación la configuración
        # del proceso (proveedor, claves, etc.) y un mismo vídeo use ajustes distintos
        # al principio y al final.
        with config.runtime_config_lock():
            return tm.start(
                task_id=task_id,
                params=params,
                voice_preview=voice_preview,
                loomloom_video_request=loomloom_video_request,
                voxcpm_reference_audio=voxcpm_reference_audio,
                voxcpm_prompt_audio=voxcpm_prompt_audio,
                voxcpm_prompt_text=voxcpm_prompt_text,
            )
    except Exception as exc:
        # tm.start ya convierte las excepciones del flujo en estado de fallo; aquí se
        # protege además la capa envolvente de la WebUI (sink de logs, bloqueo de
        # configuración, etc.). Cualquier excepción del hilo en segundo plano debe dejar
        # un estado final; el gestor de tareas no puede seguir mostrando "generando"
        # para siempre tras salir el hilo de trabajo.
        error = f"{type(exc).__name__}: {exc}"
        failure = {
            "task_id": task_id,
            "state": const.TASK_STATE_FAILED,
            "progress": 0,
            "failed_stage": "webui_worker",
            "error": error,
        }
        sm.state.update_task(
            task_id,
            state=failure["state"],
            progress=failure["progress"],
            failed_stage=failure["failed_stage"],
            error=failure["error"],
        )
        logger.exception(
            f"fallo inesperado del worker de generación de la WebUI, "
            f"task_id={task_id}, error={exc}"
        )
        return failure
    finally:
        if log_handler_id is not None:
            try:
                logger.remove(log_handler_id)
            except ValueError:
                logger.debug(
                    f"el manejador de logs de la tarea de la WebUI ya se había eliminado: task_id={task_id}"
                )


def submit_generation(
    task_id: str,
    params: VideoParams,
    capture_logs: bool = True,
    voice_preview: dict | None = None,
    loomloom_video_request: LoomLoomConfirmedVideoRequest | None = None,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
) -> None:
    """
    Registra y envía la tarea de generación de vídeo de la WebUI, y vuelve de inmediato.

    El estado de la tarea debe escribirse antes de arrancar el hilo. Así, al terminar
    esta ejecución del script de la página ya se puede consultar la tarea, y un refresco
    del navegador o una reconexión del WebSocket no dependen de marcadores en la memoria
    de la página antigua.
    """
    task_params = params.model_copy(deep=True)
    # La línea de tiempo de subtítulos y los parámetros anidados de vista previa siguen
    # siendo mutables en el estado de sesión de Streamlit. Se toma una instantánea antes
    # de que este trabajo entre en la cola.
    voice_preview_snapshot = copy.deepcopy(voice_preview) if voice_preview else None
    # El audio de referencia pertenece solo a esta solicitud encolada. Está
    # deliberadamente separado de VideoParams para que no pueda llegar al historial de
    # tareas, a los preajustes, a la persistencia del estado ni a los logs. ``bytes`` es
    # inmutable; se hace una instantánea explícita antes de que arranque el worker en
    # segundo plano para que los reruns posteriores de la WebUI no compartan estado de
    # subida mutable con esta tarea.
    voxcpm_reference_audio_snapshot = (
        bytes(voxcpm_reference_audio) if voxcpm_reference_audio else None
    )
    voxcpm_prompt_audio_snapshot = (
        bytes(voxcpm_prompt_audio) if voxcpm_prompt_audio else None
    )
    voxcpm_prompt_text_snapshot = str(voxcpm_prompt_text or "")
    # La solicitud confirmada es un objeto de datos congelado que solo se pasa dentro
    # del proceso actual. La clave API no entra en VideoParams, en el estado de la
    # tarea, en los logs ni en el historial guardado en disco, y no le afectan los
    # reruns posteriores de la página.
    loomloom_request_snapshot = loomloom_video_request
    sm.state.update_task(
        task_id,
        state=const.TASK_STATE_PROCESSING,
        progress=0,
        video_subject=task_params.video_subject or task_params.video_script or task_id,
    )
    try:
        _task_manager.add_task(
            _run_generation,
            task_id=task_id,
            params=task_params,
            capture_logs=capture_logs,
            voice_preview=voice_preview_snapshot,
            loomloom_video_request=loomloom_request_snapshot,
            voxcpm_reference_audio=voxcpm_reference_audio_snapshot,
            voxcpm_prompt_audio=voxcpm_prompt_audio_snapshot,
            voxcpm_prompt_text=voxcpm_prompt_text_snapshot,
        )
    except Exception as exc:
        # Un fallo de planificación, igual que un fallo del flujo, debe convertirse en
        # un estado consultable, para que el gestor de tareas no muestre "generando"
        # para siempre. Se conserva el tipo de excepción para localizar rápidamente
        # problemas de la cola en los logs de Docker o del equipo.
        error = f"{type(exc).__name__}: {exc}"
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_FAILED,
            progress=0,
            failed_stage="scheduling",
            error=error,
        )
        logger.exception(
            f"no se pudo enviar la tarea de generación de la WebUI, task_id={task_id}, error={exc}"
        )
        raise
