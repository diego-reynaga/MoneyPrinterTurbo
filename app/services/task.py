import json
import math
import os
import re
import socket
import tempfile
import threading
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from functools import partial
from os import path
from uuid import uuid4

from loguru import logger

from app.config import config
from app.models import const
from app.models.schema import VideoConcatMode, VideoParams
from app.services import bgm as bgm_service
from app.services import (
    elevenlabs_music,
    infistar_video,
    llm,
    loomloom,
    material,
    metaso_minimax,
    muapi,
    ofox,
    sonilo,
    subtitle,
    task_artifacts,
    twelvelabs,
    video,
    volcengine_seedance,
    voice,
)
from app.services import upload_post
from app.services import state as sm
from app.utils import file_security, utils
from app.utils.subtitle_writer import staged_subtitle_file


# Una solicitud de publicación puede esperar varios minutos y no puede seguir ocupando
# una plaza de concurrencia de las tareas de generación de vídeo. El pool de hilos de
# tamaño fijo limita el rendimiento de publicación a un rango controlable, y a la vez
# permite que el vídeo pase al estado completado en cuanto se genera.
_cross_post_executor = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="mpt-cross-post",
)
_cross_post_max_pending_tasks = max(
    1,
    int(config.app.get("upload_post_max_pending_tasks", 10)),
)
_cross_post_slots = threading.BoundedSemaphore(_cross_post_max_pending_tasks)
_cross_post_registry_lock = threading.RLock()
_cross_post_futures: dict[str, Future] = {}
_cross_post_process_owner = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex}"
_ACTIVE_CROSS_POST_STATES = {
    const.CROSS_POST_STATE_PENDING,
    const.CROSS_POST_STATE_PROCESSING,
}
_CROSS_POST_STATE_WRITE_ATTEMPTS = 3
_CROSS_POST_STATE_RETRY_DELAY_SECONDS = 0.1
_TASK_FAILURE_STATE_WRITE_ATTEMPTS = 3
_TASK_FAILURE_STATE_RETRY_DELAY_SECONDS = 0.1
_LOOMLOOM_STATE_WRITE_ATTEMPTS = 3
_LOOMLOOM_STATE_RETRY_DELAY_SECONDS = 0.1
_INTERRUPTED_CROSS_POST_ERROR = (
    "la publicación multiplataforma se interrumpió antes de que terminara el proceso"
)
# Asigna los ids de plataforma de upload-post a los nombres de plataforma social que
# acepta llm.py.
_CROSS_POST_SOCIAL_PLATFORMS = {
    "tiktok": "tiktok",
    "instagram": "instagram_reels",
    "facebook": "facebook_reels",
}
# Un servicio de música para el vídeo solo tiene que implementar ``is_enabled`` y
# ``generate_bgm``. Las diferencias entre proveedores se concentran en la extensión del
# archivo, las excepciones de dominio y los códigos de aviso de la WebUI; la
# orquestación de tareas, el cortocircuito con volumen 0 y la degradación ante fallos
# reutilizan siempre la misma ruta, para no mantener varios flujos parecidos al añadir
# proveedores.
_VIDEO_MUSIC_PROVIDERS = {
    "sonilo": {
        "service": sonilo,
        "error_type": sonilo.SoniloError,
        "suffix": ".m4a",
        "warning_code": "sonilo_bgm_failed",
        "display_name": "Sonilo",
    },
    "elevenlabs": {
        "service": elevenlabs_music,
        "error_type": elevenlabs_music.ElevenLabsMusicError,
        "suffix": ".mp3",
        "warning_code": "elevenlabs_bgm_failed",
        "display_name": "ElevenLabs",
    },
}


_SUPPORTED_VIDEO_SOURCES = frozenset({
    "pexels", "pixabay", "coverr", "local", "wavespeed",
    "volcengine_seedance", "ofox", "metaso_minimax", "muapi",
    "loomloom", "openai_image", "infistar",
})


def _get_video_music_prompt(params: VideoParams) -> str:
    """
    Lee el prompt que usa realmente el proveedor de música actual.

    Las tareas nuevas usan siempre el campo independiente del proveedor; los parámetros
    antiguos de la CLI de Sonilo y las tareas históricas pueden tener solo
    ``sonilo_bgm_prompt``, así que el campo antiguo solo se lee cuando el campo común
    está vacío para Sonilo.
    """
    prompt = str(params.video_music_prompt or "").strip()
    if params.bgm_type == "sonilo" and not prompt:
        prompt = str(params.sonilo_bgm_prompt or "").strip()
    return prompt


def is_task_busy(task: dict | None) -> bool:
    """Decide si la tarea sigue generándose o publicándose, para reutilizarlo en todas las
    entradas de borrado.
    """
    if not task:
        return False

    state = task.get("state")
    try:
        state = int(state)
    except (TypeError, ValueError):
        # La propia docstring de la función promete tratar los estados que no se pueden
        # analizar como ocupados, para que la API de borrado conserve una tarea sobre la
        # que no podemos razonar. Devolver False aquí permitiría que delete_video()
        # cancelara una tarea que sigue ejecutándose o publicándose cada vez que una
        # instantánea antigua de Redis o un punto de llamada heredado guardara el estado
        # como cadena.
        return True

    # Tanto la generación de vídeo como la publicación multiplataforma pueden seguir
    # leyendo el directorio de la tarea. Tratarlas siempre como ocupadas evita que, si
    # la API y la WebUI mantuvieran cada una sus reglas, una permitiera borrar y la otra
    # no.
    return (
        state == const.TASK_STATE_PROCESSING
        or task.get("cross_post_state") in _ACTIVE_CROSS_POST_STATES
    )


def _register_cross_post_future(task_id: str, future: Future) -> None:
    """Registra el Future de publicación que tiene el proceso actual, para que la
    recuperación al arrancar y los tests puedan saber el estado real de ejecución.
    """
    with _cross_post_registry_lock:
        _cross_post_futures[task_id] = future


def _unregister_cross_post_future(task_id: str, future: Future | None = None) -> None:
    """Solo elimina el Future que coincide, para que un callback antiguo no borre por error
    el trabajo nuevo registrado después para la misma tarea.
    """
    with _cross_post_registry_lock:
        current = _cross_post_futures.get(task_id)
        if current is None or (future is not None and current is not future):
            return
        _cross_post_futures.pop(task_id, None)


def _is_cross_post_active_in_process(task_id: str) -> bool:
    """Decide si el proceso actual sigue teniendo tareas de publicación sin terminar."""
    with _cross_post_registry_lock:
        future = _cross_post_futures.get(task_id)
        return future is not None and not future.done()


def _is_windows_process_alive(process_id: int) -> bool:
    """Comprueba el estado del proceso con una API Win32 de solo lectura, para no terminar
    por error el proceso con os.kill.
    """
    import ctypes

    process_query_limited_information = 0x1000
    still_active = 259
    error_access_denied = 5
    error_invalid_parameter = 87
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # ctypes trata por defecto los valores de retorno no declarados como int de 32 bits.
    # Por eso un handle de proceso de 64 bits en Windows podría truncarse, así que hay
    # que declarar explícitamente la firma de la función Win32 antes de llamarla.
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(
        process_query_limited_information,
        False,
        process_id,
    )
    if not handle:
        error_code = ctypes.get_last_error()
        if error_code == error_invalid_parameter:
            return False
        if error_code == error_access_denied:
            # Si el proceso existe pero el usuario actual no tiene permiso para
            # consultarlo, debe considerarse vivo de forma conservadora, para no
            # recuperar por error tareas de publicación que está ejecutando otra cuenta.
            return True
        logger.warning(
            "no se pudo abrir en Windows el proceso propietario de la publicación multiplataforma, "
            f"process_id: {process_id}, error_code: {error_code}"
        )
        return True

    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            error_code = ctypes.get_last_error()
            logger.warning(
                "no se pudo leer en Windows el estado del proceso propietario de la publicación multiplataforma, "
                f"process_id: {process_id}, error_code: {error_code}"
            )
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _is_cross_post_owner_alive(owner: str | None) -> bool:
    """Decide si sigue existiendo el proceso local de una tarea de publicación persistida."""
    if not owner:
        return False

    try:
        hostname, process_id_text, _ = owner.split(":", 2)
        process_id = int(process_id_text)
    except (TypeError, ValueError):
        logger.warning(f"metadatos del propietario de la publicación multiplataforma no válidos: {owner}")
        return False

    # No se pueden detectar de forma fiable los procesos de otros hosts. En despliegues
    # de varios hosts que comparten Redis, debe considerarse de forma conservadora que
    # siguen en marcha, para que este nodo no borre por error archivos de vídeo que está
    # leyendo otro nodo.
    if hostname != socket.gethostname():
        return True

    # Si en el proceso actual sigue habiendo trabajo de publicación real ya lo determina
    # con exactitud el registro de Futures. Llegar aquí significa que el registro no
    # tiene el Future correspondiente, así que aunque el owner coincida totalmente con
    # el proceso actual debe considerarse interrumpido; esto cubre el caso en que la
    # escritura del estado final sigue fallando y el Future ya ha terminado.
    if process_id == os.getpid():
        return False

    # os.kill(pid, 0) en Windows no tiene la misma semántica que en POSIX y puede
    # terminar directamente el proceso de destino. Se usa una API Win32 que solo pide
    # permiso de consulta, sin enviar ninguna señal al proceso de destino.
    if os.name == "nt":
        return _is_windows_process_alive(process_id)

    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        logger.warning(
            f"no se pudo inspeccionar el proceso propietario de la publicación multiplataforma, owner: {owner}, error: {exc}"
        )
        return True
    return True


def _mark_task_failed(
    task_id: str,
    stage: str,
    error: str,
    details: dict | None = None,
) -> dict:
    """Registra la información de fallo estructurada, conservando el progreso que había
    alcanzado la tarea antes de fallar.
    """
    existing_task = None
    try:
        existing_task = sm.state.get_task(task_id)
    except Exception as exc:
        logger.warning(f"no se pudo leer el estado de la tarea antes de registrar el fallo: {exc}")

    # La función de servicio concreta suele tener un motivo de error más preciso que la
    # capa de orquestación. Las comprobaciones posteriores de resultado vacío no pueden
    # sobrescribirlo con un texto genérico, o quien llama a la API solo vería
    # información vaga.
    if (
        existing_task
        and existing_task.get("state") == const.TASK_STATE_FAILED
        and existing_task.get("error")
    ):
        return existing_task

    message = str(error or "error de tarea desconocido").strip()
    progress = int((existing_task or {}).get("progress", 0) or 0)
    logger.error(f"la tarea falló, task_id: {task_id}, stage: {stage}, error: {message}")
    failure = {
        "task_id": task_id,
        "state": const.TASK_STATE_FAILED,
        "progress": progress,
        "failed_stage": stage,
        "error": message,
    }
    # Algunas tareas externas ya han creado un ID remoto que sirve para recuperar o
    # diagnosticar. El estado de fallo debe conservar esos campos no sensibles, pero sin
    # permitir que quien llama sobrescriba la estructura común de estado, progreso y
    # error. Conserva los resultados de lotes completados tanto en el estado persistido
    # como en el valor de retorno directo que usa la CLI. Estos campos se reinician
    # antes de un nuevo intento.
    failure_details = {
        key: existing_task[key]
        for key in ("videos", "combined_videos", "warnings", "failed_video_index")
        if existing_task and key in existing_task
    }
    failure_details.update({
        key: value for key, value in dict(details or {}).items() if key not in failure
    })
    failure.update(failure_details)
    fields = {key: value for key, value in failure.items() if key != "task_id"}
    for attempt in range(_TASK_FAILURE_STATE_WRITE_ATTEMPTS):
        try:
            if attempt == 0:
                sm.state.update_task(task_id, **fields)
            else:
                # Un acuse perdido puede significar que la primera escritura tuvo éxito.
                # Solo se parchea un registro existente para que los reintentos no
                # puedan revivir una tarea que el usuario ha borrado después.
                sm.state.patch_task(task_id, **fields)
            break
        except Exception as exc:
            if attempt + 1 == _TASK_FAILURE_STATE_WRITE_ATTEMPTS:
                logger.exception(
                    f"no se pudo guardar el fallo de la tarea, task_id: {task_id}, error: {exc}"
                )
            else:
                time.sleep(_TASK_FAILURE_STATE_RETRY_DELAY_SECONDS)
    # Una caída del estado no debe descartar la única instantánea de archivos
    # completados que tiene quien llama. Se devuelve incluso cuando fallan todos los
    # intentos de persistencia acotados.
    return failure


def _stage_progress_reporter(task_id: str, stage_start: float, stage_end: float):
    """
    Devuelve un callback que convierte la proporción de avance 0~1 de una etapa en
    progreso de la tarea y lo escribe en el estado.

    Antes el flujo solo actualizaba el progreso en los límites de las etapas, así que
    durante fases largas como la descarga de material o el procesamiento de clips la
    barra no se movía en absoluto, y el usuario no podía distinguir entre que siguiera
    en marcha o que se hubiera colgado. El progreso es solo información para mostrar: si
    la proporción no se puede analizar se ignora, y si falla la escritura en el backend
    de estado solo se registra en el log, sin interrumpir por ello la descarga ni la
    composición.
    """

    def report(fraction) -> None:
        try:
            fraction = max(0.0, min(1.0, float(fraction)))
        except (TypeError, ValueError):
            return
        try:
            sm.state.update_task(
                task_id,
                state=const.TASK_STATE_PROCESSING,
                progress=stage_start + (stage_end - stage_start) * fraction,
            )
        except Exception as exc:
            logger.warning(
                f"no se pudo actualizar el progreso de la tarea: task_id={task_id}, "
                f"error={type(exc).__name__}, detail={exc}"
            )

    return report


def generate_script(task_id, params):
    logger.info("\n\n## generando el guion del vídeo")
    video_script = params.video_script.strip()
    if not video_script:
        video_script = llm.generate_script(
            video_subject=params.video_subject,
            language=params.video_language,
            paragraph_number=params.paragraph_number,
            video_script_prompt=params.video_script_prompt,
            custom_system_prompt=params.custom_system_prompt,
        )
    else:
        logger.debug(f"guion del vídeo: \n{video_script}")

    if not video_script:
        _mark_task_failed(task_id, "script", "no se pudo generar el guion del vídeo")
        return None

    return video_script


def generate_terms(task_id, params, video_script):
    logger.info("\n\n## generando los términos del vídeo")
    video_terms = params.video_terms
    if not video_terms:
        # Con la correspondencia de material en el orden del guion activada, las propias
        # palabras clave también deben generarse en el orden narrativo del guion; si no,
        # aunque después se descargue y se monte en orden, solo se podría reutilizar un
        # conjunto global de temas, sin resolver el problema de que "las imágenes de lo
        # que viene después aparecen antes".
        video_terms = llm.generate_terms(
            video_subject=params.video_subject,
            video_script=utils.remove_pause_tags(video_script),
            amount=8 if params.match_materials_to_script else 5,
            match_script_order=params.match_materials_to_script,
        )
    else:
        if isinstance(video_terms, str):
            video_terms = [term.strip() for term in re.split(r"[,，]", video_terms)]
        elif isinstance(video_terms, list):
            video_terms = [term.strip() for term in video_terms]
        else:
            raise ValueError("video_terms debe ser una cadena o una lista de cadenas.")

        # Una entrada formada solo por delimitadores no debe llegar al reordenamiento de
        # pago ni a la búsqueda de material.
        video_terms = [term for term in video_terms if term]

        logger.debug(f"términos del vídeo: {utils.to_json(video_terms)}")

    if not video_terms:
        _mark_task_failed(
            task_id,
            "terms",
            "no se pudieron generar los términos de búsqueda del vídeo",
        )
        return None

    # Reordenamiento semántico opcional con TwelveLabs Marengo: si no está activado
    # devuelve el orden original, sin efectos secundarios. En el modo de correspondencia
    # ordenada, el orden de las palabras clave ya es el orden narrativo del guion y debe
    # mantenerse, así que se omite.
    if not params.match_materials_to_script:
        video_terms = twelvelabs.rerank_terms_by_subject(
            video_subject=params.video_subject,
            search_terms=video_terms,
        )

    return video_terms


def save_script_data(task_id, video_script, video_terms, params):
    script_data = {
        "script": video_script,
        "search_terms": video_terms,
        "params": params,
    }
    task_artifacts.write_script_data(task_id, script_data)


def resolve_custom_audio_file(
    task_id: str,
    custom_audio_file: str | None,
    *,
    allow_server_file_input: bool = False,
) -> str:
    requested_file = (custom_audio_file or "").strip()
    if not requested_file:
        return ""

    task_dir = utils.task_dir(task_id)
    try:
        return file_security.resolve_path_within_directory(
            task_dir,
            requested_file,
        )
    except ValueError as exc:
        task_dir_error = exc

    # Una ruta que falta pero que por lo demás queda dentro del directorio de la tarea
    # se puede informar con precisión. Las rutas fuera de ese límite usan el mismo error
    # genérico existan o no, para que quien llama no pueda sondear el sistema de
    # archivos del host.
    if str(task_dir_error) == "el archivo no existe":
        raise task_dir_error

    # Las solicitudes HTTP y demás llamadores no fiables nunca deben convertir una ruta
    # enviada en una lectura de archivos en el servidor. Las subidas de la WebUI ya
    # viven en el directorio de la tarea; solo la CLI local opta explícitamente por
    # resolver archivos en otros lugares del host.
    if not allow_server_file_input:
        raise ValueError(
            "el archivo de audio personalizado debe estar dentro del directorio de la tarea actual"
        ) from task_dir_error

    server_audio_file = path.realpath(
        requested_file
        if path.isabs(requested_file)
        else path.join(utils.root_dir(), requested_file)
    )
    if not path.isabs(requested_file):
        project_root = path.realpath(utils.root_dir())
        try:
            if path.commonpath([project_root, server_audio_file]) != project_root:
                raise ValueError(
                    "las rutas relativas de audio personalizado deben quedar dentro del directorio del proyecto"
                )
        except ValueError as exc:
            raise ValueError(
                "el archivo de audio personalizado debe ser local de la tarea o un archivo existente del servidor"
            ) from exc

    if not path.isfile(server_audio_file):
        raise ValueError(
            "el archivo de audio personalizado no existe o no es un archivo"
        ) from task_dir_error

    return server_audio_file


def _resolve_reusable_voice_preview(
    task_id: str,
    params,
    video_script: str,
    voice_preview: dict | None,
) -> tuple[str, float, object] | None:
    """
    Valida y analiza la caché de vista previa completa enviada por la WebUI.

    Esta carga útil no es un parámetro público de la API y solo puede venir de la WebUI
    del proceso actual. Aun así, la tarea en segundo plano vuelve a comprobar el texto y
    todos los parámetros de locución, y exige que el audio esté en el directorio de la
    tarea actual; ante cualquier diferencia vuelve al TTS normal, para que una vista
    previa caducada no contamine el vídeo final definitivo.
    """
    if not voice_preview:
        return None

    expected_values = {
        "script": str(video_script or "").strip(),
        "voice_name": params.voice_name,
        "voice_rate": float(params.voice_rate),
        "voice_volume": float(params.voice_volume),
    }
    if not math.isclose(float(params.voice_volume), 1.0) or any(
        voice_preview.get(key) != value for key, value in expected_values.items()
    ):
        logger.info(
            f"se omite una caché de vista previa de voz obsoleta, task_id: {task_id}, "
            "motivo: cambiaron los parámetros de la voz"
        )
        return None

    preview_file = path.realpath(str(voice_preview.get("audio_file") or ""))
    task_root = path.realpath(utils.task_dir(task_id))
    try:
        preview_is_task_local = path.commonpath([task_root, preview_file]) == task_root
    except ValueError:
        preview_is_task_local = False

    duration = voice_preview.get("duration")
    sub_maker = voice_preview.get("sub_maker")
    if (
        not preview_is_task_local
        or not path.isfile(preview_file)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
        or sub_maker is None
    ):
        logger.warning(
            f"se omite una caché de vista previa de voz no válida, task_id: {task_id}, "
            f"audio_file: {preview_file or '<empty>'}"
        )
        return None

    logger.info(
        f"se usa el audio de la vista previa completa de la voz, task_id: {task_id}, duration: {duration:.2f}s"
    )
    return preview_file, math.ceil(duration), sub_maker


def generate_audio(
    task_id,
    params,
    video_script,
    voice_preview=None,
    *,
    allow_server_file_input: bool = False,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
):
    """
    Genera el audio para el guion del vídeo. Si se proporciona un archivo de audio
    personalizado, se usa directamente. En ese caso no se devuelve ningún objeto
    generador de subtítulos. Si no, se usa el TTS para generar el audio.
    Returns:
        - audio_file: ruta del archivo de audio generado o proporcionado
        - audio_duration: duración del audio en segundos
        - sub_maker: objeto generador de subtítulos si se usa el TTS; None en otro caso
    """
    logger.info("\n\n## generando el audio")
    # Los modelos de solicitud de /audio y /subtitle no incluyen custom_audio_file, así
    # que aquí se lee de forma compatible, para no lanzar un error de atributo al llamar
    # directamente a la interfaz.
    requested_custom_audio_file = getattr(params, "custom_audio_file", None)
    try:
        custom_audio_file = resolve_custom_audio_file(
            task_id,
            requested_custom_audio_file,
            allow_server_file_input=allow_server_file_input,
        )
    except ValueError as exc:
        _mark_task_failed(
            task_id,
            "audio",
            f"archivo de audio personalizado no válido: {exc}",
        )
        return None, None, None

    if not custom_audio_file:
        reusable_preview = _resolve_reusable_voice_preview(
            task_id,
            params,
            video_script,
            voice_preview,
        )
        if reusable_preview:
            return reusable_preview

        logger.info("no se proporcionó un archivo de audio personalizado; se genera el audio con TTS.")
        audio_file = path.join(utils.task_dir(task_id), "audio.mp3")
        tts_kwargs = {
            "text": video_script,
            "voice_name": voice.parse_voice_name(params.voice_name),
            "voice_rate": params.voice_rate,
            "voice_file": audio_file,
        }
        if voxcpm_reference_audio is not None:
            tts_kwargs["voxcpm_reference_audio"] = voxcpm_reference_audio
        if voxcpm_prompt_audio is not None:
            tts_kwargs["voxcpm_prompt_audio"] = voxcpm_prompt_audio
            tts_kwargs["voxcpm_prompt_text"] = voxcpm_prompt_text
        sub_maker = voice.tts(**tts_kwargs)
        if sub_maker is None:
            _mark_task_failed(
                task_id,
                "audio",
                "no se pudo sintetizar el audio; verifica la voz elegida y la conectividad del TTS",
            )
            return None, None, None
        # Se mide el audio_file escrito de verdad, no sub_maker.cues[-1].end:
        # este último es el último LÍMITE DE PALABRA, y el TTS deja una cola fija
        # después (Edge TTS: ~0,88 s con cualquier longitud, el 19 % de un clip de 7
        # palabras pero el 1,4 % de uno de 153, así que los guiones cortos son los más
        # perjudicados). La infravaloración dimensiona las llamadas de pago a
        # generate_bgm(), se informa como audio_duration a la API/WebUI y se queda corta
        # de material en download_videos(), multiplicado por video_count.
        file_duration = voice.get_audio_duration(audio_file)
        audio_duration = math.ceil(
            file_duration if file_duration > 0 else voice.get_audio_duration(sub_maker)
        )
        if audio_duration == 0:
            _mark_task_failed(task_id, "audio", "la duración del audio generado es cero")
            return None, None, None
        return audio_file, audio_duration, sub_maker
    else:
        logger.info(f"se usa el archivo de audio personalizado: {custom_audio_file}")
        audio_duration = voice.get_audio_duration(custom_audio_file)
        if audio_duration == 0:
            _mark_task_failed(
                task_id,
                "audio",
                "la duración del audio personalizado es cero",
            )
            return None, None, None
        return custom_audio_file, audio_duration, None


def generate_subtitle(task_id, params, video_script, sub_maker, audio_file):
    """
    Genera los subtítulos para el guion del vídeo. Si la generación de subtítulos está
    desactivada o no se proporciona generador de subtítulos, devuelve una cadena vacía.
    Si no, genera los subtítulos con el proveedor indicado.
    Returns:
        - subtitle_path: ruta del archivo de subtítulos generado
    """
    logger.info("\n\n## generando los subtítulos")
    if not params.subtitle_enabled:
        return ""

    final_subtitle_path = path.join(utils.task_dir(task_id), "subtitle.srt")
    subtitle_provider = config.app.get("subtitle_provider", "edge").strip().lower()
    logger.info(f"\n\n## generando los subtítulos, proveedor: {subtitle_provider}")

    if not subtitle_provider:
        logger.info("el proveedor de subtítulos está vacío; se omite la generación de subtítulos")
        return ""

    if sub_maker is None and subtitle_provider != "whisper":
        # El audio personalizado no pasa por el TTS, así que no tiene la línea de tiempo
        # sub_maker que devuelven los TTS como Edge/Azure. Solo Whisper puede
        # transcribir los subtítulos directamente desde el archivo de audio; los demás
        # proveedores de subtítulos mantienen el comportamiento original, para no
        # generar líneas de tiempo vacías erróneas.
        logger.warning(
            "falta el generador de subtítulos; se omite la generación de subtítulos para el proveedor: "
            f"{subtitle_provider}"
        )
        return ""

    is_word_level = getattr(params, "subtitle_display_mode", "sentence") == "word_by_word"

    # Un reintento fallido nunca debe reutilizar los subtítulos de una narración
    # anterior.
    with staged_subtitle_file(final_subtitle_path) as subtitle_path:
        if subtitle_provider == "edge":
            voice.create_subtitle(
                text=video_script,
                sub_maker=sub_maker,
                subtitle_file=subtitle_path,
                word_level=is_word_level,
            )
            if not os.path.exists(subtitle_path):
                # Los subtítulos de Edge a veces no producen archivo porque la línea de
                # tiempo no se puede hacer coincidir con el texto. Aquí no se puede
                # pasar automáticamente a Whisper, o el primer fallo descargaría un
                # modelo de varios GB sin que el usuario lo sepa. Solo se permite cargar
                # el modelo cuando Whisper está configurado explícitamente; si Edge
                # falla se conserva el vídeo sin subtítulos y se registra el motivo,
                # evitando gastos inesperados de red y de disco.
                logger.warning(
                    "la generación de subtítulos de edge no produjo un archivo de subtítulos; "
                    "se omiten los subtítulos sin volver a whisper"
                )
                return ""

        if subtitle_provider == "whisper":
            subtitle.create(
                audio_file=audio_file,
                subtitle_file=subtitle_path,
                word_level=is_word_level,
            )
            if not subtitle.file_to_subtitles(subtitle_path):
                logger.warning("whisper no produjo subtítulos utilizables")
                return ""
            if not is_word_level:
                logger.info("\n\n## corrigiendo los subtítulos")
                subtitle.correct(subtitle_file=subtitle_path, video_script=video_script)

        subtitle_lines = subtitle.file_to_subtitles(subtitle_path)
        if not subtitle_lines:
            logger.warning(f"el archivo de subtítulos no es válido: {subtitle_path}")
            return ""

        os.replace(subtitle_path, final_subtitle_path)
        return final_subtitle_path


def get_video_materials(
    task_id,
    params,
    video_terms,
    audio_duration,
    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,
):
    if params.video_source == "local":
        logger.info("\n\n## preprocesando el material local")
        materials = video.preprocess_video(
            materials=params.video_materials,
            clip_duration=params.video_clip_duration,
            image_motion=params.image_motion,
        )
        if not materials:
            _mark_task_failed(
                task_id,
                "materials",
                "no se encontró material de vídeo local válido",
            )
            return None
        return [material_info.url for material_info in materials]
    elif params.video_source == "loomloom":
        if not isinstance(
            loomloom_video_request, loomloom.LoomLoomConfirmedVideoRequest
        ):
            _mark_task_failed(
                task_id,
                "materials",
                "la generación de vídeo con LoomLoom requiere un presupuesto confirmado",
            )
            return None

        request = loomloom_video_request
        logger.info(
            "\n\n## generando "
            f"{len(request.batch.input_rows)} materiales de vídeo con LoomLoom"
        )
        run_id = ""
        try:
            request.validate()
            backend = loomloom.LoomLoomVideoBackend(request.settings)
            execution = backend.execute(
                request.batch,
                client_request_id=request.client_request_id,
                listing_version_id=request.listing_version_id,
                confirm=True,
            )
            run_id = execution.run_id
            # Que execute vuelva significa que el servidor ya ha aceptado la tarea de
            # pago. Hay que escribir primero el run ID en el log del proceso, para que
            # aunque después el backend de estado (Redis, etc.) no esté disponible, el
            # personal de operaciones pueda localizar la tarea en Shengsuan Cloud con el
            # log; el único identificador no puede quedarse solo en una variable local.
            logger.info(
                "ejecución de vídeo de pago de LoomLoom creada: "
                f"task_id={task_id}, run_id={run_id}, "
                f"listing_version_id={request.listing_version_id}"
            )
            # En cuanto se crea una tarea de pago se registra de inmediato el ID remoto.
            # Aunque más tarde el sondeo agote el tiempo, el log y el estado de la tarea
            # siguen ayudando al usuario o al soporte de la plataforma a localizar y
            # recuperar el producto ya generado. Un fallo del backend de estado solo
            # puede reducir la observabilidad; no puede interrumpir una tarea remota que
            # ya ha empezado a facturarse ni la descarga del producto.
            _record_loomloom_run_reference(
                task_id=task_id,
                run_id=run_id,
                listing_version_id=request.listing_version_id,
            )
            backend.wait_for_run(run_id)
            return list(
                backend.download_video_results(
                    run_id,
                    utils.task_dir(task_id),
                )
            )
        except (loomloom.LoomLoomError, ValueError) as exc:
            _mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details={
                    "loomloom_run_id": run_id,
                    "loomloom_listing_version_id": request.listing_version_id,
                },
            )
            return None
    else:
        logger.info(f"\n\n## descargando vídeos de {params.video_source}")
        # El modo de correspondencia ordenada solo tiene efecto cuando el usuario lo
        # activa explícitamente. Aquí se fuerza que la descarga de material vaya rotando
        # por las palabras clave en orden, para que una palabra clave temprana no
        # descargue demasiado material y saque de la línea de tiempo final los temas
        # posteriores del guion.
        try:
            downloaded_videos = material.download_videos(
                task_id=task_id,
                search_terms=video_terms,
                source=params.video_source,
                video_aspect=params.video_aspect,
                video_concat_mode=(
                    VideoConcatMode.sequential
                    if params.match_materials_to_script
                    else params.video_concat_mode
                ),
                audio_duration=audio_duration * params.video_count,
                max_clip_duration=params.video_clip_duration,
                match_script_order=params.match_materials_to_script,
                image_motion=params.image_motion,
                # La fase de material ocupa del 40 % al 50 %: avanza cada vez que se
                # descarga un archivo, para que con redes lentas la barra de progreso no
                # se quede mucho tiempo en el 40 %.
                progress_callback=_stage_progress_reporter(task_id, 40, 50),
            )
        except volcengine_seedance.VolcEngineSeedanceError as exc:
            # Tanto el estado sin confirmar como el generado pero con descarga fallida
            # corresponden a una tarea remota que se puede recuperar en la consola de
            # Ark. El estado de fallo se escribe siempre a partir del task_id que lleva
            # la excepción, para que cada rama de excepción no mantenga su propia
            # información de recuperación y se olvide otra vez en futuras ampliaciones.
            remote_task_id = str(getattr(exc, "task_id", "") or "").strip()
            details = (
                {"volcengine_seedance_task_id": remote_task_id}
                if remote_task_id
                else None
            )
            _mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details=details,
            )
            return None
        except (
            material.WaveSpeedUnconfirmedTaskError,
            material.WaveSpeedDownloadError,
        ) as exc:
            # Una vez que existe una predicción de pago, un fallo de sondeo o de
            # descarga debe dejar su ID en el estado de la tarea. Los clips anteriores
            # correctos no pueden convertir esta ejecución incompleta en una tarea de
            # vídeo aparentemente completada.
            prediction_id = str(getattr(exc, "prediction_id", "") or "").strip()
            details = {"wavespeed_prediction_id": prediction_id} if prediction_id else None
            _mark_task_failed(task_id, "materials", str(exc), details=details)
            return None
        except material.OpenAIImagePaidResultError as exc:
            # Una solicitud de pago con una respuesta incierta, una descarga del
            # resultado fallida o un renderizado local fallido debe detenerse antes de
            # otro pedido de imagen.
            _mark_task_failed(task_id, "materials", str(exc))
            return None
        except ofox.OFoxError as exc:
            # La misma semántica de recuperación que en Ark: tanto el estado sin
            # confirmar como el generado pero con descarga fallida corresponden a una
            # tarea remota que se puede recuperar en la consola de OFox, y el estado de
            # fallo se escribe siempre a partir del task_id que lleva la excepción.
            remote_task_id = str(getattr(exc, "task_id", "") or "").strip()
            details = (
                {"ofox_task_id": remote_task_id} if remote_task_id else None
            )
            _mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details=details,
            )
            return None
        except metaso_minimax.MetasoMiniMaxError as exc:
            # Las tareas de Metaso y las de Ark usan entradas de recuperación y nombres
            # de campo distintos, y no se pueden fusionar en un remote_task_id ambiguo.
            # Conservar un prefijo de proveedor claro permite a la API, la WebUI y los
            # logs de operaciones localizar directamente la plataforma correspondiente.
            remote_task_id = str(getattr(exc, "task_id", "") or "").strip()
            details = (
                {"metaso_minimax_task_id": remote_task_id} if remote_task_id else None
            )
            _mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details=details,
            )
            return None
        except infistar_video.InfistarVideoError as exc:
            remote_task_id = str(exc.task_id or "").strip()
            _mark_task_failed(
                task_id, "materials", str(exc),
                details={"infistar_task_id": remote_task_id} if remote_task_id else None,
            )
            return None
        except muapi.MuAPIError as exc:
            # Las tareas de MuAPI son facturables tras aceptarse. Se conserva el ID de
            # solicitud remoto ante cualquier fallo ambiguo o posterior a la
            # finalización para que el usuario pueda recuperarlo en el panel del
            # proveedor.
            remote_task_id = str(getattr(exc, "task_id", "") or "").strip()
            details = {"muapi_task_id": remote_task_id} if remote_task_id else None
            _mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details=details,
            )
            return None
        if not downloaded_videos:
            _mark_task_failed(
                task_id,
                "materials",
                f"no se pudo descargar el material de vídeo de {params.video_source}",
            )
            return None
        return downloaded_videos


def _record_loomloom_run_reference(
    *, task_id: str, run_id: str, listing_version_id: str
) -> bool | None:
    """
    Hace todo lo posible por guardar el Run de LoomLoom de pago ya creado, sin que un
    fallo del estado interrumpa la tarea remota.

    Devuelve True si se guardó correctamente, False si el registro de la tarea ya no
    existe, y None si el backend de estado sigue sin estar disponible tras los
    reintentos limitados. Sea cual sea el resultado, quien llama debe seguir sondeando y
    descargando, porque execute ya ha producido un efecto externo de pago, y detener el
    flujo local solo haría más difícil recuperar el producto.
    """
    fields = {
        "loomloom_run_id": run_id,
        "loomloom_listing_version_id": listing_version_id,
    }
    for attempt in range(1, _LOOMLOOM_STATE_WRITE_ATTEMPTS + 1):
        try:
            updated = sm.state.patch_task(task_id, **fields)
        except Exception as exc:
            if attempt >= _LOOMLOOM_STATE_WRITE_ATTEMPTS:
                logger.exception(
                    "no se pudo guardar la ejecución de pago de LoomLoom tras los reintentos: "
                    f"task_id={task_id}, run_id={run_id}, attempts={attempt}, "
                    f"error={exc}"
                )
                return None
            logger.warning(
                "se reintenta la actualización del estado de la ejecución de pago de LoomLoom: "
                f"task_id={task_id}, run_id={run_id}, attempt={attempt}, "
                f"error={exc}"
            )
            time.sleep(_LOOMLOOM_STATE_RETRY_DELAY_SECONDS)
            continue

        if updated is False:
            logger.warning(
                "no se pudo guardar la ejecución de pago de LoomLoom porque falta la tarea: "
                f"task_id={task_id}, run_id={run_id}"
            )
        return updated

    return None


def _get_material_source_groups(task_id: str, video_paths: list[str]) -> dict[str, str]:
    """Recupera los grupos de palabras clave a partir del manifiesto de material
    descargado.
    """
    try:
        with open(path.join(utils.task_dir(task_id), "script.json"), encoding="utf-8") as file:
            payload = json.load(file)
        groups = {
            record["local_file"]: record["search_term"]
            for record in payload.get("material_sources", [])
            if isinstance(record, dict)
            and isinstance(record.get("local_file"), str)
            and isinstance(record.get("search_term"), str)
            and record["search_term"]
        }
        return {
            file: groups[path.basename(file)]
            for file in video_paths
            if path.basename(file) in groups
        }
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        logger.warning(f"No se pueden leer los grupos de palabras clave del material: task_id={task_id}, error={exc}")
        return {}


def generate_final_videos(
    task_id, params, downloaded_videos, audio_file, subtitle_path, audio_duration,
    *, partial_results: dict | None = None,
):
    """Registra los resultados completados aunque un clip posterior lance una excepción
    antes de volver.
    """
    final_video_paths = []
    combined_video_paths = []
    warnings = []
    if partial_results is not None:
        # Quien llama es dueño de esta instantánea para poder informar de los archivos
        # completados incluso cuando el backend de estado falla durante una escritura de
        # progreso/punto de control.
        partial_results.update(
            videos=final_video_paths, combined_videos=combined_video_paths,
            warnings=warnings,
        )
    allocate_batch_materials = params.video_count > 1 and params.video_source in {
        "pexels", "pixabay", "coverr", "local"
    }
    source_usage = {}
    material_selections = []
    source_groups = (
        _get_material_source_groups(task_id, downloaded_videos)
        if (allocate_batch_materials and params.match_materials_to_script
            and params.video_source != "local")
        else {}
    )
    video_music_provider = _VIDEO_MUSIC_PROVIDERS.get(params.bgm_type)
    video_music_requested = (
        video_music_provider is not None
        and bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
    )
    # La correspondencia conserva el orden de las palabras clave; la asignación por
    # lotes varía los candidatos de cada palabra clave.
    if params.match_materials_to_script:
        video_concat_mode = VideoConcatMode.sequential
    elif params.video_count == 1:
        video_concat_mode = params.video_concat_mode
    else:
        video_concat_mode = VideoConcatMode.random
    video_transition_mode = params.video_transition_mode

    _progress = 50
    for i in range(params.video_count):
        index = i + 1
        if partial_results is not None:
            partial_results["failed_video_index"] = index
        combined_video_path = path.join(
            utils.task_dir(task_id), f"combined-{index}.mp4"
        )
        logger.info(f"\n\n## montando el vídeo: {index} => {combined_video_path}")
        used_video_paths = []
        batch_options = (
            {
                "source_usage": source_usage,
                "source_groups": source_groups,
                "used_video_paths": used_video_paths,
            }
            if allocate_batch_materials else {}
        )
        # Cada vídeo ocupa una parte del 50 % del progreso, mitad para la composición y
        # mitad para la codificación final. El procesamiento de clips es la parte más
        # lenta de la composición, y esa mitad del progreso avanza segmento a segmento.
        combine_share = 50 / params.video_count / 2
        video.combine_videos(
            combined_video_path=combined_video_path,
            video_paths=downloaded_videos,
            audio_file=audio_file,
            video_aspect=params.video_aspect,
            video_fit_mode=params.video_fit_mode,
            video_concat_mode=video_concat_mode,
            video_transition_mode=video_transition_mode,
            max_clip_duration=params.video_clip_duration,
            threads=params.n_threads,
            clip_speed=params.video_clip_speed,
            progress_callback=_stage_progress_reporter(
                task_id, _progress, _progress + combine_share
            ),
            **batch_options,
        )
        if allocate_batch_materials:
            selected_sources = list(dict.fromkeys(used_video_paths))
            reused_sources = [file for file in selected_sources if source_usage.get(file, 0)]
            for file in selected_sources:
                source_usage[file] = source_usage.get(file, 0) + 1
            material_selections.append({
                "video_index": index,
                "local_files": [path.basename(file) for file in used_video_paths],
                "reused_files": [path.basename(file) for file in reused_sources],
            })
            task_artifacts.patch_script_data(task_id, material_selections=material_selections)
            logger.info(
                f"Asignación de material por lotes: video_index={index}, "
                f"sources={len(selected_sources)}, reused_sources={len(reused_sources)}"
            )
            if reused_sources:
                warnings.append({
                    "code": "batch_materials_reused",
                    "video_index": index,
                    "count": len(reused_sources),
                })

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_path = path.join(utils.task_dir(task_id), f"final-{index}.mp4")

        # El modo de música para el vídeo desactiva primero explícitamente la resolución
        # de la BGM por defecto, para que no se use por error un bgm_file residual de
        # una tarea antigua. Solo con volumen mayor que 0 se genera el proxy y se llama
        # a la API de pago; con volumen 0 se omite siempre.
        bgm_file_override = "" if video_music_provider else None
        if video_music_requested:
            service = video_music_provider["service"]
            display_name = video_music_provider["display_name"]
            warning_code = video_music_provider["warning_code"]
            generated_bgm_path = path.join(
                utils.task_dir(task_id),
                (f"{params.bgm_type}-bgm-{index}{video_music_provider['suffix']}"),
            )
            try:
                service.generate_bgm(
                    video_path=combined_video_path,
                    output_path=generated_bgm_path,
                    video_duration=audio_duration,
                    prompt=_get_video_music_prompt(params),
                )
                bgm_file_override = generated_bgm_path
            except video_music_provider["error_type"] as exc:
                # Cuando el vídeo, la narración y los subtítulos ya se han generado, un
                # fallo temporal de la música de un tercero no debería desperdiciar toda
                # la tarea. El vídeo actual desactiva explícitamente la BGM y el
                # resultado degradado se devuelve a la WebUI para avisar al usuario.
                logger.warning(
                    f"falló la generación de la BGM de {display_name}: task_id={task_id}, "
                    f"video_index={index}, error={exc}"
                )
                bgm_file_override = ""
                warnings.append({"code": warning_code, "video_index": index})

        logger.info(f"\n\n## generando el vídeo: {index} => {final_video_path}")
        bgm_mix_succeeded = video.generate_video(
            video_path=combined_video_path,
            audio_path=audio_file,
            subtitle_path=subtitle_path,
            output_file=final_video_path,
            params=params,
            bgm_file_override=bgm_file_override,
        )
        if (
            video_music_provider is not None
            and bgm_file_override
            and not bgm_mix_succeeded
        ):
            # El tercero ya ha devuelto la música con éxito y ha pasado la validación de
            # FFmpeg, pero la mezcla final de MoviePy aún puede fallar por el entorno de
            # ejecución. El servicio de vídeo conservará el vídeo final sin BGM; si la
            # generación por API falla, override está vacío, así que no se añade el
            # aviso dos veces.
            warnings.append(
                {
                    "code": video_music_provider["warning_code"],
                    "video_index": index,
                }
            )

        final_video_paths.append(final_video_path)
        combined_video_paths.append(combined_video_path)
        _progress += 50 / params.video_count / 2
        sm.state.update_task(
            task_id, progress=_progress, videos=list(final_video_paths),
            combined_videos=list(combined_video_paths), warnings=list(warnings) or None,
        )

    return final_video_paths, combined_video_paths, warnings


def _patch_cross_post_state(task_id: str, **kwargs) -> bool | None:
    """Actualiza de forma segura los campos de publicación, con reintentos limitados ante
    fallos breves del backend de estado.
    """
    for attempt in range(1, _CROSS_POST_STATE_WRITE_ATTEMPTS + 1):
        try:
            return sm.state.patch_task(task_id, **kwargs)
        except Exception as exc:
            # Una desconexión breve de Redis no debe dejar la tarea para siempre en
            # pending/processing. El estado de publicación se escribe con muy poca
            # frecuencia, así que aquí basta un número fijo de intentos con esperas
            # cortas para cubrir los fallos pasajeros, evitando a la vez que el hilo en
            # segundo plano se bloquee indefinidamente. El último fallo conserva la
            # traza completa para poder localizarlo.
            if attempt >= _CROSS_POST_STATE_WRITE_ATTEMPTS:
                logger.exception(
                    f"no se pudo actualizar el estado de la publicación multiplataforma tras los reintentos, "
                    f"task_id: {task_id}, fields: {', '.join(kwargs)}, "
                    f"attempts: {attempt}, error: {exc}"
                )
                return None

            logger.warning(
                f"se reintenta la actualización del estado de la publicación multiplataforma, task_id: {task_id}, "
                f"fields: {', '.join(kwargs)}, attempt: {attempt}, error: {exc}"
            )
            time.sleep(_CROSS_POST_STATE_RETRY_DELAY_SECONDS)

    return None


def _record_cross_post_failure(
    task_id: str,
    error: Exception,
    results: list[dict] | None = None,
) -> None:
    """Hace todo lo posible por guardar el fallo de publicación; si el backend de estado no
    está disponible, el log conserva la información de diagnóstico.
    """
    updated = _patch_cross_post_state(
        task_id,
        cross_post_state=const.CROSS_POST_STATE_FAILED,
        cross_post_results=results or None,
        cross_post_error=str(error),
        cross_post_owner=None,
    )
    if updated is False:
        logger.warning(f"se descarta el fallo de publicación multiplataforma de una tarea que no existe: {task_id}")


def _ensure_cross_post_terminal_state(task_id: str) -> None:
    """Cuando termina el Future, las tareas que siguen en estado activo se cierran como
    fallidas.
    """
    try:
        task = sm.state.get_task(task_id)
    except Exception as exc:
        # Esto ya es el callback final del Future, y no hay ningún llamador síncrono
        # posterior que pueda tratar la excepción. Cuando el backend de estado se
        # recupere, el siguiente arranque del proceso tratará los estados residuales con
        # la lógica de recuperación.
        logger.exception(
            f"no se pudo verificar el estado final de la publicación multiplataforma, task_id: {task_id}, error: {exc}"
        )
        return

    if not task or task.get("cross_post_state") not in _ACTIVE_CROSS_POST_STATES:
        return

    logger.warning(
        f"el worker de publicación multiplataforma terminó sin estado final, task_id: {task_id}, "
        f"state: {task.get('cross_post_state')}"
    )
    _record_cross_post_failure(
        task_id,
        RuntimeError("el proceso de publicación multiplataforma terminó sin guardar un estado final"),
        task.get("cross_post_results"),
    )


def recover_interrupted_cross_posts(page_size: int = 100) -> int | None:
    """
    Marca como fallidas las tareas de publicación que no se pueden recuperar tras
    reiniciar el proceso.

    La publicación multiplataforma usa el pool de hilos del proceso actual, no una cola
    de tareas persistente. Al arrancar el proceso, los pending/processing que quedan en
    Redis no siguen ejecutándose solos; si se siguieran considerando en ejecución, el
    usuario nunca podría borrar la tarea. Aquí solo se recogen una vez los ID de tarea y
    luego se lee el estado actual; no se puede avanzar por número de página a lo largo
    de varios SCAN de Redis, porque un cambio en el orden del escaneo haría que se
    perdieran tareas residuales. Solo se tratan los registros activos sin Future
    correspondiente en el proceso actual, conservando los resultados de vídeo ya
    generados.
    """
    recovered = 0
    try:
        task_ids = sm.state.list_task_ids(scan_count=page_size)
        for task_id in task_ids:
            # Tras el escaneo, la tarea puede haberse borrado o haber pasado a un estado
            # final; se decide si recuperar según el estado más reciente, sin
            # sobrescribir un resultado de publicación ya completado con la instantánea
            # antigua del escaneo.
            task = sm.state.get_task(task_id)
            if (
                not task
                or task.get("cross_post_state") not in _ACTIVE_CROSS_POST_STATES
                or _is_cross_post_active_in_process(task_id)
                or _is_cross_post_owner_alive(task.get("cross_post_owner"))
            ):
                continue

            updated = _patch_cross_post_state(
                task_id,
                cross_post_state=const.CROSS_POST_STATE_FAILED,
                cross_post_error=_INTERRUPTED_CROSS_POST_ERROR,
                cross_post_owner=None,
            )
            if updated is True:
                recovered += 1
    except Exception as exc:
        logger.exception(f"no se pudieron recuperar las tareas de publicación multiplataforma interrumpidas: {exc}")
        return None

    if recovered:
        logger.warning(f"tareas de publicación multiplataforma interrumpidas recuperadas: {recovered}")
    return recovered


def _run_cross_post(
    task_id: str,
    video_paths: tuple[str, ...],
    video_subject: str,
    video_script: str,
    video_language: str,
    platforms: tuple[str, ...],
    youtube_privacy_status: str,
    youtube_made_for_kids: bool = False,
    upload_account: dict | None = None,
) -> None:
    """Ejecuta la publicación multiplataforma en segundo plano, completando solo los campos
    de la tarea relacionados con la publicación.
    """
    results = []
    try:
        state_updated = _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_PROCESSING,
            cross_post_error=None,
            cross_post_owner=_cross_post_process_owner,
        )
        if state_updated is not True:
            # False significa que la tarea se ha borrado y None que el backend de estado
            # no está disponible temporalmente. En ambos casos no se debe seguir
            # llamando a la interfaz de terceros, o el usuario no podría consultar ni
            # controlar esta publicación.
            if state_updated is False:
                logger.warning(f"se omite la publicación multiplataforma de una tarea que no existe: {task_id}")
            else:
                _record_cross_post_failure(
                    task_id,
                    RuntimeError("no se pudo guardar el estado de procesamiento de la publicación multiplataforma"),
                )
            return

        logger.info(
            f"publicación multiplataforma iniciada, task_id: {task_id}, platforms: {', '.join(platforms)}"
        )
        youtube_extra = None
        post_title = video_subject or "¡Mira este vídeo! #shorts #viral"
        if platforms:
            has_youtube = any(platform.startswith("youtube") for platform in platforms)
            social_platform = "youtube_shorts"
            if not has_youtube:
                first = (platforms[0] or "").strip().lower()
                # llm.py resuelve los ids desconocidos a su plataforma por defecto.
                social_platform = _CROSS_POST_SOCIAL_PLATFORMS.get(first, first)
            metadata = llm.generate_social_metadata(
                video_subject=video_subject,
                video_script=video_script,
                language=video_language or "",
                platform=social_platform,
            )
            if has_youtube:
                youtube_extra = {
                    "youtube_title": metadata.get("title", video_subject),
                    "youtube_description": metadata.get("caption", ""),
                    "tags": metadata.get("hashtags", []),
                    "privacyStatus": youtube_privacy_status,
                    "selfDeclaredMadeForKids": youtube_made_for_kids,
                    "containsSyntheticMedia": True,
                }
            post_title = (
                metadata.get("caption")
                or metadata.get("title")
                or video_subject
                or "¡Mira este vídeo! #shorts #viral"
            )

        for video_path in video_paths:
            pending_result_index = None

            def record_background_request(request_id: str) -> None:
                nonlocal pending_result_index
                # Se persiste el identificador remoto antes de sondear. Si este proceso
                # sale a mitad de la subida, la recuperación puede mostrar el ID al
                # usuario.
                pending_result_index = len(results)
                results.append(
                    {
                        "video_index": pending_result_index + 1,
                        "request_id": request_id,
                        "status": "processing",
                    }
                )
                if _patch_cross_post_state(
                    task_id, cross_post_results=list(results)
                ) is not True:
                    logger.warning(
                        "no se pudo guardar el ID de solicitud de la subida en segundo plano: "
                        f"task_id={task_id}, request_id={request_id}"
                    )

            upload_kwargs = {}
            if upload_account is not None:
                upload_kwargs["account"] = upload_account
            result = upload_post.cross_post_video(
                video_path=video_path,
                title=post_title,
                platforms=list(platforms),
                youtube_extra=youtube_extra,
                on_background_start=record_background_request,
                **upload_kwargs,
            )
            if not isinstance(result, dict):
                result = {
                    "success": False,
                    "error": "Upload-Post devolvió una respuesta no válida",
                }
            if pending_result_index is None:
                results.append(result)
            else:
                results[pending_result_index] = result

        failures = [result for result in results if not result.get("success")]
        if failures:
            error_messages = [
                str(
                    result.get("error")
                    or result.get("message")
                    or "error de subida desconocido"
                )
                for result in failures
            ]
            cross_post_state = const.CROSS_POST_STATE_FAILED
            cross_post_error = "; ".join(error_messages)
            logger.warning(
                f"publicación multiplataforma completada con fallos, task_id: {task_id}, "
                f"failed: {len(failures)}, total: {len(results)}"
            )
        else:
            cross_post_state = const.CROSS_POST_STATE_COMPLETE
            cross_post_error = None
            logger.success(
                f"publicación multiplataforma completada, task_id: {task_id}, videos: {len(results)}"
            )

        state_updated = _patch_cross_post_state(
            task_id,
            cross_post_state=cross_post_state,
            cross_post_results=results,
            cross_post_error=cross_post_error,
            cross_post_owner=None,
        )
        if state_updated is False:
            logger.warning(f"se descarta el resultado de publicación multiplataforma de una tarea que no existe: {task_id}")
        elif state_updated is None:
            # Si la subida ya ha terminado pero el resultado no se ha persistido, no se
            # puede seguir manteniendo processing. La escritura del estado de fallo
            # vuelve a pasar por reintentos limitados, para que al menos quien llama
            # obtenga un estado final claro.
            _record_cross_post_failure(
                task_id,
                RuntimeError("no se pudo guardar el resultado final de la publicación multiplataforma"),
                results,
            )
    except Exception as exc:
        # Un fallo de publicación solo afecta al estado de publicación y no puede
        # sobrescribir a la inversa una tarea de vídeo ya completada. El texto original
        # de la excepción se escribe en el estado de la tarea, para que quien llama a la
        # API pueda localizar el problema sin acceder a los logs del servidor.
        logger.exception(f"falló la publicación multiplataforma, task_id: {task_id}, error: {exc}")
        _record_cross_post_failure(task_id, exc, results)


def _run_cross_post_with_slot(*args) -> None:
    """Ejecuta la tarea de publicación, asegurando que se devuelve la capacidad de la cola
    tanto si tiene éxito como si falla o lanza una excepción.
    """
    try:
        _run_cross_post(*args)
    except Exception as exc:
        # _run_cross_post ya trata las excepciones previstas; esto es la última
        # protección, para que las excepciones que lance lógica añadida en el futuro no
        # se queden solo en un Future que nadie lee.
        task_id = str(args[0]) if args else "unknown"
        logger.exception(f"el worker de publicación multiplataforma falló, task_id: {task_id}, error: {exc}")
        if args:
            _record_cross_post_failure(task_id, exc)
    finally:
        _cross_post_slots.release()


def _finalize_cross_post_future(task_id: str, future: Future) -> None:
    """Limpia el registro de Futures, asegurando que la cancelación, las excepciones y los
    fallos de escritura del estado se cierran.
    """
    _unregister_cross_post_future(task_id, future)

    try:
        error = future.exception()
    except CancelledError:
        logger.warning(f"se canceló el future de publicación multiplataforma, task_id: {task_id}")
        # Si un Future se cancela antes de empezar a ejecutarse, el finally del worker
        # no se ejecuta, así que hay que devolver la capacidad de la cola en el callback
        # y cambiar el estado persistido a fallo.
        _cross_post_slots.release()
        _record_cross_post_failure(
            task_id,
            RuntimeError("el trabajo de publicación multiplataforma se canceló antes de ejecutarse"),
        )
        return
    except Exception as exc:
        logger.exception(
            f"no se pudo inspeccionar el future de publicación multiplataforma, task_id: {task_id}, error: {exc}"
        )
        _ensure_cross_post_terminal_state(task_id)
        return

    if error is not None:
        logger.error(
            f"falló el future de publicación multiplataforma, task_id: {task_id}, "
            f"error: {type(error).__name__}: {error}"
        )

    _ensure_cross_post_terminal_state(task_id)


def _schedule_cross_post(
    task_id: str,
    video_paths: list[str],
    params: VideoParams,
    video_script: str,
    platforms: list[str],
    youtube_privacy_status: str,
    youtube_made_for_kids: bool = False,
) -> str | None:
    """Envía la tarea de publicación en segundo plano; si tiene éxito devuelve None, y si
    falla la planificación devuelve un motivo de error consultable.
    """
    if not _cross_post_slots.acquire(blocking=False):
        error = "la cola de publicación multiplataforma está llena; se omitió la publicación"
        logger.warning(
            f"se omite la publicación multiplataforma porque la cola está llena, task_id: {task_id}, "
            f"capacity: {_cross_post_max_pending_tasks}"
        )
        _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_FAILED,
            cross_post_error=error,
            cross_post_owner=None,
        )
        return error

    try:
        future = _cross_post_executor.submit(
            _run_cross_post_with_slot,
            task_id,
            tuple(video_paths),
            params.video_subject or "",
            video_script,
            params.video_language or "",
            tuple(platforms),
            youtube_privacy_status,
            youtube_made_for_kids,
            upload_post.upload_post_service.snapshot_account(),
        )
        _register_cross_post_future(task_id, future)
        future.add_done_callback(partial(_finalize_cross_post_future, task_id))
    except RuntimeError as exc:
        _unregister_cross_post_future(task_id)
        _cross_post_slots.release()
        logger.exception(
            f"no se pudo programar la publicación multiplataforma, task_id: {task_id}, error: {exc}"
        )
        _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_FAILED,
            cross_post_error=f"no se pudo programar la publicación multiplataforma: {exc}",
            cross_post_owner=None,
        )
        return f"no se pudo programar la publicación multiplataforma: {exc}"

    return None


def _run_pipeline(
    task_id,
    params: VideoParams,
    stop_at: str = "video",
    voice_preview: dict | None = None,
    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,
    allow_server_file_input: bool = False,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
):
    logger.info(f"inicio de la tarea: {task_id}, stop_at: {stop_at}")
    sm.state.update_task(
        task_id, state=const.TASK_STATE_PROCESSING, progress=5,
        failed_stage=None, error=None,
        videos=[], combined_videos=[], warnings=None, failed_video_index=None,
    )

    if stop_at == "video" and params.subtitle_enabled:
        try:
            video.validate_subtitle_colors(params)
        except ValueError as exc:
            return _mark_task_failed(task_id, "preflight", str(exc))

    if stop_at in {"materials", "video"} and params.video_source not in _SUPPORTED_VIDEO_SOURCES:
        return _mark_task_failed(
            task_id, "preflight", f"fuente de vídeo no admitida: {params.video_source!r}"
        )

    if (
        stop_at in {"materials", "video"}
        and params.video_source == "volcengine_seedance"
        and not volcengine_seedance.is_enabled()
    ):
        return _mark_task_failed(
            task_id,
            "preflight",
            "Volcano Engine Seedance requiere una API Key de Ark",
        )

    if stop_at in {"materials", "video"} and params.video_source == "infistar":
        try:
            if not infistar_video.is_enabled():
                raise infistar_video.InfistarVideoError("La generación de vídeo de Infistar necesita una clave API")
            infistar_video.model_id()
            if params.video_aspect not in {"16:9", "9:16"}:
                raise infistar_video.InfistarVideoError("El vídeo de Infistar admite actualmente 16:9 y 9:16")
        except infistar_video.InfistarVideoError as exc:
            return _mark_task_failed(task_id, "preflight", str(exc))

    if (
        stop_at in {"materials", "video"}
        and params.video_source == "ofox"
        and not ofox.is_enabled()
    ):
        return _mark_task_failed(
            task_id,
            "preflight",
            "la generación de vídeo con OFox requiere una API Key de OFox",
        )

    if (
        stop_at in {"materials", "video"}
        and params.video_source == "metaso_minimax"
        and not metaso_minimax.is_enabled()
    ):
        return _mark_task_failed(
            task_id,
            "preflight",
            "Metaso MiniMax requiere una API Key",
        )

    if (
        stop_at in {"materials", "video"}
        and params.video_source == "muapi"
        and not muapi.is_enabled()
    ):
        return _mark_task_failed(
            task_id,
            "preflight",
            "la generación de vídeo con MuAPI requiere una API Key de MuAPI",
        )

    if (
        stop_at in {"materials", "video"}
        and params.video_source == "openai_image"
        and not material.is_openai_image_enabled(
            config.snapshot_config_with_pending(config.app)
        )
    ):
        return _mark_task_failed(
            task_id,
            "preflight",
            "la fuente de imágenes OpenAI requiere openai_image_base_url y "
            "openai_image_model en config.toml (openai_image_api_keys es "
            "opcional para puertas de enlace locales sin autenticación)",
        )

    # Solo el flujo de vídeo final completo necesita el proveedor de música para el
    # vídeo. Las tareas completas a las que les falta la clave se bloquean lo antes
    # posible, para no gastar antes el saldo del LLM, del TTS y de los servicios de
    # material; las interfaces de productos intermedios siguen pudiendo usarse por
    # separado.
    video_music_provider = _VIDEO_MUSIC_PROVIDERS.get(params.bgm_type)
    video_music_enabled = (
        stop_at == "video"
        and video_music_provider is not None
        and bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
    )
    if video_music_enabled:
        service = video_music_provider["service"]
        display_name = video_music_provider["display_name"]
        if not service.is_enabled():
            return _mark_task_failed(
                task_id,
                "preflight",
                f"la música de fondo de {display_name} requiere una API Key",
            )

        # La WebUI limita la longitud de la entrada, pero la API, la CLI y las tareas
        # históricas pueden saltarse los controles del frontend. Antes de generar el
        # guion, la locución y el material se vuelve a validar con el límite del
        # proveedor, para no esperar a que una solicitud del tercero lo rechace tras
        # componer el vídeo entero. La capa de servicio mantiene la misma validación
        # como última defensa ante llamadas directas.
        music_prompt = _get_video_music_prompt(params)
        max_prompt_length = int(getattr(service, "MAX_PROMPT_LENGTH", 0) or 0)
        if max_prompt_length and len(music_prompt) > max_prompt_length:
            return _mark_task_failed(
                task_id,
                "preflight",
                (f"el prompt de música de {display_name} supera los {max_prompt_length} caracteres"),
            )

        # Un proveedor puede ofrecer una comprobación previa de la cuenta que no se
        # factura. La función de comprobación solo debe lanzar errores deterministas; si
        # por fluctuaciones de red o alcance de permisos no se puede confirmar, la capa
        # de servicio registra un aviso y sigue con la generación real.
        validate_access = getattr(service, "validate_generation_access", None)
        if callable(validate_access):
            try:
                validate_access()
            except video_music_provider["error_type"] as exc:
                return _mark_task_failed(task_id, "preflight", str(exc))

    # Solo los productos intermedios script/terms no necesitan FFmpeg (no generan audio
    # ni vídeo). La API, la CLI y la WebUI ejecutan las tareas a través de esta entrada
    # compartida, así que comprobarlo aquí de forma común, en lugar de repetirlo en cada
    # entrada, garantiza que las tres rutas se comporten igual. Va después de la
    # validación de la clave de música para no cambiar el orden de "fallar primero" ni
    # los mensajes de error originales de esas validaciones.
    if stop_at not in ("script", "terms") and not utils.check_ffmpeg_ready():
        return _mark_task_failed(
            task_id,
            "preflight",
            "ffmpeg no está disponible; instala ffmpeg o configura app.ffmpeg_path "
            "en config.toml con un ejecutable de ffmpeg que funcione",
        )

    # 1. Genera el guion
    video_script = generate_script(task_id, params)
    # El adaptador del LLM usa un "Error: " inicial como centinela de fallo. Un guion
    # proporcionado por el usuario puede citar ese texto legítimamente en cualquier
    # sitio, incluido el principio, así que solo se interpreta en un guion generado por
    # el LLM.
    is_provider_error = (
        not (params.video_script or "").strip()
        and isinstance(video_script, str)
        and video_script.startswith("Error: ")
    )
    if not video_script or is_provider_error:
        error = (
            video_script.removeprefix("Error: ").strip()
            if is_provider_error
            else "no se pudo generar el guion del vídeo"
        )
        return _mark_task_failed(task_id, "script", error)

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=10)

    if stop_at == "script":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, script=video_script
        )
        return {"script": video_script}

    # 2. Genera los términos
    video_terms = ""
    if stop_at in {"terms", "materials", "video"} and params.video_source != "local":
        video_terms = generate_terms(task_id, params, video_script)
        if not video_terms:
            return _mark_task_failed(
                task_id,
                "terms",
                "no se pudieron generar los términos de búsqueda del vídeo",
            )

    save_script_data(task_id, video_script, video_terms, params)

    if stop_at == "terms":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, terms=video_terms
        )
        return {"script": video_script, "terms": video_terms}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=20)

    # 3. Genera el audio
    generate_audio_kwargs = {
        "voice_preview": voice_preview,
        "allow_server_file_input": allow_server_file_input,
    }
    if voxcpm_reference_audio is not None:
        generate_audio_kwargs["voxcpm_reference_audio"] = voxcpm_reference_audio
    if voxcpm_prompt_audio is not None:
        generate_audio_kwargs["voxcpm_prompt_audio"] = voxcpm_prompt_audio
        generate_audio_kwargs["voxcpm_prompt_text"] = voxcpm_prompt_text
    audio_file, audio_duration, sub_maker = generate_audio(
        task_id,
        params,
        video_script,
        **generate_audio_kwargs,
    )
    if not audio_file:
        return _mark_task_failed(
            task_id,
            "audio",
            "no se pudo preparar el audio de la narración",
        )

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=30)

    if stop_at == "audio":
        # Los trabajos de vídeo completos aplican la ganancia en el mezclador final. Las
        # exportaciones solo de audio no tienen mezclador, así que aquí se aplica a una
        # salida propia (nunca a una entrada personalizada).
        audio_volume = 1.0 if params.voice_volume is None else float(params.voice_volume)
        if audio_volume != 1.0:
            descriptor, output_file = tempfile.mkstemp(
                prefix="audio-export-", suffix=".mp3", dir=utils.task_dir(task_id)
            )
            os.close(descriptor)
            export_ready = False
            try:
                export_ready = voice.apply_audio_volume(audio_file, output_file, audio_volume)
                if not export_ready:
                    return _mark_task_failed(task_id, "audio", "no se pudo aplicar el volumen de exportación del audio")
            finally:
                if not export_ready:
                    try:
                        os.remove(output_file)
                    except OSError as exc:
                        logger.warning(f"no se pudo eliminar la exportación de audio incompleta: {exc}")
            audio_file = output_file
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            audio_file=audio_file,
        )
        return {"audio_file": audio_file, "audio_duration": audio_duration}

    # 4. Genera los subtítulos
    subtitle_path = generate_subtitle(
        task_id, params, video_script, sub_maker, audio_file
    )

    if stop_at == "subtitle":
        if not subtitle_path:
            return _mark_task_failed(
                task_id,
                "subtitle",
                "no se pudieron generar los subtítulos; verifica el proveedor de subtítulos y la línea de tiempo",
            )
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            subtitle_path=subtitle_path,
        )
        return {"subtitle_path": subtitle_path}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=40)

    # 5. Obtiene el material de vídeo
    downloaded_videos = get_video_materials(
        task_id,
        params,
        video_terms,
        audio_duration,
        loomloom_video_request=loomloom_video_request,
    )
    if not downloaded_videos:
        return _mark_task_failed(
            task_id,
            "materials",
            "no se pudo preparar el material de vídeo",
        )

    if stop_at == "materials":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            materials=downloaded_videos,
        )
        return {"materials": downloaded_videos}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=50)

    # Solo el flujo de generación de vídeo completo necesita tratar el modo de montaje
    # del vídeo; así se evita que solicitudes como /subtitle y /audio accedan a campos
    # que no existen.
    if type(params.video_concat_mode) is str:
        params.video_concat_mode = VideoConcatMode(params.video_concat_mode)

    # 6. Genera los vídeos finales
    partial_results = {}
    try:
        final_video_paths, combined_video_paths, generation_warnings = generate_final_videos(
            task_id,
            params,
            downloaded_videos,
            audio_file,
            subtitle_path,
            audio_duration,
            partial_results=partial_results,
        )
    except Exception as exc:
        logger.exception(f"falló el lote de vídeos, task_id: {task_id}, error: {exc}")
        # Los avisos del clip que falla no describen ninguna salida utilizable.
        completed_count = len(partial_results.get("videos", []))
        partial_results["warnings"] = [
            warning for warning in partial_results.get("warnings", [])
            if warning.get("video_index", 0) <= completed_count
        ] or None
        return _mark_task_failed(
            task_id, "video", f"{type(exc).__name__}: {exc}", details=partial_results,
        )

    if not final_video_paths:
        return _mark_task_failed(
            task_id,
            "video",
            "no se pudo generar el vídeo final",
        )

    logger.success(
        f"la tarea {task_id} terminó y generó {len(final_video_paths)} vídeos."
    )

    # 7. Primero se completa la tarea de generación de vídeo y después, si hace falta,
    #    se envía la publicación multiplataforma. Las subidas a terceros pueden tardar
    #    varios minutos, así que no deben bloquear la devolución del resultado del vídeo
    #    ni afectar a la inversa al vídeo final ya generado.
    cross_post_enabled = (
        upload_post.upload_post_service.is_configured()
        and upload_post.upload_post_service.auto_upload
    )
    platforms = (
        list(upload_post.upload_post_service.platforms) if cross_post_enabled else []
    )
    should_cross_post = cross_post_enabled and bool(platforms)
    if cross_post_enabled and not platforms:
        logger.warning(
            f"se omite la publicación multiplataforma porque no hay plataformas configuradas, task_id: {task_id}"
        )
    cross_post_state = const.CROSS_POST_STATE_PENDING if should_cross_post else None

    kwargs = {
        "videos": final_video_paths,
        "combined_videos": combined_video_paths,
        "script": video_script,
        "terms": video_terms,
        "audio_file": audio_file,
        "audio_duration": audio_duration,
        "subtitle_path": subtitle_path,
        "materials": downloaded_videos,
        "cross_post_state": cross_post_state,
        "cross_post_results": None,
        "cross_post_error": None,
        "cross_post_owner": _cross_post_process_owner if should_cross_post else None,
        "warnings": generation_warnings or None,
    }
    sm.state.update_task(
        task_id, state=const.TASK_STATE_COMPLETE, progress=100, **kwargs
    )

    if should_cross_post:
        scheduling_error = _schedule_cross_post(
            task_id=task_id,
            video_paths=final_video_paths,
            params=params,
            video_script=video_script,
            platforms=platforms,
            youtube_privacy_status=(
                upload_post.upload_post_service.youtube_privacy_status
            ),
            # Se fija la elección de público en el momento de encolar; cambiar después
            # la WebUI no debe cambiar la declaración del vídeo ya encolado.
            youtube_made_for_kids=(
                upload_post.upload_post_service.youtube_made_for_kids
            ),
        )
        # Una cola llena o un pool de hilos cerrado son fallos de planificación que se
        # conocen de forma síncrona. La función de planificación ya ha actualizado el
        # estado de la tarea, así que aquí se corrige de forma síncrona la instantánea
        # devuelta, para que quien llama no reciba un pending distinto de lo que dirán
        # las consultas posteriores.
        if scheduling_error:
            kwargs["cross_post_state"] = const.CROSS_POST_STATE_FAILED
            kwargs["cross_post_error"] = scheduling_error
            kwargs["cross_post_owner"] = None

    return kwargs


def start(
    task_id,
    params: VideoParams,
    stop_at: str = "video",
    voice_preview: dict | None = None,
    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,
    allow_server_file_input: bool = False,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
):
    """
    Ejecuta el flujo de la tarea, asegurando que también las excepciones inesperadas se
    conviertan en un estado de fallo consultable.

    ``allow_server_file_input`` solo lo usa la CLI local. La API HTTP y la WebUI deben
    mantener el valor por defecto, para que el audio personalizado siga siempre limitado
    al directorio de la tarea actual.
    """
    try:
        return _run_pipeline(
            task_id,
            params,
            stop_at=stop_at,
            voice_preview=voice_preview,
            loomloom_video_request=loomloom_video_request,
            allow_server_file_input=allow_server_file_input,
            voxcpm_reference_audio=voxcpm_reference_audio,
            voxcpm_prompt_audio=voxcpm_prompt_audio,
            voxcpm_prompt_text=voxcpm_prompt_text,
        )
    except Exception as exc:
        logger.exception(
            f"fallo inesperado del flujo de la tarea, task_id: {task_id}, error: {exc}"
        )
        return _mark_task_failed(
            task_id,
            "pipeline",
            f"{type(exc).__name__}: {exc}",
        )


if __name__ == "__main__":
    task_id = "task_id"
    params = VideoParams(
        video_subject="El papel del dinero",
        voice_name="zh-CN-XiaoyiNeural-Female",
        voice_rate=1.0,
    )
    start(task_id, params, stop_at="video")
