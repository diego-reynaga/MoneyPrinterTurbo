import inspect
import json
from typing import Dict

import redis
from loguru import logger
from pydantic import ValidationError

from app.controllers.manager.base_manager import TaskManager, TaskQueueFullError
from app.models import const
from app.models.schema import AudioRequest, SubtitleRequest, VideoParams
from app.services import state as sm
from app.services import task as tm

FUNC_MAP = {
    "start": tm.start,
    # 'start_test': tm.start_test
}

_ADMIT_QUEUED_TASK_SCRIPT = """
if redis.call("LLEN", KEYS[1]) >= tonumber(ARGV[1]) then
    return 0
end
redis.call("RPUSH", KEYS[1], ARGV[2])
return 1
"""


class RedisTaskManager(TaskManager):
    def __init__(
        self,
        max_concurrent_tasks: int,
        redis_url: str,
        max_queued_tasks: int = 100,
    ):
        self.redis_client = redis.Redis.from_url(redis_url)
        super().__init__(max_concurrent_tasks, max_queued_tasks=max_queued_tasks)

    def create_queue(self):
        return "task_queue"

    def resume_queued_tasks(self):
        """Despacha las entradas de cola persistidas cuando arranca un nuevo proceso de la
        API.

        Los elementos en cola de Redis sobreviven a un reinicio, pero ningún worker
        llama a check_queue hasta que termina una tarea activa. Se ocupa cada plaza de
        worker local disponible para que un proceso inactivo no deje las solicitudes
        antiguas atascadas en processing.
        """
        for _ in range(max(0, self.max_concurrent_tasks - self.current_tasks)):
            self.check_queue()

    def enqueue(self, task: Dict):
        self.redis_client.rpush(self.queue, self._serialize_task(task))

    def enqueue_new_task(self, task: Dict):
        # Los bloqueos del gestor son locales al proceso. La validación de capacidad y
        # la admisión se hacen en una sola operación de Redis para que otros workers de
        # la API no puedan ocupar la misma plaza.
        admitted = self.redis_client.eval(
            _ADMIT_QUEUED_TASK_SCRIPT,
            1,
            self.queue,
            self.max_queued_tasks,
            self._serialize_task(task),
        )
        if not admitted:
            raise TaskQueueFullError("la cola de tareas está llena, inténtalo de nuevo más tarde")

    @staticmethod
    def _serialize_task(task: Dict) -> str:
        task_with_serializable_params = task.copy()
        # task.copy() solo copia el diccionario exterior; si se modificaran directamente
        # los kwargs anidados, el VideoParams que tiene quien llama se sustituiría
        # también por un dict. Los logs o reintentos posteriores aún podrían leer la
        # tarea original, así que aquí se copian los kwargs por separado, para que la
        # serialización no tenga efectos secundarios inesperados.
        task_kwargs = task.get("kwargs", {})
        task_with_serializable_params["kwargs"] = task_kwargs.copy()

        if "params" in task_kwargs and isinstance(
            task_kwargs["params"], (VideoParams, AudioRequest, SubtitleRequest)
        ):
            task_with_serializable_params["kwargs"]["params"] = task_kwargs[
                "params"
            ].model_dump(warnings=False)

        # Convierte el objeto función en su nombre
        task_with_serializable_params["func"] = task["func"].__name__
        return json.dumps(task_with_serializable_params)

    def dequeue(self):
        # Se usa un bucle en lugar de una sola extracción: una tarea pudo cumplir las
        # reglas de validación al encolarse, pero las reglas de validación y los
        # miembros de FUNC_MAP cambian con los despliegues (por ejemplo, VideoParams
        # añade una restricción ge=1 o se elimina una función de entrada), así que en la
        # cola pueden quedar entradas escritas con un esquema antiguo o que ya no se
        # pueden analizar. lpop es destructivo: una vez extraída no se puede devolver a
        # su sitio; esa tarea ya ha salido de la cola para siempre y no se puede fingir
        # que sigue ahí. En lugar de dejar que la excepción suba desde aquí (check_queue
        # llama a este método con el bloqueo, y la excepción tumbaría el hilo de trabajo
        # a través del finally de task_done → run_task; a partir de ahí no habría tareas
        # en marcha, nadie volvería a llamar a check_queue y las tareas siguientes de la
        # cola se quedarían para siempre en processing), es mejor descartarla en el
        # sitio y seguir con la siguiente, manteniendo el contrato de "obtener una tarea
        # utilizable o que la cola esté realmente vacía".
        while True:
            task_json = self.redis_client.lpop(self.queue)
            # Solo que lpop no extraiga nada significa que la cola está vacía. Una
            # cadena vacía (o bytes vacíos) también es una entrada inutilizable y puede
            # haber tareas válidas detrás, así que debe ir por la ruta de descarte de
            # abajo en lugar de tratarse como "fin de la cola" y volver.
            if task_json is None:
                return None

            task_info = None
            try:
                task_info = json.loads(task_json)
                # Convierte el nombre de la función de vuelta en el objeto función. Si
                # falta el nombre o ya no está en FUNC_MAP no se puede indexar
                # directamente, o un KeyError se saltaría la política de descarte de
                # params de abajo.
                task_info["func"] = FUNC_MAP[task_info["func"]]
                task_kwargs = task_info["kwargs"]
                if not isinstance(task_kwargs, dict):
                    raise ValueError("la tarea en cola no tiene un mapa de argumentos con nombre")
                # Si falta args entero se usa el valor por defecto de check_queue; si se
                # escribe como null u otra forma que no sea un array, check_queue
                # lanzaría TypeError al expandir `*args`, y allí la entrada volvería a
                # encolarse y la excepción escaparía del hilo de trabajo, así que hay
                # que interceptarlo aquí antes.
                if not isinstance(task_info.get("args", []), list):
                    raise ValueError("los argumentos posicionales de la tarea en cola no son una lista")
                # Una solicitud persistida puede sobrevivir a la firma de una función.
                # Se marca como fallida antes de despachar: los errores de enlace de
                # argumentos de Python ocurren antes de que start() entre en su
                # manejador de fallos del flujo.
                inspect.signature(task_info["func"]).bind(
                    *task_info.get("args", []), **task_kwargs
                )
            except (TypeError, ValueError, KeyError) as e:
                logger.error(f"se descarta una tarea en cola inutilizable: {e}")
                # Igual que la ruta de fallo de validación de params de abajo: si se
                # puede leer un task_id utilizable, esta tarea que ya ha salido para
                # siempre de la cola se cierra como fallida; si no, la API/WebUI la
                # mostrarían siempre en processing. Si la carga útil no se puede
                # analizar o el task_id no es una cadena (por ejemplo un array JSON), no
                # hay registro que actualizar y solo se puede descartar; pasar un valor
                # que no es cadena a patch_task haría que redis lanzara DataError e
                # interrumpiera a su vez el bucle de descarte.
                stale_kwargs = (
                    task_info.get("kwargs") if isinstance(task_info, dict) else None
                )
                task_id = (
                    stale_kwargs.get("task_id")
                    if isinstance(stale_kwargs, dict)
                    else None
                )
                if isinstance(task_id, str) and task_id:
                    sm.state.patch_task(
                        task_id,
                        state=const.TASK_STATE_FAILED,
                        failed_stage="dequeue",
                        error=f"se descartó una tarea en cola obsoleta: {e}",
                    )
                continue

            if "params" in task_kwargs and isinstance(task_kwargs["params"], dict):
                try:
                    params_model = VideoParams
                    # /audio y /subtitle usan modelos de solicitud abreviados sin
                    # video_subject. Se conservan esos esquemas en el límite de la cola;
                    # un VideoParams completo también puede detenerse en cualquiera de
                    # esas etapas.
                    if "video_subject" not in task_kwargs["params"]:
                        if task_kwargs.get("stop_at") == "audio":
                            params_model = AudioRequest
                        elif task_kwargs.get("stop_at") == "subtitle":
                            params_model = SubtitleRequest
                    task_kwargs["params"] = params_model(**task_kwargs["params"])
                except ValidationError as e:
                    logger.error(
                        "se descarta una tarea en cola cuyos parámetros no pasan la validación actual "
                        f"del modelo de solicitud (encolada con un esquema anterior más "
                        f"permisivo, o dañada): {e}"
                    )
                    # El registro de estado de la tarea se crea antes de encolar y por
                    # defecto está en processing; si solo se descartara esta entrada de
                    # la cola sin tocar el estado, la API/WebUI mostrarían siempre la
                    # tarea en ejecución y nunca pasaría a fallo. Se usa patch_task y no
                    # update_task para no volver a crear la tarea si el usuario ya la ha
                    # borrado. Si task_id no es una cadena, igual que arriba: no hay
                    # registro que actualizar y se omite la actualización de estado.
                    task_id = task_kwargs.get("task_id")
                    if isinstance(task_id, str) and task_id:
                        sm.state.patch_task(
                            task_id,
                            state=const.TASK_STATE_FAILED,
                            failed_stage="dequeue",
                            error=f"se descartó una tarea en cola obsoleta: {e}",
                        )
                    continue

            return task_info

    def is_queue_empty(self):
        return self.redis_client.llen(self.queue) == 0

    def queue_size(self):
        return self.redis_client.llen(self.queue)
