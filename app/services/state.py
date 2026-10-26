import ast
import copy
import threading
from abc import ABC, abstractmethod

from redis.exceptions import ResponseError
from itertools import islice

from app.config import config
from app.models import const


_PATCH_EXISTING_TASK_SCRIPT = """
-- Replica las comprobaciones de propiedad de discovery/get_task en la misma operación atómica.
-- Otros servicios y nuestra cola pueden compartir esta base de datos o reutilizar una clave borrada.
if redis.call("TYPE", KEYS[1]).ok ~= "hash" then
    return 0
end
if redis.call("HGET", KEYS[1], "task_id") ~= KEYS[1] then
    return 0
end

for index = 1, #ARGV, 2 do
    redis.call("HSET", KEYS[1], ARGV[index], ARGV[index + 1])
end

return 1
"""


_DELETE_OWNED_TASK_SCRIPT = """
if redis.call("TYPE", KEYS[1]).ok ~= "hash" then
    return 0
end
if redis.call("HGET", KEYS[1], "task_id") ~= KEYS[1] then
    return 0
end

return redis.call("DEL", KEYS[1])
"""


# Clase base de la gestión de estado
class BaseState(ABC):
    @abstractmethod
    def update_task(self, task_id: str, state: int, progress: int = 0, **kwargs):
        pass

    @abstractmethod
    def get_task(self, task_id: str):
        pass

    @abstractmethod
    def get_all_tasks(self, page: int, page_size: int):
        pass

    @abstractmethod
    def list_task_ids(self, scan_count: int = 100) -> list[str]:
        """Obtiene los ID de tarea de un recorrido, para operaciones completas como la
        recuperación al arrancar.
        """
        pass

    @abstractmethod
    def patch_task(self, task_id: str, **kwargs) -> bool:
        """Solo actualiza los campos indicados de una tarea existente; si la tarea no
        existe devuelve False.
        """
        pass


# Gestión de estado en memoria
class MemoryState(BaseState):
    def __init__(self):
        self._tasks = {}
        self._lock = threading.RLock()

    def get_all_tasks(self, page: int, page_size: int):
        start = (page - 1) * page_size
        end = start + page_size
        with self._lock:
            total = len(self._tasks)
            tasks = [
                copy.deepcopy(task)
                for task in islice(self._tasks.values(), start, end)
            ]
        return tasks, total

    def list_task_ids(self, scan_count: int = 100) -> list[str]:
        with self._lock:
            return list(self._tasks)

    def update_task(
        self,
        task_id: str,
        state: int = const.TASK_STATE_PROCESSING,
        progress: int = 0,
        **kwargs,
    ):
        progress = int(progress)
        if progress > 100:
            progress = 100

        with self._lock:
            self._tasks[task_id] = {
                # Conserva los campos de etapas anteriores del flujo, igual que las
                # actualizaciones HSET de Redis. Una actualización que solo trae el
                # progreso no debe borrar el tema de la WebUI ni los detalles de
                # diagnóstico ya guardados.
                **self._tasks.get(task_id, {}),
                "task_id": task_id,
                "state": state,
                "progress": progress,
                **copy.deepcopy(kwargs),
            }

    def get_task(self, task_id: str):
        with self._lock:
            task = self._tasks.get(task_id, None)
            return copy.deepcopy(task) if task is not None else None

    def patch_task(self, task_id: str, **kwargs) -> bool:
        # La publicación asíncrona solo debe añadir el estado de publicación, sin
        # sobrescribir los resultados ya guardados del vídeo, los subtítulos, etc.
        # Comprobar la existencia y fusionar los campos dentro del mismo bloqueo también
        # evita que un hilo en segundo plano vuelva a crear la tarea después de
        # borrarla.
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            task.update(copy.deepcopy(kwargs))
            return True

    def delete_task(self, task_id: str):
        with self._lock:
            self._tasks.pop(task_id, None)


# Gestión de estado en Redis
class RedisState(BaseState):
    """
    Estado de tareas respaldado por Redis.

    Límite de confianza: se espera que Redis sea privado de esta aplicación. Los valores
    de las tareas los escribe MoneyPrinterTurbo y se convierten de nuevo desde cadenas
    por compatibilidad con los registros de estado existentes. No expongas esta base de
    datos Redis a escritores no fiables sin sustituir la deserialización por un formato
    más estricto basado en esquemas.
    """

    def __init__(self, host="localhost", port=6379, db=0, password=None):
        import redis

        self._redis = redis.StrictRedis(host=host, port=port, db=db, password=password)

    def get_all_tasks(self, page: int, page_size: int):
        start = (page - 1) * page_size
        end = start + page_size
        # Cada página se corta con el mismo orden determinista, en lugar de depender del
        # orden que devuelve SCAN. Esto sigue sin ser una instantánea transaccional
        # cuando se añaden o borran tareas a la vez; sin índice, se prioriza que la
        # paginación de un conjunto de tareas estático sea correcta, y total cuenta las
        # claves de tarea sin duplicados.
        task_ids = self.list_task_ids(scan_count=page_size)
        tasks = []
        for task_id in task_ids[start:end]:
            task = self.get_task(task_id)
            if task is not None:
                tasks.append(task)
        return tasks, len(task_ids)

    def list_task_ids(self, scan_count: int = 100) -> list[str]:
        """Devuelve solo los hashes de tareas de esta aplicación de una base de datos que
        puede ser compartida.
        """
        task_keys = set()
        cursor = 0
        while True:
            # Además de los Hash de tareas, la base de datos Redis puede contener colas
            # List que usa RedisTaskManager. Escanear solo los Hash evita provocar
            # WRONGTYPE al ejecutar HGETALL sobre una cola. COUNT es una indicación del
            # trabajo de escaneo y no garantiza el número de elementos de cada lote.
            cursor, keys = self._redis.scan(
                cursor,
                count=scan_count,
                _type="HASH",
            )
            # La db 0 de Redis también puede contener hashes de otros servicios. Se
            # comprueba el marcador de tarea en una ida y vuelta por pipeline por cada
            # lote de SCAN; exponer aquí todas las claves HASH podría filtrar sus campos
            # a través de /tasks.
            candidates = [key for key in dict.fromkeys(keys) if key not in task_keys]
            if candidates:
                with self._redis.pipeline(transaction=False) as pipeline:
                    for key in candidates:
                        pipeline.hget(key, "task_id")
                    # Una clave puede cambiar de tipo tras el SCAN. Se aísla esa fila en
                    # lugar de dejar que un WRONGTYPE aborte todo el listado de tareas.
                    embedded_ids = pipeline.execute(raise_on_error=False)
                for key, embedded_id in zip(candidates, embedded_ids):
                    if isinstance(embedded_id, ResponseError):
                        if str(embedded_id).startswith("WRONGTYPE"):
                            continue
                        raise embedded_id
                    if embedded_id == key:
                        task_keys.add(key)
            if cursor == 0:
                break
        # Ordenar por clave de tarea no depende del orden de escaneo de los Hash; no se
        # mantiene un índice extra ni se cambian las tareas antiguas.
        return [key.decode("utf-8") for key in sorted(task_keys)]

    def update_task(
        self,
        task_id: str,
        state: int = const.TASK_STATE_PROCESSING,
        progress: int = 0,
        **kwargs,
    ):
        progress = int(progress)
        if progress > 100:
            progress = 100

        fields = {
            "task_id": task_id,
            "state": state,
            "progress": progress,
            **kwargs,
        }

        # Un solo HSET escribe todo el estado de la tarea de forma atómica. Comandos
        # separados podrían mostrar a los lectores un estado nuevo con el progreso o los
        # resultados anteriores, y dejar un registro a medio actualizar ante un fallo de
        # red.
        self._redis.hset(
            task_id,
            mapping={
                field: self._serialize_field(field, value)
                for field, value in fields.items()
            },
        )

    def get_task(self, task_id: str):
        try:
            task_data = self._redis.hgetall(task_id)
        except ResponseError as exc:
            # Un ID de tarea arbitrario puede nombrar la cola List de la aplicación o
            # una clave que no es hash de otro servicio. Esos no son registros de tarea.
            if str(exc).startswith("WRONGTYPE"):
                return None
            raise
        # Quien llama a la API puede pedir cualquier clave de Redis por su nombre. Se
        # exige el mismo marcador que list_task_ids antes de devolver el contenido de un
        # hash.
        if not task_data or task_data.get(b"task_id") != task_id.encode("utf-8"):
            return None

        task = {
            key.decode("utf-8"): (
                value.decode("utf-8")
                if key == b"task_id"
                else self._convert_to_original_type(value)
            )
            for key, value in task_data.items()
        }
        return task

    def patch_task(self, task_id: str, **kwargs) -> bool:
        if not kwargs:
            return False

        arguments = []
        for field, value in kwargs.items():
            arguments.extend((field, self._serialize_field(field, value)))

        # Si EXISTS y HSET fueran dos comandos separados, cuando el hilo de publicación
        # en segundo plano y una solicitud de borrado coincidieran, HSET podría volver a
        # crear una tarea incompleta tras el borrado. Redis ejecuta el script Lua de
        # forma atómica, lo que garantiza que no se escriba nada si la tarea no existe y
        # que no se cambien datos fuera de los campos existentes.
        updated = self._redis.eval(
            _PATCH_EXISTING_TASK_SCRIPT,
            1,
            task_id,
            *arguments,
        )
        return bool(updated)

    def delete_task(self, task_id: str):
        # Quien llama todavía limpia el directorio tras leer la tarea, y en ese tiempo
        # otro servicio puede reutilizar una clave con el mismo nombre. Se comprueban el
        # tipo y el marcador de tarea y se borra en una sola operación Lua, para evitar
        # la condición de carrera entre la comprobación y el borrado.
        self._redis.eval(_DELETE_OWNED_TASK_SCRIPT, 1, task_id)

    @staticmethod
    def _serialize_field(field, value):
        # Se entrecomillan las cadenas para que literal_eval no convierta un tema como
        # "2026" o un error como "None" en un entero/None. El marcador de pertenencia se
        # mantiene en bruto: el descubrimiento de tareas compara sus bytes directamente
        # con la clave de Redis.
        if isinstance(value, str) and field != "task_id":
            return repr(value)
        return str(value)

    @staticmethod
    def _convert_to_original_type(value):
        """
        Convierte los valores escritos por esta aplicación de nuevo a tipos comunes de
        Python.

        Este analizador de compatibilidad supone que Redis está dentro del límite de
        confianza de la aplicación. Si clientes no fiables pueden escribir en Redis, el
        estado de las tareas debería pasar a un analizador JSON/esquema estricto en
        lugar de una conversión literal abierta.
        """
        value_str = value.decode("utf-8")

        try:
            # intenta convertir un array de cadenas de bytes en lista
            return ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            pass

        if value_str.isdigit():
            return int(value_str)
        # Añade aquí más conversiones si hace falta
        return value_str


# Estado global
_enable_redis = config.app.get("enable_redis", False)
_redis_host = config.app.get("redis_host", "localhost")
_redis_port = config.app.get("redis_port", 6379)
_redis_db = config.app.get("redis_db", 0)
_redis_password = config.app.get("redis_password", None)

state = (
    RedisState(
        host=_redis_host, port=_redis_port, db=_redis_db, password=_redis_password
    )
    if _enable_redis
    else MemoryState()
)
