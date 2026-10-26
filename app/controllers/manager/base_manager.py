import math
import threading
from typing import Any, Callable, Dict

from loguru import logger


class TaskQueueFullError(ValueError):
    pass


def _coerce_task_limit(value: Any, name: str) -> int:
    """
    Convierte a entero los límites de concurrencia / cola de la configuración.

    Un valor numérico de TOML puede escribirse como `max_concurrent_tasks = 5` o como
    `"5"`. La cadena llegaría hasta la comparación de `add_task` y se convertiría en un
    TypeError que no indica el nombre de la opción; y si `max_queued_tasks` está mal,
    solo salta cuando se agotan las plazas de concurrencia. `app/services/webui_task.py`
    ya hace la misma normalización con la clave `max_queued_tasks`; aquí se sube a la
    única entrada de construcción.

    Cualquier valor que no se pueda convertir a entero informa del nombre de la opción:
    `0.5` lo truncaría `int()` en silencio a 0 (las tareas solo entran en cola y ningún
    worker las ejecuta), y los `inf` / `-inf` / `nan` que permite TOML tampoco son
    límites válidos. 0 y los negativos se mantienen: el primero significa no ejecutar
    tareas por ahora, y los segundos hacen que las solicitudes vayan directamente a la
    rama de cola.
    """
    if isinstance(value, bool):
        # bool es subclase de int, pero `true` claramente no es el límite que quiere el
        # usuario.
        raise ValueError(f"{name} debe ser un entero, se recibió {value!r}")
    if isinstance(value, float):
        # Un valor no finito no es un límite, y un decimal se degradaría tras truncarse
        # al significado de «0: solo encolar sin ejecutar».
        if not math.isfinite(value):
            raise ValueError(f"{name} debe ser un entero finito, se recibió {value!r}")
        if not value.is_integer():
            raise ValueError(f"{name} debe ser un entero, se recibió {value!r}")
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} debe ser un entero, se recibió {value!r}") from None


class TaskManager:
    def __init__(self, max_concurrent_tasks: int, max_queued_tasks: int = 100):
        self.max_concurrent_tasks = _coerce_task_limit(
            max_concurrent_tasks, "max_concurrent_tasks"
        )
        self.max_queued_tasks = _coerce_task_limit(max_queued_tasks, "max_queued_tasks")
        self.current_tasks = 0
        self.lock = threading.Lock()
        self.queue = self.create_queue()

    def create_queue(self):
        raise NotImplementedError()

    def add_task(self, func: Callable, *args: Any, **kwargs: Any):
        with self.lock:
            if self.current_tasks < self.max_concurrent_tasks:
                logger.info(
                    f"se añade la tarea: {func.__name__}, current_tasks: {self.current_tasks}"
                )
                # Reserva la plaza de concurrencia antes de arrancar el hilo. La
                # implementación original incrementaba dentro del hilo, y varias
                # solicitudes seguidas podían ver current_tasks=0 antes de que el hilo
                # hijo obtuviera el bloqueo, superando el límite de concurrencia. Si el
                # arranque falla se devuelve la plaza, para que las solicitudes
                # siguientes se puedan planificar con normalidad.
                self.current_tasks += 1
                try:
                    self.execute_task(func, *args, **kwargs)
                except Exception:
                    self.current_tasks -= 1
                    raise
            else:
                queue_size = self.queue_size()
                # Solo se encola cuando la concurrencia está llena. La cola debe tener
                # un límite; si no, una interfaz anónima podría acumular sin fin objetos
                # de tarea y parámetros de solicitud, hasta agotar la memoria o
                # descontrolar el coste de API de terceros.
                if queue_size >= self.max_queued_tasks:
                    logger.warning(
                        f"se rechaza la tarea: {func.__name__}, queue_size: {queue_size}, "
                        f"max_queued_tasks: {self.max_queued_tasks}"
                    )
                    raise TaskQueueFullError("la cola de tareas está llena, inténtalo de nuevo más tarde")

                logger.info(
                    f"se encola la tarea: {func.__name__}, current_tasks: {self.current_tasks}, "
                    f"queue_size: {queue_size}"
                )
                self.enqueue_new_task({"func": func, "args": args, "kwargs": kwargs})

    def execute_task(self, func: Callable, *args: Any, **kwargs: Any):
        thread = threading.Thread(
            target=self.run_task, args=(func, *args), kwargs=kwargs
        )
        thread.start()

    def run_task(self, func: Callable, *args: Any, **kwargs: Any):
        try:
            func(*args, **kwargs)  # aquí se llama a la función, pasándole *args y **kwargs.
        finally:
            self.task_done()

    def check_queue(self):
        with self.lock:
            self._check_queue_locked()

    def _check_queue_locked(self):
        """Despacha una tarea en cola mientras quien llama tiene ``self.lock``."""
        if (
            self.current_tasks < self.max_concurrent_tasks
            and not self.is_queue_empty()
        ):
            task_info = self.dequeue()
            if task_info is None:
                # dequeue() puede saltarse y descartar entradas de la cola que ya no
                # pasan la validación actual (ver RedisTaskManager.dequeue) y devolver
                # None cuando no queda nada utilizable, aunque is_queue_empty() fuera
                # False un momento antes.
                return
            func = task_info["func"]
            args = task_info.get("args", ())
            kwargs = task_info.get("kwargs", {})
            # Reserva la plaza liberada antes de que otro add_task pueda ocuparla.
            self.current_tasks += 1
            try:
                self.execute_task(func, *args, **kwargs)
            except Exception:
                self.current_tasks -= 1
                self.enqueue(task_info)
                raise

    def task_done(self):
        with self.lock:
            self.current_tasks -= 1
            self._check_queue_locked()

    def enqueue(self, task: Dict):
        raise NotImplementedError()

    def enqueue_new_task(self, task: Dict):
        """Admite trabajo nuevo; las colas compartidas pueden necesitar su propia
        comprobación atómica del límite.
        """
        self.enqueue(task)

    def dequeue(self):
        raise NotImplementedError()

    def is_queue_empty(self):
        raise NotImplementedError()

    def queue_size(self):
        raise NotImplementedError()
