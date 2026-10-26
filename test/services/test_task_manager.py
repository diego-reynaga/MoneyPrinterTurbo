import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from app.controllers.manager.base_manager import TaskQueueFullError
from app.controllers.manager.memory_manager import InMemoryTaskManager
from app.controllers.manager.redis_manager import RedisTaskManager
from app.models import const
from app.models.schema import VideoParams
from app.services import task as task_service


def _queued_payload(func: str, **kwargs) -> str:
    """Construye una entrada de cola con el formato en disco de RedisTaskManager.enqueue."""
    return json.dumps({"func": func, "args": [], "kwargs": kwargs})


def _video_params() -> dict:
    """Devuelve una serialización de VideoParams que pasa la validación actual."""
    return VideoParams(video_subject="Tea").model_dump(warnings=False)


class TestInMemoryTaskManager(unittest.TestCase):
    def test_queue_operations_preserve_task_payload(self):
        """La cola en memoria debe conservar la función, los argumentos posicionales y los
        de palabra clave, sin cambiar el contenido de la tarea.
        """
        manager = InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks=2)
        task = {"func": len, "args": ([1, 2],), "kwargs": {}}

        manager.enqueue(task)

        self.assertFalse(manager.is_queue_empty())
        self.assertEqual(manager.queue_size(), 1)
        self.assertEqual(manager.dequeue(), task)
        self.assertTrue(manager.is_queue_empty())

    def test_add_task_rejects_only_after_queue_limit(self):
        """Cuando se agotan las plazas de concurrencia se permite encolar hasta el límite;
        solo al superarlo se devuelve un error claro.
        """
        manager = InMemoryTaskManager(max_concurrent_tasks=0, max_queued_tasks=1)

        manager.add_task(len, [1])

        with self.assertRaises(TaskQueueFullError):
            manager.add_task(len, [2])

    def test_add_task_reserves_slot_before_background_thread_runs(self):
        """
        La plaza de concurrencia debe reservarse antes de arrancar el hilo; aunque el
        hilo simulado aún no haya entrado en run_task, la segunda solicitud debe ir a la
        cola y no puede superar max_concurrent_tasks.
        """
        manager = InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks=1)

        with patch.object(manager, "execute_task") as execute_task:
            manager.add_task(len, [1])
            manager.add_task(len, [2])

        self.assertEqual(manager.current_tasks, 1)
        execute_task.assert_called_once_with(len, [1])
        self.assertEqual(manager.queue_size(), 1)

    def test_add_task_rolls_back_slot_when_thread_cannot_start(self):
        """Un fallo al arrancar el hilo no puede ocupar para siempre una plaza de
        concurrencia, y la excepción debe seguir llegando a quien llama.
        """
        manager = InMemoryTaskManager(max_concurrent_tasks=1)

        with patch.object(
            manager,
            "execute_task",
            side_effect=RuntimeError("thread unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "thread unavailable"):
                manager.add_task(len, [1])

        self.assertEqual(manager.current_tasks, 0)

    def test_task_done_starts_next_queued_task(self):
        """Al terminar la tarea actual debe liberarse la plaza de concurrencia y
        planificarse de inmediato la siguiente tarea de la cola.
        """
        manager = InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks=2)
        manager.current_tasks = 1
        manager.enqueue({"func": len, "args": ([1, 2],), "kwargs": {}})

        with patch.object(manager, "execute_task") as execute_task:
            manager.task_done()

        self.assertEqual(manager.current_tasks, 1)
        execute_task.assert_called_once_with(len, [1, 2])
        self.assertTrue(manager.is_queue_empty())

    def test_new_request_cannot_take_slot_before_waiting_task(self):
        """Una finalización debe reservar la plaza liberada para la tarea más antigua de la
        cola.
        """
        manager = InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks=2)
        manager.current_tasks = 1

        def queued_task():
            pass

        def incoming_task():
            pass

        manager.enqueue({"func": queued_task, "args": (), "kwargs": {}})
        released = threading.Event()
        resume = threading.Event()

        class PausingLock:
            def __init__(self):
                self.lock = threading.Lock()
                self.worker = None
                self.paused = False

            def __enter__(self):
                self.lock.acquire()
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self.lock.release()
                if threading.current_thread() is self.worker and not self.paused:
                    self.paused = True
                    released.set()
                    resume.wait(timeout=2)

        manager.lock = PausingLock()
        started = []
        with patch.object(manager, "execute_task", side_effect=lambda func: started.append(func)):
            worker = threading.Thread(target=manager.task_done)
            manager.lock.worker = worker
            worker.start()
            try:
                self.assertTrue(released.wait(timeout=2))
                manager.add_task(incoming_task)
            finally:
                resume.set()
                worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(started, [queued_task])
        self.assertEqual(manager.dequeue()["func"], incoming_task)

    def test_task_done_requeues_task_when_thread_cannot_start(self):
        """Si tras sacar de la cola falla el arranque del hilo, hay que devolver la plaza y
        volver a poner la tarea en la cola, para no perderla.
        """
        manager = InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks=1)
        manager.current_tasks = 1
        queued_task = {"func": len, "args": ([1, 2],), "kwargs": {}}
        manager.enqueue(queued_task)

        with patch.object(
            manager,
            "execute_task",
            side_effect=RuntimeError("thread unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "thread unavailable"):
                manager.task_done()

        self.assertEqual(manager.current_tasks, 0)
        self.assertEqual(manager.dequeue(), queued_task)

    def test_run_task_releases_slot_after_failure(self):
        """Cuando la función de la tarea lanza una excepción, el finally debe liberar
        igualmente la plaza, para que la cola no se bloquee para siempre.
        """
        manager = InMemoryTaskManager(max_concurrent_tasks=1)
        manager.current_tasks = 1

        with patch.object(manager, "task_done") as task_done:
            with self.assertRaisesRegex(RuntimeError, "task failed"):
                manager.run_task(MagicMock(side_effect=RuntimeError("task failed")))

        self.assertEqual(manager.current_tasks, 1)
        task_done.assert_called_once_with()

    def test_check_queue_handles_dequeue_returning_none(self):
        """
        dequeue() puede saltarse internamente todas las tareas en cola que ya no cumplen
        la validación actual y devolver None, aunque is_queue_empty() fuera False antes
        de llamar a check_queue. check_queue no puede suponer que dequeue siempre
        obtiene una tarea utilizable, o fallaría en task_info["func"].
        """
        manager = InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks=1)

        with patch.object(manager, "is_queue_empty", return_value=False), patch.object(
            manager, "dequeue", return_value=None
        ), patch.object(manager, "execute_task") as execute_task:
            manager.check_queue()

        execute_task.assert_not_called()
        self.assertEqual(manager.current_tasks, 0)

    def test_execute_task_starts_background_thread(self):
        """La entrada de ejecución de tareas debe arrancar el hilo y pasar todos los
        argumentos de la función a run_task.
        """
        manager = InMemoryTaskManager(max_concurrent_tasks=1)
        fake_thread = MagicMock()

        with patch(
            "app.controllers.manager.base_manager.threading.Thread",
            return_value=fake_thread,
        ) as thread:
            manager.execute_task(len, [1, 2])

        thread.assert_called_once_with(
            target=manager.run_task,
            args=(len, [1, 2]),
            kwargs={},
        )
        fake_thread.start.assert_called_once_with()

    def test_limits_accept_quoted_toml_integers(self):
        """
        Cuando en TOML el límite se escribe como cadena también debe compararse como
        entero; antes de la corrección, la comparación de plazas de concurrencia lanzaba
        TypeError, y la del límite de la cola solo saltaba al agotarse las plazas.
        """
        manager = InMemoryTaskManager(max_concurrent_tasks="1", max_queued_tasks="1")

        with patch.object(manager, "execute_task") as execute_task:
            manager.add_task(len, [1])
            manager.add_task(len, [2])
            with self.assertRaises(TaskQueueFullError):
                manager.add_task(len, [3])

        self.assertEqual(manager.max_concurrent_tasks, 1)
        self.assertEqual(manager.max_queued_tasks, 1)
        execute_task.assert_called_once_with(len, [1])
        self.assertEqual(manager.queue_size(), 1)

    def test_invalid_limits_raise_a_named_error(self):
        """Un límite que no se puede analizar debe indicar de inmediato el nombre de la
        opción, sin aplazarse a una excepción anónima durante la planificación.
        """
        with self.assertRaisesRegex(ValueError, "max_concurrent_tasks"):
            InMemoryTaskManager(max_concurrent_tasks="abc")

        with self.assertRaisesRegex(ValueError, "max_queued_tasks"):
            InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks="abc")

        with self.assertRaisesRegex(ValueError, "max_queued_tasks"):
            InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks=True)

    def test_non_integral_limits_are_rejected_by_name(self):
        """Antes de la corrección, `int()` truncaba `0.5` a 0 (las tareas entraban en la
        cola y ningún worker las ejecutaba), y `inf` dejaba escapar un OverflowError.
        """
        cases = (
            (0.5, "un entero"),
            (1.5, "un entero"),
            ("0.5", "un entero"),
            (float("nan"), "un entero finito"),
            (float("inf"), "un entero finito"),
            (float("-inf"), "un entero finito"),
        )
        for field in ("max_concurrent_tasks", "max_queued_tasks"):
            for value, msg in cases:
                with self.subTest(field=field, value=value):
                    limits = {"max_concurrent_tasks": 1, "max_queued_tasks": 1}
                    limits[field] = value
                    pattern = f"{field} debe ser {msg}"
                    with self.assertRaisesRegex(ValueError, pattern):
                        InMemoryTaskManager(**limits)

        # Los float de valor entero sin riesgo de truncamiento siguen siendo válidos
        # (TOML permite `max_queued_tasks = 2.0`).
        integral = {"max_concurrent_tasks": 1, "max_queued_tasks": 2.0}
        self.assertEqual(InMemoryTaskManager(**integral).max_queued_tasks, 2)

    def test_zero_concurrency_keeps_queueing_without_executing(self):
        """0 y los negativos siguen siendo valores válidos: solo encolar, sin ejecutar;
        casos existentes dependen de ese significado.
        """
        manager = InMemoryTaskManager(max_concurrent_tasks=0, max_queued_tasks=1)

        with patch.object(manager, "execute_task") as execute_task:
            manager.add_task(len, [1])

        execute_task.assert_not_called()
        self.assertEqual(manager.queue_size(), 1)


class TestRedisTaskManager(unittest.TestCase):
    def setUp(self):
        self.redis_client = MagicMock()
        patcher = patch(
            "app.controllers.manager.redis_manager.redis.Redis.from_url",
            return_value=self.redis_client,
        )
        self.addCleanup(patcher.stop)
        from_url = patcher.start()
        self.manager = RedisTaskManager(
            max_concurrent_tasks=1,
            redis_url="redis://localhost:6379/0",
            max_queued_tasks=3,
        )
        from_url.assert_called_once_with("redis://localhost:6379/0")

    def test_resume_queued_tasks_fills_available_slots_after_restart(self):
        """Las entradas de Redis persistidas deben ejecutarse sin esperar a una nueva
        solicitud a la API.
        """
        self.manager.max_concurrent_tasks = 2
        self.redis_client.llen.return_value = 2
        self.redis_client.lpop.side_effect = [
            _queued_payload("start", task_id="first", params=_video_params()),
            _queued_payload("start", task_id="second", params=_video_params()),
        ]

        with patch.object(self.manager, "execute_task") as execute_task:
            self.manager.resume_queued_tasks()

        self.assertEqual(execute_task.call_count, 2)
        self.assertEqual(self.manager.current_tasks, 2)
        self.assertEqual(self.redis_client.lpop.call_count, 2)

    def test_enqueue_serializes_video_params_without_mutating_task(self):
        """
        Redis solo puede guardar JSON; VideoParams debe convertirse a diccionario, pero
        la tarea original debe conservar el modelo, para que los efectos secundarios de
        la serialización no afecten a los logs, los reintentos ni las lecturas
        posteriores de quien llama.
        """
        params = VideoParams(video_subject="Coffee")
        task = {
            "func": task_service.start,
            "args": (),
            "kwargs": {"task_id": "task-1", "params": params},
        }

        self.manager.enqueue(task)

        self.assertIs(task["kwargs"]["params"], params)
        queue_name, payload = self.redis_client.rpush.call_args.args
        decoded = json.loads(payload)
        self.assertEqual(queue_name, "task_queue")
        self.assertEqual(decoded["func"], "start")
        self.assertEqual(decoded["kwargs"]["task_id"], "task-1")
        self.assertEqual(decoded["kwargs"]["params"]["video_subject"], "Coffee")

    def test_dequeue_restores_function_and_video_params(self):
        """Una tarea sacada de Redis debe recuperar la función invocable y el modelo
        VideoParams.
        """
        payload = {
            "func": "start",
            "args": [],
            "kwargs": {
                "task_id": "task-1",
                "params": VideoParams(video_subject="Coffee").model_dump(
                    warnings=False
                ),
            },
        }
        self.redis_client.lpop.return_value = json.dumps(payload)

        task = self.manager.dequeue()

        self.redis_client.lpop.assert_called_once_with("task_queue")
        self.assertIs(task["func"], task_service.start)
        self.assertIsInstance(task["kwargs"]["params"], VideoParams)
        self.assertEqual(task["kwargs"]["params"].video_subject, "Coffee")

    def test_empty_queue_and_size_use_redis_length(self):
        """La comprobación de cola vacía y su longitud deben reflejar directamente la
        longitud actual de la lista en Redis.
        """
        self.redis_client.lpop.return_value = None
        self.redis_client.llen.side_effect = [0, 2]

        self.assertIsNone(self.manager.dequeue())
        self.assertTrue(self.manager.is_queue_empty())
        self.assertEqual(self.manager.queue_size(), 2)

    def test_dequeue_skips_task_that_fails_current_validation(self):
        """
        Una tarea puede haberse encolado antes de endurecer las reglas de validación
        (por ejemplo, cuando video_count admitía 0). lpop es destructivo: si falla
        reconstruir VideoParams, la tarea ya se ha eliminado de Redis para siempre y no
        se puede fingir que sigue ahí; dequeue no debe lanzar la excepción de validación
        a quien llama (eso haría fallar a quien tiene el bloqueo y perdería la tarea sin
        dejar log), sino saltársela y seguir con la siguiente de la cola, hasta obtener
        una tarea utilizable o que la cola esté realmente vacía.
        """
        stale_payload = {
            "func": "start",
            "args": [],
            "kwargs": {
                "task_id": "task-stale",
                "params": {**VideoParams(video_subject="Coffee").model_dump(
                    warnings=False
                ), "video_count": 0},
            },
        }
        valid_payload = {
            "func": "start",
            "args": [],
            "kwargs": {
                "task_id": "task-valid",
                "params": VideoParams(video_subject="Tea").model_dump(
                    warnings=False
                ),
            },
        }
        self.redis_client.lpop.side_effect = [
            json.dumps(stale_payload),
            json.dumps(valid_payload),
        ]

        task = self.manager.dequeue()

        self.assertEqual(self.redis_client.lpop.call_count, 2)
        self.assertEqual(task["kwargs"]["task_id"], "task-valid")
        self.assertIsInstance(task["kwargs"]["params"], VideoParams)
        self.assertEqual(task["kwargs"]["params"].video_subject, "Tea")

    def test_dequeue_returns_none_when_every_queued_task_is_stale(self):
        """Cuando todas las tareas restantes se descartan por las reglas de validación
        actuales, debe devolverse None en lugar de lanzar una excepción.
        """
        stale_payload = {
            "func": "start",
            "args": [],
            "kwargs": {
                "task_id": "task-stale",
                "params": {**VideoParams(video_subject="Coffee").model_dump(
                    warnings=False
                ), "video_count": -1},
            },
        }
        self.redis_client.lpop.side_effect = [json.dumps(stale_payload), None]

        self.assertIsNone(self.manager.dequeue())
        self.assertEqual(self.redis_client.lpop.call_count, 2)

    def test_dequeue_marks_stale_task_failed_instead_of_leaving_it_processing(self):
        """
        El registro de estado de la tarea se crea antes de encolar y por defecto está en
        processing. Si dequeue solo se salta y descarta la entrada de la cola sin
        actualizar el estado, la tarea aparecería para siempre en ejecución en la
        API/WebUI. Debe marcarse como fallida con patch_task (y no con update_task),
        para no volver a crear su registro de estado si el usuario ya la ha borrado.
        """
        stale_payload = {
            "func": "start",
            "args": [],
            "kwargs": {
                "task_id": "task-stale",
                "params": {**VideoParams(video_subject="Coffee").model_dump(
                    warnings=False
                ), "video_count": 0},
            },
        }
        self.redis_client.lpop.side_effect = [json.dumps(stale_payload), None]

        with patch("app.controllers.manager.redis_manager.sm.state") as state:
            state.patch_task.return_value = True
            result = self.manager.dequeue()

        self.assertIsNone(result)
        state.patch_task.assert_called_once()
        call_args = state.patch_task.call_args
        self.assertEqual(call_args.args[0], "task-stale")
        self.assertEqual(call_args.kwargs["state"], const.TASK_STATE_FAILED)
        self.assertEqual(call_args.kwargs["failed_stage"], "dequeue")
        self.assertIn("video_count", call_args.kwargs["error"])

    def test_dequeue_does_not_recreate_state_for_already_deleted_task(self):
        """patch_task devuelve False si la tarea ya se ha borrado; dequeue no debe tratarlo
        como error.
        """
        stale_payload = {
            "func": "start",
            "args": [],
            "kwargs": {
                "task_id": "task-deleted",
                "params": {**VideoParams(video_subject="Coffee").model_dump(
                    warnings=False
                ), "video_count": 0},
            },
        }
        self.redis_client.lpop.side_effect = [json.dumps(stale_payload), None]

        with patch("app.controllers.manager.redis_manager.sm.state") as state:
            state.patch_task.return_value = False
            result = self.manager.dequeue()

        self.assertIsNone(result)
        state.patch_task.assert_called_once()
        state.update_task.assert_not_called()

    def test_dequeue_skips_task_with_unknown_function_name(self):
        """
        Los miembros de FUNC_MAP cambian con los despliegues (este mismo archivo
        conserva una segunda entrada comentada), así que en la cola pueden quedar tareas
        de funciones de entrada antiguas. Indexar directamente un nombre que no está en
        FUNC_MAP lanza KeyError y se salta la política de descarte de dequeue, así que
        hay que saltarlo igual que en la ruta de fallo de validación.
        """
        self.redis_client.lpop.side_effect = [
            _queued_payload("start_test", task_id="task-unknown-func"),
            _queued_payload("start", task_id="task-valid", params=_video_params()),
        ]

        with patch("app.controllers.manager.redis_manager.sm.state") as state:
            state.patch_task.return_value = True
            task = self.manager.dequeue()

        self.assertEqual(self.redis_client.lpop.call_count, 2)
        self.assertIs(task["func"], task_service.start)
        self.assertEqual(task["kwargs"]["task_id"], "task-valid")
        state.patch_task.assert_called_once()
        call_args = state.patch_task.call_args
        self.assertEqual(call_args.args[0], "task-unknown-func")
        self.assertEqual(call_args.kwargs["state"], const.TASK_STATE_FAILED)
        self.assertIn("start_test", call_args.kwargs["error"])

    def test_dequeue_skips_every_unusable_payload_shape(self):
        """
        En la cola pueden quedar tres tipos de entradas inutilizables: JSON truncado al
        escribirse, JSON que no es un objeto, y tareas que apuntan a una función de
        entrada eliminada (en FUNC_MAP solo queda el comentario). Ninguna puede hacer
        que dequeue lance una excepción, o las tareas válidas que van detrás no se
        planificarían nunca; las entradas de las que se puede leer task_id deben además
        cerrar como fallida la tarea que ya ha salido para siempre de la cola.
        """
        self.redis_client.lpop.side_effect = [
            '{"func": "start", "args": [',
            json.dumps(["not", "a", "mapping"]),
            _queued_payload("retired-entry", task_id="task-retired"),
            _queued_payload("start", task_id="task-valid", params=_video_params()),
        ]

        with patch("app.controllers.manager.redis_manager.sm.state") as state:
            state.patch_task.return_value = True
            task = self.manager.dequeue()

        self.assertEqual(self.redis_client.lpop.call_count, 4)
        self.assertIs(task["func"], task_service.start)
        self.assertEqual(task["kwargs"]["task_id"], "task-valid")
        state.patch_task.assert_called_once()
        self.assertEqual(state.patch_task.call_args.args[0], "task-retired")
        self.assertEqual(
            state.patch_task.call_args.kwargs["state"], const.TASK_STATE_FAILED
        )

    def test_dequeue_returns_none_when_only_unusable_entries_remain(self):
        """Cuando todas las entradas restantes son inutilizables, debe devolverse None en
        lugar de lanzar una excepción.
        """
        self.redis_client.lpop.side_effect = [
            _queued_payload("retired-entry"),
            None,
        ]

        with patch("app.controllers.manager.redis_manager.sm.state") as state:
            result = self.manager.dequeue()

        self.assertIsNone(result)
        self.assertEqual(self.redis_client.lpop.call_count, 2)
        state.patch_task.assert_not_called()

    def test_task_done_keeps_draining_queue_when_entry_is_unusable(self):
        """
        Cuando una tarea termina y libera su plaza, task_done llama a check_queue para
        planificar la siguiente. Si una entrada inutilizable hiciera que check_queue
        lanzara una excepción, esta tumbaría el hilo de trabajo a través del finally de
        run_task; a partir de ahí no habría tareas en marcha, nadie volvería a llamar a
        check_queue y las tareas siguientes de la cola se quedarían para siempre en
        processing. Por eso task_done debe poder seguir planificando la siguiente tarea
        utilizable en lugar de dejar toda la cola parada.
        """
        self.manager.current_tasks = 1
        self.redis_client.lpop.side_effect = [
            _queued_payload("retired-entry"),
            _queued_payload("start", task_id="task-next", params=_video_params()),
        ]

        with patch.object(self.manager, "execute_task") as execute_task:
            self.manager.task_done()

        self.assertEqual(self.manager.current_tasks, 1)
        self.assertEqual(execute_task.call_args.kwargs["task_id"], "task-next")
        self.assertIs(execute_task.call_args.args[0], task_service.start)

    def test_dequeue_discards_payloads_that_cannot_be_dispatched(self):
        """
        Tres tipos de entradas que se pueden analizar pero no despachar: lpop devuelve
        un valor vacío, o args se escribe como null / objeto / número / cadena. Ninguna
        puede hacer que dequeue lance una excepción ni tratarse como "cola vacía"; si
        no, check_queue volvería a encolar o saldría antes, y las tareas válidas que van
        detrás nunca se planificarían.
        """
        cases = (
            ("empty bytes", b"", None),
            ("empty string", "", None),
            ("args null", {"args": None}, "t-1"),
            ("args mapping", {"args": {}}, "t-2"),
            ("args number", {"args": 5}, "t-3"),
            ("args string", {"args": "x"}, "t-4"),
        )

        for case, broken, expected_task_id in cases:
            with self.subTest(case=case):
                if isinstance(broken, dict):
                    broken = json.dumps(
                        {
                            "func": "start",
                            "kwargs": {"task_id": expected_task_id},
                            **broken,
                        }
                    )
                self.redis_client.reset_mock()
                self.redis_client.lpop.side_effect = [
                    broken,
                    _queued_payload(
                        "start", task_id="task-valid", params=_video_params()
                    ),
                ]

                with patch("app.controllers.manager.redis_manager.sm.state") as state:
                    state.patch_task.return_value = True
                    task = self.manager.dequeue()

                self.assertEqual(self.redis_client.lpop.call_count, 2)
                self.assertEqual(task["kwargs"]["task_id"], "task-valid")
                self.assertIs(task["func"], task_service.start)
                if expected_task_id is None:
                    state.patch_task.assert_not_called()
                else:
                    self.assertEqual(
                        state.patch_task.call_args.args[0], expected_task_id
                    )
                    self.assertIn(
                        "argumentos posicionales",
                        state.patch_task.call_args.kwargs["error"],
                    )

    def test_dequeue_dispatches_payload_without_args_field(self):
        """Que falte el campo args entero es un formato existente (check_queue toma por
        defecto una tupla vacía) y no puede descartarse.
        """
        self.redis_client.lpop.return_value = json.dumps(
            {
                "func": "start",
                "kwargs": {"task_id": "task-no-args", "params": _video_params()},
            }
        )

        with patch("app.controllers.manager.redis_manager.sm.state") as state:
            task = self.manager.dequeue()

        self.assertEqual(task["kwargs"]["task_id"], "task-no-args")
        self.assertIs(task["func"], task_service.start)
        state.patch_task.assert_not_called()

    def test_dequeue_ignores_non_string_task_id_in_every_discard_path(self):
        """
        Cuando task_id es un valor que no es cadena, como un array u objeto JSON,
        pasarlo a patch_task haría que redis lanzara DataError, rompiendo el bucle de
        descarte en una entrada dañada e impidiendo que arranquen las tareas válidas que
        van detrás. Ambas rutas de descarte deben omitir la actualización del estado y
        descartar solo la propia entrada.
        """
        cases = (
            ("unknown function", _queued_payload("retired-entry", task_id=["task-1"])),
            (
                "stale params",
                _queued_payload(
                    "start",
                    task_id={"nested": "task-2"},
                    params={**_video_params(), "video_count": 0},
                ),
            ),
        )

        for case, payload in cases:
            with self.subTest(case=case):
                self.redis_client.reset_mock()
                self.redis_client.lpop.side_effect = [
                    payload,
                    _queued_payload(
                        "start", task_id="task-valid", params=_video_params()
                    ),
                ]

                with patch("app.controllers.manager.redis_manager.sm.state") as state:
                    task = self.manager.dequeue()

                self.assertEqual(self.redis_client.lpop.call_count, 2)
                self.assertEqual(task["kwargs"]["task_id"], "task-valid")
                state.patch_task.assert_not_called()
                state.update_task.assert_not_called()


if __name__ == "__main__":
    unittest.main()
