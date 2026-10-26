import os
import sys
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models import const
from app.services.state import MemoryState, RedisState


class _FakeRedisPipeline:
    def __init__(self, redis):
        self.redis = redis
        self.keys = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def hget(self, key, field):
        self.keys.append((key, field))
        return self

    def execute(self, raise_on_error=True):
        return [self.redis.data.get(key, {}).get(field.encode("utf-8")) for key, field in self.keys]


class _FakeRedis:
    def __init__(self, batches):
        self.batches = batches
        self.scan_types = []
        self.data = {}
        for key in [key for batch in batches for key in batch]:
            index = int(key.decode("utf-8").split(":")[-1])
            self.data[key] = {
                b"task_id": key,
                b"state": b"1",
                b"progress": str(index).encode("utf-8"),
            }

    def scan(self, cursor, count, _type=None):
        self.scan_types.append(_type)
        batch_index = int(cursor)
        next_cursor = batch_index + 1
        if next_cursor >= len(self.batches):
            next_cursor = 0
        return next_cursor, self.batches[batch_index]

    def hgetall(self, key):
        if isinstance(key, str):
            key = key.encode("utf-8")
        return self.data[key]

    def pipeline(self, transaction=False):
        return _FakeRedisPipeline(self)

    def exists(self, key):
        if isinstance(key, str):
            key = key.encode("utf-8")
        return key in self.data

    def hset(self, key, field=None, value=None, mapping=None):
        if isinstance(key, str):
            key = key.encode("utf-8")
        target = self.data.setdefault(key, {})
        if mapping:
            target.update(
                {
                    str(item_key).encode("utf-8"): str(item_value).encode("utf-8")
                    for item_key, item_value in mapping.items()
                }
            )
        elif field is not None:
            target[str(field).encode("utf-8")] = str(value).encode("utf-8")

    def eval(self, script, numkeys, key, *arguments):
        if isinstance(key, str):
            key = key.encode("utf-8")
        if key not in self.data:
            return 0

        target = self.data[key]
        for index in range(0, len(arguments), 2):
            field = str(arguments[index]).encode("utf-8")
            value = str(arguments[index + 1]).encode("utf-8")
            target[field] = value
        return 1


class TestMemoryState(unittest.TestCase):
    def test_progress_update_preserves_existing_task_details(self):
        state = MemoryState()
        state.update_task(
            "task-1",
            state=const.TASK_STATE_PROCESSING,
            video_subject="A day in Shanghai",
            material_sources=["source.mp4"],
        )

        state.update_task("task-1", progress=25)

        task = state.get_task("task-1")
        self.assertEqual(task["progress"], 25)
        self.assertEqual(task["video_subject"], "A day in Shanghai")
        self.assertEqual(task["material_sources"], ["source.mp4"])

    def test_get_task_and_get_all_tasks_return_isolated_snapshots(self):
        state = MemoryState()
        state.update_task(
            "task-1",
            state=const.TASK_STATE_PROCESSING,
            progress=25,
            videos=["first.mp4"],
        )

        task = state.get_task("task-1")
        task["videos"].append("mutated.mp4")

        tasks, total = state.get_all_tasks(page=1, page_size=10)
        tasks[0]["videos"].append("mutated-again.mp4")

        self.assertEqual(total, 1)
        self.assertEqual(state.get_task("task-1")["videos"], ["first.mp4"])

    def test_get_all_tasks_copies_only_the_requested_page(self):
        """Una página pequeña no debe clonar la carga útil de todas las tareas históricas."""

        class CopyTracked:
            def __init__(self):
                self.copies = 0

            def __deepcopy__(self, memo):
                self.copies += 1
                return CopyTracked()

        state = MemoryState()
        off_page = CopyTracked()
        state.update_task("task-1", videos=["first.mp4"])
        state.update_task("task-2", payload=off_page)
        # Los valores de entrada se copian al actualizar; se rastrea la carga útil
        # guardada para que este test mida solo las copias que hace la paginación.
        off_page = state._tasks["task-2"]["payload"]

        first_page, total = state.get_all_tasks(page=1, page_size=1)

        self.assertEqual(total, 2)
        self.assertEqual([task["task_id"] for task in first_page], ["task-1"])
        self.assertEqual(off_page.copies, 0)

        second_page, _ = state.get_all_tasks(page=2, page_size=1)
        self.assertEqual([task["task_id"] for task in second_page], ["task-2"])
        self.assertEqual(off_page.copies, 1)

    def test_concurrent_memory_updates_are_preserved(self):
        state = MemoryState()
        thread_count = 5
        tasks_per_thread = 50

        def update_tasks(thread_index):
            for task_index in range(tasks_per_thread):
                state.update_task(
                    f"task-{thread_index}-{task_index}",
                    state=const.TASK_STATE_PROCESSING,
                    progress=task_index,
                )

        threads = [
            threading.Thread(target=update_tasks, args=(thread_index,))
            for thread_index in range(thread_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        tasks, total = state.get_all_tasks(page=1, page_size=thread_count * tasks_per_thread)

        self.assertEqual(total, thread_count * tasks_per_thread)
        self.assertEqual(len(tasks), total)

    def test_patch_task_preserves_generated_outputs(self):
        """Las actualizaciones de publicación asíncronas no deben sobrescribir los campos
        de una tarea de vídeo ya completada.
        """
        state = MemoryState()
        state.update_task(
            "task-1",
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
        )

        patched = state.patch_task(
            "task-1",
            cross_post_state=const.CROSS_POST_STATE_COMPLETE,
            cross_post_results=[{"success": True}],
        )

        self.assertTrue(patched)
        self.assertEqual(
            state.get_task("task-1"),
            {
                "task_id": "task-1",
                "state": const.TASK_STATE_COMPLETE,
                "progress": 100,
                "videos": ["final.mp4"],
                "cross_post_state": const.CROSS_POST_STATE_COMPLETE,
                "cross_post_results": [{"success": True}],
            },
        )
        self.assertFalse(state.patch_task("missing", value="ignored"))


class TestRedisState(unittest.TestCase):
    def test_update_task_writes_all_fields_in_one_redis_command(self):
        state = RedisState.__new__(RedisState)
        state._redis = Mock()

        state.update_task(
            "task-1",
            state=const.TASK_STATE_COMPLETE,
            progress=120,
            videos=["final.mp4"],
        )

        state._redis.hset.assert_called_once_with(
            "task-1",
            mapping={
                "task_id": "task-1",
                "state": str(const.TASK_STATE_COMPLETE),
                "progress": "100",
                "videos": "['final.mp4']",
            },
        )

    def _build_state(self, batch_sizes):
        keys = [f"task:{i}".encode("utf-8") for i in range(sum(batch_sizes))]
        batches = []
        offset = 0
        for batch_size in batch_sizes:
            batches.append(keys[offset : offset + batch_size])
            offset += batch_size

        state = RedisState.__new__(RedisState)
        state._redis = _FakeRedis(batches)
        return state

    def test_get_all_tasks_paginates_across_scan_batches(self):
        """
        Cuando Redis SCAN devuelve las claves por lotes, la paginación debe cortar según
        un orden estable de las claves de tarea.

        Este caso reproduce el escenario del PR #890: 18 tareas con page_size=10; el
        primer lote trae 10 y el segundo 8, y entre las dos páginas deben cubrirse todas
        las tareas.
        """
        state = self._build_state([10, 8])

        first_page, first_total = state.get_all_tasks(page=1, page_size=10)
        second_page, second_total = state.get_all_tasks(page=2, page_size=10)

        self.assertEqual(first_total, 18)
        self.assertEqual(second_total, 18)
        self.assertEqual(len(first_page), 10)
        self.assertEqual(len(second_page), 8)
        expected_ids = sorted(f"task:{i}" for i in range(18))
        self.assertEqual(
            [task["task_id"] for task in first_page],
            expected_ids[:10],
        )
        self.assertEqual(
            [task["task_id"] for task in second_page],
            expected_ids[10:],
        )
        self.assertTrue(state._redis.scan_types)
        self.assertEqual(set(state._redis.scan_types), {"HASH"})

    def test_get_all_tasks_deduplicates_and_stabilizes_scan_order(self):
        """Aunque dos escaneos independientes tengan distinto orden o un mismo escaneo
        repita claves, no debe haber duplicados ni omisiones.
        """
        state = self._build_state([3])
        state._redis.batches = [
            [b"task:1", b"task:0"],
            [],
            [b"task:2", b"task:1"],
        ]
        first_page, first_total = state.get_all_tasks(page=1, page_size=2)

        state._redis.batches = [
            [b"task:2", b"task:1"],
            [b"task:0", b"task:2"],
        ]
        second_page, second_total = state.get_all_tasks(page=2, page_size=2)

        self.assertEqual(first_total, 3)
        self.assertEqual(second_total, 3)
        self.assertEqual(
            [task["task_id"] for task in first_page + second_page],
            ["task:0", "task:1", "task:2"],
        )
        self.assertEqual(state.list_task_ids(scan_count=1), ["task:0", "task:1", "task:2"])

    def test_shared_redis_db_does_not_expose_unrelated_hashes(self):
        """Solo los hashes cuyo task_id incrustado coincide con la clave pertenecen a esta
        aplicación.
        """
        state = self._build_state([3])
        state._redis.data[b"task:1"] = {b"secret": b"another service's token"}
        state._redis.data[b"task:2"] = {
            b"task_id": b"different-task",
            b"secret": b"another service's token",
        }

        self.assertIsNone(state.get_task("task:1"))
        self.assertIsNone(state.get_task("task:2"))
        self.assertEqual(state.list_task_ids(), ["task:0"])
        tasks, total = state.get_all_tasks(page=1, page_size=10)
        self.assertEqual(total, 1)
        self.assertEqual([task["task_id"] for task in tasks], ["task:0"])

    @unittest.skipUnless(
        os.getenv("MPT_TEST_REDIS_HOST"),
        "MPT_TEST_REDIS_HOST not set",
    )
    def test_real_redis_get_all_tasks_ignores_queue_keys(self):
        """Las colas List del Redis real no deben leerse por error como Hash en la lista de
        tareas.
        """
        state = RedisState(
            host=os.environ["MPT_TEST_REDIS_HOST"],
            port=int(os.getenv("MPT_TEST_REDIS_PORT", "6379")),
            db=int(os.getenv("MPT_TEST_REDIS_DB", "15")),
        )
        suffix = uuid.uuid4()
        task_ids = [f"ci-list-{suffix}-{index}" for index in range(3)]
        queue_key = f"ci-queue-{suffix}"

        try:
            for task_id in task_ids:
                state.update_task(
                    task_id,
                    state=const.TASK_STATE_COMPLETE,
                    progress=100,
                )
            state._redis.rpush(queue_key, *task_ids)

            tasks, _ = state.get_all_tasks(page=1, page_size=1000)
            returned_ids = {task["task_id"] for task in tasks}

            self.assertTrue(set(task_ids).issubset(returned_ids))
            self.assertNotIn(queue_key, returned_ids)
        finally:
            state._redis.delete(queue_key, *task_ids)

    @unittest.skipUnless(
        os.getenv("MPT_TEST_REDIS_HOST"),
        "MPT_TEST_REDIS_HOST not set",
    )
    def test_real_redis_pagination_keeps_a_stable_task_order(self):
        """Un SCAN de varios lotes en el Redis real no debe duplicar ni perder tareas entre
        páginas.
        """
        state = RedisState(
            host=os.environ["MPT_TEST_REDIS_HOST"],
            port=int(os.getenv("MPT_TEST_REDIS_PORT", "6379")),
            db=int(os.getenv("MPT_TEST_REDIS_DB", "15")),
        )
        task_ids = [f"ci-page-{uuid.uuid4()}-{index}" for index in range(13)]
        try:
            for task_id in reversed(task_ids):
                state.update_task(task_id, state=const.TASK_STATE_COMPLETE)

            all_tasks = []
            page = 1
            while True:
                tasks, total = state.get_all_tasks(page=page, page_size=3)
                all_tasks.extend(tasks)
                if page * 3 >= total:
                    break
                page += 1

            returned_ids = [task["task_id"] for task in all_tasks if "task_id" in task]
            self.assertEqual(len(returned_ids), len(set(returned_ids)))
            self.assertEqual(returned_ids, sorted(returned_ids))
            self.assertTrue(set(task_ids).issubset(returned_ids))
        finally:
            state._redis.delete(*task_ids)

    def test_patch_task_updates_only_existing_redis_task(self):
        state = self._build_state([1])

        self.assertTrue(
            state.patch_task(
                "task:0",
                cross_post_state=const.CROSS_POST_STATE_FAILED,
                cross_post_error="upload failed",
            )
        )
        task = state.get_task("task:0")
        self.assertEqual(task["progress"], 0)
        self.assertEqual(task["cross_post_state"], const.CROSS_POST_STATE_FAILED)
        self.assertEqual(task["cross_post_error"], "upload failed")
        self.assertFalse(state.patch_task("missing", value="ignored"))

    @unittest.skipUnless(
        os.getenv("MPT_TEST_REDIS_HOST"),
        "MPT_TEST_REDIS_HOST not set",
    )
    def test_real_redis_patch_and_delete_are_atomic(self):
        """En el Redis real, los borrados concurrentes y las actualizaciones parciales no
        deben recrear tareas incompletas.
        """
        state = RedisState(
            host=os.environ["MPT_TEST_REDIS_HOST"],
            port=int(os.getenv("MPT_TEST_REDIS_PORT", "6379")),
            db=int(os.getenv("MPT_TEST_REDIS_DB", "15")),
        )

        for _ in range(50):
            task_id = f"ci-atomic-{uuid.uuid4()}"
            state.update_task(
                task_id,
                state=const.TASK_STATE_COMPLETE,
                progress=100,
            )
            barrier = threading.Barrier(2)

            def patch_task():
                barrier.wait()
                state.patch_task(
                    task_id,
                    cross_post_state=const.CROSS_POST_STATE_COMPLETE,
                )

            def delete_task():
                barrier.wait()
                state.delete_task(task_id)

            # Future.result() vuelve a lanzar en el hilo del test las excepciones del
            # hilo de trabajo, para que un comando de Redis que falla de verdad no se
            # limite a imprimir la excepción del hilo y acabe dando el test por bueno.
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(patch_task),
                    executor.submit(delete_task),
                ]
                for future in futures:
                    future.result(timeout=5)

            self.assertIsNone(state.get_task(task_id))


if __name__ == "__main__":
    unittest.main()
