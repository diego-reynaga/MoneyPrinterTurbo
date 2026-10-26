import unittest
import sys
from pathlib import Path

# añade la raíz del proyecto al path de Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import task as tm


class TestIsTaskBusyDefensiveDefault(unittest.TestCase):
    """is_task_busy() debe conservar las tareas cuyo estado no se puede analizar."""

    def setUp(self):
        # El registro es global al proceso; se vacía para que los futures de publicación
        # cruzada que dejaron tests anteriores no afecten a las aserciones.
        with tm._cross_post_registry_lock:
            tm._cross_post_futures.clear()

    def tearDown(self):
        with tm._cross_post_registry_lock:
            tm._cross_post_futures.clear()

    def test_string_state_is_treated_as_busy(self):
        # Las instantáneas antiguas de Redis pueden devolver el estado como cadena. El
        # endpoint de borrado no debe poder cancelar una tarea cuyo estado no se puede
        # analizar.
        self.assertTrue(tm.is_task_busy({"state": "processing"}))
        self.assertTrue(tm.is_task_busy({"state": "4"}))
        self.assertTrue(tm.is_task_busy({"state": "PROCESSING"}))

    def test_none_state_is_treated_as_busy(self):
        # Un estado ausente o None sigue indicando una tarea sobre la que no podemos
        # razonar, así que debe aplicarse el valor por defecto conservador.
        self.assertTrue(tm.is_task_busy({"state": None}))
        self.assertTrue(tm.is_task_busy({"state": "not-an-int"}))

    def test_string_state_with_idle_cross_post_still_busy(self):
        # La ruta de código anterior devolvía False aquí, lo que permitía que la API de
        # borrado cancelara en silencio tareas en curso.
        self.assertTrue(
            tm.is_task_busy(
                {
                    "state": "processing",
                    "cross_post_state": tm.const.CROSS_POST_STATE_COMPLETE,
                }
            )
        )

    def test_integer_processing_state_remains_busy(self):
        # El caso normal existente no cambia.
        self.assertTrue(tm.is_task_busy({"state": tm.const.TASK_STATE_PROCESSING}))

    def test_integer_complete_with_idle_cross_post_remains_idle(self):
        # El caso normal existente no cambia.
        self.assertFalse(
            tm.is_task_busy(
                {
                    "state": tm.const.TASK_STATE_COMPLETE,
                    "cross_post_state": tm.const.CROSS_POST_STATE_COMPLETE,
                }
            )
        )

    def test_none_task_remains_idle(self):
        # La comprobación de None existente al principio de is_task_busy sigue teniendo
        # prioridad.
        self.assertFalse(tm.is_task_busy(None))


if __name__ == "__main__":
    unittest.main()
