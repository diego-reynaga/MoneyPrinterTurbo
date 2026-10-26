from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.config import config
from app.utils.logging_utils import get_logging_options, set_logging_options


def test_logging_form_applies_runtime_and_persisted_settings():
    """Ejecuta de verdad los controles de Streamlit del diálogo de ajustes; tras cambiar el
    nivel y el rastreo no hace falta reiniciar.
    """
    previous = get_logging_options()
    diagnostics = {}
    try:
        set_logging_options("DEBUG", False)
        with (
            patch.object(config, "diagnostics", diagnostics),
            patch.object(
                config,
                "app",
                dict(config.app, llm_provider="openai", openai_api_key=""),
            ),
            patch.object(config, "ui", dict(config.ui, language="es")),
            patch.object(config, "try_save_config", return_value=True),
        ):
            app = AppTest.from_file(
                str(Path(__file__).resolve().parents[2] / "webui/Main.py"),
                default_timeout=15,
            )
            app.session_state["settings_dialog_open"] = True
            app.run()
            assert not app.exception
            app.selectbox(key="diagnostic_log_level").select("ERROR")
            app.checkbox(key="diagnostic_log_requests").check()
            button = next(b for b in app.button if b.label == "Aplicar ajustes de registro")
            button.click().run()
            assert not app.exception
            assert get_logging_options() == ("ERROR", True)
            assert diagnostics == {"log_level": "ERROR", "log_requests": True}
            app.checkbox(key="diagnostic_log_requests").uncheck()
            next(
                b for b in app.button if b.label == "Aplicar ajustes de registro"
            ).click().run()
            assert not app.exception
            assert get_logging_options() == ("ERROR", False)
    finally:
        set_logging_options(*previous)
