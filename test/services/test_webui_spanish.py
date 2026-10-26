from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from app.config import config
from app.services import state as sm
from app.services import version_checker, voice


WEBUI_MAIN = Path(__file__).resolve().parents[2] / "webui" / "Main.py"


def _selectbox(app, key):
    """Los controles llevan un sufijo de locale; se buscan por su clave estable."""
    return next(
        item for item in app.selectbox
        if item.key == key or str(item.key).startswith(f"{key}_")
    )


@pytest.fixture
def spanish_app():
    # Usa una configuración sin credenciales, una lista de tareas vacía y una función de
    # guardado que prohíbe escribir en disco, para que los tests de interfaz no lean las
    # tareas del usuario ni modifiquen la configuración real. El idioma guardado es de
    # una versión antigua ("zh"), que ya no existe: la interfaz debe seguir en español.
    with (
        patch.object(config, "app", {"llm_provider": "openai", "video_source": "pexels"}),
        patch.object(config, "ui", {
            "language": "zh", "voice_mode": "tts", "tts_server": "azure-tts-v1",
            "voice_name": "es-ES-ElviraNeural-Female", "video_language": "es-ES",
        }),
        patch.object(config, "try_save_config", return_value=True),
        patch.object(sm.state, "get_all_tasks", return_value=([], 0)),
        # Se fija como completada y sin actualizaciones para no lanzar solicitudes
        # reales a GitHub ni refrescos en segundo plano.
        patch.object(
            version_checker,
            "poll_available_update",
            return_value=version_checker.UpdateCheckSnapshot(complete=True),
        ),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.run()
        assert not app.exception
        yield app


def test_interface_is_always_spanish_without_language_selector(spanish_app):
    """La interfaz solo tiene español y no muestra ningún selector de idioma."""
    app = spanish_app
    assert app.session_state["ui_language"] == "es"
    assert not any(
        str(item.key).startswith("top_language_code_selector")
        for item in app.selectbox
    )
    assert _selectbox(app, "script_language_select").label == "Idioma del guion"
    assert _selectbox(app, "script_language_select").value == "es-ES"


def test_spanish_builtin_voices_and_settings_dialog(spanish_app):
    """Hay voces españolas integradas para elegir y el diálogo de ajustes se abre en
    español.
    """
    assert set(voice.get_all_azure_voices(["es-ES"])) >= {
        "es-ES-AlvaroNeural-Male", "es-ES-ElviraNeural-Female",
    }
    app = spanish_app
    assert any("Elvira" in option for item in app.selectbox for option in item.options)
    next(item for item in app.button if item.key == "open_settings_dialog_button").click().run()
    assert not app.exception
    assert any(tab.label == "Configuración LLM" for tab in app.tabs)
    assert any(tab.label == "Configuración de fuentes de material" for tab in app.tabs)
