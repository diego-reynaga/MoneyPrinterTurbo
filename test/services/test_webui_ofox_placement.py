"""Verifica el cambio de orden de presentación de OFox sin modificar la configuración de
generación guardada ni lanzar solicitudes de pago.
"""

import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from app.config import config


WEBUI = Path(__file__).resolve().parents[2] / "webui" / "Main.py"


def test_ai_video_source_order_keeps_ofox_below_metaso():
    # La constante controla el selector de fuente de vídeo; el diálogo de ajustes se
    # verifica con los tests de controles reales de abajo.
    tree = ast.parse(WEBUI.read_text(encoding="utf-8"))
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "VIDEO_SOURCE_GROUPS"
                for target in node.targets)
    )
    groups = ast.literal_eval(assignment.value)
    assert groups["ai_video"] == (
        "metaso_minimax", "ofox", "loomloom", "volcengine_seedance", "infistar", "wavespeed", "muapi"
    )
    assert groups["stock_video"] == ("pexels", "pixabay", "coverr")


@pytest.mark.parametrize("language", sorted(file.stem for file in (WEBUI.parent / "i18n").glob("*.json")))
def test_ofox_settings_order_and_saved_values(language):
    # Usa una configuración ficticia y prohíbe guardar; abrir los ajustes no debe
    # sobrescribir las credenciales locales ni cambiar la selección existente de OFox.
    values = {
        "llm_provider": "openai",
        "ofox_api_key": "review-only-fake-key",
        "ofox_text_to_video_model": "custom-model",
        "ofox_base_url": "https://example.invalid/v1",
        "ofox_provider": "volcengine",
    }
    expected = dict(values)
    with (
        patch.object(config, "app", values),
        patch.object(config, "try_save_config", return_value=True),
    ):
        app = AppTest.from_file(str(WEBUI), default_timeout=60)
        app.session_state["ui_language"] = language
        app.run()
        app.session_state["settings_dialog_open"] = True
        app.run()
        assert not app.exception
        keys = [item.key for item in app.text_input]
        assert keys.index("metaso_minimax_api_key_input") < keys.index("ofox_api_key_input")
        assert keys.index("ofox_api_key_input") < keys.index("loomloom_api_token_input")
        # El enlace para obtener la clave también debe llevar los mismos parámetros de
        # seguimiento, para no perder el origen de la promoción.
        api_key_input = next(item for item in app.text_input if item.key == "ofox_api_key_input")
        assert "https://ofox.ai/?utm_source=github&utm_medium=sponsorship&utm_content=moneyprinterturbo" in api_key_input.label
        assert any(item.value == "**OfoxAI**" for item in app.markdown)
        # Se renderiza de verdad; solo el nombre de la marca conserva el enlace
        # promocional y la descripción oficial del proveedor usa texto normal.
        messages = json.loads((WEBUI.parent / "i18n" / f"{language}.json").read_text(encoding="utf-8"))
        help_text = messages["Translation"]["OFox AI Video Help"]
        assert "http" not in help_text
        assert any(
            "[OfoxAI](https://ofox.ai/?utm_source=github&utm_medium=sponsorship&utm_content=moneyprinterturbo)" in item.value
            and help_text in item.value
            for item in app.caption
        )
        app.run()
        assert not app.exception
        for key, value in expected.items():
            assert values[key] == value
