from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.config import config


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"


def _widget_by_key(elements, key):
    """Busca controles de Streamlit con sufijo de idioma a partir de su clave de negocio
    estable.
    """
    return next(
        item
        for item in elements
        if str(getattr(item, "key", "")) == key
        or str(getattr(item, "key", "")).startswith(f"{key}_")
    )


def test_metaso_source_requires_confirmation_and_never_enters_task_params():
    test_config = dict(
        config.app,
        llm_provider="openai",
        video_source="pexels",
        metaso_minimax_api_key="metaso-secret",
        metaso_minimax_base_url="https://metaso.cn/api/minimax",
        metaso_minimax_resolution="2K",
    )
    with (
        patch.object(config, "app", test_config),
        patch.object(config, "ui", dict(config.ui, voice_mode="none")),
        patch.object(config, "try_save_config", return_value=True),
        patch("app.services.webui_task.submit_generation") as submit_generation,
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "es"
        app.run()

        _widget_by_key(app.text_area, "video_subject").set_value("Space fleet").run()
        _widget_by_key(app.text_area, "video_script").set_value(
            "A fleet jumps through space while its captain watches."
        ).run()
        _widget_by_key(app.text_area, "video_terms").set_value(
            "cinematic space fleet"
        ).run()
        app.session_state["video_source_select_es"] = "metaso_minimax"
        app.run()

        clip_duration = _widget_by_key(app.selectbox, "video_clip_duration_select")
        assert min(map(int, clip_duration.options)) == 4
        assert max(map(int, clip_duration.options)) == 15
        clip_duration.set_value(15).run()

        _widget_by_key(app.button, "generate_video_button").click().run()
        assert submit_generation.call_count == 0
        assert any("confirm" in str(item.value).lower() for item in app.error)

        _widget_by_key(app.checkbox, "metaso_minimax_confirm_charge").check().run()
        _widget_by_key(app.button, "generate_video_button").click().run()

        assert submit_generation.call_count == 1
        submitted_params = submit_generation.call_args.kwargs["params"]
        assert submitted_params.video_source == "metaso_minimax"
        assert submitted_params.video_clip_duration == 15
        assert "metaso-secret" not in submitted_params.model_dump_json()
        assert [str(item.value) for item in app.exception] == []


def test_invalid_metaso_resolution_requires_an_explicit_replacement():
    """Abrir los ajustes no debe cambiar en silencio una resolución no válida por el 2K por
    defecto, que es más caro.
    """
    test_config = dict(
        config.app,
        llm_provider="openai",
        video_source="pexels",
        metaso_minimax_resolution="720P",
    )
    test_ui = dict(config.ui, language="es")
    with (
        patch.object(config, "app", test_config),
        patch.object(config, "ui", test_ui),
        patch.object(config, "try_save_config", return_value=True),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "es"
        app.session_state["settings_dialog_open"] = True
        app.session_state["settings_dialog_target_tab"] = "material"
        app.run()

        resolution = _widget_by_key(
            app.selectbox, "metaso_minimax_resolution_input"
        )
        assert resolution.value is None
        assert test_config["metaso_minimax_resolution"] == "720P"
        assert any("720P" in str(item.value) for item in app.error)

        resolution.set_value("768P").run()
        assert test_config["metaso_minimax_resolution"] == "768P"
        assert [str(item.value) for item in app.exception] == []


def test_metaso_upload_voiceover_uses_actual_audio_billing_copy():
    """Al subir una locución no se debe usar la longitud del texto del guion como si fuera
    el número real de tareas de pago.
    """
    test_config = dict(
        config.app,
        video_source="metaso_minimax",
        metaso_minimax_api_key="metaso-secret",
        metaso_minimax_resolution="2K",
    )
    test_ui = dict(config.ui, language="es", voice_mode="upload")
    with (
        patch.object(config, "app", test_config),
        patch.object(config, "ui", test_ui),
        patch.object(config, "try_save_config", return_value=True),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "es"
        app.run()

        billing_warnings = [
            str(item.value)
            for item in app.warning
            if "Metaso factura cada clip generado" in str(item.value)
        ]
        assert len(billing_warnings) == 1
        assert "duración real del archivo de audio" in billing_warnings[0]
        assert "estimada" not in billing_warnings[0].lower()
        assert [str(item.value) for item in app.exception] == []
