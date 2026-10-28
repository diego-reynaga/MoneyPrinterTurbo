"""Tests del panel de publicación automática diaria de la WebUI."""

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.config import config


WEBUI_MAIN = str(Path(__file__).resolve().parents[2] / "webui/Main.py")


def _open_publish_settings(autopilot_section, state_file):
    """Abre el diálogo de ajustes con la sección autopilot aislada del config.toml real."""
    app = AppTest.from_file(WEBUI_MAIN, default_timeout=30)
    app.session_state["settings_dialog_open"] = True
    with (
        patch.object(config, "autopilot", autopilot_section),
        patch.object(config, "ui", dict(config.ui, language="es")),
        patch.object(config, "try_save_config", return_value=True),
        # El panel lee el plan guardado; se apunta a un archivo temporal para que el
        # test no toque storage/autopilot/state.json ni dependa de que exista.
        patch.dict(
            config.app,
            {
                "upload_post_enabled": True,
                "upload_post_api_key": "clave",
                "upload_post_username": "usuario",
                "upload_post_platforms": ["tiktok"],
            },
        ),
    ):
        autopilot_section.setdefault("state_file", state_file)
        app.run()
    return app


def test_the_panel_shows_the_configured_daily_rhythm(tmp_path):
    section = {
        "enabled": True,
        "niche": "finanzas personales",
        "videos_per_day": 3,
        "window_start_hour": 10,
        "window_end_hour": 22,
    }
    app = _open_publish_settings(section, str(tmp_path / "state.json"))
    assert not app.exception
    assert app.checkbox(key="autopilot_enabled_checkbox").value is True
    assert app.number_input(key="autopilot_videos_per_day_input").value == 3
    assert app.number_input(key="autopilot_window_start_input").value == 10
    assert app.number_input(key="autopilot_window_end_input").value == 22
    assert app.text_area(key="autopilot_niche_input").value == "finanzas personales"
    assert any(
        "3 vídeos al día entre las 10:00-22:00" in str(message.value)
        for message in app.success
    )


def test_editing_the_panel_writes_the_autopilot_section(tmp_path):
    section = {"enabled": False, "state_file": str(tmp_path / "state.json")}
    app = AppTest.from_file(WEBUI_MAIN, default_timeout=30)
    app.session_state["settings_dialog_open"] = True
    with (
        patch.object(config, "autopilot", section),
        patch.object(config, "ui", dict(config.ui, language="es")),
        patch.object(config, "try_save_config", return_value=True),
    ):
        app.run()
        assert not app.exception
        app.checkbox(key="autopilot_enabled_checkbox").check().run()
        app.text_area(key="autopilot_niche_input").set_value("recetas rápidas").run()
        app.number_input(key="autopilot_videos_per_day_input").set_value(4).run()
        app.text_area(key="autopilot_subjects_input").set_value(
            "Tortilla en 5 minutos\n\nPasta de una olla"
        ).run()
        assert not app.exception

    assert section["enabled"] is True
    assert section["niche"] == "recetas rápidas"
    assert section["videos_per_day"] == 4
    assert section["subjects"] == ["Tortilla en 5 minutos", "Pasta de una olla"]


def test_an_inverted_window_is_reported_instead_of_saved(tmp_path):
    section = {
        "enabled": True,
        "niche": "un nicho",
        "window_start_hour": 10,
        "window_end_hour": 22,
        "state_file": str(tmp_path / "state.json"),
    }
    app = AppTest.from_file(WEBUI_MAIN, default_timeout=30)
    app.session_state["settings_dialog_open"] = True
    with (
        patch.object(config, "autopilot", section),
        patch.object(config, "ui", dict(config.ui, language="es")),
        patch.object(config, "try_save_config", return_value=True),
    ):
        app.run()
        app.number_input(key="autopilot_window_end_input").set_value(9).run()
        assert not app.exception
        assert any(
            "mayor que la de inicio" in str(message.value) for message in app.error
        )


def test_missing_credentials_are_listed_as_problems(tmp_path):
    section = {
        "enabled": True,
        "niche": "un nicho",
        "state_file": str(tmp_path / "state.json"),
    }
    app = AppTest.from_file(WEBUI_MAIN, default_timeout=30)
    app.session_state["settings_dialog_open"] = True
    with (
        patch.object(config, "autopilot", section),
        patch.object(config, "ui", dict(config.ui, language="es")),
        patch.object(config, "try_save_config", return_value=True),
        patch.dict(
            config.app,
            {
                "upload_post_enabled": False,
                "upload_post_api_key": "",
                "upload_post_username": "",
            },
        ),
    ):
        app.run()
        assert not app.exception
        assert any(
            "upload_post_enabled" in str(message.value) for message in app.warning
        )


def test_invalid_cli_options_are_reported_in_the_panel(tmp_path):
    """La validación de cli_options importa la CLI de vídeo; aquí se comprueba que
    también funciona desde el proceso de Streamlit.
    """
    section = {
        "enabled": True,
        "niche": "un nicho",
        "cli_options": ["--esta-opcion-no-existe"],
        "state_file": str(tmp_path / "state.json"),
    }
    app = _open_publish_settings(section, str(tmp_path / "state.json"))
    assert not app.exception
    assert any("cli_options" in str(message.value) for message in app.warning)


def test_the_panel_lists_todays_plan(tmp_path):
    """El plan ya sorteado se muestra tal cual; el panel no planifica ni publica nada."""
    import json
    from datetime import datetime

    from app.services import autopilot

    state_file = tmp_path / "state.json"
    today = datetime.now().date()
    state = autopilot.empty_state()
    state["date"] = today.isoformat()
    state["slots"] = [
        autopilot._new_slot(1, datetime(today.year, today.month, today.day, 11, 7)),
        autopilot._new_slot(2, datetime(today.year, today.month, today.day, 18, 42)),
    ]
    state["slots"][0]["status"] = autopilot.SLOT_STATE_DONE
    state["slots"][0]["subject"] = "Fondo de emergencia"
    autopilot.save_state(str(state_file), state)

    section = {
        "enabled": True,
        "niche": "finanzas personales",
        "videos_per_day": 2,
        "state_file": str(state_file),
    }
    app = _open_publish_settings(section, str(state_file))
    assert not app.exception
    rendered = app.dataframe[0].value
    assert list(rendered["Hora"]) == ["11:07", "18:42"]
    assert list(rendered["Estado"]) == ["done", "pending"]
    assert list(rendered["Tema del vídeo"]) == ["Fondo de emergencia", "-"]
    # El panel solo lee: el plan guardado sigue intacto tras renderizar.
    assert json.loads(state_file.read_text(encoding="utf-8"))["slots"][0]["status"] == "done"
