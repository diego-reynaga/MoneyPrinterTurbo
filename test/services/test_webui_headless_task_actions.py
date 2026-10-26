import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from app.config import config

# webui/Main.py importa moviepy al importar app.services.material, y moviepy resuelve el
# ejecutable de FFmpeg al importarse (elige el binario según sys.platform); numpy
# también decide según sys.platform si llama a os.uname(). Esa resolución debe hacerse
# antes que el fixture de abajo; si no, al simular en Windows un servidor sin escritorio
# buscaría el FFmpeg de Linux o lanzaría directamente AttributeError.
from app.services import material as _material  # noqa: F401
from app.services import state as sm
from app.utils import utils

ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"


@pytest.fixture
def headless_task_app(tmp_path, monkeypatch):
    # Usa un directorio de tareas independiente para no leer ni modificar los registros
    # de generación reales del desarrollador. El archivo de prueba solo tiene que
    # registrarse en Streamlit como recurso multimedia y no participa en la
    # decodificación de vídeo, así que no hace falta llamar a FFmpeg en el test unitario
    # para generar material, y se cubre de forma estable la rama de interfaz del
    # servidor sin escritorio.
    tasks_dir = tmp_path / "storage" / "tasks"
    task_dir = tasks_dir / "headless-test"
    task_dir.mkdir(parents=True)
    video_file = task_dir / "final-1.mp4"
    video_file.write_bytes(b"test video payload")

    monkeypatch.setattr(utils, "task_dir", lambda: str(tasks_dir))
    monkeypatch.setattr(sm.state, "get_all_tasks", lambda *_args, **_kwargs: ([], 0))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    # AppTest vuelve a ejecutar el script de la página varias veces; el guardado de
    # configuración debe mantenerse aislado durante todo el test, para que la
    # inicialización de controles no escriba por accidente en el config.toml del
    # desarrollador.
    with patch.object(config, "try_save_config", return_value=True):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.run()
        yield app, video_file


def _button_by_key_prefix(app, key_prefix):
    return next(button for button in app.button if str(button.key).startswith(key_prefix))


def test_headless_play_renders_and_closes_browser_preview(headless_task_app):
    app, video_file = headless_task_app

    _button_by_key_prefix(app, "play_task_all_headless-test").click()
    app.run()

    assert not app.exception
    assert app.session_state["task_preview_video_file"] == str(video_file.resolve())
    assert len(app.get("video")) == 1

    _button_by_key_prefix(app, "close_task_video_preview").click()
    app.run()

    assert "task_preview_video_file" not in app.session_state
    assert len(app.get("video")) == 0


def test_headless_open_folder_shows_host_mapped_path(headless_task_app):
    app, _ = headless_task_app

    _button_by_key_prefix(app, "open_task_all_headless-test").click()
    app.run()

    assert not app.exception
    # _open_task_path() genera la parte relativa con os.path.relpath(), así que el
    # separador sigue la plataforma en la que se ejecuta el test. Se entra en la rama
    # headless parcheando sys.platform; en Windows el texto del aviso sigue llevando
    # barras invertidas, así que la aserción no puede fijar el separador POSIX.
    expected_folder = os.path.join("tasks", "headless-test")
    assert any(
        f"./storage/{expected_folder}" in toast.value for toast in app.get("toast")
    )
