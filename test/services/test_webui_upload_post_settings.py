from unittest.mock import patch
from pathlib import Path
from streamlit.testing.v1 import AppTest
import pytest

from app.config import config

ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"

def _widget_by_key(elements, key):
    return next(
        item
        for item in elements
        if str(getattr(item, "key", "")) == key
        or str(getattr(item, "key", "")).startswith(f"{key}_")
    )


def test_webui_upload_post_setup_guide_links_to_required_pages():
    # La primera vez que se usa la publicación automática, la clave API y el usuario de
    # publicación se configuran en páginas distintas de Upload-Post. Aquí se fijan las
    # dos entradas exactas, para que futuros cambios de texto no vuelvan a degradarse a
    # un aviso genérico sin enlace y el usuario confunda el correo de inicio de sesión
    # con el nombre del usuario de publicación.
    with patch.object(config, "try_save_config", return_value=True):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.run()
        app.session_state["settings_dialog_open"] = True
        app.run()

        setup_guide = next(
            item
            for item in app.info
            if "https://app.upload-post.com/api-keys" in item.value
        )
        assert "https://app.upload-post.com/manage-users" in setup_guide.value


def test_webui_upload_post_checkboxes_stay_decoupled():
    # enabled=True / auto_upload=False es una separación deliberada: un flujo externo
    # puede llamar a Upload-Post mientras la publicación automática sigue desactivada.
    # Abrir el diálogo de ajustes no debe reescribir ninguna de las dos claves, y cada
    # casilla debe reflejar y escribir solo su propia clave.
    test_app_config = dict(
        config.app,
        upload_post_enabled=True,
        upload_post_auto_upload=False
    )

    with (
        patch.object(config, "app", test_app_config),
        patch.object(config, "try_save_config", return_value=True),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.run()
        app.session_state["settings_dialog_open"] = True
        app.run()

        enabled_box = _widget_by_key(app.checkbox, "upload_post_enabled_checkbox")
        auto_box = _widget_by_key(app.checkbox, "upload_post_auto_upload_checkbox")
        assert enabled_box.value is True
        assert auto_box.value is False

        # El simple renderizado del diálogo no debe haber reescrito la configuración
        # separada
        assert config.app["upload_post_enabled"] is True
        assert config.app["upload_post_auto_upload"] is False

        # Activar auto_upload debe dejar enabled intacto
        auto_box.set_value(True)
        app.run()
        assert config.app["upload_post_enabled"] is True
        assert config.app["upload_post_auto_upload"] is True

def test_webui_upload_post_youtube_privacy_fallback_to_public():
    # Si la configuración contiene por algún motivo un valor de privacidad de YouTube no
    # válido (p. ej. "draft"), no debe romper la interfaz y debe volver en silencio a
    # "public".
    test_app_config = dict(
        config.app,
        upload_post_platforms=["youtube"],
        upload_post_youtube_privacy_status="draft"
    )

    with (
        patch.object(config, "app", test_app_config),
        patch.object(config, "try_save_config", return_value=True),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.run()
        app.session_state["settings_dialog_open"] = True
        app.run()

        # El selectbox debe renderizarse y su valor debe volver a "public"
        yt_privacy_selectbox = _widget_by_key(app.selectbox, "upload_post_youtube_privacy_status_selectbox")
        assert yt_privacy_selectbox.value == "public"


@pytest.mark.parametrize("saved", [False, True, "false"])
def test_youtube_audience_selection_persists_on_first_change(saved):
    """Ejecuta de verdad los controles de Streamlit, verificando los booleanos, la
    configuración no válida y la persistencia del primer cambio.
    """
    values = dict(config.app, upload_post_platforms=["youtube"],
                  upload_post_youtube_made_for_kids=saved)
    with patch.object(config, "app", values), patch.object(config, "try_save_config", return_value=True):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60).run()
        app.session_state["settings_dialog_open"] = True
        app.run()
        assert not app.exception
        selector = _widget_by_key(app.selectbox, "upload_post_youtube_made_for_kids_selectbox")
        assert selector.value is (saved if isinstance(saved, bool) else None)
        assert values["upload_post_youtube_made_for_kids"] == saved
        for value in (True, False):
            _widget_by_key(app.selectbox, "upload_post_youtube_made_for_kids_selectbox").set_value(value)
            app.run()
            assert not app.exception
            assert values["upload_post_youtube_made_for_kids"] is value


def test_youtube_audience_hidden_for_other_platforms():
    """Si no se elige YouTube, no se muestra la opción de público ni se sobrescribe la
    declaración guardada.
    """
    values = dict(config.app, upload_post_platforms=["tiktok", "instagram"],
                  upload_post_youtube_made_for_kids=True)
    with patch.object(config, "app", values), patch.object(config, "try_save_config", return_value=True):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60).run()
        app.session_state["settings_dialog_open"] = True
        app.run()
        assert not app.exception
        assert not any(item.key == "upload_post_youtube_made_for_kids_selectbox" for item in app.selectbox)
        assert values["upload_post_youtube_made_for_kids"] is True
