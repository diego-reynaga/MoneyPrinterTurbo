from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from app.config import config
from app.services import voice


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"


class _GroupedSelectHarness:
    """Solo sustituye el componente de fuente de vídeo y conserva la implementación real
    del resto de Components v2 de la página.
    """

    def __init__(self):
        self.selected = None
        self.calls = []
        self.declaration = None
        self._original_component = st.components.v2.component

    def declare(self, name, *args, **kwargs):
        # Componentes de terceros como la guía de bienvenida también usan Components v2.
        # Dejar pasar esas declaraciones evita que el stub del test cambie otras
        # funciones de la página y solo controla el resultado de selección de fuente de
        # vídeo que interesa aquí.
        if name != "mpt_grouped_select":
            return self._original_component(name, *args, **kwargs)

        self.declaration = kwargs

        def render(**render_kwargs):
            self.calls.append(render_kwargs)
            return SimpleNamespace(selected=self.selected)

        return render


@contextmanager
def _running_app(harness, *, saved_video_source="pexels"):
    """Mantiene aislados el componente, la configuración y las consultas de voces externas
    durante todo el caso.
    """
    test_app_config = dict(config.app, video_source=saved_video_source)
    test_ui_config = dict(config.ui, language="es")
    with (
        patch(
            "streamlit.components.v2.component",
            side_effect=harness.declare,
        ),
        patch.object(config, "app", test_app_config),
        patch.object(config, "ui", test_ui_config),
        patch.object(config, "try_save_config", return_value=True),
        patch.object(
            voice,
            "get_all_azure_voices",
            return_value=["en-US-JennyNeural-Female"],
        ),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "es"
        app.run()
        assert [str(item.value) for item in app.exception] == []
        yield app


def test_grouped_video_source_applies_first_change_and_allows_switching_back():
    """Un solo evento change debe actualizar el estado de negocio, sin obligar al usuario a
    elegir dos veces.
    """
    harness = _GroupedSelectHarness()
    with _running_app(harness) as app:
        assert app.session_state["video_source_select_es"] == "pexels"
        assert harness.calls[-1]["data"]["value"] == "pexels"

        harness.selected = "pixabay"
        app.run()
        assert [str(item.value) for item in app.exception] == []
        assert app.session_state["video_source_select_es"] == "pixabay"
        # grouped_selectbox hace un rerun activo en la vuelta del evento; el último
        # renderizado debe devolver el nuevo valor al frontend, o el componente podría
        # quedar sobrescrito por los datos antiguos.
        assert harness.calls[-1]["data"]["value"] == "pixabay"

        harness.selected = "pexels"
        app.run()
        assert [str(item.value) for item in app.exception] == []
        assert app.session_state["video_source_select_es"] == "pexels"
        assert harness.calls[-1]["data"]["value"] == "pexels"


def test_grouped_video_source_ignores_unknown_event_and_repairs_saved_value():
    """Ni una configuración obsoleta ni un evento falsificado deben dejar la página con una
    fuente de material desconocida.
    """
    harness = _GroupedSelectHarness()
    with _running_app(harness, saved_video_source="removed-provider") as app:
        assert app.session_state["video_source_select_es"] == "pexels"
        assert harness.calls[-1]["data"]["value"] == "pexels"

        harness.selected = "unknown-provider"
        app.run()
        assert [str(item.value) for item in app.exception] == []
        assert app.session_state["video_source_select_es"] == "pexels"
        assert harness.calls[-1]["data"]["value"] == "pexels"


def test_grouped_video_source_keeps_groups_and_accessible_label_binding():
    """Los datos del componente deben mantener el orden de los grupos y dar un ID de
    control estable a cada etiqueta visible.
    """
    harness = _GroupedSelectHarness()
    with _running_app(harness):
        data = harness.calls[-1]["data"]
        assert data["controlId"] == "video_source_select_es_control"
        assert [group["label"] for group in data["groups"]] == [
            "Vídeos de stock",
            "Vídeo con IA",
            "Imagen con IA",
            "Archivos locales",
        ]
        assert [
            option["value"] for group in data["groups"] for option in group["options"]
        ] == [
            "pexels",
            "pixabay",
            "coverr",
            "metaso_minimax",
            "ofox",
            "loomloom",
            "volcengine_seedance",
            "infistar",
            "wavespeed",
            "muapi",
            "openai_image",
            "local",
        ]

        # AppTest no expone actualmente el DOM interno de Components v2, así que también
        # se verifica que la declaración del componente usa de verdad controlId para
        # asociar label/select y que permite que la fila de etiquetas se ajuste de forma
        # natural en pantallas estrechas.
        assert "label.htmlFor = data.controlId" in harness.declaration["js"]
        assert "select.id = data.controlId" in harness.declaration["js"]
        assert "flex-wrap: wrap" in harness.declaration["css"]


def test_stock_concurrency_only_appears_for_stock_sources():
    """La concurrencia de stock solo se muestra para los tres proveedores de material de
    stock, y al cambiar de fuente se conserva el ajuste explícito.
    """
    harness = _GroupedSelectHarness()
    with _running_app(harness) as app:
        stock = next(
            item for item in app.selectbox
            if item.key.startswith("material_concurrency_select_")
        )
        clip = next(
            item for item in app.selectbox
            if item.key.startswith("clip_rendering_concurrency_select_")
        )
        assert stock.value == 1
        assert clip.value == 1

        stock.set_value(4).run()
        clip = next(
            item for item in app.selectbox
            if item.key.startswith("clip_rendering_concurrency_select_")
        )
        clip.set_value(2).run()
        assert config.app["material_concurrency"] == 4
        assert config.app["video_clip_concurrency"] == 2

        for source, show_stock in (
            ("pixabay", True),
            ("coverr", True),
            ("wavespeed", False),
            ("local", False),
            ("pexels", True),
        ):
            harness.selected = source
            app.run()
            assert [str(item.value) for item in app.exception] == []
            stock_widgets = [
                item for item in app.selectbox
                if item.key.startswith("material_concurrency_select_")
            ]
            assert bool(stock_widgets) is show_stock
            if show_stock:
                assert stock_widgets[0].value == 4
            assert next(
                item for item in app.selectbox
                if item.key.startswith("clip_rendering_concurrency_select_")
            ).value == 2
            assert config.app["material_concurrency"] == 4
