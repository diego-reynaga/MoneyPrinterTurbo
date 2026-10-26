"""El nombre de archivo de descarga del historial de tareas debe eliminar los caracteres de
control y los de control de texto bidireccional.
"""
import ast
import os
import re
from collections.abc import Mapping
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"
DOWNLOAD_NAME_HELPERS = {"_build_video_download_name"}
DOWNLOAD_NAME_CONSTANTS = {
    "_DOWNLOAD_FILENAME_INVALID_PATTERN",
    "_WINDOWS_RESERVED_FILENAMES",
}


def _load_download_name_helpers():
    """
    Carga de forma aislada, desde la entrada de la WebUI, las funciones puras
    relacionadas con el nombre de descarga.

    Importar Main.py directamente ejecutaría todo el renderizado de la página. Aquí solo
    se compilan las constantes y funciones objetivo; lo que se ejecuta sigue siendo la
    implementación real integrada, así que las aserciones cubren el comportamiento de
    producción y no una copia dentro del test.
    """
    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    selected_nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in DOWNLOAD_NAME_CONSTANTS
            for target in node.targets
        ):
            selected_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in DOWNLOAD_NAME_HELPERS:
            selected_nodes.append(node)

    namespace = {"os": os, "re": re, "Mapping": Mapping}
    module = ast.fix_missing_locations(ast.Module(body=selected_nodes, type_ignores=[]))
    exec(compile(module, str(WEBUI_MAIN), "exec"), namespace)
    return namespace


build_video_download_name = _load_download_name_helpers()[
    "_build_video_download_name"
]

# Enumera explícitamente los caracteres que deben eliminarse, sin copiar la derivación
# con range del código de producción: si la implementación tiene mal el rango, la
# aserción falla directamente en lugar de dejar pasar el error junto con la
# implementación. Los caracteres de control C0 y U+0085, U+2028 y U+2029 ya se tratan al
# fusionar espacios en blanco, pero aquí se confirman uno a uno para que futuros cambios
# no los vuelvan a meter en el nombre de archivo.
C0_CONTROL_CHARACTERS = ("\x00", "\x08", "\x1b", "\x1f")
C1_CONTROL_CHARACTERS = ("\x7f", "\x80", "\x85", "\x9f")
BIDIRECTIONAL_CONTROL_CHARACTERS = (
    "‎",
    "‏",
    "‪",
    "‫",
    "‬",
    "‭",
    "‮",
    "⁦",
    "⁧",
    "⁨",
    "⁩",
)
LINE_SEPARATOR_CHARACTERS = (" ", " ")


def test_download_name_drops_bidirectional_override_from_subject():
    """U+202E invierte el renderizado de los caracteres siguientes y puede falsificar otra
    extensión de apariencia fiable.
    """
    download_name = build_video_download_name("holiday‮4pm", 1, 1)

    assert "‮" not in download_name
    assert download_name == "holiday 4pm.mp4"


def test_download_name_drops_every_unsafe_character_from_subject():
    """Ninguno de los caracteres que rechaza la entrada de subida puede entrar en el nombre
    de descarga.
    """
    unsafe_characters = (
        C0_CONTROL_CHARACTERS
        + C1_CONTROL_CHARACTERS
        + BIDIRECTIONAL_CONTROL_CHARACTERS
        + LINE_SEPARATOR_CHARACTERS
    )
    for character in unsafe_characters:
        download_name = build_video_download_name(f"trip{character}note", 1, 1)

        assert character not in download_name, f"U+{ord(character):04X} survived"
        assert download_name == "trip note.mp4"


def test_download_name_keeps_ordinary_multilingual_subjects():
    """Eliminar los caracteres de control no debe reescribir de paso temas normales,
    incluidos chino, letras acentuadas y puntuación.
    """
    assert build_video_download_name("上海一日游", 1, 1) == "上海一日游.mp4"
    assert build_video_download_name("Café déjà vu", 2, 3) == "Café déjà vu-2.mp4"
    assert (
        build_video_download_name("A day: in / Shanghai?", 2, 3)
        == "A day in Shanghai-2.mp4"
    )
