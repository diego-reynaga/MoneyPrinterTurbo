import ast
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.models import const


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"
TASK_HISTORY_HELPERS = {
    "_find_final_task_video",
    "_build_video_download_name",
    "_build_restore_upload_requirements",
    "_get_unmet_restore_upload_requirements",
}
TASK_HISTORY_CONSTANTS = {
    "_FINAL_VIDEO_PATTERN",
    "_DOWNLOAD_FILENAME_INVALID_PATTERN",
    "_WINDOWS_RESERVED_FILENAMES",
    "VOICE_MODE_TTS",
    "VOICE_MODE_UPLOAD",
    "VOICE_MODE_NONE",
}


def _load_task_history_helpers():
    """
    Carga de forma aislada, desde la entrada de la WebUI, las funciones puras de
    historial de tareas que no dependen de Streamlit.

    Importar Main.py directamente ejecutaría todo el renderizado de la página. El test
    solo compila las constantes y funciones objetivo, así verifica la implementación
    real integrada y evita separar un módulo de producción con unas pocas funciones solo
    para los tests unitarios.
    """
    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    selected_nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in TASK_HISTORY_CONSTANTS
            for target in node.targets
        ):
            selected_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in TASK_HISTORY_HELPERS:
            selected_nodes.append(node)

    namespace = {"os": os, "re": re, "Mapping": Mapping}
    module = ast.fix_missing_locations(ast.Module(body=selected_nodes, type_ignores=[]))
    exec(compile(module, str(WEBUI_MAIN), "exec"), namespace)
    return namespace


TASK_HISTORY_NAMESPACE = _load_task_history_helpers()
find_final_task_video = TASK_HISTORY_NAMESPACE["_find_final_task_video"]
build_video_download_name = TASK_HISTORY_NAMESPACE["_build_video_download_name"]
windows_reserved_filenames = TASK_HISTORY_NAMESPACE["_WINDOWS_RESERVED_FILENAMES"]
build_restore_upload_requirements = TASK_HISTORY_NAMESPACE[
    "_build_restore_upload_requirements"
]
get_unmet_restore_upload_requirements = TASK_HISTORY_NAMESPACE[
    "_get_unmet_restore_upload_requirements"
]


def test_find_final_task_video_ignores_intermediate_files(tmp_path):
    """El historial de tareas solo puede considerar completado el vídeo final, no los
    archivos intermedios de composición.
    """
    for file_name in (
        "combined-1.mp4",
        "temp-clip-1.mp4",
        "final-1TEMP_MPY_wvf_snd.mp4",
    ):
        (tmp_path / file_name).touch()

    assert find_final_task_video(str(tmp_path)) == ""


def test_find_final_task_video_returns_first_numbered_output(tmp_path):
    """Las tareas con varios vídeos finales coinciden con el resultado en ejecución y
    reproducen por defecto el vídeo final con el número más bajo.
    """
    (tmp_path / "final-10.mp4").touch()
    (tmp_path / "final-2.mp4").touch()
    (tmp_path / "final-1.mp4").touch()

    assert find_final_task_video(str(tmp_path)) == str(tmp_path / "final-1.mp4")


def test_build_video_download_name_uses_subject_and_output_index():
    assert (
        build_video_download_name("A day: in / Shanghai?", 2, 3)
        == "A day in Shanghai-2.mp4"
    )


def test_build_video_download_name_handles_empty_and_long_subjects():
    assert build_video_download_name("  ...  ", 1, 1) == "video.mp4"
    assert len(build_video_download_name("a" * 100, 1, 1)) == 84


def test_build_video_download_name_avoids_windows_reserved_names():
    # Usa una lista explícita de las reglas oficiales para que el test no copie el
    # range/comprehension del código de producción; si la implementación tiene mal el
    # rango, la aserción de igualdad de conjuntos falla de inmediato en lugar de dejar
    # pasar el error junto con la implementación.
    reserved_names = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "COM¹",
        "COM²",
        "COM³",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
        "LPT¹",
        "LPT²",
        "LPT³",
    }
    assert windows_reserved_filenames == reserved_names

    for reserved_name in reserved_names:
        assert (
            build_video_download_name(reserved_name.lower(), 1, 1)
            == f"_{reserved_name.lower()}.mp4"
        )
        assert (
            build_video_download_name(f"{reserved_name}.topic", 2, 3)
            == f"_{reserved_name}.topic-2.mp4"
        )

    assert build_video_download_name("con .topic", 1, 1) == "_con .topic.mp4"


def test_build_video_download_name_does_not_overmatch_similar_names():
    """Solo trata los nombres realmente reservados de Windows, sin afectar a temas normales
    parecidos pero válidos.
    """

    for safe_name in ("COM0", "COM10", "LPT0", "LPT10", "COM⁴", "LPT⁴"):
        assert build_video_download_name(safe_name, 1, 1) == f"{safe_name}.mp4"


def test_restore_requirements_block_missing_uploaded_files():
    params = {
        "video_source": "local",
        "custom_audio_file": "/old-task/custom-audio.wav",
        "voice_name": "zh-CN-XiaoxiaoNeural-Female",
    }
    requirements = build_restore_upload_requirements(params)

    assert get_unmet_restore_upload_requirements(
        requirements,
        video_source="local",
        voice_name=params["voice_name"],
        has_local_materials=False,
        has_custom_audio=False,
    ) == {"local_materials", "custom_audio"}


def test_restore_requirements_allow_explicit_replacements():
    requirements = build_restore_upload_requirements(
        {
            "video_source": "local",
            "custom_audio_file": "/old-task/custom-audio.wav",
            "voice_name": "zh-CN-XiaoxiaoNeural-Female",
        }
    )

    assert not get_unmet_restore_upload_requirements(
        requirements,
        video_source="pexels",
        voice_name="en-US-JennyNeural-Female",
        has_local_materials=False,
        has_custom_audio=False,
    )


def test_restore_requirements_require_file_in_upload_voice_mode():
    """Al restaurar una tarea con locución subida, seguir en modo subida obliga a volver a
    elegir el archivo de audio.
    """
    requirements = build_restore_upload_requirements(
        {
            "video_source": "pexels",
            "custom_audio_file": "/old-task/custom-audio.wav",
            "voice_name": "zh-CN-XiaoxiaoNeural-Female",
        }
    )

    assert get_unmet_restore_upload_requirements(
        requirements,
        video_source="pexels",
        voice_name="zh-CN-XiaoxiaoNeural-Female",
        has_local_materials=False,
        has_custom_audio=False,
        voice_mode="upload",
    ) == {"custom_audio"}


def test_restore_requirements_allow_replacing_upload_with_other_voice_modes():
    """Si el usuario cambia a locución automática o sin locución, ya no se fuerza la
    restauración del archivo subido anterior.
    """
    requirements = build_restore_upload_requirements(
        {
            "video_source": "pexels",
            "custom_audio_file": "/old-task/custom-audio.wav",
            "voice_name": "zh-CN-XiaoxiaoNeural-Female",
        }
    )

    for voice_mode in ("tts", "none"):
        assert not get_unmet_restore_upload_requirements(
            requirements,
            video_source="pexels",
            voice_name="zh-CN-XiaoxiaoNeural-Female",
            has_local_materials=False,
            has_custom_audio=False,
            voice_mode=voice_mode,
        )


@pytest.mark.parametrize("payload", [[], {"params": []}, {"script": []}])
def test_history_scan_skips_non_object_script_payload(tmp_path, payload):
    """Un artefacto histórico mal formado no debe romper el panel de tareas."""
    tasks_root = tmp_path / "tasks"
    task_dir = tasks_root / "broken-task"
    task_dir.mkdir(parents=True)
    (task_dir / "script.json").write_text(json.dumps(payload), encoding="utf-8")

    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_safe_load_task_script", "_scan_history_tasks"}
    ]
    namespace = {
        "os": os,
        "json": json,
        "logger": MagicMock(),
        "utils": SimpleNamespace(task_dir=lambda: str(tasks_root)),
        "const": const,
        "_find_final_task_video": lambda _task_path: "",
    }
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(WEBUI_MAIN), "exec"), namespace)

    history = namespace["_scan_history_tasks"]()

    assert len(history) == 1
    assert history[0]["task_id"] == "broken-task"
    assert history[0]["subject"] == "broken-task"


def test_find_final_task_video_ignores_directories_and_broken_links(tmp_path):
    (tmp_path / "final-1.mp4").mkdir()
    (tmp_path / "final-2.mp4").symlink_to(tmp_path / "missing.mp4")
    assert find_final_task_video(str(tmp_path)) == ""
    (tmp_path / "final-3.mp4").touch()
    assert find_final_task_video(str(tmp_path)) == str(tmp_path / "final-3.mp4")
