"""Lectura y escritura seguras de los archivos persistentes del directorio de la tarea."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping

from loguru import logger

from app.utils import utils


_SCRIPT_LOCKS = tuple(threading.RLock() for _ in range(256))


def _script_lock(task_id: str):
    return _SCRIPT_LOCKS[hash(task_id) % len(_SCRIPT_LOCKS)]


def _script_file(task_id: str) -> Path:
    """Devuelve la ruta del manifiesto del guion de la tarea, reutilizando la lógica común
    de creación del directorio de la tarea.
    """
    return Path(utils.task_dir(task_id)) / "script.json"


def _write_json_atomic(target: Path, payload: Mapping[str, Any]) -> None:
    """
    Escribe JSON de forma atómica dentro del directorio de destino, para que una
    interrupción del proceso no deje un archivo a medias.

    El archivo temporal y el de destino deben estar en el mismo directorio para que
    ``os.replace`` mantenga la semántica de sustitución atómica en los sistemas de
    archivos locales habituales y en los directorios montados de Docker. No se modifica
    el archivo existente hasta que la escritura tiene éxito; ante una excepción solo se
    limpia el temporal creado en esta llamada y el error se pasa a quien llama para que
    decida si afecta al flujo principal.
    """
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(
                payload,
                temp_file,
                ensure_ascii=False,
                indent=4,
                default=lambda value: value.__dict__,
            )
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def write_script_data(task_id: str, payload: Mapping[str, Any]) -> None:
    """Crea o sustituye por completo el manifiesto ``script.json`` de la tarea."""
    with _script_lock(task_id):
        _write_json_atomic(_script_file(task_id), payload)


def patch_script_data(task_id: str, **updates: Any) -> bool:
    """
    Completa el manifiesto de la tarea conservando los campos existentes; devuelve
    ``False`` si falla.

    El origen del material es información de diagnóstico auxiliar, y no debe bloquear la
    generación del vídeo por permisos de archivo, fallos momentáneos del disco o
    archivos históricos dañados. Por eso esta entrada registra la excepción completa y
    degrada; la primera creación del manifiesto sigue usando ``write_script_data``, y el
    flujo principal decide qué hacer si falla la escritura de los datos básicos de la
    tarea.
    """
    try:
        with _script_lock(task_id):
            target = _script_file(task_id)
            with target.open("r", encoding="utf-8") as script_file:
                payload = json.load(script_file)
            if not isinstance(payload, dict):
                raise ValueError("los datos del guion de la tarea deben ser un objeto JSON")

            payload.update(updates)
            _write_json_atomic(target, payload)
            return True
    except FileNotFoundError:
        # ``download_videos`` también puede llamarse de forma independiente desde tests,
        # scripts o código de terceros; en ese caso que no haya manifiesto de tarea es
        # normal, y no se debe emitir un aviso ni crear un archivo incompleto solo para
        # el registro auxiliar.
        logger.debug(
            f"se omite la actualización del guion de la tarea porque no existe script.json: "
            f"task_id={task_id}"
        )
        return False
    except Exception as exc:
        logger.warning(
            "no se pudieron actualizar los datos del guion de la tarea: "
            f"task_id={task_id}, fields={sorted(updates)}, "
            f"error={type(exc).__name__}, detail={exc}"
        )
        return False
