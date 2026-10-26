"""Prepara los artefactos de subtítulos junto a su destino antes de publicarlos."""
from contextlib import contextmanager
import os
import re
import tempfile

from loguru import logger


@contextmanager
def staged_subtitle_file(destination):
    descriptor, staged = tempfile.mkstemp(
        prefix=".subtitle-", suffix=".srt",
        dir=os.path.dirname(os.path.abspath(destination)),
    )
    os.close(descriptor)
    try:
        yield staged
    finally:
        try:
            os.remove(staged)
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            logger.warning(f"no se pudo eliminar el subtítulo temporal: {cleanup_error}")


def has_subtitle_cue(content) -> bool:
    """Devuelve True cuando el cuerpo contiene al menos un subtítulo SRT con tiempos.

    Un cuerpo SRT sin una línea de tiempos completa no contiene ningún subtítulo que
    pueda mostrar un reproductor, así que una publicación hecha con él es una escritura
    fallida y no un resultado vacío válido.
    """
    timestamp = r"[0-9]{2,}:[0-5][0-9]:[0-5][0-9],[0-9]{3}"
    return any(
        re.fullmatch(rf"\s*{timestamp}\s+-->\s+{timestamp}(?:\s+.*)?\s*", line)
        for line in str(content or "").splitlines()
    )


def write_subtitle_file(destination, content) -> bool:
    """Publica un archivo de subtítulos completo sin destruir los subtítulos existentes.

    Todos los demás puntos de publicación de subtítulos validan el artefacto que van a
    sustituir: ``voice._write_subtitle_items`` vuelve a analizar el archivo preparado y
    ``task.generate_subtitle`` exige subtítulos analizables antes de ``os.replace``.
    Publicar un cuerpo vacío borraría en silencio los subtítulos ya guardados en
    ``destination``, que es justo lo que esas comprobaciones existen para evitar, así
    que aquí se aplica la misma regla una sola vez para todos los llamadores.

    Devuelve ``True`` tras una sustitución correcta y ``False`` cuando no se publicó
    nada porque el cuerpo no tenía subtítulos.
    """
    if not has_subtitle_cue(content):
        logger.warning(
            f"se rechaza publicar un archivo de subtítulos sin entradas: path={destination}"
        )
        return False

    with staged_subtitle_file(destination) as staged:
        with open(staged, "w", encoding="utf-8") as output:
            output.write(content)
        os.replace(staged, destination)
    return True
