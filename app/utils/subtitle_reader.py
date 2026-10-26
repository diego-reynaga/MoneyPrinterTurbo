"""Lee los límites de los subtítulos SRT antes de adaptarlos a tuplas de tiempos de
MoviePy.
"""
import os
import re


def file_to_subtitles(filename):
    if not filename or not os.path.isfile(filename):
        return []

    times_texts = []
    current_times = None
    current_text = ""
    index = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            times = re.findall("([0-9]*:[0-9]*:[0-9]*,[0-9]*)", line)
            # Una vez empezado un subtítulo, las marcas de tiempo pertenecen a su texto
            # hasta el separador de línea en blanco; no deben sobrescribir el tiempo del
            # subtítulo.
            if times and current_times is None:
                current_times = line
            elif line.strip() == "" and current_times:
                index += 1
                times_texts.append((index, current_times.strip(), current_text.strip()))
                current_times, current_text = None, ""
            elif current_times:
                current_text += line

    # Vuelca el último bloque. Los archivos SRT cuyo último subtítulo no va seguido de
    # una línea en blanco final nunca llegan a la rama de línea en blanco de arriba, así
    # que sin esto el último subtítulo se perdería en silencio.
    if current_times:
        index += 1
        times_texts.append((index, current_times.strip(), current_text.strip()))
    return times_texts


def moviepy_subtitles(filename):
    """Solo convierte las cabeceras de subtítulo; el texto que contiene marcas de tiempo
    sigue siendo texto del subtítulo.
    """
    from moviepy.tools import convert_to_seconds

    cues = []
    for _index, timing, text in file_to_subtitles(filename):
        start, end = re.findall("([0-9]*:[0-9]*:[0-9]*,[0-9]*)", timing)
        cues.append(((convert_to_seconds(start), convert_to_seconds(end)), text))
    return cues
