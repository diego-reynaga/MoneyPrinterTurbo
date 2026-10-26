from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import sys
import textwrap
from typing import TYPE_CHECKING, Any, Sequence
from uuid import UUID, uuid4

from loguru import logger

if TYPE_CHECKING:
    from app.models.schema import MaterialInfo, VideoParams


DEFAULT_VOICE_NAME = "es-ES-ElviraNeural-Female"
# Corresponden a VOICE_MODE_NONE y VOICE_MODE_UPLOAD de webui/Main.py. Ambos lados no
# comparten aún estas constantes, así que aquí se mantienen los valores literales
# indicando su origen.
UI_VOICE_MODE_NONE = "none"
# Valor por defecto integrado de la posición de los subtítulos. VideoParams también
# tiene un valor por defecto del mismo nombre, pero es el valor por defecto de un campo
# Pydantic, que solo lee config.ui una vez al importar el módulo: un valor no válido de
# config.toml quedaría congelado directamente en el modelo, sin validarse ni poder
# sustituirse en los tests. Por eso la CLI guarda su propia copia, para garantizar que
# "un valor guardado no válido vuelve al valor por defecto" se cumpla siempre en la
# línea de comandos.
DEFAULT_SUBTITLE_POSITION = "bottom"
DEFAULT_CUSTOM_POSITION = 70.0
UI_VOICE_MODE_UPLOAD = "upload"
# Ambos modos de locución guardados significan no generar locución automática.
UI_VOICE_MODES_WITHOUT_TTS = frozenset({UI_VOICE_MODE_NONE, UI_VOICE_MODE_UPLOAD})
_PIPELINE_STAGES = ("script", "terms", "audio", "subtitle", "materials", "video")
_CUSTOM_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
_BATCH_FILE_MAX_BYTES = 1024 * 1024
_BATCH_TASK_MAX_COUNT = 100
# El argparse de tarea única y la lista de lotes deben compartir el mismo conjunto de
# fuentes. Antes se mantenían a mano en dos sitios, y openai_image solo estaba
# disponible en la entrada de tarea única; con una definición centralizada, los
# proveedores nuevos ya no se olvidarán en la validación de lotes. Aquí solo se incluyen
# las fuentes que la CLI ya admite públicamente, sin exponer a la fuerza flujos
# exclusivos de la WebUI.
_CLI_VIDEO_SOURCES = (
    "pexels",
    "pixabay",
    "coverr",
    "wavespeed",
    "volcengine_seedance",
    "ofox",
    "infistar",
    "metaso_minimax",
    "muapi",
    "openai_image",
    "local",
)


class _CliHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Muestra automáticamente los valores por defecto significativos, conservando el
    formato de los ejemplos de varias líneas.
    """

    def add_usage(self, usage, actions, groups, prefix=None):
        return super().add_usage(usage, actions, groups, prefix or "uso: ")

    def _split_lines(self, text, width):
        # No se parte por los guiones, para que identificadores como
        # es-ES-ElviraNeural-Female queden enteros en la ayuda.
        text = self._whitespace_matcher.sub(" ", text).strip()
        return textwrap.wrap(text, width, break_on_hyphens=False)

    def _get_help_string(self, action):
        help_text = action.help or ""
        if (
            "%(default)" not in help_text
            and action.default not in (None, "", argparse.SUPPRESS)
            and action.option_strings
            and "por defecto:" not in help_text.lower()
        ):
            help_text += " (por defecto: %(default)s)"
        return help_text


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"el valor debe ser >= 1, se recibió {parsed}")
    return parsed


def _paragraph_count(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > 10:
        raise argparse.ArgumentTypeError(
            f"paragraph-number debe estar entre 1 y 10, se recibió {parsed}"
        )
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(f"el valor debe ser un número finito >= 0, se recibió {value!r}")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(f"el valor debe ser un número finito > 0, se recibió {value!r}")
    return parsed


# Rango de valores de la velocidad de reproducción del clip. normalize_clip_speed() de
# app/utils/utils.py limita con el mismo rango, y el control deslizante de webui/Main.py
# también usa 0,5~2,0. Con una definición centralizada, los parámetros de tarea única,
# la validación de la lista de lotes y los valores guardados en [ui] no volverán a tener
# rangos distintos.
_CLIP_SPEED_MIN = 0.5
_CLIP_SPEED_MAX = 2.0


def _clip_speed(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not _CLIP_SPEED_MIN <= parsed <= _CLIP_SPEED_MAX:
        raise argparse.ArgumentTypeError(
            "video-clip-speed debe ser un número finito entre "
            f"{_CLIP_SPEED_MIN} y {_CLIP_SPEED_MAX}, se recibió {value!r}"
        )
    return parsed


def _percent_position(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or parsed > 100:
        raise argparse.ArgumentTypeError(
            f"custom-position debe ser un número finito entre 0 y 100, se recibió {value!r}"
        )
    return parsed


def _hex_color(value: str) -> str:
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise argparse.ArgumentTypeError(
            f"el color debe usar el formato #RRGGBB, se recibió {value!r}"
        )
    return value


def _subtitle_position(value: str) -> str:
    """Valida la posición de subtítulos guardada, con el mismo rango de valores que el
    parámetro de línea de comandos.
    """
    if value not in _SUBTITLE_POSITION_VALUES:
        raise argparse.ArgumentTypeError(
            "subtitle-position debe ser uno de: "
            f"{', '.join(_SUBTITLE_POSITION_VALUES)}, se recibió {value!r}"
        )
    return value


def _subtitle_display_mode(value: str) -> str:
    """Valida el modo de visualización de subtítulos guardado, con el mismo rango de
    valores que el parámetro de línea de comandos.
    """
    if value not in _SUBTITLE_DISPLAY_MODE_VALUES:
        raise argparse.ArgumentTypeError(
            "subtitle-display-mode debe ser uno de: "
            f"{', '.join(_SUBTITLE_DISPLAY_MODE_VALUES)}, se recibió {value!r}"
        )
    return value


def _subtitle_animation(value: str) -> str:
    """Valida la animación de subtítulos guardada, con el mismo rango de valores que el
    parámetro de línea de comandos.
    """
    if value not in _SUBTITLE_ANIMATION_VALUES:
        raise argparse.ArgumentTypeError(
            "subtitle-animation debe ser uno de: "
            f"{', '.join(_SUBTITLE_ANIMATION_VALUES)}, se recibió {value!r}"
        )
    return value


def _task_id(value: str) -> str:
    """El identificador de tarea personalizado de la CLI solo acepta UUID, para que el
    valor no se interprete como una ruta del sistema de archivos.
    """
    try:
        return str(UUID(value.strip()))
    except (AttributeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"task-id debe ser un UUID válido, se recibió {value!r}"
        ) from exc


_TRANSITION_MODE_VALUES = {
    "none": None,
    "shuffle": "Shuffle",
    "fade-in": "FadeIn",
    "fade-out": "FadeOut",
    "slide-in": "SlideIn",
    "slide-out": "SlideOut",
    "zoom-in": "ZoomIn",
    "zoom-out": "ZoomOut",
}

# El argparse de tarea única y la lista de lotes deben compartir los mismos valores de
# posición de subtítulos, para que no vuelva a pasar que «la WebUI lo puede guardar y la
# CLI lo rechaza». Los valores deben coincidir con las ramas de renderizado de
# app/services/video.py y con el desplegable de webui/Main.py.
_SUBTITLE_POSITION_VALUES = (
    "top",
    "center",
    "bottom",
    "two_thirds_bottom",
    "custom",
)

# El modo de visualización de subtítulos y la animación de entrada los guarda la WebUI
# en [ui]. Llegaron con los «subtítulos palabra a palabra + animación de rebote», y
# entonces solo se cambiaron la WebUI y los valores por defecto de los campos del
# modelo, así que a la línea de comandos siempre le faltó el interruptor. Los valores
# deben coincidir con _SUBTITLE_DISPLAY_MODES / _SUBTITLE_ANIMATIONS de
# app/models/schema.py.
_SUBTITLE_DISPLAY_MODE_VALUES = ("sentence", "word_by_word")
_SUBTITLE_ANIMATION_VALUES = ("none", "pop_spring")


def _transition_mode(value: str) -> str | None:
    normalized = value.strip().lower()
    if normalized not in _TRANSITION_MODE_VALUES:
        allowed = ", ".join(_TRANSITION_MODE_VALUES)
        raise argparse.ArgumentTypeError(
            f"video-transition-mode debe ser uno de: {allowed}"
        )
    return _TRANSITION_MODE_VALUES[normalized]


def _video_fit_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"cover", "contain"}:
        raise argparse.ArgumentTypeError(
            "video-fit-mode debe ser uno de: cover, contain"
        )
    return normalized


def _image_motion(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"static", "zoom"}:
        raise argparse.ArgumentTypeError("image-motion debe ser uno de: static, zoom")
    return normalized


# Solo estos valores piden un servicio externo de generación de música, así que solo
# ellos consumen el prompt de música. Los valores deben coincidir con
# _VIDEO_MUSIC_PROVIDERS de app/services/task.py; con una definición centralizada, los
# nuevos proveedores de música no se olvidarán en la validación de --bgm-type, en la
# validación de lotes ni en el parámetro de prompt.
_CLI_MUSIC_BGM_TYPES = ("sonilo", "elevenlabs")
_BGM_TYPE_CHOICES_TEXT = "none, random, custom, " + ", ".join(_CLI_MUSIC_BGM_TYPES)
_BGM_TYPE_METAVAR = "{none,random,custom," + ",".join(_CLI_MUSIC_BGM_TYPES) + "}"


def _bgm_type(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "none":
        return ""
    if normalized in {"", "random", "custom", *_CLI_MUSIC_BGM_TYPES}:
        return normalized
    raise argparse.ArgumentTypeError(
        f"bgm-type debe ser uno de: {_BGM_TYPE_CHOICES_TEXT}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    command = "mpt cli" if __name__ == "app.cli" else "uv run python cli.py"
    parser = argparse.ArgumentParser(
        prog="mpt cli" if __name__ == "app.cli" else None,
        description=(
            "Genera vídeos de MoneyPrinterTurbo sin la WebUI.\n\n"
            "Los ajustes y credenciales de los proveedores se leen de config.toml.\n"
            "La generación completa por defecto requiere un LLM configurado y una API Key de Pexels.\n"
            "La voz Edge TTS por defecto no requiere API Key."
        ),
        epilog="""
Ejemplos:
  Generar un vídeo completo con la voz Edge TTS por defecto:
    uv run python cli.py --video-subject "Cómo la IA está cambiando la vida cotidiana"

  Generar a partir de archivos locales. Las rutas relativas usan el directorio de
  trabajo actual; también se aceptan rutas absolutas:
    uv run python cli.py --video-subject "Cómo la IA está cambiando la vida cotidiana" \\
      --video-source local --video-materials "./1.mp4,./2.mp4"

  Generar con un guion preparado y sin locución:
    uv run python cli.py --video-script "Tu guion completo" \\
      --voice-name no-voice --stop-at video

  Detenerse tras generar el guion:
    uv run python cli.py --video-subject "Cómo la IA está cambiando la vida cotidiana" --stop-at script

  Ejecutar una lista JSON o un manifiesto JSONL. Las opciones de la CLI aportan los
  valores por defecto y cada objeto sobrescribe campos de VideoParams para una tarea:
    uv run python cli.py --batch-file ./tasks.jsonl --stop-at script

Etapas del flujo:
  script     Genera o devuelve el guion.
  terms      Genera los términos de búsqueda de material; no disponible con material local.
  audio      Genera TTS, audio en silencio o usa --custom-audio-file.
  subtitle   Genera los subtítulos si están activados.
  materials  Descarga material en línea o preprocesa los archivos locales.
  video      Genera el vídeo final y ejecuta la publicación cruzada configurada.
  El comando se detiene justo después de la etapa elegida e imprime su resultado.

Salida y código de salida:
  Los archivos de la tarea se escriben en storage/tasks/<task-id>/. Un comando correcto
  imprime un objeto JSON en stdout y sale con 0. El modo por lotes imprime un resumen
  JSON cuando terminan todas las tareas. Los fallos de tarea salen con 1; los errores de
  argumentos o del manifiesto salen con 2 antes de que empiece cualquier tarea del lote.
  Los logs de ejecución se escriben en stderr.

Manifiestos de lotes:
  --batch-file acepta una lista JSON en UTF-8 o un archivo JSONL con un objeto por cada
  línea no vacía (hasta 100 tareas y 1 MiB). Los objetos pueden sobrescribir campos de
  VideoParams; los campos desconocidos se rechazan. Cada tarea combinada necesita
  video_subject o video_script. La ruta del manifiesto es relativa al directorio de
  trabajo actual. Los valores relativos de custom_audio_file y de video_materials[].url
  locales declarados en un manifiesto son relativos al directorio del manifiesto. Las
  rutas relativas indicadas con opciones de la CLI siguen siendo relativas al directorio
  de trabajo actual. bgm_file y font_name mantienen sus reglas de búsqueda gestionadas
  en storage/resource. --stop-at es una opción de la CLI que afecta a todo el lote. El
  resumen tiene las claves total, succeeded, failed y tasks; cada entrada de tarea tiene
  index, task_id, status, result, failed_stage y error.
""".replace("uv run python cli.py", command),
        formatter_class=_CliHelpFormatter,
        add_help=False,
    )
    # argparse no traduce sus propios textos; se sustituyen los visibles en la ayuda.
    parser._optionals.title = "opciones"
    parser.add_argument(
        "-h", "--help", action="help", help="muestra esta ayuda y sale"
    )

    content_group = parser.add_argument_group("guion y contenido")
    content_group.add_argument(
        "--video-subject",
        default="",
        help="tema del vídeo; obligatorio salvo que se indique --video-script",
    )
    content_group.add_argument(
        "--video-script",
        default="",
        help="guion completo; si se indica, se omite la generación del guion con el LLM",
    )
    content_group.add_argument(
        "--video-terms",
        default=None,
        help="términos de búsqueda de material separados por comas; se generan automáticamente si se omiten",
    )
    content_group.add_argument(
        "--video-language",
        default=None,
        help=(
            "código de idioma del guion, como es-ES o en-US (por defecto: detección automática)"
        ),
    )
    content_group.add_argument(
        "--paragraph-number",
        type=_paragraph_count,
        default=None,
        help="número de párrafos del guion generado, de 1 a 10 (por defecto: 1)",
    )
    content_group.add_argument(
        "--video-script-prompt",
        default=None,
        help="requisitos adicionales para la generación del guion con el LLM",
    )
    content_group.add_argument(
        "--custom-system-prompt",
        default=None,
        help="sustituye el prompt de sistema por defecto del LLM para generar el guion",
    )

    material_group = parser.add_argument_group("material y flujo")
    material_group.add_argument(
        "--video-source",
        default="pexels",
        choices=_CLI_VIDEO_SOURCES,
        help="proveedor de material de vídeo; los proveedores en línea requieren su API Key en config.toml",
    )
    material_group.add_argument(
        "--video-materials",
        default="",
        metavar="PATH[,PATH...]",
        help=(
            "rutas locales de imágenes/vídeos separadas por comas para --video-source local; "
            "las rutas relativas usan el directorio de trabajo actual y, por compatibilidad, "
            "después storage/local_videos; se aceptan rutas absolutas"
        ),
    )
    material_group.add_argument(
        "--stop-at",
        default="video",
        choices=_PIPELINE_STAGES,
        help="se detiene tras esta etapa del flujo; consulta el orden de etapas más abajo",
    )
    material_group.add_argument(
        "--confirm-wavespeed-charge",
        action="store_true",
        help=(
            "confirma que la generación de vídeo con WaveSpeed crea tareas de pago; "
            "obligatorio con --video-source wavespeed para la salida materials o video"
        ),
    )
    material_group.add_argument(
        "--confirm-seedance-charge",
        action="store_true",
        help=(
            "confirma que Volcano Engine Seedance crea tareas de pago en Ark; "
            "obligatorio con --video-source volcengine_seedance para la salida materials o video"
        ),
    )
    material_group.add_argument(
        "--confirm-infistar-charge", action="store_true",
        help="confirma la generación de vídeo de pago con Infistar para la salida materials o video",
    )
    material_group.add_argument(
        "--confirm-ofox-charge",
        action="store_true",
        help=(
            "confirma que la generación de vídeo con OFox crea tareas de pago; "
            "obligatorio con --video-source ofox para la salida materials o video"
        ),
    )
    material_group.add_argument(
        "--confirm-metaso-minimax-charge",
        action="store_true",
        help=(
            "confirma que Metaso MiniMax H3 crea tareas de vídeo de pago; "
            "obligatorio con --video-source metaso_minimax para la salida materials o video"
        ),
    )
    material_group.add_argument(
        "--confirm-muapi-charge",
        action="store_true",
        help=(
            "confirma que la generación de vídeo con MuAPI crea tareas de pago; "
            "obligatorio con --video-source muapi para la salida materials o video"
        ),
    )

    video_group = parser.add_argument_group("salida de vídeo")
    video_group.add_argument(
        "--video-count",
        type=_positive_int,
        default=1,
        help="número de vídeos de salida, al menos 1",
    )
    video_group.add_argument(
        "--video-aspect",
        choices=["9:16", "16:9", "1:1"],
        default="9:16",
        help="relación de aspecto de salida: vertical, horizontal o cuadrada",
    )
    video_group.add_argument(
        "--video-fit-mode",
        type=_video_fit_mode,
        choices=["cover", "contain"],
        default=None,
        help=(
            "ajusta los clips de origen que no encajan rellenando y recortando al centro "
            "(cover) o conservando el encuadre completo con barras negras (contain); "
            "por defecto: cover"
        ),
    )
    video_group.add_argument(
        "--video-concat-mode",
        choices=["random", "sequential"],
        default=None,
        help="orden de concatenación de los clips de origen (por defecto: random)",
    )
    video_group.add_argument(
        "--image-motion",
        type=_image_motion,
        choices=["static", "zoom"],
        default=None,
        help="movimiento de las imágenes locales o generadas con IA (por defecto: zoom)",
    )
    video_group.add_argument(
        "--video-transition-mode",
        type=_transition_mode,
        default=None,
        metavar="{none,shuffle,fade-in,fade-out,slide-in,slide-out,zoom-in,zoom-out}",
        help="transición aplicada entre los clips de origen (por defecto: none)",
    )
    video_group.add_argument(
        "--video-clip-duration",
        type=_positive_int,
        default=None,
        help=(
            "duración máxima de cada clip de origen en segundos, al menos 1 (por defecto: 5)"
        ),
    )
    video_group.add_argument(
        "--video-clip-speed",
        type=_clip_speed,
        default=None,
        help=(
            "multiplicador de velocidad de reproducción de cada clip de origen, entre "
            "0.5 y 2.0 (por defecto: 1.0)"
        ),
    )
    video_group.add_argument(
        "--match-materials-to-script",
        default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "mantiene el orden de las palabras clave del guion al elegir y concatenar el "
            "material (por defecto: desactivado)"
        ),
    )
    video_group.add_argument(
        "--n-threads",
        type=_positive_int,
        default=None,
        help="número de hilos de trabajo de FFmpeg, al menos 1 (por defecto: 2)",
    )

    audio_group = parser.add_argument_group("locución y música de fondo")
    audio_group.add_argument(
        "--voice-name",
        default=None,
        help=(
            f"identificador de la voz TTS. Por defecto, [ui].voice_name de "
            f"config.toml o, si no, {DEFAULT_VOICE_NAME}. Un [ui].voice_mode "
            "guardado como 'none' o 'upload' equivale a no-voice, salvo que "
            "se indique esta opción. "
            "Usa 'no-voice' para una salida sin voz. Los identificadores propios "
            "de cada proveedor usan prefijos como gemini:, mimo:, elevenlabs:, "
            "chatterbox:, kokoro: y voxcpm:"
        ),
    )
    audio_group.add_argument(
        "--voice-volume",
        type=_non_negative_float,
        default=None,
        help=(
            "multiplicador del volumen final de la locución, un número finito >= 0 "
            "(por defecto: [ui].voice_volume de config.toml; 1.0 si no está definido)"
        ),
    )
    audio_group.add_argument(
        "--voice-rate",
        type=_positive_float,
        default=None,
        help=(
            "multiplicador de la velocidad del habla, un número finito > 0 "
            "(por defecto: [ui].voice_rate de config.toml; 1.0 si no está definido)"
        ),
    )
    audio_group.add_argument(
        "--custom-audio-file",
        default=None,
        metavar="PATH",
        help=(
            "locución MP3/WAV/M4A/AAC/FLAC/OGG existente; las rutas relativas usan el "
            "directorio de trabajo actual. Omite el TTS; define subtitle_provider=whisper "
            "para transcribirla"
        ),
    )
    audio_group.add_argument(
        "--bgm-type",
        type=_bgm_type,
        default=None,
        metavar=_BGM_TYPE_METAVAR,
        help=(
            "modo de música de fondo; Sonilo lee su API Key de config.toml o de "
            "SONILO_API_KEY, y ElevenLabs de config.toml o de ELEVENLABS_API_KEY; "
            "si se omite, --bgm-file implica custom (por defecto: random)"
        ),
    )
    audio_group.add_argument(
        "--sonilo-bgm-prompt",
        default=None,
        help="prompt opcional de estilo musical para Sonilo, hasta 2000 caracteres",
    )
    audio_group.add_argument(
        "--video-music-prompt",
        default=None,
        help=(
            "prompt opcional de estilo musical para el proveedor de música de fondo "
            "con IA elegido (sonilo o elevenlabs); el límite de longitud propio del "
            "proveedor se aplica antes de generar"
        ),
    )
    audio_group.add_argument(
        "--bgm-file",
        default=None,
        metavar="PATH",
        help=(
            "archivo de audio personalizado compatible dentro de storage/bgm o "
            "resource/songs; acepta un nombre de archivo o una ruta gestionada permitida"
        ),
    )
    audio_group.add_argument(
        "--bgm-volume",
        type=_non_negative_float,
        default=None,
        help=(
            "multiplicador del volumen de la música de fondo, un número finito >= 0 "
            "(por defecto: 0.2)"
        ),
    )

    subtitle_group = parser.add_argument_group("subtítulos")
    subtitle_group.add_argument(
        "--subtitle-enabled",
        default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "activa los subtítulos; usa --no-subtitle-enabled para desactivarlos "
            "(por defecto: [ui].subtitle_enabled de config.toml; activados si no "
            "está definido)"
        ),
    )
    subtitle_group.add_argument(
        "--font-name",
        default=None,
        help=(
            "nombre del archivo de fuente de subtítulos dentro de resource/fonts "
            "(por defecto: [ui].font_name de config.toml; "
            "STHeitiMedium.ttc si no está definido)"
        ),
    )
    subtitle_group.add_argument(
        "--subtitle-position",
        choices=_SUBTITLE_POSITION_VALUES,
        default=None,
        help=(
            "posición vertical de los subtítulos (por defecto: [ui].subtitle_position "
            "de config.toml; bottom si no está definido)"
        ),
    )
    subtitle_group.add_argument(
        "--subtitle-display-mode",
        choices=_SUBTITLE_DISPLAY_MODE_VALUES,
        default=None,
        help=(
            "ritmo de los subtítulos: frase a frase o palabra a palabra "
            "(por defecto: [ui].subtitle_display_mode de config.toml; sentence)"
        ),
    )
    subtitle_group.add_argument(
        "--subtitle-animation",
        choices=_SUBTITLE_ANIMATION_VALUES,
        default=None,
        help=(
            "animación de entrada de los subtítulos (por defecto: "
            "[ui].subtitle_animation de config.toml; none si no está definido)"
        ),
    )
    subtitle_group.add_argument(
        "--custom-position",
        type=_percent_position,
        default=None,
        help=(
            "posición personalizada en porcentaje desde arriba, 0-100; requiere "
            "--subtitle-position custom (por defecto: [ui].custom_position de "
            "config.toml; 70 si no está definido)"
        ),
    )
    subtitle_group.add_argument(
        "--text-fore-color",
        type=_hex_color,
        default=None,
        help=(
            "color del texto de los subtítulos en formato #RRGGBB; entrecomilla el "
            "valor en shells que tratan # como comentario (por defecto: "
            "[ui].text_fore_color de config.toml; #FFFFFF si no está definido)"
        ),
    )
    subtitle_group.add_argument(
        "--font-size",
        type=_positive_int,
        default=None,
        help=(
            "tamaño de fuente de los subtítulos (por defecto: [ui].font_size de "
            "config.toml; 60 si no está definido)"
        ),
    )
    subtitle_group.add_argument(
        "--stroke-color",
        type=_hex_color,
        default=None,
        help=(
            "color del contorno de los subtítulos en formato #RRGGBB (por defecto: "
            "[ui].stroke_color de config.toml; #000000 si no está definido)"
        ),
    )
    subtitle_group.add_argument(
        "--stroke-width",
        type=_non_negative_float,
        default=None,
        help=(
            "grosor del contorno de los subtítulos, un número finito >= 0 (por "
            "defecto: [ui].stroke_width de config.toml; 1.5 si no está definido)"
        ),
    )
    subtitle_group.add_argument(
        "--subtitle-background-enabled",
        default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "activa el fondo de los subtítulos; usa --no-subtitle-background-enabled "
            "para desactivarlo (por defecto: [ui].subtitle_background_enabled de "
            "config.toml; desactivado si no está definido)"
        ),
    )
    subtitle_group.add_argument(
        "--subtitle-background-color",
        type=_hex_color,
        default=None,
        help=(
            "color de fondo de los subtítulos en formato #RRGGBB (por defecto: "
            "[ui].subtitle_background_color de config.toml)"
        ),
    )
    subtitle_group.add_argument(
        "--rounded-subtitle-background",
        default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "usa un fondo de subtítulos redondeado (por defecto: "
            "[ui].rounded_subtitle_background de config.toml; "
            "desactivado si no está definido)"
        ),
    )

    execution_group = parser.add_argument_group("ejecución")
    execution_mode = execution_group.add_mutually_exclusive_group()
    execution_mode.add_argument(
        "--task-id",
        type=_task_id,
        default=None,
        help="UUID personalizado para storage/tasks/<task-id>; se genera automáticamente si se omite",
    )
    execution_mode.add_argument(
        "--batch-file",
        default=None,
        metavar="PATH",
        help=(
            "manifiesto de tareas como lista JSON en UTF-8 o JSONL; incompatible con "
            "--task-id y genera un UUID para cada tarea"
        ),
    )
    args = parser.parse_args(argv)

    if (
        not args.batch_file
        and not args.video_subject.strip()
        and not args.video_script.strip()
    ):
        parser.error("se requiere --video-subject o --video-script")

    if not args.batch_file and args.video_source == "local" and args.stop_at == "terms":
        parser.error(
            "--stop-at terms no tiene efecto con --video-source local "
            "(no se generan términos de búsqueda para fuentes locales)"
        )

    stage_requires_materials = args.stop_at in {"materials", "video"}
    has_video_materials = bool((args.video_materials or "").strip())
    if (
        not args.batch_file
        and args.video_source == "local"
        and stage_requires_materials
        and not has_video_materials
    ):
        parser.error(
            "--video-materials es obligatorio con --video-source local cuando "
            "--stop-at es materials o video"
        )
    if not args.batch_file and args.video_source != "local" and has_video_materials:
        parser.error("--video-materials solo puede usarse con --video-source local")
    if (
        not args.batch_file
        and args.video_source == "wavespeed"
        and stage_requires_materials
        and not args.confirm_wavespeed_charge
    ):
        parser.error(
            "--confirm-wavespeed-charge es obligatorio con "
            "--video-source wavespeed"
        )
    if (
        not args.batch_file
        and args.video_source == "volcengine_seedance"
        and stage_requires_materials
        and not args.confirm_seedance_charge
    ):
        parser.error(
            "--confirm-seedance-charge es obligatorio con "
            "--video-source volcengine_seedance"
        )
    if (
        not args.batch_file
        and args.video_source == "ofox"
        and stage_requires_materials
        and not args.confirm_ofox_charge
    ):
        parser.error(
            "--confirm-ofox-charge es obligatorio con --video-source ofox"
        )
    if (
        not args.batch_file
        and args.video_source == "metaso_minimax"
        and stage_requires_materials
        and not args.confirm_metaso_minimax_charge
    ):
        parser.error(
            "--confirm-metaso-minimax-charge es obligatorio con "
            "--video-source metaso_minimax"
        )
    if (
        not args.batch_file
        and args.video_source == "muapi"
        and stage_requires_materials
        and not args.confirm_muapi_charge
    ):
        parser.error(
            "--confirm-muapi-charge es obligatorio con --video-source muapi"
        )

    if (not args.batch_file and args.video_source == "infistar"
            and stage_requires_materials and not args.confirm_infistar_charge):
        parser.error("--confirm-infistar-charge es obligatorio con --video-source infistar")

    if args.bgm_file:
        if args.bgm_type in (None, "custom"):
            args.bgm_type = "custom"
        elif not args.batch_file:
            parser.error("--bgm-file solo puede combinarse con --bgm-type custom")

    if args.sonilo_bgm_prompt:
        if args.bgm_type in (None, "sonilo"):
            args.bgm_type = "sonilo"
        elif not args.batch_file:
            parser.error(
                "--sonilo-bgm-prompt solo puede combinarse con --bgm-type sonilo"
            )

    # El campo del prompt en sí no depende del proveedor, así que no se deduce bgm_type
    # como con el parámetro propio de Sonilo: el proveedor deducido podría no ser el que
    # quiere el usuario. Hay que elegir explícitamente un proveedor de música con IA; si
    # no, el prompt se descartaría en silencio. La lista de lotes puede seguir fijando
    # este campo por tarea.
    if (
        not args.batch_file
        and args.video_music_prompt
        and args.bgm_type not in _CLI_MUSIC_BGM_TYPES
    ):
        parser.error(
            "--video-music-prompt requiere --bgm-type sonilo o elevenlabs"
        )

    if (
        not args.batch_file
        and args.custom_position is not None
        and args.subtitle_position != "custom"
    ):
        parser.error("--custom-position requiere --subtitle-position custom")
    # Solo un --no-subtitle-enabled explícito cuenta como conflicto. El valor por
    # defecto ahora es None, y el estado desactivado guardado se trata en
    # build_video_params, así que aquí no debe dar error de parámetros.
    if (
        not args.batch_file
        and args.stop_at == "subtitle"
        and args.subtitle_enabled is False
    ):
        parser.error("--stop-at subtitle no puede combinarse con --no-subtitle-enabled")
    if not args.batch_file and args.subtitle_background_enabled is False and (
        args.subtitle_background_color is not None
        or args.rounded_subtitle_background is True
    ):
        parser.error(
            "el color o el redondeo del fondo de los subtítulos no pueden activarse junto "
            "con --no-subtitle-background-enabled"
        )

    return args


def _ui_config_value(ui_config, key: str, expected_type, checker=None):
    """
    Lee los ajustes de la WebUI guardados en ``[ui]``; si no hay un valor utilizable,
    devuelve ``None``.

    Esta sección también puede editarse a mano, así que aquí se descartan directamente
    las entradas inutilizables y se deja que quien llama vuelva a los valores por
    defecto integrados, en lugar de seguir pasando datos sucios que provoquen una traza
    o un error de validación de ``VideoParams``.

    ``checker`` reutiliza las mismas funciones de tipo que usa la línea de comandos
    (como ``_hex_color``), así que los valores guardados y los parámetros de línea de
    comandos siguen exactamente las mismas reglas: por ejemplo, el volumen no puede ser
    negativo y el color debe ser ``#RRGGBB``.
    """
    value = ui_config.get(key)
    if value is None:
        return None
    # ``isinstance(True, int)`` es verdadero, así que cuando se espera un número hay que
    # excluir bool explícitamente.
    if isinstance(value, bool) != (expected_type is bool):
        return None
    # En TOML ``1`` es un entero, pero para campos como el volumen o la velocidad
    # también es un valor válido.
    if expected_type is float and isinstance(value, int):
        value = float(value)
    if not isinstance(value, expected_type):
        return None
    if expected_type is str and not value.strip():
        return None
    if checker is not None:
        try:
            return checker(str(value))
        except argparse.ArgumentTypeError:
            return None
    return value


def _resolve_subtitle_enabled(args: argparse.Namespace, ui_config) -> bool:
    """
    Resuelve el interruptor de subtítulos por prioridad: línea de comandos > valor
    guardado de la WebUI > activado por defecto.

    ``--stop-at subtitle`` pide explícitamente generar subtítulos, así que el estado
    desactivado guardado no puede convertir esa etapa en una operación vacía; la
    combinación de un --no-subtitle-enabled explícito con esa etapa ya se rechaza en la
    validación de parámetros, así que aquí solo hay que tratar el valor guardado.
    """
    if args.subtitle_enabled is not None:
        return args.subtitle_enabled
    if args.stop_at == "subtitle":
        return True
    saved = _ui_config_value(ui_config, "subtitle_enabled", bool)
    return True if saved is None else saved


def _resolve_voice_name(args: argparse.Namespace, ui_config) -> str:
    """
    Resuelve la voz por prioridad: línea de comandos > modo de locución y voz guardados
    en la WebUI > valor por defecto integrado.

    La WebUI guarda "sin locución" como un voice_mode independiente, y conserva la
    última voz que el usuario eligió de verdad para restaurarla al volver a la locución
    automática. Por eso aquí no basta con leer voice_name; si no, se ignoraría el estado
    sin locución guardado y podrían volver a lanzarse solicitudes a proveedores de pago.
    """
    from app.services.voice import NO_VOICE_NAME

    if args.voice_name:
        return args.voice_name
    # Tanto "sin locución" como "subir audio propio" significan que el usuario no quiere
    # locución automática. La ruta del archivo del modo de subida no se escribe en [ui],
    # y la CLI no puede reproducir esa subida; usar en ese caso la voz guardada lanzaría
    # en silencio una solicitud de TTS de pago, así que ambos modos se asignan a
    # no-voice, y si se quiere locución hay que pasar --voice-name explícitamente.
    if _ui_config_value(ui_config, "voice_mode", str) in UI_VOICE_MODES_WITHOUT_TTS:
        return NO_VOICE_NAME
    return _ui_config_value(ui_config, "voice_name", str) or DEFAULT_VOICE_NAME


def build_video_params(args: argparse.Namespace) -> VideoParams:
    # La ayuda y la validación de parámetros no necesitan cargar la configuración de la
    # aplicación. Los modelos solo se importan al construir de verdad los parámetros de
    # la tarea, para que ``cli.py -h`` no genere logs de inicialización de
    # configuración.
    from app.config import config
    from app.models.schema import MaterialInfo, VideoParams

    ui_config = config.ui

    video_terms = args.video_terms
    if video_terms:
        video_terms = [
            term.strip() for term in re.split(r"[,，]", video_terms) if term.strip()
        ]

    video_materials = None
    materials_arg = args.video_materials or ""
    if materials_arg.strip():
        video_materials = [
            # La duración real se detectará durante el procesamiento del vídeo; se usa 0
            # como marcador.
            MaterialInfo(provider="local", url=item.strip(), duration=0)
            for item in materials_arg.split(",")
            if item.strip()
        ]

    params_kwargs = {
        "video_subject": args.video_subject.strip(),
        "video_script": args.video_script,
        "video_terms": video_terms,
        "video_source": args.video_source,
        "video_materials": video_materials,
        "video_count": args.video_count,
        "video_aspect": args.video_aspect,
        "voice_name": _resolve_voice_name(args, ui_config),
        "subtitle_enabled": _resolve_subtitle_enabled(args, ui_config),
    }

    optional_arg_names = [
        "video_language",
        "paragraph_number",
        "video_script_prompt",
        "custom_system_prompt",
        "video_fit_mode",
        "image_motion",
        "video_concat_mode",
        "video_transition_mode",
        "video_clip_duration",
        "video_clip_speed",
        "match_materials_to_script",
        "n_threads",
        "voice_volume",
        "voice_rate",
        "custom_audio_file",
        "bgm_type",
        "bgm_file",
        "bgm_volume",
        "sonilo_bgm_prompt",
        "video_music_prompt",
        "font_name",
        "subtitle_position",
        "subtitle_display_mode",
        "subtitle_animation",
        "custom_position",
        "text_fore_color",
        "font_size",
        "stroke_color",
        "stroke_width",
        "rounded_subtitle_background",
    ]
    for name in optional_arg_names:
        value = getattr(args, name)
        if value is not None:
            params_kwargs[name] = value

    # Si no se ha pasado explícitamente un parámetro de línea de comandos, se usa el
    # valor guardado por la WebUI. Solo se completan los campos de arriba que aún no ha
    # fijado la línea de comandos; si falta el valor guardado, se sigue usando el valor
    # por defecto de VideoParams. La mayoría de los campos guardables de VideoParams
    # (subtitle_position, custom_position, etc.) leen config.ui al importar el modelo,
    # pero video_clip_speed está fijado a 1.0, así que hay que completarlo aquí
    # explícitamente; si no, la velocidad guardada por la WebUI se descartaría en
    # silencio en la línea de comandos.
    ui_defaults = (
        ("video_clip_speed", float, _clip_speed),
        ("video_fit_mode", str, _video_fit_mode),
        ("image_motion", str, _image_motion),
        ("font_name", str, None),
        ("subtitle_display_mode", str, _subtitle_display_mode),
        ("subtitle_animation", str, _subtitle_animation),
        ("text_fore_color", str, _hex_color),
        ("font_size", int, _positive_int),
        ("rounded_subtitle_background", bool, None),
        ("voice_volume", float, _non_negative_float),
        ("voice_rate", float, _positive_float),
        ("stroke_color", str, _hex_color),
        ("stroke_width", float, _non_negative_float),
    )
    for name, expected_type, checker in ui_defaults:
        if name in params_kwargs:
            continue
        value = _ui_config_value(ui_config, name, expected_type, checker)
        if value is not None:
            params_kwargs[name] = value

    # La posición de los subtítulos debe tener un valor determinado dado por la CLI, sin
    # dejarlo al valor por defecto del campo en tiempo de importación descrito arriba.
    if "subtitle_position" not in params_kwargs:
        params_kwargs["subtitle_position"] = (
            _ui_config_value(ui_config, "subtitle_position", str, _subtitle_position)
            or DEFAULT_SUBTITLE_POSITION
        )
    if "custom_position" not in params_kwargs:
        saved_custom_position = _ui_config_value(
            ui_config, "custom_position", float, _percent_position
        )
        params_kwargs["custom_position"] = (
            DEFAULT_CUSTOM_POSITION
            if saved_custom_position is None
            else saved_custom_position
        )

    if args.subtitle_background_enabled is False:
        params_kwargs["text_background_color"] = False
        params_kwargs["rounded_subtitle_background"] = False
    elif args.subtitle_background_color is not None:
        params_kwargs["text_background_color"] = args.subtitle_background_color
    elif args.subtitle_background_enabled is True:
        # El usuario solo ha activado el fondo sin sobrescribir el color, así que se usa
        # primero el color guardado por la WebUI, y solo si no hay un valor guardado
        # utilizable se vuelve al fondo por defecto.
        params_kwargs["text_background_color"] = (
            _ui_config_value(
                ui_config, "subtitle_background_color", str, _hex_color
            )
            or True
        )
    else:
        # "Desactivar el fondo" junto con un color es un error de parámetros como
        # combinación de línea de comandos; pero como ajuste guardado, la misma
        # combinación no debe interrumpir la ejecución y solo significa que el fondo
        # está desactivado.
        ui_enabled = _ui_config_value(
            ui_config, "subtitle_background_enabled", bool
        )
        ui_color = _ui_config_value(
            ui_config, "subtitle_background_color", str, _hex_color
        )
        if ui_enabled is False:
            params_kwargs["text_background_color"] = False
            if args.rounded_subtitle_background is None:
                params_kwargs["rounded_subtitle_background"] = False
        elif ui_color is not None:
            params_kwargs["text_background_color"] = ui_color
        elif ui_enabled is True:
            params_kwargs["text_background_color"] = True

    return VideoParams(**params_kwargs)


def _load_batch_manifest(raw_path: str) -> tuple[str, list[dict[str, Any]]]:
    expanded_path = os.path.expanduser(raw_path.strip())
    if not expanded_path:
        raise ValueError("la ruta de --batch-file no puede estar vacía")

    candidate = (
        expanded_path
        if os.path.isabs(expanded_path)
        else os.path.join(os.getcwd(), expanded_path)
    )
    manifest_path = os.path.realpath(candidate)
    if not os.path.isfile(manifest_path):
        raise ValueError(f"el manifiesto de lotes no existe o no es un archivo: {raw_path}")

    with open(manifest_path, "rb") as manifest_file:
        payload = manifest_file.read(_BATCH_FILE_MAX_BYTES + 1)
    if len(payload) > _BATCH_FILE_MAX_BYTES:
        raise ValueError("el manifiesto de lotes supera el límite de 1 MiB")

    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("el manifiesto de lotes debe estar codificado en UTF-8") from exc
    if not text.strip():
        raise ValueError("el manifiesto de lotes debe contener al menos una tarea")

    if text.lstrip().startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"lista JSON no válida en el manifiesto de lotes: {exc.msg}"
            ) from exc
        if not isinstance(parsed, list):
            raise ValueError("el JSON del manifiesto de lotes debe ser una lista")
        entries = parsed
    else:
        entries = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL no válido en el manifiesto de lotes, línea {line_number}: {exc.msg}"
                ) from exc

    if not entries:
        raise ValueError("el manifiesto de lotes debe contener al menos una tarea")
    if len(entries) > _BATCH_TASK_MAX_COUNT:
        raise ValueError(
            f"el manifiesto de lotes contiene {len(entries)} tareas; "
            f"el límite es {_BATCH_TASK_MAX_COUNT}"
        )

    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"la tarea {index} del lote debe ser un objeto JSON")
    return manifest_path, entries


def _validate_batch_entry_fields(
    entry: dict[str, Any],
    *,
    index: int,
    allowed_fields: set[str],
) -> None:
    unknown_fields = sorted(set(entry) - allowed_fields)
    if unknown_fields:
        raise ValueError(
            f"la tarea {index} del lote contiene campos de VideoParams desconocidos: "
            f"{', '.join(unknown_fields)}"
        )

    materials = entry.get("video_materials")
    if not isinstance(materials, list):
        return
    allowed_material_fields = {"provider", "url", "duration", "source_info"}
    for material_index, material in enumerate(materials, start=1):
        if not isinstance(material, dict):
            continue
        unknown_material_fields = sorted(set(material) - allowed_material_fields)
        if unknown_material_fields:
            raise ValueError(
                f"el material {material_index} de la tarea {index} del lote contiene "
                f"campos de MaterialInfo desconocidos: {', '.join(unknown_material_fields)}"
            )


def _manifest_relative_path(raw_path: str, manifest_directory: str) -> str:
    expanded_path = os.path.expanduser(raw_path.strip())
    if not expanded_path or os.path.isabs(expanded_path):
        return expanded_path
    return os.path.realpath(os.path.join(manifest_directory, expanded_path))


def _resolve_batch_entry_paths(
    params: VideoParams,
    *,
    override_fields: set[str],
    manifest_directory: str,
) -> None:
    if "custom_audio_file" in override_fields and params.custom_audio_file:
        params.custom_audio_file = _manifest_relative_path(
            params.custom_audio_file,
            manifest_directory,
        )

    if (
        "video_materials" in override_fields
        and params.video_source == "local"
        and params.video_materials
    ):
        for material in params.video_materials:
            material.url = _manifest_relative_path(
                material.url,
                manifest_directory,
            )


def _validate_batch_task_params(
    params: VideoParams,
    *,
    stop_at: str,
    custom_position_is_explicit: bool,
    wavespeed_charge_confirmed: bool,
    seedance_charge_confirmed: bool,
    ofox_charge_confirmed: bool,
    metaso_minimax_charge_confirmed: bool,
    muapi_charge_confirmed: bool,
    infistar_charge_confirmed: bool = False,
) -> None:
    if not params.video_subject.strip() and not params.video_script.strip():
        raise ValueError("se requiere video_subject o video_script")

    if params.video_source not in _CLI_VIDEO_SOURCES:
        raise ValueError(
            "video_source debe ser uno de: " + ", ".join(_CLI_VIDEO_SOURCES)
        )
    for field_name, value in (
        ("video_aspect", params.video_aspect),
        ("video_concat_mode", params.video_concat_mode),
    ):
        # Estos campos del esquema siguen siendo Optional por compatibilidad con cargas
        # útiles históricas de la API, pero el flujo de vídeo siempre desreferencia sus
        # valores enum. Un null explícito en un manifiesto debe fallar antes de que
        # empiece cualquier tarea del lote.
        if value is None:
            raise ValueError(f"{field_name} no puede ser null")
    if params.video_source == "local" and stop_at == "terms":
        raise ValueError(
            "stop_at=terms no tiene efecto con video_source=local"
        )
    if (
        params.video_source == "local"
        and stop_at in {"materials", "video"}
        and not params.video_materials
    ):
        raise ValueError(
            "video_materials es obligatorio con video_source=local cuando "
            "stop_at es materials o video"
        )
    if params.video_source != "local" and params.video_materials:
        raise ValueError("video_materials solo puede usarse con video_source=local")
    if (
        params.video_source == "wavespeed"
        and stop_at in {"materials", "video"}
        and not wavespeed_charge_confirmed
    ):
        raise ValueError(
            "--confirm-wavespeed-charge es obligatorio para generar vídeo con WaveSpeed"
        )
    if (
        params.video_source == "volcengine_seedance"
        and stop_at in {"materials", "video"}
        and not seedance_charge_confirmed
    ):
        raise ValueError(
            "--confirm-seedance-charge es obligatorio para Volcano Engine Seedance"
        )
    if (
        params.video_source == "ofox"
        and stop_at in {"materials", "video"}
        and not ofox_charge_confirmed
    ):
        raise ValueError("--confirm-ofox-charge es obligatorio para generar vídeo con OFox")
    if (
        params.video_source == "metaso_minimax"
        and stop_at in {"materials", "video"}
        and not metaso_minimax_charge_confirmed
    ):
        raise ValueError(
            "--confirm-metaso-minimax-charge es obligatorio para Metaso MiniMax H3"
        )
    if (
        params.video_source == "muapi"
        and stop_at in {"materials", "video"}
        and not muapi_charge_confirmed
    ):
        raise ValueError("--confirm-muapi-charge es obligatorio para generar vídeo con MuAPI")

    if (params.video_source == "infistar" and stop_at in {"materials", "video"}
            and not infistar_charge_confirmed):
        raise ValueError("--confirm-infistar-charge es obligatorio para generar vídeo con Infistar")

    if stop_at == "subtitle" and not params.subtitle_enabled:
        raise ValueError("stop_at=subtitle no puede combinarse con los subtítulos desactivados")
    if params.subtitle_position not in _SUBTITLE_POSITION_VALUES:
        raise ValueError(
            "subtitle_position debe ser uno de: "
            + ", ".join(_SUBTITLE_POSITION_VALUES)
        )
    if custom_position_is_explicit and params.subtitle_position != "custom":
        raise ValueError("custom_position requiere subtitle_position=custom")
    if not math.isfinite(params.custom_position) or not 0 <= params.custom_position <= 100:
        raise ValueError("custom_position debe ser un número finito entre 0 y 100")
    if params.video_clip_speed is not None and (
        not math.isfinite(params.video_clip_speed)
        or not _CLIP_SPEED_MIN <= params.video_clip_speed <= _CLIP_SPEED_MAX
    ):
        raise ValueError(
            "video_clip_speed debe ser un número finito entre "
            f"{_CLIP_SPEED_MIN} y {_CLIP_SPEED_MAX}"
        )
    if params.text_background_color is False and params.rounded_subtitle_background:
        raise ValueError(
            "rounded_subtitle_background requiere un fondo de subtítulos activado"
        )

    color_values = {
        "text_fore_color": params.text_fore_color,
        "stroke_color": params.stroke_color,
        "text_background_color": (
            params.text_background_color
            if isinstance(params.text_background_color, str)
            else None
        ),
    }
    for name, value in color_values.items():
        if value is not None and not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
            raise ValueError(f"{name} debe usar el formato #RRGGBB")

    numeric_constraints = (
        ("voice_volume", params.voice_volume, 0, False),
        ("voice_rate", params.voice_rate, 0, True),
        ("bgm_volume", params.bgm_volume, 0, False),
        ("stroke_width", params.stroke_width, 0, False),
    )
    for name, value, minimum, exclusive in numeric_constraints:
        if value is None:
            continue
        if not math.isfinite(value) or (
            value <= minimum if exclusive else value < minimum
        ):
            comparison = ">" if exclusive else ">="
            raise ValueError(f"{name} debe ser un número finito {comparison} {minimum}")

    for name, value in (
        ("n_threads", params.n_threads),
        ("font_size", params.font_size),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} debe ser >= 1")

    if params.bgm_type not in {"", "random", "custom", *_CLI_MUSIC_BGM_TYPES}:
        raise ValueError(f"bgm_type debe ser uno de: {_BGM_TYPE_CHOICES_TEXT}")
    if params.bgm_file and params.bgm_type != "custom":
        raise ValueError("bgm_file requiere bgm_type=custom")
    if params.sonilo_bgm_prompt and params.bgm_type != "sonilo":
        raise ValueError("sonilo_bgm_prompt requiere bgm_type=sonilo")
    if params.video_music_prompt and params.bgm_type not in _CLI_MUSIC_BGM_TYPES:
        raise ValueError(
            "video_music_prompt requiere bgm_type=sonilo o elevenlabs"
        )


def _build_batch_tasks(args: argparse.Namespace) -> list[VideoParams]:
    from app.models.schema import VideoParams

    manifest_path, entries = _load_batch_manifest(args.batch_file)
    manifest_directory = os.path.dirname(manifest_path)
    base_params = build_video_params(args)
    allowed_fields = set(VideoParams.model_fields)
    base_values = {
        field_name: getattr(base_params, field_name) for field_name in allowed_fields
    }
    tasks: list[VideoParams] = []

    for index, entry in enumerate(entries, start=1):
        _validate_batch_entry_fields(
            entry,
            index=index,
            allowed_fields=allowed_fields,
        )
        try:
            params = VideoParams.model_validate(
                {**copy.deepcopy(base_values), **entry}
            )
            override_fields = set(entry)
            _resolve_batch_entry_paths(
                params,
                override_fields=override_fields,
                manifest_directory=manifest_directory,
            )
            _validate_batch_task_params(
                params,
                stop_at=args.stop_at,
                custom_position_is_explicit=(
                    args.custom_position is not None
                    or "custom_position" in override_fields
                ),
                seedance_charge_confirmed=args.confirm_seedance_charge,
                wavespeed_charge_confirmed=args.confirm_wavespeed_charge,
                ofox_charge_confirmed=args.confirm_ofox_charge,
                metaso_minimax_charge_confirmed=(
                    args.confirm_metaso_minimax_charge
                ),
                muapi_charge_confirmed=args.confirm_muapi_charge,
                infistar_charge_confirmed=args.confirm_infistar_charge,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"tarea {index} del lote no válida: {exc}") from exc
        tasks.append(params)

    validation_plans = []
    for index, params in enumerate(tasks, start=1):
        try:
            validation_plans.append(_validate_cli_files(params, stop_at=args.stop_at))
        except (OSError, ValueError) as exc:
            raise ValueError(f"tarea {index} del lote no válida: {exc}") from exc

    prepared_paths: dict[str, str] = {}
    created_paths: list[str] = []
    try:
        for index, (local_videos_dir, resolved_materials) in enumerate(
            validation_plans, start=1
        ):
            try:
                _prepare_cli_materials(
                    local_videos_dir,
                    resolved_materials,
                    prepared_paths=prepared_paths,
                    created_paths=created_paths,
                )
            except OSError as exc:
                raise ValueError(f"tarea {index} del lote no válida: {exc}") from exc
    except Exception:
        _remove_cli_material_copies(created_paths)
        raise
    return tasks


def _resolve_cli_file(
    raw_path: str,
    *,
    description: str,
    fallback_dir: str | None = None,
) -> str:
    """
    Resuelve los parámetros de archivo de la CLI a rutas absolutas según el directorio
    de trabajo actual, y confirma que existen antes de empezar la tarea.

    Las versiones antiguas resolvían siempre el material local relativo a
    ``storage/local_videos``. Por compatibilidad con los scripts existentes, si no se
    encuentra una ruta relativa en el directorio actual se permite volver a ese
    directorio; las rutas absolutas se resuelven siempre tal como las escribe el
    usuario.
    """
    expanded_path = os.path.expanduser(raw_path.strip())
    if not expanded_path:
        raise ValueError(f"la ruta de {description} no puede estar vacía")

    candidate = (
        expanded_path
        if os.path.isabs(expanded_path)
        else os.path.join(os.getcwd(), expanded_path)
    )
    resolved_path = os.path.realpath(candidate)
    if not os.path.isfile(resolved_path) and fallback_dir and not os.path.isabs(expanded_path):
        resolved_path = os.path.realpath(os.path.join(fallback_dir, expanded_path))

    if not os.path.isfile(resolved_path):
        raise ValueError(f"el archivo de {description} no existe: {raw_path}")
    return resolved_path


def _path_is_within_directory(file_path: str, directory: str) -> bool:
    try:
        return os.path.commonpath(
            [os.path.realpath(directory), os.path.realpath(file_path)]
        ) == os.path.realpath(directory)
    except ValueError:
        # En Windows no se puede calcular commonpath entre unidades distintas, y en ese
        # caso el archivo claramente no está en el directorio de destino.
        return False


def _resolve_managed_resource_file(
    raw_path: str,
    *,
    resource_dir: str,
    description: str,
) -> str:
    """Resuelve un archivo de recursos del proyecto, asegurando que la ruta absoluta sigue
    dentro del directorio de recursos correspondiente.
    """
    from app.utils import utils

    expanded_path = os.path.expanduser(raw_path.strip())
    candidates = (
        [expanded_path]
        if os.path.isabs(expanded_path)
        else [
            os.path.join(resource_dir, expanded_path),
            os.path.join(utils.root_dir(), expanded_path),
        ]
    )
    for candidate in candidates:
        resolved_path = os.path.realpath(candidate)
        if os.path.isfile(resolved_path) and _path_is_within_directory(
            resolved_path, resource_dir
        ):
            return resolved_path
    raise ValueError(
        f"el archivo de {description} debe existir dentro de {resource_dir}: {raw_path}"
    )


def _validate_cli_files(
    params: VideoParams, stop_at: str
) -> tuple[str, list[tuple[MaterialInfo, str, str]]]:
    """
    Resuelve y valida los archivos de la CLI sin efectos secundarios, para que la
    comprobación previa de lotes no deje copias del material.

    El audio personalizado, la BGM y las fuentes se normalizan a rutas o nombres que
    puede usar la capa de servicio; del material local solo se resuelven el origen y la
    extensión, y la copia real la hace ``_prepare_cli_materials`` de una vez cuando
    todas las entradas del lote han pasado la validación.
    """
    from app.models import const
    from app.services import bgm as bgm_service
    from app.utils import utils

    # La comprobación de FFmpeg se ha movido al flujo de tareas compartido de
    # app/services/task.py (task.start), donde se hace como comprobación obligatoria
    # común: si falla, la tarea termina con fallo en la etapa preflight, y run_cli()
    # devuelve según eso un código de salida distinto de cero. Aquí ya no se repite una
    # comprobación que no bloquea el flujo, para que no difiera del resultado del flujo.

    local_material_extensions = {
        *(f".{extension}" for extension in const.FILE_TYPE_VIDEOS),
        *(f".{extension}" for extension in const.FILE_TYPE_IMAGES),
        ".avi",
        ".flv",
    }

    if params.custom_audio_file and stop_at not in {"script", "terms"}:
        params.custom_audio_file = _resolve_cli_file(
            params.custom_audio_file,
            description="audio personalizado",
        )
        audio_extension = os.path.splitext(params.custom_audio_file)[1].lower()
        if audio_extension not in _CUSTOM_AUDIO_EXTENSIONS:
            allowed = ", ".join(sorted(_CUSTOM_AUDIO_EXTENSIONS))
            raise ValueError(
                f"tipo de audio personalizado no admitido {audio_extension or '<none>'}; "
                f"extensiones permitidas: {allowed}"
            )

    if params.bgm_type == "custom" and stop_at == "video":
        if not bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume):
            # Con volumen 0, más adelante se omite toda BGM. Aquí también se vacía el
            # parámetro de archivo, para que la CLI no resuelva la ruta, compruebe la
            # existencia ni valide el formato de un archivo que nunca se leerá.
            params.bgm_file = ""
        elif not params.bgm_file:
            # Que falte el archivo sea o no un error depende del interruptor general de
            # BGM, así que no se puede interceptar sin condiciones en la fase de
            # argparse; si no, ``custom + 0%`` no se comportaría igual que la WebUI y la
            # capa de servicio.
            raise ValueError("--bgm-file es obligatorio cuando --bgm-type es custom")
        else:
            try:
                # La CLI, la WebUI y el servicio de tareas deben compartir los mismos
                # límites de archivos de BGM. Aquí se reutiliza directamente la
                # resolución de la capa de servicio, que admite tanto el directorio de
                # subidas del usuario como el de canciones integradas y hereda
                # automáticamente los nuevos formatos de audio y las reglas de seguridad
                # de rutas, para que varias entradas no mantengan cada una su propia
                # lista blanca.
                params.bgm_file = bgm_service.resolve_bgm_file(params.bgm_file)
            except ValueError as exc:
                supported_extensions = ", ".join(
                    bgm_service.SUPPORTED_BGM_EXTENSIONS
                )
                raise ValueError(
                    "la música de fondo debe ser un archivo de audio compatible dentro de "
                    f"storage/bgm o resource/songs ({supported_extensions}): "
                    f"{params.bgm_file}"
                ) from exc

    if params.subtitle_enabled and params.font_name and stop_at == "video":
        font_path = _resolve_managed_resource_file(
            params.font_name,
            resource_dir=utils.font_dir(),
            description="fuente de subtítulos",
        )
        if not font_path.lower().endswith((".ttf", ".ttc")):
            raise ValueError("la fuente de subtítulos debe tener la extensión .ttf o .ttc")
        # Más adelante la ruta se construye a partir del nombre de archivo dentro de
        # resource/fonts, así que se sigue conservando el nombre de archivo puro.
        params.font_name = os.path.basename(font_path)

    if params.video_source != "local" or stop_at not in {"materials", "video"}:
        return "", []

    local_videos_dir = utils.storage_dir("local_videos")
    resolved_materials: list[tuple[MaterialInfo, str, str]] = []
    for material in params.video_materials or []:
        source_path = _resolve_cli_file(
            material.url,
            description="material local",
            fallback_dir=local_videos_dir,
        )
        extension = os.path.splitext(source_path)[1].lower()
        if extension not in local_material_extensions:
            allowed = ", ".join(sorted(local_material_extensions))
            raise ValueError(
                f"tipo de material local no admitido {extension or '<none>'}: "
                f"{material.url}; extensiones permitidas: {allowed}"
            )
        resolved_materials.append((material, source_path, extension))

    return local_videos_dir, resolved_materials


def _remove_cli_material_copies(created_paths: Sequence[str]) -> None:
    """Limpieza en lo posible de las copias gestionadas creadas por una preparación de la
    CLI.
    """
    for file_path in reversed(created_paths):
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except OSError as exc:
            logger.warning(
                f"no se pudo eliminar el material preparado de la CLI: path={file_path}, "
                f"error={exc}"
            )


def _prepare_cli_materials(
    local_videos_dir: str,
    resolved_materials: Sequence[tuple[MaterialInfo, str, str]],
    *,
    prepared_paths: dict[str, str] | None = None,
    created_paths: list[str] | None = None,
) -> None:
    """Copia una vez el material local validado y actualiza todas las referencias que
    coinciden.
    """
    if not resolved_materials:
        return

    os.makedirs(local_videos_dir, exist_ok=True)
    if prepared_paths is None:
        prepared_paths = {}
    if created_paths is None:
        created_paths = []

    for material, source_path, extension in resolved_materials:
        prepared_path = prepared_paths.get(source_path)
        if prepared_path is None:
            if _path_is_within_directory(source_path, local_videos_dir):
                prepared_path = source_path
            else:
                prepared_path = os.path.join(
                    local_videos_dir,
                    f"cli-material-{uuid4().hex}{extension}",
                )
                try:
                    shutil.copy2(source_path, prepared_path)
                except OSError:
                    try:
                        if os.path.exists(prepared_path):
                            os.remove(prepared_path)
                    except OSError:
                        pass
                    raise
                created_paths.append(prepared_path)
                logger.info(
                    "material local de la CLI copiado al almacenamiento gestionado: "
                    f"source={source_path}, target={prepared_path}"
                )
            prepared_paths[source_path] = prepared_path

        material.url = prepared_path


def prepare_cli_files(params: VideoParams, stop_at: str) -> None:
    """Valida y prepara los archivos de una tarea local de la CLI de confianza."""
    local_videos_dir, resolved_materials = _validate_cli_files(params, stop_at)
    created_paths: list[str] = []
    try:
        _prepare_cli_materials(
            local_videos_dir,
            resolved_materials,
            created_paths=created_paths,
        )
    except Exception:
        _remove_cli_material_copies(created_paths)
        raise


def _run_batch_tasks(args: argparse.Namespace, tasks: list[VideoParams]) -> int:
    from app.services import task as tm
    from app.utils import utils

    task_summaries: list[dict[str, Any]] = []
    used_task_ids: set[str] = set()
    failed_count = 0

    for index, params in enumerate(tasks, start=1):
        task_id = utils.get_uuid()
        if task_id in used_task_ids:
            task_id = str(uuid4())
        used_task_ids.add(task_id)
        logger.info(
            f"inicio de la tarea por lotes de la CLI: index={index}, task_id={task_id}, "
            f"stop_at={args.stop_at}"
        )

        result = None
        failed_stage = None
        error = None
        try:
            result = tm.start(
                task_id=task_id,
                params=params,
                stop_at=args.stop_at,
                allow_server_file_input=True,
            )
        except Exception as exc:
            failed_stage = "runtime"
            error = str(exc)
            logger.exception(
                "la tarea por lotes de la CLI falló con un error inesperado: "
                f"index={index}, task_id={task_id}, error={exc}"
            )
        else:
            if not isinstance(result, dict) or not result:
                failed_stage = "unknown"
                error = "resultado de tarea vacío o no válido"
            elif result.get("state") == tm.const.TASK_STATE_FAILED:
                failed_stage = result.get("failed_stage") or "unknown"
                error = result.get("error") or "error de tarea desconocido"

            if error is not None:
                logger.error(
                    f"la tarea por lotes de la CLI falló: index={index}, task_id={task_id}, "
                    f"stop_at={args.stop_at}, stage={failed_stage}, error={error}"
                )

        status = "failed" if error is not None else "succeeded"
        if status == "failed":
            failed_count += 1
        task_summaries.append(
            {
                "index": index,
                "task_id": task_id,
                "status": status,
                "result": result,
                "failed_stage": failed_stage,
                "error": error,
            }
        )

    print(
        json.dumps(
            {
                "total": len(task_summaries),
                "succeeded": len(task_summaries) - failed_count,
                "failed": failed_count,
                "tasks": task_summaries,
            },
            ensure_ascii=False,
        )
    )
    return 1 if failed_count else 0


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import app.config  # noqa: F401 -- inicializa la configuración y los logs del proceso.
    from app.utils.logging_utils import configure_terminal_logger, get_logging_options

    # El módulo de configuración envía por defecto los logs del terminal a stdout. El
    # stdout de la CLI es JSON legible por máquina, así que hay que cambiar a stderr
    # antes de empezar pasos como el preprocesado del material; la ayuda y los errores
    # de parámetros siguen manteniendo la importación diferida.
    configure_terminal_logger(sys.stderr, level=get_logging_options()[0], colorize=False)
    if args.batch_file:
        try:
            tasks = _build_batch_tasks(args)
        except (ValueError, OSError) as exc:
            logger.error(f"entrada por lotes de la CLI no válida: {exc}")
            return 2
        return _run_batch_tasks(args, tasks)

    try:
        params = build_video_params(args)
        prepare_cli_files(params, stop_at=args.stop_at)
    except (ValueError, OSError) as exc:
        logger.error(f"entrada de la CLI no válida: {exc}")
        return 2

    # Los parámetros de ayuda salen directamente en parse_args. Importar los servicios
    # de negocio aquí, de forma diferida, garantiza que la salida de -h/--help quede
    # limpia, sin cambiar el flujo de inicialización de las tareas reales.
    from app.services import task as tm
    from app.utils import utils

    task_id = args.task_id or utils.get_uuid()
    logger.info(f"inicio de la tarea de la CLI: task_id={task_id}, stop_at={args.stop_at}")
    try:
        result = tm.start(
            task_id=task_id,
            params=params,
            stop_at=args.stop_at,
            # Las entradas de la CLI vienen del operador local y no de un cliente HTTP.
            # Se mantiene el soporte de rutas de audio locales arbitrarias sin debilitar
            # el valor por defecto seguro del servicio de tareas para quienes llaman
            # desde la API y la WebUI.
            allow_server_file_input=True,
        )
    except Exception as exc:
        logger.exception(
            f"la tarea de la CLI falló con un error inesperado: task_id={task_id}, error={exc}"
        )
        return 1
    if not result or result.get("state") == tm.const.TASK_STATE_FAILED:
        failed_stage = result.get("failed_stage", "unknown") if result else "unknown"
        error = result.get("error", "error de tarea desconocido") if result else "resultado vacío"
        logger.error(
            f"la tarea de la CLI falló: task_id={task_id}, stop_at={args.stop_at}, "
            f"stage={failed_stage}, error={error}"
        )
        if result and result.get("videos"):
            print(json.dumps({"task_id": task_id, "result": result}, ensure_ascii=False))
        return 1

    print(json.dumps({"task_id": task_id, "result": result}, ensure_ascii=False))
    return 0


def _force_utf8_console() -> None:
    """Pone stdout/stderr en UTF-8 antes de imprimir nada.

    Las consolas de Windows usan por defecto una página de códigos antigua (cp1252 en
    Europa occidental). Generar un vídeo en francés produce U+202F, el espacio estrecho
    de no separación que la tipografía francesa pone antes de ':' y '!', y las propias
    líneas de progreso de Loguru llevan números en círculo. Cualquiera de los dos lanza
    UnicodeEncodeError, que mata el proceso *después* de que el vídeo se haya escrito
    correctamente, así que la ejecución informa de un fallo y nunca imprime dónde está
    el archivo.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            continue


if __name__ == "__main__":
    _force_utf8_console()
    raise SystemExit(run_cli())
