import itertools
import hashlib
from concurrent.futures import ThreadPoolExecutor
import io
import math
import os
import random
import gc
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from contextlib import ExitStack, contextmanager, redirect_stdout
from functools import lru_cache
from typing import Callable, List
from loguru import logger
import numpy as np
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    afx,
)
from moviepy.video.tools.subtitles import SubtitlesClip
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

from app.config import config
from app.models import const
from app.models.schema import (
    ImageMotion,
    MaterialInfo,
    VideoAspect,
    VideoConcatMode,
    VideoFitMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services import bgm as bgm_service
from app.services.utils import video_effects
from app.utils import file_security, logging_utils, utils
from app.utils.subtitle_reader import moviepy_subtitles

class SubClippedVideoClip:
    def __init__(
        self,
        file_path,
        start_time=None,
        end_time=None,
        width=None,
        height=None,
        duration=None,
        source_file_path=None,
    ):
        self.file_path = file_path
        self.start_time = start_time
        self.end_time = end_time
        self.width = width
        self.height = height
        self.source_file_path = source_file_path or file_path
        if duration is None:
            self.duration = end_time - start_time
        else:
            self.duration = duration

    def __str__(self):
        return f"SubClippedVideoClip(file_path={self.file_path}, start_time={self.start_time}, end_time={self.end_time}, duration={self.duration}, width={self.width}, height={self.height})"


audio_codec = "aac"
# La combinación ffmpeg/AAC de Docker tiende más a fluctuaciones de calidad de audio con
# la configuración por defecto, así que aquí se sube explícitamente la tasa de bits del
# audio, para que el vídeo final no introduzca una distorsión evidente por un valor por
# defecto demasiado bajo.
audio_bitrate = "192k"
fps = 30
# Cuando FFmpeg monta/transcodifica por fotogramas, la duración final puede ser unas
# decenas de milisegundos más corta que la duración teórica que lee MoviePy. Aquí se
# deja al material de vídeo un pequeño margen de seguridad, para que el final del audio
# no tenga pantalla negra, tirones o un último trozo de narración sin imagen por el
# redondeo de fotogramas.
_VIDEO_DURATION_SAFETY_MARGIN = 0.1
_MIN_MATERIAL_DIMENSION = 480
# Las aplicaciones de mensajería y algunos codificadores redondean hacia abajo el tamaño
# de la imagen; por ejemplo, WhatsApp comprime el material 9:16 a 478x850, dos píxeles
# menos que 480. Exigir estrictamente 480 haría que se descartara todo ese material y
# que todo fallara con "no valid materials found". Aquí se deja una pequeña tolerancia,
# que deja pasar el material que queda ligeramente por debajo del umbral solo por el
# redondeo, y aun así bloquea el material de baja resolución de verdad.
_MIN_DIMENSION_TOLERANCE = 10
# Por defecto los clips se procesan en serie, para que tras actualizar el uso de CPU y
# memoria no suba de golpe a los usuarios antiguos. Si se quiere acelerar se puede
# aumentar la concurrencia, pero cada vía arranca una tarea de codificación
# independiente.
_CLIP_PROCESSING_CONCURRENCY = 1


def _get_clip_processing_concurrency() -> int:
    try:
        concurrency = int(config.app.get("video_clip_concurrency", _CLIP_PROCESSING_CONCURRENCY))
    except (TypeError, ValueError):
        concurrency = _CLIP_PROCESSING_CONCURRENCY
    return max(1, min(8, concurrency))
_DEFAULT_VIDEO_CODEC = "libx264"
# Mientras ffmpeg encadena los clips no hay logs de etapa, y `subprocess.run` bloquea
# hasta que termina el proceso, así que un montaje largo aparece en el log como "sin
# salida". Aquí se registra a intervalos que sigue vivo, para distinguir entre que esté
# codificando o que se haya colgado.
_FFMPEG_CONCAT_HEARTBEAT_SECONDS = 30.0
# MoviePy tampoco produce ningún log al escribir el vídeo final, y un vídeo vertical de
# un minuto tarda varios minutos en codificarse en CPU. Se sigue lo mismo que en la fase
# de montaje y se registra a intervalos que sigue vivo.
_STAGE_HEARTBEAT_SECONDS = 30.0
_DEFAULT_FFMPEG_CONCAT_TIMEOUT_SECONDS = 3600
_SUBTITLE_SPRING_DURATION_SECONDS = 0.18
_MIN_SUBTITLE_SPRING_SCALE = 0.05
_MAX_SUBTITLE_SPRING_SCALE = 1.35
_SUPPORTED_VIDEO_CODECS = (
    "libx264",
    "h264_nvenc",
    "h264_amf",
    "h264_qsv",
    "h264_mf",
    "h264_videotoolbox",
)
_runtime_disabled_video_codecs = set()
# MoviePy envía fotogramas sRGB, y la matriz RGB→YUV por defecto de ffmpeg es BT.601 sin
# etiquetas de color, mientras que los reproductores y YouTube decodifican el HD sin
# etiquetar como BT.709 y desplazan los colores. Se convierte con la matriz BT.709 y se
# etiqueta el stream; setparams escribe las etiquetas que los flags -color_* por sí
# solos no escriben. El formato de píxel sigue siendo elección del codificador: MoviePy
# deja los fotogramas de tamaño impar a libx264.
_BT709_VIDEO_FILTER = (
    "scale=out_color_matrix=bt709:out_range=tv,"
    "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
)
_BT709_FFMPEG_PARAMS = ["-vf", _BT709_VIDEO_FILTER]


def _get_subtitle_spring_scale(time_seconds: float, duration_seconds: float) -> float:
    """Devuelve el factor de escala que usa la animación de rebote de los subtítulos en el
    instante indicado.
    """
    if duration_seconds <= 0 or time_seconds >= duration_seconds:
        return 1.0

    progress = max(0.0, min(time_seconds / duration_seconds, 1.0))
    scale = 1.0 - math.exp(-6.0 * progress) * math.cos(2.5 * math.pi * progress)
    return max(
        _MIN_SUBTITLE_SPRING_SCALE,
        min(scale, _MAX_SUBTITLE_SPRING_SCALE),
    )


def _scale_subtitle_frame_on_canvas(frame: np.ndarray, scale: float) -> np.ndarray:
    """
    Escala la imagen del subtítulo o la máscara de transparencia alrededor del centro,
    manteniendo el tamaño del lienzo.

    MoviePy guarda por separado el fotograma de color del subtítulo y la máscara de
    transparencia. La animación de rebote debe aplicar exactamente el mismo escalado y
    recorte a ambos; si no, el primer fotograma de la animación compondría sobre el
    vídeo la zona transparente como un contorno de texto negro. Un array bidimensional
    representa una máscara con valores de 0 a 1, y uno tridimensional un fotograma de
    color RGB/RGBA.
    """
    if frame.ndim not in (2, 3):
        raise ValueError("el fotograma del subtítulo debe ser una máscara 2D o un fotograma de color 3D")

    height, width = frame.shape[:2]
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    offset = ((width - scaled_width) // 2, (height - scaled_height) // 2)

    if frame.ndim == 2:
        # La máscara de MoviePy usa coma flotante de 0 a 1, y el modo L de Pillow usa de
        # 0 a 255; tras convertir se recuperan el tipo y el rango originales, para que
        # no cambie la semántica de transparencia de CompositeVideoClip.
        mask_image = Image.fromarray(
            np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        )
        resized_mask = mask_image.resize(
            (scaled_width, scaled_height),
            Image.Resampling.BILINEAR,
        )
        mask_canvas = Image.new("L", (width, height), 0)
        mask_canvas.paste(resized_mask, offset)
        return (np.asarray(mask_canvas) / 255.0).astype(frame.dtype, copy=False)

    if frame.shape[2] not in (3, 4):
        raise ValueError("el fotograma de color del subtítulo debe usar canales RGB o RGBA")
    color_image = Image.fromarray(frame)
    resized_color = color_image.resize(
        (scaled_width, scaled_height),
        Image.Resampling.BILINEAR,
    )
    background = (0, 0, 0, 0) if frame.shape[2] == 4 else (0, 0, 0)
    color_canvas = Image.new(color_image.mode, (width, height), background)
    color_canvas.paste(resized_color, offset)
    return np.asarray(color_canvas).astype(frame.dtype, copy=False)


def _apply_subtitle_spring_animation(clip, subtitle_duration: float):
    """Escala a la vez el fotograma de color y la máscara del subtítulo, para que la
    animación de rebote no tenga un primer fotograma negro.
    """
    animation_duration = min(
        _SUBTITLE_SPRING_DURATION_SECONDS,
        max(0.0, subtitle_duration),
    )
    if animation_duration <= 0:
        return clip

    def transform_frame(get_frame, time_seconds):
        frame = get_frame(time_seconds)
        scale = _get_subtitle_spring_scale(time_seconds, animation_duration)
        if scale == 1.0:
            return frame
        return _scale_subtitle_frame_on_canvas(frame, scale)

    # apply_to=["mask"] es la clave de la corrección: MoviePy solo trata por defecto el
    # fotograma de color, y por eso la implementación antigua conservaba brevemente la
    # máscara de tamaño original al aparecer cada subtítulo, mostrando un contorno de
    # texto negro.
    return clip.transform(transform_frame, apply_to=["mask"])


def _get_required_video_duration(audio_duration: float) -> float:
    """
    Devuelve la duración objetivo del montaje del material de vídeo.

    Uso: al componer el vídeo, la duración del material debe cubrir el audio de la
    narración. Si solo se llega a "exactamente igual" a la duración del audio, FFmpeg
    puede dejar el vídeo final un poco más corto por el redondeo de la tasa de
    fotogramas, así que siempre se añade un margen ligero. La función va aparte para
    facilitar los tests y para ajustar el margen según la experiencia real.
    """
    return max(0.0, float(audio_duration) + _VIDEO_DURATION_SAFETY_MARGIN)


def is_material_resolution_acceptable(width: int, height: int) -> bool:
    """
    Decide si la resolución del material es suficiente para la composición.

    El mínimo nominal es 480x480, pero se permiten `_MIN_DIMENSION_TOLERANCE` píxeles
    menos, para admitir tamaños que redondean hacia abajo los codificadores/aplicaciones
    de mensajería (por ejemplo, el 478x850 de WhatsApp).
    """
    min_dimension = _MIN_MATERIAL_DIMENSION - _MIN_DIMENSION_TOLERANCE
    return width >= min_dimension and height >= min_dimension


def _prioritize_unique_source_clips(
    subclipped_items: List[SubClippedVideoClip],
    concat_mode: VideoConcatMode,
    source_usage: dict[str, int] | None = None,
    source_groups: dict[str, str] | None = None,
) -> List[SubClippedVideoClip]:
    """
    Intenta que cada material de origen aparezca una sola vez, reduciendo la
    probabilidad de que el mismo material se repita en el vídeo final.

    El material en línea a menudo es "un vídeo largo cortado en varios clips cortos". La
    lógica antigua barajaba directamente todos los clips cortos en modo random, así que
    varios cortes del mismo vídeo de origen podían repartirse entre el principio y la
    mitad, y el usuario lo percibía como material repetido. Esta función solo ajusta el
    orden de los clips: primero pone el clip más largo de cada archivo de origen y deja
    el resto como respaldo; si la duración total del material no basta, sigue
    permitiendo que los clips posteriores completen la duración del audio, para no
    empeorar la tasa de éxito de la generación del vídeo. Se prefiere el clip más largo
    para no elegir al azar fragmentos cortos sueltos del final del vídeo, que harían
    reutilizar material antes de tiempo aunque haya suficiente.
    """
    if not subclipped_items:
        return []

    concat_mode_value = getattr(concat_mode, "value", concat_mode)
    if concat_mode_value != VideoConcatMode.random.value:
        if source_usage is None:
            return subclipped_items
        if not source_groups:
            return sorted(
                subclipped_items,
                key=lambda item: source_usage.get(item.source_file_path, 0),
            )
        # Mantiene en orden las rondas de palabras clave, rotando los candidatos dentro
        # de cada palabra clave.
        groups = {}
        for item in subclipped_items:
            key = source_groups.get(item.source_file_path, item.source_file_path)
            groups.setdefault(key, []).append(item)
        for items in groups.values():
            items.sort(key=lambda item: source_usage.get(item.source_file_path, 0))
        return [
            item
            for row in itertools.zip_longest(*groups.values())
            for item in row
            if item is not None
        ]

    grouped_items: dict[str, list[SubClippedVideoClip]] = {}
    for item in subclipped_items:
        grouped_items.setdefault(item.source_file_path, []).append(item)

    primary_items = []
    overflow_items = []
    for items in grouped_items.values():
        primary_item = max(items, key=lambda item: item.duration)
        primary_items.append(primary_item)
        overflow_items.extend(item for item in items if item is not primary_item)

    random.shuffle(primary_items)
    random.shuffle(overflow_items)
    if source_usage is not None:
        # La ordenación estable conserva la aleatoriedad entre fuentes usadas por igual.
        primary_items.sort(key=lambda item: source_usage.get(item.source_file_path, 0))
        overflow_items.sort(key=lambda item: source_usage.get(item.source_file_path, 0))
    logger.info(
        "material de vídeo priorizado sin repeticiones, "
        f"fuentes: {len(grouped_items)}, "
        f"clips principales: {len(primary_items)}, "
        f"clips de respaldo: {len(overflow_items)}"
    )
    return primary_items + overflow_items


def get_ffmpeg_binary():
    """
    Compatible con los llamadores históricos que leían la ruta de FFmpeg directamente
    del servicio de vídeo.

    La lógica real de resolución ya se ha extraído a
    `app.utils.utils.get_ffmpeg_binary()`, y el vídeo, la voz y las cadenas que se
    añadan después deben reutilizar la misma prioridad; aquí se mantiene un envoltorio
    fino, para que los scripts externos o los tests antiguos que importan directamente
    `app.services.video.get_ffmpeg_binary` no reciban un AttributeError.
    """
    return utils.get_ffmpeg_binary()


def _get_configured_video_codec() -> str:
    """
    Lee el codificador de vídeo configurado por el usuario.

    Esta opción es para usuarios avanzados que quieran probar la codificación por
    hardware NVENC/AMF/QSV/VideoToolbox, etc. Aquí solo se permite deliberadamente una
    lista blanca fija, para que abrir parámetros arbitrarios de FFmpeg no haga que un
    parámetro mal escrito vuelva incontrolable el formato de salida, o incluso que la
    tarea de generación falle en una fase posterior.
    """
    configured_codec = str(
        config.app.get("video_codec", _DEFAULT_VIDEO_CODEC) or _DEFAULT_VIDEO_CODEC
    ).strip()
    if configured_codec not in _SUPPORTED_VIDEO_CODECS:
        logger.warning(
            f"códec de vídeo configurado no admitido: {configured_codec}, "
            f"se usa {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC
    return configured_codec


@lru_cache(maxsize=16)
def _ffmpeg_encoder_exists(ffmpeg_binary: str, codec: str) -> bool:
    """
    Comprueba si el FFmpeg actual declara admitir el codificador indicado.

    Esto solo demuestra que FFmpeg se compiló con ese encoder, no que el hardware y el
    controlador de este equipo lo puedan usar. Por eso, si la codificación real falla,
    se sigue volviendo a libx264.
    """
    try:
        result = subprocess.run(
            [ffmpeg_binary, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "no se pudieron inspeccionar los codificadores de ffmpeg, "
            f"se usa {_DEFAULT_VIDEO_CODEC}: {str(exc)}"
        )
        return False

    if result.returncode != 0:
        logger.warning(
            "no se pudieron inspeccionar los codificadores de ffmpeg, "
            f"se usa {_DEFAULT_VIDEO_CODEC}: {(result.stderr or result.stdout or '').strip()}"
        )
        return False
    return codec in result.stdout


def _get_effective_video_codec(preferred_codec: str | None = None) -> str:
    """
    Devuelve el codificador de vídeo que se usa realmente esta vez.

    Cuando el usuario elige un codificador por hardware, primero se comprueba la lista
    de encoders de FFmpeg; si en este proceso ya ha fallado una codificación real,
    también se vuelve directamente, para que no falle repetidamente cada clip de una
    tarea.
    """
    selected_codec = preferred_codec or _get_configured_video_codec()
    if selected_codec == _DEFAULT_VIDEO_CODEC:
        return _DEFAULT_VIDEO_CODEC

    if selected_codec in _runtime_disabled_video_codecs:
        logger.warning(
            f"el códec de vídeo {selected_codec} se desactivó tras un fallo en ejecución, "
            f"se usa {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    ffmpeg_binary = utils.get_ffmpeg_binary()
    if not _ffmpeg_encoder_exists(ffmpeg_binary, selected_codec):
        logger.warning(
            f"el codificador de ffmpeg {selected_codec} no está disponible, "
            f"se usa {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    return selected_codec


def _disable_runtime_video_codec(codec: str, reason: str):
    if codec == _DEFAULT_VIDEO_CODEC:
        return
    _runtime_disabled_video_codecs.add(codec)
    logger.warning(
        f"falló el códec de vídeo {codec}, se usa {_DEFAULT_VIDEO_CODEC}. "
        f"motivo: {reason}"
    )


def _get_temp_audio_dir(output_dir: str) -> str:
    """
    Devuelve el directorio que se usa para el archivo de audio temporal de MoviePy.

    En Windows, Windows Defender puede bloquear los archivos escritos en el directorio
    de salida de la tarea mientras los analiza, haciendo que MoviePy falle con un
    PermissionError (WinError 32) en el archivo temporal TEMP_MPY_wvf_snd y dejando el
    MP4 final con 0 bytes. Usar el directorio temporal del sistema evita el análisis sin
    cambiar el comportamiento en otras plataformas.

    En Linux/macOS/Docker se devuelve el directorio de salida sin cambios para conservar
    el comportamiento existente.
    """
    if sys.platform == "win32":
        return tempfile.gettempdir()
    return output_dir


def _fallback_write_videofile(clip, output_file: str, failed_codec: str, reason: str, **kwargs):
    """
    Tras un fallo de la codificación por hardware se reintenta con libx264, y solo si el
    reintento tiene éxito se desactiva ese codificador por hardware.

    En Windows las causas de fallo de FFmpeg son complejas: puede que la tarjeta gráfica
    o el controlador no lo admitan, o que haya problemas generales de E/S como un
    archivo de salida en uso, permisos del directorio o el bloqueo de un antivirus. Solo
    cuando libx264 consigue escribir el archivo se puede concluir que el fallo original
    viene probablemente del propio codificador por hardware, para no perjudicar a las
    tareas posteriores.
    """
    clip.write_videofile(output_file, codec=_DEFAULT_VIDEO_CODEC, **kwargs)
    _disable_runtime_video_codec(failed_codec, reason)
    return _DEFAULT_VIDEO_CODEC


def _write_videofile_with_codec_fallback(
    clip, output_file: str, codec: str, atomic_output: bool = False, **kwargs
):
    """
    Escribe el vídeo con el codificador indicado y, si falla, reintenta automáticamente
    una vez con libx264.

    Que un codificador por hardware funcione no depende solo de FFmpeg, sino también de
    la tarjeta gráfica, el controlador y el entorno de ejecución actual. Una tarea de
    generación no puede fallar entera porque no esté disponible un codificador avanzado,
    así que el respaldo se gestiona aquí de forma centralizada.
    """
    kwargs.setdefault("ffmpeg_params", _BT709_FFMPEG_PARAMS)
    if atomic_output:
        # Los vídeos finales se pueden descargar por su ruta mientras se renderizan. Se
        # mantienen fuera de esa ruta tanto las codificaciones fallidas como las
        # escrituras en curso.
        output_dir = os.path.dirname(os.path.abspath(output_file))
        descriptor, temp_output = tempfile.mkstemp(
            prefix=f".{os.path.basename(output_file)}.",
            suffix=os.path.splitext(output_file)[1] or ".mp4",
            dir=output_dir,
        )
        os.close(descriptor)
        os.unlink(temp_output)
        try:
            used_codec = _write_videofile_with_codec_fallback(
                clip, temp_output, codec, **kwargs
            )
            os.replace(temp_output, output_file)
            return used_codec
        finally:
            try:
                os.unlink(temp_output)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(
                    f"no se pudo eliminar el vídeo final temporal: {temp_output}, "
                    f"error: {exc}"
                )

    effective_codec = _get_effective_video_codec(codec)
    try:
        clip.write_videofile(output_file, codec=effective_codec, **kwargs)
        return effective_codec
    except Exception as exc:
        if effective_codec == _DEFAULT_VIDEO_CODEC:
            raise
        return _fallback_write_videofile(
            clip,
            output_file,
            failed_codec=effective_codec,
            reason=str(exc),
            **kwargs,
        )


def _escape_ffmpeg_concat_path(file_path: str) -> str:
    # El demuxer concat envuelve las rutas entre comillas simples, así que antes hay que
    # escapar las comillas simples de la ruta.
    return file_path.replace("'", "'\\''")


def _format_ffmpeg_concat_path(file_path: str) -> str:
    """
    Genera la ruta para la lista de archivos del demuxer concat.

    La documentación oficial de FFmpeg exige escapar los caracteres especiales y los
    espacios de la lista concat; las barras invertidas de las rutas absolutas de Windows
    también se interpretan fácilmente como caracteres de escape. Aquí se convierten
    siempre a barras normales, de modo que `C:\\Users\\...` pasa a `C:/Users/...`, y
    luego se tratan las comillas simples, compatible con macOS/Linux.
    """
    absolute_path = os.path.abspath(file_path)
    # Solo se normaliza el separador nativo. Una barra invertida puede ser un componente
    # literal de una ruta POSIX válida y debe seguir identificando el mismo archivo.
    return _escape_ffmpeg_concat_path(absolute_path.replace(os.sep, "/"))


def _describe_concat_output_progress(output_file: str) -> str:
    """Devuelve una descripción legible del tamaño actual del archivo de salida, para el
    log de latido del montaje.
    """
    try:
        size = os.path.getsize(output_file)
    except OSError:
        # Cuando el archivo de salida aún no existe también hay que degradar con
        # seguridad, sin afectar al propio montaje.
        return "tamaño de salida no disponible"
    return f"tamaño de salida: {size / (1024 * 1024):.2f} MB"


@contextmanager
def _stage_heartbeat(description: str):
    """
    Registra a intervalos que sigue vivo durante una fase larga que no produce logs
    propios.

    El hilo de latido se vincula al hilo que entra en esa fase, para que sus logs
    aparezcan en el panel de la WebUI de la tarea correspondiente. Cuando la fase
    termina o lanza una excepción, se detiene y espera a que salga el hilo de latido,
    para que no siga escribiendo "sigue en curso" cuando la fase ya ha terminado.
    """
    started_at = time.monotonic()
    stop_event = threading.Event()

    def log_heartbeat() -> None:
        while not stop_event.wait(_STAGE_HEARTBEAT_SECONDS):
            logger.info(
                f"{description} sigue en curso: "
                f"elapsed={time.monotonic() - started_at:.0f}s"
            )

    reporter = threading.Thread(
        target=logging_utils.bind_log_scope(log_heartbeat), daemon=True
    )
    reporter.start()
    try:
        yield
    finally:
        stop_event.set()
        reporter.join(timeout=1)


def _report_clip_progress(
    progress_callback: Callable[[float], None] | None,
    covered_duration: float,
    required_duration: float,
) -> None:
    """Convierte la duración del vídeo final ya cubierta en una proporción de 0~1 y avisa a
    quien llama.
    """
    if progress_callback is None:
        return
    fraction = 1.0
    if required_duration > 0:
        fraction = min(1.0, covered_duration / required_duration)
    try:
        progress_callback(fraction)
    except Exception as exc:
        # El progreso es solo información para mostrar. Un fallo del callback (por
        # ejemplo, que el backend de estado no esté disponible temporalmente) no puede
        # invalidar los clips ya procesados.
        logger.warning(
            "no se pudo informar del progreso del procesamiento de clips: "
            f"error={type(exc).__name__}, detail={exc}"
        )


def _run_concat_with_heartbeat(command: list[str], output_file: str):
    """
    Espera bloqueado a que termine ffmpeg, registrando a intervalos que sigue vivo.

    Mientras ffmpeg encadena los clips no hay logs de etapa, y `subprocess.run` además
    almacena la salida hasta que termina el proceso, así que un montaje largo aparece en
    el log como "sin salida". Se registran el tiempo de espera y el tamaño del archivo
    de salida, para distinguir entre que siga codificando o que se haya colgado.
    """
    started_at = time.monotonic()
    stop_event = threading.Event()

    def log_heartbeat() -> None:
        while not stop_event.wait(_FFMPEG_CONCAT_HEARTBEAT_SECONDS):
            logger.info(
                "el montaje con ffmpeg sigue en curso: "
                f"elapsed={time.monotonic() - started_at:.0f}s, "
                f"{_describe_concat_output_progress(output_file)}"
            )

    reporter = threading.Thread(
        target=logging_utils.bind_log_scope(log_heartbeat), daemon=True
    )
    reporter.start()
    try:
        configured_timeout = config.app.get(
            "ffmpeg_concat_timeout_seconds", _DEFAULT_FFMPEG_CONCAT_TIMEOUT_SECONDS
        )
        try:
            timeout_seconds = float(configured_timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("ffmpeg_concat_timeout_seconds debe ser positivo") from exc
        if (
            isinstance(configured_timeout, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("ffmpeg_concat_timeout_seconds debe ser positivo")

        try:
            # subprocess.run mata y espera a FFmpeg al agotar el tiempo, así que un
            # codificador atascado no puede dejar un hijo huérfano ni una tarea activa
            # para siempre.
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"el montaje con ffmpeg superó los {timeout_seconds:g} segundos"
            ) from exc
    finally:
        stop_event.set()


def concat_video_clips_with_ffmpeg(
    clip_files: List[str],
    output_file: str,
    threads: int,
    output_dir: str,
    max_duration: float | None = None,
):
    # Distintos renderizados pueden compartir un directorio. Cada proceso de FFmpeg debe
    # conservar su propio manifiesto hasta que terminen todos los intentos de códec, sin
    # sobrescribir ni borrar la lista de otro renderizado.
    concat_list_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="ffmpeg-concat-", suffix=".txt",
            dir=output_dir, delete=False,
        ) as fp:
            concat_list_file = fp.name
            for clip_file in clip_files:
                fp.write(f"file '{_format_ffmpeg_concat_path(clip_file)}'\n")
    except Exception:
        if concat_list_file:
            delete_files(concat_list_file)
        raise

    staged_output = None

    def build_command(codec: str) -> list[str]:
        command = [
            utils.get_ffmpeg_binary(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_file,
            "-c:v",
            codec,
            "-threads",
            str(threads or 2),
            "-vf",
            _BT709_VIDEO_FILTER,
            "-pix_fmt",
            "yuv420p",
        ]
        if max_duration is not None and max_duration > 0:
            command.extend(["-t", f"{max_duration:.3f}"])
        command.append(staged_output)
        return command

    def run_concat(codec: str):
        command = build_command(codec)
        # Se usa ffmpeg para encadenar y codificar una sola vez, evitando la
        # recodificación repetida que haría MoviePy al fusionar clip a clip, lo que
        # reduce el riesgo de pérdida de calidad y de desplazamiento de color. Durante
        # la espera bloqueada, el log de latido muestra que la tarea sigue en marcha.
        result = _run_concat_with_heartbeat(command, staged_output)
        if result.returncode != 0:
            error_message = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(error_message or "falló la concatenación con ffmpeg")
        return codec

    try:
        descriptor, staged_output = tempfile.mkstemp(
            prefix=".ffmpeg-concat-",
            suffix=os.path.splitext(output_file)[1] or ".mp4",
            dir=os.path.dirname(os.path.abspath(output_file)),
        )
        os.close(descriptor)
        effective_codec = _get_effective_video_codec()
        try:
            result_codec = run_concat(effective_codec)
        except TimeoutError:
            # Un codificador colgado no demuestra que otro códec vaya a funcionar. No se
            # gasta un segundo periodo de espera reintentando la misma entrada.
            raise
        except Exception as exc:
            if effective_codec == _DEFAULT_VIDEO_CODEC:
                raise
            result_codec = run_concat(_DEFAULT_VIDEO_CODEC)
            _disable_runtime_video_codec(effective_codec, str(exc))
        # Los intentos fallidos y la salida en curso se mantienen privados hasta que
        # FFmpeg ha terminado. Los errores de publicación no deben provocar otro intento
        # de códec.
        if os.path.getsize(staged_output) == 0:
            raise RuntimeError("el montaje con ffmpeg no produjo ninguna salida")
        os.replace(staged_output, output_file)
        return result_codec
    finally:
        delete_files([concat_list_file, staged_output])


def _sanitize_image_file(image_path: str) -> str:
    # Algunas imágenes locales se pueden abrir con Pillow, pero unos metadatos EXIF/eXIf
    # dañados hacen que ImageClip lance una excepción directamente en la fase de
    # análisis. Aquí se vuelve a exportar una "imagen limpia" que elimina los metadatos
    # dañados. Se limita el nombre base intermedio conservando la identidad completa del
    # origen. Las distintas extensiones no deben sobrescribir píxeles, y los nombres de
    # archivo largos válidos no deben superar el límite de componente del sistema de
    # archivos tras añadir un sufijo.
    source_digest = hashlib.sha256(os.fsencode(os.path.abspath(image_path)))

    temp_path = ""
    try:
        with Image.open(image_path) as image:
            with ImageOps.exif_transpose(image) as upright:
                upright.load()
                # Elimina los metadatos tras aplicar la orientación de la cámara. La
                # transparencia de paleta pertenece a los píxeles, no a metadatos
                # eliminables.
                mode = "RGBA" if "A" in upright.getbands() or "transparency" in upright.info else "RGB"
                if upright.mode == "I;16":
                    # Las muestras de escala de grises PNG van de 0 a 65535;
                    # convertirlas a RGB las recorta a blanco o las desborda cuando
                    # MoviePy las codifica.
                    with upright.point(lambda value: value / 257 + 0.5) as byte_range:
                        cleaned_image = byte_range.convert(mode)
                else:
                    cleaned_image = upright.convert(mode)
                cleaned_image.info.clear()
                source_digest.update(str(cleaned_image.size).encode())
                source_digest.update(cleaned_image.mode.encode())
                source_digest.update(cleaned_image.tobytes())
                sanitized_path = os.path.join(
                    os.path.dirname(image_path),
                    f"{source_digest.hexdigest()}.sanitized.png",
                )
                descriptor, temp_path = tempfile.mkstemp(
                    prefix=".image-sanitize-", suffix=".png",
                    dir=os.path.dirname(os.path.abspath(sanitized_path)),
                )
                os.close(descriptor)
                cleaned_image.save(temp_path)
                cleaned_image.close()
        os.replace(temp_path, sanitized_path)
        temp_path = ""
    finally:
        if temp_path:
            delete_files(temp_path)

    return sanitized_path


def _open_image_clip_with_fallback(image_path: str):
    # MoviePy no aplica la orientación EXIF de la cámara al decodificar una imagen. Los
    # canales JPEG CMYK deben pasar a RGB; LA debe pasar a RGBA para que MoviePy cree
    # una máscara de opacidad correcta en lugar de tratar el alfa como una banda de
    # color. Los píxeles de dos niveles deben pasar a RGB en el rango de bytes en lugar
    # de valores bool 0/1. Las entradas RGB normales mantienen la ruta directa.
    try:
        with Image.open(image_path) as image:
            orientation = image.getexif().get(274, 1)
            image_mode = image.mode
    except Exception:
        orientation = 1
        image_mode = None
    if orientation in range(2, 9) or image_mode in {"CMYK", "LA", "1", "I;16"}:
        sanitized_path = _sanitize_image_file(image_path)
        return ImageClip(sanitized_path), sanitized_path

    # Primero se intenta abrir directamente la imagen original; si falla por metadatos
    # dañados, se intenta generar una copia sin metadatos.
    try:
        return ImageClip(image_path), image_path
    except Exception as exc:
        logger.warning(
            f"no se pudo abrir la imagen directamente, se prueba con una copia saneada: {image_path}, error: {str(exc)}"
        )
        sanitized_path = _sanitize_image_file(image_path)
        return ImageClip(sanitized_path), sanitized_path


_moviepy_reader_open_lock = threading.Lock()


def _open_video_clip_quietly(video_path: str, audio: bool = False) -> VideoFileClip:
    """
    Abre el archivo de vídeo en silencio, para que MoviePy 2.1.x no imprima directamente
    en stdout la información de detección de ffmpeg.

    Contexto:
    El `FFMPEG_VideoReader` de la versión de la dependencia actual contiene internamente
    `print(self.infos)` y `print(ffmpeg command)`, y al leer vídeos intermedios sin
    pista de audio muestra `audio_found: False`. Eso son solo metadatos del material de
    entrada y no significa que el vídeo final no tenga audio, pero confunde a los
    usuarios de la WebUI/terminal haciéndoles creer que la generación ha fallado.

    Implementación:
    1. Solo se redirige stdout en la breve ventana en que se abre el VideoFileClip;
    2. Por defecto `audio=False`, porque la fase de material de vídeo del proyecto no
       necesita conservar el sonido original del material,
       y el audio final se monta de forma común en la fase `generate_video()`;
    3. Si la librería de la dependencia llega a imprimir algo, se degrada a log debug,
       para poder diagnosticarlo si hace falta.
    """
    captured_stdout = io.StringIO()
    # redirect_stdout cambia un estado global del proceso. Aperturas de lectores
    # solapadas pueden restaurar los búferes de captura de las otras en lugar del stdout
    # original. Se serializa esta breve ventana de construcción; el procesamiento de
    # clips sigue en paralelo.
    with _moviepy_reader_open_lock:
        with redirect_stdout(captured_stdout):
            clip = VideoFileClip(video_path, audio=audio)

    moviepy_stdout = captured_stdout.getvalue().strip()
    if moviepy_stdout:
        logger.debug(
            "se silenció el stdout del lector de vídeo de MoviePy para "
            f"{video_path}, caracteres: {len(moviepy_stdout)}"
        )

    return clip


def close_clip(clip):
    if clip is None:
        return
        
    try:
        # cierra los recursos principales
        if hasattr(clip, 'reader') and clip.reader is not None:
            clip.reader.close()
            
        # cierra los recursos de audio
        if hasattr(clip, 'audio') and clip.audio is not None:
            if hasattr(clip.audio, 'reader') and clip.audio.reader is not None:
                clip.audio.reader.close()
            del clip.audio
            
        # cierra los recursos de máscara
        if hasattr(clip, 'mask') and clip.mask is not None:
            if hasattr(clip.mask, 'reader') and clip.mask.reader is not None:
                clip.mask.reader.close()
            del clip.mask
            
        # trata los clips hijos de los clips compuestos
        if hasattr(clip, 'clips') and clip.clips:
            for child_clip in clip.clips:
                if child_clip is not clip:  # evita posibles referencias circulares
                    close_clip(child_clip)
            
        # vacía la lista de clips
        if hasattr(clip, 'clips'):
            clip.clips = []
            
    except Exception as e:
        logger.error(f"no se pudo cerrar el clip: {str(e)}")
    
    del clip
    gc.collect()

def delete_files(files: List[str] | str):
    if isinstance(files, str):
        files = [files]

    # Al completar el vídeo en bucle, la misma ruta de clip temporal aparece varias
    # veces en la lista de montaje de FFmpeg. El montaje debe conservar los repetidos,
    # pero la limpieza solo puede borrar una vez; aquí se eliminan los duplicados
    # conservando el orden original, para que todos los llamadores tengan un
    # comportamiento idempotente y no se emitan FileNotFoundError seguidos tras un
    # primer borrado correcto.
    unique_files = dict.fromkeys(file for file in files if file)
    for file in unique_files:
        try:
            os.remove(file)
        except FileNotFoundError:
            # La limpieza admite que el archivo ya no exista, por ejemplo en la ruta de
            # fallo de FFmpeg o porque una limpieza concurrente ya lo recuperó; no es un
            # problema que tenga que resolver el usuario y no debe ensuciar el log de
            # generación.
            continue
        except OSError as e:
            # Los permisos, un sistema de archivos de solo lectura o una anomalía del
            # disco dejan archivos temporales reales, así que se conserva el warning
            # para localizar el problema del entorno a partir de la ruta concreta y el
            # error del sistema.
            logger.warning(f"no se pudo borrar el archivo temporal {file}: {str(e)}")


def get_bgm_file(bgm_type: str = "random", bgm_file: str = ""):
    if not bgm_type:
        return ""

    if bgm_file:
        try:
            resolved_bgm_file = bgm_service.resolve_bgm_file(bgm_file)
        except ValueError as exc:
            # El bgm_file de una solicitud a la API viene de la entrada del usuario y
            # solo se permite resolver al directorio de BGM del usuario o al de
            # canciones integradas, impidiendo que MoviePy lea archivos arbitrarios del
            # servidor como la configuración o las claves.
            logger.warning(
                f"se rechaza un archivo de bgm inseguro: {bgm_file}, error: {str(exc)}"
            )
            return ""
        return resolved_bgm_file

    if bgm_type == "random":
        files = bgm_service.list_bgm_files()
        # Cuando el directorio de música de fondo está vacío, se vuelve directamente a
        # "no usar BGM", para que random.choice([]) no lance una excepción.
        if not files:
            logger.warning("no se encontraron archivos de música de fondo")
            return ""
        return random.choice(files)

    return ""


def _fit_clip_to_canvas(
    clip,
    *,
    target_width: int,
    target_height: int,
    fit_mode: VideoFitMode | str = VideoFitMode.cover,
):
    """Redimensiona un clip a un lienzo exacto con cover/recorte o contain/letterbox."""
    source_width, source_height = (int(value) for value in clip.size)
    target_width = int(target_width)
    target_height = int(target_height)
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError(
            "las dimensiones del vídeo deben ser positivas: "
            f"source={source_width}x{source_height}, "
            f"target={target_width}x{target_height}"
        )

    mode = VideoFitMode(fit_mode)
    if (source_width, source_height) == (target_width, target_height):
        return clip

    # Las coincidencias exactas de relación de aspecto no necesitan ni recorte ni fondo.
    if source_width * target_height == source_height * target_width:
        return clip.resized(new_size=(target_width, target_height))

    width_scale = target_width / source_width
    height_scale = target_height / source_height

    if mode == VideoFitMode.cover:
        # ceil garantiza que el clip redimensionado cubre el lienzo completo pese al
        # redondeo de coma flotante. Cualquier exceso se elimina de forma simétrica.
        scale_factor = max(width_scale, height_scale)
        resized_width = max(target_width, math.ceil(source_width * scale_factor))
        resized_height = max(target_height, math.ceil(source_height * scale_factor))
        resized_clip = clip.resized(new_size=(resized_width, resized_height))
        crop_x = max(0, (resized_width - target_width) // 2)
        crop_y = max(0, (resized_height - target_height) // 2)
        return resized_clip.cropped(
            x1=crop_x,
            y1=crop_y,
            width=target_width,
            height=target_height,
        )

    # contain conserva el comportamiento heredado: muestra el fotograma de origen
    # completo, centrado sobre un lienzo negro cuando las relaciones de aspecto
    # difieren.
    scale_factor = min(width_scale, height_scale)
    resized_width = max(1, min(target_width, int(source_width * scale_factor)))
    resized_height = max(1, min(target_height, int(source_height * scale_factor)))
    background = ColorClip(
        size=(target_width, target_height), color=(0, 0, 0)
    ).with_duration(clip.duration)
    resized_clip = clip.resized(
        new_size=(resized_width, resized_height)
    ).with_position("center")
    return CompositeVideoClip(
        [background, resized_clip], size=(target_width, target_height)
    ).with_duration(clip.duration)


def combine_videos(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    video_transition_mode: VideoTransitionMode = None,
    max_clip_duration: int = 5,
    threads: int = 2,
    clip_speed: float = 1.0,
    video_fit_mode: VideoFitMode = VideoFitMode.cover,
    source_usage: dict[str, int] | None = None,
    source_groups: dict[str, str] | None = None,
    used_video_paths: List[str] | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> str:
    # Pydantic deja este valor por defecto opcional como la cadena "random" salvo que se
    # proporcione explícitamente. Se normaliza una vez para todos los accesos
    # posteriores al enum.
    video_concat_mode = VideoConcatMode(video_concat_mode or VideoConcatMode.random)
    audio_clip = AudioFileClip(audio_file)
    try:
        # Aquí solo hace falta leer la duración del audio de la narración para decidir
        # la longitud del montaje del material de vídeo; audio_clip ya no se usa
        # después. Se cierra en cuanto termina la lectura, para no filtrar el descriptor
        # de archivo en salidas anticipadas o rutas de excepción.
        audio_duration = audio_clip.duration
    finally:
        close_clip(audio_clip)
    logger.info(f"duración del audio: {audio_duration} segundos")
    logger.info(f"duración máxima del clip: {max_clip_duration} segundos")
    required_video_duration = _get_required_video_duration(audio_duration)
    logger.info(
        f"duración de vídeo necesaria: {required_video_duration:.2f} segundos "
        f"(duración del audio + {_VIDEO_DURATION_SAFETY_MARGIN:.2f}s de margen de seguridad)"
    )

    # Admite que la API se llame directamente sin pasar el modo de transición, para que
    # no falle después al acceder a .value.
    transition_value = getattr(video_transition_mode, "value", video_transition_mode)
    normalized_clip_speed = utils.normalize_clip_speed(clip_speed)
    if normalized_clip_speed != 1.0:
        # Solo se registra una vez el valor final efectivo, lo que ayuda a localizar
        # parámetros de la API fuera de rango que se han normalizado sin repetir el
        # mismo log en la ruta crítica clip a clip.
        logger.info(f"velocidad de reproducción del clip: {normalized_clip_speed:.2f}x")
    # max_clip_duration limita la duración de reproducción final en el vídeo, no la
    # duración que se lee del vídeo de origen. MoviePy reproduciendo 1,5 segundos de
    # origen a 0,5x da un clip de 3 segundos, y reproduciendo 6 segundos de origen a 2x
    # también da un clip de 3 segundos. Por eso, antes de cortar hay que deducir la
    # duración de origen según la velocidad; si se siguiera leyendo siempre 3 segundos
    # para luego ralentizar y recortar, mientras el siguiente segmento empieza en el
    # segundo 3 del vídeo de origen, se saltaría 1,5 segundos de imagen intermedia. Este
    # cálculo garantiza además que, a distintas velocidades, la línea de tiempo de
    # origen sea continua y sin solapes.
    source_clip_duration = max_clip_duration * normalized_clip_speed
    output_dir = os.path.dirname(combined_video_path)

    aspect = VideoAspect(video_aspect)
    fit_mode = VideoFitMode(video_fit_mode)
    video_width, video_height = aspect.to_resolution()

    processed_clips = []
    subclipped_items = []
    video_duration = 0
    for video_path in video_paths:
        clip = None
        try:
            clip = _open_video_clip_quietly(video_path)
            clip_duration = float(clip.duration)
            clip_w, clip_h = clip.size
            if (
                not math.isfinite(clip_duration)
                or clip_duration <= 0
                or not all(
                    math.isfinite(float(dimension)) and float(dimension) > 0
                    for dimension in (clip_w, clip_h)
                )
            ):
                raise ValueError("duración o dimensiones de vídeo no válidas")
        except Exception as exc:
            logger.warning(
                f"se omite una fuente de vídeo ilegible: path={video_path}, error={exc}"
            )
            continue
        finally:
            close_clip(clip)
        
        start_time = 0

        while start_time < clip_duration:
            end_time = min(start_time + source_clip_duration, clip_duration)

            # Conserva todos los segmentos válidos. Así no se pierde el material "cuyo
            # vídeo entero ya es más corto que max_clip_duration", ni se traga el
            # pequeño trozo final que sobra de un vídeo largo.
            if end_time > start_time:
                subclipped_items.append(
                    SubClippedVideoClip(
                        file_path=video_path,
                        start_time=start_time,
                        end_time=end_time,
                        width=clip_w,
                        height=clip_h,
                        source_file_path=video_path,
                    )
                )

            start_time = end_time
            if video_concat_mode.value == VideoConcatMode.sequential.value:
                break

    subclipped_items = _prioritize_unique_source_clips(
        subclipped_items=subclipped_items,
        concat_mode=video_concat_mode,
        **({"source_usage": source_usage, "source_groups": source_groups}
           if source_usage is not None else {}),
    )
        
    logger.debug(f"total de fragmentos recortados: {len(subclipped_items)}")
    
    # Añade los clips descargados una y otra vez hasta alcanzar la duración del audio
    # (max_duration)
    def process_one_clip(indexed_item):
        """Recorta, cambia la velocidad, aplica la transición y escribe un clip de origen;
        si falla devuelve None.
        """
        index, subclipped_item = indexed_item
        source_clip = None
        clip = None
        clip_file = None
        try:
            logger.debug(
                f"procesando el clip {index + 1}: {subclipped_item.width}x{subclipped_item.height}, "
                f"origen: {os.path.basename(subclipped_item.source_file_path)}"
            )
            source_clip = _open_video_clip_quietly(subclipped_item.file_path)
            clip = source_clip.subclipped(
                subclipped_item.start_time, subclipped_item.end_time
            )
            # La velocidad de reproducción es una propiedad del propio material y debe
            # aplicarse antes de la transición. Así, las transiciones de un segundo como
            # Fade/Slide no se convierten en 0,5 o 2 segundos siguiendo la velocidad del
            # material; el recorte posterior a la duración máxima sigue sirviendo de
            # protección ante errores de coma flotante o duraciones anómalas del
            # material, para que el clip final no supere el máximo configurado.
            if normalized_clip_speed != 1.0:
                clip = clip.with_speed_scaled(normalized_clip_speed)
            # Normaliza cada clip de origen antes de aplicar las transiciones. En modo
            # cover el clip llena el lienzo y se recortan los bordes sobrantes; contain
            # conserva el fotograma de origen completo y usa franjas negras para la zona
            # no usada.
            clip_w, clip_h = clip.size
            if clip_w != video_width or clip_h != video_height:
                clip_ratio = clip.w / clip.h
                video_ratio = video_width / video_height
                logger.debug(
                    "redimensionando el clip, "
                    f"origen: {clip_w}x{clip_h}, proporción: {clip_ratio:.2f}, "
                    f"destino: {video_width}x{video_height}, proporción: {video_ratio:.2f}, "
                    f"fit_mode: {fit_mode.value}"
                )
                clip = _fit_clip_to_canvas(
                    clip,
                    target_width=video_width,
                    target_height=video_height,
                    fit_mode=fit_mode,
                )

            shuffle_side = random.choice(["left", "right", "top", "bottom"])
            # El material corto y la reproducción más rápida pueden dejar menos de un
            # segundo. Se mantiene la transición elegida dentro de la línea de tiempo
            # del clip renderizado.
            transition_duration = min(1.0, clip.duration)
            if transition_value == VideoTransitionMode.fade_in.value:
                clip = video_effects.fadein_transition(clip, transition_duration)
            elif transition_value == VideoTransitionMode.fade_out.value:
                clip = video_effects.fadeout_transition(clip, transition_duration)
            elif transition_value == VideoTransitionMode.slide_in.value:
                clip = video_effects.slidein_transition(clip, transition_duration, shuffle_side)
            elif transition_value == VideoTransitionMode.slide_out.value:
                clip = video_effects.slideout_transition(clip, transition_duration, shuffle_side)
            elif transition_value == VideoTransitionMode.zoom_in.value:
                clip = video_effects.zoomin_transition(clip, 1)
            elif transition_value == VideoTransitionMode.zoom_out.value:
                clip = video_effects.zoomout_transition(clip, 1)
            elif transition_value == VideoTransitionMode.shuffle.value:
                transition_funcs = [
                    lambda c: video_effects.fadein_transition(c, transition_duration),
                    lambda c: video_effects.fadeout_transition(c, transition_duration),
                    lambda c: video_effects.slidein_transition(c, transition_duration, shuffle_side),
                    lambda c: video_effects.slideout_transition(c, transition_duration, shuffle_side),
                    lambda c: video_effects.zoomin_transition(c, 1),
                    lambda c: video_effects.zoomout_transition(c, 1),
                ]
                shuffle_transition = random.choice(transition_funcs)
                clip = shuffle_transition(clip)

            if clip.duration > max_clip_duration:
                clip = clip.subclipped(0, max_clip_duration)

            # Escribe cada clip candidato en un archivo temporal único. Los hilos no
            # deben compartir la misma ruta de salida. Combinaciones distintas pueden
            # compartir un directorio de tarea/salida. Se reserva una ruta propia para
            # que sus codificadores y su limpieza nunca compartan un clip.
            with tempfile.NamedTemporaryFile(
                dir=output_dir or ".", prefix="temp-clip-", suffix=".mp4", delete=False
            ) as temporary_clip:
                clip_file = temporary_clip.name
            _write_videofile_with_codec_fallback(
                clip,
                clip_file,
                codec=_get_configured_video_codec(),
                logger=None,
                fps=fps,
            )

            clip_duration_saved = clip.duration
            processed_clip = SubClippedVideoClip(
                file_path=clip_file,
                duration=clip_duration_saved,
                width=clip_w,
                height=clip_h,
                source_file_path=subclipped_item.source_file_path,
            )
            clip_file = None
            return processed_clip
        except Exception as exc:
            logger.error(f"no se pudo procesar el clip: {str(exc)}")
            return None
        finally:
            # El clip derivado comparte su lector de FFmpeg con el de origen. Si
            # el propio subclipping falló, se cierra en su lugar el origen original.
            close_clip(clip if clip is not None else source_clip)
            # MoviePy puede dejar un MP4 truncado aunque la codificación lance una
            # excepción. Nunca se devolvió, así que la limpieza de concat no puede
            # verlo. Se cierra primero el lector para que Windows pueda eliminar el
            # archivo parcial.
            if clip_file:
                delete_files(clip_file)

    # Los clips siempre se procesan en el pool de hilos. El log clip a clip es la única
    # información de progreso de esta fase, y solo vinculándolo al hilo que lanzó la
    # composición puede recogerlo el log de tareas de la WebUI.
    process_clip_in_task_scope = logging_utils.bind_log_scope(process_one_clip)
    clip_processing_workers = 1
    if len(subclipped_items) >= 2:
        clip_processing_workers = min(_get_clip_processing_concurrency(), len(subclipped_items))
    with ThreadPoolExecutor(
        max_workers=clip_processing_workers,
        thread_name_prefix="clip-process",
    ) as executor:
        next_candidate_index = 0
        while (
            next_candidate_index < len(subclipped_items)
            and video_duration < required_video_duration
        ):
            remaining_duration = required_video_duration - video_duration
            batch = []
            batch_duration = 0.0
            candidate_index = next_candidate_index
            while candidate_index < len(subclipped_items) and batch_duration < remaining_duration:
                subclipped_item = subclipped_items[candidate_index]
                source_duration = subclipped_item.end_time - subclipped_item.start_time
                output_duration = min(
                    max_clip_duration,
                    source_duration / normalized_clip_speed,
                )
                batch.append((candidate_index, subclipped_item))
                batch_duration += output_duration
                candidate_index += 1
            if not batch:
                break
            for processed_clip in executor.map(process_clip_in_task_scope, batch):
                if processed_clip is None:
                    continue
                processed_clips.append(processed_clip)
                video_duration += processed_clip.duration
                # Cada segmento de material 4K tarda más de diez segundos en procesarse;
                # informando segmento a segmento de la duración cubierta, el log y la
                # barra de progreso reflejan que esta fase sigue avanzando.
                logger.info(
                    f"clip procesado {len(processed_clips)}: "
                    f"{video_duration:.1f} de {required_video_duration:.1f}s cubiertos"
                )
                _report_clip_progress(
                    progress_callback, video_duration, required_video_duration
                )
            next_candidate_index = candidate_index
    
    # repite los clips procesados hasta que la duración del vídeo cubra la duración del
    # audio más el pequeño margen de seguridad.
    if video_duration < required_video_duration:
        logger.warning(
            f"la duración del vídeo ({video_duration:.2f}s) es menor que la necesaria "
            f"({required_video_duration:.2f}s); se repiten clips en bucle para igualar la duración del audio."
        )
        base_clips = processed_clips.copy()
        for clip in itertools.cycle(base_clips):
            if video_duration >= required_video_duration:
                break
            processed_clips.append(clip)
            video_duration += clip.duration
        logger.info(
            f"duración del vídeo: {video_duration:.2f}s, duración del audio: {audio_duration:.2f}s, "
            f"duración necesaria: {required_video_duration:.2f}s, "
            f"clips repetidos: {len(processed_clips)-len(base_clips)}"
        )
     
    # fusiona los clips de vídeo de forma progresiva, sin cargar todos los vídeos a la
    # vez para no desbordar la memoria
    logger.info("empieza la fusión de los clips")
    if not processed_clips:
        if video_paths:
            raise RuntimeError("no hay clips de vídeo legibles para fusionar")
        logger.warning("no hay clips para fusionar")
        return combined_video_path
    
    clip_files = [clip.file_path for clip in processed_clips]
    logger.info(f"montando {len(clip_files)} clips con ffmpeg")
    try:
        concat_video_clips_with_ffmpeg(
            clip_files=clip_files,
            output_file=combined_video_path,
            threads=threads,
            output_dir=output_dir,
            max_duration=audio_duration,
        )
        if used_video_paths is not None:
            # Excluye los clips del margen de seguridad que FFmpeg recorta por completo
            # de la salida.
            elapsed = 0.0
            for clip in processed_clips:
                if elapsed >= audio_duration:
                    break
                used_video_paths.append(clip.source_file_path)
                elapsed += clip.duration
    finally:
        # Los fallos y tiempos agotados de FFmpeg no deben dejar abandonado un MP4
        # codificado por clip. Los clips repetidos comparten ruta; delete_files ya
        # elimina los duplicados.
        delete_files(clip_files)
            
    logger.info("composición del vídeo completada")
    return combined_video_path


def wrap_text(text, max_width, font="Arial", fontsize=60):
    # El ajuste de línea de los subtítulos debe hacerse antes de crear realmente el
    # TextClip; si no, MoviePy calcularía la zona de renderizado solo con el texto
    # original. Aquí se mide el ancho con PIL según la fuente y el tamaño actuales, para
    # que cada línea quede lo más posible dentro del ancho útil del vídeo, evitando que
    # los tamaños grandes o las frases largas en chino se salgan directamente de la
    # imagen.
    if "\n" in text:
        # Los saltos de línea duros del texto SRT son líneas de maquetación separadas.
        # Medir un token que cruza un salto de línea haría que Pillow contara ambas
        # líneas como una sola cadena ancha, y luego el ajuste por caracteres podría
        # partir una palabra que sí cabía.
        wrapped_lines = [
            wrap_text(line, max_width, font=font, fontsize=fontsize)
            for line in text.split("\n")
        ]
        return "\n".join(line for line, _ in wrapped_lines), sum(
            height for _, height in wrapped_lines
        )

    font = ImageFont.truetype(font, fontsize)
    max_width = int(max_width)

    # getbbox() devuelve la "altura de tinta visible del glifo actual", no la altura de
    # línea de la fuente. Por ejemplo, un texto en inglés con solo caracteres sin
    # descendente como A, m o n carece de descent, y con varias líneas ese error se
    # acumula línea a línea, hasta que el lienzo recorta la última línea del TextClip.
    # ascent + descent vienen de la propia fuente, no dependen del idioma ni de la
    # combinación de caracteres, y coinciden con el modelo de dibujo por baseline de
    # MoviePy.
    ascent, descent = font.getmetrics()
    line_height = int(ascent + descent)
    if line_height <= 0:
        # Las fuentes TrueType/OpenType normales no llegan aquí; se mantienen un log de
        # diagnóstico y un respaldo por tamaño de fuente, para que una fuente dañada o
        # poco habitual que devuelva métricas anómalas no genere subtítulos de altura
        # cero.
        logger.warning(
            "métricas de la fuente de subtítulos no válidas; se usa el tamaño de la fuente: "
            f"ascent={ascent}, descent={descent}, fontsize={fontsize}"
        )
        line_height = max(1, int(fontsize))

    def get_text_size(inner_text):
        inner_text = inner_text.strip()
        if not inner_text:
            return 0, line_height
        left, top, right, bottom = font.getbbox(inner_text)
        # bbox sigue sirviendo para medir el ancho real que necesita el ajuste de línea;
        # la altura debe usar siempre la altura de línea estable de la fuente.
        return right - left, line_height

    width, height = get_text_size(text)
    if width <= max_width:
        # Las entradas SRT permiten que el autor haga saltos de línea manuales. Aunque
        # todo el texto no necesite volver a ajustarse por ancho, la altura del lienzo
        # debe calcularse con el número de líneas existente; si no, se recortarían la
        # segunda línea y las siguientes.
        return text, (text.count("\n") + 1) * line_height

    def attached_text_units(token):
        units = []
        for char in token:
            if units and unicodedata.category(char).startswith("M"):
                units[-1] += char
            else:
                units.append(char)
        return units

    def split_long_token(token):
        # Cuando un token por sí solo ya es demasiado ancho (habitual en frases largas
        # en chino sin espacios o palabras muy largas en inglés), se degrada a una
        # división por caracteres. Lo clave es: al detectar que candidate es demasiado
        # ancho, primero se confirma el current anterior que todavía era válido y luego
        # se pone el carácter actual en la línea siguiente; no se puede meter el
        # carácter demasiado ancho en la línea anterior.
        lines = []
        current = ""
        for char in attached_text_units(token):
            candidate = f"{current}{char}"
            candidate_width, _ = get_text_size(candidate)
            if candidate_width <= max_width or not current:
                current = candidate
                continue
            lines.append(current)
            current = char
        if current:
            lines.append(current)
        return lines

    lines = []
    current = ""
    words = text.split(" ")
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)

        word_width, _ = get_text_size(word)
        if word_width <= max_width:
            current = word
        else:
            # El último fragmento todavía no llena la línea, así que las palabras
            # siguientes deben seguir intentando colocarse a su lado; si no, después de
            # cada palabra demasiado larga saldría una línea extra.
            pieces = split_long_token(word)
            lines.extend(pieces[:-1])
            current = pieces[-1] if pieces else ""

    if current:
        lines.append(current)

    line_start_punctuation = "，。！？；：、,.!?;:)]}）】》」』”’"
    for index in range(1, len(lines)):
        # Cuando una frase larga en chino se divide por caracteres, la puntuación de
        # cierre final, como el punto o la coma, puede quedar sola en la línea
        # siguiente, estirando anormalmente el fondo del subtítulo y pareciendo
        # visualmente un punto caído bajo el texto. Aquí, sin rediseñar el algoritmo de
        # ajuste de línea, se mueve el último carácter de la línea anterior delante de
        # la línea de la puntuación, para que la puntuación acompañe al texto;
        # compatible con la puntuación de cierre habitual en chino e inglés.
        if not lines[index] or lines[index][0] not in line_start_punctuation:
            continue
        if len(lines[index - 1]) <= 1:
            continue

        previous_units = attached_text_units(lines[index - 1])
        if len(previous_units) <= 1:
            continue
        last_unit = previous_units[-1]
        candidate = f"{last_unit}{lines[index]}"
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            lines[index] = candidate
            lines[index - 1] = lines[index - 1][:-len(last_unit)]

    result = "\n".join(line.strip() for line in lines if line.strip()).strip()
    # La altura se basa en el resultado final. Los saltos de línea explícitos del texto
    # original pueden quedar dentro de un token, y en ese caso la longitud de la lista
    # lines temporal no es igual al número de líneas que renderiza realmente MoviePy.
    height = (result.count("\n") + 1) * line_height
    return result, height


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    # Coincide con la sintaxis de color de Pillow que acepta la comprobación previa de
    # subtítulos. El clip de fondo fija la opacidad aparte del valor del color.
    if isinstance(color, str):
        try:
            return ImageColor.getcolor(color, "RGB")
        except (ValueError, TypeError):
            pass
    return (0, 0, 0)


def _rounded_subtitle_background_clip(
    width: int,
    height: int,
    color: str,
    alpha: int = 140,
    radius: int = 16,
) -> ImageClip:
    # El nuevo fondo de subtítulos solo se usa cuando el usuario lo activa
    # explícitamente: se dibuja una placa semitransparente con esquinas redondeadas en
    # una imagen RGBA, que luego entra en la composición como ImageClip transparente de
    # MoviePy. Así la ruta por defecto no cambia en absoluto, y se puede probar con poco
    # coste un efecto visual de subtítulos más suave.
    rgb = _hex_to_rgb(color)
    safe_alpha = max(0, min(255, int(alpha)))
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [0, 0, max(0, width - 1), max(0, height - 1)],
        radius=max(0, int(radius)),
        fill=(rgb[0], rgb[1], rgb[2], safe_alpha),
    )
    return ImageClip(np.array(img), transparent=True)


def _get_visible_center_position(
    text_clip: TextClip,
    container_width: int,
    container_height: int,
) -> tuple[int, int]:
    """
    Coloca el TextClip en el centro del contenedor de fondo según los píxeles visibles
    reales del texto.

    El TextClip de MoviePy crea un lienzo transparente según la altura de línea y la
    baseline de la fuente. En muchas fuentes, el glifo visible no está en el centro
    geométrico de ese lienzo, y usar directamente `with_position("center")` centraría
    todo el lienzo transparente, de modo que el subtítulo parecería desplazado hacia
    arriba o hacia abajo. Aquí se lee la máscara transparente del TextClip y se calcula
    el desplazamiento solo a partir del bbox con píxeles reales, para que el texto que
    ve el usuario quede visualmente centrado en el fondo del subtítulo.
    """
    x = int(round((container_width - text_clip.w) / 2))
    y = int(round((container_height - text_clip.h) / 2))

    try:
        if text_clip.mask is None:
            return x, y

        mask_frame = text_clip.mask.get_frame(0)
        ys, _ = np.where(mask_frame > 0.01)
        if len(ys) == 0:
            return x, y

        visible_top = int(ys.min())
        visible_bottom = int(ys.max())
        visible_height = visible_bottom - visible_top + 1
        y = int(round((container_height - visible_height) / 2 - visible_top))
    except Exception as exc:
        logger.debug(f"no se pudo centrar el texto del subtítulo según la máscara visible: {str(exc)}")

    return x, y


def validate_subtitle_colors(params: VideoParams) -> None:
    """Rechaza los colores que Pillow no puede renderizar, antes de que se ejecuten los
    servicios de generación.
    """
    if not params.subtitle_enabled:
        return
    colors = [("text_fore_color", params.text_fore_color)]
    # El renderizado trunca el ancho a un entero; un contorno de ancho cero no se usa.
    if int(params.stroke_width) > 0:
        colors.append(("stroke_color", params.stroke_color))
    if isinstance(params.text_background_color, str):
        colors.append(("text_background_color", params.text_background_color))
    for field, value in colors:
        if value is None:
            continue
        try:
            ImageColor.getcolor(value, "RGBA")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"color de subtítulo no válido: {field}") from exc


def subtitle_colors_are_indistinguishable(params: VideoParams) -> bool:
    """Comprueba si el texto y el fondo del subtítulo son del mismo color, para avisar al
    usuario de que puede que no se lea el subtítulo.
    """
    if not params.subtitle_enabled or not params.text_background_color:
        return False

    def normalize_color(value):
        if isinstance(value, bool):
            return "#000000" if value else ""
        return str(value or "").strip().lower()

    text_color = normalize_color(params.text_fore_color)
    background_color = normalize_color(params.text_background_color)
    return bool(text_color and text_color == background_color)


@lru_cache(maxsize=64)
def _subtitle_font_supports_sample(font_path: str, sample: str) -> bool:
    """Comprueba si la fuente contiene los glifos que necesita el texto de muestra,
    guardando en caché los resultados de comprobaciones repetidas.
    """
    try:
        font = ImageFont.truetype(font_path, 30)
        missing_mask = font.getmask("\U0010ffff")
        missing_signature = (
            missing_mask.size,
            missing_mask.getbbox(),
            bytes(missing_mask),
        )
        for char in sample:
            char_mask = font.getmask(char)
            char_signature = (
                char_mask.size,
                char_mask.getbbox(),
                bytes(char_mask),
            )
            if char_mask.getbbox() is None or char_signature == missing_signature:
                return False
        return True
    except Exception as e:
        # Un fallo al detectar la fuente no debe impedir que el usuario genere; se
        # conserva el log para investigar problemas de compatibilidad del entorno.
        logger.warning(f"no se pudieron inspeccionar los glifos de la fuente de subtítulos: {font_path}, {e}")
        return True


def subtitle_font_supports_text(font_path: str, text: str) -> bool:
    """Comprueba si la fuente puede dibujar las letras y los dígitos del texto, ignorando
    los espacios y la puntuación.
    """
    sample = "".join(
        dict.fromkeys(
            char
            for char in str(text or "")
            if unicodedata.category(char)[0] in {"L", "N"}
        )
    )[:64]
    if not sample:
        return True
    return _subtitle_font_supports_sample(font_path, sample)


def generate_video(
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_file: str,
    params: VideoParams,
    bgm_file_override: str | None = None,
) -> bool:
    """
    Compone el vídeo final y devuelve si el tratamiento de la música de fondo de esta
    vez ha tenido éxito.

    El valor devuelto solo describe el estado del tratamiento de la BGM: devuelve True
    si no se pidió BGM o si se mezcló bien, y False si se pidió BGM pero falló la carga,
    el efecto o la mezcla. Aunque falle la BGM se sigue generando el vídeo solo con la
    narración, y la capa de orquestación de tareas decide si muestra al usuario un aviso
    de degradación.
    """
    aspect = VideoAspect(params.video_aspect)
    video_width, video_height = aspect.to_resolution()

    logger.info(f"generando el vídeo: {video_width} x {video_height}")
    logger.info(f"  ① video: {video_path}")
    logger.info(f"  ② audio: {audio_path}")
    logger.info(f"  ③ subtitle: {subtitle_path}")
    logger.info(f"  ④ output: {output_file}")

    # https://github.com/harry0703/MoneyPrinterTurbo/issues/217 PermissionError:
    # [WinError 32] The process cannot access the file because it is being used by
    # another process: 'final-1.mp4.tempTEMP_MPY_wvf_snd.mp3' escribe en el mismo
    # directorio que el archivo de salida
    output_dir = os.path.dirname(output_file)

    font_path = ""
    if params.subtitle_enabled:
        if not params.font_name:
            params.font_name = "STHeitiMedium.ttc"
        # Aunque la entrada de la API ya lo comprueba antes, la WebUI, la CLI y las
        # llamadas internas pueden llegar directamente a la capa de renderizado; se
        # valida siempre con la ruta real que la fuente permanece en resource/fonts,
        # bloqueando rutas absolutas, recorridos ../ y enlaces simbólicos que apuntan
        # fuera del directorio, antes de dejar que PIL/MoviePy la abran.
        font_path = file_security.resolve_path_within_directory(
            utils.font_dir(), params.font_name
        )
        if os.name == "nt":
            font_path = font_path.replace("\\", "/")

        logger.info(f"  ⑤ font: {font_path}")

    def resolve_subtitle_background_color():
        # Compatibilidad con parámetros históricos: en la API, `text_background_color`
        # puede ser tanto un booleano como una cadena de color real. Aquí se normaliza
        # siempre, para que pasar True/False directamente a TextClip no produzca un
        # renderizado imprevisible.
        if isinstance(params.text_background_color, bool):
            return "#000000" if params.text_background_color else None
        return params.text_background_color

    def create_text_clip(subtitle_item):
        params.font_size = int(params.font_size)
        params.stroke_width = int(params.stroke_width)
        phrase = subtitle_item[1]
        max_width = video_width * 0.9
        bg_color = resolve_subtitle_background_color()
        rounded_bg_enabled = bool(
            getattr(params, "rounded_subtitle_background", False) and bg_color
        )
        has_subtitle_background = bool(bg_color)
        # El fondo redondeado se genera según el ancho real del texto, y los márgenes
        # laterales deben ser más contenidos; el fondo rectangular antiguo mantiene un
        # margen de seguridad mayor, para que los subtítulos largos de configuraciones
        # históricas no toquen el borde ni se recorten.
        padding_ratio = 0.4 if rounded_bg_enabled else 0.6
        pad_x = int(params.font_size * padding_ratio) if has_subtitle_background else 0
        # El fondo del subtítulo necesita un margen interior claro a izquierda y derecha
        # del texto. Primero se resta el padding del ancho útil y luego se ajustan las
        # líneas, para que un texto largo en inglés o un tamaño grande que llene justo
        # el 90 % del ancho del vídeo no quede pegado al borde del fondo, pareciendo
        # recortado. Tanto el fondo rectangular normal como el redondeado siguen esta
        # lógica; los subtítulos sin fondo mantienen el ancho máximo original.
        text_max_width = max(1, int(max_width) - 2 * pad_x)
        wrapped_txt, txt_height = wrap_text(
            phrase,
            max_width=text_max_width,
            font=font_path,
            fontsize=params.font_size,
        )
        interline = int(params.font_size * 0.25)
        line_count = wrapped_txt.count("\n") + 1
        vertical_padding = int(params.font_size * 0.35)
        # Pillow/MoviePy extienden el contorno hacia arriba y hacia abajo del glifo, y
        # lo cuentan en la altura de avance de cada línea. Si solo se añadiera una vez
        # el margen del contorno fuera de todo el bloque de subtítulo, un texto de
        # varias líneas con contorno grueso seguiría acumulando error línea a línea.
        # Aquí se cuenta el espacio del contorno a ambos lados según el número real de
        # líneas: el contorno fino por defecto solo añade un poco de altura, y "fuente
        # pequeña + contorno grueso + varias líneas" también se muestra completo.
        stroke_padding = int(params.stroke_width * 2 * line_count)
        text_clip_margin_y = max(
            int(params.font_size * 0.3), int(params.stroke_width * 2)
        )
        # Con `method=label`, MoviePy reduce automáticamente la altura del cuadro de
        # texto, y con subtítulos de varias líneas, contorno o color de fondo tiende a
        # recortar la mitad inferior de la última línea. Aquí se pasa explícitamente una
        # altura más conservadora, que incluye el interlineado y un margen extra arriba
        # y abajo, para que tanto el cuadro de fondo del subtítulo como el propio texto
        # se rendericen completos.
        clip_h = int(
            txt_height
            + vertical_padding
            + (interline * line_count)
            + stroke_padding
        )

        if rounded_bg_enabled:
            # El fondo redondeado debe ajustarse al ancho del texto, en lugar de usar el
            # 90 % del ancho del vídeo. Aquí se mide primero con PIL la línea de texto
            # más larga y se le añade el margen interior horizontal, para que los
            # subtítulos cortos no tengan una placa demasiado ancha.
            try:
                font = ImageFont.truetype(font_path, params.font_size)
                text_w = max(
                    int(font.getbbox(line)[2] - font.getbbox(line)[0])
                    for line in wrapped_txt.split("\n")
                )
            except Exception as exc:
                logger.warning(
                    f"no se pudo medir el ancho del texto del subtítulo; se usa el ancho máximo: {str(exc)}"
                )
                text_w = int(max_width)

            box_w = max(1, min(int(max_width), text_w + 2 * pad_x))
            radius = max(8, int(params.font_size * 0.4))
            text_clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=(box_w, None),
                text_align="center",
                margin=(0, text_clip_margin_y),
            )
            clip_h = max(clip_h, text_clip.h)
            bg_clip = _rounded_subtitle_background_clip(
                width=box_w,
                height=clip_h,
                color=bg_color,
                alpha=140,
                radius=radius,
            )
            text_position = _get_visible_center_position(text_clip, box_w, clip_h)
            _clip = CompositeVideoClip(
                [bg_clip, text_clip.with_position(text_position)],
                size=(box_w, clip_h),
            )
        elif bg_color:
            size = (
                int(max_width),
                clip_h,
            )
            text_clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=(int(max_width), None),
                text_align="center",
                margin=(0, text_clip_margin_y),
            )
            size = (size[0], max(size[1], text_clip.h))
            bg_clip = _rounded_subtitle_background_clip(
                width=size[0],
                height=size[1],
                color=bg_color,
                alpha=255,
                radius=0,
            )
            text_position = _get_visible_center_position(text_clip, size[0], size[1])
            _clip = CompositeVideoClip(
                [bg_clip, text_clip.with_position(text_position)],
                size=size,
            )
        else:
            size = (
                int(max_width),
                clip_h,
            )
            _clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=size,
                text_align="center",
            )
        duration = subtitle_item[0][1] - subtitle_item[0][0]
        _clip = _clip.with_start(subtitle_item[0][0])
        _clip = _clip.with_end(subtitle_item[0][1])
        _clip = _clip.with_duration(duration)

        # La animación de rebote solo se activa cuando el usuario la elige
        # explícitamente; por defecto, none sigue exactamente la ruta de renderizado de
        # subtítulos original.
        anim_type = getattr(params, "subtitle_animation", "none")
        if anim_type in ("pop_spring", "spring", "pop"):
            _clip = _apply_subtitle_spring_animation(_clip, duration)

        if params.subtitle_position == "bottom":
            _clip = _clip.with_position(("center", video_height * 0.95 - _clip.h))
        elif params.subtitle_position == "top":
            _clip = _clip.with_position(("center", video_height * 0.05))
        elif params.subtitle_position in ("two_thirds_bottom", "two_thirds", "2/3_bottom"):
            # 2/3 desde abajo = 1/3 desde arriba: y = (video_height - _clip.h) * (1/3)
            y_two_thirds = (video_height - _clip.h) / 3.0
            _clip = _clip.with_position(("center", y_two_thirds))
        elif params.subtitle_position == "custom":
            # Asegura que el subtítulo quede completamente dentro de los límites de la
            # pantalla
            margin = 10  # Margen adicional, en píxeles
            max_y = video_height - _clip.h - margin
            min_y = margin
            custom_y = (video_height - _clip.h) * (params.custom_position / 100)
            custom_y = max(
                min_y, min(custom_y, max_y)
            )  # Limita el valor de y al rango válido
            _clip = _clip.with_position(("center", custom_y))
        else:  # centro
            _clip = _clip.with_position(("center", "center"))
        return _clip

    # CompositeAudioClip.close() de MoviePy no cierra los AudioFileClip hijos. Aquí se
    # usa un ExitStack para tener explícitamente todos los lectores de archivos
    # originales, asegurando que en todas las rutas (éxito, excepción en los subtítulos,
    # fallo de la mezcla o fallo al escribir el vídeo) se liberen los subprocesos de
    # FFmpeg, sobre todo para evitar archivos bloqueados en Windows.
    with ExitStack() as clip_stack:
        source_video_clip = clip_stack.enter_context(
            _open_video_clip_quietly(video_path)
        )
        voice_source_clip = clip_stack.enter_context(AudioFileClip(audio_path))
        video_clip = source_video_clip
        audio_clip = voice_source_clip.with_effects(
            [afx.MultiplyVolume(1.0 if params.voice_volume is None else params.voice_volume)]
        )

        def make_textclip(text):
            return TextClip(
                text=text,
                font=font_path,
                font_size=params.font_size,
            )

        if params.subtitle_enabled and subtitle_path and os.path.exists(subtitle_path):
            sub = clip_stack.enter_context(
                SubtitlesClip(
                    subtitles=moviepy_subtitles(subtitle_path),
                    encoding="utf-8",
                    make_textclip=make_textclip,
                )
            )
            text_clips = []
            for item in sub.subtitles:
                clip = create_text_clip(subtitle_item=item)
                text_clips.append(clip)
            video_clip = CompositeVideoClip([video_clip, *text_clips])
            clip_stack.callback(video_clip.close)

        bgm_enabled = bgm_service.should_use_bgm(
            params.bgm_type, params.bgm_volume
        )
        if not bgm_enabled and params.bgm_type:
            # Todas las fuentes de BGM comparten esta regla de cortocircuito. Con un
            # volumen no mayor que 0 no se puede resolver un archivo aleatorio o
            # personalizado ni cargar el archivo devuelto por un proveedor, para evitar
            # E/S y mezclas sin sentido.
            logger.info(
                f"se omite la música de fondo porque el volumen no es positivo: "
                f"type={params.bgm_type}, volume={params.bgm_volume}"
            )

        # La música de un proveedor la puede pasar directamente la capa de orquestación
        # de tareas como archivo. None significa seguir con la resolución de BGM
        # aleatoria/personalizada, y una cadena vacía desactiva explícitamente la BGM de
        # este vídeo; pero cualquier fuente debe pasar antes la regla común de volumen.
        bgm_file = ""
        if bgm_enabled:
            bgm_file = (
                bgm_file_override
                if bgm_file_override is not None
                else get_bgm_file(
                    bgm_type=params.bgm_type,
                    bgm_file=params.bgm_file,
                )
            )
        bgm_mix_succeeded = True
        if bgm_file:
            try:
                bgm_effects = [afx.MultiplyVolume(params.bgm_volume)]
                # La música aleatoria/personalizada que resuelve el servicio puede ser
                # más corta que el vídeo final y debe repetirse en bucle para cubrirlo;
                # el archivo que pasa la capa de tareas mediante override indica que el
                # proveedor ya la ha ajustado a la duración. Aquí se decide si repetir
                # en bucle según el origen del archivo, para no tener que modificar una
                # lista blanca de nombres cada vez que se añade un proveedor.
                if bgm_file_override is None:
                    bgm_effects.append(afx.AudioLoop(duration=video_clip.duration))
                # Aplica el fundido a toda la línea de tiempo de reproducción, no a cada
                # bucle de origen.
                bgm_effects.append(afx.AudioFadeOut(3))
                bgm_source_clip = clip_stack.enter_context(AudioFileClip(bgm_file))
                bgm_clip = bgm_source_clip.with_effects(bgm_effects)
                audio_clip = CompositeAudioClip([audio_clip, bgm_clip])
            except Exception:
                bgm_mix_succeeded = False
                # Registra la traza completa y un contexto estable, para distinguir los
                # fallos de decodificación del archivo, de los efectos de MoviePy y de
                # CompositeAudioClip; el contenido del archivo y la clave API no llegan
                # al log.
                logger.exception(
                    f"no se pudo mezclar la música de fondo: type={params.bgm_type}, "
                    f"file={bgm_file}"
                )

        final_video_clip = video_clip.with_audio(audio_clip)
        clip_stack.callback(final_video_clip.close)
        # Se usa explícitamente la frecuencia de muestreo del audio de entrada; si no se
        # puede obtener, se vuelve a los 44100 Hz por defecto de MoviePy. Así se reducen
        # las fluctuaciones de calidad por volver a remuestrear en distintos entornos,
        # sobre todo en Docker.
        output_audio_fps = int(getattr(audio_clip, "fps", 0) or 44100)
        with _stage_heartbeat("renderizado del vídeo final"):
            _write_videofile_with_codec_fallback(
                final_video_clip,
                output_file=output_file,
                codec=_get_configured_video_codec(),
                atomic_output=True,
                audio_codec=audio_codec,
                audio_fps=output_audio_fps,
                audio_bitrate=audio_bitrate,
                temp_audiofile_path=_get_temp_audio_dir(output_dir),
                threads=params.n_threads or 2,
                logger=None,
                fps=fps,
            )
        return bgm_mix_succeeded


def render_image_zoom_video(
    image_path: str, clip_duration: int = 5, image_motion: ImageMotion = ImageMotion.zoom
) -> str:
    """
    Renderiza una sola imagen como un clip mp4 estático o con zoom lento, y devuelve la
    ruta del archivo de salida.

    El preprocesado del material local y el material de texto a imagen compatible con
    OpenAI comparten esta lógica de renderizado "imagen → clip": por defecto se mantiene
    un zoom dinámico de aproximadamente un 3 % por segundo, y el modo estático solo fija
    la duración de reproducción. Este ajuste es independiente de la transición del clip.
    Las excepciones de renderizado las trata quien llama según el contrato de fallo de
    cada fuente de material.
    """
    motion = ImageMotion(image_motion)
    clip, _ = _open_image_clip_with_fallback(image_path)
    clip = clip.with_duration(clip_duration).with_position("center")
    temp_path = ""
    try:
        # Solo el modo de zoom cambia el tamaño de la imagen; el modo estático conserva
        # la composición original y el tratamiento del fondo transparente.
        image_clip = (
            clip.resized(lambda t: 1 + (clip_duration * 0.03) * (t / clip.duration))
            if motion == ImageMotion.zoom
            else clip
        )

        # Opcionalmente, crea un clip de vídeo compuesto que contenga el clip con zoom.
        # Es útil si quieres añadir otros elementos al vídeo.
        final_clip = CompositeVideoClip([image_clip], bg_color=(0, 0, 0))
        try:
            # La duración cambia el contenido renderizado, así que debe formar parte de
            # la identidad de la salida. Tareas distintas pueden renderizar la misma
            # imagen a la vez; solo se publica un MP4 completo después de que MoviePy lo
            # cierre.
            source_digest = hashlib.sha256(os.fsencode(os.path.abspath(image_path)))
            source_frame = clip.get_frame(0)
            source_digest.update(str(source_frame.shape).encode())
            source_digest.update(source_frame.tobytes())
            if clip.mask is not None:
                source_digest.update(clip.mask.get_frame(0).tobytes())
            source_identity = source_digest.hexdigest()
            duration_identity = hashlib.sha256(str(clip_duration).encode()).hexdigest()[:16]
            video_file = os.path.join(
                os.path.dirname(image_path),
                f"{source_identity}.{motion.value}-{duration_identity}.mp4",
            )
            descriptor, temp_path = tempfile.mkstemp(
                prefix=".image-zoom-",
                suffix=".mp4",
                dir=os.path.dirname(os.path.abspath(video_file)),
            )
            os.close(descriptor)
            final_clip.write_videofile(
                temp_path, fps=30, logger=None, ffmpeg_params=_BT709_FFMPEG_PARAMS
            )
        finally:
            close_clip(final_clip)
        os.replace(temp_path, video_file)
        temp_path = ""
        return video_file
    finally:
        close_clip(clip)
        if temp_path:
            delete_files(temp_path)


def preprocess_video(
    materials: List[MaterialInfo], clip_duration=4, image_motion: ImageMotion = ImageMotion.zoom
):
    # En algunos escenarios de regeneración, la WebUI puede pasar una lista de material
    # vacía; aquí se devuelve directamente un resultado vacío, para no lanzar una
    # excepción NoneType.
    if not materials:
        return []

    # Solo se devuelve el material que ha pasado la validación del preprocesado, para
    # que las imágenes de baja resolución no sigan al flujo posterior de composición del
    # vídeo.
    valid_materials = []
    local_videos_dir = utils.storage_dir("local_videos", create=True)

    for material in materials:
        if not material.url:
            continue

        try:
            material_source_path = file_security.resolve_path_within_directory(
                local_videos_dir, material.url
            )
        except ValueError as exc:
            # Las rutas de material del video_source local vienen de los parámetros de
            # la API y deben limitarse al directorio de material propio. Se permite que
            # el usuario pase un nombre de archivo, y también se admiten las rutas
            # absolutas devueltas históricamente, pero no se permite escapar a otros
            # directorios del sistema, para impedir la lectura de archivos arbitrarios o
            # el sondeo de archivos locales sensibles a través de MoviePy.
            logger.warning(
                f"se omite un material local inseguro: {material.url}, "
                f"local_videos_dir: {local_videos_dir}, error: {str(exc)}"
            )
            continue

        ext = utils.parse_extension(material_source_path)
        is_image = ext in const.FILE_TYPE_IMAGES
        try:
            # El material de imagen se lee directamente como imagen, para no pasar antes
            # por VideoFileClip, equivocarse y activar una rama de respaldo inestable.
            if is_image:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
            else:
                clip = _open_video_clip_quietly(material_source_path)
        except Exception:
            # Con una extensión no estándar o si falla la detección, se vuelve después
            # al modo imagen, compatible con el caso histórico de pasar directamente la
            # ruta de una imagen local.
            try:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
                # El decodificador que tiene éxito determina el tipo de material,
                # incluso cuando el nombre del archivo subido tiene un sufijo de vídeo o
                # desconocido.
                is_image = True
            except Exception as exc:
                logger.warning(
                    f"se omite un material local ilegible: {material.url}, error: {str(exc)}"
                )
                continue
        try:
            width = clip.size[0]
            height = clip.size[1]
            if not is_material_resolution_acceptable(width, height):
                logger.warning(
                    f"material de baja resolución: {width}x{height}, se requiere un mínimo de "
                    f"{_MIN_MATERIAL_DIMENSION}x{_MIN_MATERIAL_DIMENSION} "
                    f"(tolerancia {_MIN_DIMENSION_TOLERANCE}px)"
                )
                # Al detectar material de baja resolución se cierran de inmediato los
                # recursos, y ese material no se devuelve al flujo posterior.
                close_clip(clip)
                continue

            if is_image:
                logger.info(f"procesando la imagen: {material_source_path}")
                # Al detectar el tamaño ya se abrió el material una vez; aquí se libera
                # primero el handle de detección y luego se renderiza el clip de imagen
                # para exportar.
                close_clip(clip)
                video_file = render_image_zoom_video(
                    material_source_path, clip_duration, image_motion=image_motion
                )
                material.url = video_file
                logger.success(f"imagen procesada: {video_file}")
            else:
                # El material de vídeo normal solo necesita leer el tamaño para
                # validarlo, y tras la validación se libera el handle de inmediato.
                close_clip(clip)
                # Actualiza url a la ruta absoluta resuelta para que las etapas
                # posteriores (combine_videos) puedan abrir el archivo sin volver a
                # resolverlo.
                material.url = material_source_path
        except Exception:
            close_clip(clip)
            raise

        valid_materials.append(material)

    return valid_materials
