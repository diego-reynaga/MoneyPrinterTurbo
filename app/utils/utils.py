import json
import math
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
import threading
from typing import Any, Iterable
from uuid import uuid4

from loguru import logger

from app import paths
from app.models import const


def get_response(status: int, data: Any = None, message: str = ""):
    obj = {
        "status": status,
    }
    if data is not None:
        obj["data"] = data
    if message:
        obj["message"] = message
    return obj


def to_json(obj):
    try:
        # Define una función auxiliar para tratar distintos tipos de objetos
        def serialize(o):
            # Si el objeto es de un tipo serializable, se devuelve directamente
            if isinstance(o, (int, float, bool, str)) or o is None:
                return o
            # Si el objeto son datos binarios, se convierte en una cadena codificada en
            # base64
            elif isinstance(o, bytes):
                return "*** datos binarios ***"
            # Si el objeto es un diccionario, se procesa recursivamente cada par
            # clave-valor
            elif isinstance(o, dict):
                return {k: serialize(v) for k, v in o.items()}
            # Si el objeto es una lista o tupla, se procesa recursivamente cada elemento
            elif isinstance(o, (list, tuple)):
                return [serialize(item) for item in o]
            # Si el objeto es de un tipo personalizado, se intenta devolver su atributo
            # __dict__
            elif hasattr(o, "__dict__"):
                return serialize(o.__dict__)
            # En otros casos se devuelve None (o se puede optar por lanzar una
            # excepción)
            else:
                return None

        # Usa la función serialize para procesar el objeto de entrada
        serialized_obj = serialize(obj)

        # Serializa el objeto procesado en una cadena JSON
        return json.dumps(serialized_obj, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"no se pudo serializar el objeto a json: {str(e)}")
        return None


def get_uuid(remove_hyphen: bool = False):
    u = str(uuid4())
    if remove_hyphen:
        u = u.replace("-", "")
    return u


_CLIP_SPEED_MIN = 0.5
_CLIP_SPEED_MAX = 2.0


def normalize_clip_speed(value, default: float = 1.0) -> float:
    """Normaliza la velocidad de reproducción del clip al rango seguro que admite la WebUI."""
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return default

    # NaN se salta las comparaciones normales de tamaño y se propaga cuando MoviePy
    # calcula duration; los infinitos tampoco son una entrada de usuario válida. Ambos
    # vuelven al valor por defecto, para que ni la API ni las llamadas internas directas
    # generen líneas de tiempo no válidas. Cero y los negativos tampoco pueden
    # representar una velocidad de reproducción normal.
    if not math.isfinite(speed) or speed <= 0:
        return default

    return min(max(speed, _CLIP_SPEED_MIN), _CLIP_SPEED_MAX)


def root_dir():
    return str(paths.runtime_root())


def storage_dir(sub_dir: str = "", create: bool = False):
    d = os.path.join(root_dir(), "storage")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if create:
        os.makedirs(d, exist_ok=True)

    return d


def resource_dir(sub_dir: str = ""):
    return str(paths.resource_directory(sub_dir))


def task_dir(sub_dir: str = ""):
    d = os.path.join(storage_dir(), "tasks")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    os.makedirs(d, exist_ok=True)
    return d


def font_dir(sub_dir: str = ""):
    d = resource_dir("fonts")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    os.makedirs(d, exist_ok=True)
    return d


def song_dir(sub_dir: str = ""):
    d = resource_dir("songs")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    os.makedirs(d, exist_ok=True)
    return d


def public_dir(sub_dir: str = ""):
    d = resource_dir("public")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    os.makedirs(d, exist_ok=True)
    return d


def get_ffmpeg_binary() -> str:
    """
    Resuelve el ejecutable de FFmpeg que debe usar el proceso actual.

    Motivos:
    1. La codificación de vídeo, la generación de audio silencioso y la
       transcodificación de audio de pydub dependen de FFmpeg;
    2. En el paquete portátil de Windows, en Docker y en directorios de instalación
       personalizados del usuario el PATH suele ser incoherente;
    3. Resolverlo en un solo sitio hace que todos los llamadores usen la misma
       prioridad, reduciendo los casos en que una ruta funciona y otra no encuentra
       FFmpeg.

    Prioridad:
    1. IMAGEIO_FFMPEG_EXE: configuración explícita según la convención de
       MoviePy/imageio;
    2. ffmpeg en el PATH del sistema;
    3. el binario integrado que proporciona la dependencia imageio-ffmpeg;
    4. la cadena "ffmpeg" como último recurso, dejando que subprocess muestre un error
       más concreto en tiempo de ejecución.
    """
    configured_ffmpeg = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if configured_ffmpeg:
        return configured_ffmpeg

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg

        bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled_ffmpeg:
            return bundled_ffmpeg
    except Exception as exc:
        logger.warning(f"no se pudo resolver el binario de ffmpeg incluido: {str(exc)}")

    return "ffmpeg"


_FFMPEG_INSTALL_HINT = (
    "Instala FFmpeg en tu sistema o configura app.ffmpeg_path en config.toml con "
    "la ruta completa de un ejecutable de ffmpeg (por ejemplo, descargado de "
    "https://www.gyan.dev/ffmpeg/builds/)."
)


def check_ffmpeg_ready(timeout: int = 10) -> bool:
    """
    Comprueba si FFmpeg está disponible antes de empezar de verdad a generar el vídeo.

    Motivo:
    Antes, que FFmpeg faltara o no funcionara solo aparecía en la composición del vídeo,
    la generación de la pista silenciosa, etc., como ``RuntimeError: No ffmpeg exe could
    be found`` o como error de subprocess; el usuario a menudo veía este error por
    primera vez con la tarea ya muy avanzada, y el propio error no indicaba ninguna
    solución. Aquí se hace una comprobación anticipada en el flujo de tareas compartido
    (``_run_pipeline`` de app/services/task.py), dando cuanto antes un aviso útil; la
    API, la CLI y la WebUI pasan todas por este flujo, así que las tres rutas quedan
    cubiertas por igual.

    Solo hace una llamada ligera a ``-version``, sin provocar descargas ni cambiar el
    flujo principal; quien llama debe tratar el valor devuelto como requisito previo
    obligatorio: el imageio-ffmpeg==0.6.0 que fija el proyecto no descarga
    automáticamente un binario utilizable al usarse, así que si la comprobación falla
    las etapas que necesitan FFmpeg deben terminar directamente, en lugar de seguir
    hasta fallar en la composición del vídeo.
    """
    ffmpeg_bin = get_ffmpeg_binary()
    try:
        completed = subprocess.run(
            [ffmpeg_bin, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except FileNotFoundError:
        logger.warning(
            f"no se encontró ningún ejecutable de ffmpeg utilizable (probado: {ffmpeg_bin}). "
            f"{_FFMPEG_INSTALL_HINT}"
        )
        return False
    except Exception as exc:
        logger.warning(
            f"no se pudo comprobar ffmpeg ({ffmpeg_bin}): {exc}. {_FFMPEG_INSTALL_HINT}"
        )
        return False

    if completed.returncode != 0:
        logger.warning(
            f"la comprobación de ffmpeg ({ffmpeg_bin}) terminó con el estado {completed.returncode}; "
            f"la generación de vídeo puede fallar después. {_FFMPEG_INSTALL_HINT}"
        )
        return False

    logger.info(f"comprobación de ffmpeg correcta, se usa: {ffmpeg_bin}")
    return True


def run_in_background(func, *args, **kwargs):
    def run():
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"error en run_in_background: {e}", exc_info=True)

    thread = threading.Thread(target=run, daemon=False)
    thread.start()
    return thread


def time_convert_seconds_to_hmsm(seconds) -> str:
    # Redondea una vez a milisegundos enteros y deriva todos los campos de ese valor.
    # Truncar productos de coma flotante como 2.01 * 1000 == 2009.999... escribía los
    # límites de los subtítulos un milisegundo antes.
    total_milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return "{:02d}:{:02d}:{:02d},{:03d}".format(hours, minutes, seconds, milliseconds)


def text_to_srt(idx: int, msg: str, start_time: float, end_time: float) -> str:
    start_time = time_convert_seconds_to_hmsm(start_time)
    end_time = time_convert_seconds_to_hmsm(end_time)
    srt = """%d
%s --> %s
%s
        """ % (
        idx,
        start_time,
        end_time,
        msg,
    )
    return srt


def str_contains_punctuation(word):
    for p in const.PUNCTUATIONS:
        if p in word:
            return True
    return False


def split_string_by_punctuations(s):
    result = []
    txt = ""

    previous_char = ""
    next_char = ""
    for i in range(len(s)):
        char = s[i]
        if char == "\n":
            result.append(txt.strip())
            txt = ""
            continue

        if i > 0:
            previous_char = s[i - 1]
        if i < len(s) - 1:
            next_char = s[i + 1]

        if char == "." and previous_char.isdigit() and next_char.isdigit():
            # # En un caso como "retirar 10,000, con una comisión del 2.5%", el punto de
            # "2.5" no debe tratarse como marca de salto de línea
            txt += char
            continue

        if char == "." and previous_char.isalpha() and next_char.isalpha():
            # Edge TTS lee "U.S.", "e.g.", "9 a.m." y "example.com" como una sola
            # palabra cada uno.
            txt += char
            continue

        if char == ":" and previous_char.isdigit() and next_char.isdigit():
            # Edge TTS devuelve una hora como "10:30" como una sola palabra, así que no
            # es un salto de línea.
            txt += char
            continue

        if char == "," and previous_char.isdigit() and next_char.isdigit():
            # La coma de los millares en los números en inglés no es un signo de corte,
            # por ejemplo "1,000 years". El word boundary de Edge TTS suele devolver ese
            # número como contenido continuo; si aquí se dividiera en "1" y "000 years",
            # la agregación de subtítulos posterior no podría coincidir con el texto
            # original del guion y volvería por error a Whisper.
            txt += char
            continue

        if char not in const.PUNCTUATIONS:
            txt += char
        else:
            result.append(txt.strip())
            txt = ""
    result.append(txt.strip())
    # filtra las cadenas vacías
    result = list(filter(None, result))
    return result


PAUSE_TAG_KEYWORDS = (
    r"pause|pausa|silence|silencio|silêncio|silenzio|stille|"
    r"пауза|тишина|停顿|暂停|静音|ポーズ|一時停止|無音|일시중지|정지"
)
# Coincide con todas las etiquetas (entre corchetes o paréntesis) que contienen una
# palabra clave de pausa, sean válidos o no sus parámetros, para que las etiquetas no
# válidas (como [pause: -2s], [pause: nope] o [pause: 0s]) se eliminen por completo
# antes de la síntesis y no lleguen al TTS. No se puede usar `\b` tras la palabra clave:
# los caracteres CJK y los dígitos son `\w`, así que la forma habitual sin dos puntos
# `[停顿3秒]` no coincidiría por falta de límite de palabra y se leería el texto de la
# etiqueta. Aquí solo se prohíbe que tras la palabra clave venga directamente una letra
# (se permiten dígitos), para que textos normales como `[paused]` o `[暂停一下]` sigan sin
# tomarse por etiquetas.
PAUSE_TAG_PATTERN = re.compile(
    rf"[\[\(]\s*(?:{PAUSE_TAG_KEYWORDS})(?![^\W\d])(?:\s*[:：]?\s*([^\]\)]*?))?\s*[\]\)]",
    re.IGNORECASE,
)

# Umbrales de seguridad de la duración de pausa (en segundos):
# la pausa válida mínima es de 0,1 segundos (100 ms), y las solicitudes menores se
# validan y corrigen a 0,1 s; las etiquetas de pausa menores o iguales a 0 segundos o no
# numéricas se consideran no válidas y se eliminan directamente, sin leerse ni generar
# silencio; el máximo de pausa es de 10,0 segundos, y lo que lo supere se recorta con
# seguridad.
MIN_PAUSE_DURATION_SECONDS = 0.1
MAX_PAUSE_DURATION_SECONDS = 10.0


def has_pause_tags(text: str) -> bool:
    """Comprueba si el texto contiene etiquetas de pausa."""
    if not text:
        return False
    return bool(PAUSE_TAG_PATTERN.search(text))


def remove_pause_tags(text: str) -> str:
    """
    Elimina todas las etiquetas de pausa del texto del guion (tanto válidas como no
    válidas).

    Al dividir los subtítulos en frases, extraer palabras clave con el LLM o pasar el
    texto de pronunciación al TTS, hay que eliminar estas marcas no pronunciables, para
    que las etiquetas no válidas o sin tratar no se lean ni se usen como términos de
    búsqueda visual.
    """
    if not text:
        return ""
    cleaned = PAUSE_TAG_PATTERN.sub(" ", text)
    # Fusiona los espacios horizontales consecutivos, conservando los saltos de línea
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def parse_script_with_pauses(text: str) -> list[tuple[str, Any]]:
    """
    Analiza el texto y las etiquetas de pausa del guion.

    Las etiquetas de pausa consecutivas se fusionan automáticamente en un solo segmento
    de pausa; las etiquetas no válidas (con parámetros no numéricos o duraciones menores
    o iguales a 0) se eliminan e ignoran directamente, y nunca se pasan al TTS como
    texto; las pausas demasiado cortas, menores que MIN_PAUSE_DURATION_SECONDS (0,1
    s/100 ms), se validan y suben a 0,1 s; las pausas demasiado largas, mayores que
    MAX_PAUSE_DURATION_SECONDS (10,0 s), se limitan al máximo seguro.

    Returns:
        Lista ordenada de tuplas, de la forma [("speech", "texto"), ("pause", 2.0), ...]
    """
    if not text:
        return []

    segments: list[tuple[str, Any]] = []
    last_idx = 0

    for match in PAUSE_TAG_PATTERN.finditer(text):
        start, end = match.span()
        if start > last_idx:
            speech_text = text[last_idx:start].strip()
            if speech_text:
                segments.append(("speech", speech_text))

        raw_arg = match.group(1)
        if raw_arg is None or not raw_arg.strip():
            # Sin parámetro indicado, la pausa por defecto es de 1,0 segundos
            duration = 1.0
        else:
            raw_arg_str = raw_arg.strip()
            num_match = re.match(
                r"^([+-]?\d+(?:\.\d+)?)\s*(s|sec|secs|second|seconds|ms|msec|msecs|秒|毫秒)?$",
                raw_arg_str,
                re.IGNORECASE,
            )
            if not num_match:
                logger.warning(
                    f"valor de etiqueta de pausa no válido '{raw_arg_str}'; la etiqueta se elimina y se ignora"
                )
                last_idx = end
                continue

            val = float(num_match.group(1))
            unit = (num_match.group(2) or "s").lower()
            duration = val / 1000.0 if ("ms" in unit or "毫秒" in unit) else val

        if duration <= 0:
            logger.warning(
                f"duración de pausa no positiva no válida {duration}s; la etiqueta se elimina y se ignora"
            )
            last_idx = end
            continue

        if duration < MIN_PAUSE_DURATION_SECONDS:
            logger.warning(
                f"la duración de pausa {duration:.3f}s es inferior al mínimo de {MIN_PAUSE_DURATION_SECONDS}s (100 ms); "
                f"se ajusta a {MIN_PAUSE_DURATION_SECONDS}s"
            )
            duration = MIN_PAUSE_DURATION_SECONDS

        if duration > MAX_PAUSE_DURATION_SECONDS:
            logger.warning(
                f"la duración de pausa {duration}s supera el máximo de {MAX_PAUSE_DURATION_SECONDS}s; "
                f"se ajusta a {MAX_PAUSE_DURATION_SECONDS}s"
            )
            duration = MAX_PAUSE_DURATION_SECONDS

        # Las etiquetas de pausa consecutivas se fusionan en un solo segmento de pausa,
        # para no generar archivos de silencio fragmentados
        if segments and segments[-1][0] == "pause":
            merged_duration = min(
                segments[-1][1] + duration, MAX_PAUSE_DURATION_SECONDS
            )
            segments[-1] = ("pause", merged_duration)
        else:
            segments.append(("pause", duration))

        last_idx = end

    if last_idx < len(text):
        speech_text = text[last_idx:].strip()
        if speech_text:
            segments.append(("speech", speech_text))

    return segments


def normalize_script_for_subtitle_matching(video_script: str) -> str:
    """
    Limpia el texto del guion antes de hacer coincidir los subtítulos.

    El usuario puede escribir a mano separadores Markdown, énfasis en títulos o símbolos
    de formato como `_`. Estos caracteres normalmente no aparecen en los resultados de
    reconocimiento de TTS/Whisper; si siguieran participando en la correspondencia línea
    a línea de los subtítulos, el guion tendría más líneas que los subtítulos reales, y
    al final podría añadirse `00:00:00,000 --> 00:00:00,000`, de modo que los programas
    de edición no podrían importar el SRT.
    """
    video_script = remove_pause_tags(video_script or "")
    underscore_count = video_script.count("_")
    video_script = video_script.replace("_", "")
    cleaned_lines = []
    removed_separator_lines = 0
    for line in video_script.splitlines():
        line = line.strip()
        # Cuando un separador o un símbolo de énfasis de Markdown ocupa una línea solo,
        # el TTS no lo lee, así que hay que quitarlo de las líneas del guion para que la
        # agregación de subtítulos no se quede atascada en estas líneas "no
        # pronunciables". Tampoco lo es una línea sin ninguna letra ni dígito: "* * *",
        # "———", "• • •".
        if line and not re.sub(r"[_\W]+", "", line):
            removed_separator_lines += 1
            continue
        cleaned_lines.append(line)

    normalized_script = "\n".join(cleaned_lines).strip()
    if underscore_count or removed_separator_lines:
        logger.debug(
            "guion normalizado para la correspondencia de subtítulos, "
            f"guiones bajos eliminados: {underscore_count}, "
            f"líneas separadoras markdown eliminadas: {removed_separator_lines}"
        )
    return normalized_script


def md5(text):
    import hashlib

    return hashlib.md5(text.encode("utf-8")).hexdigest()


def resolve_ui_language(
    saved_language: str | None,
    browser_locale: str | None,
    supported_languages: Iterable[str],
    default_language: str = "es",
) -> str:
    """
    Elige el idioma de la interfaz con la prioridad "ajuste guardado, idioma del
    navegador, idioma por defecto".

    El navegador suele devolver un locale con región, como ``es-ES`` o ``es-MX``. Los
    archivos de idioma usan códigos base como ``es``, así que primero se intenta la
    coincidencia completa y luego se vuelve al código de idioma anterior al guion. La
    función se mantiene como lógica pura, para no acoplar el contexto del navegador ni
    la escritura de configuración a la capa de utilidades, lo que facilita los tests.
    """
    supported = [str(language).strip() for language in supported_languages]
    supported_by_lower = {
        language.lower(): language for language in supported if language
    }

    def match_language(value: str | None) -> str | None:
        normalized = str(value or "").strip().replace("_", "-").lower()
        if not normalized:
            return None
        if normalized in supported_by_lower:
            return supported_by_lower[normalized]
        base_language = normalized.split("-", 1)[0]
        return supported_by_lower.get(base_language)

    saved_match = match_language(saved_language)
    if saved_match:
        return saved_match

    browser_match = match_language(browser_locale)
    if browser_match:
        return browser_match

    default_match = match_language(default_language)
    if default_match:
        return default_match

    # Un proyecto normal siempre incluye el idioma por defecto; se mantiene un respaldo
    # para un conjunto de idiomas vacío, para que un directorio de idiomas dañado no
    # haga fallar la inicialización de la página, y la función de traducción seguirá
    # mostrando la clave original para facilitar el diagnóstico.
    return supported[0] if supported else default_language


@lru_cache(maxsize=8)
def load_locales(i18n_dir):
    # Cada interacción con la WebUI hace que Streamlit vuelva a ejecutar el script, y
    # los archivos de idioma no cambian durante la ejecución, así que se guarda en caché
    # el resultado del análisis para no leer y analizar una y otra vez todos los
    # archivos JSON de i18n.
    _locales = {}
    for root, dirs, files in os.walk(i18n_dir):
        for file in files:
            if file.endswith(".json"):
                lang = file.split(".")[0]
                with open(os.path.join(root, file), "r", encoding="utf-8-sig") as f:
                    _locales[lang] = json.loads(f.read())
    return _locales


def parse_extension(filename):
    return Path(filename).suffix.lower().lstrip('.')
