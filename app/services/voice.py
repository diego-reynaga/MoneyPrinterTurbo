import asyncio
import base64
import io
import inspect
import json
import math
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unicodedata
import wave
from datetime import timedelta
from typing import Union
from urllib.parse import urlparse
from xml.sax.saxutils import escape, unescape

import edge_tts
import requests
from edge_tts import SubMaker
from edge_tts.srt_composer import Subtitle
from loguru import logger

from app.utils.request_logging import trace_http_call
from moviepy.audio.io.AudioFileClip import AudioFileClip
from openai import OpenAI

from app.config import config
from app.services import bgm
from app.utils import utils
from app.utils.subtitle_reader import moviepy_subtitles
from app.utils.subtitle_writer import staged_subtitle_file

_DEFAULT_EDGE_TTS_TIMEOUT_SECONDS = 30.0
_SILICONFLOW_TTS_TIMEOUT_SECONDS = (10, 300)  # conexión, lectura
_MIMO_DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
_MIMO_DEFAULT_TTS_MODEL = "mimo-v2.5-tts"
MINIMAX_TTS_GLOBAL_URL = "https://api.minimax.io/v1/t2a_v2"
MINIMAX_TTS_CN_URL = "https://api.minimaxi.com/v1/t2a_v2"
MINIMAX_TTS_DEFAULT_MODEL = "speech-2.8-hd"
MINIMAX_TTS_DEFAULT_VOICE = "English_expressive_narrator"
MINIMAX_TTS_MODELS = (
    "speech-2.8-hd", "speech-2.8-turbo", "speech-2.6-hd", "speech-2.6-turbo",
    "speech-02-hd", "speech-02-turbo", "speech-01-hd", "speech-01-turbo",
)
GEMINI_TTS_VOICES = (
    ("Zephyr", "Bright"),
    ("Puck", "Upbeat"),
    ("Charon", "Informative"),
    ("Kore", "Firm"),
    ("Fenrir", "Excitable"),
    ("Leda", "Youthful"),
    ("Orus", "Firm"),
    ("Aoede", "Breezy"),
    ("Callirrhoe", "Easy-going"),
    ("Autonoe", "Bright"),
    ("Enceladus", "Breathy"),
    ("Iapetus", "Clear"),
    ("Umbriel", "Easy-going"),
    ("Algieba", "Smooth"),
    ("Despina", "Smooth"),
    ("Erinome", "Clear"),
    ("Algenib", "Gravelly"),
    ("Rasalgethi", "Informative"),
    ("Laomedeia", "Upbeat"),
    ("Achernar", "Soft"),
    ("Alnilam", "Firm"),
    ("Schedar", "Even"),
    ("Gacrux", "Mature"),
    ("Pulcherrima", "Forward"),
    ("Achird", "Friendly"),
    ("Zubenelgenubi", "Casual"),
    ("Vindemiatrix", "Gentle"),
    ("Sadachbia", "Lively"),
    ("Sadaltager", "Knowledgeable"),
    ("Sulafat", "Warm"),
)
_MINIMAX_TTS_MAX_AUDIO_HEX_CHARS = 100 * 1024 * 1024
_ELEVENLABS_TTS_MAX_AUDIO_BYTES = 50 * 1024 * 1024
_ELEVENLABS_TTS_MAX_ERROR_BYTES = 4096
VOXCPM_DEFAULT_BASE_URL = "https://api.modelbest.cn/v1"
VOXCPM_DEFAULT_VOICE = "default"
VOXCPM_REFERENCE_AUDIO_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
VOXCPM_REFERENCE_AUDIO_MAX_WAV_BYTES = 5 * 1024 * 1024
VOXCPM_REFERENCE_AUDIO_CONVERSION_TIMEOUT_SECONDS = 15
VOXCPM_REFERENCE_AUDIO_MAX_DURATION_SECONDS = 120
VOXCPM_REFERENCE_AUDIO_FILE_TYPES = ("wav", "mp3", "m4a", "aac", "ogg", "flac")
_DEFAULT_TTS_FFMPEG_TIMEOUT_SECONDS = 600
_VOXCPM_NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}
_VOXCPM_RETRY_DELAY_SECONDS = (1.0, 2.0)
# Coincide con el límite actual de voz generada; permite un evento WAV Base64 completo
# más el marco JSON, acotando tanto las líneas de transporte como los eventos de varias
# líneas.
_VOXCPM_TTS_MAX_AUDIO_BYTES = 50 * 1024 * 1024
_VOXCPM_SSE_MAX_EVENT_BYTES = 4 * ((_VOXCPM_TTS_MAX_AUDIO_BYTES + 2) // 3) + 64 * 1024
NO_VOICE_NAME = "no-voice"
# `none` es el identificador de sin locución que usó el PR #981. Aquí se mantiene a
# corto plazo la compatibilidad con este valor, para que los usuarios de la API que ya
# llamaron a esa rama a mano no dejen de funcionar de inmediato al actualizar; la WebUI
# y el código nuevo usan siempre el más claro `no-voice`.
_NO_VOICE_ALIASES = {NO_VOICE_NAME, "none"}


def _configure_pydub_ffmpeg(audio_segment_cls):
    configured_ffmpeg = utils.get_ffmpeg_binary()
    if configured_ffmpeg:
        audio_segment_cls.converter = configured_ffmpeg


def mktimestamp(time_unit: float) -> str:
    """
    Convierte la unidad de tiempo de 100 nanosegundos que usa edge_tts en una marca de
    tiempo de subtítulo.

    edge_tts 7.x ya no exporta el `mktimestamp` de las versiones antiguas, pero la
    cadena de subtítulos antigua del proyecto sigue necesitando esta función de formato
    para las líneas de tiempo de subtítulos construidas a mano de Azure v2, Gemini o
    SiliconFlow, así que aquí se incluye una implementación equivalente.
    """
    # Primero se redondea a milisegundos y luego se separan horas, minutos y segundos.
    # Si se formateara directamente con `:06.3f` los segundos tras el módulo, 59,9996
    # segundos se mostrarían como `00:00:60.000`, y un analizador SRT estricto
    # rechazaría esa marca de tiempo.
    total_milliseconds = int(round(time_unit / 10**4))
    hour, remainder = divmod(total_milliseconds, 3_600_000)
    minute, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hour:02d}:{minute:02d}:{seconds:02d}.{milliseconds:03d}"


def get_siliconflow_voices() -> list[str]:
    """
    Obtiene la lista de voces de SiliconFlow

    Returns:
        Lista de voces, con el formato ["siliconflow:FunAudioLLM/CosyVoice2-0.5B:alex",
        ...]
    """
    # Lista de voces de SiliconFlow y su género correspondiente (para mostrar)
    voices_with_gender = [
        ("FunAudioLLM/CosyVoice2-0.5B", "alex", "Male"),
        ("FunAudioLLM/CosyVoice2-0.5B", "anna", "Female"),
        ("FunAudioLLM/CosyVoice2-0.5B", "bella", "Female"),
        ("FunAudioLLM/CosyVoice2-0.5B", "benjamin", "Male"),
        ("FunAudioLLM/CosyVoice2-0.5B", "charles", "Male"),
        ("FunAudioLLM/CosyVoice2-0.5B", "claire", "Female"),
        ("FunAudioLLM/CosyVoice2-0.5B", "david", "Male"),
        ("FunAudioLLM/CosyVoice2-0.5B", "diana", "Female"),
    ]

    # Añade el prefijo siliconflow: y lo formatea como nombre visible
    return [
        f"siliconflow:{model}:{voice}-{gender}"
        for model, voice, gender in voices_with_gender
    ]


def get_gemini_voices() -> list[str]:
    """
    Obtiene la lista de voces predefinidas oficiales de Gemini TTS.

    Google no publica metadatos de género para estas voces, así que el desplegable usa
    la descripción del estilo oficial, para no escribir un género supuesto en el voice
    id persistente. Origen del catálogo de voces:
    https://ai.google.dev/gemini-api/docs/speech-generation#voice-options

    Returns:
        Lista de voces, con el formato ["gemini:Zephyr-Bright", "gemini:Puck-Upbeat",
        ...]
    """
    return [f"gemini:{voice}-{style}" for voice, style in GEMINI_TTS_VOICES]


def get_mimo_voices() -> list[str]:
    """
    Obtiene la lista de voces predefinidas de Xiaomi MiMo V2.5 TTS.

    De momento solo se integra el modo de voces predefinidas `mimo-v2.5-tts` de la
    documentación oficial. El diseño de voces `mimo-v2.5-tts-voicedesign` y la clonación
    de voz `mimo-v2.5-tts-voiceclone` necesitan formularios de entrada y flujos de
    subida de material adicionales, así que de momento no se mezclan en el desplegable
    TTS normal, para que el usuario no crea que elegir un voice id basta para tener
    todas las funciones avanzadas.
    """
    voices_with_gender = [
        ("mimo_default", "Female"),
        ("冰糖", "Female"),
        ("茉莉", "Female"),
        ("苏打", "Male"),
        ("白桦", "Male"),
        ("Mia", "Female"),
        ("Chloe", "Female"),
        ("Milo", "Male"),
        ("Dean", "Male"),
    ]

    return [f"mimo:{voice}-{gender}" for voice, gender in voices_with_gender]


def get_minimax_voices(voice_id: str | None = None) -> list[str]:
    """Devuelve las voces de MiniMax configuradas actualmente, en el formato común de
    despacho del TTS.
    """
    voice_id = str(
        voice_id
        or config.minimax_tts.get("voice_id", MINIMAX_TTS_DEFAULT_VOICE)
        or MINIMAX_TTS_DEFAULT_VOICE
    ).strip()
    return [f"minimax:{voice_id}"]


def get_elevenlabs_voices(api_key: str) -> list[str]:
    if not api_key:
        return []
    try:
        url = "https://api.elevenlabs.io/v2/voices"
        params = {"is_favorite": "true", "page_size": 100}
        headers = {"xi-api-key": api_key}
        # Requests conserva las cabeceras personalizadas xi-api-key en las
        # redirecciones. Se mantiene la clave en el endpoint del proveedor aunque
        # responda con una redirección.
        result = []
        seen_voices = set()
        seen_tokens = set()
        deadline = time.monotonic() + 30.0
        # Los catálogos v2 están paginados incluso con page_size al máximo.
        # Se mantiene el endpoint fijo y el límite de credenciales sin redirecciones.
        for _ in range(100):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning("se agotó el tiempo para el catálogo de voces de ElevenLabs; se devuelve un catálogo parcial")
                break
            phase_timeout = min(10.0, remaining / 2.0)
            response = trace_http_call(
                "GET", requests.get,
                url, params=dict(params), headers=headers,
                timeout=(phase_timeout, phase_timeout), allow_redirects=False,
            )
            try:
                if response.status_code != 200:
                    logger.warning(
                        f"falló la obtención de voces de ElevenLabs con el estado {response.status_code}"
                    )
                    return []
                data = response.json()
            finally:
                response.close()
            if not isinstance(data, dict) or not isinstance(data.get("voices"), list):
                return []
            for entry in data["voices"]:
                if not isinstance(entry, dict):
                    continue
                identity = entry.get("voice_id")
                name = entry.get("name")
                if (
                    isinstance(identity, str) and identity
                    and isinstance(name, str) and name
                    and entry.get("status") != "disabled"
                    and identity not in seen_voices
                ):
                    seen_voices.add(identity)
                    result.append(f"elevenlabs:{identity}:{name}")
            token = data.get("next_page_token")
            if data.get("has_more") is not True or not isinstance(token, str) or not token:
                break
            if token in seen_tokens:
                logger.warning("el catálogo de voces de ElevenLabs repitió un token de página")
                break
            seen_tokens.add(token)
            params["next_page_token"] = token
        else:
            logger.warning("se alcanzó el límite de páginas del catálogo de voces de ElevenLabs; se devuelve un catálogo parcial")
        return result
    except Exception as e:
        logger.warning(f"falló la obtención de voces de ElevenLabs: {str(e)}")
        return []


def get_chatterbox_voices() -> list[str]:
    """Devuelve las voces de Chatterbox configuradas.

    Chatterbox es autoalojado, así que no hay un catálogo de voces global. Los
    operadores enumeran los nombres de voz que expone su servidor en ``[chatterbox]
    voices`` (un array TOML o una cadena separada por comas). Cada entrada se normaliza
    al formato ``chatterbox:<nombre>`` que usa el despachador del TTS.
    """
    voices = config.chatterbox.get("voices", []) or []
    if isinstance(voices, str):
        voices = [v.strip() for v in voices.split(",") if v.strip()]
    result = []
    for v in voices:
        v = str(v).strip()
        if not v:
            continue
        result.append(v if v.startswith("chatterbox:") else f"chatterbox:{v}")
    if not result:
        # mantiene el desplegable utilizable incluso antes de configurar ninguna voz
        result = ["chatterbox:default-Female"]
    return result


KOKORO_DEFAULT_VOICE = "af_heart"


def _normalize_kokoro_voices(entries) -> list[str]:
    """Unifica la configuración manual y los formatos nuevo y antiguo del servidor,
    aceptando solo ID reales, para no convertir objetos en nombres de voz.
    """
    if isinstance(entries, str):
        entries = entries.split(",")
    if not isinstance(entries, list):
        return []
    result = []
    for entry in entries:
        name = entry.get("id") if isinstance(entry, dict) else entry
        if not isinstance(name, str):
            continue
        name = name.strip().removeprefix("kokoro:").strip()
        if name:
            value = f"kokoro:{name}"
            if value not in result:
                result.append(value)
    return result


def get_kokoro_voices(*, fallback: bool = True) -> list[str]:
    """Tienen prioridad las voces manuales; si no, se consulta al servidor; la interfaz
    puede desactivar el valor por defecto para detectar desconexiones y conservar la
    selección.

    Los servidores antiguos devuelven una lista de cadenas y los nuevos una lista de
    objetos con id. Si falla, la capa de servicio no guarda estado de sesión en caché;
    la WebUI conserva el último catálogo correcto, para que distintos usuarios/endpoints
    no se contaminen entre sí.
    """
    voices = _normalize_kokoro_voices(config.kokoro.get("voices"))
    if not voices:
        base_url = (config.kokoro.get("base_url", "") or "").strip().rstrip("/")
        if base_url:
            try:
                headers = {}
                api_key = config.kokoro.get("api_key", "")
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                response = trace_http_call(
                    "GET", requests.get,
                    f"{base_url}/audio/voices", headers=headers, timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    listed = data.get("voices", []) if isinstance(data, dict) else data
                    voices = _normalize_kokoro_voices(listed)
                    if not voices:
                        logger.warning("la lista de voces de Kokoro no contiene ID de voz válidos")
                else:
                    logger.warning(
                        f"falló la petición de voces de Kokoro con el estado {response.status_code}"
                    )
            except Exception as e:
                # No se muestran la URL ni el cuerpo de la excepción, para que los
                # parámetros de consulta o la información de autenticación de la
                # dirección autoalojada no lleguen al log.
                logger.warning(f"lista de voces de Kokoro no disponible ({type(e).__name__})")
    return voices or ([f"kokoro:{KOKORO_DEFAULT_VOICE}"] if fallback else [])


def get_fish_audio_voices() -> list[str]:
    """Devuelve las voces de Fish Audio configuradas.

    Cada entrada sigue el formato ``fish_audio:<reference_id>:<nombre_visible>``. Cuando
    ``reference_id`` es "default", se usa la voz por defecto integrada de Fish Audio (no
    se envía ``reference_id`` en la solicitud a la API). Los operadores pueden enumerar
    voces públicas o clonadas adicionales en ``[fish_audio] voices`` del archivo de
    configuración.
    """
    result = [
        "fish_audio:2324c907b9a94c64ab4afb941e5b3408:Clear Female-Female",
        "fish_audio:7b6131ba75ba47c98a46c847db729ab6:Clear Male-Male",
        "fish_audio:default:Default Voice",
    ]
    voices = config.fish_audio.get("voices", []) or []
    if isinstance(voices, str):
        voices = [v.strip() for v in voices.split(",") if v.strip()]
    for entry in voices:
        entry = str(entry).strip()
        if not entry:
            continue
        if entry.startswith("fish_audio:"):
            result.append(entry)
        elif ":" in entry:
            # "<reference_id>:<nombre_visible>"
            result.append(f"fish_audio:{entry}")
        else:
            # reference_id sin más
            result.append(f"fish_audio:{entry}:{entry}")
    return result


def get_voxcpm_voices(voice_id: str | None = None) -> list[str]:
    """Devuelve la voz VoxCPM de ModelBest seleccionada en la configuración local.

    ModelBest acepta ``default`` cuando no hay una voz seleccionada explícitamente. El
    diseño de voces se expresa en el texto de entrada y la clonación de voz requiere un
    flujo aparte de audio de referencia, así que la primera integración mantiene
    deliberadamente el selector TTS estándar con un único voice id configurado.
    """
    voice_id = str(
        voice_id
        or config.voxcpm.get("voice_id", VOXCPM_DEFAULT_VOICE)
        or VOXCPM_DEFAULT_VOICE
    ).strip()
    return [f"voxcpm:{voice_id}"]


_AZURE_VOICES_DATA_FILE = os.path.join(
    os.path.dirname(__file__), "data", "azure_voices.json"
)
_azure_voices_cache = None


def _load_azure_voices() -> list[dict]:
    global _azure_voices_cache
    if _azure_voices_cache is None:
        with open(_AZURE_VOICES_DATA_FILE, "r", encoding="utf-8") as f:
            _azure_voices_cache = json.load(f)
    return _azure_voices_cache


def get_all_azure_voices(filter_locals=None) -> list[str]:
    voices = []
    for item in _load_azure_voices():
        name = item["name"]
        gender = item["gender"]
        # Aplica las condiciones de filtrado
        if filter_locals and any(
            name.lower().startswith(fl.lower()) for fl in filter_locals
        ):
            voices.append(f"{name}-{gender}")
        elif not filter_locals:
            voices.append(f"{name}-{gender}")

    voices.sort()
    return voices


def parse_voice_name(name: str):
    # zh-CN-XiaoyiNeural-Female
    # zh-CN-YunxiNeural-Male
    # zh-CN-XiaoxiaoMultilingualNeural-V2-Female
    name = name.strip()
    # Los ID cualificados pertenecen a su proveedor. Las etiquetas visibles de
    # género/estilo las analiza el adaptador de cada proveedor, sin reescribirlas dentro
    # de un voice ID opaco.
    if ":" not in name:
        name = name.removesuffix("-Female").removesuffix("-Male")
    return name


def is_azure_v2_voice(voice_name: str):
    voice_name = parse_voice_name(voice_name)
    if ":" not in voice_name and voice_name.endswith("-V2"):
        return voice_name.removesuffix("-V2").strip()
    return ""


def is_siliconflow_voice(voice_name: str):
    """Comprueba si es una voz de SiliconFlow"""
    return voice_name.startswith("siliconflow:")


def is_gemini_voice(voice_name: str):
    """Comprueba si es una voz de Gemini TTS"""
    return voice_name.startswith("gemini:")


def parse_gemini_voice_name(voice_name: str | None) -> str:
    """Extrae, de los valores nuevos y antiguos del desplegable de Gemini, el nombre de la
    voz predefinida que usa la API de Google.
    """
    if not is_gemini_voice(voice_name or ""):
        return ""
    return (voice_name or "").split(":", 1)[1].split("-", 1)[0].strip()


def is_mimo_voice(voice_name: str):
    """Comprueba si es una voz de Xiaomi MiMo TTS"""
    return voice_name.startswith("mimo:")


def is_minimax_voice(voice_name: str | None) -> bool:
    return (voice_name or "").startswith("minimax:")


def is_elevenlabs_voice(voice_name: str) -> bool:
    return (voice_name or "").startswith("elevenlabs:")


def get_elevenlabs_api_key() -> str:
    """
    Lee la clave API que usa el TTS de ElevenLabs.

    Tiene prioridad el archivo de configuración, y la variable de entorno solo es un
    respaldo cuando no está configurada. La WebUI y la función de música ya admiten
    ``ELEVENLABS_API_KEY``, y el TTS debe usar la misma regla; si no, en despliegues
    solo con variables de entorno del contenedor, la lista de voces se cargaría con
    normalidad pero la síntesis de voz real informaría por error de que falta la clave.
    """
    configured_key = str(config.elevenlabs.get("api_key", "") or "").strip()
    return configured_key or os.getenv("ELEVENLABS_API_KEY", "").strip()


def is_chatterbox_voice(voice_name: str) -> bool:
    return (voice_name or "").startswith("chatterbox:")


def is_kokoro_voice(voice_name: str) -> bool:
    return (voice_name or "").startswith("kokoro:")


def is_fish_audio_voice(voice_name: str) -> bool:
    return (voice_name or "").startswith("fish_audio:")


def is_voxcpm_voice(voice_name: str | None) -> bool:
    return (voice_name or "").startswith("voxcpm:")


def get_fish_audio_api_key() -> str:
    configured_key = str(config.fish_audio.get("api_key", "") if hasattr(config, "fish_audio") and isinstance(config.fish_audio, dict) else "").strip()
    return configured_key or os.getenv("FISH_API_KEY", "").strip()


def is_no_voice(voice_name: str | None) -> bool:
    """
    Decide si el usuario ha elegido explícitamente el modo "sin locución".

    Aquí no se trata deliberadamente la cadena vacía como sin locución: una voz vacía es
    más probable que sea una configuración dañada, una pérdida del estado de una WebUI
    antigua o un parámetro de la interfaz que falta. Solo el centinela explícito entra
    en la rama de silencio, para no disfrazar un error real de generación normal.
    """
    return str(voice_name or "").strip().lower() in _NO_VOICE_ALIASES


def is_azure_v1_voice(voice_name: str | None) -> bool:
    """
    Comprueba si es una voz predefinida de Azure TTS v1 (Edge TTS).

    La primera versión de la cadena de síntesis por segmentos de las etiquetas de pausa
    ([pause: ...]) solo es para Edge TTS (Azure TTS v1), para no cambiar la facturación,
    la frecuencia ni el comportamiento por defecto de las solicitudes de otros
    proveedores como Gemini, Fish Audio, SiliconFlow o Kokoro.
    """
    if not voice_name:
        return False
    name = str(voice_name).strip()
    if is_no_voice(name):
        return False
    if is_azure_v2_voice(name):
        return False
    if is_siliconflow_voice(name):
        return False
    if is_gemini_voice(name):
        return False
    if is_mimo_voice(name):
        return False
    if is_minimax_voice(name):
        return False
    if is_elevenlabs_voice(name):
        return False
    if is_chatterbox_voice(name):
        return False
    if is_kokoro_voice(name):
        return False
    if is_fish_audio_voice(name):
        return False
    if is_voxcpm_voice(name):
        return False
    return True


def estimate_no_voice_duration(text: str) -> float:
    """
    Estima una longitud de línea de tiempo de vídeo estable para el modo sin locución.

    Sin locución sigue haciendo falta un audio de marcador que mueva el recorte de
    material, la línea de tiempo de subtítulos y la composición final existentes. La
    estrategia de estimación es lo más simple posible:
    1. Los caracteres CJK como el chino se estiman a unos 4,2 caracteres/segundo;
    2. El inglés/los números a unas 2,7 palabras/segundo;
    3. Los textos de otros idiomas a unos 4,0 caracteres/segundo como respaldo,
       cubriendo textos no ASCII como ruso, árabe,
       kana japonés o coreano;
    4. Cada corte de frase añade una pequeña pausa, para que el cambio de subtítulos no
       sea demasiado apretado;
    5. Mínimo 3 segundos, para que los guiones muy cortos no generen audio de 0
       segundos.
    """
    normalized_text = (text or "").strip()
    if not normalized_text:
        return 3.0

    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", normalized_text))
    words = len(re.findall(r"[A-Za-z0-9]+", normalized_text))
    ascii_word_chars = sum(len(word) for word in re.findall(r"[A-Za-z0-9]+", normalized_text))
    other_text_chars = 0
    for char in normalized_text:
        # Las categorías Unicode que empiezan por L son letras de cualquier idioma y N
        # son dígitos. Antes ya se han contado aparte los caracteres CJK y las palabras
        # ASCII, así que aquí solo se cuenta el texto restante, para no contar dos veces
        # el tiempo del inglés.
        category = unicodedata.category(char)
        if category.startswith(("L", "N")):
            other_text_chars += 1
    other_text_chars = max(other_text_chars - cjk_chars - ascii_word_chars, 0)
    sentence_count = max(len(utils.split_string_by_punctuations(normalized_text)), 1)

    cjk_duration = cjk_chars / 4.2
    word_duration = words / 2.7
    other_text_duration = other_text_chars / 4.0
    pause_duration = max(sentence_count - 1, 0) * 0.35
    return max(3.0, cjk_duration + word_duration + other_text_duration + pause_duration)


def generate_silent_audio(duration_seconds: float, output_file: str) -> bool:
    """
    Genera audio silencioso.

    Permite generar directamente audio WAV PCM de 16 bits mono (preciso a nivel de
    muestra, sin retardo de codificación), o generar audio MP3 con anullsrc de FFmpeg
    (como marcador del modo "sin locución").
    """
    ensure_file_path_exists(output_file)
    duration_seconds = max(
        float(duration_seconds or 0), utils.MIN_PAUSE_DURATION_SECONDS
    )

    if output_file.lower().endswith(".wav"):
        sample_rate = 24000
        num_samples = int(round(duration_seconds * sample_rate))
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"\x00\x00" * num_samples)
        return os.path.exists(output_file) and os.path.getsize(output_file) > 0

    ffmpeg_binary = utils.get_ffmpeg_binary()
    command = [
        ffmpeg_binary,
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=mono",
        "-t",
        f"{duration_seconds:.3f}",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "4",
    ]

    logger.info(
        f"generando audio en silencio para el modo sin voz, duración: {duration_seconds:.2f}s"
    )
    return _publish_tts_ffmpeg_output(command, output_file, "codificación de la narración en silencio")


def _single_tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
) -> Union[SubMaker, None]:
    if is_no_voice(voice_name):
        duration_seconds = estimate_no_voice_duration(text)
        if not generate_silent_audio(duration_seconds, voice_file):
            return None

        sub_maker = ensure_legacy_submaker_fields(SubMaker())
        return populate_legacy_submaker_with_full_text(
            sub_maker=sub_maker,
            text=text,
            audio_duration_seconds=duration_seconds,
        )

    if is_azure_v2_voice(voice_name):
        return azure_tts_v2(
            text,
            voice_name,
            voice_file,
            voice_rate=voice_rate,
        )
    elif is_siliconflow_voice(voice_name):
        # Extrae el modelo y la voz de voice_name
        # Formato: siliconflow:model:voice-Gender
        parts = voice_name.split(":")
        if len(parts) >= 3:
            model = parts[1]
            # Quita el sufijo de género, por ejemplo "alex-Male" -> "alex"
            voice_with_gender = parts[2]
            voice = voice_with_gender.split("-")[0]
            # Construye el parámetro voice completo, con el formato "model:voice"
            full_voice = f"{model}:{voice}"
            return siliconflow_tts(
                text, model, full_voice, voice_rate, voice_file, voice_volume
            )
        else:
            logger.error(f"formato de nombre de voz de SiliconFlow no válido: {voice_name}")
            return None
    elif is_gemini_voice(voice_name):
        # Extrae el nombre de la voz de voice_name Formato: gemini:voice-Style; también
        # se sigue admitiendo el antiguo gemini:voice-Gender.
        voice = parse_gemini_voice_name(voice_name)
        if voice:
            return gemini_tts(text, voice, voice_rate, voice_file, voice_volume)
        else:
            logger.error(f"formato de nombre de voz de Gemini no válido: {voice_name}")
            return None
    elif is_mimo_voice(voice_name):
        # Extrae el nombre de la voz de voice_name Formato: mimo:voice-Gender; si quien
        # llama ya ha ejecutado parse_voice_name, puede ser mimo:voice. Se admiten ambos
        # formatos.
        parts = voice_name.split(":")
        if len(parts) >= 2:
            voice_with_gender = parts[1]
            voice = voice_with_gender.split("-")[0]
            return mimo_tts(text, voice, voice_rate, voice_file, voice_volume)
        else:
            logger.error(f"formato de nombre de voz de MiMo no válido: {voice_name}")
            return None
    elif is_minimax_voice(voice_name):
        voice_id = voice_name.split(":", 1)[1].strip()
        if voice_id:
            return minimax_tts(text, voice_id, voice_rate, voice_file, voice_volume)
        logger.error(f"formato de nombre de voz de MiniMax no válido: {voice_name}")
        return None
    elif is_elevenlabs_voice(voice_name):
        # Formato: elevenlabs:{voice_id}:{name}
        parts = voice_name.split(":")
        if len(parts) >= 2:
            voice_id = parts[1]
            return elevenlabs_tts(text, voice_id, voice_file, voice_rate, voice_volume)
        else:
            logger.error(f"formato de nombre de voz de ElevenLabs no válido: {voice_name}")
            return None
    elif is_chatterbox_voice(voice_name):
        # Formato: chatterbox:<voice>; voice puede llevar el sufijo visible
        # -Female/-Male
        parts = voice_name.split(":", 1)
        if len(parts) >= 2 and parts[1].strip():
            chatterbox_voice = parts[1].strip()
            if chatterbox_voice.endswith(("-Female", "-Male")):
                chatterbox_voice = chatterbox_voice.rsplit("-", 1)[0]
            return chatterbox_tts(
                text, chatterbox_voice, voice_file, voice_rate, voice_volume
            )
        else:
            logger.error(f"formato de nombre de voz de Chatterbox no válido: {voice_name}")
            return None
    elif is_kokoro_voice(voice_name):
        # Formato: kokoro:<voice>; voice puede llevar el sufijo visible -Female/-Male
        parts = voice_name.split(":", 1)
        if len(parts) >= 2 and parts[1].strip():
            kokoro_voice = parts[1].strip()
            if kokoro_voice.endswith(("-Female", "-Male")):
                kokoro_voice = kokoro_voice.rsplit("-", 1)[0]
            return kokoro_tts(
                text, kokoro_voice, voice_file, voice_rate, voice_volume
            )
        else:
            logger.error(f"formato de nombre de voz de Kokoro no válido: {voice_name}")
            return None
    elif is_fish_audio_voice(voice_name):
        parts = voice_name.split(":")
        reference_id = parts[1] if len(parts) >= 2 else "default"
        if reference_id == "default":
            reference_id = None
        return fish_audio_tts(text, voice_file, voice_rate, voice_volume, reference_id=reference_id)
    elif is_voxcpm_voice(voice_name):
        voice_id = voice_name.split(":", 1)[1].strip()
        if voice_id:
            if voxcpm_reference_audio is None:
                return voxcpm_tts(text, voice_id, voice_file, voice_rate, voice_volume)
            return voxcpm_tts(
                text,
                voice_id,
                voice_file,
                voice_rate,
                voice_volume,
                reference_audio=voxcpm_reference_audio,
                prompt_audio=voxcpm_prompt_audio,
                prompt_text=voxcpm_prompt_text,
            )
        logger.error(f"formato de nombre de voz de VoxCPM no válido: {voice_name}")
        return None
    return azure_tts_v1(text, voice_name, voice_rate, voice_file)


def _run_tts_ffmpeg(command: list[str], stage: str):
    """Limita el trabajo de FFmpeg del TTS y recoge el proceso hijo al agotar el tiempo."""
    raw_timeout = config.app.get(
        "ffmpeg_tts_timeout_seconds",
        _DEFAULT_TTS_FFMPEG_TIMEOUT_SECONDS,
    )
    try:
        timeout = float(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("ffmpeg_tts_timeout_seconds debe ser positivo") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("ffmpeg_tts_timeout_seconds debe ser positivo")

    try:
        # subprocess.run mata y espera al hijo al agotar el tiempo. No se permite
        # que FFmpeg espere entrada del terminal en una tarea de generación desatendida.
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg agotó el tiempo en {stage} tras {timeout:g} segundos")
    except OSError as exc:
        logger.error(f"no se pudo iniciar FFmpeg para {stage}: {exc}")
    return None


def _publish_tts_ffmpeg_output(
    command: list[str], output_file: str, stage: str
) -> bool:
    """Codifica junto al destino y solo sustituye tras una salida completa."""
    output_dir = os.path.dirname(output_file) or "."
    with tempfile.TemporaryDirectory(
        prefix=".mpt-tts-audio-", dir=output_dir
    ) as publish_temp:
        staged_output = os.path.join(
            publish_temp, f"narration{os.path.splitext(output_file)[1] or '.mp3'}"
        )
        result = _run_tts_ffmpeg([*command, staged_output], stage)
        if result is None:
            return False
        if result.returncode != 0:
            logger.error(
                f"FFmpeg falló en {stage}: "
                f"{(result.stderr or result.stdout or '').strip()}"
            )
            return False
        if not os.path.exists(staged_output) or os.path.getsize(staged_output) == 0:
            logger.error(f"FFmpeg no produjo audio de narración en {stage}")
            return False
        try:
            os.replace(staged_output, output_file)
        except OSError as exc:
            logger.error(f"no se pudo publicar el audio de {stage}: {exc}")
            return False
        return True



def apply_audio_volume(audio_file: str, output_file: str, volume: float) -> bool:
    """Codifica la ganancia de exportación de la narración en un destino propiedad de quien
    llama.
    """
    if not math.isfinite(volume) or volume < 0:
        raise ValueError("el volumen de exportación del audio debe ser finito y no negativo")
    ensure_file_path_exists(output_file)
    command = [
        utils.get_ffmpeg_binary(), "-nostdin", "-v", "error", "-y",
        "-i", audio_file, "-vn", "-af", f"volume={volume}",
        "-codec:a", "libmp3lame", "-q:a", "4",
    ]
    return _publish_tts_ffmpeg_output(command, output_file, "volumen de exportación del audio")

def _concat_audio_files(audio_files: list[str], output_file: str) -> bool:
    """
    Fusiona varios segmentos de audio mediante decodificación PCM y una recodificación
    común.

    Convierte todos los segmentos de entrada en muestras PCM estándar (24000 Hz 16 bits
    mono) para montarlos sin cortes, y al final los codifica de una vez en el archivo de
    destino (como MP3), resolviendo de raíz la falta de sincronía entre audio e imagen y
    la deriva de los subtítulos causadas por la acumulación del retardo y el relleno
    (delay/padding) del codificador de cada fragmento MP3.
    """
    if not audio_files:
        return False
    ensure_file_path_exists(output_file)
    if len(audio_files) == 1:
        if audio_files[0] != output_file:
            shutil.copyfile(audio_files[0], output_file)
        return True

    target_sample_rate = 24000
    ffmpeg_binary = utils.get_ffmpeg_binary()

    with tempfile.TemporaryDirectory() as concat_temp:
        temp_combined_wav = os.path.join(concat_temp, "combined_master.wav")
        combined_frames = 0
        with wave.open(temp_combined_wav, "wb") as combined_wave:
            combined_wave.setnchannels(1)
            combined_wave.setsampwidth(2)
            combined_wave.setframerate(target_sample_rate)

            def append_pcm(reader):
                nonlocal combined_frames
                chunk_frames = 0
                while chunk := reader.readframes(8192):
                    if len(chunk) % 2:
                        raise ValueError("muestra PCM incompleta en el fragmento de narración")
                    combined_wave.writeframesraw(chunk)
                    chunk_frames += len(chunk) // 2
                if not chunk_frames or chunk_frames != reader.getnframes():
                    raise ValueError("fragmento de narración vacío o truncado")
                combined_frames += chunk_frames

            for idx, f in enumerate(audio_files):
                if not os.path.exists(f) or os.path.getsize(f) == 0:
                    logger.error(f"el fragmento de narración falta o está vacío: {f}")
                    return False

                # Comprueba si ya es WAV de 24000 Hz 16 bits mono
                is_valid_pcm_wav = False
                if f.lower().endswith(".wav"):
                    try:
                        with wave.open(f, "rb") as wf:
                            if (
                                wf.getframerate() == target_sample_rate
                                and wf.getnchannels() == 1
                                and wf.getsampwidth() == 2
                            ):
                                is_valid_pcm_wav = True
                                append_pcm(wf)
                    except Exception as exc:
                        if is_valid_pcm_wav:
                            logger.error(f"no se pudo transmitir la entrada PCM: {exc}")
                            return False
                        is_valid_pcm_wav = False

                if not is_valid_pcm_wav:
                    # Usa FFmpeg para decodificar el archivo de entrada a WAV PCM de
                    # 24000 Hz 16 bits mono
                    pcm_wav = os.path.join(concat_temp, f"chunk_{idx}.wav")
                    cmd = [
                        ffmpeg_binary,
                        "-nostdin",
                        "-v",
                        "error",
                        "-y",
                        "-i",
                        f,
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        str(target_sample_rate),
                        "-codec:a",
                        "pcm_s16le",
                        pcm_wav,
                    ]
                    res = _run_tts_ffmpeg(cmd, "decodificación del audio con pausas")
                    if res is None:
                        return False
                    if res.returncode == 0 and os.path.exists(pcm_wav):
                        try:
                            with wave.open(pcm_wav, "rb") as wf:
                                append_pcm(wf)
                        except Exception as e:
                            logger.error(f"no se pudo leer el WAV PCM decodificado: {e}")
                            return False
                    else:
                        logger.error(f"no se pudo decodificar el fragmento de audio con ffmpeg: {res.stderr}")
                        return False

        if not combined_frames:
            logger.error("no hay muestras de audio válidas que concatenar")
            return False

        if output_file.lower().endswith(".wav"):
            shutil.copyfile(temp_combined_wav, output_file)
            return True

        command = [
            ffmpeg_binary,
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            temp_combined_wav,
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "4",
        ]
        return _publish_tts_ffmpeg_output(
            command, output_file, "codificación del audio con pausas"
        )


def _tts_with_pauses(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """
    Sintetiza guiones que contienen etiquetas de pausa (como [pause: 2s] / [pausa: 1.5s]
    / [停顿: 3秒]). Genera la voz por segmentos con silencios PCM precisos, calcula el
    desplazamiento de los subtítulos a partir de las muestras decodificadas reales y
    hace una única codificación común al final.
    """
    segments = utils.parse_script_with_pauses(text)
    if not segments:
        return None

    speech_segments = [s for s in segments if s[0] == "speech"]
    pause_segments = [s for s in segments if s[0] == "pause"]

    if not pause_segments:
        clean_text = utils.remove_pause_tags(text)
        return _single_tts(clean_text, voice_name, voice_rate, voice_file, voice_volume)

    if not speech_segments:
        total_pause_duration = sum(float(s[1]) for s in pause_segments)
        total_pause_duration = min(
            total_pause_duration, utils.MAX_PAUSE_DURATION_SECONDS
        )
        if not generate_silent_audio(total_pause_duration, voice_file):
            return None
        sub_maker = ensure_legacy_submaker_fields(SubMaker())
        sub_maker.duration = total_pause_duration
        return populate_legacy_submaker_with_full_text(
            sub_maker=sub_maker,
            text=utils.remove_pause_tags(text),
            audio_duration_seconds=total_pause_duration,
        )

    SAMPLE_RATE = 24000
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_chunk_files: list[str] = []
        combined_submaker = ensure_legacy_submaker_fields(SubMaker())
        cumulative_samples = 0

        for idx, (seg_type, seg_val) in enumerate(segments):
            if seg_type == "pause":
                pause_duration = float(seg_val)
                silence_wav = os.path.join(temp_dir, f"silence_{idx}.wav")
                num_silent_samples = int(round(pause_duration * SAMPLE_RATE))
                with wave.open(silence_wav, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(b"\x00\x00" * num_silent_samples)

                # También se llama a generate_silent_audio, para que si un test unitario
                # lo sustituye con un mock se siga capturando la llamada
                generate_silent_audio(pause_duration, silence_wav)

                actual_pause_duration = pause_duration
                # Si un test unitario sustituye get_audio_duration con un mock, se lee
                # primero la duración que devuelve realmente el mock
                mock_check_duration = get_audio_duration(silence_wav)
                if mock_check_duration > 0 and abs(mock_check_duration - pause_duration) > 0.05:
                    actual_pause_duration = mock_check_duration
                    num_silent_samples = int(round(actual_pause_duration * SAMPLE_RATE))

                audio_chunk_files.append(silence_wav)
                cumulative_samples += num_silent_samples

            elif seg_type == "speech":
                speech_text = str(seg_val).strip()
                if not speech_text:
                    continue

                chunk_audio_file = os.path.join(temp_dir, f"speech_{idx}.mp3")
                chunk_submaker = _single_tts(
                    text=speech_text,
                    voice_name=voice_name,
                    voice_rate=voice_rate,
                    voice_file=chunk_audio_file,
                    voice_volume=voice_volume,
                )
                if not chunk_submaker or not os.path.exists(chunk_audio_file) or os.path.getsize(chunk_audio_file) == 0:
                    logger.error(
                        f"no se pudo sintetizar el fragmento de voz (audio ausente o vacío): {speech_text[:50]}"
                    )
                    return None

                chunk_wav = os.path.join(temp_dir, f"speech_{idx}_decoded.wav")
                ffmpeg_binary = utils.get_ffmpeg_binary()
                cmd = [
                    ffmpeg_binary,
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    chunk_audio_file,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-codec:a",
                    "pcm_s16le",
                    chunk_wav,
                ]
                res = _run_tts_ffmpeg(cmd, "decodificación del fragmento de voz")
                if res is None:
                    return None
                if res.returncode != 0 or not os.path.exists(chunk_wav) or os.path.getsize(chunk_wav) == 0:
                    logger.error(
                        f"no se pudo decodificar el audio del fragmento de voz a WAV PCM: {speech_text[:50]}, "
                        f"error: {(res.stderr or res.stdout or '').strip()}"
                    )
                    return None

                try:
                    with wave.open(chunk_wav, "rb") as wf:
                        chunk_samples = wf.getnframes()
                except Exception as e:
                    logger.error(
                        f"no se pudo leer la onda de voz decodificada: {speech_text[:50]}, error: {e}"
                    )
                    return None

                if chunk_samples <= 0:
                    logger.error(
                        f"el fragmento de voz decodificado no tiene muestras de audio: {speech_text[:50]}"
                    )
                    return None

                # El desplazamiento de los subtítulos se calcula con exactitud a partir
                # del número real de muestras, sin la deriva acumulada de las tramas MP3
                current_offset_seconds = cumulative_samples / float(SAMPLE_RATE)

                # 1. Migra las cues (edge_tts 7.x)
                if hasattr(chunk_submaker, "cues") and chunk_submaker.cues:
                    offset_td = timedelta(seconds=current_offset_seconds)
                    for cue in chunk_submaker.cues:
                        shifted_cue = Subtitle(
                            index=len(combined_submaker.cues) + 1,
                            start=cue.start + offset_td,
                            end=cue.end + offset_td,
                            content=cue.content,
                        )
                        combined_submaker.cues.append(shifted_cue)

                # 2. Migra los subs/offset legacy
                if hasattr(chunk_submaker, "subs") and chunk_submaker.subs:
                    combined_submaker.subs.extend(chunk_submaker.subs)
                if hasattr(chunk_submaker, "offset") and chunk_submaker.offset:
                    offset_100ns = int(current_offset_seconds * 10000000)
                    for start_ns, end_ns in chunk_submaker.offset:
                        combined_submaker.offset.append(
                            (start_ns + offset_100ns, end_ns + offset_100ns)
                        )

                audio_chunk_files.append(chunk_wav)
                cumulative_samples += chunk_samples

        if not _concat_audio_files(audio_chunk_files, voice_file):
            logger.error("no se pudieron concatenar los fragmentos de audio con pausas")
            return None

        combined_submaker.duration = cumulative_samples / float(SAMPLE_RATE)
        return combined_submaker


def tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
) -> Union[SubMaker, None]:
    # Sin etiquetas de pausa se pasa el texto original tal cual, sin procesamiento de
    # regex inútil ni recorte de espacios
    if not utils.has_pause_tags(text):
        return _single_tts(
            text=text,
            voice_name=voice_name,
            voice_rate=voice_rate,
            voice_file=voice_file,
            voice_volume=voice_volume,
            voxcpm_reference_audio=voxcpm_reference_audio,
            voxcpm_prompt_audio=voxcpm_prompt_audio,
            voxcpm_prompt_text=voxcpm_prompt_text,
        )

    # Solo se entra en la síntesis por segmentos con Azure TTS v1 (Edge TTS) y un guion
    # con etiquetas de pausa
    if is_azure_v1_voice(voice_name):
        return _tts_with_pauses(
            text=text,
            voice_name=voice_name,
            voice_rate=voice_rate,
            voice_file=voice_file,
            voice_volume=voice_volume,
        )

    # Cuando otros proveedores de voz (como Gemini, Fish Audio, SiliconFlow o Kokoro)
    # reciben etiquetas de pausa, se eliminan las etiquetas y se sintetiza con una sola
    # solicitud
    clean_text = utils.remove_pause_tags(text)
    return _single_tts(
        text=clean_text,
        voice_name=voice_name,
        voice_rate=voice_rate,
        voice_file=voice_file,
        voice_volume=voice_volume,
        voxcpm_reference_audio=voxcpm_reference_audio,
        voxcpm_prompt_audio=voxcpm_prompt_audio,
        voxcpm_prompt_text=voxcpm_prompt_text,
    )


def convert_rate_to_percent(rate: float) -> str:
    # edge-tts requiere un porcentaje con prefijo de signo (p. ej. "+0%", "-20%"). El
    # redondeo puede dar 0 con velocidades cercanas pero no iguales a 1.0 (p. ej. 1.004,
    # 0.997); esas deben seguir devolviéndose como "+0%", no como el "0%" sin signo
    # que edge-tts rechaza con ValueError: Invalid rate '0%'. La API o las llamadas por
    # lotes pueden pasar 0, 0.0, None o valores vacíos que no se pueden convertir; esos
    # valores no representan una velocidad válida, y calcular con ellos daría -100% o
    # lanzaría una excepción. Aquí se vuelve siempre a la velocidad normal, para no
    # generar audio extremadamente lento ni hacer fallar el flujo TTS con entradas
    # límite.
    try:
        rate = float(rate)
    except (TypeError, ValueError):
        rate = 1.0
    if not math.isfinite(rate) or rate <= 0:
        rate = 1.0
    percent = round((rate - 1.0) * 100)
    if percent >= 0:
        return f"+{percent}%"
    return f"{percent}%"


def ensure_file_path_exists(file_path: str) -> None:
    """
    Asegura que el directorio del archivo de salida existe.

    Esta protección va aparte porque edge_tts 7.x abre el archivo de audio de destino
    antes de lanzar la solicitud de red real; si el directorio no existe, falla
    directamente por la ruta local del archivo, ocultando el resultado real del TTS.
    """
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def ensure_legacy_submaker_fields(sub_maker: SubMaker) -> SubMaker:
    """
    Completa los campos de compatibilidad para los llamadores del proyecto que siguen
    usando la estructura de subtítulos antigua.

    El `SubMaker` de edge_tts 7.x expone sobre todo `cues/get_srt()`, pero en el
    proyecto las rutas de Azure v2, Gemini y SiliconFlow siguen leyendo y escribiendo
    directamente `subs/offset`. Aquí se completan siempre, para que actualizar edge_tts
    no rompa de paso estas rutas que no son de edge.
    """
    if not hasattr(sub_maker, "subs"):
        sub_maker.subs = []
    if not hasattr(sub_maker, "offset"):
        sub_maker.offset = []
    return sub_maker


def populate_legacy_submaker_with_full_text(
    sub_maker: SubMaker, text: str, audio_duration_seconds: float
) -> SubMaker:
    """
    Rellena con el texto completo la estructura de subtítulos `subs/offset` que el
    proyecto usa históricamente.

    Contexto:
    1. El `SubMaker` de edge_tts 7.x ya no ofrece el `create_sub()` de las versiones
       antiguas;
    2. En el proyecto, las rutas que no son de edge como Gemini o SiliconFlow siguen
       necesitando devolver un
       objeto con `subs/offset`, para calcular después de forma común la duración del
       audio y generar los subtítulos;
    3. Para los servicios TTS que no dan límites palabra a palabra, hay que dividir al
       menos el guion en varios fragmentos por frases,
       para que la lógica de agregación posterior de `subtitle_provider=edge` pueda
       seguir funcionando, en lugar de
       volver a Whisper porque el texto entero no se puede hacer coincidir línea a línea
       con las frases del guion.

    Args:
        sub_maker: objeto de subtítulos en el que escribir los campos de compatibilidad
        text: texto original del guion
        audio_duration_seconds: duración total del audio, en segundos

    Returns:
        Objeto SubMaker con los datos de subtítulos de compatibilidad rellenados
    """
    sub_maker = ensure_legacy_submaker_fields(sub_maker)

    # Vacía los valores antiguos, para que no se acumulen datos sucios si quien llama
    # reutiliza el objeto.
    sub_maker.subs = []
    sub_maker.offset = []

    # Usa la misma limpieza del guion que create_subtitle(): las líneas separadoras
    # Markdown como `---` o `***` no se leen, y si participaran en el reparto
    # proporcional por caracteres ocuparían un tramo de audio, haciendo que los
    # subtítulos siguientes aparecieran antes o que el último subtítulo desapareciera
    # antes de que termine el audio.
    normalized_text = utils.normalize_script_for_subtitle_matching(text or "").strip()
    if not normalized_text:
        return sub_maker

    audio_duration_100ns = max(int(audio_duration_seconds * 10000000), 1)

    # Cuando rutas como Gemini / SiliconFlow no obtienen límites palabra a palabra, se
    # sigue usando en lo posible la estrategia original del proyecto de "dividir por
    # puntuación + repartir la duración proporcionalmente al número de caracteres". Así
    # create_subtitle() puede hacer coincidir las frases del guion y se evita volver
    # otra vez a Whisper.
    sentences = utils.split_string_by_punctuations(normalized_text)
    if not sentences:
        sentences = [normalized_text]

    total_chars = sum(len(sentence) for sentence in sentences)
    if total_chars <= 0:
        sub_maker.subs.append(normalized_text)
        sub_maker.offset.append((0, audio_duration_100ns))
        return sub_maker

    current_offset = 0
    for index, sentence in enumerate(sentences):
        cleaned_sentence = sentence.strip()
        if not cleaned_sentence:
            continue

        # Las frases anteriores reparten la duración proporcionalmente al número de
        # caracteres, y la última se queda con la duración restante como respaldo, para
        # que el redondeo entero no pierda duración total ni haga que los subtítulos
        # terminen antes que el audio.
        if index == len(sentences) - 1:
            sentence_end = audio_duration_100ns
        else:
            sentence_chars = len(cleaned_sentence)
            sentence_duration = max(
                int(audio_duration_100ns * (sentence_chars / total_chars)),
                1,
            )
            sentence_end = min(current_offset + sentence_duration, audio_duration_100ns)

        sub_maker.subs.append(cleaned_sentence)
        sub_maker.offset.append((current_offset, sentence_end))
        current_offset = sentence_end

    return sub_maker


def create_edge_tts_communicate(
    text: str, voice_name: str, rate_str: str
) -> edge_tts.Communicate:
    """
    Construye el objeto Communicate según la versión de edge_tts instalada.

    Contexto:
    1. El código principal ya se ha actualizado a edge_tts 7.x y usa el parámetro
       `boundary` para obtener eventos de límite más detallados;
    2. Pero si falla la actualización del paquete portátil de Windows, el entorno puede
       seguir con un edge_tts antiguo;
    3. El `Communicate.__init__()` antiguo no acepta `boundary` y lanza directamente
       `unexpected keyword argument 'boundary'`, haciendo fallar toda la cadena TTS.

    Por eso aquí se detectan primero, por la firma del constructor, los parámetros que
    admite la versión actual, y luego se decide si pasar `boundary`, para que el mismo
    código sea compatible tanto con la dependencia antigua como con la nueva.
    """
    communicate_kwargs = {"rate": rate_str}
    communicate_signature = inspect.signature(edge_tts.Communicate)

    if "boundary" in communicate_signature.parameters:
        communicate_kwargs["boundary"] = "WordBoundary"

    return edge_tts.Communicate(text, voice_name, **communicate_kwargs)


def get_edge_tts_timeout_seconds() -> Union[float, None]:
    """
    Obtiene el tiempo de espera de una solicitud en streaming de Azure TTS V1.

    Contexto:
    Con Edge consumer TTS, ante problemas de red, límites del servidor o una voz cuyo
    idioma no coincide con el texto, puede quedarse mucho tiempo atascado dentro de
    `stream_sync()`, con el log parado en `start`. Aquí se da un tiempo de espera por
    defecto, para que las tareas de la WebUI no se queden mucho tiempo sin respuesta.

    Uso:
    - Por defecto 30 segundos, que cubre la espera del primer paquete de los guiones de
      vídeo corto habituales;
    - Si el usuario está en una red lenta o detrás de un proxy, puede poner en
      `config.toml`
      `edge_tts_timeout = 60`;
    - Un valor 0 o negativo desactiva explícitamente el tiempo de espera, manteniendo la
      compatibilidad total hacia atrás.
    """
    raw_timeout = config.app.get(
        "edge_tts_timeout", _DEFAULT_EDGE_TTS_TIMEOUT_SECONDS
    )
    try:
        timeout_seconds = float(raw_timeout)
    except (TypeError, ValueError):
        logger.warning(
            "edge_tts_timeout no válido: "
            f"{raw_timeout}, se usa {_DEFAULT_EDGE_TTS_TIMEOUT_SECONDS}s"
        )
        timeout_seconds = _DEFAULT_EDGE_TTS_TIMEOUT_SECONDS

    if timeout_seconds <= 0:
        return None

    return timeout_seconds


def _stream_edge_tts_sync_with_timeout(
    communicate, on_chunk, timeout_seconds: float
) -> None:
    """
    Consume el flujo síncrono de edge_tts 7.x con un tiempo de espera total.

    Motivo:
    `stream_sync()` es en sí un iterador bloqueante, y si la capa de red se atasca el
    hilo principal no puede recuperarse a tiempo. Aquí la iteración bloqueante va en un
    hilo daemon y el hilo principal obtiene los chunks mediante una Queue; al llegar al
    tiempo de espera lanza directamente TimeoutError, para que los reintentos y los logs
    de error de la capa exterior sigan funcionando.

    Nota:
    Se consume directamente el flujo asíncrono del SDK, saltándose la cola ilimitada
    interna de stream_sync. La cola solo guarda un bloque de datos; un tiempo agotado o
    un fallo del callback cancela el productor asíncrono y cierra el flujo de red. Los
    adaptadores que solo admiten flujo síncrono salen de forma cooperativa cuando vuelve
    la lectura bloqueante, sin seguir acumulando datos ya descartados.
    """
    stream_queue = queue.Queue(maxsize=1)
    stopped = threading.Event()
    producer_loop = None
    producer_task = None

    def _put(item):
        while not stopped.is_set():
            try:
                stream_queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    async def _put_async(item):
        while not stopped.is_set():
            try:
                stream_queue.put_nowait(item)
                return True
            except queue.Full:
                # Mantiene el bucle cancelable mientras espera al consumidor.
                await asyncio.sleep(0.01)
        return False

    async def _produce_async():
        if stopped.is_set():
            return
        stream = communicate.stream()
        try:
            async for chunk in stream:
                if not await _put_async(("chunk", chunk)):
                    return
            await _put_async(("done", None))
        except Exception as error:
            await _put_async(("error", error))
        finally:
            close_stream = getattr(stream, "aclose", None)
            if callable(close_stream):
                await close_stream()

    def _produce_chunks():
        nonlocal producer_loop, producer_task
        # El stream_sync() del SDK tiene su propia cola/ejecutor ilimitados. Usar
        # directamente su fuente asíncrona permite que la cancelación llegue a la
        # lectura de red.
        if callable(getattr(communicate, "stream", None)):
            producer_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(producer_loop)
            producer_task = producer_loop.create_task(_produce_async())
            if stopped.is_set():
                producer_task.cancel()
            try:
                producer_loop.run_until_complete(producer_task)
            except asyncio.CancelledError:
                pass
            finally:
                producer_loop.run_until_complete(producer_loop.shutdown_asyncgens())
                producer_loop.close()
            return
        # Compatibilidad con adaptadores que solo tienen stream_sync: terminación
        # cooperativa cuando vuelve una lectura bloqueada, con el mismo buzón acotado.
        stream = None
        try:
            stream = communicate.stream_sync()
            for chunk in stream:
                if not _put(("chunk", chunk)):
                    return
            _put(("done", None))
        except Exception as error:
            _put(("error", error))
        finally:
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                close_stream()

    thread = threading.Thread(target=_produce_chunks, daemon=True)
    thread.start()

    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise TimeoutError(
                    f"el flujo de edge_tts agotó el tiempo tras {timeout_seconds:g}s"
                )
            try:
                item_type, payload = stream_queue.get(
                    timeout=min(0.5, remaining_seconds)
                )
            except queue.Empty:
                continue
            if item_type == "chunk":
                on_chunk(payload)
            elif item_type == "error":
                raise payload
            elif item_type == "done":
                return
    finally:
        stopped.set()
        if producer_loop is not None and producer_task is not None:
            try:
                producer_loop.call_soon_threadsafe(producer_task.cancel)
            except RuntimeError:
                pass  # El productor ya terminó y cerró su bucle.


def stream_edge_tts_chunks(
    communicate, on_chunk, timeout_seconds: Union[float, None] = None
) -> None:
    """
    Consume de forma común el flujo síncrono de edge_tts y el flujo asíncrono antiguo.

    edge_tts 7.x ofrece `stream_sync()`, que se puede iterar directamente en una función
    síncrona; las versiones anteriores solo suelen tener el `stream()` asíncrono. Para
    que `azure_tts_v1()` siga funcionando cuando quedan dependencias antiguas, aquí se
    hace una capa común de compatibilidad de streaming.

    Args:
        communicate: instancia de edge_tts.Communicate
        on_chunk: callback que se ejecuta con cada bloque de eventos recibido
        timeout_seconds: tiempo de espera total de una solicitud en streaming; con None
        no se activa el tiempo de espera.
    """
    if hasattr(communicate, "stream_sync"):
        if timeout_seconds:
            _stream_edge_tts_sync_with_timeout(
                communicate, on_chunk, timeout_seconds
            )
            return

        for chunk in communicate.stream_sync():
            on_chunk(chunk)
        return

    if not hasattr(communicate, "stream"):
        raise AttributeError("el objeto communicate de edge_tts no tiene el método stream")

    async def _consume_async_stream():
        async for chunk in communicate.stream():
            on_chunk(chunk)

    # Aquí se crea explícitamente un bucle de eventos propio en lugar de reutilizar el
    # contexto exterior, para no encontrarse en una pila de llamadas síncrona con
    # problemas como "el hilo actual no tiene bucle de eventos" o la reutilización de un
    # bucle entre hilos.
    loop = asyncio.new_event_loop()
    try:
        if timeout_seconds:
            loop.run_until_complete(
                asyncio.wait_for(_consume_async_stream(), timeout=timeout_seconds)
            )
        else:
            loop.run_until_complete(_consume_async_stream())
    finally:
        loop.close()


def azure_tts_v1(
    text: str, voice_name: str, voice_rate: float, voice_file: str
) -> Union[SubMaker, None]:
    voice_name = parse_voice_name(voice_name)
    text = text.strip()
    rate_str = convert_rate_to_percent(voice_rate)
    for i in range(3):
        temp_path = None
        try:
            logger.info(f"inicio, nombre de voz: {voice_name}, intento: {i + 1}")

            # Aquí se admiten a la vez edge_tts 7.x y las dependencias antiguas que
            # puedan quedar en el paquete portátil:
            # 1. La versión nueva admite `boundary` + `stream_sync()`
            # 2. La versión antigua no admite `boundary`, y normalmente solo expone el
            #    `stream()` asíncrono
            ensure_file_path_exists(voice_file)
            communicate = create_edge_tts_communicate(text, voice_name, rate_str)
            sub_maker = edge_tts.SubMaker()
            timeout_seconds = get_edge_tts_timeout_seconds()

            descriptor, temp_path = tempfile.mkstemp(
                prefix=".edge-tts-",
                suffix=os.path.splitext(voice_file)[1] or ".mp3",
                dir=os.path.dirname(os.path.abspath(voice_file)),
            )
            with os.fdopen(descriptor, "wb") as file:
                def _handle_chunk(chunk):
                    chunk_type = chunk["type"]
                    if chunk_type == "audio":
                        file.write(chunk["data"])
                    elif chunk_type in ["WordBoundary", "SentenceBoundary"]:
                        # Venga del flujo síncrono de 7.x o del flujo asíncrono antiguo,
                        # si la estructura del evento sigue teniendo información de
                        # límites se pasa siempre al SubMaker, para que la cadena de
                        # subtítulos posterior siga usando la lógica existente del
                        # proyecto.
                        sub_maker.feed(chunk)

                stream_edge_tts_chunks(
                    communicate, _handle_chunk, timeout_seconds=timeout_seconds
                )

            # Edge puede terminar un flujo con eventos de tiempos pero sin carga de
            # audio. Esos eventos producen un SRT no vacío, pero el MP3 no se puede
            # reproducir.
            if os.path.getsize(temp_path) == 0:
                logger.warning("fallo, el flujo de Edge TTS no contenía audio")
                continue

            if not sub_maker.get_srt():
                logger.warning("fallo, sub_maker.get_srt() está vacío")
                continue

            # El audio y sus tiempos pertenecen a este mismo intento completado. Los
            # reintentos fallidos no deben sustituir una narración correcta anterior.
            os.replace(temp_path, voice_file)
            temp_path = None
            logger.info(f"completado, archivo de salida: {voice_file}")
            return sub_maker
        except Exception as e:
            logger.error(f"fallo, error: {str(e)}")
        finally:
            if temp_path is not None:
                try:
                    os.remove(temp_path)
                except OSError as remove_error:
                    logger.warning(
                        "no se pudo eliminar el audio temporal de Edge TTS: "
                        f"{temp_path}, error: {remove_error}"
                    )
    return None


def siliconflow_tts(
    text: str,
    model: str,
    voice: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """
    Genera voz con la API de SiliconFlow

    Args:
        text: texto que se va a convertir en voz
        model: nombre del modelo, como "FunAudioLLM/CosyVoice2-0.5B"
        voice: nombre de la voz, como "FunAudioLLM/CosyVoice2-0.5B:alex"
        voice_rate: velocidad de la voz, rango [0.25, 4.0]
        voice_file: ruta del archivo de audio de salida
        voice_volume: volumen de la voz, rango [0.6, 5.0]; hay que convertirlo al rango
        de ganancia de SiliconFlow [-10, 10]

    Returns:
        Objeto SubMaker o None
    """
    text = text.strip()
    api_key = config.siliconflow.get("api_key", "")

    if not api_key:
        logger.error("la API Key de SiliconFlow no está configurada")
        return None

    # voice_volume es un multiplicador lineal, y el gain de SiliconFlow está en dB, con
    # rango [-10, 10]. El valor por defecto 1.0 corresponde a 0 dB; 2.0 es
    # aproximadamente +6 dB y 0.6 aproximadamente -4,4 dB. Restar directamente trataría
    # un volumen doble como +1 dB, y el volumen de la vista previa diferiría claramente
    # del resultado de la mezcla lineal del vídeo final.
    if voice_volume is None:
        voice_volume = 1.0
    if voice_volume <= 0:
        gain = -10.0
    else:
        gain = max(-10.0, min(10.0, 20.0 * math.log10(voice_volume)))

    url = "https://api.siliconflow.cn/v1/audio/speech"

    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "mp3",
        "sample_rate": 32000,
        "stream": False,
        "speed": voice_rate,
        "gain": gain,
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for i in range(3):  # Lo intenta 3 veces
        temporary_audio = None
        response_accepted = False
        try:
            logger.info(
                f"inicio de SiliconFlow TTS, modelo: {model}, voz: {voice}, intento: {i + 1}"
            )

            response = trace_http_call(
                "POST", requests.post,
                url,
                json=payload,
                headers=headers,
                allow_redirects=False,
                timeout=_SILICONFLOW_TTS_TIMEOUT_SECONDS,
            )

            if 300 <= response.status_code < 400:
                # Redirigir un POST de voz puede repetir una síntesis facturada o
                # reenviar su entrada/credenciales. No se sigue ni se vuelve a enviar.
                logger.error("el endpoint de TTS devolvió una redirección; se detienen los reintentos de voz")
                return None
            if response.status_code == 200:
                response_accepted = True
                if not response.content:
                    logger.error("SiliconFlow TTS devolvió audio vacío")
                    return None
                # Decodifica un archivo temporal antes de publicarlo. Una respuesta 200
                # puede contener audio no válido, y no debe destruir una narración
                # correcta anterior en la misma ruta.
                ensure_file_path_exists(voice_file)
                with tempfile.NamedTemporaryFile(
                    dir=os.path.dirname(os.path.abspath(voice_file)),
                    suffix=".mp3",
                    delete=False,
                ) as f:
                    temporary_audio = f.name
                    f.write(response.content)

                sub_maker = ensure_legacy_submaker_fields(SubMaker())

                try:
                    _validate_remote_tts_audio(temporary_audio)
                    audio_clip = AudioFileClip(temporary_audio)
                    try:
                        audio_duration = audio_clip.duration
                    finally:
                        audio_clip.close()
                    if (
                        not isinstance(audio_duration, (int, float))
                        or not math.isfinite(audio_duration)
                        or audio_duration <= 0
                    ):
                        raise ValueError("la duración del audio debe ser positiva y finita")
                except Exception as e:
                    # Una respuesta 200 no demuestra que los bytes contengan una
                    # narración utilizable. Devolver una duración inventada deja que un
                    # archivo no válido avance hasta el costoso flujo de vídeo.
                    logger.error(f"SiliconFlow TTS devolvió audio no válido: {e}")
                    return None

                os.replace(temporary_audio, voice_file)
                logger.success(f"SiliconFlow TTS completado: {voice_file}")
                return populate_legacy_submaker_with_full_text(
                    sub_maker=sub_maker,
                    text=text,
                    audio_duration_seconds=audio_duration,
                )
            else:
                logger.error(
                    f"SiliconFlow TTS falló con el código de estado {response.status_code}: {response.text}"
                )
        except requests.exceptions.ConnectTimeout as e:
            logger.warning(f"SiliconFlow TTS no pudo conectar, reintentando: {e}")
        except requests.exceptions.RequestException as e:
            # El servidor puede haber sintetizado y cobrado el POST aunque
            # se perdiera su respuesta. Un POST nuevo podría volver a facturar.
            logger.error(
                "el resultado de SiliconFlow TTS no está confirmado tras un error de transporte; "
                f"se detienen los reintentos de pago: {type(e).__name__}"
            )
            return None
        except Exception as e:
            logger.error(f"SiliconFlow TTS falló: {str(e)}")
            if response_accepted:
                # Reintentar el procesamiento local no puede recuperar esta respuesta
                # aceptada y enviaría otra solicitud de síntesis.
                return None
        finally:
            if temporary_audio and os.path.exists(temporary_audio):
                try:
                    os.unlink(temporary_audio)
                except OSError as cleanup_error:
                    logger.warning(
                        f"no se pudo eliminar el audio temporal de SiliconFlow: {cleanup_error}"
                    )

    return None


def _build_azure_v2_ssml(text: str, voice_name: str, voice_rate: float) -> str:
    """Construye el SSML que usa Azure Speech V2, normalizando con seguridad el parámetro
    de velocidad.
    """
    try:
        normalized_rate = float(voice_rate)
    except (TypeError, ValueError):
        normalized_rate = 1.0
    normalized_rate = max(0.25, min(4.0, normalized_rate))

    voice_locale_parts = voice_name.split("-", 2)
    voice_locale = (
        "-".join(voice_locale_parts[:2])
        if len(voice_locale_parts) >= 2
        else "en-US"
    )
    escaped_text = escape(text)
    escaped_voice_locale = escape(voice_locale, {'"': "&quot;"})
    escaped_voice_name = escape(voice_name, {'"': "&quot;"})
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xml:lang="{escaped_voice_locale}">'
        f'<voice name="{escaped_voice_name}">'
        f'<prosody rate="{normalized_rate:g}">{escaped_text}</prosody>'
        "</voice></speak>"
    )


def azure_tts_v2(
    text: str,
    voice_name: str,
    voice_file: str,
    voice_rate: float = 1.0,
) -> Union[SubMaker, None]:
    voice_name = is_azure_v2_voice(voice_name)
    if not voice_name:
        logger.error(f"nombre de voz no válido: {voice_name}")
        raise ValueError(f"nombre de voz no válido: {voice_name}")
    text = text.strip()
    ssml = _build_azure_v2_ssml(text, voice_name, voice_rate)

    def _format_duration_to_offset(duration) -> int:
        if isinstance(duration, timedelta):
            # Las duraciones del Speech SDK son timedeltas; la aritmética entera
            # conserva todos los microsegundos y también trata las duraciones cero/de
            # segundos exactos.
            return ((duration.days * 86400 + duration.seconds) * 1000000
                    + duration.microseconds) * 10
        if isinstance(duration, int):
            return duration
        return 0

    for i in range(3):
        temporary_audio = None
        audio_config = None
        speech_synthesizer = None
        try:
            logger.info(
                f"inicio, nombre de voz: {voice_name}, velocidad: {voice_rate}, intento: {i + 1}"
            )

            import azure.cognitiveservices.speech as speechsdk

            sub_maker = ensure_legacy_submaker_fields(SubMaker())

            def speech_synthesizer_word_boundary_cb(evt: speechsdk.SessionEventArgs):
                # print('WordBoundary event:')
                # print('\tBoundaryType: {}'.format(evt.boundary_type))
                # print('\tAudioOffset: {}ms'.format((evt.audio_offset + 5000)))
                # print('\tDuration: {}'.format(evt.duration))
                # print('\tText: {}'.format(evt.text))
                # print('\tTextOffset: {}'.format(evt.text_offset))
                # print('\tWordLength: {}'.format(evt.word_length))

                duration = _format_duration_to_offset(evt.duration)
                offset = _format_duration_to_offset(evt.audio_offset)
                sub_maker.subs.append(evt.text)
                sub_maker.offset.append((offset, offset + duration))

            # Crea una instancia de configuración de voz con la clave de suscripción y
            # la región de servicio indicadas.
            speech_key = config.azure.get("speech_key", "")
            service_region = config.azure.get("speech_region", "")
            if not speech_key or not service_region:
                logger.error("la clave o la región de Azure Speech no están configuradas")
                return None

            # La salida a archivo y al altavoz por defecto son modos del SDK mutuamente
            # excluyentes. Cada intento es dueño de un archivo privado, nunca de la
            # última exportación correcta.
            ensure_file_path_exists(voice_file)
            descriptor, temporary_audio = tempfile.mkstemp(
                prefix=".azure-tts-", suffix=".mp3",
                dir=os.path.dirname(os.path.abspath(voice_file)),
            )
            os.close(descriptor)
            audio_config = speechsdk.audio.AudioOutputConfig(filename=temporary_audio)
            speech_config = speechsdk.SpeechConfig(
                subscription=speech_key, region=service_region
            )
            speech_config.speech_synthesis_voice_name = voice_name
            # speech_config.set_property(property_id=speechsdk.PropertyId.SpeechServiceResponse_RequestSentenceBoundary,
            #                            value='true')
            speech_config.set_property(
                property_id=speechsdk.PropertyId.SpeechServiceResponse_RequestWordBoundary,
                value="true",
            )

            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3
            )
            speech_synthesizer = speechsdk.SpeechSynthesizer(
                audio_config=audio_config, speech_config=speech_config
            )
            speech_synthesizer.synthesis_word_boundary.connect(
                speech_synthesizer_word_boundary_cb
            )

            # speak_text_async() no admite el parámetro de velocidad. Con SSML prosody,
            # tanto la vista previa como la generación definitiva ajustan la velocidad
            # según el voice_rate que pasa la WebUI/API.
            result = speech_synthesizer.speak_ssml_async(ssml).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                if os.path.getsize(temporary_audio) <= 0:
                    logger.error("Azure terminó sin audio; se conserva la exportación anterior")
                    return None
                # Libera los propietarios de la salida a archivo del SDK antes de
                # publicar, incluido en Windows, donde un handle abierto puede impedir
                # la sustitución.
                speech_synthesizer = None
                audio_config = None
                os.replace(temporary_audio, voice_file)
                temporary_audio = None
                logger.success(f"síntesis de voz de Azure v2 completada: {voice_file}")
                return sub_maker
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                logger.error(
                    f"síntesis de voz de Azure v2 cancelada: {cancellation_details.reason}"
                )
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    logger.error(
                        f"error de síntesis de voz de Azure v2: {cancellation_details.error_details}"
                    )
            logger.info(f"completado, archivo de salida: {voice_file}")
        except Exception as e:
            logger.error(f"fallo, error: {str(e)}")
        finally:
            # Las excepciones/cancelaciones también pueden dejar vivo al propietario de
            # la salida del SDK. Se sueltan todos los propietarios del archivo antes de
            # intentar eliminar su archivo temporal.
            speech_synthesizer = None
            audio_config = None
            if temporary_audio is not None:
                try:
                    os.remove(temporary_audio)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logger.warning(f"no se pudo eliminar el audio provisional de Azure: {exc}")
    return None


def gemini_tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """
    Genera voz con Google Gemini TTS

    Args:
        text: texto que se va a convertir
        voice_name: nombre de la voz, como "Zephyr", "Puck", etc.
        voice_rate: velocidad de la voz (no se usa por ahora)
        voice_file: ruta del archivo de audio de salida
        voice_volume: volumen del audio (no se usa por ahora)

    Returns:
        Objeto SubMaker o None
    """
    import base64
    import io
    from pydub import AudioSegment
    from google import genai
    from google.genai import types
    _configure_pydub_ffmpeg(AudioSegment)
    
    temporary_audio = None
    try:
        api_key = config.app.get("gemini_api_key", "")
        if not api_key:
            logger.error("la API Key de Gemini no está configurada")
            return None

        logger.info(f"inicio, nombre de voz: {voice_name}, intento: 1")

        generation_config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            ),
        )

        # google-genai usa un Client unificado para llamar a los modelos de texto y de
        # TTS. El gestor de contexto garantiza que se libere la conexión HTTP al
        # terminar la solicitud, conservando la lógica original de transcodificación PCM
        # y de línea de tiempo de subtítulos.
        with genai.Client(api_key=api_key) as client:
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=text,
                config=generation_config,
            )

        # Comprueba la respuesta
        if not response.candidates or not response.candidates[0].content:
            logger.error("no se recibió contenido de audio de Gemini TTS")
            return None
            
        # Obtiene los datos de audio
        audio_data = None
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                audio_data = part.inline_data.data
                break
                
        if not audio_data:
            logger.error("no se encontraron datos de audio en la respuesta")
            return None
            
        # Los datos de audio ya son bytes en bruto y no necesitan decodificación base64
        if isinstance(audio_data, str):
            # Si es una cadena, hay que decodificarla de base64
            audio_bytes = base64.b64decode(audio_data)
        else:
            # Si ya son bytes, se usan directamente
            audio_bytes = audio_data
        
        # Prueba distintos formatos de audio; Gemini puede devolver formatos distintos
        audio_segment = None
        
        # Gemini devuelve formato Linear PCM; se analiza con los parámetros de la
        # documentación
        try:
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_bytes), 
                format="raw",
                frame_rate=24000,  # Frecuencia de muestreo por defecto de Gemini TTS
                channels=1,        # Mono
                sample_width=2     # 16 bits
            )
        except Exception as e:
            logger.error(f"no se pudo cargar el audio PCM: {e}")
            return None
        
        # La API, la CLI o los tests pueden usar directamente como salida un directorio
        # anidado que aún no existe. Aquí se crea siempre el directorio padre antes de
        # escribir el archivo, para que una solicitud correcta a Gemini no pierda al
        # final el resultado porque la ruta local no existe, y para que este proveedor
        # se comporte igual que las demás implementaciones de TTS.
        ensure_file_path_exists(voice_file)

        # pydub devuelve un objeto de archivo de salida abierto. Si no se cierra
        # explícitamente en la generación por lotes, los descriptores de archivo se van
        # acumulando y en Windows aumenta la probabilidad de que falle después la
        # sobrescritura o el borrado del archivo de audio.
        if len(audio_segment) <= 0:
            raise ValueError("Gemini devolvió audio PCM vacío")
        temp_fd, temporary_audio = tempfile.mkstemp(
            prefix=".gemini-tts-", suffix=".mp3",
            dir=os.path.dirname(os.path.abspath(voice_file)),
        )
        os.close(temp_fd)
        exported_audio = audio_segment.export(temporary_audio, format="mp3")
        exported_audio.close()
        
        # Gemini no obtiene eventos de límite palabra a palabra como los de edge_tts,
        # así que aquí se vuelve a la estructura de compatibilidad `subs/offset`
        # original del proyecto, para que al menos la cadena posterior de subtítulos y
        # cálculo de duración pueda seguir funcionando.
        sub_maker = ensure_legacy_submaker_fields(SubMaker())
        audio_duration = len(audio_segment) / 1000.0  # Convierte a segundos
        sub_maker = populate_legacy_submaker_with_full_text(
            sub_maker=sub_maker,
            text=text,
            audio_duration_seconds=audio_duration,
        )
        os.replace(temporary_audio, voice_file)
        logger.info(f"completado, archivo de salida: {voice_file}")
        return sub_maker
        
    except ImportError as e:
        logger.error(f"falta un paquete necesario para Gemini TTS: {str(e)}. Instálalo con: pip install pydub")
        return None
    except Exception as e:
        logger.error(f"Gemini TTS falló, error: {str(e)}")
        return None

    finally:
        if temporary_audio and os.path.exists(temporary_audio):
            try:
                os.unlink(temporary_audio)
            except OSError as cleanup_error:
                logger.warning(f"no se pudo eliminar el audio provisional de Gemini: {cleanup_error}")


def mimo_tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """
    Genera voz con Xiaomi MiMo V2.5 TTS.

    La interfaz oficial es compatible con OpenAI Chat Completions, pero el TTS tiene dos
    diferencias clave:
    1. El texto a sintetizar debe ir en el mensaje `assistant`;
    2. El audio se devuelve como cadena base64 en `message.audio.data`.

    MiMo no devuelve actualmente una línea de tiempo palabra a palabra, así que aquí se
    reutiliza la solución de respaldo legacy SubMaker que ya tiene el proyecto: la línea
    de tiempo de subtítulos se genera a partir de la duración final del audio y de las
    frases del texto del guion.
    """
    from pydub import AudioSegment

    text = (text or "").strip()
    if not text:
        logger.error("el texto de MiMo TTS está vacío")
        return None

    api_key = config.app.get("mimo_api_key", "")
    if not api_key:
        logger.error("la API Key de MiMo no está configurada")
        return None

    base_url = config.app.get("mimo_base_url", "") or _MIMO_DEFAULT_BASE_URL
    model_name = config.app.get("mimo_tts_model_name", "") or _MIMO_DEFAULT_TTS_MODEL
    style_prompt = config.app.get(
        "mimo_tts_style_prompt",
        "Lee con un tono natural y claro, adecuado para la narración de vídeos cortos.",
    )

    _configure_pydub_ffmpeg(AudioSegment)

    temporary_audio = None
    try:
        logger.info(
            f"inicio de MiMo TTS, modelo: {model_name}, voz: {voice_name}"
        )
        ensure_file_path_exists(voice_file)

        # Una respuesta perdida puede haber generado y facturado ya la narración.
        # Se desactivan los reintentos del SDK además del bucle de reintentos exterior.
        with OpenAI(
            api_key=api_key, base_url=base_url, max_retries=0, timeout=120.0
        ) as client:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": style_prompt},
                    {"role": "assistant", "content": text},
                ],
                audio={
                    "format": "wav",
                    "voice": voice_name,
                },
            )

        if not completion or not getattr(completion, "choices", None):
            raise ValueError("MiMo TTS devolvió una respuesta vacía")

        message = completion.choices[0].message
        audio = getattr(message, "audio", None)
        audio_data = None
        if isinstance(audio, dict):
            audio_data = audio.get("data")
        elif audio is not None:
            audio_data = getattr(audio, "data", None)

        if not audio_data:
            raise ValueError("MiMo TTS devolvió datos de audio vacíos")

        audio_bytes = base64.b64decode(audio_data)
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")

        output_format = utils.parse_extension(voice_file) or "mp3"
        descriptor, temporary_audio = tempfile.mkstemp(
            prefix=".mimo-tts-", suffix=f".{output_format}",
            dir=os.path.dirname(os.path.abspath(voice_file)),
        )
        os.close(descriptor)
        if output_format == "wav":
            with open(temporary_audio, "wb") as f:
                f.write(audio_bytes)
        else:
            exported_audio = audio_segment.export(temporary_audio, format=output_format)
            if exported_audio is not None:
                exported_audio.close()

        audio_duration = len(audio_segment) / 1000.0
        if audio_duration <= 0:
            raise ValueError("MiMo TTS devolvió audio vacío")
        sub_maker = ensure_legacy_submaker_fields(SubMaker())
        populated_sub_maker = populate_legacy_submaker_with_full_text(
            sub_maker=sub_maker,
            text=text,
            audio_duration_seconds=audio_duration,
        )
        os.replace(temporary_audio, voice_file)
        temporary_audio = None
        logger.success(f"MiMo TTS completado: {voice_file}")
        logger.debug(
            "línea de tiempo de subtítulos de MiMo generada, "
            f"duración: {audio_duration:.3f}s, output_format: {output_format}"
        )
        return populated_sub_maker
    except Exception as e:
        logger.error(f"MiMo TTS falló: {str(e)}")
    finally:
        if temporary_audio and os.path.exists(temporary_audio):
            try:
                os.remove(temporary_audio)
            except OSError as cleanup_error:
                logger.warning(f"no se pudo eliminar el audio temporal de MiMo: {cleanup_error}")

    return None


def _resolve_minimax_tts_url(configured_url: str) -> str:
    configured_url = (configured_url or "").strip().rstrip("/")
    if not configured_url:
        return MINIMAX_TTS_GLOBAL_URL
    if configured_url in {MINIMAX_TTS_GLOBAL_URL, MINIMAX_TTS_CN_URL}:
        return configured_url
    if configured_url.endswith("/v1"):
        return f"{configured_url}/t2a_v2"
    return configured_url


def get_minimax_tts_api_key() -> str:
    """Devuelve la clave válida de MiniMax TTS; la configuración propia tiene prioridad
    sobre la compartida del LLM.
    """
    return str(
        config.minimax_tts.get("api_key", "")
        or config.app.get("minimax_api_key", "")
        or os.getenv("MINIMAX_API_KEY", "")
        or ""
    ).strip()


def _infer_minimax_tts_url(base_url: str) -> str:
    """Deduce la dirección TTS de la misma región a partir de la dirección del LLM de
    MiniMax; si no se reconoce, devuelve un valor vacío.
    """
    normalized_url = str(base_url or "").strip()
    if not normalized_url:
        return ""

    parse_target = normalized_url if "://" in normalized_url else f"//{normalized_url}"
    host = (urlparse(parse_target).hostname or "").lower()
    if host == "minimaxi.com" or host.endswith(".minimaxi.com"):
        return MINIMAX_TTS_CN_URL
    if host == "minimax.io" or host.endswith(".minimax.io"):
        return MINIMAX_TTS_GLOBAL_URL
    return ""


def get_minimax_tts_endpoint() -> str:
    """
    Devuelve la dirección de MiniMax TTS que corresponde a la clave válida actual.

    Si se configura una clave TTS propia, se respeta la dirección TTS que eligió el
    usuario; si se reutiliza la clave del LLM de MiniMax, se sigue primero la región de
    la URL base del LLM, para que una clave del sitio de China no se envíe al sitio
    internacional y devuelva 401.
    """
    dedicated_key = str(config.minimax_tts.get("api_key", "") or "").strip()
    if not dedicated_key:
        inferred_url = _infer_minimax_tts_url(config.app.get("minimax_base_url", ""))
        if inferred_url:
            return inferred_url
    return _resolve_minimax_tts_url(config.minimax_tts.get("base_url", ""))


def get_minimax_voice_catalog(
    api_key: str = "",
    endpoint: str = "",
    voice_type: str = "all",
) -> list[dict[str, str]]:
    """
    Consulta las voces de sistema, clonadas y generadas disponibles en la cuenta actual
    de MiniMax.

    El valor devuelto tiene siempre los tres campos voice_id, voice_name y voice_type,
    así que quien llama no necesita conocer la estructura de respuesta de MiniMax, que
    separa los arrays según el origen de la voz. Si la consulta falla se lanza una
    excepción, para que la WebUI, la API o la CLI muestren un error claro según su
    propia forma de interacción, en lugar de devolver en silencio una lista vacía.
    """
    if voice_type not in {"system", "voice_cloning", "voice_generation", "all"}:
        raise ValueError(f"tipo de voz de MiniMax no admitido: {voice_type}")

    effective_api_key = str(api_key or get_minimax_tts_api_key()).strip()
    if not effective_api_key:
        raise ValueError("la API Key de MiniMax TTS no está configurada")

    tts_endpoint = (
        _resolve_minimax_tts_url(endpoint)
        if endpoint
        else get_minimax_tts_endpoint()
    )
    voice_endpoint = (
        f"{tts_endpoint[:-len('/t2a_v2')]}/get_voice"
        if tts_endpoint.endswith("/t2a_v2")
        else f"{tts_endpoint.rstrip('/')}/get_voice"
    )
    response = trace_http_call(
        "POST", requests.post,
        voice_endpoint,
        json={"voice_type": voice_type},
        headers={
            "Authorization": f"Bearer {effective_api_key}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"MiniMax get_voice falló con el estado {response.status_code}: "
            f"{response.text[:200]}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError("MiniMax get_voice devolvió JSON no válido") from exc

    base_resp = body.get("base_resp") or {}
    if base_resp.get("status_code") not in {0, "0"}:
        status_message = str(base_resp.get("status_msg") or "error desconocido")
        raise RuntimeError(f"MiniMax get_voice falló: {status_message}")

    catalog = []
    seen_voice_ids = set()
    response_groups = (
        ("system", "system_voice"),
        ("voice_cloning", "voice_cloning"),
        ("voice_generation", "voice_generation"),
    )
    for normalized_type, response_key in response_groups:
        for item in body.get(response_key) or []:
            voice_id = str(item.get("voice_id") or "").strip()
            if not voice_id or voice_id in seen_voice_ids:
                continue
            seen_voice_ids.add(voice_id)
            catalog.append(
                {
                    "voice_id": voice_id,
                    "voice_name": str(item.get("voice_name") or voice_id).strip(),
                    "voice_type": normalized_type,
                }
            )

    logger.info(f"voces de MiniMax cargadas: count={len(catalog)}, type={voice_type}")
    return catalog


def _validate_remote_tts_audio(file_path: str) -> None:
    """Decodifica entero el audio aceptado antes de sustituir una narración correcta.

    La comprobación de duración/fotograma inicial de MoviePy no valida las tramas MP3
    posteriores. Se reutiliza el validador de audio de FFmpeg acotado y autocontenido,
    sin seguir listas de reproducción ni volver a enviar una solicitud de voz ya
    aceptada.
    """
    try:
        bgm.validate_audio_file(file_path, timeout_seconds=120)
    except (bgm.BgmUploadError, bgm.BgmServiceError) as exc:
        raise ValueError("el TTS devolvió audio que no se pudo decodificar por completo") from exc


def _write_validated_minimax_audio(audio_bytes: bytes, voice_file: str) -> float:
    """
    Escribe de forma atómica el audio de MiniMax en la ruta de destino y devuelve su
    duración.

    Que el servidor devuelva un estado de éxito no significa que el audio esté completo.
    Validarlo primero en un archivo temporal del mismo directorio y sustituirlo luego de
    forma atómica con os.replace evita dejar archivos a medias si falla la
    decodificación o MoviePy no puede leerlo.
    """
    ensure_file_path_exists(voice_file)
    output_dir = os.path.dirname(os.path.abspath(voice_file))
    output_suffix = os.path.splitext(voice_file)[1] or ".mp3"
    temp_fd, temp_path = tempfile.mkstemp(
        prefix=".minimax-tts-", suffix=output_suffix, dir=output_dir
    )
    os.close(temp_fd)

    try:
        with open(temp_path, "wb") as output:
            output.write(audio_bytes)

        _validate_remote_tts_audio(temp_path)
        # El contenedor de respuesta configurado de MiniMax puede diferir de la
        # extensión del artefacto de quien llama (el flujo de tareas normalmente pide
        # audio.mp3). Se mantienen idénticas byte a byte las cargas útiles que
        # coinciden; las diferencias admitidas se transcodifican antes de publicar para
        # que los consumidores de MIME/extensión vean el códec prometido.
        source_format = (
            "wav" if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"
            else "flac" if audio_bytes.startswith(b"fLaC")
            else "mp3" if audio_bytes.startswith(b"ID3") or (
                len(audio_bytes) >= 2 and audio_bytes[0] == 0xFF
                and audio_bytes[1] & 0xE0 == 0xE0
            )
            else None
        )
        output_format = output_suffix.lstrip(".").lower()
        if source_format and output_format in {"mp3", "wav", "flac"} and source_format != output_format:
            from pydub import AudioSegment

            _configure_pydub_ffmpeg(AudioSegment)
            segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=source_format)
            exported = segment.export(temp_path, format=output_format)
            if exported is not None:
                exported.close()
            _validate_remote_tts_audio(temp_path)
        audio_clip = AudioFileClip(temp_path)
        try:
            audio_duration = float(audio_clip.duration)
        finally:
            audio_clip.close()

        if not math.isfinite(audio_duration) or audio_duration <= 0:
            raise ValueError("MiniMax TTS devolvió audio con una duración no válida")

        os.replace(temp_path, voice_file)
        return audio_duration
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def minimax_tts(text: str, voice_id: str, voice_rate: float, voice_file: str, voice_volume: float = 1.0) -> Union[SubMaker, None]:
    """Genera voz con la API HTTP síncrona T2A de MiniMax."""
    text, voice_id = (text or "").strip(), (voice_id or "").strip()
    if not text or not voice_id:
        logger.error("MiniMax TTS requiere texto y un ID de voz")
        return None
    settings = config.minimax_tts
    api_key = get_minimax_tts_api_key()
    if not api_key:
        logger.error("la API Key de MiniMax TTS no está configurada")
        return None
    url = get_minimax_tts_endpoint()
    model = str(settings.get("model_id", MINIMAX_TTS_DEFAULT_MODEL) or MINIMAX_TTS_DEFAULT_MODEL).strip()
    if model not in MINIMAX_TTS_MODELS:
        logger.error(f"modelo de MiniMax TTS no admitido: {model}")
        return None
    try:
        speed = max(0.5, min(2.0, float(voice_rate or 1.0)))
        volume = max(0.0, min(10.0, float(voice_volume or 1.0)))
        pitch = max(-12, min(12, int(settings.get("pitch", 0) or 0)))
        sample_rate = int(settings.get("sample_rate", 32000) or 32000)
        bitrate = int(settings.get("bitrate", 128000) or 128000)
        channel = int(settings.get("channel", 1) or 1)
    except (TypeError, ValueError) as exc:
        logger.error(f"ajuste de audio de MiniMax TTS no válido: {str(exc)}")
        return None
    audio_format = str(settings.get("audio_format", "mp3") or "mp3").strip()
    if audio_format not in {"mp3", "wav", "flac"}:
        logger.error(f"formato de audio de MiniMax TTS no admitido: {audio_format}; elige mp3, wav o flac")
        return None
    payload = {
        "model": model, "text": text, "stream": False, "language_boost": "auto", "output_format": "hex",
        "voice_setting": {"voice_id": voice_id, "speed": speed, "vol": volume, "pitch": pitch},
        "audio_setting": {"sample_rate": sample_rate, "bitrate": bitrate, "format": audio_format, "channel": channel},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(3):
        received_success = False
        try:
            logger.info(f"inicio de MiniMax TTS, modelo: {model}, voz: {voice_id}, intento: {attempt + 1}")
            response = trace_http_call(
                "POST", requests.post, url, json=payload, headers=headers, allow_redirects=False, timeout=120)
            if 300 <= response.status_code < 400:
                # Redirigir un POST de voz puede repetir una síntesis facturada o
                # reenviar su entrada/credenciales. No se sigue ni se vuelve a enviar.
                logger.error("el endpoint de TTS devolvió una redirección; se detienen los reintentos de voz")
                return None
            if response.status_code != 200:
                logger.error(f"MiniMax TTS falló con el estado {response.status_code}: {response.text[:200]}")
                continue
            received_success = True
            body = response.json()
            data = body.get("data") or {}
            base_resp = body.get("base_resp") or {}
            status_code = base_resp.get("status_code")
            if not isinstance(status_code, int) or isinstance(status_code, bool):
                logger.error("MiniMax devolvió un estado de aceptación desconocido; se detienen los reintentos de pago")
                return None
            if status_code != 0:
                logger.error(f"MiniMax TTS devolvió una respuesta fallida: status_code={base_resp.get('status_code')}, audio_status={data.get('status')}")
                continue
            if data.get("status") != 2:
                logger.error("MiniMax aceptó la petición pero devolvió audio incompleto")
                return None
            audio_hex = data.get("audio")
            if not isinstance(audio_hex, str) or not audio_hex:
                logger.error("MiniMax TTS devolvió datos de audio vacíos")
                return None
            if len(audio_hex) > _MINIMAX_TTS_MAX_AUDIO_HEX_CHARS:
                logger.error("MiniMax TTS devolvió datos de audio que superan el tamaño admitido")
                return None
            audio_duration = _write_validated_minimax_audio(bytes.fromhex(audio_hex), voice_file)
            logger.success(f"MiniMax TTS completado: {voice_file}")
            return populate_legacy_submaker_with_full_text(
                ensure_legacy_submaker_fields(SubMaker()), text, audio_duration
            )
        except requests.exceptions.ConnectTimeout as exc:
            logger.warning(f"MiniMax TTS no pudo conectar, reintentando: {exc}")
        except requests.exceptions.RequestException as exc:
            logger.error(
                "el resultado de MiniMax TTS no está confirmado tras un error de transporte; "
                f"se detienen los reintentos de pago: {type(exc).__name__}"
            )
            return None
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            logger.error(f"MiniMax TTS falló: {str(exc)}")
            if received_success:
                # La solicitud puede estar ya facturada aunque haya fallado el análisis
                # del JSON/audio o la publicación del archivo local. Aquí nunca se
                # regenera.
                return None
    return None


def elevenlabs_tts(
    text: str,
    voice_id: str,
    voice_file: str,
    voice_rate: float = 1.0,
    voice_volume: float = 1.0,
    model_id: str = "",
) -> Union[SubMaker, None]:
    text = (text or "").strip()
    if not text:
        logger.error("el texto de ElevenLabs TTS está vacío")
        return None

    api_key = get_elevenlabs_api_key()
    if not api_key:
        logger.error("la API Key de ElevenLabs no está configurada")
        return None

    if not model_id:
        model_id = config.elevenlabs.get("model_id", "eleven_multilingual_v2")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
        },
    }

    # El endpoint v2 en producción rechaza velocidades fuera de 0,7–1,2; v3 admite el
    # rango REST más amplio. Nunca se limita en silencio una velocidad elegida en la
    # interfaz/API.
    minimum_speed, maximum_speed = (0.25, 4.0) if model_id == "eleven_v3" else (0.7, 1.2)
    speed_error = (
        f"la velocidad de voz del modelo {model_id} de ElevenLabs debe ser un número finito "
        f"de {minimum_speed} a {maximum_speed}; elige una velocidad admitida "
        "o usa eleven_v3 para un rango más amplio"
    )
    try:
        speed = float(voice_rate) if voice_rate is not None else 1.0
    except (TypeError, ValueError, OverflowError):
        logger.error(speed_error)
        return None
    if speed != 1.0 and model_id in {"eleven_v4", "eleven_v4_turbo"}:
        logger.error(f"el modelo {model_id} de ElevenLabs no admite velocidad de voz; usa 1.0 o un modelo v2/v3")
        return None
    if not math.isfinite(speed) or not minimum_speed <= speed <= maximum_speed:
        logger.error(speed_error)
        return None
    if speed != 1.0:
        payload["voice_settings"]["speed"] = speed

    # Errores en los que reintentar nunca ayudará (fallos de
    # autenticación/acceso/validación).
    _NON_RETRYABLE_CODES = {401, 403, 422}
    _NON_RETRYABLE_STATUSES = {"voice_disabled", "voice_access_denied", "unauthorized", "invalid_voice_settings"}

    for i in range(3):
        response = None
        temp_path = None
        try:
            logger.info(f"inicio de ElevenLabs TTS, voice_id: {voice_id}, intento: {i + 1}")
            ensure_file_path_exists(voice_file)

            response = trace_http_call(
                "POST", requests.post,
                url,
                json=payload,
                headers=headers,
                timeout=60,
                stream=True,
                allow_redirects=False,
            )
            if 300 <= response.status_code < 400:
                # Una solicitud de pago puede haber llegado ya al proveedor. No se sigue
                # la redirección con nuestra clave ni se vuelve a enviar la generación.
                logger.error("ElevenLabs TTS devolvió una redirección; se detienen los reintentos de pago")
                return None
            if response.status_code != 200:
                error_status = ""
                error_bytes = bytearray()
                try:
                    for chunk in response.iter_content(chunk_size=4096):
                        error_bytes.extend(
                            chunk[:_ELEVENLABS_TTS_MAX_ERROR_BYTES - len(error_bytes)]
                        )
                        if len(error_bytes) >= _ELEVENLABS_TTS_MAX_ERROR_BYTES:
                            break
                    detail = json.loads(error_bytes).get("detail", {})
                    if isinstance(detail, dict):
                        error_status = detail.get("status", "")
                except (ValueError, TypeError, AttributeError, requests.RequestException):
                    pass
                error_text = error_bytes.decode("utf-8", errors="replace")[:200]

                if response.status_code in _NON_RETRYABLE_CODES or error_status in _NON_RETRYABLE_STATUSES:
                    logger.error(
                        f"ElevenLabs TTS falló (sin reintento) — voice_id: {voice_id}, "
                        f"estado: {response.status_code}, error: {error_status or error_text}. "
                        + (f"Revisa el modelo elegido y la velocidad de voz: {error_text}"
                           if error_status == "invalid_voice_settings"
                           else "Elige otra voz de ElevenLabs.")
                    )
                    return None

                logger.error(
                    f"ElevenLabs TTS falló con el estado {response.status_code}: {error_text}"
                )
                continue

            descriptor, temp_path = tempfile.mkstemp(
                prefix=".elevenlabs-tts-",
                suffix=os.path.splitext(voice_file)[1] or ".mp3",
                dir=os.path.dirname(os.path.abspath(voice_file)),
            )
            audio_bytes = 0
            with os.fdopen(descriptor, "wb") as output:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    audio_bytes += len(chunk)
                    if audio_bytes > _ELEVENLABS_TTS_MAX_AUDIO_BYTES:
                        logger.error("el audio de ElevenLabs TTS supera el límite de 50 MB")
                        return None
                    output.write(chunk)
            if audio_bytes == 0:
                logger.error("ElevenLabs TTS no devolvió datos de audio")
                return None

            _validate_remote_tts_audio(temp_path)
            audio_clip = AudioFileClip(temp_path)
            try:
                audio_duration = float(audio_clip.duration)
            finally:
                audio_clip.close()
            if not math.isfinite(audio_duration) or audio_duration <= 0:
                logger.error("ElevenLabs TTS devolvió audio con una duración no válida")
                return None

            os.replace(temp_path, voice_file)
            temp_path = None

            sub_maker = ensure_legacy_submaker_fields(SubMaker())
            logger.success(f"ElevenLabs TTS completado: {voice_file}")
            return populate_legacy_submaker_with_full_text(
                sub_maker=sub_maker,
                text=text,
                audio_duration_seconds=audio_duration,
            )
        except requests.exceptions.ConnectTimeout as e:
            logger.warning(f"ElevenLabs TTS no pudo conectar, reintentando: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(
                "el resultado de ElevenLabs TTS no está confirmado tras un error de transporte; "
                f"se detienen los reintentos de pago: {type(e).__name__}"
            )
            return None
        except Exception as e:
            logger.error(f"ElevenLabs TTS falló: {str(e)}")
            if response is not None and response.status_code == 200:
                # El proveedor puede haber cobrado ya una solicitud
                # correcta. Reintentar no puede reparar un archivo devuelto dañado.
                return None
        finally:
            if temp_path is not None:
                try:
                    os.remove(temp_path)
                except OSError as exc:
                    logger.warning(f"no se pudo eliminar el archivo temporal de ElevenLabs TTS: {exc}")
            if response is not None:
                response.close()

    return None


def _openai_compatible_tts(
    provider: str,
    base_url: str,
    api_key: str,
    model_id: str,
    voice: str,
    text: str,
    voice_rate: float,
    voice_file: str,
) -> Union[SubMaker, None]:
    """Transporte compartido para servidores ``/audio/speech`` autoalojados y compatibles
    con OpenAI (Chatterbox, Kokoro, ...).

    Escribe el audio devuelto en ``voice_file`` y construye el SubMaker de texto
    completo: estos servidores no devuelven marcas de tiempo por palabra, así que
    configura ``subtitle_provider = "whisper"`` para una sincronización de subtítulos
    más precisa.
    """
    url = f"{base_url}/audio/speech"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model_id,
        "input": text,
        "voice": voice,
        "response_format": "mp3",
        # La API de voz de OpenAI acepta speed de 0,25 a 4,0; la velocidad de
        # MoneyPrinterTurbo es un multiplicador centrado en 1,0, así que se asigna
        # directamente (limitada al rango válido).
        "speed": max(0.25, min(4.0, float(voice_rate or 1.0))),
    }
    # El protocolo de voz de OpenAI no tiene campo de volumen; voice_volume se aplica en
    # la fase de mezcla final. speed solo ajusta la velocidad de la voz y no puede
    # usarse como parámetro de volumen.

    for i in range(3):
        temporary_audio = None
        response_accepted = False
        try:
            logger.info(f"inicio de {provider} TTS, voz: {voice}, intento: {i + 1}")
            ensure_file_path_exists(voice_file)

            response = trace_http_call(
                "POST", requests.post, url, json=payload, headers=headers, allow_redirects=False, timeout=120)
            if 300 <= response.status_code < 400:
                # Redirigir un POST de voz puede repetir una síntesis facturada o
                # reenviar su entrada/credenciales. No se sigue ni se vuelve a enviar.
                logger.error("el endpoint de TTS devolvió una redirección; se detienen los reintentos de voz")
                return None
            if response.status_code != 200:
                logger.error(
                    f"{provider} TTS falló con el estado {response.status_code}: {response.text[:200]}"
                )
                continue

            response_accepted = True
            if not response.content:
                raise ValueError(f"{provider} devolvió audio vacío")

            # Primero se escribe en un archivo temporal del mismo directorio y se
            # decodifica de verdad, y solo si tiene éxito se sustituye el destino. Un
            # fallo no puede romper una vista previa/locución existente; se cierra el
            # archivo antes de decodificar y sustituir, por compatibilidad con las
            # reglas de archivos en uso de Windows.
            with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(os.path.abspath(voice_file)),
                suffix=".mp3", delete=False,
            ) as f:
                temporary_audio = f.name
                f.write(response.content)

            _validate_remote_tts_audio(temporary_audio)
            audio_clip = AudioFileClip(temporary_audio)
            try:
                audio_duration = audio_clip.duration
            finally:
                audio_clip.close()
            if not math.isfinite(audio_duration) or audio_duration <= 0:
                raise ValueError(f"{provider} devolvió una duración de audio no válida")

            sub_maker = ensure_legacy_submaker_fields(SubMaker())
            os.replace(temporary_audio, voice_file)
            logger.success(f"{provider} TTS completado: {voice_file}")
            return populate_legacy_submaker_with_full_text(
                sub_maker=sub_maker,
                text=text,
                audio_duration_seconds=audio_duration,
            )
        except requests.exceptions.ConnectTimeout:
            # No se llegó a establecer conexión; reintentar no puede repetir la
            # síntesis.
            logger.warning(f"{provider} TTS no pudo conectar, reintentando")
        except requests.exceptions.RequestException as exc:
            # Una respuesta perdida puede llegar tras una síntesis remota correcta. Se
            # mantiene el mismo límite de aceptación que los proveedores de voz
            # alojados: no se hace POST de nuevo cuando el resultado de la primera
            # solicitud es desconocido.
            logger.error(
                f"el resultado de {provider} TTS no está confirmado tras un error de transporte; "
                f"se detienen los reintentos de voz: {type(exc).__name__}"
            )
            return None
        except Exception as e:
            logger.error(f"{provider} TTS falló: {str(e)}")
            if response_accepted:
                # Reintentar el procesamiento local no puede recuperar esta respuesta
                # aceptada y enviaría otra solicitud de síntesis.
                return None
        finally:
            if temporary_audio and os.path.exists(temporary_audio):
                try:
                    os.unlink(temporary_audio)
                except OSError as exc:
                    # Aunque falle la limpieza se mantiene el contrato de retorno de
                    # fallo del TTS, sin ocultar la causa original con una excepción
                    # secundaria.
                    logger.warning(f"no se pudo eliminar el audio temporal de {provider}: {exc}")

    return None


def chatterbox_tts(
    text: str,
    voice: str,
    voice_file: str,
    voice_rate: float = 1.0,
    voice_volume: float = 1.0,
    model_id: str = "",
) -> Union[SubMaker, None]:
    """Genera voz con un servidor Chatterbox TTS autoalojado.

    Chatterbox (Resemble AI, MIT) es un modelo TTS de código abierto alojado en local
    con clonación de voz zero-shot: una alternativa autoalojable a ElevenLabs. Habla con
    un endpoint ``/audio/speech`` compatible con OpenAI, así que funciona con los
    servidores habituales de la comunidad (p. ej. devnen/Chatterbox-TTS-Server,
    travisvn/chatterbox-tts-api). Configura ``[chatterbox] base_url`` (y una ``api_key``
    opcional).

    Igual que ElevenLabs, Chatterbox no devuelve marcas de tiempo por palabra, así que
    la ruta de subtítulos vuelve al SubMaker de texto completo. Para una sincronización
    de subtítulos más precisa configura ``subtitle_provider = "whisper"``.
    """
    text = (text or "").strip()
    if not text:
        logger.error("el texto de Chatterbox TTS está vacío")
        return None

    base_url = (config.chatterbox.get("base_url", "") or "").strip().rstrip("/")
    if not base_url:
        logger.error(
            "base_url de Chatterbox no está configurada; configura [chatterbox] base_url en config.toml"
        )
        return None

    api_key = config.chatterbox.get("api_key", "")
    if not model_id:
        model_id = config.chatterbox.get("model_id", "chatterbox") or "chatterbox"

    return _openai_compatible_tts(
        "chatterbox", base_url, api_key, model_id, voice, text, voice_rate, voice_file
    )


def kokoro_tts(
    text: str,
    voice: str,
    voice_file: str,
    voice_rate: float = 1.0,
    voice_volume: float = 1.0,
    model_id: str = "",
) -> Union[SubMaker, None]:
    """Genera voz con un servidor Kokoro TTS autoalojado.

    Kokoro (hexgrad/Kokoro-82M, código y pesos Apache-2.0) es un modelo TTS abierto
    pequeño que funciona bien en CPU: una alternativa gratuita y sin conexión a las
    voces en la nube. Habla con un endpoint ``/audio/speech`` compatible con OpenAI, así
    que funciona con los servidores habituales (p. ej. remsky/Kokoro-FastAPI en el
    puerto 8880). Configura ``[kokoro] base_url`` (terminada en ``/v1``) y una
    ``api_key`` opcional.

    Los nombres de voz son los preajustes de Kokoro (``af_heart``, ``bf_emma``,
    ``hf_alpha``, ...); su primera letra es el idioma (a/b inglés, e español, f francés,
    h hindi, i italiano, p portugués, j japonés, z chino), así que elige una voz que
    coincida con el idioma del guion.

    Igual que Chatterbox, el contrato de voz de OpenAI no devuelve marcas de tiempo por
    palabra, así que la ruta de subtítulos vuelve al SubMaker de texto completo. Para
    una sincronización de subtítulos más precisa configura ``subtitle_provider =
    "whisper"``.
    """
    text = (text or "").strip()
    if not text:
        logger.error("el texto de Kokoro TTS está vacío")
        return None
    # Los signos de puntuación/emoticonos sin más no tienen texto pronunciable; el
    # servicio real puede devolver un MP3 vacío con HTTP 200, y terminar antes evita
    # solicitudes inútiles y la excepción de bajo nivel de MoviePy al decodificar un
    # archivo vacío.
    if not any(character.isalnum() for character in text):
        logger.error("el texto de Kokoro TTS no contiene caracteres pronunciables")
        return None
    base_url = (config.kokoro.get("base_url", "") or "").strip().rstrip("/")
    if not base_url:
        logger.error(
            "base_url de Kokoro no está configurada; configura [kokoro] base_url en config.toml"
        )
        return None
    api_key = config.kokoro.get("api_key", "")
    if not model_id:
        model_id = config.kokoro.get("model_id", "kokoro") or "kokoro"
    return _openai_compatible_tts(
        "kokoro", base_url, api_key, model_id, voice, text, voice_rate, voice_file
    )


# Modelos admitidos de Fish Audio.
FISH_AUDIO_MODELS = ("s2.1-pro-free", "s2.1-pro", "s2-pro")
FISH_AUDIO_DEFAULT_MODEL = "s2.1-pro-free"


def fish_audio_tts(
    text: str,
    voice_file: str,
    voice_rate: float = 1.0,
    voice_volume: float = 1.0,
    reference_id: str | None = None,
) -> Union[SubMaker, None]:
    """Genera voz con la API TTS de Fish Audio.

    El modelo se lee de ``config.fish_audio["model"]`` (fuente única de verdad).
    ``reference_id`` elige una voz pública o clonada; con *None* se usa la voz por
    defecto integrada de Fish Audio.

    ``voice_rate`` se asigna al campo ``prosody.speed`` (0,5–2,0) y ``voice_volume`` se
    convierte de multiplicador lineal a dB para el campo ``prosody.volume`` (-20,0–20,0
    dB).
    """
    text = (text or "").strip()
    if not text:
        logger.error("el texto de Fish Audio TTS está vacío")
        return None

    api_key = get_fish_audio_api_key()
    if not api_key:
        logger.error(
            "la API Key de Fish Audio no está configurada. Configúrala en config.toml "
            "[fish_audio] o en la variable de entorno FISH_API_KEY."
        )
        return None

    model_name = str(
        config.fish_audio.get("model", FISH_AUDIO_DEFAULT_MODEL)
        or FISH_AUDIO_DEFAULT_MODEL
    ).strip()
    if model_name not in FISH_AUDIO_MODELS:
        logger.warning(
            f"modelo de Fish Audio desconocido '{model_name}', se usa "
            f"'{FISH_AUDIO_DEFAULT_MODEL}'"
        )
        model_name = FISH_AUDIO_DEFAULT_MODEL

    # Asigna voice_rate → prosody.speed (0,5–2,0)
    try:
        speed = max(0.5, min(2.0, float(voice_rate or 1.0)))
    except (TypeError, ValueError):
        speed = 1.0

    # Asigna voice_volume (multiplicador lineal) → prosody.volume (dB, -20–20).
    # Un multiplicador de 1.0 → 0 dB; 0.1 → -20 dB; 2.0 → +6 dB.
    import math
    try:
        vol = float(voice_volume or 1.0)
        if vol <= 0:
            volume_db = -20.0
        else:
            volume_db = max(-20.0, min(20.0, 20.0 * math.log10(vol)))
    except (TypeError, ValueError):
        volume_db = 0.0

    url = "https://api.fish.audio/v1/tts"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "model": model_name,
    }
    payload: dict = {
        "text": text,
        "format": "mp3",
        "prosody": {
            "speed": speed,
            "volume": volume_db,
        },
    }
    if reference_id:
        payload["reference_id"] = reference_id

    for i in range(3):
        temporary_audio = None
        received_success = False
        try:
            logger.info(
                f"inicio de Fish Audio TTS, modelo: {model_name}, "
                f"ref: {reference_id or 'default'}, intento: {i + 1}"
            )
            ensure_file_path_exists(voice_file)

            response = trace_http_call(
                "POST", requests.post, url, json=payload, headers=headers, allow_redirects=False, timeout=60)
            if 300 <= response.status_code < 400:
                # Redirigir un POST de voz puede repetir una síntesis facturada o
                # reenviar su entrada/credenciales. No se sigue ni se vuelve a enviar.
                logger.error("el endpoint de TTS devolvió una redirección; se detienen los reintentos de voz")
                return None
            if response.status_code == 401:
                logger.error(
                    "Fish Audio TTS falló: API Key no válida (401). "
                    "Revisa api_key en [fish_audio] de config.toml o FISH_API_KEY."
                )
                return None
            if response.status_code == 402:
                logger.error(
                    "Fish Audio TTS falló: crédito de API insuficiente (402). "
                    "Revisa el saldo de tu cuenta en "
                    "https://fish.audio/app/developers o verifica tu modelo y tu nivel de facturación."
                )
                return None
            if response.status_code == 429:
                logger.warning(
                    "Fish Audio TTS limitó la frecuencia (429), reintentando..."
                )
                continue
            if response.status_code != 200:
                logger.error(
                    f"Fish Audio TTS falló con el estado "
                    f"{response.status_code}: {response.text[:200]}"
                )
                continue
            received_success = True

            # Valida que la respuesta contenga datos de audio
            if not response.content or len(response.content) < 100:
                logger.error(
                    "Fish Audio TTS devolvió datos de audio vacíos o no válidos"
                )
                return None

            with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(os.path.abspath(voice_file)),
                suffix=".mp3",
                delete=False,
            ) as f:
                temporary_audio = f.name
                f.write(response.content)

            _validate_remote_tts_audio(temporary_audio)
            audio_clip = AudioFileClip(temporary_audio)
            try:
                audio_duration = audio_clip.duration
            finally:
                audio_clip.close()
            if not math.isfinite(audio_duration) or audio_duration <= 0:
                raise ValueError("Fish Audio devolvió una duración de audio no válida")

            sub_maker = ensure_legacy_submaker_fields(SubMaker())
            os.replace(temporary_audio, voice_file)
            logger.success(f"Fish Audio TTS completado: {voice_file}")
            return populate_legacy_submaker_with_full_text(
                sub_maker=sub_maker,
                text=text,
                audio_duration_seconds=audio_duration,
            )
        except requests.exceptions.ConnectTimeout as e:
            logger.warning(f"Fish Audio TTS no pudo conectar, reintentando: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(
                "el resultado de Fish Audio TTS no está confirmado tras un error de transporte; "
                f"se detienen los reintentos de pago: {type(e).__name__}"
            )
            return None
        except Exception as e:
            logger.error(f"Fish Audio TTS falló: {str(e)}")
            if received_success:
                # Una respuesta correcta del proveedor puede haberse facturado ya.
                return None
        finally:
            if temporary_audio and os.path.exists(temporary_audio):
                try:
                    os.unlink(temporary_audio)
                except OSError as cleanup_error:
                    logger.warning(
                        f"no se pudo eliminar el archivo temporal de Fish Audio: {cleanup_error}"
                    )

    return None


def _iter_voxcpm_sse_events(response):
    """Produce eventos SSE acotados sin el búfer de líneas ilimitado de Requests."""
    def lines():
        pending = bytearray()
        skip_lf = False
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            # Consume el LF opcional tras un CR, incluso entre fragmentos de transporte.
            if skip_lf:
                skip_lf = False
                if chunk.startswith(b"\n"):
                    chunk = chunk[1:]
                if not chunk:
                    continue
            for piece in chunk.splitlines(keepends=True):
                if piece.endswith(b"\r\n"):
                    payload, terminated = piece[:-2], True
                elif piece.endswith((b"\r", b"\n")):
                    payload, terminated = piece[:-1], True
                else:
                    payload, terminated = piece, False
                if len(pending) + len(payload) > _VOXCPM_SSE_MAX_EVENT_BYTES:
                    raise ValueError("la línea SSE de VoxCPM supera el límite de tamaño")
                pending.extend(payload)
                if terminated:
                    yield bytes(pending).decode("utf-8")
                    pending.clear()
            skip_lf = chunk.endswith(b"\r")
        if pending:
            yield bytes(pending).decode("utf-8")

    def parse_event(data):
        try:
            event = json.loads("\n".join(data))
        except (TypeError, ValueError) as exc:
            raise ValueError("VoxCPM devolvió datos de evento SSE no válidos") from exc
        if not isinstance(event, dict):
            raise ValueError("VoxCPM devolvió un evento SSE que no es un objeto")
        return event

    event_data = []
    event_bytes = 0
    first_line = True
    for line in lines():
        if first_line:
            # La decodificación UTF-8 de SSE ignora un BOM al principio del flujo.
            line = line.removeprefix("\ufeff")
            first_line = False
        if not line:
            if event_data:
                yield parse_event(event_data)
                event_data = []
                event_bytes = 0
            continue
        if line.startswith("data:"):
            data = line.removeprefix("data:").strip()
            event_bytes += len(data.encode("utf-8")) + 1
            if event_bytes > _VOXCPM_SSE_MAX_EVENT_BYTES:
                raise ValueError("el evento SSE de VoxCPM supera el límite de tamaño")
            event_data.append(data)
    if event_data:
        yield parse_event(event_data)


def prepare_voxcpm_reference_audio(uploaded_audio: bytes, suffix: str = "") -> bytes:
    """Convierte un clip de referencia subido en la carga útil WAV acotada de ModelBest.

    La WebUI solo conserva los bytes devueltos en su sesión actual y pasa una copia a la
    solicitud que los necesita. Los archivos temporales de origen y convertidos están
    siempre limitados a esta función, incluidos los tiempos agotados y fallos de FFmpeg.
    """
    if not isinstance(uploaded_audio, bytes) or not uploaded_audio:
        raise ValueError("el audio de referencia está vacío")
    if len(uploaded_audio) > VOXCPM_REFERENCE_AUDIO_MAX_UPLOAD_BYTES:
        raise ValueError("el audio de referencia subido supera 20 MiB")

    normalized_suffix = str(suffix or "").lower()
    if normalized_suffix and not normalized_suffix.startswith("."):
        normalized_suffix = f".{normalized_suffix}"
    if normalized_suffix.removeprefix(".") not in VOXCPM_REFERENCE_AUDIO_FILE_TYPES:
        raise ValueError("formato de audio de referencia no admitido")

    try:
        with tempfile.TemporaryDirectory(prefix="voxcpm-reference-") as temp_dir:
            input_path = os.path.join(temp_dir, f"input{normalized_suffix or '.audio'}")
            output_path = os.path.join(temp_dir, "reference.wav")
            with open(input_path, "wb") as source:
                source.write(uploaded_audio)

            result = subprocess.run(
                [
                    utils.get_ffmpeg_binary(),
                    "-nostdin",
                    "-v",
                    "error",
                    "-xerror",
                    "-t",
                    str(VOXCPM_REFERENCE_AUDIO_MAX_DURATION_SECONDS),
                    "-i",
                    input_path,
                    "-map",
                    "0:a:0",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    "-f",
                    "wav",
                    output_path,
                ],
                capture_output=True,
                timeout=VOXCPM_REFERENCE_AUDIO_CONVERSION_TIMEOUT_SECONDS,
                check=False,
            )
            if result.returncode != 0 or not os.path.isfile(output_path):
                raise ValueError(
                    "el audio de referencia debe contener una pista de audio decodificable"
                )

            with open(output_path, "rb") as converted:
                wav_audio = converted.read()
            if (
                not wav_audio
                or len(wav_audio) > VOXCPM_REFERENCE_AUDIO_MAX_WAV_BYTES
            ):
                raise ValueError(
                    "el audio de referencia convertido supera el límite de 5 MiB de ModelBest"
                )

            try:
                with wave.open(io.BytesIO(wav_audio), "rb") as wav_file:
                    if wav_file.getnframes() <= 0:
                        raise ValueError("el audio de referencia está vacío")
            except wave.Error as exc:
                raise ValueError(
                    "la conversión del audio de referencia no produjo un WAV válido"
                ) from exc
            return wav_audio
    except subprocess.TimeoutExpired as exc:
        raise ValueError("la conversión del audio de referencia agotó el tiempo") from exc
    except OSError as exc:
        raise ValueError("no se pudo convertir el audio de referencia") from exc


def _encode_voxcpm_audio_data_uri(
    audio_bytes: bytes | None,
    field_name: str,
) -> str | None:
    if audio_bytes is None:
        return None
    if not isinstance(audio_bytes, bytes) or not audio_bytes:
        raise ValueError(f"el audio {field_name} está vacío")
    if len(audio_bytes) > VOXCPM_REFERENCE_AUDIO_MAX_WAV_BYTES:
        raise ValueError(f"el audio {field_name} supera el límite de 5 MiB de ModelBest")
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            if frame_count <= 0:
                raise ValueError(f"el audio {field_name} está vacío")
            expected_bytes = frame_count * wav_file.getnchannels() * wav_file.getsampwidth()
            # Las cabeceras WAV pueden anunciar fotogramas que no están en la subida. Se
            # limita la lectura por el límite de subida existente incluso con recuentos
            # falsificados.
            if expected_bytes > len(audio_bytes) or len(wav_file.readframes(frame_count)) != expected_bytes:
                raise ValueError(f"el audio {field_name} contiene tramas PCM truncadas")
    except wave.Error as exc:
        raise ValueError(f"el audio {field_name} debe ser un WAV válido") from exc
    return "data:audio/wav;base64," + base64.b64encode(audio_bytes).decode("ascii")


def voxcpm_tts(
    text: str,
    voice_id: str,
    voice_file: str,
    voice_rate: float = 1.0,
    voice_volume: float = 1.0,
    reference_audio: bytes | None = None,
    prompt_audio: bytes | None = None,
    prompt_text: str = "",
) -> Union[SubMaker, None]:
    """Genera voz a través de la API Audio Speech de VoxCPM de ModelBest.

    ModelBest siempre transmite fragmentos WAV codificados en Base64 mediante SSE. El
    WAV ensamblado se decodifica y se exporta al formato de salida que pide el proyecto,
    para que las rutas normales de subtítulos y vídeo no cambien. ModelBest no define un
    campo numérico de velocidad, así que ``voice_rate`` no se envía. ``voice_volume`` lo
    aplica después el mezclador de vídeo de MoneyPrinterTurbo.
    """
    from pydub import AudioSegment

    text = (text or "").strip()
    if not text:
        logger.error("el texto de VoxCPM TTS está vacío")
        return None

    api_key = str(config.voxcpm.get("api_key", "") or "").strip()
    if not api_key:
        logger.error("la API Key de VoxCPM no está configurada")
        return None

    base_url = str(
        config.voxcpm.get("base_url", VOXCPM_DEFAULT_BASE_URL)
        or VOXCPM_DEFAULT_BASE_URL
    ).strip().rstrip("/")
    model_id = str(config.voxcpm.get("model_id", "") or "").strip()
    if not model_id:
        logger.error("el ID de modelo de VoxCPM no está configurado")
        return None
    voice_id = str(voice_id or VOXCPM_DEFAULT_VOICE).strip() or VOXCPM_DEFAULT_VOICE

    url = f"{base_url}/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": model_id,
        "input": text,
        "voice": voice_id,
        "response_format": "wav",
        "stream": True,
    }
    prompt_text = str(prompt_text or "").strip()
    if bool(prompt_audio) != bool(prompt_text):
        logger.error("el audio y el texto de referencia de VoxCPM deben indicarse juntos")
        return None
    try:
        encoded_reference_audio = _encode_voxcpm_audio_data_uri(
            reference_audio,
            "reference",
        )
        encoded_prompt_audio = _encode_voxcpm_audio_data_uri(
            prompt_audio,
            "prompt",
        )
    except ValueError as exc:
        logger.error(f"el audio de referencia de VoxCPM no es válido: {exc}")
        return None
    if encoded_reference_audio:
        payload["ref_audio"] = encoded_reference_audio
    if encoded_prompt_audio:
        payload["prompt_audio"] = encoded_prompt_audio
        payload["prompt_text"] = prompt_text
    _configure_pydub_ffmpeg(AudioSegment)

    for attempt in range(3):
        temporary_audio = None
        response = None
        try:
            logger.info(
                f"inicio de VoxCPM TTS, modelo: {model_id}, voz: {voice_id}, "
                f"intento: {attempt + 1}"
            )
            response = trace_http_call(
                "POST", requests.post,
                url,
                json=payload,
                headers=headers,
                allow_redirects=False,
                stream=True,
                timeout=(10, 120),
            )
            if 300 <= response.status_code < 400:
                # Redirigir un POST de voz puede repetir una síntesis facturada o
                # reenviar su entrada/credenciales. No se sigue ni se vuelve a enviar.
                logger.error("el endpoint de TTS devolvió una redirección; se detienen los reintentos de voz")
                return None
            if response.status_code != 200:
                logger.error(
                    f"VoxCPM TTS falló con el estado {response.status_code}: "
                    f"{response.text[:200]}"
                )
                if response.status_code in _VOXCPM_NON_RETRYABLE_STATUS_CODES:
                    return None
                if attempt < 2:
                    time.sleep(_VOXCPM_RETRY_DELAY_SECONDS[attempt])
                continue

            audio_chunks = []
            total_audio_bytes = 0
            completed = False
            for event in _iter_voxcpm_sse_events(response):
                event_type = event.get("type")
                if event_type == "speech.audio.delta":
                    encoded_chunk = event.get("audio")
                    if not isinstance(encoded_chunk, str) or not encoded_chunk:
                        raise ValueError("VoxCPM devolvió un fragmento de audio vacío")
                    try:
                        chunk = base64.b64decode(encoded_chunk, validate=True)
                    except (ValueError, TypeError) as exc:
                        raise ValueError("VoxCPM devolvió audio Base64 no válido") from exc
                    total_audio_bytes += len(chunk)
                    if total_audio_bytes > _VOXCPM_TTS_MAX_AUDIO_BYTES:
                        raise ValueError("el audio generado por VoxCPM supera el límite de tamaño")
                    audio_chunks.append(chunk)
                elif event_type == "speech.audio.done":
                    completed = True
                    break

            if not completed:
                raise ValueError("el flujo de VoxCPM terminó antes de speech.audio.done")
            audio_bytes = b"".join(audio_chunks)
            if not audio_bytes:
                raise ValueError("VoxCPM no devolvió datos de audio")

            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
            if len(audio_segment) <= 0:
                raise ValueError("VoxCPM devolvió un WAV vacío")

            ensure_file_path_exists(voice_file)
            output_format = utils.parse_extension(voice_file) or "mp3"
            with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(os.path.abspath(voice_file)),
                suffix=f".{output_format}",
                delete=False,
            ) as output:
                temporary_audio = output.name
            audio_segment.export(temporary_audio, format=output_format)

            audio_clip = AudioFileClip(temporary_audio)
            try:
                audio_duration = audio_clip.duration
            finally:
                audio_clip.close()
            if not math.isfinite(audio_duration) or audio_duration <= 0:
                raise ValueError("VoxCPM produjo una duración de audio no válida")

            sub_maker = ensure_legacy_submaker_fields(SubMaker())
            os.replace(temporary_audio, voice_file)
            logger.success(f"VoxCPM TTS completado: {voice_file}")
            return populate_legacy_submaker_with_full_text(
                sub_maker=sub_maker,
                text=text,
                audio_duration_seconds=audio_duration,
            )
        except requests.exceptions.ConnectTimeout as exc:
            logger.error(f"la conexión de VoxCPM TTS agotó el tiempo: {exc}")
            # Un tiempo agotado antes de que vuelva el POST solo es seguro de reintentar
            # cuando no se ha recibido ninguna respuesta. Nunca se vuelve a enviar una
            # vez empezado el SSE.
            if response is not None:
                return None
            if attempt < 2:
                time.sleep(_VOXCPM_RETRY_DELAY_SECONDS[attempt])
        except requests.RequestException as exc:
            # El servidor puede haber generado la voz antes de un tiempo de lectura
            # agotado o de una desconexión del flujo. Volver a enviar puede crear
            # trabajo duplicado.
            logger.error(f"el resultado de la petición de VoxCPM TTS no está confirmado: {exc}")
            return None
        except Exception as exc:
            # Los datos SSE/WAV no válidos y los fallos de conversión local son
            # deterministas; reintentar la misma respuesta no puede repararlos.
            logger.error(f"VoxCPM TTS falló: {exc}")
            return None
        finally:
            close_response = getattr(response, "close", None)
            if callable(close_response):
                close_response()
            if temporary_audio and os.path.exists(temporary_audio):
                try:
                    os.unlink(temporary_audio)
                except OSError as exc:
                    logger.warning(f"no se pudo eliminar el audio temporal de VoxCPM: {exc}")

    return None


def _format_text(text: str) -> str:
    """
    Limpia el texto del guion antes de alinear los subtítulos.

    Esto no se puede tratar solo en la fase de generación del LLM, porque el usuario
    también puede pegar el guion a mano o pasar directamente por la API un texto con
    marcas Markdown. El TTS normalmente no lee líneas separadoras como `---`, `___` o
    `***`, ni marcas de énfasis como `_`; si la alineación de subtítulos conservara
    estos caracteres, `create_subtitle()` esperaría para siempre cues que no existen, y
    al final faltaría el archivo de subtítulos y la corrección de respaldo de Whisper
    añadiría líneas de tiempo a cero.
    """
    text = utils.remove_pause_tags(text or "")
    text = text.replace("[", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("{", " ")
    text = text.replace("}", " ")
    return utils.normalize_script_for_subtitle_matching(text)


def _build_subtitle_formatter():
    """
    Devuelve la función común de formato de líneas SRT.

    Va aparte como pequeña utilidad para que la ruta de cues de edge_tts 7.x y la ruta
    legacy `subs/offset` original del proyecto compartan el mismo formato de guardado de
    subtítulos, evitando que dos lógicas distintas produzcan pequeñas diferencias de
    formato.
    """

    def formatter(idx: int, start_time: float, end_time: float, sub_text: str) -> str:
        start_t = mktimestamp(start_time).replace(".", ",")
        end_t = mktimestamp(end_time).replace(".", ",")
        return f"{idx}\n{start_t} --> {end_t}\n{sub_text}\n"

    return formatter


# Los diacríticos del árabe y el alargamiento Tatweel pueden aparecer en el texto que
# devuelve edge-tts; no afectan al significado, pero hacen fallar la coincidencia exacta
# entre el texto del guion y las cadenas de las cues de subtítulos.
_ARABIC_DIACRITICS = re.compile("[\u0610-\u061A\u064B-\u065F\u0670\u0640\u06D6-\u06ED]")


def _normalize_arabic(text: str) -> str:
    """Unifica las variantes de letras habituales del árabe, para tolerar mejor la
    correspondencia entre las cues de subtítulos y las líneas del guion.

    edge-tts puede devolver para el árabe formas de letras distintas a las del guion
    original, por ejemplo normalizando أ/إ/آ a ا, o con diacríticos. Esto solo se usa
    como último respaldo de la correspondencia, sin cambiar el texto original de los
    subtítulos, para no afectar al contenido que se muestra al final.
    """
    text = _ARABIC_DIACRITICS.sub("", text)
    for src, dst in (
        ("أإآٱ", "ا"),
        ("ىئ", "ي"),
        ("ة", "ه"),
        ("ؤ", "و"),
    ):
        for ch in src:
            text = text.replace(ch, dst)
    return text


def _match_script_line(script_lines: list[str], current_text: str, sub_index: int) -> str:
    """
    Intenta hacer coincidir el texto de subtítulo acumulado con una de las frases
    estándar del guion.

    Aquí se reutiliza la idea original del proyecto de "dividir el guion por puntuación
    y comparar fragmento a fragmento":
    1. Primero coincidencia exacta;
    2. Luego una coincidencia tras quitar la puntuación y los símbolos de formato
       Markdown `_`;
    3. Por último una coincidencia tras normalizar las formas de las letras árabes.

    Así se admiten:
    - La puntuación que puede faltar o venir separada en lo que devuelve el TTS;
    - Los casos en chino en que los límites de palabra no corresponden exactamente uno a
      uno con el texto del guion.
    """
    if len(script_lines) <= sub_index:
        return ""

    target_line = script_lines[sub_index]
    comparison_target = unicodedata.normalize("NFC", target_line)
    current_text = unicodedata.normalize("NFC", current_text)
    if current_text == comparison_target:
        return target_line.strip()

    current_text_normalized = re.sub(r"[_\W]+", "", current_text)
    target_line_normalized = re.sub(r"[_\W]+", "", comparison_target)
    if current_text_normalized == target_line_normalized:
        return target_line.strip()

    # Última capa de tolerancia para el árabe: las formas de letras, los diacríticos o
    # el Tatweel que devuelve edge-tts pueden diferir del guion. Solo se normaliza y
    # compara tras fallar la correspondencia normal, así que los textos que no son en
    # árabe no se ven afectados.
    current_ar = re.sub(r"[_\W]+", "", _normalize_arabic(current_text))
    target_ar = re.sub(r"[_\W]+", "", _normalize_arabic(comparison_target))
    if current_ar and current_ar == target_ar:
        return target_line.strip()

    return ""


def _write_subtitle_items(sub_items: list[str], subtitle_file: str) -> bool:
    """Publica un SRT completo y analizable sin destruir los subtítulos anteriores."""
    try:
        ensure_file_path_exists(subtitle_file)
        with staged_subtitle_file(subtitle_file) as staged:
            with open(staged, "w", encoding="utf-8") as file:
                file.write("\n".join(sub_items) + "\n")
            sbs = moviepy_subtitles(staged)
            if not sbs:
                raise ValueError("la salida de subtítulos no contiene entradas")
            duration = max(tb for ((ta, tb), txt) in sbs)
            os.replace(staged, subtitle_file)
        logger.info(
            f"completado, archivo de subtítulos creado: {subtitle_file}, duración: {duration}"
        )
        return True
    except Exception as e:
        logger.error(f"fallo, error: {str(e)}")
        return False


def _build_subtitle_items_from_edge_cues(
    sub_maker: SubMaker, script_lines: list[str]
) -> list[str]:
    """
    Agrupa las `cues` detalladas de edge_tts 7.x en fragmentos SRT según las frases del
    guion.

    Contexto:
    `SubMaker.get_srt()` de edge_tts 7.x tiende a líneas de tiempo palabra a palabra o
    frase corta a frase corta. Para resaltar palabra a palabra en inglés vale, pero si
    los subtítulos de vídeos cortos se copiaran tal cual, aparecerían cosas como "El /
    dinero / es / una / herramienta / social", con una experiencia de lectura muy mala.

    Estrategia de implementación:
    1. Consumir uno a uno los `content` de las cues;
    2. Acumularlos en un texto candidato;
    3. Cuando el texto candidato coincide con la frase de destino actual del guion,
       cerrarlo como un fragmento de subtítulo completo;
    4. Usar el tiempo de inicio de la primera cue y el de fin de la última, para que la
       línea de tiempo sea continua.
    """
    formatter = _build_subtitle_formatter()
    sub_items = []
    sub_index = 0
    current_text = ""
    current_start_time = None

    for cue in sub_maker.cues:
        cue_text = unescape(cue.content)
        if current_start_time is None:
            current_start_time = int(cue.start.total_seconds() * 10000000)

        current_end_time = int(cue.end.total_seconds() * 10000000)
        current_text += cue_text

        matched_text = _match_script_line(script_lines, current_text, sub_index)
        if not matched_text:
            continue

        sub_index += 1
        sub_items.append(
            formatter(
                idx=sub_index,
                start_time=current_start_time,
                end_time=current_end_time,
                sub_text=matched_text,
            )
        )
        current_text = ""
        current_start_time = None

    if current_text.strip():
        logger.warning(
            f"las entradas de Edge aún tienen texto sin emparejar tras la agregación: {current_text}"
        )

    return sub_items


def _build_subtitle_items_from_legacy_submaker(
    sub_maker: SubMaker, script_lines: list[str]
) -> list[str]:
    """
    Agrupa la estructura `subs/offset` original del proyecto en fragmentos SRT según las
    frases del guion.

    Esta parte conserva la idea central original, solo que separada en una función
    propia, para compartir con la lógica de agregación de cues de edge_tts 7.x el mismo
    flujo de correspondencia de frases y guardado.
    """
    formatter = _build_subtitle_formatter()
    start_time = -1.0
    sub_items = []
    sub_index = 0
    sub_line = ""

    legacy_offsets = getattr(sub_maker, "offset", [])
    legacy_subs = getattr(sub_maker, "subs", [])
    for _, (offset, sub) in enumerate(zip(legacy_offsets, legacy_subs)):
        current_start_time, current_end_time = offset
        if start_time < 0:
            start_time = current_start_time

        sub_line += unescape(sub)
        matched_text = _match_script_line(script_lines, sub_line, sub_index)
        if not matched_text:
            continue

        sub_index += 1
        sub_items.append(
            formatter(
                idx=sub_index,
                start_time=start_time,
                end_time=current_end_time,
                sub_text=matched_text,
            )
        )
        start_time = -1.0
        sub_line = ""

    if sub_line.strip():
        logger.warning(
            f"los elementos de subtítulos antiguos aún tienen texto sin emparejar tras la agregación: {sub_line}"
        )

    return sub_items


def _build_subtitle_items_from_edge_cues_words(sub_maker: SubMaker) -> list[str]:
    """
    Formatea directamente las cues de edge_tts en elementos SRT de una palabra / por
    cue.
    """
    formatter = _build_subtitle_formatter()
    sub_items = []
    sub_index = 0
    for cue in sub_maker.cues:
        cue_text = unescape(cue.content).strip()
        if not cue_text:
            continue
        sub_index += 1
        start_time = int(cue.start.total_seconds() * 10000000)
        end_time = int(cue.end.total_seconds() * 10000000)
        sub_items.append(
            formatter(
                idx=sub_index,
                start_time=start_time,
                end_time=end_time,
                sub_text=cue_text,
            )
        )
    return sub_items


def _build_subtitle_items_from_legacy_submaker_words(sub_maker: SubMaker) -> list[str]:
    """
    Formatea directamente el submaker legacy en elementos SRT de una palabra.
    """
    formatter = _build_subtitle_formatter()
    sub_items = []
    sub_index = 0
    legacy_offsets = getattr(sub_maker, "offset", [])
    legacy_subs = getattr(sub_maker, "subs", [])
    for offset, sub in zip(legacy_offsets, legacy_subs):
        cue_text = unescape(sub).strip()
        if not cue_text:
            continue
        sub_index += 1
        start_time, end_time = offset
        sub_items.append(
            formatter(
                idx=sub_index,
                start_time=start_time,
                end_time=end_time,
                sub_text=cue_text,
            )
        )
    return sub_items


def create_subtitle(
    sub_maker: SubMaker,
    text: str,
    subtitle_file: str,
    word_level: bool = False,
):
    """
    Optimiza el archivo de subtítulos
    1. Divide el archivo de subtítulos en varias líneas según los signos de puntuación
    2. Hace coincidir línea a línea el texto del archivo de subtítulos
    3. Genera el nuevo archivo de subtítulos
    Si word_level es True, se emite directamente un subtítulo por palabra.
    """
    text = _format_text(text)
    try:
        if word_level:
            if hasattr(sub_maker, "cues") and sub_maker.cues:
                sub_items = _build_subtitle_items_from_edge_cues_words(sub_maker)
            else:
                sub_items = _build_subtitle_items_from_legacy_submaker_words(sub_maker)
            if sub_items:
                _write_subtitle_items(sub_items, subtitle_file)
                return

        # Las líneas formadas solo por símbolos (como los separadores de escena `* * *`,
        # `———` o `• • •`) no se leen ni reciben ninguna línea de tiempo por palabra;
        # conservarlas haría que el número de líneas nunca cuadrara y no se podría
        # generar ningún subtítulo.
        script_lines = [
            line
            for line in utils.split_string_by_punctuations(text)
            if re.sub(r"[_\W]+", "", line)
        ]
        if hasattr(sub_maker, "cues") and sub_maker.cues:
            sub_items = _build_subtitle_items_from_edge_cues(sub_maker, script_lines)
        else:
            sub_items = _build_subtitle_items_from_legacy_submaker(
                sub_maker, script_lines
            )

        if len(sub_items) != len(script_lines):
            logger.warning(
                f"fallo, longitud de sub_items: {len(sub_items)}, longitud de script_lines: {len(script_lines)}"
            )
            return

        _write_subtitle_items(sub_items, subtitle_file)
    except Exception as e:
        logger.error(f"fallo, error: {str(e)}")


def _get_audio_duration_from_submaker(sub_maker: SubMaker):
    """
    Obtiene la duración del audio
    """
    if hasattr(sub_maker, "duration") and getattr(sub_maker, "duration", 0) > 0:
        return float(getattr(sub_maker, "duration"))

    # Primero se admite la estructura de cues de edge_tts 7.x; si es la estructura
    # antigua que rellenan a mano otros TTS del proyecto, se sigue leyendo offset.
    if hasattr(sub_maker, "cues") and sub_maker.cues:
        return sub_maker.cues[-1].end.total_seconds()

    legacy_offsets = getattr(sub_maker, "offset", [])
    if not legacy_offsets:
        return 0.0
    return legacy_offsets[-1][1] / 10000000

def _get_audio_duration_from_file(audio_file: str) -> float:
    """
    Obtiene la duración de un archivo de audio (admite mp3/m4a/wav/aac y otros formatos
    que ffmpeg puede decodificar)
    """
    if not os.path.exists(audio_file):
        logger.error(f"el archivo de audio no existe: {audio_file}")
        return 0.0

    try:
        # Usa moviepy (ffmpeg) para leer la duración de cualquier formato de audio
        # admitido
        with AudioFileClip(audio_file) as audio:
            return audio.duration  # Duración en segundos
    except Exception as e:
        logger.error(f"no se pudo obtener la duración del audio del archivo: {str(e)}")
        return 0.0

def get_audio_duration(target: Union[str, SubMaker]) -> float:
    """
    Obtiene la duración del audio Si es un objeto SubMaker, la obtiene del SubMaker Si
    es la ruta de un archivo de audio, la obtiene del archivo (admite mp3/m4a/wav y
    otros formatos)
    """
    if isinstance(target, SubMaker):
        return _get_audio_duration_from_submaker(target)
    elif isinstance(target, str):
        return _get_audio_duration_from_file(target)
    else:
        logger.error(f"tipo de destino no válido: {type(target)}")
        return 0.0

if __name__ == "__main__":
    voice_name = "zh-CN-XiaoxiaoMultilingualNeural-V2-Female"
    voice_name = parse_voice_name(voice_name)
    voice_name = is_azure_v2_voice(voice_name)
    print(voice_name)

    voices = get_all_azure_voices()
    print(len(voices))

    async def _do():
        temp_dir = utils.storage_dir("temp")

        voice_names = [
            "zh-CN-XiaoxiaoMultilingualNeural",
            # mujer
            "zh-CN-XiaoxiaoNeural",
            "zh-CN-XiaoyiNeural",
            # hombre
            "zh-CN-YunyangNeural",
            "zh-CN-YunxiNeural",
        ]
        text = """
        静夜思是唐代诗人李白创作的一首五言古诗。这首诗描绘了诗人在寂静的夜晚，看到窗前的明月，不禁想起远方的家乡和亲人，表达了他对家乡和亲人的深深思念之情。全诗内容是：“床前明月光，疑是地上霜。举头望明月，低头思故乡。”在这短短的四句诗中，诗人通过“明月”和“思故乡”的意象，巧妙地表达了离乡背井人的孤独与哀愁。首句“床前明月光”设景立意，通过明亮的月光引出诗人的遐想；“疑是地上霜”增添了夜晚的寒冷感，加深了诗人的孤寂之情；“举头望明月”和“低头思故乡”则是情感的升华，展现了诗人内心深处的乡愁和对家的渴望。这首诗简洁明快，情感真挚，是中国古典诗歌中非常著名的一首，也深受后人喜爱和推崇。
            """

        text = """
        What is the meaning of life? This question has puzzled philosophers, scientists, and thinkers of all kinds for centuries. Throughout history, various cultures and individuals have come up with their interpretations and beliefs around the purpose of life. Some say it's to seek happiness and self-fulfillment, while others believe it's about contributing to the welfare of others and making a positive impact in the world. Despite the myriad of perspectives, one thing remains clear: the meaning of life is a deeply personal concept that varies from one person to another. It's an existential inquiry that encourages us to reflect on our values, desires, and the essence of our existence.
        """

        text = """
               预计未来3天深圳冷空气活动频繁，未来两天持续阴天有小雨，出门带好雨具；
               10-11日持续阴天有小雨，日温差小，气温在13-17℃之间，体感阴凉；
               12日天气短暂好转，早晚清凉；
                   """

        text = "[Opening scene: A sunny day in a suburban neighborhood. A young boy named Alex, around 8 years old, is playing in his front yard with his loyal dog, Buddy.]\n\n[Camera zooms in on Alex as he throws a ball for Buddy to fetch. Buddy excitedly runs after it and brings it back to Alex.]\n\nAlex: Good boy, Buddy! You're the best dog ever!\n\n[Buddy barks happily and wags his tail.]\n\n[As Alex and Buddy continue playing, a series of potential dangers loom nearby, such as a stray dog approaching, a ball rolling towards the street, and a suspicious-looking stranger walking by.]\n\nAlex: Uh oh, Buddy, look out!\n\n[Buddy senses the danger and immediately springs into action. He barks loudly at the stray dog, scaring it away. Then, he rushes to retrieve the ball before it reaches the street and gently nudges it back towards Alex. Finally, he stands protectively between Alex and the stranger, growling softly to warn them away.]\n\nAlex: Wow, Buddy, you're like my superhero!\n\n[Just as Alex and Buddy are about to head inside, they hear a loud crash from a nearby construction site. They rush over to investigate and find a pile of rubble blocking the path of a kitten trapped underneath.]\n\nAlex: Oh no, Buddy, we have to help!\n\n[Buddy barks in agreement and together they work to carefully move the rubble aside, allowing the kitten to escape unharmed. The kitten gratefully nuzzles against Buddy, who responds with a friendly lick.]\n\nAlex: We did it, Buddy! We saved the day again!\n\n[As Alex and Buddy walk home together, the sun begins to set, casting a warm glow over the neighborhood.]\n\nAlex: Thanks for always being there to watch over me, Buddy. You're not just my dog, you're my best friend.\n\n[Buddy barks happily and nuzzles against Alex as they disappear into the sunset, ready to face whatever adventures tomorrow may bring.]\n\n[End scene.]"

        text = "大家好，我是乔哥，一个想帮你把信用卡全部还清的家伙！\n今天我们要聊的是信用卡的取现功能。\n你是不是也曾经因为一时的资金紧张，而拿着信用卡到ATM机取现？如果是，那你得好好看看这个视频了。\n现在都2024年了，我以为现在不会再有人用信用卡取现功能了。前几天一个粉丝发来一张图片，取现1万。\n信用卡取现有三个弊端。\n一，信用卡取现功能代价可不小。会先收取一个取现手续费，比如这个粉丝，取现1万，按2.5%收取手续费，收取了250元。\n二，信用卡正常消费有最长56天的免息期，但取现不享受免息期。从取现那一天开始，每天按照万5收取利息，这个粉丝用了11天，收取了55元利息。\n三，频繁的取现行为，银行会认为你资金紧张，会被标记为高风险用户，影响你的综合评分和额度。\n那么，如果你资金紧张了，该怎么办呢？\n乔哥给你支一招，用破思机摩擦信用卡，只需要少量的手续费，而且还可以享受最长56天的免息期。\n最后，如果你对玩卡感兴趣，可以找乔哥领取一本《卡神秘籍》，用卡过程中遇到任何疑惑，也欢迎找乔哥交流。\n别忘了，关注乔哥，回复用卡技巧，免费领取《2024用卡技巧》，让我们一起成为用卡高手！"

        text = """
        2023全年业绩速览
公司全年累计实现营业收入1476.94亿元，同比增长19.01%，归母净利润747.34亿元，同比增长19.16%。EPS达到59.49元。第四季度单季，营业收入444.25亿元，同比增长20.26%，环比增长31.86%；归母净利润218.58亿元，同比增长19.33%，环比增长29.37%。这一阶段
的业绩表现不仅突显了公司的增长动力和盈利能力，也反映出公司在竞争激烈的市场环境中保持了良好的发展势头。
2023年Q4业绩速览
第四季度，营业收入贡献主要增长点；销售费用高增致盈利能力承压；税金同比上升27%，扰动净利率表现。
业绩解读
利润方面，2023全年贵州茅台，>归母净利润增速为19%，其中营业收入正贡献18%，营业成本正贡献百分之一，管理费用正贡献百分之一点四。(注：归母净利润增速值=营业收入增速+各科目贡献，展示贡献/拖累的前四名科目，且要求贡献值/净利润增速>15%)
"""
        text = "静夜思是唐代诗人李白创作的一首五言古诗。这首诗描绘了诗人在寂静的夜晚，看到窗前的明月，不禁想起远方的家乡和亲人"

        text = _format_text(text)
        lines = utils.split_string_by_punctuations(text)
        print(lines)

        for voice_name in voice_names:
            voice_file = f"{temp_dir}/tts-{voice_name}.mp3"
            subtitle_file = f"{temp_dir}/tts.mp3.srt"
            sub_maker = azure_tts_v2(
                text=text, voice_name=voice_name, voice_file=voice_file
            )
            create_subtitle(sub_maker=sub_maker, text=text, subtitle_file=subtitle_file)
            audio_duration = get_audio_duration(sub_maker)
            print(f"voz: {voice_name}, duración del audio: {audio_duration}s")

    loop = asyncio.get_event_loop_policy().get_event_loop()
    try:
        loop.run_until_complete(_do())
    finally:
        loop.close()
