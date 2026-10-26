import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from loguru import logger

from app.utils import file_security, utils


# Streamlit permite por defecto subir archivos bastante grandes, pero la música de fondo
# suele ocupar solo unos MB. Aquí se fija un límite explícito en el servidor, para que
# la API o la WebUI no escriban en disco archivos enormes que afecten a las tareas de
# vídeo del mismo proceso.
MAX_BGM_UPLOAD_BYTES = 30 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_INTERNAL_UPLOAD_PREFIX = ".bgm-upload-"
_WINDOWS_INVALID_FILENAME_CHARS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_FILENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)}
    # Win32 también reconoce como número de dispositivo los dígitos en superíndice
    # Latin-1 ¹, ² y ³, así que se tratan igual que los nombres reservados con dígitos
    # normales, para que COM¹.mp3 no se salte la protección. La lista coincide con las
    # reglas de nombres de descarga de webui/Main.py.
    | {f"{prefix}{number}" for prefix in ("COM", "LPT") for number in ("¹", "²", "³")}
)
# El nombre del archivo llega tal cual a los logs, a las respuestas de la API y a la
# interfaz de la WebUI, así que, además de los separadores de ruta y los nombres
# reservados de Win32, hay que rechazar todos los caracteres de control y los que
# cambian la forma de visualización. Antes solo se bloqueaba ord < 32 (caracteres de
# control C0), y se pasaba por alto la otra mitad del mismo problema:
# * Caracteres de control C1 U+007F-U+009F: algunos visores de logs muestran U+0085 como
#   salto de línea, así que un nombre de archivo podría falsificar líneas de log
#   adicionales que parecen independientes.
# * Caracteres de control de texto bidireccional
#   U+200E/U+200F/U+202A-U+202E/U+2066-U+2069: los caracteres tras U+202E se muestran
#   invertidos, y "photo‮gnp.mp3" parece tener otra extensión en el explorador de
#   archivos y en los logs.
# * U+2028/U+2029 (separadores Unicode de línea/párrafo) también pueden crear saltos de
#   línea visuales en una línea de log.
# Ninguno de estos caracteres se puede teclear en el campo del nombre de archivo, así
# que las subidas normales no se ven afectadas; el criterio coincide con
# normalize_task_id de app/controllers/base.py (que usa directamente isprintable).
_UNSAFE_FILENAME_CHARACTERS = frozenset(
    chr(code)
    for code in (
        *range(0x00, 0x20),
        *range(0x7F, 0xA0),
        0x200E,
        0x200F,
        0x2028,
        0x2029,
        *range(0x202A, 0x202F),
        *range(0x2066, 0x206A),
    )
)
# MoviePy decodifica al final la música de fondo con FFmpeg, así que no hace falta
# limitarla artificialmente a MP3. Aquí solo se abren las extensiones de audio
# habituales y de significado claro, para que no se suba por error un contenedor con
# vídeo como MP4 como música de fondo. La tupla es además la única fuente de datos del
# control de subida de la WebUI, para que al añadir o quitar formatos no haya
# incoherencias entre frontend y backend.
SUPPORTED_BGM_EXTENSIONS = (
    ".mp3",
    ".m4a",
    ".aac",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
)


class BgmUploadError(ValueError):
    """Indica que el archivo subido no cumple los requisitos de seguridad o de formato de
    la música de fondo.
    """


class BgmServiceError(RuntimeError):
    """Indica un fallo de ejecución del servidor, como que FFmpeg o el sistema de archivos
    no estén disponibles.
    """


def should_use_bgm(bgm_type: str | None, bgm_volume: float | None) -> bool:
    """
    Decide de forma común si la tarea actual necesita tratar alguna música de fondo.

    Esta regla no depende de la fuente concreta: si no se ha elegido fuente, el volumen
    no es válido o no es mayor que 0, tanto la aleatoria como la personalizada, Sonilo y
    los proveedores que se añadan en el futuro deben saltarse el análisis de archivos,
    la generación externa y la mezcla final. Ponerlo en el servicio común de BGM evita
    copiar la comprobación del volumen 0 cada vez que se añade un proveedor.
    """
    if not str(bgm_type or "").strip():
        return False
    try:
        normalized_volume = float(bgm_volume or 0)
    except (TypeError, ValueError):
        return False
    return math.isfinite(normalized_volume) and normalized_volume > 0


def uploaded_bgm_dir(create: bool = True) -> str:
    """
    Devuelve el directorio persistente de la música de fondo del usuario.

    Las canciones integradas son recursos del código y siguen en resource/songs; lo que
    sube el usuario son datos de ejecución y debe ir en el storage que Docker ya monta,
    para conservarse al reconstruir el contenedor y no ensuciar el área de trabajo de
    Git.
    """
    return utils.storage_dir("bgm", create=create)


def _remove_staged_file(file_path: str) -> None:
    """Intenta limpiar el archivo temporal de subida sin sobrescribir la excepción original
    que está tratando quien llama.
    """
    if not file_path or not os.path.exists(file_path):
        return
    try:
        os.remove(file_path)
    except OSError as exc:
        # El archivo temporal usa un prefijo reservado y no entra en la lista de BGM; un
        # fallo de limpieza no debe sobrescribir una excepción original más precisa como
        # "audio no válido", pero debe dejar la ruta y el error del sistema para que
        # operaciones lo localice.
        logger.warning(
            f"no se pudo eliminar la música de fondo temporal: path={file_path}, "
            f"error={str(exc)}"
        )


def sanitize_upload_filename(filename: str) -> str:
    """Extrae un nombre de archivo de audio que se pueda mostrar en cualquier plataforma,
    rechazando nombres no válidos y extensiones no admitidas.
    """
    safe_name = (filename or "").replace("\\", "/").split("/")[-1].strip()
    if (
        not safe_name
        or safe_name in {".", ".."}
        or len(safe_name) > 255
        or any(character in _UNSAFE_FILENAME_CHARACTERS for character in safe_name)
        or any(character in _WINDOWS_INVALID_FILENAME_CHARS for character in safe_name)
        or safe_name.lower().startswith(_INTERNAL_UPLOAD_PREFIX)
    ):
        raise BgmUploadError("nombre de archivo de música de fondo no válido")

    # Windows reconoce como nombre de dispositivo el primer segmento antes de la
    # extensión; por ejemplo, CON.mp3 y LPT1.wav no pueden crearse como archivos
    # normales. Aunque el servidor acabe usando un UUID, rechazar antes estos nombres
    # garantiza que la API se comporte igual con las entradas en todas las plataformas.
    windows_basename = safe_name.split(".", 1)[0].rstrip(" .").upper()
    if windows_basename in _WINDOWS_RESERVED_FILENAMES:
        raise BgmUploadError("nombre de archivo de música de fondo no válido")
    if Path(safe_name).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS:
        supported_formats = ", ".join(
            extension.removeprefix(".").upper()
            for extension in SUPPORTED_BGM_EXTENSIONS
        )
        raise BgmUploadError(
            f"formato de música de fondo no admitido; formatos admitidos: {supported_formats}"
        )
    return safe_name


def _validate_audio(file_path: str, timeout_seconds: int = 30) -> None:
    """
    Usa solo el FFmpeg configurado actualmente en el proyecto para verificar que el
    archivo contiene un stream de audio que se puede decodificar entero.

    El proyecto permite que imageio-ffmpeg proporcione un FFmpeg portátil, y esa forma
    de instalación no garantiza que exista también FFprobe, así que no se puede añadir
    una dependencia de otro binario. `-map 0:a:0` falla si no hay stream de audio, y
    `-xerror` convierte los errores de decodificación en fallo; decodificar entero
    también detecta archivos cifrados o datos aleatorios que coincidan por casualidad
    con una cabecera de trama de audio. El archivo puede contener streams adicionales
    como la portada del álbum, pero solo se valida el primer stream de audio.
    """
    try:
        decoded = subprocess.run(
            [
                utils.get_ffmpeg_binary(),
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                # Los archivos subidos deben ser medios autocontenidos, no listas de
                # reproducción que hagan que la validación o el renderizado lean
                # archivos o URL contiguos.
                "-protocol_whitelist",
                "file,pipe",
                "-format_whitelist",
                "mp3,mov,aac,wav,flac,ogg,asf",
                "-i",
                file_path,
                "-map",
                "0:a:0",
                # Normaliza las marcas de tiempo a partir de las muestras decodificadas.
                # Un WAV que solo tiene cabecera puede terminar con éxito sin producir
                # audio.
                "-af",
                "asetpts=N/SR/TB",
                "-progress",
                "pipe:1",
                "-nostats",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BgmServiceError("se agotó el tiempo de la validación de la música de fondo con FFmpeg") from exc
    except OSError as exc:
        raise BgmServiceError("no se pudo ejecutar FFmpeg para validar la música de fondo") from exc
    if decoded.returncode != 0:
        raise BgmUploadError("el archivo subido debe contener un stream de audio que se pueda decodificar")
    decoded_audio = False
    for line in decoded.stdout.splitlines():
        key, separator, value = line.partition(b"=")
        if separator and key == b"out_time_us":
            try:
                decoded_audio = decoded_audio or int(value) > 0
            except ValueError:
                continue
    if not decoded_audio:
        raise BgmUploadError("el archivo subido debe contener audio no vacío que se pueda decodificar")


def validate_audio_file(file_path: str, timeout_seconds: int = 120) -> None:
    """
    Verifica que el archivo de audio en disco se puede decodificar entero con el FFmpeg
    del proyecto.

    La validación previa de una subida suele necesitar solo 30 segundos; la música que
    genera Sonilo puede durar hasta 6 minutos, así que se ofrece una entrada
    reutilizable con tiempo de espera ajustable. El servicio solo depende de FFmpeg y no
    exige que el sistema tenga instalado FFprobe.
    """
    if not os.path.isfile(file_path) or os.path.getsize(file_path) <= 0:
        raise BgmUploadError("el archivo de música de fondo está vacío o no existe")
    _validate_audio(file_path, timeout_seconds=timeout_seconds)


def _stage_bgm_upload(filename: str, source: BinaryIO) -> tuple[str, str, int]:
    """
    Escribe el flujo subido en un archivo temporal del mismo directorio y devuelve el
    nombre seguro, la ruta temporal y el número de bytes.

    La validación previa de subidas de la WebUI y la persistencia final deben usar
    exactamente la misma lectura por bloques, el mismo límite de tamaño y las mismas
    reglas de nombre de archivo; si no, podría ocurrir que la interfaz lo muestre como
    utilizable y que, al pulsar generar, el servidor lo rechace. Quien llama borra o
    sustituye de forma atómica el archivo temporal tras completar la detección del
    audio.
    """
    safe_name = sanitize_upload_filename(filename)
    try:
        target_dir = uploaded_bgm_dir(create=True)
    except OSError as exc:
        raise BgmServiceError("no se pudo preparar el almacenamiento de la música de fondo") from exc
    temp_path = ""
    total_bytes = 0

    try:
        try:
            source.seek(0)
        except (AttributeError, OSError) as exc:
            raise BgmUploadError("la subida de música de fondo no permite el acceso aleatorio (seek)") from exc

        # Se conserva la extensión original para que FFmpeg elija el demuxer correcto en
        # formatos sin cabecera de contenedor como AAC; el archivo temporal sigue en el
        # directorio de destino, para que el os.replace final sea atómico.
        descriptor, temp_path = tempfile.mkstemp(
            prefix=_INTERNAL_UPLOAD_PREFIX,
            suffix=Path(safe_name).suffix.lower(),
            dir=target_dir,
        )
        with os.fdopen(descriptor, "wb") as output:
            while True:
                chunk = source.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise BgmUploadError("la subida de música de fondo debe ser binaria")
                total_bytes += len(chunk)
                if total_bytes > MAX_BGM_UPLOAD_BYTES:
                    raise BgmUploadError("el archivo de música de fondo supera el límite de 30 MB")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        if total_bytes == 0:
            raise BgmUploadError("el archivo de música de fondo está vacío")
        return safe_name, temp_path, total_bytes
    except Exception as exc:
        _remove_staged_file(temp_path)
        if isinstance(exc, BgmUploadError):
            raise
        if isinstance(exc, OSError):
            raise BgmServiceError("no se pudo preparar la subida de música de fondo") from exc
        raise
    finally:
        # Streamlit también necesita usar el mismo UploadedFile para la vista previa en
        # el navegador; restaurar el puntero del archivo evita que tras la validación el
        # reproductor o el guardado final lean contenido vacío.
        try:
            source.seek(0)
        except (AttributeError, OSError):
            pass


def validate_bgm_upload(filename: str, source: BinaryIO) -> str:
    """Valida por completo el audio subido sin persistirlo, para la comprobación previa de
    la WebUI antes de mostrar "listo".
    """
    safe_name, temp_path, total_bytes = _stage_bgm_upload(filename, source)
    try:
        _validate_audio(temp_path)
        logger.debug(
            f"subida de música de fondo validada: name={safe_name}, "
            f"size={total_bytes} bytes"
        )
        return safe_name
    finally:
        _remove_staged_file(temp_path)


def save_bgm_upload(filename: str, source: BinaryIO) -> str:
    """
    Guarda la música de fondo del usuario por bloques, con límite de tamaño y
    sustitución atómica.

    Se usa tanto con UploadFile de FastAPI como con UploadedFile de Streamlit, que
    ofrecen ambos una interfaz de archivo binario. Escribir primero en un temporal del
    mismo directorio, validarlo y luego guardarlo de forma atómica con os.replace evita
    que las subidas concurrentes o las interrupciones del proceso dejen medio archivo de
    audio, y además da a cada subida con el mismo nombre una clave de almacenamiento
    UUID distinta, de modo que las tareas en cola o en ejecución siempre usan el archivo
    inmutable original.
    """
    safe_name, temp_path, total_bytes = _stage_bgm_upload(filename, source)
    stored_name = f"{uuid4().hex}{Path(safe_name).suffix.lower()}"
    target_path = os.path.join(os.path.dirname(temp_path), stored_name)

    try:
        _validate_audio(temp_path)
        try:
            os.replace(temp_path, target_path)
        except OSError as exc:
            raise BgmServiceError("no se pudo guardar la subida de música de fondo") from exc
        temp_path = ""
        logger.info(
            f"música de fondo subida: original_name={safe_name}, "
            f"stored_name={stored_name}, size={total_bytes} bytes"
        )
        return stored_name
    finally:
        _remove_staged_file(temp_path)


def _list_bgm_files(directories: tuple[str, ...]) -> list[str]:
    """Enumera, por orden de prioridad de directorio, los archivos de música de fondo
    seguros y admitidos.
    """
    files_by_name: dict[str, str] = {}
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory), key=str.lower):
            # Tanto la validación previa de subidas como el guardado final crean
            # brevemente archivos en el mismo directorio. Aunque el archivo temporal
            # tenga una extensión de audio válida, todavía no se ha validado y la lista
            # de BGM aleatoria no puede elegirlo antes de tiempo.
            if name.startswith(_INTERNAL_UPLOAD_PREFIX):
                continue
            if Path(name).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS:
                continue
            file_path = os.path.join(directory, name)
            try:
                # El resultado de la enumeración también necesita validar la ruta real.
                # Si no, un atacante podría colocar en un directorio permitido un enlace
                # simbólico de audio que apunte a un archivo externo y pasárselo a
                # MoviePy a través de la ruta de la BGM aleatoria.
                resolved_path = file_security.resolve_path_within_directory(
                    directory, file_path
                )
            except ValueError as exc:
                logger.warning(
                    f"se omite un archivo de música de fondo inseguro: name={name}, error={str(exc)}"
                )
                continue
            files_by_name[name] = resolved_path
    return [files_by_name[name] for name in sorted(files_by_name, key=str.lower)]


def list_builtin_bgm_files() -> list[str]:
    """
    Lista la música de fondo integrada que se distribuye con el proyecto.

    Las "canciones predefinidas" de la WebUI y la importación/exportación de preajustes
    solo usan esta lista, para que el nombre de archivo guardado se pueda restaurar en
    otro equipo con la misma versión; los archivos subidos por el usuario siguen
    gestionándose en la música personalizada.
    """
    return _list_bgm_files((utils.song_dir(),))


def list_bgm_files() -> list[str]:
    """Lista la música de fondo disponible, subida por el usuario e integrada; con nombres
    repetidos tiene prioridad el archivo subido.
    """
    return _list_bgm_files((utils.song_dir(), uploaded_bgm_dir(create=True)))


def resolve_builtin_bgm_file(unsafe_path: str) -> str:
    """Resuelve la música de fondo integrada por nombre de archivo, rechazando rutas,
    archivos desconocidos y archivos subidos por el usuario.
    """
    if not unsafe_path:
        raise ValueError("hace falta el nombre del archivo de música de fondo")

    filename = str(unsafe_path)
    if filename != os.path.basename(filename):
        raise ValueError("la música de fondo predefinida debe usar un nombre de archivo")

    files_by_name = {
        os.path.basename(file_path): file_path for file_path in list_builtin_bgm_files()
    }
    if filename not in files_by_name:
        raise ValueError("la música de fondo predefinida no está disponible")
    return files_by_name[filename]


def resolve_bgm_file(unsafe_path: str) -> str:
    """
    Resuelve la BGM en el directorio de subidas del usuario y en el de canciones
    integradas, rechazando las rutas fuera de esas dos listas blancas y los archivos
    temporales que dejan subidas interrumpidas.

    El nombre de archivo busca primero en el directorio del usuario, manteniendo los
    usos antiguos como `output000.mp3`, rutas absolutas en lista blanca y
    `./resource/songs/output000.mp3`. Las subidas nuevas usan UUID, así que normalmente
    no coinciden en nombre con las canciones integradas ni con subidas históricas.
    """
    if (
        not unsafe_path
        or Path(unsafe_path).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS
    ):
        raise ValueError("ruta de música de fondo no admitida")
    if os.path.basename(str(unsafe_path)).lower().startswith(_INTERNAL_UPLOAD_PREFIX):
        # Tanto la validación previa de subidas como el guardado final crean brevemente
        # en el mismo directorio archivos intermedios con el prefijo ``.bgm-upload-``.
        # Tienen una extensión de audio válida pero aún no se han validado, y
        # ``_list_bgm_files`` ya los excluye de la BGM aleatoria; la entrada de
        # resolución debe llegar a la misma conclusión, o un nombre pasado por la
        # API/CLI podría apuntar a un archivo intermedio incompleto cuya escritura se
        # interrumpió. Los nombres que pasa el usuario se comparan sin distinguir
        # mayúsculas, igual que los sistemas de archivos de Windows/macOS.
        raise ValueError("los archivos temporales de subida de música de fondo no se pueden elegir")

    candidates = [unsafe_path]
    if not os.path.isabs(unsafe_path):
        candidates.append(os.path.join(utils.root_dir(), unsafe_path))

    last_error = ValueError("el archivo de música de fondo no existe")
    for directory in (uploaded_bgm_dir(create=True), utils.song_dir()):
        for candidate in candidates:
            try:
                return file_security.resolve_path_within_directory(directory, candidate)
            except ValueError as exc:
                last_error = exc
    raise ValueError(str(last_error)) from last_error
