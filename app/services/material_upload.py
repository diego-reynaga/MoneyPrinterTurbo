import os
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import BinaryIO, Literal
from uuid import uuid4

from loguru import logger
from PIL import Image, UnidentifiedImageError

from app.utils import utils


# El material local suele ser clips cortos. Esto coincide con el límite de subida por
# defecto de Streamlit y además impone un límite explícito en el servidor a los clientes
# que usan la API directamente.
MAX_VIDEO_MATERIAL_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_IMAGE_MATERIAL_UPLOAD_BYTES = 20 * 1024 * 1024
MATERIAL_VALIDATION_TIMEOUT_SECONDS = 120

# Se mantienen alineados con los formatos de material local que acepta la CLI (cli.py
# deriva su lista de const.FILE_TYPE_VIDEOS / const.FILE_TYPE_IMAGES) y que clasifica el
# flujo de renderizado (video.py lee const.FILE_TYPE_IMAGES). Aceptar aquí un conjunto
# más reducido rechazaría subidas que el resto del flujo ya admite.
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".flv", ".mkv", ".webm")
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
SUPPORTED_MATERIAL_EXTENSIONS = (
    *SUPPORTED_VIDEO_EXTENSIONS,
    *SUPPORTED_IMAGE_EXTENSIONS,
)

_COPY_CHUNK_BYTES = 1024 * 1024
_INTERNAL_UPLOAD_PREFIX = ".material-upload-"
_WINDOWS_INVALID_FILENAME_CHARS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_FILENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)}
    # Se mantiene la misma lista oficial que bgm.sanitize_upload_filename y las reglas
    # de nombres de descarga de webui/Main.py: Win32 también trata los dígitos en
    # superíndice Latin-1 ¹, ² y ³ como número de dispositivo.
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
#   invertidos, y "photo‮gnp.mp4" parece tener otra extensión en el explorador de
#   archivos y en los logs.
# * U+2028/U+2029 (separadores Unicode de línea/párrafo) también pueden crear saltos de
#   línea visuales en una línea de log.
# Ninguno de estos caracteres se puede teclear en el campo del nombre de archivo, así
# que las subidas normales no se ven afectadas; el criterio coincide con
# bgm.sanitize_upload_filename y con normalize_task_id de app/controllers/base.py.
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
# Pillow informa como MPO de los JPEG de cámara que llevan un índice Multi-Picture
# Format (habitual en fotos de móvil y de doble lente); la imagen principal sigue siendo
# un JPEG normal.
_IMAGE_FORMATS_BY_EXTENSION = {
    ".jpg": frozenset({"JPEG", "MPO"}),
    ".jpeg": frozenset({"JPEG", "MPO"}),
    ".png": frozenset({"PNG"}),
    ".bmp": frozenset({"BMP"}),
}


class MaterialUploadError(ValueError):
    """El material subido no cumple los requisitos de archivo o de medio."""


class MaterialServiceError(RuntimeError):
    """El servidor no pudo preparar, validar o persistir un material subido."""


def uploaded_material_dir(create: bool = True) -> str:
    return utils.storage_dir("local_videos", create=create)


def _remove_staged_file(file_path: str) -> None:
    if not file_path or not os.path.exists(file_path):
        return
    try:
        os.remove(file_path)
    except OSError as exc:
        logger.warning(
            f"no se pudo eliminar el material local temporal: path={file_path}, "
            f"error={str(exc)}"
        )


def _material_kind(filename: str) -> Literal["video", "image"]:
    suffix = Path(filename).suffix.lower()
    if suffix in SUPPORTED_VIDEO_EXTENSIONS:
        return "video"
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"

    supported_formats = ", ".join(
        extension.removeprefix(".") for extension in SUPPORTED_MATERIAL_EXTENSIONS
    )
    raise MaterialUploadError(
        f"formato de material local no admitido; formatos admitidos: {supported_formats}"
    )


def sanitize_material_filename(filename: str) -> str:
    """Devuelve un nombre base seguro para mostrar y rechaza los nombres mal formados o no
    admitidos.
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
        raise MaterialUploadError("nombre de archivo de material local no válido")

    # Se mantiene la misma regla que bgm.sanitize_upload_filename: Windows resuelve el
    # segmento anterior a la extensión como nombre de dispositivo, así que CON.mp4 y
    # LPT1.webm no pueden crearse allí como archivos normales. Aunque el nombre guardado
    # sea un UUID en ambos endpoints, rechazar estos nombres de entrada hace que las dos
    # API de subida se comporten igual en todas las plataformas.
    windows_basename = safe_name.split(".", 1)[0].rstrip(" .").upper()
    if windows_basename in _WINDOWS_RESERVED_FILENAMES:
        raise MaterialUploadError("nombre de archivo de material local no válido")
    _material_kind(safe_name)
    return safe_name


def _validate_image(file_path: str, extension: str) -> None:
    try:
        # Pillow protege contra dimensiones en píxeles excesivamente grandes. Su umbral
        # de aviso también se trata como error de subida, en lugar de rechazar solo en
        # el umbral más alto de DecompressionBombError.
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(file_path) as image:
                image_format = str(image.format or "").upper()
                if image_format not in _IMAGE_FORMATS_BY_EXTENSION[extension]:
                    raise MaterialUploadError(
                        "el contenido de la imagen subida no coincide con su extensión"
                    )
                image.verify()
            # verify() comprueba la estructura pero puede aceptar datos de píxeles JPEG
            # truncados. Se reabre tras verify() (que invalida el decodificador) y se
            # decodifican todos los píxeles antes de publicar la subida para tareas de
            # renderizado posteriores.
            with Image.open(file_path) as image:
                image.load()
    except MaterialUploadError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise MaterialUploadError(
            "el archivo subido debe contener una imagen JPEG, PNG o BMP válida"
        ) from exc


def _validate_video(
    file_path: str, timeout_seconds: int = MATERIAL_VALIDATION_TIMEOUT_SECONDS
) -> None:
    # FFmpeg trata una imagen suelta como un stream de vídeo de un fotograma. Se
    # rechazan las imágenes explícitamente para que renombrar photo.jpg a photo.mp4 no
    # pueda saltarse la clase de medio.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(file_path) as image:
                image.verify()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise MaterialUploadError(
            "el archivo subido debe contener un vídeo, no una imagen"
        ) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        pass
    else:
        raise MaterialUploadError("el archivo subido debe contener un vídeo, no una imagen")

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
                "mov,avi,flv,matroska",
                "-i",
                file_path,
                "-map",
                # La V mayúscula excluye las imágenes adjuntas/portadas de álbum. Una
                # portada dentro de un contenedor de audio no es material de vídeo
                # renderizable.
                "0:V:0",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise MaterialServiceError("se agotó el tiempo de la validación del material con FFmpeg") from exc
    except OSError as exc:
        raise MaterialServiceError(
            "no se pudo ejecutar FFmpeg para validar el material"
        ) from exc
    if utils.ffmpeg_decode_failed(decoded):
        raise MaterialUploadError(
            "el archivo subido debe contener un stream de vídeo que se pueda decodificar por completo"
        )


def _stage_material_upload(
    filename: str, source: BinaryIO
) -> tuple[str, Literal["video", "image"], str, int]:
    safe_name = sanitize_material_filename(filename)
    material_kind = _material_kind(safe_name)
    maximum_bytes = (
        MAX_VIDEO_MATERIAL_UPLOAD_BYTES
        if material_kind == "video"
        else MAX_IMAGE_MATERIAL_UPLOAD_BYTES
    )
    maximum_megabytes = maximum_bytes // (1024 * 1024)

    try:
        target_dir = uploaded_material_dir(create=True)
    except OSError as exc:
        raise MaterialServiceError("no se pudo preparar el almacenamiento del material local") from exc

    temp_path = ""
    total_bytes = 0
    try:
        try:
            source.seek(0)
        except (AttributeError, OSError) as exc:
            raise MaterialUploadError("la subida de material local no permite el acceso aleatorio (seek)") from exc

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
                    raise MaterialUploadError("la subida de material local debe ser binaria")
                total_bytes += len(chunk)
                if total_bytes > maximum_bytes:
                    raise MaterialUploadError(
                        f"el material de {material_kind} supera el límite de "
                        f"{maximum_megabytes} MB"
                    )
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        if total_bytes == 0:
            raise MaterialUploadError("el archivo de material local está vacío")
        return safe_name, material_kind, temp_path, total_bytes
    except Exception as exc:
        _remove_staged_file(temp_path)
        if isinstance(exc, MaterialUploadError):
            raise
        if isinstance(exc, OSError):
            raise MaterialServiceError("no se pudo preparar la subida de material local") from exc
        raise
    finally:
        try:
            source.seek(0)
        except (AttributeError, OSError):
            pass


def save_material_upload(filename: str, source: BinaryIO) -> str:
    """Valida y persiste de forma atómica un vídeo o imagen local subido."""
    safe_name, material_kind, temp_path, total_bytes = _stage_material_upload(
        filename, source
    )
    extension = Path(safe_name).suffix.lower()
    stored_name = f"{uuid4().hex}{extension}"
    target_path = os.path.join(os.path.dirname(temp_path), stored_name)

    try:
        if material_kind == "video":
            _validate_video(temp_path)
        else:
            _validate_image(temp_path, extension)

        try:
            os.replace(temp_path, target_path)
        except OSError as exc:
            raise MaterialServiceError(
                "no se pudo guardar la subida de material local"
            ) from exc
        temp_path = ""
        logger.info(
            f"material local subido: original_name={safe_name}, "
            f"stored_name={stored_name}, kind={material_kind}, "
            f"size={total_bytes} bytes"
        )
        return stored_name
    finally:
        _remove_staged_file(temp_path)
