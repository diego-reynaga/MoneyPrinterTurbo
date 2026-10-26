import warnings
from enum import Enum
from typing import Any, List, Literal, Optional, Union

import pydantic
from pydantic import BaseModel, ConfigDict, Field

from app.config import config

# Ignora advertencias concretas de Pydantic
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Field name.*shadows an attribute in parent.*",
)


class VideoConcatMode(str, Enum):
    random = "random"
    sequential = "sequential"


class VideoTransitionMode(str, Enum):
    none = None
    shuffle = "Shuffle"
    fade_in = "FadeIn"
    fade_out = "FadeOut"
    slide_in = "SlideIn"
    slide_out = "SlideOut"
    zoom_in = "ZoomIn"
    zoom_out = "ZoomOut"


class VideoAspect(str, Enum):
    landscape = "16:9"
    portrait = "9:16"
    square = "1:1"

    def to_resolution(self):
        if self == VideoAspect.landscape:
            return 1920, 1080
        elif self == VideoAspect.portrait:
            return 1080, 1920
        elif self == VideoAspect.square:
            return 1080, 1080
        raise ValueError(f"relación de aspecto de vídeo no admitida: {self}")


class VideoFitMode(str, Enum):
    """Cómo rellenan el lienzo de salida los clips de origen con otra relación de aspecto."""

    cover = "cover"
    contain = "contain"


class ImageMotion(str, Enum):
    static = "static"
    zoom = "zoom"


SubtitleDisplayMode = Literal["sentence", "word_by_word"]
SubtitleAnimation = Literal["none", "pop_spring"]
_SUBTITLE_DISPLAY_MODES = ("sentence", "word_by_word")
_SUBTITLE_ANIMATIONS = ("none", "pop_spring")


def _get_valid_ui_choice(key: str, allowed_values: tuple[str, ...], default: str) -> str:
    """
    Lee la configuración de enumeración validada de la WebUI, tolerando valores no
    válidos que puedan quedar de usuarios antiguos.

    El cuerpo de la solicitud lo valida estrictamente el Literal de Pydantic, y un error
    de escritura devuelve un error de validación de campo claro; el archivo de
    configuración, en cambio, debe tratarse con tolerancia, para que un error histórico
    de configuración manual no impida arrancar todo el servicio tras actualizar. El
    código de estado HTTP lo decide el manejador común de excepciones de validación de
    la aplicación, así que aquí no se fija ningún valor concreto.
    """
    configured_value = config.ui.get(key, default)
    return configured_value if configured_value in allowed_values else default


_Config = ConfigDict(
    arbitrary_types_allowed=True,
    # Nota: asegúrate de que los nombres de las claves coinciden con los parámetros
    # renombrados de V2 si hace falta
)


@pydantic.dataclasses.dataclass(config=_Config)
class MaterialInfo:
    provider: str = "pexels"
    url: str = ""
    duration: int = 0
    # La búsqueda de material en línea adjunta información de origen pública filtrada,
    # que reutilizan la caché de búsqueda y el registro de la tarea. El material subido
    # localmente no necesita rellenarla; antes de escribir el archivo de la tarea se
    # vuelve a construir con una lista blanca de campos, para que URL firmadas,
    # credenciales o campos ajenos de solicitudes externas no entren en los datos
    # persistentes.
    source_info: Optional[dict[str, Any]] = None


class VideoParams(BaseModel):
    """
    {
      "video_subject": "",
      "video_aspect": "Horizontal 16:9",
      "voice_name": "Mujer-Xiaoxiao",
      "bgm_name": "random",
      "font_name": "STHeitiMedium Heiti-Medio",
      "text_color": "#FFFFFF",
      "font_size": 60,
      "stroke_color": "#000000",
      "stroke_width": 1.5
    }
    """

    video_subject: str
    video_script: str = ""  # Guion usado para generar el vídeo
    video_terms: Optional[str | List[str]] = None  # Palabras clave usadas para generar el vídeo
    video_aspect: Optional[VideoAspect] = VideoAspect.portrait.value
    video_fit_mode: VideoFitMode = VideoFitMode.cover
    image_motion: ImageMotion = ImageMotion.zoom
    video_concat_mode: Optional[VideoConcatMode] = VideoConcatMode.random.value
    video_transition_mode: Optional[VideoTransitionMode] = None
    video_clip_duration: int = Field(default=5, ge=1)
    video_clip_speed: Optional[float] = 1.0
    match_materials_to_script: bool = False
    video_count: int = Field(default=1, ge=1)

    video_source: Optional[str] = "pexels"
    video_materials: Optional[List[MaterialInfo]] = (
        None  # Materiales usados para generar el vídeo
    )

    custom_audio_file: Optional[str] = (
        None  # Ruta de un archivo de audio personalizado; ignora el TTS y aún permite usar subtítulos de Whisper
    )
    video_language: Optional[str] = ""  # detección automática

    voice_name: Optional[str] = ""
    voice_volume: Optional[float] = Field(default=1.0, allow_inf_nan=False)
    voice_rate: Optional[float] = Field(default=1.0, allow_inf_nan=False)
    bgm_type: Optional[str] = "random"
    bgm_file: Optional[str] = ""
    bgm_volume: Optional[float] = Field(default=0.2, allow_inf_nan=False)
    # Prompt compartido por los proveedores de música para el vídeo; las tareas nuevas
    # de la WebUI siempre escriben este campo. Se mantiene el campo propio de Sonilo de
    # abajo por compatibilidad con registros de tareas antiguos y con los parámetros
    # actuales de la CLI.
    video_music_prompt: str = Field(default="", max_length=2000)
    sonilo_bgm_prompt: str = Field(default="", max_length=2000)

    subtitle_enabled: Optional[bool] = True
    subtitle_position: Optional[str] = config.ui.get(
        "subtitle_position", "bottom"
    )  # top, bottom, center, custom, two_thirds_bottom
    subtitle_display_mode: SubtitleDisplayMode = _get_valid_ui_choice(
        "subtitle_display_mode", _SUBTITLE_DISPLAY_MODES, "sentence"
    )
    subtitle_animation: SubtitleAnimation = _get_valid_ui_choice(
        "subtitle_animation", _SUBTITLE_ANIMATIONS, "none"
    )
    custom_position: float = Field(
        default=config.ui.get("custom_position", 70.0), ge=0, le=100, allow_inf_nan=False
    )
    font_name: Optional[str] = "STHeitiMedium.ttc"
    text_fore_color: Optional[str] = "#FFFFFF"
    text_background_color: Union[bool, str] = False
    rounded_subtitle_background: bool = False

    font_size: int = Field(default=60, ge=1)
    stroke_color: Optional[str] = "#000000"
    stroke_width: float = Field(default=1.5, ge=0, allow_inf_nan=False)
    n_threads: Optional[int] = 2
    paragraph_number: int = Field(default=1, ge=1, le=10)
    video_script_prompt: str = Field(default="", max_length=2000)
    custom_system_prompt: str = Field(default="", max_length=8000)


class SubtitleRequest(BaseModel):
    video_script: str
    video_language: Optional[str] = ""
    voice_name: Optional[str] = "es-ES-ElviraNeural-Female"
    voice_volume: Optional[float] = Field(default=1.0, allow_inf_nan=False)
    voice_rate: Optional[float] = Field(default=1.2, allow_inf_nan=False)
    bgm_type: Optional[str] = "random"
    bgm_file: Optional[str] = ""
    bgm_volume: Optional[float] = Field(default=0.2, allow_inf_nan=False)
    subtitle_position: Optional[str] = config.ui.get("subtitle_position", "bottom")
    subtitle_display_mode: SubtitleDisplayMode = _get_valid_ui_choice(
        "subtitle_display_mode", _SUBTITLE_DISPLAY_MODES, "sentence"
    )
    subtitle_animation: SubtitleAnimation = _get_valid_ui_choice(
        "subtitle_animation", _SUBTITLE_ANIMATIONS, "none"
    )
    font_name: Optional[str] = "STHeitiMedium.ttc"
    text_fore_color: Optional[str] = "#FFFFFF"
    text_background_color: Union[bool, str] = False
    rounded_subtitle_background: bool = False
    font_size: int = Field(default=60, ge=1)
    stroke_color: Optional[str] = "#000000"
    stroke_width: float = Field(default=1.5, ge=0, allow_inf_nan=False)
    video_source: Optional[str] = "local"
    subtitle_enabled: Optional[bool] = True


class AudioRequest(BaseModel):
    video_script: str
    video_language: Optional[str] = ""
    voice_name: Optional[str] = "es-ES-ElviraNeural-Female"
    voice_volume: Optional[float] = Field(default=1.0, allow_inf_nan=False)
    voice_rate: Optional[float] = Field(default=1.2, allow_inf_nan=False)
    bgm_type: Optional[str] = "random"
    bgm_file: Optional[str] = ""
    bgm_volume: Optional[float] = Field(default=0.2, allow_inf_nan=False)
    video_source: Optional[str] = "local"


class VideoScriptParams:
    """
    {
      "video_subject": "Un mar de flores en primavera",
      "video_language": "",
      "paragraph_number": 1,
      "video_script_prompt": "",
      "custom_system_prompt": ""
    }
    """

    video_subject: Optional[str] = "Un mar de flores en primavera"
    video_language: Optional[str] = ""
    paragraph_number: int = Field(default=1, ge=1, le=10)
    video_script_prompt: str = Field(default="", max_length=2000)
    custom_system_prompt: str = Field(default="", max_length=8000)


class VideoTermsParams:
    """
    {
      "video_subject": "",
      "video_script": "",
      "amount": 5,
      "match_materials_to_script": false
    }
    """

    video_subject: Optional[str] = "Un mar de flores en primavera"
    video_script: Optional[str] = (
        "Un mar de flores en primavera se despliega ante nuestros ojos como un poema. En la estación en que todo renace, la tierra se viste de colores: jazmines dorados, cerezos rosados, perales blancos y tulipanes vivos…"
    )
    amount: Optional[int] = 5
    match_materials_to_script: bool = False


class VideoSocialMetadataParams:
    """
    {
      "video_subject": "Un día en Shanghái",
      "video_script": "",
      "language": "auto",
      "platform": "tiktok"
    }
    """

    video_subject: Optional[str] = Field(default="Un día en Shanghái", max_length=500)
    video_script: Optional[str] = Field(default="", max_length=8000)
    language: Optional[str] = Field(default="auto", max_length=64)
    platform: Optional[str] = Field(default="tiktok", max_length=64)


class TaskVideoRequest(VideoParams, BaseModel):
    # Limita las solicitudes de recursos de la API sin cambiar los controles explícitos
    # de lotes y de hilos de FFmpeg de la CLI en VideoParams.
    video_count: int = Field(default=1, ge=1, le=5)
    video_clip_duration: int = Field(default=5, ge=1, le=15)
    n_threads: Optional[int] = Field(default=2, ge=1, le=16)


class TaskQueryRequest(BaseModel):
    pass


class VideoScriptRequest(VideoScriptParams, BaseModel):
    pass


class VideoTermsRequest(VideoTermsParams, BaseModel):
    # La generación de términos ordenados reserva una lista de ejemplos proporcional a
    # amount antes de contactar con el modelo. Se rechazan las solicitudes a la API mal
    # formadas o excesivas.
    amount: int = Field(default=5, ge=1, le=50)


class VideoSocialMetadataRequest(VideoSocialMetadataParams, BaseModel):
    pass


# ---------------------------
# ---- MODELOS DE RESPUESTA ----
# ---------------------------
class BaseResponse(BaseModel):
    status: int = 200
    message: Optional[str] = "success"
    data: Any = None


# ---- MODELOS DE DATOS ----
class TaskResponseData(BaseModel):
    task_id: str


class TaskStatusData(BaseModel):
    """Campos estables que garantiza la consulta de tareas; los campos históricos y de
    extensión se siguen pasando tal cual.
    """

    model_config = ConfigDict(extra="allow")

    task_id: str
    state: int
    progress: int = 0
    videos: Optional[List[str]] = None
    combined_videos: Optional[List[str]] = None
    failed_stage: Optional[str] = None
    error: Optional[str] = None
    cross_post_state: Optional[
        Literal["pending", "processing", "complete", "failed"]
    ] = None
    cross_post_results: Optional[List[dict[str, Any]]] = None
    cross_post_error: Optional[str] = None


class TaskListData(BaseModel):
    """Estructura de la lista de tareas paginada."""

    tasks: List[TaskStatusData]
    total: int
    page: int
    page_size: int


class VideoScriptData(BaseModel):
    video_script: str


class VideoTermsData(BaseModel):
    video_terms: List[str]


class VideoSocialMetadataData(BaseModel):
    title: str
    caption: str
    hashtags: List[str]


class FileData(BaseModel):
    name: str
    size: int
    file: str


class BgmRetrieveData(BaseModel):
    files: List[FileData]


class BgmUploadData(BaseModel):
    file: str


class VideoMaterialRetrieveData(BaseModel):
    files: List[FileData]


class VideoMaterialUploadData(BaseModel):
    file: str


# ---- MODELOS DE RESPUESTA ----
class TaskResponse(BaseResponse):
    data: TaskResponseData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "task_id": "6c85c8cc-a77a-42b9-bc30-947815aa0558",
                },
            },
        }
    )


class TaskQueryResponse(BaseResponse):
    """
    La consulta de tareas devuelve el estado de generación y el estado opcional de
    publicación multiplataforma.

    Si la generación falla incluye `failed_stage` y `error`; si tras completarse la
    generación está activada la publicación automática, `cross_post_state` pasa
    sucesivamente por pending, processing, complete o failed.
    """

    data: TaskStatusData

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": 200,
                    "message": "success",
                    "data": {
                        "task_id": "6c85c8cc-a77a-42b9-bc30-947815aa0558",
                        "state": 1,
                        "progress": 100,
                        "videos": ["/tasks/example/final-1.mp4"],
                        "cross_post_state": "complete",
                        "cross_post_results": [{"success": True}],
                    },
                },
                {
                    "status": 200,
                    "message": "success",
                    "data": {
                        "task_id": "6c85c8cc-a77a-42b9-bc30-947815aa0558",
                        "state": -1,
                        "progress": 30,
                        "failed_stage": "audio",
                        "error": "la petición de TTS agotó el tiempo",
                    },
                },
            ],
        }
    )


class TaskListResponse(BaseResponse):
    """La lista de tareas usa un modelo de respuesta propio, para no mezclar la estructura
    documentada con la de la consulta de una sola tarea.
    """

    data: TaskListData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "tasks": [
                        {
                            "task_id": "6c85c8cc-a77a-42b9-bc30-947815aa0558",
                            "state": 4,
                            "progress": 50,
                        }
                    ],
                    "total": 1,
                    "page": 1,
                    "page_size": 10,
                },
            }
        }
    )


class TaskDeletionResponse(BaseResponse):
    data: None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": None,
            },
        }
    )


class VideoScriptResponse(BaseResponse):
    data: VideoScriptData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "video_script": "Un mar de flores en primavera es un hermoso cuadro de la naturaleza. En esta estación la tierra despierta, todo crece y las flores compiten por abrirse, formando un mar multicolor..."
                },
            },
        }
    )


class VideoTermsResponse(BaseResponse):
    data: VideoTermsData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {"video_terms": ["sky", "tree"]},
            },
        }
    )


class VideoSocialMetadataResponse(BaseResponse):
    data: VideoSocialMetadataData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "title": "Un día en Shanghái que no te puedes perder",
                    "caption": "Guarda esta inspiración rápida de Shanghái y síguenos para más ideas de viaje breves.",
                    "hashtags": ["#shorts", "#travel", "#shanghai", "#viral", "#fyp"],
                },
            },
        }
    )


class BgmRetrieveResponse(BaseResponse):
    data: BgmRetrieveData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "files": [
                        {
                            "name": "4fca18fce7344f3aa824777a40d45c8c.mp3",
                            "size": 1891269,
                            "file": "4fca18fce7344f3aa824777a40d45c8c.mp3",
                        }
                    ]
                },
            },
        }
    )


class BgmUploadResponse(BaseResponse):
    data: BgmUploadData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {"file": "4fca18fce7344f3aa824777a40d45c8c.mp3"},
            },
        }
    )


class VideoMaterialRetrieveResponse(BaseResponse):
    data: VideoMaterialRetrieveData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "files": [
                        {
                            "name": "example.mp4",
                            "size": 12345678,
                            "file": "/MoneyPrinterTurbo/resource/videos/example.mp4",
                        }
                    ]
                },
            },
        }
    )


class VideoMaterialUploadResponse(BaseResponse):
    data: VideoMaterialUploadData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "file": "/MoneyPrinterTurbo/resource/videos/example.mp4",
                },
            },
        }
    )
