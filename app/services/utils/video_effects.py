import numpy as np
from moviepy import Clip, ColorClip, CompositeVideoClip, vfx
from PIL import Image


# FadeIn
def fadein_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeIn(t)])


# FadeOut
def fadeout_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeOut(t)])


# SlideIn
def slidein_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size

    # El SlideIn integrado de MoviePy no es estable con material a pantalla completa en
    # esta cadena de procesamiento, y puede ocurrir que "la transición se aplica en
    # teoría pero la imagen apenas cambia". Aquí se sustituye por un fondo negro
    # explícito más una animación de desplazamiento, para que la transición sea visible
    # y su comportamiento controlable.
    def position(current_time: float):
        progress = min(max(current_time / max(t, 0.001), 0), 1)

        if side == "left":
            return (-width + width * progress, 0)
        if side == "right":
            return (width - width * progress, 0)
        if side == "top":
            return (0, -height + height * progress)
        if side == "bottom":
            return (0, height - height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# SlideOut
def slideout_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size
    transition_duration = min(t, clip.duration)
    transition_start = max(clip.duration - transition_duration, 0)

    # SlideOut también se cambia a un desplazamiento explícito, para que el clip salga
    # de la imagen de forma estable al final.
    def position(current_time: float):
        if current_time <= transition_start:
            return (0, 0)

        progress = min(
            max((current_time - transition_start) / max(transition_duration, 0.001), 0), 1
        )

        if side == "left":
            return (-width * progress, 0)
        if side == "right":
            return (width * progress, 0)
        if side == "top":
            return (0, -height * progress)
        if side == "bottom":
            return (0, height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# Se mantiene el 20 % de zoom del diseño original, para que incluso los clips cortos de
# unos tres segundos tengan un movimiento Ken Burns claramente visible. La estabilidad
# del zoom la garantiza el muestreo subpíxel centrado de abajo; no se reduce la
# intensidad del efecto para ocultar el parpadeo de codificación del vídeo de origen.
_ZOOM_MAX_SCALE = 1.2


def _zoom_frame(frame: np.ndarray, scale_factor: float) -> np.ndarray:
    """Implementa un zoom estable y sin bordes negros mediante recorte centrado subpíxel.

    No se pueden convertir antes a enteros el ancho y alto del recorte: cuando la escala
    cambia de forma continua, los límites enteros saltan con pasos irregulares y, al
    alternar entre tamaños pares e impares, cambia la fase de muestreo de medio píxel,
    lo que se ve como temblor de la imagen. La transformación EXTENT de Pillow acepta
    límites de coma flotante y hace el muestreo subpíxel sobre un lienzo de salida fijo;
    los límites izquierdo/derecho y superior/inferior siempre son simétricos respecto al
    mismo centro en coma flotante, por eso sirve para un zoom lento y continuo a lo
    largo de todo el vídeo.
    """
    if scale_factor <= 0:
        raise ValueError("scale_factor debe ser mayor que cero")

    # Con zoom 1x se devuelve directamente el fotograma original, para que un remuestreo
    # innecesario no difumine ligeramente el primer fotograma.
    if abs(scale_factor - 1.0) < 1e-9:
        return frame

    height, width = frame.shape[:2]
    crop_width = width / scale_factor
    crop_height = height / scale_factor
    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = left + crop_width
    bottom = top + crop_height

    image = Image.fromarray(frame)
    transformed = image.transform(
        (width, height),
        Image.Transform.EXTENT,
        (left, top, right, bottom),
        # En un zoom continuo de vídeo importa más la coherencia entre fotogramas
        # contiguos. BICUBIC/LANCZOS dan fotogramas más nítidos, pero las texturas de
        # alta frecuencia que cruzan la rejilla de muestreo tienden a producir ringing y
        # parpadeo de brillo; BILINEAR es más suave y cambia un poco de nitidez por una
        # sensación de movimiento más estable.
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(transformed)


def zoomin_transition(clip: Clip, t: float) -> Clip:
    """Amplía suavemente desde la imagen original hasta 1,2x a lo largo de todo el clip."""
    # t se mantiene de momento para conservar la misma firma de llamada que las demás
    # funciones de transición; el zoom debe cubrir todo el clip, porque si terminara
    # pronto la imagen se quedaría quieta de golpe, lo que no conviene para material
    # estático o con poco movimiento.
    _ = t
    duration = max(clip.duration, 0.001)

    def scale_effect(get_frame, current_time: float):
        progress = min(max(current_time / duration, 0), 1)
        scale_factor = 1 + (_ZOOM_MAX_SCALE - 1) * progress
        return _zoom_frame(get_frame(current_time), scale_factor)

    return clip.transform(scale_effect, apply_to=["mask"])


def zoomout_transition(clip: Clip, t: float) -> Clip:
    """Reduce suavemente desde 1,2x hasta la imagen original a lo largo de todo el clip."""
    # Igual que en zoomin_transition, t solo sirve para ser compatible con la interfaz
    # común de llamada de transiciones.
    _ = t
    duration = max(clip.duration, 0.001)

    def scale_effect(get_frame, current_time: float):
        progress = min(max(current_time / duration, 0), 1)
        scale_factor = _ZOOM_MAX_SCALE - (_ZOOM_MAX_SCALE - 1) * progress
        return _zoom_frame(get_frame(current_time), scale_factor)

    return clip.transform(scale_effect, apply_to=["mask"])
