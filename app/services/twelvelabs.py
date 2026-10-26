"""
Integración con TwelveLabs (https://twelvelabs.io): funciones auxiliares opcionales.

Este módulo envuelve dos modelos de TwelveLabs para que MoneyPrinterTurbo aproveche
mejor el material de stock/B-roll que descarga:

  * Marengo (embeddings multimodales de 512 dimensiones): se usa para *reordenar
    semánticamente*
    los términos de búsqueda generados por el LLM respecto al tema del vídeo, para que,
    cuando se
    acabe el presupuesto de la línea de tiempo, el material que entre sea el más
    relacionado con el tema
    (en lugar de lo que el LLM haya puesto primero por casualidad).

  * Pegasus (comprensión de vídeo): se usa para revisar/describir un clip generado a
    partir de
    una URL pública, por ejemplo para comprobar que un clip descargado coincide de
    verdad con
    el guion antes de publicarlo.

La integración es totalmente opcional y no rompe nada:
  * Si no se configura `twelvelabs_api_keys`, todas las funciones públicas de aquí son
    no-op que devuelven su entrada sin cambios (o None), así que el comportamiento por
    defecto es
    idéntico al de una versión sin TwelveLabs.
  * El SDK `twelvelabs` se importa de forma diferida, así que la dependencia solo hace
    falta
    cuando se usa de verdad la función.

Configuración (config.toml, sección [app]):
    twelvelabs_api_keys = ["tlk_xxx"]   # necesario para activarlo
    twelvelabs_rerank_terms = true      # opcional: reordena los términos de búsqueda
    por relevancia
    twelvelabs_marengo_model = "marengo3.0"   # sobrescritura opcional
    twelvelabs_pegasus_model = "pegasus1.5"   # sobrescritura opcional

Configura una clave API de TwelveLabs desde su panel (https://twelvelabs.io) para
activar esta integración opcional.
"""

import math
from contextlib import contextmanager
from functools import lru_cache
from typing import List, Optional

from loguru import logger

from app.config import config
from app.services import material

DEFAULT_MARENGO_MODEL = "marengo3.0"
DEFAULT_PEGASUS_MODEL = "pegasus1.5"
# Pegasus exige max_tokens en [512, 98304]; 512 basta de sobra para una revisión de una
# línea.
_PEGASUS_MIN_MAX_TOKENS = 512


def is_enabled() -> bool:
    """True solo cuando hay al menos una clave API de TwelveLabs configurada."""
    keys = config.app.get("twelvelabs_api_keys")
    return bool(keys)


def _client(httpx_client=None):
    # La importación diferida y la reutilización de claves en rotación imitan a los
    # demás proveedores de material.py (get_api_key rota entre las claves configuradas).
    from twelvelabs import TwelveLabs

    api_key = material.get_api_key("twelvelabs_api_keys")
    return TwelveLabs(api_key=api_key, httpx_client=httpx_client)


@contextmanager
def _managed_client():
    # El SDK no cierra su cliente HTTP al salir del gestor de contexto. Se gestiona el
    # transporte explícitamente, conservando su tiempo de espera y su política de
    # redirecciones por defecto.
    import httpx

    with httpx.Client(timeout=600, follow_redirects=True) as transport:
        yield _client(httpx_client=transport)


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def embed_text(text: str, model: Optional[str] = None) -> Optional[List[float]]:
    """
    Devuelve un embedding de texto Marengo de 512 dimensiones, o None si falla / está
    desactivado.

    Se guarda en caché para que los términos repetidos en una sesión no vuelvan a llamar
    a la API.
    """
    if not is_enabled() or not text or not text.strip():
        return None
    model = model or config.app.get("twelvelabs_marengo_model", DEFAULT_MARENGO_MODEL)
    try:
        # lru_cache solo memoriza los retornos correctos; una excepción lanzada no se
        # guarda en caché, así que un error pasajero de la API nunca envenena la caché.
        return _embed_text_cached(text.strip(), model)
    except Exception as e:  # noqa: BLE001 - never break the pipeline on TL errors
        logger.warning(f"falló embed_text de TwelveLabs; se omite el reordenamiento: {e}")
        return None


@lru_cache(maxsize=512)
def _embed_text_cached(text: str, model: str) -> List[float]:
    with _managed_client() as client:
        resp = client.embed.create(model_name=model, text=text)
        # El SDK renombra la clave de vector 'float' del JSON en bruto a `float_`.
        vector = list(resp.text_embedding.segments[0].float_)
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("el embedding debe contener componentes finitos")
        if not any(vector):
            raise ValueError("el embedding debe tener una norma distinta de cero")
        return vector


def rerank_terms_by_subject(
    video_subject: str,
    search_terms: List[str],
    model: Optional[str] = None,
) -> List[str]:
    """
    Reordena `search_terms` para que los términos semánticamente más relevantes para
    `video_subject` vayan primero (similitud coseno de Marengo).

    Opcional: solo se ejecuta cuando TwelveLabs está activado Y
    `twelvelabs_rerank_terms` es verdadero. Ante cualquier fallo vuelve al orden
    original, así que nunca puede empeorar el flujo.
    """
    if not is_enabled() or not config.app.get("twelvelabs_rerank_terms"):
        return search_terms
    if not video_subject or len(search_terms) < 2:
        return search_terms

    subject_vec = embed_text(video_subject, model)
    if subject_vec is None:
        return search_terms

    scored = []
    for term in search_terms:
        vec = embed_text(term, model)
        if vec is None or len(vec) != len(subject_vec):
            # El coseno requiere vectores del mismo espacio dimensional. zip() truncaría
            # en silencio una respuesta mal formada y produciría una puntuación
            # engañosa, así que se mantiene el respaldo de todo o nada.
            return search_terms
        scored.append((term, _cosine(subject_vec, vec)))

    ranked = [term for term, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
    logger.info(
        f"TwelveLabs Marengo reordenó {len(ranked)} términos de búsqueda por relevancia "
        f"respecto al tema '{video_subject}': {ranked}"
    )
    return ranked


def analyze_clip(
    video_url: str,
    prompt: str = "Describe what happens in this video in one sentence.",
    model: Optional[str] = None,
    max_tokens: int = _PEGASUS_MIN_MAX_TOKENS,
) -> Optional[str]:
    """
    Revisa/describe un clip a partir de una URL pública con Pegasus y devuelve la
    respuesta de texto del modelo (o None si está desactivado / falla).

    Notas (restricciones de la API de TwelveLabs):
      * Pegasus necesita una URL accesible públicamente (o un recurso subido), no una
        ruta local sin más; la ventana analizada debe ser de >= 4 s.
      * max_tokens debe ser >= 512 para este modelo.
    """
    if not is_enabled() or not video_url:
        return None
    model = model or config.app.get("twelvelabs_pegasus_model", DEFAULT_PEGASUS_MODEL)
    try:
        from twelvelabs.types import VideoContext_Url

        with _managed_client() as client:
            resp = client.analyze(
                model_name=model,
                video=VideoContext_Url(url=video_url),
                prompt=prompt,
                max_tokens=max(max_tokens, _PEGASUS_MIN_MAX_TOKENS),
            )
            return resp.data
    except Exception as e:  # noqa: BLE001
        logger.warning(f"falló analyze_clip de TwelveLabs: {e}")
        return None
