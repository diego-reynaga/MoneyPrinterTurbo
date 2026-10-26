from fastapi import Depends, Request

from app.controllers import base
from app.controllers.v1.base import new_router
from app.models.exception import HttpException
from app.models.schema import (
    VideoScriptRequest,
    VideoScriptResponse,
    VideoSocialMetadataRequest,
    VideoSocialMetadataResponse,
    VideoTermsRequest,
    VideoTermsResponse,
)
from app.services import llm
from app.utils import utils

# Las rutas de LLM y de vídeo comparten la misma regla de autenticación, para no olvidar
# la protección al añadir endpoints. Si api_key está vacía, verify_token deja pasar
# directamente y no cambia la experiencia local por defecto.
router = new_router(dependencies=[Depends(base.verify_token)])


@router.post(
    "/scripts",
    response_model=VideoScriptResponse,
    summary="Crear un guion para el vídeo",
)
def generate_video_script(request: Request, body: VideoScriptRequest):
    video_script = llm.generate_script(
        video_subject=body.video_subject,
        language=body.video_language,
        paragraph_number=body.paragraph_number,
        video_script_prompt=body.video_script_prompt,
        custom_system_prompt=body.custom_system_prompt,
    )
    if not video_script or video_script.startswith("Error: "):
        raise HttpException(
            task_id=base.get_task_id(request), status_code=503,
            message="la generación de guiones no está disponible",
        )
    response = {"video_script": video_script}
    return utils.get_response(200, response)


@router.post(
    "/terms",
    response_model=VideoTermsResponse,
    summary="Generar los términos del vídeo a partir del guion",
)
def generate_video_terms(request: Request, body: VideoTermsRequest):
    video_terms = llm.generate_terms(
        video_subject=body.video_subject,
        video_script=body.video_script,
        amount=body.amount,
        match_script_order=body.match_materials_to_script,
    )
    if not video_terms:
        raise HttpException(
            task_id=base.get_task_id(request), status_code=503,
            message="la generación de términos de búsqueda no está disponible",
        )
    response = {"video_terms": video_terms}
    return utils.get_response(200, response)


@router.post(
    "/social-metadata",
    response_model=VideoSocialMetadataResponse,
    summary="Generar metadatos de publicación en redes sociales",
)
def generate_video_social_metadata(
    request: Request, body: VideoSocialMetadataRequest
):
    metadata = llm.generate_social_metadata(
        video_subject=body.video_subject,
        video_script=body.video_script,
        language=body.language,
        platform=body.platform,
    )
    return utils.get_response(200, metadata)
