"""Configuración de la aplicación - APIRouter raíz

Define todos los endpoints de la aplicación FastAPI.

Recursos:
    1. https://fastapi.tiangolo.com/tutorial/bigger-applications
"""

from fastapi import APIRouter

from app.controllers import ping
from app.controllers.v1 import llm, video

root_api_router = APIRouter()
root_api_router.include_router(ping.router)

# v1
root_api_router.include_router(video.router)
root_api_router.include_router(llm.router)
