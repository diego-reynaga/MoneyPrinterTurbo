import secrets
from typing import Annotated
from uuid import uuid4

from fastapi import Header, Request

from app.config import config
from app.models.exception import HttpException

MAX_TASK_ID_LENGTH = 128


def normalize_task_id(value: object) -> str:
    """Devuelve un ID de solicitud seguro para los logs, sustituyendo la entrada no válida
    del cliente por un UUID.
    """
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_TASK_ID_LENGTH
        or not value.isprintable()
    ):
        return str(uuid4())
    return value


def get_task_id(request: Request) -> str:
    return normalize_task_id(request.headers.get("x-task-id"))


def get_api_key(request: Request):
    api_key = request.headers.get("x-api-key")
    return api_key


def get_api_key_values(request: Request) -> list[str]:
    """Devuelve todas las cabeceras de clave API de la solicitud, conservando los valores
    repetidos para la validación de seguridad.
    """

    # Los Headers de Starlette ofrecen getlist(), que permite distinguir cabeceras
    # repetidas enviadas por un proxy o por el cliente. El sustituto ligero de Request
    # de los tests unitarios solo usa un dict normal, así que se mantiene un respaldo
    # compatible.
    get_list = getattr(request.headers, "getlist", None)
    if callable(get_list):
        return [value for value in get_list("x-api-key") if isinstance(value, str)]

    api_key = get_api_key(request)
    return [api_key] if isinstance(api_key, str) else []


def verify_token(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="x-api-key")] = None,
):
    """Decide según la configuración si se valida la clave API.

    Una clave vacía mantiene el modo local actual sin autenticación; cuando el
    administrador configura explícitamente una clave no vacía, tanto las rutas de la API
    como las descargas de artefactos de tareas exigen que el cliente envíe ese mismo
    valor en la cabecera ``x-api-key``. La declaración del parámetro también hace que
    Swagger muestre la cabecera, lo que facilita depurar en entornos protegidos.
    """

    configured_key = config.app.get("api_key", "")
    if configured_key in (None, ""):
        return None

    # La opción debe ser una cadena. Aquí se rechazan listas, números y otros tipos
    # erróneos, para evitar comportamientos de autenticación difíciles de detectar por
    # conversiones implícitas a cadena; el mensaje de error tampoco incluye la clave
    # real.
    if not isinstance(configured_key, str):
        raise HttpException(
            task_id=get_task_id(request),
            status_code=500,
            message="la autenticación de la API está mal configurada",
        )

    # El parámetro de FastAPI sirve para declarar x-api-key en OpenAPI; la validación
    # real siempre lee la Request, para poder detectar cabeceras del mismo nombre
    # enviadas varias veces. Los clientes normales y los proxys inversos pueden tomar
    # los valores repetidos en distinto orden, así que hay que rechazarlos en lugar de
    # usar implícitamente el primero o el último.
    token_values = get_api_key_values(request)
    if not token_values and isinstance(x_api_key, str):
        token_values = [x_api_key]

    if len(token_values) != 1:
        raise HttpException(
            task_id=get_task_id(request),
            status_code=401,
            message="clave API no válida",
        )

    # compare_digest solo admite ASCII para str. Las cabeceras de la solicitud son
    # entrada no fiable, y un atacante puede enviar caracteres Latin-1 para provocar un
    # TypeError. Codificar todo a bytes UTF-8 mantiene la comparación en tiempo
    # constante y además admite las claves Unicode válidas en TOML.
    token = token_values[0]
    if not secrets.compare_digest(
        token.encode("utf-8"), configured_key.encode("utf-8")
    ):
        raise HttpException(
            task_id=get_task_id(request),
            status_code=401,
            message="clave API no válida",
        )

    return None
