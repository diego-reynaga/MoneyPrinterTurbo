"""Implementación de la aplicación - ASGI."""

import math
import os
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import config
from app.controllers import base
from app.models.exception import HttpException
from app.router import root_api_router
from app.utils import utils


@asynccontextmanager
async def application_lifespan(_: FastAPI):
    """Gestiona de forma centralizada la recuperación al arrancar y los logs de cierre del
    proceso de la API.
    """
    logger.info("evento de arranque")

    configured_api_key = config.app.get("api_key", "")
    if configured_api_key in (None, ""):
        logger.warning(
            "la autenticación por clave API está desactivada; mantén la API en una red de confianza"
        )
    elif isinstance(configured_api_key, str):
        # Solo registra el alcance de la protección; no debe mostrar la clave, su
        # longitud ni un resumen, para que las credenciales no lleguen al sistema de
        # logs.
        logger.info("la autenticación por clave API está activada para /api/v1 y /tasks")
    else:
        logger.error(
            "la autenticación por clave API está mal configurada: app.api_key debe ser una cadena"
        )

    # La publicación multiplataforma la ejecuta el pool de hilos del proceso actual y no
    # se recupera tras reiniciar el servicio. Al arrancar, los estados activos de Redis
    # cuyo proceso de ejecución ya se ha perdido se cierran como fallidos, para que las
    # tareas no queden imposibles de borrar para siempre.
    from app.services import task as task_service

    task_service.recover_interrupted_cross_posts()

    # Las entradas de la cola de Redis persisten entre reinicios de la API. En el
    # proceso nuevo no existe ningún callback task_done que las despache, así que se
    # ocupan ya sus plazas de worker.
    from app.controllers.manager.redis_manager import RedisTaskManager
    from app.controllers.v1 import video as video_controller

    if isinstance(video_controller.task_manager, RedisTaskManager):
        video_controller.task_manager.resume_queued_tasks()
    try:
        yield
    finally:
        logger.info("evento de cierre")


def exception_handler(request: Request, e: HttpException):
    return JSONResponse(
        status_code=e.status_code,
        content=utils.get_response(e.status_code, e.data, e.message),
    )


def validation_exception_handler(request: Request, e: RequestValidationError):
    # Las entradas rechazadas pueden contener NaN/Infinity, y los validadores
    # personalizados adjuntan objetos de excepción a ctx. Ninguno de los dos puede
    # emitirlo directamente JSONResponse.
    errors = jsonable_encoder(
        e.errors(),
        custom_encoder={
            float: lambda value: value if math.isfinite(value) else str(value),
            Exception: str,
        },
    )
    # Describe el primer campo rechazado en lugar de afirmar siempre que falta un campo:
    # los valores fuera de rango y el JSON mal formado no son "field required".
    message = "parámetros de la petición no válidos"
    if errors and isinstance(errors[0], dict):
        location = ".".join(str(part) for part in errors[0].get("loc") or ())
        detail = errors[0].get("msg") or message
        message = f"{location}: {detail}" if location else str(detail)
    return JSONResponse(
        status_code=400,
        content=utils.get_response(status=400, data=errors, message=message),
    )


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _normalize_allowed_origin(raw_origin: str) -> str | None:
    """Reduce la opción a la forma de Origin que envía realmente el navegador.

    La cabecera Origin la fija el RFC 6454 como ``scheme://host[:port]``: sin ruta ni
    barra final, con el esquema y el host normalizados a minúsculas, y omitiendo el
    puerto por defecto del protocolo. ``https://frontend.example/``, copiado de la barra
    de direcciones, y ``https://frontend.example:443``, con el puerto por defecto
    explícito, no son iguales carácter a carácter que el ``https://frontend.example``
    que envía realmente el navegador, y la lista blanca deja de funcionar en silencio:
    el frontend solo ve un error de CORS, el servidor solo deja una línea de log
    blocked, y ninguno de los dos apunta a la configuración.

    Devolver ``None`` significa que esa forma no puede coincidir con ningún Origin
    (falta el esquema, el esquema no es http/https, el host está vacío o contiene
    espacios, el literal IPv6 está mal formado, o el puerto no es válido o está fuera de
    rango), y quien llama debe descartarla y avisar. Los fallos de análisis deben
    reducirse a ``None``: esta función se ejecuta al importar el módulo, y dejar escapar
    el ``ValueError`` de ``urlsplit()`` equivale a que toda la API no pueda arrancar.
    """

    if raw_origin == "*":
        return raw_origin

    try:
        parsed = urlsplit(raw_origin)
        scheme = parsed.scheme.lower()
        # El netloc solo se valida al leer hostname / port: urlsplit() lanza la
        # excepción de los literales IPv6 mal formados, y los puertos no numéricos o
        # fuera de 0-65535 la lanzan al leer port.
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None

    # Solo http/https tienen la semántica de Origin del navegador, y ambos tienen un
    # puerto por defecto que se puede omitir.
    if scheme not in _DEFAULT_PORTS or not host:
        return None
    if any(character.isspace() for character in host):
        return None

    # Minúsculas explícitas, sin depender de cómo trate las mayúsculas urlsplit en
    # distintas versiones de Python.
    host = host.lower()
    # Al serializar el Origin, el navegador mantiene los corchetes de los literales
    # IPv6; las formas con el puerto por defecto como ``:443``, ``:0443`` o ``:80``
    # deben reducirse a la forma sin puerto para coincidir con el frontend real.
    if ":" in host:
        host = f"[{host}]"
    if port is None or port == _DEFAULT_PORTS[scheme]:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def parse_cors_allowed_origins(raw_origins: str | None) -> list[str]:
    """Analiza la lista blanca de orígenes de peticiones entre orígenes del navegador.

    CORS solo restringe el JavaScript entre orígenes del navegador; no afecta a curl,
    Postman, n8n ni a los SDK de servidor. Sin configurar devuelve una lista vacía, lo
    que significa que por defecto no se abre el acceso entre orígenes; cuando el usuario
    despliega de verdad un frontend web independiente, lo activa explícitamente con
    ``CORS_ALLOWED_ORIGINS``.

    Cada origen se reduce primero a la forma canónica de la cabecera Origin, así que
    ``https://a.example/``, ``HTTPS://A.Example`` y ``https://a.example:443`` con el
    puerto por defecto explícito dan el mismo resultado, y las formas que no pueden
    coincidir con ningún Origin se descartan con un aviso. Una entrada mal formada solo
    se descarta a sí misma, sin interrumpir el análisis: esta función se llama al
    importar el módulo, y una excepción de análisis impediría arrancar la API.
    """

    if not raw_origins:
        return []

    # Quita los espacios a ambos lados de cada elemento separado por comas e ignora los
    # vacíos, para que el formato habitual de variable de entorno ``https://a.example,
    # https://b.example,`` no produzca orígenes que nunca coinciden.
    origins: list[str] = []
    for candidate in raw_origins.split(","):
        item = candidate.strip()
        if not item:
            continue
        origin = _normalize_allowed_origin(item)
        if origin is None:
            logger.warning(
                f"se ignora un origen CORS configurado que no puede coincidir con la cabecera "
                f"Origin de un navegador: {item!r}"
            )
            continue
        if origin not in origins:
            origins.append(origin)
    return origins


def configure_cors(instance: FastAPI, allowed_origins: list[str]) -> None:
    """Configura CORS según la lista blanca explícita; una lista vacía mantiene la política
    de mismo origen por defecto.
    """

    if not allowed_origins:
        logger.info(
            "el acceso a la API entre orígenes desde el navegador está desactivado; define "
            "CORS_ALLOWED_ORIGINS para permitir orígenes de confianza"
        )
        return

    allow_all_origins = "*" in allowed_origins
    configured_api_key = config.app.get("api_key", "")
    if allow_all_origins and configured_api_key in (None, ""):
        # ``*`` es un modo de compatibilidad que elige el usuario explícitamente, así
        # que no se impide arrancar; pero sin autenticación permitiría que cualquier
        # página leyera y llamara a la API, así que debe dejar un aviso de seguridad
        # localizable.
        logger.warning(
            "CORS permite cualquier origen de navegador mientras la autenticación por clave API está "
            "desactivada; configura app.api_key o restringe CORS_ALLOWED_ORIGINS"
        )

    instance.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        # Starlette refleja cualquier Origin cuando ``*`` y credentials están activos a
        # la vez. El modo comodín no necesita autenticación por cookies, así que se
        # desactiva credentials; los orígenes explícitos mantienen el comportamiento
        # antiguo, para no afectar al modo de solicitudes con credentials de los
        # frontends web independientes existentes.
        allow_credentials=not allow_all_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        # Cuando un frontend HTTPS remoto accede a una API local o de la red local, los
        # navegadores modernos envían además un preflight de Private Network Access.
        # Solo los orígenes de la lista blanca exacta pueden obtener permiso; el modo
        # comodín sigue rechazándolo, para que cualquier sitio web no pueda sondear los
        # servicios de la red privada del usuario.
        allow_private_network=not allow_all_origins,
    )


def is_browser_origin_allowed(
    request: Request, allowed_origins: list[str]
) -> bool:
    """Decide si el origen de una solicitud del navegador es el mismo origen o un origen de
    la lista blanca explícita.
    """

    origin = request.headers.get("origin")
    if not origin:
        # curl, Postman, n8n y los SDK de servidor normalmente no envían Origin. Se
        # mantienen esas solicitudes, para que la corrección de seguridad no cambie por
        # error el contrato de llamada de los clientes de API existentes.
        return True
    if "*" in allowed_origins or origin in allowed_origins:
        return True

    # El navegador también puede enviar Origin en un POST del mismo origen. Solo se
    # compara scheme + authority, ignorando la ruta y los parámetros de consulta; si un
    # despliegue tras un proxy inverso no reenvía correctamente el esquema/host
    # públicos, se puede declarar el origen externo con CORS_ALLOWED_ORIGINS explícito,
    # sin depender de cabeceras reenviadas no fiables.
    request_url = urlsplit(str(request.url))
    request_origin = f"{request_url.scheme}://{request_url.netloc}"
    return origin == request_origin


def configure_browser_access(instance: FastAPI, allowed_origins: list[str]) -> None:
    """Configura a la vez la protección de Origin del servidor y la política de respuestas
    CORS del navegador.
    """

    @instance.middleware("http")
    async def reject_untrusted_browser_origin(request: Request, call_next):
        """Rechaza activamente los orígenes de navegador no fiables, cubriendo las
        solicitudes simples que no necesitan preflight de CORS.
        """

        if not is_browser_origin_allowed(request, allowed_origins):
            origin = request.headers.get("origin", "")
            logger.warning(
                f"se bloquea un origen de navegador no fiable: method={request.method}, "
                f"path={request.url.path}, origin={origin}"
            )
            return JSONResponse(
                status_code=403,
                content=utils.get_response(
                    status=403,
                    message="no se permite la petición de navegador de origen cruzado",
                ),
            )

        return await call_next(request)

    # El middleware CORS se registra el último y queda por fuera de la protección de
    # Origin: los preflight fiables pueden tener éxito directamente y los no fiables los
    # rechaza CORS; las solicitudes reales que no necesitan preflight siguen pasando por
    # la protección 403 de arriba.
    configure_cors(instance, allowed_origins)


def get_application() -> FastAPI:
    """Inicializa la aplicación FastAPI.

    Returns:
       FastAPI: Instancia del objeto de la aplicación.
    """
    instance = FastAPI(
        title=config.project_name,
        description=config.project_description,
        version=config.project_version,
        debug=False,
        lifespan=application_lifespan,
    )
    instance.include_router(root_api_router)
    instance.add_exception_handler(HttpException, exception_handler)
    instance.add_exception_handler(RequestValidationError, validation_exception_handler)
    return instance


app = get_application()


@app.middleware("http")
async def protect_generated_task_files(request: Request, call_next):
    """Protege la ruta estática de artefactos de tareas, para impedir descargas directas
    que se salten la autenticación de la API.

    ``/tasks`` lo monta StaticFiles por separado y no puede reutilizar las dependencias
    de APIRouter, así que el middleware llama al mismo verify_token. La función de
    autenticación deja pasar si no hay api_key configurada; los preflight OPTIONS se
    dejan también al middleware CORS.
    """

    request_path = request.url.path
    is_task_file = request_path == "/tasks" or request_path.startswith("/tasks/")
    if is_task_file and request.method != "OPTIONS":
        try:
            base.verify_token(request)
        except HttpException as exception:
            return exception_handler(request, exception)

    return await call_next(request)


# Por defecto se sigue la política de mismo origen del navegador; solo se abre el acceso
# entre orígenes cuando el usuario configura explícitamente orígenes web de confianza.
cors_allowed_origins = parse_cors_allowed_origins(
    os.getenv("CORS_ALLOWED_ORIGINS", "")
)
configure_browser_access(app, cors_allowed_origins)

task_dir = utils.task_dir()
app.mount("/tasks", StaticFiles(directory=task_dir, html=True), name="")

public_dir = utils.public_dir()
app.mount("/", StaticFiles(directory=public_dir, html=True), name="")
