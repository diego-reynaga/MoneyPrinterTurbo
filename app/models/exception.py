import traceback
from typing import Any

from loguru import logger


class HttpException(Exception):
    def __init__(
        self, task_id: str, status_code: int, message: str = "", data: Any = None
    ):
        self.message = message
        self.status_code = status_code
        self.data = data
        # Obtiene la traza de pila de la excepción.
        tb_str = traceback.format_exc().strip()
        if not tb_str or tb_str == "NoneType: None":
            msg = f"HttpException: {status_code}, {task_id}, {message}"
        else:
            msg = f"HttpException: {status_code}, {task_id}, {message}\n{tb_str}"

        # Tanto 400 como 401 son problemas previsibles de entrada del cliente. Sobre
        # todo con la autenticación activada, los escaneos desde internet pueden generar
        # muchas claves no válidas; usar WARNING conserva la información para
        # localizarlos sin ensuciar las alertas de ERROR. Los errores de configuración
        # del servidor y el resto de excepciones siguen siendo ERROR.
        if status_code in (400, 401):
            logger.warning(msg)
        else:
            logger.error(msg)


class FileNotFoundException(Exception):
    pass
