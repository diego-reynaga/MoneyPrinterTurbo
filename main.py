import uvicorn
from loguru import logger

from app.config import config

if __name__ == "__main__":
    logger.info(
        "start server, docs: http://127.0.0.1:" + str(config.listen_port) + "/docs"
    )
    # La comprobación de FFmpeg se ha movido al flujo de tareas compartido de
    # app/services/task.py, para que las tres rutas (API, CLI y WebUI) queden cubiertas
    # por igual; aquí ya no se comprueba aparte.
    uvicorn.run(
        app="app.asgi:app",
        host=config.listen_host,
        port=config.listen_port,
        reload=config.reload_debug,
        log_level="warning",
    )
