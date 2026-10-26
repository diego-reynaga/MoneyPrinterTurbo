import functools
import os
import re
import threading

from loguru import logger


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
_PROJECT_ROOT_IN_MESSAGE_RE = re.compile(re.escape(PROJECT_ROOT) + r"(?![\w.\-])")
LOG_RECORD_FORMAT = (
    "<green>{time:%Y-%m-%d %H:%M:%S}</> | "
    "<level>{level}</> | "
    '"{file.path}:{line}":<blue> {function}</> '
    "- <level>{message}</>\n"
)
# Al arrancar, el handler de terminal por defecto de Loguru tiene el ID 0. Al recargar
# la WebUI solo se puede sustituir esa salida de terminal básica; no se puede llamar a
# logger.remove() para vaciar todos los handlers, o también se borraría el sink temporal
# que usan las tareas en ejecución para recoger los logs de la WebUI.
_terminal_handler_id: int | None = 0
_terminal_handler_lock = threading.RLock()
LOG_LEVELS = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")
# Solo se sustituye la instantánea inmutable; el hilo de logs no obtiene el bloqueo de
# configuración que tiene la tarea de generación.
_runtime_logging = ("DEBUG", False)


def get_logging_options() -> tuple[str, bool]:
    return _runtime_logging


def set_logging_options(level: str, log_requests: bool = False) -> None:
    """Ajusta al momento los logs de este proceso, sin reconstruir handlers ni interrumpir
    las tareas en ejecución.
    """
    global _runtime_logging
    level = str(level).strip().upper()
    if level not in LOG_LEVELS:
        raise ValueError(f"nivel de registro no admitido: {level}")
    if not isinstance(log_requests, bool):
        raise ValueError("log_requests debe ser un booleano")
    _runtime_logging = (level, log_requests)


def runtime_log_filter(record) -> bool:
    level, log_requests = _runtime_logging
    # El rastreo de solicitudes es un interruptor independiente; incluso con nivel ERROR
    # el usuario puede activar explícitamente el diagnóstico.
    if record["extra"].get("request_trace"):
        return log_requests
    return record["level"].no >= logger.level(level).no


# La WebUI filtra los logs por el ID del hilo de trabajo de la tarea. Los hilos
# auxiliares que la tarea arranca para descargas en paralelo, codificación de clips y
# latidos tienen sus propios ID, y los logs que escriben se descartarían enteros, con lo
# que justo las fases más largas no mostrarían nada en la WebUI. Aquí se registra la
# correspondencia "hilo auxiliar -> hilo que lo lanzó", para que el filtro de logs pueda
# devolver los registros de los hilos auxiliares a su tarea.
_log_scope_roots: dict[int, int] = {}
_log_scope_lock = threading.Lock()


def log_scope_thread_id(thread_id: int | None = None) -> int:
    """Devuelve el ID del hilo raíz del ámbito de logs al que pertenece el hilo; un hilo
    sin vincular es su propio raíz.
    """
    if thread_id is None:
        thread_id = threading.get_ident()
    with _log_scope_lock:
        return _log_scope_roots.get(thread_id, thread_id)


def bind_log_scope(func):
    """
    Envuelve una función que se ejecutará en otro hilo para que sus logs pertenezcan al
    hilo que llama a esta función.

    Debe llamarse en el hilo que envía la tarea: el ámbito se fija al envolver, y los
    hilos que arranquen a su vez los hilos auxiliares también pertenecerán al hilo de
    tarea original. Al terminar se deshace el vínculo, porque el sistema reutiliza los
    ID de hilo y una correspondencia residual contaría los logs de hilos posteriores no
    relacionados en una tarea ya terminada.
    """
    root_thread_id = log_scope_thread_id()

    @functools.wraps(func)
    def run_in_log_scope(*args, **kwargs):
        thread_id = threading.get_ident()
        if thread_id == root_thread_id:
            return func(*args, **kwargs)

        with _log_scope_lock:
            previous_root = _log_scope_roots.get(thread_id)
            _log_scope_roots[thread_id] = root_thread_id
        try:
            return func(*args, **kwargs)
        finally:
            with _log_scope_lock:
                # El pool de hilos reutiliza hilos: se restaura la pertenencia anterior
                # en lugar de borrarla siempre, para que un envoltorio anidado no
                # elimine antes de tiempo el vínculo que el exterior aún usa.
                if previous_root is None:
                    _log_scope_roots.pop(thread_id, None)
                else:
                    _log_scope_roots[thread_id] = previous_root

    return run_in_log_scope


def _project_relative_path(file_path):
    """
    Acorta una ruta absoluta a una ruta relativa al proyecto que empieza por ``./`` y
    usa siempre barras normales.

    En Windows el proyecto puede arrancarse desde una unidad de red mapeada o una unidad
    ``subst``. Entonces la ruta de la pila de llamadas sigue siendo
    ``X:\\MoneyPrinterTurbo\\...``, mientras que ``PROJECT_ROOT`` resuelto con
    ``realpath`` está en ``C:\\...``, y ``os.path.relpath`` lanza directamente
    ``ValueError``. Si la función de formato lanza una excepción, loguru la captura y
    descarta el registro entero, y tanto el terminal como el panel de logs de la WebUI
    se quedan vacíos; por eso aquí hay que devolver la ruta original como respaldo. Lo
    mismo para los archivos fuera del directorio del proyecto: anteponer ``./`` a una
    ruta que retrocede con ``..`` solo da un resultado más difícil de leer.
    """
    try:
        relative_path = os.path.relpath(file_path, PROJECT_ROOT)
    except ValueError:
        return file_path
    if relative_path == os.pardir or relative_path.startswith(os.pardir + os.sep):
        return file_path
    # El relpath de Windows devuelve rutas separadas por barras invertidas, y
    # concatenarlo tal cual daría una salida con separadores mezclados como
    # ``./app\\utils``, distinta de los logs de otras plataformas.
    return f"./{relative_path.replace(os.sep, '/')}"


def format_log_record(record):
    """
    Da el mismo formato a los logs del terminal y de la WebUI.

    Loguru entrega el mismo registro a varios sinks. El primer sink puede haber
    convertido ya la ruta absoluta en relativa al proyecto, así que aquí se admiten
    tanto rutas absolutas como rutas ya formateadas que empiezan por ``./``. El sink de
    la WebUI desactiva los colores, pero la hora, el nivel, la ubicación de la llamada y
    el mensaje coinciden con los del terminal.
    """
    file_path = record["file"].path
    if os.path.isabs(file_path):
        record["file"].path = _project_relative_path(file_path)

    # Los mensajes de log a veces incluyen la ruta absoluta de archivos de tarea.
    # Acortarlas todas a rutas relativas al proyecto evita que la WebUI y el terminal
    # muestren contenidos distintos según el punto de entrada de la inicialización. Solo
    # se sustituye la ruta raíz completa del proyecto; directorios hermanos como
    # `<root>-backup` o `<root>2` solo comparten el prefijo, y sustituir por subcadena
    # los convertiría en rutas erróneas como `.-backup` o `.2`.
    record["message"] = _PROJECT_ROOT_IN_MESSAGE_RE.sub(".", record["message"])
    return LOG_RECORD_FORMAT


def configure_terminal_logger(sink, level: str, colorize: bool = True) -> int:
    """
    Sustituye de forma segura el handler de logs de terminal del proceso, conservando
    los handlers propios de las tareas.

    Streamlit puede volver a ejecutar la inicialización de logs al recargar el código en
    caliente o al invalidar la caché. Aquí solo se elimina con precisión la salida de
    terminal antigua por el ID de handler registrado, así que no se interrumpen los logs
    de la WebUI que están escribiendo las tareas en segundo plano. El bloqueo protege la
    actualización del ID cuando varias sesiones del navegador inicializan a la vez.
    """
    global _terminal_handler_id

    set_logging_options(level, get_logging_options()[1])

    with _terminal_handler_lock:
        if _terminal_handler_id is not None:
            try:
                logger.remove(_terminal_handler_id)
            except ValueError:
                # Puede que un test o una entrada externa ya hayan eliminado ese
                # handler. Se sigue creando una nueva salida de terminal, sin afectar a
                # los demás sinks de logs que siguen siendo válidos.
                pass

        _terminal_handler_id = logger.add(
            sink,
            level=0,
            filter=runtime_log_filter,
            format=format_log_record,
            colorize=colorize,
        )
        return _terminal_handler_id
