import copy
import errno
import os
import shutil
import socket
import tempfile
import threading
from contextlib import contextmanager

import toml
from loguru import logger

from app import __version__, paths

root_dir = str(paths.runtime_root())
config_file = f"{root_dir}/config.toml"
_CONTAINER_CGROUP_MARKERS = ("docker", "containerd", "kubepods", "libpod", "podman")
_DOCKER_HOST_GATEWAY_NAME = "host.docker.internal"
_config_save_lock = threading.RLock()
_diagnostics_update_lock = threading.Lock()
_pending_config_lock = threading.RLock()
_pending_config_updates = {}
_pending_config_save_requested = False
_pending_config_flush_scheduled = False
_MISSING = object()
_DELETE = object()
_UTF8_BOM = "\ufeff"


class _SynchronizedConfig(dict):
    """Mantiene el uso como dict, haciendo a la vez que las escrituras de configuración en
    tiempo de ejecución respeten el mismo bloqueo.
    """

    def __setitem__(self, key, value):
        # En cada rerun completo de la página, Streamlit vuelve a escribir en la
        # configuración los valores actuales de los controles. Cuando una tarea de vídeo
        # tiene runtime_config_lock y el valor no ha cambiado, esa escritura no tiene
        # ningún efecto, y no debe dejar la página refrescada atascada a mitad del
        # formulario. Las escrituras que cambian de verdad la configuración siguen
        # pasando por el bloqueo de abajo, así que no se puede cambiar de proveedor,
        # clave u otro ajuste global a mitad de un vídeo en generación.
        current = super().get(key, _MISSING)
        if current is not _MISSING and current == value:
            return
        with _config_save_lock:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        with _config_save_lock:
            super().__delitem__(key)

    def clear(self):
        if not self:
            return
        with _config_save_lock:
            super().clear()

    def pop(self, key, default=_MISSING):
        # ``pop(key, default)`` tampoco cambia la configuración cuando la clave no
        # existe. La WebUI usa esta forma para expresar "usar la estrategia por
        # defecto", y al refrescar debe permitirse que termine directamente.
        if key not in self:
            if default is _MISSING:
                raise KeyError(key)
            return default
        with _config_save_lock:
            if default is _MISSING:
                return super().pop(key)
            return super().pop(key, default)

    def setdefault(self, key, default=None):
        # Igual que __setitem__, setdefault sobre una clave existente es de solo
        # lectura. Volver antes permite que los refrescos de página que solo leen la
        # configuración por defecto no se vean afectados por el bloqueo de configuración
        # de las tareas largas.
        current = super().get(key, _MISSING)
        if current is not _MISSING:
            return current
        with _config_save_lock:
            return super().setdefault(key, default)

    def update(self, *args, **kwargs):
        changes = dict(*args, **kwargs)
        if all(
            (current := dict.get(self, key, _MISSING)) is not _MISSING
            and current == value
            for key, value in changes.items()
        ):
            return
        with _config_save_lock:
            super().update(changes)


def _pending_update_key(config_section, key):
    """Genera la clave pendiente de actualizar para una sección de configuración fija del
    proceso.
    """
    return id(config_section), key


def update_config_nonblocking(config_section, key, value):
    """
    Actualiza sin bloquear la configuración en tiempo de ejecución de la WebUI.

    La generación de vídeo tiene ``runtime_config_lock``, para que una misma tarea no
    cambie de proveedor, clave o configuración de voz a mitad de la ejecución. Cuando
    cambia un control de Streamlit no se puede esperar a este bloqueo de tarea larga, o
    el navegador parecería congelado. Si el bloqueo está libre se actualiza de
    inmediato; si está ocupado solo se guarda el último valor de cada opción, y se
    aplican todos juntos cuando la tarea actual libera el bloqueo.

    Devolver True indica que el valor ya tiene efecto, y False que ha entrado en la cola
    de actualizaciones pendientes.
    """
    return update_config_values_nonblocking(config_section, {key: value})


def update_config_values_nonblocking(config_section, changes):
    """Encola juntos los campos relacionados de una misma operación, para que la
    configuración no se separe al aplicarse de forma concurrente.
    """
    # Todas las actualizaciones entran primero en la misma cola y luego se intenta
    # obtener el bloqueo de configuración. Así, cuando varias páginas cambian a la vez
    # la misma opción, el orden en que se escriben en la cola es el orden final, y un
    # hilo anterior no puede borrar por error, tras obtener el bloqueo, el valor que un
    # hilo más reciente ya había encolado.
    with _pending_config_lock:
        for key, value in changes.items():
            _pending_config_updates[_pending_update_key(config_section, key)] = (
                config_section,
                key,
                copy.deepcopy(value),
            )

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        # Quien llama normalmente pide guardar al final de este rerun de Streamlit, pero
        # no se puede depender de que ese paso se ejecute siempre. Por ejemplo, si la
        # página falla a mitad o la actualización coincide con la fase de guardado de
        # salida de la tarea, hace falta un hilo de refresco en segundo plano que
        # garantice que los valores en cola acaben aplicándose.
        _schedule_deferred_config_flush()
        return False

    try:
        _apply_pending_config_updates_locked()
        return all(
            config_section.get(key, _MISSING) == value
            for key, value in changes.items()
        )
    finally:
        _config_save_lock.release()


def update_logging_options(level, log_requests):
    """Publica en serie la instantánea de ejecución y la intención de persistir, sin
    esperar al bloqueo de configuración largo de la tarea de generación.
    """
    from app.utils.logging_utils import get_logging_options, set_logging_options

    with _diagnostics_update_lock:
        set_logging_options(level, log_requests)
        level, log_requests = get_logging_options()
        return update_config_values_nonblocking(
            diagnostics, {"log_level": level, "log_requests": log_requests}
        )


def delete_config_nonblocking(config_section, key):
    """
    Borra sin bloquear una opción de configuración de la WebUI.

    "Usar el valor por defecto" necesita eliminar de verdad la opción, no escribir una
    cadena vacía. Mientras una tarea de vídeo ocupa el bloqueo de configuración, la
    intención de borrado sustituye a las actualizaciones encoladas antes para esa misma
    opción y se ejecuta al terminar la tarea.
    """
    with _pending_config_lock:
        _pending_config_updates[_pending_update_key(config_section, key)] = (
            config_section,
            key,
            _DELETE,
        )

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        _schedule_deferred_config_flush()
        return False

    try:
        _apply_pending_config_updates_locked()
        return key not in config_section
    finally:
        _config_save_lock.release()


def _apply_pending_config_updates_locked():
    """Aplica los valores de configuración más recientes que tiene guardados la WebUI
    mientras se tiene el bloqueo de escritura de configuración.
    """
    with _pending_config_lock:
        updates = list(_pending_config_updates.values())
        _pending_config_updates.clear()
        # Se sigue teniendo el bloqueo de pendientes mientras se aplica la
        # configuración. Así, los hilos que leen la instantánea "valor actual + valor
        # pendiente" solo pueden ver el estado completo de antes o de después de
        # aplicar, y nunca un conjunto de configuración actualizado a medias.
        for config_section, key, value in updates:
            if value is _DELETE:
                config_section.pop(key, None)
            else:
                config_section[key] = value
    return bool(updates)


def snapshot_config_with_pending(config_section):
    """
    Devuelve la instantánea efectiva de una sección de configuración, fusionando las
    actualizaciones de la WebUI aún no aplicadas.

    Mientras una tarea de vídeo tiene el bloqueo no se puede reescribir la configuración
    global, pero el usuario puede ir preparando el siguiente contenido. Cuando una
    solicitud al LLM usa esta instantánea, el proveedor, el modelo y la clave recién
    elegidos en la interfaz participan en la nueva solicitud, sin cambiar la tarea de
    vídeo en ejecución.
    """
    with _pending_config_lock:
        snapshot = dict(config_section)
        section_id = id(config_section)
        for (pending_section_id, key), (_, _, value) in _pending_config_updates.items():
            if pending_section_id != section_id:
                continue
            if value is _DELETE:
                snapshot.pop(key, None)
            else:
                snapshot[key] = copy.deepcopy(value)
    return snapshot


def _flush_pending_config_locked(*, suppress_save_errors):
    """Aplica y guarda toda la configuración pendiente actual mientras se tiene el bloqueo
    de escritura de configuración.
    """
    global _pending_config_save_requested

    updates_applied = _apply_pending_config_updates_locked()
    with _pending_config_lock:
        save_requested = _pending_config_save_requested
        _pending_config_save_requested = False

    if not updates_applied and not save_requested:
        return True

    try:
        save_config()
        return True
    except Exception as exc:
        # La configuración en memoria ya se ha aplicado correctamente, y si falla el
        # guardado solo se conserva la marca de pendiente de guardar. Una tarea de vídeo
        # no debe marcarse como fallida porque el archivo de configuración no se pueda
        # escribir temporalmente; la siguiente interacción con la página volverá a
        # intentar guardar.
        with _pending_config_lock:
            _pending_config_save_requested = True
        if not suppress_save_errors:
            raise
        logger.exception(f"no se pudo guardar la configuración de ejecución diferida: {exc}")
        return False


def _run_deferred_config_flush():
    """Espera a que la tarea larga libere el bloqueo de configuración y vacía de forma
    fiable las actualizaciones acumuladas mientras tanto.
    """
    global _pending_config_flush_scheduled

    while True:
        with _config_save_lock:
            flush_succeeded = _flush_pending_config_locked(
                suppress_save_errors=True
            )

        with _pending_config_lock:
            has_pending_work = bool(
                _pending_config_updates or _pending_config_save_requested
            )
            if not flush_succeeded or not has_pending_work:
                _pending_config_flush_scheduled = False
                return


def _schedule_deferred_config_flush():
    """Garantiza que como mucho un hilo en segundo plano esté esperando a la vez para
    refrescar la configuración.
    """
    global _pending_config_flush_scheduled

    with _pending_config_lock:
        if _pending_config_flush_scheduled:
            return
        _pending_config_flush_scheduled = True

    try:
        threading.Thread(
            target=_run_deferred_config_flush,
            name="mpt-config-flush",
            daemon=True,
        ).start()
    except Exception:
        # Ningún worker es dueño de esta reserva cuando falla la construcción o el
        # arranque. Se dejan intactas las actualizaciones en cola y la solicitud de
        # guardado para que una llamada posterior pueda programar otro volcado en lugar
        # de quedarse esperando en silencio para siempre.
        with _pending_config_lock:
            _pending_config_flush_scheduled = False
        raise


def try_save_config():
    """
    Guarda sin bloquear la configuración de la WebUI; si el bloqueo está ocupado, se
    guarda cuando termine la tarea larga actual.

    La API normal, la CLI y los scripts de mantenimiento pueden seguir llamando a
    ``save_config`` para obtener la semántica de escritura bloqueante original; solo los
    reruns de Streamlit usan esta función, para que la página no deje de responder mucho
    tiempo esperando a una tarea de vídeo.
    """
    global _pending_config_save_requested

    with _pending_config_lock:
        _pending_config_save_requested = True

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        _schedule_deferred_config_flush()
        return False

    try:
        return _flush_pending_config_locked(suppress_save_errors=False)
    finally:
        _config_save_lock.release()


@contextmanager
def runtime_config_lock():
    """
    Impide que otras sesiones de la WebUI reescriban la configuración durante una
    operación completa que depende de la configuración global.

    El proyecto se enlaza por defecto a la dirección de loopback local, y la
    configuración sigue siendo global y de un solo usuario. Este bloqueo ligero protege
    sobre todo operaciones largas como la generación o la vista previa, para que otra
    pestaña no cambie de proveedor o de clave a mitad de la operación.
    """
    with _config_save_lock:
        # Si, cuando la operación corta anterior liberó el bloqueo, el hilo de refresco
        # en segundo plano todavía no se había planificado, la nueva tarea debe aplicar
        # la cola antes de leer configuraciones globales como el proveedor o la clave, y
        # no puede ejecutar todo el flujo con la configuración antigua.
        _flush_pending_config_locked(suppress_save_errors=True)
        try:
            yield
        finally:
            _flush_pending_config_locked(suppress_save_errors=True)


@contextmanager
def try_runtime_config_lock():
    """
    Intenta obtener el bloqueo de configuración en tiempo de ejecución y devuelve de
    inmediato si lo consiguió.

    La vista previa de la WebUI es una operación corta que lanza el usuario, y no debe
    esperar varios minutos mientras una tarea de vídeo en segundo plano tiene el
    bloqueo. Si no obtiene el bloqueo, quien llama puede avisar al usuario de que lo
    reintente más tarde; si lo obtiene, sigue garantizado que durante la vista previa
    ninguna otra sesión cambie el proveedor, la clave ni la configuración del modelo.
    """
    acquired = _config_save_lock.acquire(blocking=False)
    try:
        if acquired:
            _flush_pending_config_locked(suppress_save_errors=True)
        yield acquired
    finally:
        if acquired:
            _flush_pending_config_locked(suppress_save_errors=True)
            _config_save_lock.release()


def is_running_in_container(
    dockerenv_path: str = "/.dockerenv",
    containerenv_path: str = "/run/.containerenv",
    cgroup_path: str = "/proc/1/cgroup",
) -> bool:
    """
    Decide si el proceso actual se ejecuta dentro de un contenedor.

    Esta comprobación se usa sobre todo para elegir la dirección por defecto de Ollama:
    - Al ejecutarse normalmente en el equipo, `localhost` apunta a la propia máquina del
      usuario;
    - Dentro de un contenedor Docker, `localhost` apunta al propio contenedor, y para
      acceder al Ollama del host
      normalmente hay que usar `host.docker.internal`.

    No basta con comprobar si existe `/proc/1/cgroup`, porque un Linux normal también
    tiene ese archivo. Aquí solo se devuelve True cuando se detecta una marca de
    contenedor clara, para no afectar a usuarios de Linux sin Docker. El parámetro se
    mantiene como ruta inyectable, para que los tests unitarios puedan cubrir distintos
    entornos de ejecución.
    """
    if os.path.isfile(dockerenv_path) or os.path.isfile(containerenv_path):
        return True

    try:
        with open(cgroup_path, mode="r", encoding="utf-8") as fp:
            cgroup_content = fp.read().lower()
    except OSError:
        return False

    return any(marker in cgroup_content for marker in _CONTAINER_CGROUP_MARKERS)


def _can_resolve_hostname(hostname: str) -> bool:
    try:
        socket.gethostbyname(hostname)
    except OSError:
        return False
    return True


def _decode_linux_route_gateway(hex_gateway: str) -> str:
    # El Gateway de /proc/net/route está en hexadecimal little-endian; por ejemplo,
    # 010011AC significa 172.17.0.1. Se analiza aparte para que, cuando el Docker nativo
    # de Linux no tenga un registro DNS host.docker.internal, aún se pueda intentar
    # acceder al host a través de la puerta de enlace por defecto del contenedor.
    if len(hex_gateway) != 8:
        raise ValueError("longitud de la puerta de enlace no válida")

    octets = [
        str(int(hex_gateway[index : index + 2], 16)) for index in range(6, -1, -2)
    ]
    return ".".join(octets)


def get_container_default_gateway_ip(route_path: str = "/proc/net/route") -> str:
    """
    Lee la IP de la puerta de enlace por defecto dentro de un contenedor Linux.

    Docker Desktop suele ofrecer `host.docker.internal`, pero el Docker nativo de Linux
    no siempre ofrece ese nombre DNS por defecto. La puerta de enlace por defecto suele
    servir como dirección de respaldo para acceder a los servicios del host; si el
    Ollama del usuario solo escucha en 127.0.0.1, el usuario seguirá teniendo que hacer
    que Ollama escuche en la interfaz de red del host o configurar `ollama_base_url` a
    mano.
    """
    try:
        with open(route_path, mode="r", encoding="utf-8") as fp:
            route_lines = fp.readlines()
    except OSError:
        return ""

    for line in route_lines[1:]:
        fields = line.strip().split()
        if len(fields) < 8:
            continue

        destination = fields[1]
        gateway = fields[2]
        if (
            destination != "00000000"
            or fields[7] != "00000000"
            or gateway == "00000000"
        ):
            continue

        try:
            return _decode_linux_route_gateway(gateway)
        except ValueError:
            logger.warning(f"entrada de ruta de la puerta de enlace del contenedor no válida: {line.strip()}")
            return ""

    return ""


def get_default_ollama_base_url() -> str:
    """
    Devuelve la base_url compatible con OpenAI por defecto de Ollama.

    Cuando el usuario configura explícitamente `ollama_base_url` no se pasa por aquí;
    esto solo trata "el mejor valor por defecto cuando no hay configuración". Dentro de
    un contenedor apunta por defecto al host, y en una ejecución normal en el equipo
    apunta a localhost.
    """
    if not is_running_in_container():
        return "http://localhost:11434/v1"

    if _can_resolve_hostname(_DOCKER_HOST_GATEWAY_NAME):
        return f"http://{_DOCKER_HOST_GATEWAY_NAME}:11434/v1"

    gateway_ip = get_container_default_gateway_ip()
    if gateway_ip:
        logger.info(
            "host.docker.internal no se puede resolver; se usa para Ollama la puerta de enlace "
            f"por defecto del contenedor: {gateway_ip}"
        )
        return f"http://{gateway_ip}:11434/v1"

    logger.warning(
        "no se pudieron resolver host.docker.internal ni la puerta de enlace por defecto del contenedor; "
        "se usa host.docker.internal para Ollama"
    )
    return f"http://{_DOCKER_HOST_GATEWAY_NAME}:11434/v1"


def _load_toml_config(config_path: str):
    """
    Carga el TOML, tolerando los BOM UTF-8 repetidos que pueden escribir los editores de
    Windows.

    ``utf-8-sig`` solo elimina un BOM al principio del archivo. Algunos editores de
    Windows, o procesos de descompresión y guardado, pueden volver a escribir un BOM, de
    modo que un segundo carácter invisible llega al analizador TOML y da error en la
    primera línea. Aquí solo se hace una normalización de solo lectura cuando falla el
    análisis estándar, sin reescribir el archivo original, para no sobrescribir por
    accidente las claves API que ya haya rellenado el usuario.
    """
    try:
        return toml.load(config_path)
    except (toml.TomlDecodeError, UnicodeDecodeError) as exc:
        logger.warning(
            "falló la carga de la configuración; se reintenta con compatibilidad con BOM UTF-8: "
            f"path={config_path}, error={type(exc).__name__}: {exc}"
        )

    try:
        with open(config_path, mode="r", encoding="utf-8-sig") as fp:
            config_content = fp.read()

        normalized_content = config_content.lstrip(_UTF8_BOM)
        removed_bom_count = len(config_content) - len(normalized_content)
        if removed_bom_count:
            logger.warning(
                "se eliminaron caracteres BOM UTF-8 repetidos al cargar la configuración: "
                f"path={config_path}, count={removed_bom_count}"
            )
        return toml.loads(normalized_content)
    except (toml.TomlDecodeError, UnicodeDecodeError) as exc:
        logger.error(
            "el archivo de configuración no es TOML válido tras normalizar el BOM UTF-8: "
            f"path={config_path}, error={type(exc).__name__}: {exc}"
        )
        raise



def _initialize_config_from_example(example_file):
    """Publica una configuración completa del primer arranque sin sustituir a otro
    escritor.
    """
    fd, temp_path = tempfile.mkstemp(
        prefix=".config-init-", suffix=".toml.tmp",
        dir=os.path.dirname(os.path.abspath(config_file)),
    )
    try:
        with os.fdopen(fd, "wb") as destination:
            with open(example_file, "rb") as source:
                shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        try:
            if os.name == "nt":
                # En Windows, rename se niega si el destino existe. En POSIX rename lo
                # sobrescribe, así que allí se usa una publicación exclusiva con enlace
                # duro.
                os.rename(temp_path, config_file)
            else:
                os.link(temp_path, config_file)
        except FileExistsError:
            return False
        except OSError as exc:
            raise OSError(
                exc.errno,
                "no se puede inicializar config.toml de forma segura; copia config.example.toml "
                "a config.toml antes de iniciar la aplicación",
            ) from exc
        return True
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            # Un rename correcto en Windows consumió la ruta temporal.
            pass


def load_config():
    os.makedirs(root_dir, mode=0o700, exist_ok=True)
    # Docker puede crear un directorio vacío cuando se monta con bind un config.toml que
    # falta. Solo se elimina ese marcador vacío; un directorio no vacío puede contener
    # datos del usuario.
    if os.path.isdir(config_file):
        try:
            os.rmdir(config_file)
        except FileNotFoundError:
            # Otro inicializador ya eliminó este stub vacío del bind mount.
            pass
        except OSError as exc:
            # Puede que otro inicializador ya haya sustituido el stub vacío de Docker
            # por su archivo normal completo. Solo se acepta esa carrera, no los fallos
            # de permisos ni un directorio que siga presente o no esté vacío.
            if not (
                isinstance(exc, NotADirectoryError) and os.path.isfile(config_file)
            ):
                raise IsADirectoryError(
                    f"{config_file} es un directorio y no se puede usar como archivo de configuración; "
                    "muévelo o renómbralo y vuelve a arrancar"
                ) from exc

    if not os.path.isfile(config_file):
        example_file = f"{root_dir}/config.example.toml"
        if not os.path.isfile(example_file):
            example_file = str(paths.config_template())
        if os.path.isfile(example_file):
            if _initialize_config_from_example(example_file):
                logger.info("se copia config.example.toml a config.toml")

    logger.info(f"se carga la configuración del archivo: {config_file}")

    return _load_toml_config(config_file)


def save_config():
    """
    Guarda de forma atómica la configuración en tiempo de ejecución.

    Distintas sesiones de Streamlit pueden lanzar guardados de configuración casi a la
    vez. Si se sobrescribiera config.toml directamente, otro hilo podría leer un TOML
    escrito solo en parte. Aquí se serializa el guardado con un bloqueo reentrante del
    proceso, y se escribe primero en un temporal del mismo directorio para sustituir
    después el archivo de destino de forma atómica con os.replace.

    Un bind mount de un solo archivo de Docker Desktop usa el propio config.toml como
    punto de montaje, y el kernel de Linux no permite sustituir un punto de montaje con
    rename/replace, así que devuelve EBUSY. En ese caso solo se puede sobrescribir el
    archivo en el sitio dentro del bloqueo; las demás excepciones se siguen lanzando,
    para no ocultar errores de permisos, de disco o de ruta.

    Esto mantiene la semántica actual del proyecto de configuración global de un solo
    usuario, sin añadir un sistema de configuración multiusuario complejo; sirve sobre
    todo para que varias pestañas o reruns rápidos no dañen el archivo de configuración.
    """
    with _config_save_lock:
        config_to_save = dict(_cfg)
        config_to_save["app"] = dict(app)
        config_to_save["azure"] = dict(azure)
        config_to_save["siliconflow"] = dict(siliconflow)
        config_to_save["minimax_tts"] = dict(minimax_tts)
        config_to_save["elevenlabs"] = dict(elevenlabs)
        config_to_save["chatterbox"] = dict(chatterbox)
        config_to_save["kokoro"] = dict(kokoro)
        config_to_save["fish_audio"] = dict(fish_audio)
        config_to_save["voxcpm"] = dict(voxcpm)
        config_to_save["ui"] = dict(ui)
        config_to_save["diagnostics"] = dict(diagnostics)
        config_to_save["autopilot"] = dict(autopilot)
        serialized_config = toml.dumps(config_to_save)

        # La WebUI llama al guardado al terminar cada rerun completo. Si el contenido no
        # ha cambiado vuelve directamente, para no hacer una escritura en disco y un
        # fsync cada vez que se pulsa un control normal.
        try:
            with open(config_file, mode="r", encoding="utf-8") as f:
                if f.read() == serialized_config:
                    _cfg.clear()
                    _cfg.update(config_to_save)
                    return
        except (OSError, UnicodeError):
            pass

        temp_path = ""
        try:
            fd, temp_path = tempfile.mkstemp(
                prefix=".config-",
                suffix=".toml.tmp",
                dir=root_dir,
            )
            with os.fdopen(fd, mode="w", encoding="utf-8") as f:
                f.write(serialized_config)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.replace(temp_path, config_file)
            except OSError as exc:
                if exc.errno != errno.EBUSY:
                    raise

                logger.warning(
                    "la sustitución atómica de la configuración no está disponible para el archivo "
                    f"montado; se escribe en el sitio: {config_file}"
                )
                with open(config_file, mode="w", encoding="utf-8") as f:
                    f.write(serialized_config)
                    f.flush()
                    os.fsync(f.fileno())
            _cfg.clear()
            _cfg.update(config_to_save)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


_cfg = load_config()
diagnostics = _SynchronizedConfig(_cfg.get("diagnostics", {}))
app = _SynchronizedConfig(_cfg.get("app", {}))
whisper = _cfg.get("whisper", {})
proxy = _cfg.get("proxy", {})
azure = _SynchronizedConfig(_cfg.get("azure", {}))
siliconflow = _SynchronizedConfig(_cfg.get("siliconflow", {}))
minimax_tts = _SynchronizedConfig(_cfg.get("minimax_tts", {}))
elevenlabs = _SynchronizedConfig(_cfg.get("elevenlabs", {}))
chatterbox = _SynchronizedConfig(_cfg.get("chatterbox", {}))
kokoro = _SynchronizedConfig(_cfg.get("kokoro", {}))
fish_audio = _SynchronizedConfig(_cfg.get("fish_audio", {}))
voxcpm = _SynchronizedConfig(_cfg.get("voxcpm", {}))
ui = _SynchronizedConfig(
    _cfg.get(
        "ui",
        {
            "hide_log": False,
        },
    )
)

# Publicación automática diaria. Vive en su propia sección para que activarla no toque
# ningún ajuste del flujo manual, y para que quien no la use siga teniendo la sección
# vacía y sin efecto.
autopilot = _SynchronizedConfig(_cfg.get("autopilot", {}))

hostname = socket.gethostname()

log_level = _cfg.get("log_level", "DEBUG")
listen_host = _cfg.get("listen_host", "0.0.0.0")
listen_port = _cfg.get("listen_port", 8080)
project_name = _cfg.get("project_name", "MoneyPrinterTurbo")
project_description = _cfg.get(
    "project_description",
    "<a href='https://github.com/harry0703/MoneyPrinterTurbo'>https://github.com/harry0703/MoneyPrinterTurbo</a>",
)
project_version = _cfg.get("project_version", __version__)
reload_debug = False

app["redis_host"] = os.getenv(
    "MPT_APP_REDIS_HOST",
    os.getenv("REDIS_HOST", app.get("redis_host", "localhost")),
)

ffmpeg_path = app.get("ffmpeg_path", "")
if ffmpeg_path and os.path.isfile(ffmpeg_path):
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

logger.info(f"{project_name} v{project_version}")
