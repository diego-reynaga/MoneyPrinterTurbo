"""
Publicación automática diaria en las redes sociales conectadas, sin intervención.

El nicho se configura una sola vez en ``[autopilot]`` de config.toml. A partir de ahí
este servicio planifica cada día un número fijo de huecos a horas aleatorias dentro de
una ventana horaria (por defecto tres vídeos entre las 10:00 y las 22:00 locales), y en
cada hueco genera un vídeo nuevo y lo publica con Upload-Post reutilizando el mismo
flujo de tareas de la WebUI y de la CLI.

Es una capa añadida y desactivada por defecto: no cambia ningún paso de la generación
manual. Con ``autopilot.enabled = false`` nada de este módulo se ejecuta, y el flujo de
trabajo actual sigue intacto.

El plan del día y el resultado de cada hueco se guardan en
``storage/autopilot/state.json``, así que reiniciar el equipo o el proceso no vuelve a
sortear las horas del día en curso ni republica un hueco ya publicado.
"""

from __future__ import annotations

import json
import os
import random
import socket
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable, Sequence

from loguru import logger

from app.config import config
from app.models import const
from app.services import state as sm
from app.utils import utils


STATE_VERSION = 1

SLOT_STATE_PENDING = "pending"
SLOT_STATE_RUNNING = "running"
SLOT_STATE_DONE = "done"
SLOT_STATE_FAILED = "failed"
SLOT_STATE_SKIPPED = "skipped"
# Un hueco en estos estados ya no va a volver a ejecutarse hoy.
_TERMINAL_SLOT_STATES = frozenset({SLOT_STATE_DONE, SLOT_STATE_FAILED, SLOT_STATE_SKIPPED})
# Estados de publicación en los que no se sabe si el vídeo llegó a salir. Un hueco que
# falla con la subida en uno de ellos no se reintenta nunca: republicar un vídeo que
# quizá ya está en la cuenta es peor que perder el hueco del día.
_IN_FLIGHT_CROSS_POST_STATES = frozenset(
    {const.CROSS_POST_STATE_PENDING, const.CROSS_POST_STATE_PROCESSING}
)

DEFAULT_VIDEOS_PER_DAY = 3
DEFAULT_WINDOW_START_HOUR = 10
DEFAULT_WINDOW_END_HOUR = 22
DEFAULT_MIN_GAP_MINUTES = 45
DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_RETRY_DELAY_MINUTES = 15
DEFAULT_POLL_INTERVAL_SECONDS = 30
# El propio servicio de Upload-Post deja de esperar la subida en segundo plano a la hora;
# esperar algo más aquí permite que ese límite se alcance y se registre como el fallo
# concreto que es, en lugar de como un tiempo de espera del planificador.
DEFAULT_UPLOAD_TIMEOUT_SECONDS = 3900
MAX_RECENT_SUBJECTS = 60
# Tolerancia normal de retraso del bucle. Un hueco que se pasa por menos de esto se
# ejecuta igual aunque la recuperación de huecos perdidos esté desactivada, porque no es
# un hueco perdido: es el intervalo de sondeo.
_MISSED_SLOT_TOLERANCE_MINUTES = 5
_MAX_POLL_INTERVAL_SECONDS = 300
_STATE_WRITE_ENCODING = "utf-8"


class AutopilotError(Exception):
    """Error de la publicación automática que quien llama puede mostrar y registrar."""


class AutopilotConfigError(AutopilotError):
    """La configuración de ``[autopilot]`` no permite planificar el día."""


class AutopilotBusyError(AutopilotError):
    """Ya hay otro planificador vivo publicando.

    Tiene su propio tipo porque no siempre es un error. Una tarea programada que
    comprueba cada pocos minutos se encuentra el bloqueo cada vez que una generación
    dura más que su intervalo, y eso es el funcionamiento normal, no un fallo que deba
    marcar la tarea del sistema como fallida.
    """


@dataclass(frozen=True)
class AutopilotSettings:
    """Instantánea validada de la sección ``[autopilot]``."""

    enabled: bool = False
    niche: str = ""
    subjects: tuple[str, ...] = ()
    videos_per_day: int = DEFAULT_VIDEOS_PER_DAY
    window_start_hour: int = DEFAULT_WINDOW_START_HOUR
    window_end_hour: int = DEFAULT_WINDOW_END_HOUR
    min_gap_minutes: int = DEFAULT_MIN_GAP_MINUTES
    reserve_minutes: int = 0
    catch_up: bool = True
    catch_up_grace_minutes: int = 0
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    retry_delay_minutes: int = DEFAULT_RETRY_DELAY_MINUTES
    poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS
    upload_timeout_seconds: int = DEFAULT_UPLOAD_TIMEOUT_SECONDS
    video_language: str = ""
    cli_options: tuple[str, ...] = ()
    state_file: str = ""


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------


def _setting(section, key: str, default):
    value = section.get(key, default)
    return default if value is None else value


def _coerce_bool(value, key: str) -> bool:
    if isinstance(value, bool):
        return value
    # Un booleano escrito como cadena en TOML ("false") sería verdadero al evaluarlo, y
    # en este módulo eso significaría publicar sin que el usuario lo haya pedido.
    raise AutopilotConfigError(
        f"autopilot.{key} debe ser un booleano TOML true o false, se recibió {value!r}"
    )


def _coerce_int(value, key: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AutopilotConfigError(
            f"autopilot.{key} debe ser un entero, se recibió {value!r}"
        )
    if not minimum <= value <= maximum:
        raise AutopilotConfigError(
            f"autopilot.{key} debe estar entre {minimum} y {maximum}, se recibió {value}"
        )
    return value


def _coerce_text_list(value, key: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise AutopilotConfigError(
            f"autopilot.{key} debe ser una lista TOML de cadenas, no una cadena"
        )
    if not isinstance(value, (list, tuple)):
        raise AutopilotConfigError(f"autopilot.{key} debe ser una lista TOML de cadenas")
    items = []
    for item in value:
        if not isinstance(item, str):
            raise AutopilotConfigError(
                f"autopilot.{key} solo acepta cadenas, se recibió {item!r}"
            )
        text = item.strip()
        if text:
            items.append(text)
    return tuple(items)


def load_settings(overrides: dict[str, Any] | None = None) -> AutopilotSettings:
    """
    Lee y valida ``[autopilot]``, aplicando después las sobrescrituras de la línea de
    comandos.

    Se valida todo de una vez y antes de generar nada, para que un ajuste mal escrito se
    vea al arrancar el planificador y no a las once de la noche, cuando ya no quedan
    huecos del día para recuperarlo.
    """
    section = dict(config.autopilot)
    if overrides:
        unknown = set(overrides) - set(vars(AutopilotSettings()))
        if unknown:
            raise AutopilotConfigError(
                f"sobrescrituras de autopilot desconocidas: {', '.join(sorted(unknown))}"
            )
        # Las sobrescrituras pasan por la misma validación que el archivo, para que
        # --videos-per-day 500 o --window-start 24 fallen igual que en config.toml.
        section.update(overrides)
    settings = AutopilotSettings(
        enabled=_coerce_bool(_setting(section, "enabled", False), "enabled"),
        niche=str(_setting(section, "niche", "")).strip(),
        subjects=_coerce_text_list(_setting(section, "subjects", []), "subjects"),
        videos_per_day=_coerce_int(
            _setting(section, "videos_per_day", DEFAULT_VIDEOS_PER_DAY),
            "videos_per_day",
            1,
            48,
        ),
        window_start_hour=_coerce_int(
            _setting(section, "window_start_hour", DEFAULT_WINDOW_START_HOUR),
            "window_start_hour",
            0,
            23,
        ),
        window_end_hour=_coerce_int(
            _setting(section, "window_end_hour", DEFAULT_WINDOW_END_HOUR),
            "window_end_hour",
            1,
            24,
        ),
        min_gap_minutes=_coerce_int(
            _setting(section, "min_gap_minutes", DEFAULT_MIN_GAP_MINUTES),
            "min_gap_minutes",
            0,
            12 * 60,
        ),
        reserve_minutes=_coerce_int(
            _setting(section, "reserve_minutes", 0), "reserve_minutes", 0, 12 * 60
        ),
        catch_up=_coerce_bool(_setting(section, "catch_up", True), "catch_up"),
        catch_up_grace_minutes=_coerce_int(
            _setting(section, "catch_up_grace_minutes", 0),
            "catch_up_grace_minutes",
            0,
            12 * 60,
        ),
        max_attempts=_coerce_int(
            _setting(section, "max_attempts", DEFAULT_MAX_ATTEMPTS), "max_attempts", 1, 10
        ),
        retry_delay_minutes=_coerce_int(
            _setting(section, "retry_delay_minutes", DEFAULT_RETRY_DELAY_MINUTES),
            "retry_delay_minutes",
            0,
            6 * 60,
        ),
        poll_interval_seconds=_coerce_int(
            _setting(section, "poll_interval_seconds", DEFAULT_POLL_INTERVAL_SECONDS),
            "poll_interval_seconds",
            5,
            _MAX_POLL_INTERVAL_SECONDS,
        ),
        upload_timeout_seconds=_coerce_int(
            _setting(section, "upload_timeout_seconds", DEFAULT_UPLOAD_TIMEOUT_SECONDS),
            "upload_timeout_seconds",
            60,
            24 * 3600,
        ),
        video_language=str(_setting(section, "video_language", "")).strip(),
        cli_options=_coerce_text_list(_setting(section, "cli_options", []), "cli_options"),
        state_file=str(_setting(section, "state_file", "")).strip(),
    )

    _validate_window(settings)
    return settings


def _validate_window(settings: AutopilotSettings) -> None:
    if settings.window_end_hour <= settings.window_start_hour:
        raise AutopilotConfigError(
            "autopilot.window_end_hour debe ser mayor que autopilot.window_start_hour "
            f"(se recibió {settings.window_start_hour} y {settings.window_end_hour})"
        )
    window_minutes = (settings.window_end_hour - settings.window_start_hour) * 60
    if settings.reserve_minutes >= window_minutes:
        raise AutopilotConfigError(
            f"autopilot.reserve_minutes ({settings.reserve_minutes}) no deja ninguna hora "
            f"de publicación dentro de una ventana de {window_minutes} minutos"
        )


def describe_window(settings: AutopilotSettings) -> str:
    """Texto corto de la ventana, para logs y para el resumen de estado."""
    end_label = "24:00" if settings.window_end_hour == 24 else f"{settings.window_end_hour:02d}:00"
    return f"{settings.window_start_hour:02d}:00-{end_label}"


def preflight(settings: AutopilotSettings) -> list[str]:
    """
    Enumera lo que impide publicar sin intervención.

    Devolver la lista completa en lugar de fallar en el primer problema permite arreglar
    la configuración de una sola vez, que es justo lo que hace falta en un modo cuyo
    objetivo es no necesitar a nadie delante.
    """
    from app.services import upload_post

    problems: list[str] = []
    if not settings.enabled:
        problems.append(
            "autopilot.enabled es false; ponlo a true para activar la publicación automática"
        )
    if not settings.niche and not settings.subjects:
        problems.append(
            "falta el nicho: define autopilot.niche o una lista autopilot.subjects"
        )

    service = upload_post.upload_post_service
    if not service.enabled:
        problems.append("app.upload_post_enabled es false")
    if not service.api_key:
        problems.append("falta app.upload_post_api_key")
    if not service.username:
        problems.append("falta app.upload_post_username")
    if not list(service.platforms):
        problems.append("app.upload_post_platforms está vacío")

    # Una opción mal escrita en cli_options no se nota hasta que falla el primer hueco,
    # ya entrada la mañana. Validarla aquí la convierte en algo que se ve al configurar.
    invalid_options = invalid_cli_options(settings)
    if invalid_options:
        problems.append(invalid_options)
    return problems


def invalid_cli_options(settings: AutopilotSettings) -> str | None:
    """Comprueba que ``cli_options`` lo acepta el analizador de la CLI de vídeo."""
    if not settings.cli_options:
        return None
    try:
        _load_cli_module().parse_args(
            ["--video-subject", "comprobación", *settings.cli_options]
        )
    except SystemExit:
        return (
            "autopilot.cli_options no lo acepta la CLI de vídeo: "
            + " ".join(settings.cli_options)
        )
    except Exception as exc:
        return f"no se pudo validar autopilot.cli_options: {exc}"
    return None


# ---------------------------------------------------------------------------
# Planificación de las horas del día
# ---------------------------------------------------------------------------


def window_bounds(day: date, settings: AutopilotSettings) -> tuple[datetime, datetime]:
    """Inicio y fin de la ventana de publicación de un día, en hora local."""
    midnight = datetime(day.year, day.month, day.day)
    return (
        midnight + timedelta(hours=settings.window_start_hour),
        midnight + timedelta(hours=settings.window_end_hour),
    )


def plan_slot_times(
    day: date,
    settings: AutopilotSettings,
    rng: random.Random | None = None,
    not_before: datetime | None = None,
) -> list[datetime]:
    """
    Sortea las horas de publicación de un día dentro de la ventana configurada.

    Las horas se reparten de forma uniforme por toda la ventana respetando una
    separación mínima. Para conseguirlo se sortean los huecos dentro de la ventana ya
    descontada la separación y después se vuelve a añadir; así la separación está
    garantizada sin empujar sistemáticamente las horas hacia los extremos, que es lo que
    pasaría sorteando y reintentando hasta que encajen.

    ``not_before`` recorta el principio de la ventana. Planificar un día que ya ha
    empezado tiene que repartir los vídeos por lo que queda de ventana: sortear sobre la
    ventana completa dejaría horas ya pasadas, y publicarlas todas de golpe al arrancar
    es justo lo contrario de repartir las publicaciones por el día.
    """
    if rng is None:
        rng = random.SystemRandom()

    start, end = window_bounds(day, settings)
    if not_before is not None and not_before > start:
        start = not_before.replace(second=0, microsecond=0)
    latest_start = end - timedelta(minutes=settings.reserve_minutes)
    if latest_start < start:
        return []

    span_minutes = int((latest_start - start).total_seconds() // 60)
    count = settings.videos_per_day
    gap = settings.min_gap_minutes

    if count > 1 and (count - 1) * gap > span_minutes:
        # Preferimos publicar el número de vídeos pedido antes que respetar una
        # separación que no cabe en lo que queda de ventana: la cantidad diaria es el
        # objetivo del usuario y la separación solo evita publicar en ráfaga.
        feasible_gap = span_minutes // (count - 1)
        logger.warning(
            f"autopilot.min_gap_minutes ({gap}) no cabe en los {span_minutes} minutos "
            f"disponibles de la ventana {describe_window(settings)} con {count} vídeos al "
            f"día; se usa una separación de {feasible_gap} minutos"
        )
        gap = feasible_gap

    usable_minutes = max(0, span_minutes - (count - 1) * gap)
    offsets = sorted(rng.randint(0, usable_minutes) for _ in range(count))
    return [
        start + timedelta(minutes=offset + index * gap)
        for index, offset in enumerate(offsets)
    ]


# ---------------------------------------------------------------------------
# Estado persistente
# ---------------------------------------------------------------------------


def state_path(settings: AutopilotSettings) -> str:
    if settings.state_file:
        return os.path.abspath(os.path.expanduser(settings.state_file))
    return os.path.join(utils.storage_dir("autopilot", create=True), "state.json")


def empty_state() -> dict:
    return {
        "version": STATE_VERSION,
        "date": "",
        "planned_at": "",
        "subject_cursor": 0,
        "recent_subjects": [],
        "slots": [],
    }


def load_state(path: str) -> dict:
    """
    Carga el estado guardado; un archivo ausente o dañado vuelve al estado vacío.

    Un planificador desatendido no puede quedarse parado porque el archivo de estado se
    haya truncado en un corte de luz. Perder el plan solo hace que se vuelva a sortear
    el día; negarse a arrancar haría perder el día entero.
    """
    try:
        with open(path, encoding=_STATE_WRITE_ENCODING) as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return empty_state()
    except (OSError, ValueError) as exc:
        logger.warning(
            f"no se pudo leer el estado de la publicación automática, se vuelve a planificar: {exc}"
        )
        return empty_state()

    if not isinstance(data, dict) or not isinstance(data.get("slots"), list):
        logger.warning("el estado de la publicación automática no tiene el formato esperado; se vuelve a planificar")
        return empty_state()

    merged = empty_state()
    merged.update(data)
    merged["version"] = STATE_VERSION
    if not isinstance(merged.get("recent_subjects"), list):
        merged["recent_subjects"] = []
    if not isinstance(merged.get("subject_cursor"), int) or isinstance(
        merged.get("subject_cursor"), bool
    ):
        merged["subject_cursor"] = 0
    merged["slots"] = [slot for slot in merged["slots"] if isinstance(slot, dict)]
    return merged


def save_state(path: str, state: dict) -> None:
    """Guarda el estado de forma atómica, para que un corte no deje medio plan escrito."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = ""
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=".autopilot-", suffix=".json.tmp", dir=directory
        )
        with os.fdopen(descriptor, "w", encoding=_STATE_WRITE_ENCODING) as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = ""
    finally:
        if temporary and os.path.exists(temporary):
            os.remove(temporary)


def _new_slot(index: int, at: datetime) -> dict:
    return {
        "index": index,
        "at": at.isoformat(timespec="seconds"),
        "status": SLOT_STATE_PENDING,
        "attempts": 0,
        "subject": "",
        "task_id": "",
        "videos": [],
        "retry_at": "",
        "started_at": "",
        "finished_at": "",
        "cross_post_state": None,
        "cross_post_results": None,
        "error": None,
    }


def plan_day(
    settings: AutopilotSettings,
    state: dict,
    day: date,
    rng: random.Random | None = None,
    force: bool = False,
    now: datetime | None = None,
) -> dict:
    """
    Garantiza que el estado contiene el plan del día indicado.

    Si el día ya está planificado se devuelve tal cual. Volver a sortear las horas de un
    día en curso haría que un reinicio del proceso pudiera publicar dos veces en el
    mismo hueco, o saltarse uno que ya había pasado. Un día sin ningún hueco (porque se
    arrancó cuando la ventana ya había terminado) también cuenta como planificado, para
    que el bucle no vuelva a sortearlo en cada vuelta.
    """
    if not force and state.get("date") == day.isoformat():
        return state

    now = datetime.now() if now is None else now
    # Solo se recorta la ventana cuando se planifica el día que está en curso; para un
    # día futuro la ventana entera sigue disponible.
    not_before = now if now.date() == day else None
    times = plan_slot_times(day, settings, rng=rng, not_before=not_before)
    state["date"] = day.isoformat()
    state["planned_at"] = now.isoformat(timespec="seconds")
    state["slots"] = [_new_slot(index, at) for index, at in enumerate(times, start=1)]
    if times:
        logger.info(
            f"plan de publicación automática del {day.isoformat()} "
            f"({describe_window(settings)}): "
            + ", ".join(at.strftime("%H:%M") for at in times)
        )
    else:
        logger.warning(
            f"no quedan horas de publicación el {day.isoformat()}: la ventana "
            f"{describe_window(settings)} ya ha terminado; se publicará mañana"
        )
    return state


# ---------------------------------------------------------------------------
# Selección de los huecos a ejecutar
# ---------------------------------------------------------------------------


def _parse_slot_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def slot_due_at(slot: dict) -> datetime | None:
    """Hora efectiva del hueco: la del reintento si lo hay y, si no, la planificada."""
    return _parse_slot_time(slot.get("retry_at") or slot.get("at", ""))


def _state_day(state: dict, fallback: datetime) -> date:
    """Día al que pertenece el plan guardado, cayendo al día de ``fallback``."""
    parsed = _parse_slot_time(state.get("date", ""))
    return parsed.date() if parsed is not None else fallback.date()


def _slot_deadline(settings: AutopilotSettings, day: date) -> datetime:
    _, end = window_bounds(day, settings)
    return end + timedelta(minutes=settings.catch_up_grace_minutes)


def expire_slots(settings: AutopilotSettings, state: dict, now: datetime) -> int:
    """
    Cierra los huecos que ya no se pueden publicar dentro de la ventana.

    Esto es lo que evita que el planificador publique a las tres de la mañana los vídeos
    de un día en que el equipo estuvo apagado: fuera de la ventana el hueco se marca
    omitido y el día siguiente se planifica desde cero.
    """
    deadline = _slot_deadline(settings, _state_day(state, now))
    expired = 0
    for slot in state.get("slots", []):
        if slot.get("status") in _TERMINAL_SLOT_STATES:
            continue
        due = slot_due_at(slot)
        if due is None:
            slot["status"] = SLOT_STATE_SKIPPED
            slot["error"] = "el hueco no tiene una hora de publicación válida"
            expired += 1
            continue
        if slot.get("status") == SLOT_STATE_RUNNING:
            # Un hueco quedó en ejecución al morir el proceso. No se puede saber si el
            # vídeo llegó a publicarse, así que no se vuelve a intentar: republicar es
            # peor que perder un hueco.
            slot["status"] = SLOT_STATE_FAILED
            slot["error"] = (
                "el proceso de publicación automática terminó mientras se ejecutaba este hueco"
            )
            expired += 1
            continue
        too_late = now > deadline
        if not settings.catch_up:
            too_late = too_late or now > due + timedelta(
                minutes=_MISSED_SLOT_TOLERANCE_MINUTES
            )
        if too_late:
            slot["status"] = SLOT_STATE_SKIPPED
            slot["error"] = (
                f"se pasó la hora de publicación ({due.strftime('%Y-%m-%d %H:%M')}) "
                f"fuera de la ventana {describe_window(settings)}"
            )
            expired += 1
    return expired


def next_due_slot(state: dict, now: datetime) -> dict | None:
    for slot in state.get("slots", []):
        if slot.get("status") != SLOT_STATE_PENDING:
            continue
        due = slot_due_at(slot)
        if due is not None and due <= now:
            return slot
    return None


def next_wake_time(settings: AutopilotSettings, state: dict, now: datetime) -> datetime:
    """Momento en que el bucle tiene algo que hacer: el próximo hueco o el día siguiente."""
    candidates = [
        due
        for slot in state.get("slots", [])
        if slot.get("status") == SLOT_STATE_PENDING
        and (due := slot_due_at(slot)) is not None
        and due > now
    ]
    tomorrow = datetime(now.year, now.month, now.day) + timedelta(days=1)
    candidates.append(tomorrow)
    return min(candidates)


# ---------------------------------------------------------------------------
# Tema del vídeo
# ---------------------------------------------------------------------------


def resolve_subject(settings: AutopilotSettings, state: dict) -> str:
    """
    Elige el tema del siguiente vídeo.

    Con ``subjects`` configurado se recorre la lista en orden y se guarda el cursor, de
    modo que la rotación continúa de un día para otro en lugar de empezar siempre por el
    primero. Sin lista, el LLM propone un ángulo nuevo del nicho evitando los temas ya
    publicados.
    """
    from app.services import llm

    if settings.subjects:
        cursor = state.get("subject_cursor", 0)
        if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
            cursor = 0
        subject = settings.subjects[cursor % len(settings.subjects)]
        state["subject_cursor"] = (cursor + 1) % len(settings.subjects)
        return subject

    if not settings.niche:
        raise AutopilotConfigError(
            "falta el nicho: define autopilot.niche o una lista autopilot.subjects"
        )

    recent = [item for item in state.get("recent_subjects", []) if isinstance(item, str)]
    return llm.generate_video_subject(
        niche=settings.niche,
        language=settings.video_language,
        recent_subjects=recent,
    )


def remember_subject(state: dict, subject: str) -> None:
    """Recuerda un tema ya usado, para que el LLM no vuelva a proponerlo."""
    recent = [item for item in state.get("recent_subjects", []) if isinstance(item, str)]
    recent = [item for item in recent if item != subject]
    recent.append(subject)
    state["recent_subjects"] = recent[-MAX_RECENT_SUBJECTS:]


# ---------------------------------------------------------------------------
# Ejecución de un hueco
# ---------------------------------------------------------------------------


def _load_cli_module():
    """Carga el módulo de la CLI tal como lo hace el lanzador instalado.

    La publicación automática construye sus parámetros de vídeo con el mismo código que
    la CLI, para que la configuración guardada por la WebUI, los valores por defecto y
    la validación sean exactamente los mismos en los dos modos.
    """
    from app import paths

    if paths.is_source_checkout():
        import cli

        return cli

    from app import cli

    return cli


def build_video_params(settings: AutopilotSettings, subject: str):
    """Construye los ``VideoParams`` de un hueco a partir del tema y la configuración."""
    cli = _load_cli_module()
    argv = ["--video-subject", subject]
    # Cada hueco publica un vídeo. El video_count guardado por la WebUI es un ajuste del
    # flujo manual, y respetarlo aquí multiplicaría sin avisar las publicaciones
    # diarias. Va antes que cli_options para que el usuario pueda sobrescribirlo igual
    # que cualquier otra opción.
    argv += ["--video-count", "1"]
    if settings.video_language:
        argv += ["--video-language", settings.video_language]
    argv += list(settings.cli_options)

    try:
        args = cli.parse_args(argv)
    except SystemExit as exc:
        raise AutopilotConfigError(
            f"autopilot.cli_options no es válido: {' '.join(settings.cli_options)}"
        ) from exc

    params = cli.build_video_params(args)
    cli.prepare_cli_files(params, stop_at="video")
    return params


def await_cross_post(
    task_id: str,
    timeout_seconds: int,
    poll_interval_seconds: float = 5.0,
    stop_event: threading.Event | None = None,
) -> dict:
    """
    Espera a que la publicación multiplataforma de una tarea llegue a un estado final.

    La generación de vídeo devuelve en cuanto el archivo está listo y la subida sigue en
    un hilo en segundo plano. El planificador necesita saber si el vídeo se publicó de
    verdad antes de dar el hueco por hecho, así que consulta el estado de la tarea.
    """
    deadline = time.monotonic() + timeout_seconds
    last_state = None
    while True:
        try:
            task = sm.state.get_task(task_id)
        except Exception as exc:
            # Un fallo pasajero del backend de estado no dice nada sobre la subida. Se
            # sigue consultando hasta el tiempo límite antes de declarar nada.
            logger.warning(
                f"no se pudo consultar el estado de la publicación, task_id: {task_id}, error: {exc}"
            )
            task = None

        if task:
            last_state = task.get("cross_post_state")
            if last_state == const.CROSS_POST_STATE_COMPLETE:
                return {
                    "success": True,
                    "cross_post_state": last_state,
                    "results": task.get("cross_post_results"),
                }
            if last_state == const.CROSS_POST_STATE_FAILED:
                return {
                    "success": False,
                    "cross_post_state": last_state,
                    "results": task.get("cross_post_results"),
                    "error": task.get("cross_post_error") or "falló la publicación multiplataforma",
                }
            if last_state is None:
                return {
                    "success": False,
                    "cross_post_state": None,
                    "error": "la tarea no tiene ninguna publicación multiplataforma en curso",
                }

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {
                "success": False,
                "cross_post_state": last_state,
                "error": (
                    f"la publicación multiplataforma no terminó en {timeout_seconds} segundos; "
                    f"consulta la tarea {task_id}"
                ),
            }
        if stop_event is not None and stop_event.wait(
            min(poll_interval_seconds, remaining)
        ):
            return {
                "success": False,
                "cross_post_state": last_state,
                "error": "se detuvo la publicación automática mientras se esperaba la subida",
            }
        if stop_event is None:
            time.sleep(min(poll_interval_seconds, remaining))


def publish_one_video(
    settings: AutopilotSettings,
    state: dict,
    subject: str,
    stop_event: threading.Event | None = None,
    task_id: str | None = None,
) -> dict:
    """
    Genera un vídeo del tema indicado y lo publica en las redes conectadas.

    Devuelve ``{"success": bool, "task_id": str, "videos": [...], "error": str | None}``.
    Es el único punto donde se toca el flujo de tareas, de modo que se reutiliza tal cual
    la generación de la WebUI y de la CLI.

    ``task_id`` lo pasa quien planifica, que lo ha guardado antes de llamar aquí. Si el
    proceso muere a mitad, el estado ya apunta a la tarea con la que comprobar si el
    vídeo llegó a publicarse.
    """
    from app.services import task as tm
    from app.services import upload_post

    task_id = task_id or utils.get_uuid()
    params = build_video_params(settings, subject)
    outcome: dict[str, Any] = {
        "success": False,
        "task_id": task_id,
        "videos": [],
        "cross_post_state": None,
        "cross_post_results": None,
        "error": None,
    }

    logger.info(f"hueco de publicación automática: task_id={task_id}, tema={subject!r}")
    result = tm.start(
        task_id=task_id,
        params=params,
        stop_at="video",
        # La publicación automática se ejecuta en el mismo equipo que la CLI y con la
        # misma confianza; mantener este valor permite usar material y audio locales.
        allow_server_file_input=True,
    )

    if not isinstance(result, dict) or not result:
        outcome["error"] = "la generación del vídeo devolvió un resultado vacío"
        return outcome
    if result.get("state") == const.TASK_STATE_FAILED:
        outcome["error"] = (
            f"falló la generación del vídeo en la etapa "
            f"{result.get('failed_stage') or 'desconocida'}: "
            f"{result.get('error') or 'error desconocido'}"
        )
        return outcome

    videos = [path for path in (result.get("videos") or []) if path]
    outcome["videos"] = videos
    if not videos:
        outcome["error"] = "la generación del vídeo no produjo ningún archivo"
        return outcome

    cross_post_state = result.get("cross_post_state")
    if cross_post_state is None:
        # Con upload_post_auto_upload desactivado el flujo no encola nada, y ese ajuste
        # debe seguir desactivado para que las generaciones manuales de la WebUI no se
        # publiquen solas. La publicación automática encola aquí su propia subida.
        try:
            scheduling_error = tm.schedule_cross_post_for_task(
                task_id=task_id,
                video_paths=videos,
                params=params,
                video_script=result.get("script") or "",
            )
        except Exception as exc:
            # No se sabe si el trabajo llegó a entrar en la cola. Se declara en vuelo,
            # que es el estado que impide reintentar el hueco: volver a publicar un
            # vídeo que quizá ya salió es peor que perder el hueco.
            logger.exception(f"no se pudo encolar la publicación, task_id: {task_id}, error: {exc}")
            outcome["cross_post_state"] = const.CROSS_POST_STATE_PENDING
            outcome["error"] = f"no se pudo encolar la publicación: {exc}"
            return outcome
        if scheduling_error:
            outcome["error"] = f"no se pudo encolar la publicación: {scheduling_error}"
            return outcome
    elif cross_post_state == const.CROSS_POST_STATE_FAILED:
        outcome["cross_post_state"] = cross_post_state
        outcome["error"] = (
            result.get("cross_post_error") or "no se pudo encolar la publicación"
        )
        return outcome

    try:
        upload = await_cross_post(
            task_id,
            timeout_seconds=settings.upload_timeout_seconds,
            stop_event=stop_event,
        )
    except Exception as exc:
        # La subida sigue en marcha en su hilo; solo ha fallado la espera. Igual que
        # arriba, el hueco queda en vuelo y no se reintenta.
        logger.exception(
            f"no se pudo esperar la publicación, task_id: {task_id}, error: {exc}"
        )
        outcome["cross_post_state"] = const.CROSS_POST_STATE_PROCESSING
        outcome["error"] = f"no se pudo confirmar la publicación: {exc}"
        return outcome

    outcome["cross_post_state"] = upload.get("cross_post_state")
    outcome["cross_post_results"] = upload.get("results")
    if not upload.get("success"):
        outcome["error"] = upload.get("error") or "falló la publicación multiplataforma"
        return outcome

    outcome["success"] = True
    logger.success(
        f"vídeo publicado por la publicación automática: task_id={task_id}, "
        f"plataformas={', '.join(upload_post.upload_post_service.platforms)}"
    )
    return outcome


def run_slot(
    settings: AutopilotSettings,
    state: dict,
    slot: dict,
    persist: Callable[[], None],
    stop_event: threading.Event | None = None,
    publisher: Callable[..., dict] | None = None,
    now_fn: Callable[[], datetime] = datetime.now,
) -> bool:
    """
    Ejecuta un hueco completo: tema, vídeo y publicación.

    El estado se guarda en cada paso. Así, si el proceso muere a mitad, el arranque
    siguiente ve el hueco en ejecución y no lo repite, en lugar de publicar dos veces el
    mismo contenido.
    """
    publisher = publisher or publish_one_video
    now = now_fn()
    slot["attempts"] = int(slot.get("attempts") or 0) + 1
    slot["status"] = SLOT_STATE_RUNNING
    slot["started_at"] = now.isoformat(timespec="seconds")
    slot["error"] = None
    # El resultado de publicación del intento anterior se limpia aquí, para que la
    # decisión de reintentar de más abajo mire solo lo que ha pasado en este intento.
    slot["cross_post_state"] = None
    slot["cross_post_results"] = None
    persist()

    def fail(message: str) -> bool:
        failed_at = now_fn()
        slot["finished_at"] = failed_at.isoformat(timespec="seconds")
        slot["error"] = message
        attempts_left = settings.max_attempts - int(slot.get("attempts") or 0)
        retry_at = failed_at + timedelta(minutes=settings.retry_delay_minutes)
        deadline = _slot_deadline(settings, _state_day(state, failed_at))
        if slot.get("cross_post_state") in _IN_FLIGHT_CROSS_POST_STATES:
            # El fallo es de la espera, no de la subida: el vídeo puede estar ya
            # publicado o publicándose. Reintentar sería publicarlo dos veces, así que
            # el hueco se cierra como fallido y se deja la revisión al usuario.
            slot["status"] = SLOT_STATE_FAILED
            logger.error(
                f"el hueco {slot.get('index')} de la publicación automática no se puede "
                f"reintentar porque su subida quedó sin confirmar "
                f"(task_id: {slot.get('task_id') or 'desconocido'}): {message}"
            )
            persist()
            return False
        if attempts_left > 0 and retry_at <= deadline:
            slot["status"] = SLOT_STATE_PENDING
            slot["retry_at"] = retry_at.isoformat(timespec="seconds")
            logger.warning(
                f"falló el hueco {slot.get('index')} de la publicación automática; se reintenta a las "
                f"{retry_at.strftime('%H:%M')}: {message}"
            )
        else:
            slot["status"] = SLOT_STATE_FAILED
            logger.error(
                f"falló definitivamente el hueco {slot.get('index')} de la publicación automática: {message}"
            )
        persist()
        return False

    try:
        subject = slot.get("subject") or resolve_subject(settings, state)
    except Exception as exc:
        return fail(f"no se pudo elegir el tema: {exc}")

    slot["subject"] = subject
    remember_subject(state, subject)
    # Cada intento usa su propia tarea, y queda anotada antes de generar: así un hueco
    # interrumpido deja en el estado la tarea exacta que hay que revisar.
    task_id = utils.get_uuid()
    slot["task_id"] = task_id
    persist()

    try:
        outcome = publisher(
            settings, state, subject, stop_event=stop_event, task_id=task_id
        )
    except Exception as exc:
        logger.exception(
            f"error inesperado en el hueco {slot.get('index')} de la publicación automática: {exc}"
        )
        return fail(f"{type(exc).__name__}: {exc}")

    slot["task_id"] = outcome.get("task_id") or task_id
    slot["videos"] = list(outcome.get("videos") or [])
    slot["cross_post_state"] = outcome.get("cross_post_state")
    slot["cross_post_results"] = outcome.get("cross_post_results")

    if not outcome.get("success"):
        return fail(outcome.get("error") or "falló la publicación del hueco")

    slot["status"] = SLOT_STATE_DONE
    slot["finished_at"] = now_fn().isoformat(timespec="seconds")
    slot["retry_at"] = ""
    persist()
    logger.success(
        f"hueco {slot.get('index')} de la publicación automática publicado: tema={subject!r}"
    )
    return True


# ---------------------------------------------------------------------------
# Bucle del planificador
# ---------------------------------------------------------------------------


def _slot_counts(state: dict) -> dict[str, int]:
    counts = {
        SLOT_STATE_PENDING: 0,
        SLOT_STATE_RUNNING: 0,
        SLOT_STATE_DONE: 0,
        SLOT_STATE_FAILED: 0,
        SLOT_STATE_SKIPPED: 0,
    }
    for slot in state.get("slots", []):
        status = slot.get("status")
        if status in counts:
            counts[status] += 1
    return counts


def run_due_slots(
    settings: AutopilotSettings,
    path: str | None = None,
    state: dict | None = None,
    now_fn: Callable[[], datetime] = datetime.now,
    stop_event: threading.Event | None = None,
    publisher: Callable[..., dict] | None = None,
    rng: random.Random | None = None,
) -> dict:
    """
    Planifica el día si hace falta y ejecuta todos los huecos que ya han llegado.

    Es el modo de una sola pasada: sirve para una tarea programada del sistema que se
    ejecute cada pocos minutos, además de ser el paso que repite el bucle propio.
    """
    path = path or state_path(settings)
    state = load_state(path) if state is None else state

    def persist() -> None:
        save_state(path, state)

    now = now_fn()
    plan_day(settings, state, now.date(), rng=rng, now=now)
    expire_slots(settings, state, now)
    persist()

    executed = 0
    published = 0
    while not (stop_event is not None and stop_event.is_set()):
        slot = next_due_slot(state, now_fn())
        if slot is None:
            break
        executed += 1
        if run_slot(
            settings,
            state,
            slot,
            persist,
            stop_event=stop_event,
            publisher=publisher,
            now_fn=now_fn,
        ):
            published += 1
        expire_slots(settings, state, now_fn())
        persist()

    return {
        "date": state.get("date", ""),
        "executed": executed,
        "published": published,
        "counts": _slot_counts(state),
        "state_file": path,
    }


def run_forever(
    settings: AutopilotSettings,
    path: str | None = None,
    now_fn: Callable[[], datetime] = datetime.now,
    stop_event: threading.Event | None = None,
    publisher: Callable[..., dict] | None = None,
    rng: random.Random | None = None,
    max_cycles: int | None = None,
) -> dict:
    """
    Bucle del planificador: publica el número de vídeos configurado cada día, sin parar.

    Duerme en tramos cortos en lugar de hasta la hora exacta del siguiente hueco, para
    que una parada se atienda enseguida y para que suspender e hibernar el equipo no
    deje al proceso dormido más allá de su propia ventana.

    ``max_cycles`` solo lo usan las pruebas, para ejecutar un número conocido de vueltas.
    """
    path = path or state_path(settings)
    stop_event = stop_event or threading.Event()
    state = load_state(path)
    logger.info(
        f"publicación automática en marcha: {settings.videos_per_day} vídeos al día "
        f"entre las {describe_window(settings)}, estado en {path}"
    )

    cycles = 0
    summary: dict[str, Any] = {"cycles": 0, "published": 0, "state_file": path}
    while not stop_event.is_set():
        if max_cycles is not None and cycles >= max_cycles:
            break
        cycles += 1
        try:
            result = run_due_slots(
                settings,
                path=path,
                state=state,
                now_fn=now_fn,
                stop_event=stop_event,
                publisher=publisher,
                rng=rng,
            )
            summary["published"] += result["published"]
        except AutopilotConfigError:
            # La configuración no se arregla sola esperando, y seguir el bucle solo
            # llenaría el log del mismo error cada medio minuto.
            raise
        except Exception as exc:
            logger.exception(f"falló un ciclo de la publicación automática: {exc}")

        if stop_event.is_set():
            break
        now = now_fn()
        wake_at = next_wake_time(settings, state, now)
        sleep_seconds = max(0.0, (wake_at - now).total_seconds())
        stop_event.wait(min(sleep_seconds, settings.poll_interval_seconds))

    summary["cycles"] = cycles
    summary["counts"] = _slot_counts(state)
    logger.info(f"publicación automática detenida tras {cycles} ciclos")
    return summary


# ---------------------------------------------------------------------------
# Resumen de estado y bloqueo de instancia única
# ---------------------------------------------------------------------------


def status_report(settings: AutopilotSettings, path: str | None = None) -> dict:
    """Instantánea legible del plan y de lo publicado, sin tocar el estado."""
    path = path or state_path(settings)
    state = load_state(path)
    return {
        "enabled": settings.enabled,
        "niche": settings.niche,
        "subjects": list(settings.subjects),
        "videos_per_day": settings.videos_per_day,
        "window": describe_window(settings),
        "state_file": path,
        "date": state.get("date", ""),
        "planned_at": state.get("planned_at", ""),
        "counts": _slot_counts(state),
        "problems": preflight(settings),
        "slots": [
            {
                "index": slot.get("index"),
                "at": slot.get("at"),
                "retry_at": slot.get("retry_at") or "",
                "status": slot.get("status"),
                "attempts": slot.get("attempts", 0),
                "subject": slot.get("subject", ""),
                "task_id": slot.get("task_id", ""),
                "videos": list(slot.get("videos") or []),
                "error": slot.get("error"),
            }
            for slot in state.get("slots", [])
        ],
    }


def _is_process_alive(process_id: int) -> bool:
    if process_id == os.getpid():
        return True
    if os.name == "nt":
        # Se reutiliza la comprobación Win32 de solo lectura del servicio de tareas:
        # os.kill(pid, 0) en Windows puede terminar el proceso de destino en lugar de
        # limitarse a consultarlo.
        from app.services.task import _is_windows_process_alive

        return _is_windows_process_alive(process_id)
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


class SingleInstanceLock:
    """
    Impide que dos planificadores publiquen el mismo hueco.

    El bloqueo es un archivo junto al estado con el host y el PID de quien lo tiene. Si
    el dueño ya no existe se recupera el bloqueo: un corte de luz no debe dejar la
    publicación automática incapaz de volver a arrancar.
    """

    def __init__(self, path: str):
        self.path = path
        self._acquired = False

    def owner(self) -> dict | None:
        try:
            with open(self.path, encoding=_STATE_WRITE_ENCODING) as handle:
                data = json.load(handle)
        except (FileNotFoundError, OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def acquire(self) -> None:
        owner = self.owner()
        if owner:
            host = owner.get("host")
            process_id = owner.get("pid")
            if (
                host == socket.gethostname()
                and isinstance(process_id, int)
                and process_id != os.getpid()
                and _is_process_alive(process_id)
            ):
                raise AutopilotBusyError(
                    f"ya hay una publicación automática en marcha (pid {process_id}); "
                    f"ciérrala o borra {self.path}"
                )
            if host != socket.gethostname():
                raise AutopilotError(
                    f"el bloqueo de la publicación automática pertenece a {host}; "
                    f"revísalo y borra {self.path} si ese equipo ya no publica"
                )
            logger.warning(
                f"se recupera un bloqueo de publicación automática huérfano: {self.path}"
            )

        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding=_STATE_WRITE_ENCODING) as handle:
            json.dump(
                {
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                },
                handle,
            )
        self._acquired = True

    def release(self) -> None:
        if not self._acquired:
            return
        self._acquired = False
        owner = self.owner()
        if owner and owner.get("pid") != os.getpid():
            # Otro proceso ya se quedó con el bloqueo tras considerarnos huérfanos; su
            # archivo no es nuestro para borrarlo.
            return
        try:
            os.remove(self.path)
        except OSError as exc:
            logger.warning(f"no se pudo borrar el bloqueo de la publicación automática: {exc}")

    def __enter__(self) -> "SingleInstanceLock":
        self.acquire()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.release()


def lock_path(settings: AutopilotSettings) -> str:
    return f"{state_path(settings)}.lock"


def resolve_overrides(
    videos_per_day: int | None = None,
    window_start_hour: int | None = None,
    window_end_hour: int | None = None,
    niche: str | None = None,
    subjects: Sequence[str] | None = None,
    enabled: bool | None = None,
    state_file: str | None = None,
) -> dict[str, Any]:
    """Convierte los parámetros opcionales de la línea de comandos en sobrescrituras."""
    overrides: dict[str, Any] = {}
    if videos_per_day is not None:
        overrides["videos_per_day"] = videos_per_day
    if window_start_hour is not None:
        overrides["window_start_hour"] = window_start_hour
    if window_end_hour is not None:
        overrides["window_end_hour"] = window_end_hour
    if niche is not None:
        overrides["niche"] = niche.strip()
    if subjects is not None:
        overrides["subjects"] = tuple(
            item.strip() for item in subjects if item and item.strip()
        )
    if enabled is not None:
        overrides["enabled"] = enabled
    if state_file is not None:
        overrides["state_file"] = state_file.strip()
    return overrides
