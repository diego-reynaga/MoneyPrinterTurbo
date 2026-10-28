"""
Línea de comandos de la publicación automática diaria.

La usan por igual ``python autopilot.py`` desde el código fuente y ``mpt autopilot``
tras instalar con pip, para que las dos formas de arrancar compartan los mismos modos,
los mismos códigos de salida y la misma salida JSON.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from datetime import datetime
from typing import Sequence

from loguru import logger


_MODES = ("run", "once", "plan", "status", "check", "test")

_EPILOG = """
Modos:
  run      Bucle continuo: planifica cada día y publica a las horas sorteadas. Es el
           modo pensado para dejarlo corriendo como servicio.
  once     Una sola pasada: publica los huecos del día que ya han llegado y termina.
           Pensado para una tarea programada del sistema que se ejecute cada pocos
           minutos; así la publicación sobrevive a los reinicios del equipo.
  plan     Muestra el plan de hoy sin generar nada. Con --force vuelve a sortear las
           horas de los huecos que todavía no se han publicado.
  status   Muestra el plan y el resultado de cada hueco, y los problemas de
           configuración pendientes.
  check    Solo comprueba la configuración.
  test     Genera y publica un vídeo ahora mismo, fuera del plan del día, para verificar
           la configuración completa de principio a fin.

Ejemplos:
  Comprobar la configuración:
    {command} check

  Ver a qué horas se publicará hoy:
    {command} plan

  Dejarlo publicando sin intervención:
    {command} run

  Tarea programada cada 5 minutos en lugar de un proceso siempre vivo:
    {command} once

Salida y código de salida:
  Todos los modos imprimen un objeto JSON en stdout; los logs van a stderr. 0 indica
  éxito, 1 un problema de ejecución o de configuración de cuentas, y 2 un error de
  parámetros o de la sección [autopilot] de config.toml. En el modo once el código de
  salida describe solo esa pasada, y encontrarse la pasada anterior todavía generando
  sale con 0, para que una tarea programada no aparezca como fallida mientras publica.
"""


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("el valor debe ser un entero") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"el valor debe ser >= 1, se recibió {parsed}")
    return parsed


def _hour(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("la hora debe ser un entero") from exc
    if not 0 <= parsed <= 24:
        raise argparse.ArgumentTypeError(f"la hora debe estar entre 0 y 24, se recibió {parsed}")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    command = "uv run python autopilot.py" if _is_source_entry() else "mpt autopilot"
    parser = argparse.ArgumentParser(
        prog="autopilot",
        description=(
            "Publica vídeos en las redes conectadas todos los días sin intervención.\n\n"
            "El nicho y la cantidad diaria se configuran en la sección [autopilot] de "
            "config.toml, y las credenciales de publicación en [app] con los ajustes "
            "upload_post_*. Las horas de publicación se sortean cada día dentro de la "
            "ventana configurada."
        ),
        epilog=_EPILOG.format(command=command),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="run",
        choices=_MODES,
        help="qué hacer (por defecto: run)",
        metavar="modo",
    )
    parser.add_argument(
        "--videos-per-day",
        type=_positive_int,
        help="sobrescribe autopilot.videos_per_day",
    )
    parser.add_argument(
        "--window-start",
        type=_hour,
        help="hora de inicio de la ventana de publicación (0-24)",
    )
    parser.add_argument(
        "--window-end",
        type=_hour,
        help="hora de fin de la ventana de publicación (0-24)",
    )
    parser.add_argument("--niche", help="sobrescribe autopilot.niche en esta ejecución")
    parser.add_argument(
        "--subject",
        action="append",
        dest="subjects",
        help="tema fijo que sustituye al nicho; se puede repetir y se rota en orden",
    )
    parser.add_argument("--state-file", help="ruta del archivo de estado del plan diario")
    parser.add_argument(
        "--allow-disabled",
        action="store_true",
        help="ejecuta aunque autopilot.enabled sea false; útil para probar la configuración",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="no interrumpe por los problemas de configuración detectados; solo los registra",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="en el modo plan, vuelve a sortear las horas de los huecos no publicados",
    )
    return parser.parse_args(argv)


def _is_source_entry() -> bool:
    from app import paths

    return paths.is_source_checkout()


def _print(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, default=str))


def _install_stop_handlers(stop_event: threading.Event) -> None:
    """Pide una parada ordenada en Ctrl+C y en el apagado del servicio.

    Una parada a mitad de un hueco no puede matar el proceso sin más: el vídeo puede
    estar subiéndose, y perder el estado sería arriesgarse a republicarlo al arrancar.
    """

    def handle(signal_number, _frame):
        logger.warning(
            f"se recibió la señal {signal_number}; la publicación automática terminará el hueco actual y parará"
        )
        stop_event.set()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        handler = getattr(signal, name, None)
        if handler is None:
            continue
        try:
            signal.signal(handler, handle)
        except (OSError, ValueError):
            # En algunos entornos (hilos secundarios, servicios de Windows) no se puede
            # instalar el manejador. Se sigue adelante: el bucle también para con el
            # evento que active quien llame.
            continue


def run_autopilot_cli(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    import app.config  # noqa: F401 -- inicializa la configuración y los logs del proceso.
    from app.utils.logging_utils import configure_terminal_logger, get_logging_options

    # Igual que en la CLI de vídeo: stdout queda reservado al JSON y los logs van a
    # stderr, para que la salida se pueda encadenar con otras herramientas.
    configure_terminal_logger(sys.stderr, level=get_logging_options()[0], colorize=False)

    from app.services import autopilot

    overrides = autopilot.resolve_overrides(
        videos_per_day=args.videos_per_day,
        window_start_hour=args.window_start,
        window_end_hour=args.window_end,
        niche=args.niche,
        subjects=args.subjects,
        enabled=True if args.allow_disabled else None,
        state_file=args.state_file,
    )

    try:
        settings = autopilot.load_settings(overrides=overrides)
    except autopilot.AutopilotConfigError as exc:
        logger.error(f"configuración de la publicación automática no válida: {exc}")
        _print({"mode": args.mode, "ok": False, "error": str(exc)})
        return 2

    problems = autopilot.preflight(settings)

    if args.mode == "check":
        _print(
            {
                "mode": "check",
                "ok": not problems,
                "problems": problems,
                "videos_per_day": settings.videos_per_day,
                "window": autopilot.describe_window(settings),
                "state_file": autopilot.state_path(settings),
            }
        )
        return 0 if not problems else 1

    if args.mode == "status":
        _print({"mode": "status", "ok": not problems, **autopilot.status_report(settings)})
        return 0

    if args.mode == "plan":
        path = autopilot.state_path(settings)
        state = autopilot.load_state(path)
        today = datetime.now().date()
        if args.force:
            # Solo se vuelven a sortear las horas cuando no hay nada publicado hoy;
            # replanificar por encima de huecos ya publicados duplicaría vídeos.
            published = [
                slot
                for slot in state.get("slots", [])
                if state.get("date") == today.isoformat()
                and slot.get("status") == autopilot.SLOT_STATE_DONE
            ]
            if published:
                logger.error(
                    f"hoy ya hay {len(published)} huecos publicados; no se vuelve a planificar"
                )
                _print(
                    {
                        "mode": "plan",
                        "ok": False,
                        "error": "hoy ya hay huecos publicados; no se vuelve a planificar",
                        **autopilot.status_report(settings),
                    }
                )
                return 1
        autopilot.plan_day(settings, state, today, force=args.force)
        autopilot.save_state(path, state)
        _print({"mode": "plan", "ok": not problems, **autopilot.status_report(settings)})
        return 0

    if problems and not args.skip_checks:
        for problem in problems:
            logger.error(f"la publicación automática no puede ejecutarse: {problem}")
        _print({"mode": args.mode, "ok": False, "problems": problems})
        return 1
    for problem in problems:
        logger.warning(f"problema de configuración ignorado por --skip-checks: {problem}")

    stop_event = threading.Event()
    _install_stop_handlers(stop_event)

    try:
        with autopilot.SingleInstanceLock(autopilot.lock_path(settings)):
            if args.mode == "test":
                state_file = autopilot.state_path(settings)
                state = autopilot.load_state(state_file)
                subject = autopilot.resolve_subject(settings, state)
                outcome = autopilot.publish_one_video(
                    settings, state, subject, stop_event=stop_event
                )
                # Una prueba manual no gasta ningún hueco del plan del día, pero sí
                # cuenta como tema ya publicado: pedirlo otra vez por la tarde daría un
                # vídeo repetido.
                autopilot.remember_subject(state, subject)
                autopilot.save_state(state_file, state)
                _print({"mode": "test", "ok": bool(outcome.get("success")), **outcome})
                return 0 if outcome.get("success") else 1

            if args.mode == "once":
                summary = autopilot.run_due_slots(settings, stop_event=stop_event)
                # El código de salida describe solo esta pasada. Contar los fallos del
                # día entero haría que una tarea programada cada cinco minutos se
                # marcara como fallida el resto de la jornada por un hueco que falló a
                # media mañana, y el estado del día ya viaja en "counts".
                failed_now = summary["executed"] - summary["published"]
                _print({"mode": "once", "ok": not failed_now, "failed_now": failed_now, **summary})
                return 1 if failed_now else 0

            summary = autopilot.run_forever(settings, stop_event=stop_event)
            _print({"mode": "run", "ok": True, **summary})
            return 0
    except autopilot.AutopilotConfigError as exc:
        logger.error(f"configuración de la publicación automática no válida: {exc}")
        _print({"mode": args.mode, "ok": False, "error": str(exc)})
        return 2
    except autopilot.AutopilotBusyError as exc:
        if args.mode == "once":
            # Encontrarse la pasada anterior todavía generando es lo normal cuando la
            # tarea programada comprueba cada pocos minutos. No hay nada que hacer y
            # nada que informar como fallo.
            logger.info(f"se omite esta pasada: {exc}")
            _print({"mode": "once", "ok": True, "skipped": str(exc)})
            return 0
        logger.error(f"no se pudo arrancar la publicación automática: {exc}")
        _print({"mode": args.mode, "ok": False, "error": str(exc)})
        return 1
    except autopilot.AutopilotError as exc:
        logger.error(f"no se pudo ejecutar la publicación automática: {exc}")
        _print({"mode": args.mode, "ok": False, "error": str(exc)})
        return 1
    except KeyboardInterrupt:
        logger.warning("se interrumpió la publicación automática")
        _print({"mode": args.mode, "ok": False, "error": "interrumpida"})
        return 130
    except Exception as exc:
        # Los modos de ejecución los puede lanzar una tarea programada que solo lee el
        # código de salida y la última línea. Una traza suelta en stderr, sin ningún JSON en
        # stdout dejaría a esa tarea sin forma de saber qué pasó.
        logger.exception(f"falló la publicación automática: {exc}")
        _print(
            {"mode": args.mode, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        )
        return 1
