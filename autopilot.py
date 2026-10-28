"""
Publicación automática diaria en las redes conectadas, desde el código fuente.

Equivale a ``mpt autopilot`` en la versión instalada con pip. Toda la lógica vive en
``app/services/autopilot_cli.py``, para que las dos entradas se comporten igual.

Uso rápido:
    uv run python autopilot.py check    comprueba la configuración
    uv run python autopilot.py plan     muestra las horas sorteadas para hoy
    uv run python autopilot.py run      publica sin intervención, día tras día
    uv run python autopilot.py --help   todos los modos y opciones
"""

import sys

if __name__ == "__main__":
    from app.services.autopilot_cli import run_autopilot_cli

    # Las consolas de Windows usan por defecto una página de códigos antigua, y tanto
    # los temas generados como los nombres de archivo pueden llevar caracteres que no
    # están en ella. Se reutiliza la misma preparación de consola que la CLI de vídeo,
    # para que un UnicodeEncodeError no mate el planificador después de publicar.
    import cli

    cli._force_utf8_console()
    raise SystemExit(run_autopilot_cli(sys.argv[1:]))
