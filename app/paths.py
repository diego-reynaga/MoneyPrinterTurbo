"""Unifica los directorios de recursos, configuración y datos de ejecución entre la versión
de código fuente y la instalada.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_ROOT = Path(__file__).resolve().parent / "_assets"


def is_source_checkout() -> bool:
    return (SOURCE_ROOT / "pyproject.toml").is_file() and (
        SOURCE_ROOT / "cli.py"
    ).is_file()


def runtime_root() -> Path:
    """El código fuente sigue usando el directorio del proyecto; la versión instalada usa
    el directorio del usuario, y también permite indicar MPT_HOME explícitamente.
    """
    override = os.environ.get("MPT_HOME", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if is_source_checkout():
        return SOURCE_ROOT
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "MoneyPrinterTurbo"
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
        if not base.is_absolute():
            base = Path.home() / ".local" / "share"
    return base / "MoneyPrinterTurbo"


def config_template() -> Path:
    base = SOURCE_ROOT if is_source_checkout() else BUNDLED_ROOT
    return base / "config.example.toml"


def _copy_missing_resource(source: Path, target: Path) -> None:
    """Completa de forma atómica los recursos integrados; ni los arranques concurrentes ni
    las actualizaciones sobrescriben archivos que el usuario ya tenga.
    """
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".resource-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as destination, source.open("rb") as origin:
            shutil.copyfileobj(origin, destination)
        try:
            # En Windows, rename se niega a sobrescribir; en POSIX se usan enlaces duros
            # para lograr la misma publicación exclusiva.
            if os.name == "nt":
                os.rename(temporary, target)
            else:
                os.link(temporary, target)
        except FileExistsError:
            pass
    finally:
        Path(temporary).unlink(missing_ok=True)


def resource_directory(sub_dir: str = "") -> Path:
    root = runtime_root()
    target = root / "resource" / sub_dir
    if is_source_checkout() and root == SOURCE_ROOT:
        return target

    # El usuario puede seguir añadiendo fuentes y música. Se colocan en el directorio de
    # ejecución para que las subidas de la WebUI o las actualizaciones no escriban en
    # site-packages; solo se completan los archivos que faltan y se conservan los
    # recursos que el usuario haya editado.
    origin = (SOURCE_ROOT if is_source_checkout() else BUNDLED_ROOT) / "resource" / sub_dir
    if origin.is_dir():
        for source in origin.rglob("*"):
            if source.is_file() and not source.name.startswith("."):
                _copy_missing_resource(source, target / source.relative_to(origin))
    return target
