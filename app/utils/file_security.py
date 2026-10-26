import os


def resolve_path_within_directory(
    base_dir: str,
    unsafe_path: str,
    *,
    require_file: bool = True,
) -> str:
    # La ruta que pasa el usuario puede ser un nombre de archivo, una ruta relativa, una
    # ruta absoluta o incluso contener `../`. Aquí se resuelve todo a la ruta real y se
    # usa commonpath para comprobar si sigue dentro del directorio permitido. Es más
    # fiable que comparar el prefijo de la cadena, y cubre enlaces simbólicos,
    # separadores repetidos, rutas relativas y otros casos; sirve para directorios en
    # lista blanca como el de subidas, el de material o el de artefactos de tareas.
    if not unsafe_path:
        raise ValueError("no se permite una ruta vacía")

    base_dir_real = os.path.realpath(base_dir)
    candidate_path = unsafe_path
    if not os.path.isabs(candidate_path):
        candidate_path = os.path.join(base_dir_real, candidate_path)

    resolved_path = os.path.realpath(candidate_path)
    try:
        common_path = os.path.commonpath([base_dir_real, resolved_path])
    except ValueError as exc:
        # En Windows, unidades distintas provocan ValueError; esas rutas nunca
        # pertenecen al directorio permitido.
        raise ValueError("la ruta está fuera del directorio permitido") from exc

    if common_path != base_dir_real:
        raise ValueError("la ruta está fuera del directorio permitido")

    if require_file and not os.path.isfile(resolved_path):
        raise ValueError("el archivo no existe")

    return resolved_path
