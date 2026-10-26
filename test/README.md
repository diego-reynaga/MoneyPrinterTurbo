# Directorio de tests de MoneyPrinterTurbo

Este directorio contiene los tests unitarios del proyecto **MoneyPrinterTurbo**.

## Estructura del directorio

- `services/`: tests unitarios y de controladores organizados por dominio
  - `test_task.py`: tests del flujo de tareas
  - `test_task_manager.py`: tests de las colas en memoria y de Redis
  - `test_controller_*.py`: tests de los controladores de la API, separados por dominio de controlador
  - `test_video.py`, `test_voice.py`: tests de los servicios multimedia
- `test_main.py`: test del punto de entrada de la aplicación

## Ejecutar los tests

La suite de la CI usa pytest, que también ejecuta los tests existentes basados en
`unittest.TestCase`:

```bash
# Ejecutar todos los tests
uv run python -X utf8 -m pytest -q test

# Ejecutar un archivo de test concreto
uv run python -X utf8 -m pytest -q test/services/test_video.py

# Ejecutar una clase de test concreta
uv run python -X utf8 -m pytest -q test/services/test_video.py::TestVideoService

# Ejecutar un método de test concreto
uv run python -X utf8 -m pytest -q test/services/test_video.py::TestVideoService::test_preprocess_video
```

Para ejecutar la misma comprobación de cobertura de ramas que usa la CI:

```bash
uv run python -X utf8 -m coverage run -m pytest -q test
uv run python -m coverage report
```

Los tests contra proveedores reales se omiten por defecto. Para ejecutar tests que pueden llamar a servicios
externos de TTS o LLM, define `MPT_RUN_INTEGRATION_TESTS=1` y proporciona las credenciales
necesarias de cada proveedor.

## Añadir tests nuevos

Para añadir tests de otros componentes, sigue estas pautas:

1. Nombra los archivos `test_<dominio>.py` y mantén cada archivo centrado en un solo dominio.
2. Divide las suites de controladores amplias en archivos como `test_controller_video.py`.
3. Usa funciones de pytest o `unittest.TestCase`; pytest recoge ambos.
4. Nombra las funciones y métodos de test con el prefijo `test_`.

## Recursos de los tests

Coloca los archivos de recursos que necesiten los tests en el directorio `test/resources`.
