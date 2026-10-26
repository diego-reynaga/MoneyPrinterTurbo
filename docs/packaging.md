# Paquete de instalación con pip y publicación en PyPI

El paquete de distribución se llama `mpt`, el nombre del proyecto sigue siendo MoneyPrinterTurbo, y la versión pendiente de publicar es `1.3.9`.
La `1.3.8` se publicó el 2026-10-16 con `harry0703` en
[PyPI](https://pypi.org/project/mpt/1.3.8/), con wheel y sdist.
Los archivos publicados no se pueden sobrescribir, así que los cambios de código posteriores necesitan subir la versión y volver a construir y verificar.

## Construcción e instalación

Requiere Python 3.11 o superior. Las dependencias se resuelven e instalan con la wheel, y FFmpeg usa primero la versión del sistema;
también se puede usar el binario que proporciona imageio-ffmpeg.

Instalación y arranque desde PyPI:

```sh
python -m pip install mpt
mpt webui
mpt cli --help
```

Construcción y verificación en local:

```sh
uv build --out-dir storage/package/mpt-dist
python -m pip install ./storage/package/mpt-dist/mpt-1.3.9-py3-none-any.whl
mpt --help
mpt webui
mpt cli --help
```

Antes de la subida definitiva hay que confirmar la cuenta de PyPI, la versión de destino y los hashes de los paquetes, para no subir por lotes
un directorio mezclado con paquetes antiguos. Cada publicación requiere la autorización explícita del usuario y no se publica automáticamente con los commits ni los push.

## Comprobaciones antes de publicar

```sh
uvx twine check --strict \
  storage/package/mpt-dist/mpt-1.3.9-py3-none-any.whl \
  storage/package/mpt-dist/mpt-1.3.9.tar.gz
```

Comprueba la página de descripción de PyPI, los metadatos del paquete y que las dependencias se pueden instalar, y completa una generación de vídeo en un entorno virtual nuevo.
Solo se ejecuta la publicación definitiva tras una autorización explícita de subida; las contraseñas de cuenta, los códigos de verificación y los API Token no se escriben en el repositorio,
en los ejemplos de comandos ni en los logs. El README da la entrada de instalación a los usuarios, y la descripción de PyPI se mantiene aparte en
`docs/pypi-description.md`; actualizar la documentación del repositorio no modifica automáticamente la página de presentación de PyPI ya publicada.

`mpt webui` escucha por defecto solo en `127.0.0.1`, elige un puerto libre a partir del 8501 y abre automáticamente el navegador.
Se pueden usar `--host` y `--port` para indicar la dirección, y `--headless` para no abrir el navegador automáticamente:

```sh
mpt webui --port 8503 --headless
```

La CLI reutiliza los parámetros, el flujo de tareas, los códigos de salida y la salida JSON de `python cli.py`. Ejemplo completo sin conexión:

```sh
mpt cli --video-script "Un vídeo corto a partir de material local." \
  --video-source local --video-materials /ruta/absoluta/clip.mp4 \
  --custom-audio-file /ruta/absoluta/narracion.wav \
  --no-subtitle-enabled --bgm-type none
```

Las funciones en línea, como el texto, el material o el TTS, siguen necesitando la configuración, las claves y la conexión de red de cada servicio. Los modelos de Whisper se cargan
bajo demanda y no se incluyen en la wheel. Los logs de la CLI se escriben en stderr, y si tiene éxito stdout emite un JSON con la ruta del vídeo.

## Configuración y datos

La ejecución desde el código fuente sigue usando el directorio raíz del proyecto. La versión instalada no depende del directorio de trabajo actual, y su directorio de ejecución por defecto es:

| Plataforma | Directorio de ejecución |
| --- | --- |
| macOS | `~/Library/Application Support/MoneyPrinterTurbo` |
| Windows | `%LOCALAPPDATA%\MoneyPrinterTurbo` |
| Linux | `$XDG_DATA_HOME/MoneyPrinterTurbo`, por defecto `~/.local/share/MoneyPrinterTurbo` |

Se puede indicar explícitamente el directorio de ejecución con `MPT_HOME` (también en la versión de código fuente). Dentro de él:

- `config.toml`: en la primera ejecución se inicializa a partir de una plantilla sin claves, y los arranques posteriores no sobrescriben la configuración existente.
- `storage/tasks`: tareas y vídeos generados; las demás cachés y archivos subidos están en `storage`.
- `resource/fonts`, `resource/songs`: en el primer uso se copian los recursos integrados, y puedes añadir tus propios archivos.
- `models`: aquí se pueden colocar los modelos de Whisper descargados a mano.

Los recursos del usuario solo se completan con los archivos que faltan, y las actualizaciones no sobrescriben los existentes. El paquete de instalación incluye dos fuentes chinas por defecto,
una fuente vietnamita y la música de fondo integrada; las alternativas `STHeitiLight.ttc` y `MicrosoftYaHeiNormal.ttc`
no entran en el paquete, para controlar su tamaño. Si necesitas esos pesos de fuente, colócalos en `resource/fonts` del directorio de ejecución.
Los recursos y el aspecto originales de la versión de código fuente no cambian.

## Requisitos de la verificación local

Reconstruye la wheel a partir del sdist, instálala en un entorno virtual nuevo que no herede paquetes del sistema, sal del directorio del código fuente y vacía
`PYTHONPATH`, y luego verifica `mpt cli --help`, la página real de la WebUI y la generación de un vídeo. El vídeo de salida debe poder
decodificarlo entero FFmpeg, y hay que confirmar que la configuración y los productos no se escriben en site-packages. El paquete no debe incluir
el `config.toml` local, claves, cachés, productos de tareas ni instrucciones de colaboración locales.
