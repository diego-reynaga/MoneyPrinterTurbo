# MoneyPrinterTurbo — Paquete oficial de Python

**mpt es la distribución oficial de Python de
[MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo), mantenida por
[harry0703](https://pypi.org/user/harry0703/).**

Instala el paquete oficial con **`python -m pip install mpt`**.
El paquete `MoneyPrinterTurbo` de PyPI, con otro nombre, es una distribución
de terceros y este proyecto no lo mantiene.

MoneyPrinterTurbo genera vídeos cortos a partir de un tema o de un guion ya preparado.
Combina material local o en línea, narración de voz, subtítulos y música
de fondo. Usa la WebUI de forma interactiva o ejecuta el mismo flujo de vídeo desde la CLI.

## Instalación

Requiere Python 3.11 o superior. Se recomienda un entorno virtual:

```sh
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install mpt
```

## Arrancar la WebUI

```sh
mpt webui
```

La WebUI escucha en localhost y se abre en el navegador. Para elegir un puerto o ejecutarla
sin abrir el navegador:

```sh
mpt webui --port 8503 --headless
```

## Generar un vídeo desde la CLI

```sh
mpt cli --help
mpt cli --video-script "Tu guion completo." \
  --video-source local --video-materials /ruta/absoluta/clip.mp4 \
  --custom-audio-file /ruta/absoluta/narracion.wav \
  --no-subtitle-enabled --bgm-type none
```

Se admiten imágenes y vídeos locales. Los modelos de lenguaje en línea, el material de stock
y los proveedores de voz necesitan la configuración adecuada y acceso a la red.
Elige los proveedores e introduce las credenciales en los ajustes de la WebUI o en `config.toml`.

El paquete incluye fuentes por defecto, música de fondo y un binario de FFmpeg mediante
imageio-ffmpeg. Si hay un FFmpeg del sistema instalado, tiene prioridad. Los modelos de Whisper se
cargan bajo demanda.

## Configuración y archivos generados

La aplicación instalada guarda la configuración, las credenciales, las subidas y los vídeos
generados en tu directorio de usuario:

| Sistema | Directorio por defecto |
| --- | --- |
| macOS | `~/Library/Application Support/MoneyPrinterTurbo` |
| Windows | `%LOCALAPPDATA%\MoneyPrinterTurbo` |
| Linux | `~/.local/share/MoneyPrinterTurbo` (o `$XDG_DATA_HOME/MoneyPrinterTurbo`) |

Define `MPT_HOME` antes de arrancar la aplicación para elegir otro directorio.
La configuración se guarda en `config.toml`; los vídeos generados están en
`storage/tasks/<task-id>/`. La configuración y los recursos existentes del usuario se conservan
cuando la aplicación arranca o se actualiza.

Consulta la [documentación del proyecto](https://github.com/harry0703/MoneyPrinterTurbo)
para ver los proveedores admitidos y las opciones avanzadas. Informa de los problemas en
[GitHub Issues](https://github.com/harry0703/MoneyPrinterTurbo/issues).
