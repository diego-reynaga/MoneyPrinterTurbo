---
name: moneyprinterturbo-video
description: Usa esta skill siempre que el usuario quiera crear un vídeo terminado con MoneyPrinterTurbo a partir de un tema, un título, una idea, un prompt o un guion. Incluye vídeos cortos, con locución, educativos, de marketing, para redes sociales y con material de stock. Úsala también cuando el usuario mencione MoneyPrinterTurbo, proporcione la URL de esta Skill, pida a un agente de IA que instale o configure MoneyPrinterTurbo, necesite identificar claves API que faltan, quiera reparar una generación fallida o quiera localizar y recibir un MP4 generado. Usa esta skill cuando el resultado esperado sea un archivo de vídeo final, no instrucciones de instalación.
compatibility: Requiere un agente de IA con terminal, red, sistema de archivos y soporte para comandos de larga duración. Admite macOS y Windows y usa exclusivamente uv.
metadata:
  author: "harry0703@hotmail.com"
  version: "1.3.2"
  upstream: "https://github.com/harry0703/MoneyPrinterTurbo"
---

# Generación de vídeo con MoneyPrinterTurbo

El usuario solo tiene que dar un tema o un guion de vídeo. Completa automáticamente la instalación, la reutilización de la configuración, la generación, la espera y la entrega del MP4 final. No te detengas tras dar instrucciones o comandos.

## Comportamiento obligatorio

1. Pide al usuario solo las credenciales de API necesarias que falten, se hayan rechazado o no se puedan usar. Junta todas las credenciales necesarias en una sola petición.
2. No pidas confirmación antes de instalar, generar, esperar, usar valores por defecto o devolver el resultado.
3. No crees ni actualices repetidamente un plan detallado para una solicitud de generación estándar. Envía un único aviso de progreso breve y ejecuta.
4. Ejecuta el helper como un solo comando en primer plano con un tiempo de espera de al menos 20 minutos.
5. Nunca sondees con `sleep`, `echo`, `ps`, `ls` repetidos ni `tail` repetidos. Si el terminal devuelve un ID de sesión reanudable, sigue esperando en esa misma sesión.
6. No leas el log completo tras un éxito. Tras un fallo, lee solo el breve error notificado o el final relevante del log.
7. Nunca imprimas claves API, tokens, el `config.toml` completo ni fragmentos de configuración que contengan credenciales.

## Valores por defecto

Salvo que el usuario pida otra cosa, genera un vídeo vertical `9:16` con material de Pexels, la voz de Edge TTS española por defecto, subtítulos y música de fondo; el idioma del guion se detecta a partir del tema. Instala MoneyPrinterTurbo en el directorio personal del usuario.

## Ejecución

### 1. Localiza el helper

Resuelve `SKILL_DIR` a partir de este archivo `SKILL.md`. El helper es el `mpt_agent.py` contiguo. Pon el directorio de trabajo de la herramienta de terminal en `SKILL_DIR` e invoca el helper por su nombre de archivo relativo. No pongas la ruta absoluta del helper en el comando ni ejecutes una comprobación extra con `ls` o `dir`.

Esto es obligatorio en Windows porque algunos validadores de terminal de agentes quitan las barras invertidas de las rutas absolutas incrustadas en los comandos. Usar `mpt_agent.py` con `workdir=SKILL_DIR` evita ese fallo y funciona tanto en macOS como en Windows.

Si el cliente solo cargó el `SKILL.md` remoto, descarga el helper del repositorio oficial en un directorio temporal y luego usa ese directorio temporal como directorio de trabajo del comando:

```text
https://raw.githubusercontent.com/harry0703/MoneyPrinterTurbo/main/docs/skill/mpt_agent.py
```

### 2. Ejecuta el helper

No ejecutes una comprobación previa aparte con `uv --version`. Ejecuta el helper directamente. Si la shell informa explícitamente de que falta uv, instala uv y reintenta una vez el mismo comando del helper.

Instalación de uv en macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Instalación de uv en Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Usa este comando en primer plano con `workdir=SKILL_DIR` y un tiempo de espera de al menos 20 minutos:

```bash
uv run --no-project --python 3.11 python mpt_agent.py --subject "<tema del vídeo>"
```

En Windows, no pruebes rutas absolutas con barras invertidas, rutas absolutas con barras normales ni copias en el espacio de trabajo antes de este comando relativo. Si una herramienta de terminal informa `referenced_script_path_missing`, comprueba que su directorio de trabajo sea exactamente `SKILL_DIR` y reintenta una vez el comando relativo. No vayas probando variantes de ruta.

No uses Docker, Conda, el pip del sistema ni un entorno virtual gestionado a mano.

## Tratamiento de los códigos de salida

### Código de salida 0: entrega el resultado

La salida correcta tiene esta forma:

```text
MPT_RESULT
VIDEO_FILE=<ruta absoluta>/final-1.mp4
TASK_DIR=<ruta absoluta>/storage/tasks/<task_id>
LOG_FILE=<ruta absoluta>/run-<task_id>.log
RESULT_FILE=<ruta absoluta>/latest-result.json
```

`mpt_agent.py` solo emite `VIDEO_FILE` tras confirmar que el archivo existe y no está vacío. No ejecutes otro `ls`, `stat` ni ninguna comprobación del archivo.

Si el terminal informa `exitCode=0` pero trunca la salida o devuelve una referencia a un archivo de historial sin `MPT_RESULT`, no deduzcas que ha fallado ni revises logs antiguos. Lee una vez este archivo:

```text
~/MoneyPrinterTurbo/.agent-logs/moneyprinterturbo-video/latest-result.json
```

Trata `status=completed` como éxito. Devuelve solo la ruta absoluta del vídeo y una descripción concisa, por ejemplo:

```text
El vídeo está listo.
Tema: ...
Archivo de vídeo: /ruta/absoluta/a/final-1.mp4
Resumen: vídeo vertical con locución, subtítulos y música de fondo.
```

### Código de salida 10: pide las credenciales una vez

`MPT_NEEDS_INPUT` solo incluye los campos necesarios, los proveedores LLM recomendados y sus enlaces de registro, los requisitos de un endpoint personalizado compatible con OpenAI y los enlaces de registro de los proveedores de material. Pide solo los valores indicados y no pidas credenciales que ya estén en `config.toml`.

Cuando el usuario responda, vuelve a ejecutar el mismo comando en primer plano solo con las variables de entorno necesarias:

```text
MPT_LLM_PROVIDER
MPT_LLM_API_KEY
MPT_LLM_BASE_URL
MPT_LLM_MODEL_NAME
MPT_PEXELS_API_KEY
MPT_VOLCENGINE_ARK_API_KEY
MPT_OFOX_API_KEY
MPT_METASO_MINIMAX_API_KEY
MPT_MUAPI_API_KEY
MPT_INFISTAR_VIDEO_API_KEY
```

Cuando aparezca `SEEDANCE_CHARGE_CONFIRMATION_REQUIRED`, explica que cada
clip de Seedance generado crea una tarea de Ark de pago. Solo después de que el usuario
lo confirme explícitamente, vuelve a ejecutar con `--confirm-seedance-charge`; nunca añadas este flag en silencio.

Cuando aparezca `WAVESPEED_CHARGE_CONFIRMATION_REQUIRED`, explica que cada
clip de WaveSpeed generado lo factura WaveSpeed por solicitud. Solo después de que el usuario
lo confirme explícitamente, vuelve a ejecutar con `--confirm-wavespeed-charge`; nunca añadas este
flag en silencio.

Cuando aparezca `OFOX_CHARGE_CONFIRMATION_REQUIRED`, explica que cada
clip de OFox generado crea una tarea de pago. Solo después de que el usuario
lo confirme explícitamente, vuelve a ejecutar con `--confirm-ofox-charge`; nunca añadas este flag en silencio.

Cuando aparezca `METASO_MINIMAX_CHARGE_CONFIRMATION_REQUIRED`, explica que
cada clip de MiniMax H3 generado crea una tarea de Metaso de pago. Solo después de que el usuario
lo confirme explícitamente, vuelve a ejecutar con `--confirm-metaso-minimax-charge`; nunca añadas
este flag en silencio.

Cuando aparezca `MUAPI_CHARGE_CONFIRMATION_REQUIRED`, explica que cada
clip de MuAPI generado crea una tarea de pago. Solo después de que el usuario
lo confirme explícitamente, vuelve a ejecutar con `--confirm-muapi-charge`; nunca añadas este flag en silencio.

Cuando aparezca `INFISTAR_CHARGE_CONFIRMATION_REQUIRED`, explica que cada
clip de Infistar generado crea una tarea de pago. Solo después de que el usuario
lo confirme explícitamente, vuelve a ejecutar con `--confirm-infistar-charge`; nunca añadas este flag en silencio.
Usa la clave de vídeo propia, distinta de la clave del LLM de Infistar.

### Código de salida 1: repara o informa

Usa `MPT_ERROR` y `LOG_FILE` para reparar un problema recuperable y reintenta una vez. Pregunta al usuario solo si la reparación necesita una clave API nueva. Si el reintento falla, informa de la etapa fallida, un error breve y la ruta del log.

Un error de validación de rutas de la herramienta de terminal no es un fallo de generación del vídeo, porque el helper no llegó a arrancar. Corrige el directorio de trabajo y reintenta una vez el comando relativo. Nunca pidas al usuario que copie `mpt_agent.py`, que ejecute comandos a mano ni que confirme si el agente debe continuar.

## Configuración y modo en segundo plano de respaldo

El helper puede leer el `config.toml` local completo para reutilizar los ajustes existentes, pero nunca debe imprimir su contenido. Reutiliza automáticamente un proveedor LLM que funcione y valida las claves de Pexels configuradas a través del endpoint autenticado My Collections antes de generar.

Usa el modo en segundo plano solo si la plataforma del agente no puede esperar a un proceso en primer plano. Espera la notificación de finalización del proceso de la plataforma sin sondear y luego lee `latest-result.json` una vez.

## Alcance

- Solo se admiten macOS y Windows.
- Usa solo uv y la CLI de MoneyPrinterTurbo.
- No arranques Docker, la WebUI ni los servicios de la API.
- No ejecutes varios trabajos de vídeo a la vez.
- Pasa los requisitos de vídeo adicionales después de `--`. Ejecuta `cli.py --help` una sola vez solo cuando haya que verificar una opción desconocida.
