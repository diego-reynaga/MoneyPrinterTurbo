import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from time import perf_counter
from typing import List

from loguru import logger
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from app.config import config
from app.models.llm_provider import DEFAULT_LLM_PROVIDER_ID, get_llm_provider
from app.utils import utils
from app.utils.request_logging import trace_sdk_call

_max_retries = 5
MIN_SCRIPT_PARAGRAPH_NUMBER = 1
MAX_SCRIPT_PARAGRAPH_NUMBER = 10
MAX_SCRIPT_PROMPT_LENGTH = 2000
MAX_SCRIPT_SYSTEM_PROMPT_LENGTH = 8000
# "Probar la conexión con el modelo" solo envía una solicitud mínima. El SDK de OpenAI
# tiene por defecto un tiempo de espera de 600 segundos y reintenta automáticamente 2
# veces; si el endpoint no es accesible (fallo de proxy, cortafuegos o DNS), el botón de
# prueba se queda colgado mucho tiempo y al final solo deja un "Request timed out." (ver
# issue #1642). Aquí se da a la prueba de conexión un tiempo de espera corto y acotado y
# se desactivan los reintentos del SDK, para que el fallo sea rápido y diagnosticable;
# la cadena formal de generación de guiones mantiene el comportamiento por defecto del
# SDK, para no perjudicar a modelos de razonamiento lentos ni al arranque en frío de
# Ollama local.
LLM_CONNECTION_TEST_TIMEOUT_SECONDS = 30.0
LLM_CONNECTION_TEST_MAX_RETRIES = 0
# Estos fragmentos proceden de los textos de APITimeoutError / APIConnectionError del
# SDK de OpenAI y de las excepciones de conexión subyacentes de httpx; si coincide
# cualquiera, se trata como "endpoint inaccesible".
_CONNECTION_FAILURE_MARKERS = (
    "timed out",
    "timeout",
    "connection error",
    "connection refused",
    "connection reset",
    "connection aborted",
    "failed to connect",
)
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_UNOPENED_THINK_BLOCK_RE = re.compile(r"^.*</think\s*>", re.IGNORECASE | re.DOTALL)
_URL_USERINFO_RE = re.compile(
    r"((?:https?|wss?)://)([^/\s?#@]+@)", re.IGNORECASE
)
_SENSITIVE_QUERY_RE = re.compile(
    r"([?&](?:api[_-]?key|access[_-]?token|token|key|secret|password)=)([^&#\s]+)",
    re.IGNORECASE,
)

DEFAULT_SCRIPT_SYSTEM_PROMPT = """
# Role: Video Script Generator

## Goals:
Generate a script for a video, depending on the subject of the video.

## Constrains:
1. the script is to be returned as a string with the specified number of paragraphs.
2. do not under any circumstance reference this prompt in your response.
3. get straight to the point, don't start with unnecessary things like, "welcome to this video".
4. you must not include any type of markdown or formatting in the script, never use a title.
5. only return the raw content of the script.
6. do not include "voiceover", "narrator" or similar indicators of what should be spoken at the beginning of each paragraph or line.
7. you must not mention the prompt, or anything about the script itself. also, never talk about the amount of paragraphs or lines. just write the script.
8. respond in the same language as the video subject.
""".strip()

# La CLI de Claude Code usa por defecto el prompt del sistema de su agente de
# programación, con muchas restricciones que no tienen que ver con la redacción de
# textos y que desviarían la generación de guiones y palabras clave de lo pedido, así
# que se sustituye por completo al llamarla.
CLAUDE_CODE_SYSTEM_PROMPT = (
    "You are a concise copywriter. Follow the user's instructions and output "
    "format exactly, and output nothing else."
)
CLAUDE_CODE_DEFAULT_TIMEOUT = 300.0
# `--tools ""` desactiva todas las herramientas integradas, y `--safe-mode` desactiva
# todas las personalizaciones del usuario como CLAUDE.md, skills, hooks, plugins o MCP,
# manteniendo el funcionamiento normal de la autenticación, la elección de modelo y los
# permisos. Ambos requieren una CLI reciente; las versiones antiguas salen con "unknown
# option", y quien llama lo convierte en un aviso claro.
CLAUDE_CODE_MIN_CLI_VERSION = "2.1.260"
# Estas variables de entorno hacen que la CLI pase a usar una clave API o proveedores de
# terceros (Bedrock, Vertex, Foundry, Mantle, Gateway, etc.), saltándose el inicio de
# sesión de la suscripción y generando facturación adicional. Enumerarlas una a una es
# fácil que se olvide alguna, y la CLI irá añadiendo proveedores, así que se eliminan
# por familias de prefijo:
#   ANTHROPIC_*           API Key, Auth Token, Base URL, endpoints y perfiles de cada
#   proveedor
#   CLAUDE_CODE_USE_*     interruptores de proveedor
#   CLAUDE_CODE_SKIP_*_AUTH  interruptores que se saltan la autenticación del proveedor
CLAUDE_CODE_CONFLICTING_ENV_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_USE_")
CLAUDE_CODE_CONFLICTING_ENV_VARS = (
    "AWS_BEARER_TOKEN_BEDROCK",
    "CLAUDE_CODE_GATEWAY_TOKEN_FILE_DESCRIPTOR",
)
# Estos dos tipos de variables no pueden eliminarse:
#   CLAUDE_CODE_OAUTH_TOKEN es la única forma de autenticación de la suscripción dentro
#   del contenedor (no coincide con los prefijos de arriba);
#   *_CONFIG_DIR solo indica dónde se guardan las credenciales, y eliminarlo invalidaría
#    la suscripción con la sesión ya iniciada.
CLAUDE_CODE_PRESERVED_ENV_VARS = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_CONFIG_DIR",
    "CLAUDE_CONFIG_DIR",
)


def _is_conflicting_claude_code_env(name: str) -> bool:
    """Decide si una variable de entorno haría que la CLI pasara del inicio de sesión de la
    suscripción a otra forma de autenticación.
    """
    if name in CLAUDE_CODE_PRESERVED_ENV_VARS:
        return False
    if name in CLAUDE_CODE_CONFLICTING_ENV_VARS:
        return True
    if name.startswith(CLAUDE_CODE_CONFLICTING_ENV_PREFIXES):
        return True
    return name.startswith("CLAUDE_CODE_SKIP_") and name.endswith("_AUTH")


def coerce_claude_code_timeout(value, config_key: str = "claude_code_timeout"):
    """
    Convierte el tiempo de espera de la configuración en un número de segundos positivo
    y finito.

    En TOML puede escribirse como `claude_code_timeout = 300` (int/float) o como `"300"`
    (cadena), así que no se puede llamar directamente a `strip()`. nan / inf bloquearían
    para siempre `subprocess.run(timeout=...)`, así que también se rechazan aquí.
    """
    if value is None:
        return CLAUDE_CODE_DEFAULT_TIMEOUT

    if isinstance(value, bool):
        # bool es subclase de int, pero True segundos claramente no es el tiempo de
        # espera que quiere el usuario.
        raise ValueError(f"{config_key} debe ser un número de segundos, se recibió {value!r}")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return CLAUDE_CODE_DEFAULT_TIMEOUT
        try:
            seconds = float(text)
        except ValueError:
            raise ValueError(
                f"{config_key} debe ser un número de segundos, se recibió {value!r}"
            ) from None
    elif isinstance(value, (int, float)):
        seconds = float(value)
    else:
        raise ValueError(f"{config_key} debe ser un número de segundos, se recibió {value!r}")

    if not math.isfinite(seconds):
        raise ValueError(f"{config_key} debe ser un número finito, se recibió {value!r}")
    if seconds <= 0:
        raise ValueError(f"{config_key} debe ser mayor que 0, se recibió {value!r}")
    return seconds


def _resolve_provider_field_value(raw_value, default_value):
    """
    Solo se vuelve al valor por defecto del registro cuando «no está configurado».

    Antes se usaba `raw or default_value`, que también trataba como no configurados
    valores válidos como 0 y false y los sustituía: `claude_code_timeout = 0` se
    cambiaba en silencio a 300, mientras que `"0"` daba error. El valor por defecto solo
    se aplica con None o una cadena en blanco, para que la validación de la
    configuración sea coherente con todas las formas de escribirla.
    """
    if raw_value is None:
        return default_value
    if isinstance(raw_value, str) and not raw_value.strip():
        return default_value
    return raw_value


def build_claude_code_env(base_env=None):
    """
    Construye un entorno de subproceso que solo depende del inicio de sesión de la
    suscripción.

    Devuelve (diccionario de variables de entorno, lista de nombres de variables
    eliminadas). Se eliminan las variables que cambiarían la forma de autenticación o el
    proveedor; `CLAUDE_CODE_OAUTH_TOKEN` debe conservarse: dentro del contenedor no hay
    keychain y la CLI solo puede autenticar la suscripción con él.
    """
    env = dict(os.environ if base_env is None else base_env)
    removed = sorted(name for name in env if _is_conflicting_claude_code_env(name))
    for name in removed:
        env.pop(name, None)
    return env, removed


def _normalize_text_response(content, llm_provider: str) -> str:
    # Los distintos SDK de LLM pueden devolver None, cadenas vacías o incluso objetos
    # que no son cadenas en caso de excepción o bloqueo. Aquí se hace una validación de
    # respaldo común, para que llamar después directamente a `.replace()` no lance
    # errores de atributo como `NoneType`.
    if content is None:
        raise ValueError(f"[{llm_provider}] devolvió contenido de texto vacío")

    if not isinstance(content, str):
        raise TypeError(
            f"[{llm_provider}] devolvió contenido que no es texto: {type(content).__name__}"
        )

    # Los modelos de razonamiento como MiniMax M3 o DeepSeek R1 pueden devolver el
    # razonamiento interno envuelto en `<think>...</think>`. El guion y las palabras
    # clave del vídeo solo necesitan el texto final que se puede leer; si no se limpiara
    # de forma común en la capa de servicio, la WebUI, los subtítulos y la locución
    # tratarían el razonamiento como texto.
    content = _THINK_BLOCK_RE.sub("", content)
    # Las plantillas de conversación de modelos como DeepSeek R1 o QwQ ponen el
    # `<think>` de apertura en el prompt, y el texto que devuelven gateways como
    # vLLM/Ollama solo trae `razonamiento...</think>respuesta`. Tras limpiar los pares,
    # todo lo anterior a una etiqueta de cierre residual es razonamiento y tampoco puede
    # entrar en el guion.
    content = _UNOPENED_THINK_BLOCK_RE.sub("", content)
    content = _UNCLOSED_THINK_BLOCK_RE.sub("", content).strip()
    if not content:
        raise ValueError(f"[{llm_provider}] devolvió contenido de texto vacío")

    # El ``strip()`` anterior ya ha limpiado los espacios al principio y al final. Aquí
    # hay que conservar los saltos de línea simples y dobles del texto: la generación
    # del guion usa el doble salto de línea para separar párrafos, y el tratamiento de
    # subtítulos también lee el texto del usuario línea a línea.
    return content


def _sanitize_error_message(error: object) -> str:
    """
    Limpia los mensajes de error devueltos a la WebUI/API, para que no se filtren las
    credenciales de una base_url personalizada.

    Algunos SDK compatibles con OpenAI incluyen la URL de la solicitud tal cual en el
    mensaje de la excepción. Si el usuario configuró `https://user:pass@example.com/v1`
    para un gateway proxy, devolver directamente `str(e)` expondría la contraseña a la
    página, a quien llama a la API o a los logs posteriores. Aquí solo se trata el texto
    del error, sin cambiar la dirección real de la solicitud, para no afectar a la
    cadena de llamadas normal.
    """
    message = str(error)
    message = _URL_USERINFO_RE.sub(r"\1***:***@", message)
    message = _SENSITIVE_QUERY_RE.sub(r"\1***", message)
    return message


def _extract_chat_completion_text(response, llm_provider: str) -> str:
    # En casos anómalos, una interfaz compatible con OpenAI puede devolver un objeto de
    # respuesta sin choices, o con choices/message/content vacíos. Aquí se hace una
    # validación común de la estructura, para evitar errores de acceso a atributos de
    # bajo nivel como `NoneType is not subscriptable`.
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(f"[{llm_provider}] devolvió choices vacío")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError(f"[{llm_provider}] devolvió un mensaje vacío")

    content = getattr(message, "content", None)
    return _normalize_text_response(content, llm_provider)


def _get_response_field(value, key: str):
    """Lee campos tanto de un dict como de un objeto de respuesta del SDK."""
    if isinstance(value, dict):
        return value.get(key)

    try:
        return value[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(value, key, None)


def _sdk_transport_options(request_timeout, max_retries) -> dict:
    """Convierte las restricciones opcionales de tiempo de espera/reintentos en parámetros
    de construcción del cliente del SDK de OpenAI.

    Si ambos parámetros son None devuelve un diccionario vacío, y el cliente sigue
    usando los valores por defecto del SDK (600 segundos de espera y 2 reintentos
    automáticos), así que las cadenas normales como la generación de guiones no cambian;
    el presupuesto de la solicitud solo se ajusta cuando un llamador que necesita fallar
    rápido, como la prueba de conexión, pasa explícitamente valores acotados.
    """
    options = {}
    if request_timeout is not None:
        options["timeout"] = request_timeout
    if max_retries is not None:
        options["max_retries"] = max_retries
    return options


def _explain_connection_test_failure(error_message: str) -> str:
    """Añade pistas para diagnosticar los fallos de tiempo de espera/conexión, en lugar de
    mostrar solo el texto original del SDK.

    En el issue #1642 la API del usuario funcionaba, y el fallo venía del enlace entre
    el equipo y el endpoint (proxy, cortafuegos o DNS), pero "Request timed out." no
    apuntaba a ninguna causa posible. Los demás errores (autenticación, cuota, etc.) se
    devuelven tal cual.
    """
    normalized = (error_message or "").lower()
    if not any(marker in normalized for marker in _CONNECTION_FAILURE_MARKERS):
        return error_message
    return (
        f"{error_message} (la petición no terminó en "
        f"{LLM_CONNECTION_TEST_TIMEOUT_SECONDS:.0f}s). Comprueba que la URL base "
        "configurada es accesible desde esta máquina y que ningún proxy ni "
        "cortafuegos está bloqueando la petición."
    )


def _extract_qwen_generation_text(response) -> str:
    """
    Extrae el texto de una respuesta Generation de DashScope.

    Cuando Qwen se llama con `messages` devuelve una estructura chat:
    `output.choices[0].message.content`; solo la forma completion antigua devuelve
    `output.text`. Aquí se admiten ambas rutas, para que, cuando `output.text` sea None,
    seguir con `.replace()` no provoque un AttributeError imposible de diagnosticar.
    """
    output = _get_response_field(response, "output")
    choices = _get_response_field(output, "choices") if output else None
    if choices is not None:
        if not choices:
            logger.warning("Qwen devolvió una lista choices vacía")
            raise ValueError("[qwen] devolvió choices vacío")

        first_choice = choices[0]
        message = _get_response_field(first_choice, "message")
        content = _get_response_field(message, "content") if message else None
        if content is not None:
            return _normalize_text_response(content, "qwen")

    text = _get_response_field(output, "text") if output else None
    return _normalize_text_response(text, "qwen")


def _generate_response(
    prompt: str,
    app_config=None,
    request_timeout: float | None = None,
    max_retries: int | None = None,
) -> str:
    sdk_client = None
    sdk_stream = None
    try:
        # La WebUI permite al usuario preparar el siguiente texto mientras se genera un
        # vídeo. Quien llama puede pasar una instantánea de la configuración en el
        # momento del envío, para que durante los reintentos de la solicitud al modelo
        # no se cambie a otro proveedor, URL base o modelo porque una tarea en segundo
        # plano terminó y aplicó la configuración nueva.
        runtime_app_config = app_config if app_config is not None else config.app
        llm_provider = str(
            runtime_app_config.get("llm_provider", DEFAULT_LLM_PROVIDER_ID)
        ).lower()
        provider = get_llm_provider(llm_provider)
        if provider is None:
            raise ValueError(f"{llm_provider}: proveedor LLM no admitido")

        logger.info(f"proveedor LLM: {llm_provider}")
        api_key = runtime_app_config.get(provider.config_key("api_key"), "")
        configured_model = runtime_app_config.get(provider.config_key("model_name"), "")
        model_name = provider.resolve_model_name(configured_model)
        if configured_model and model_name != configured_model:
            logger.warning(
                f"el modelo '{configured_model}' de {llm_provider} está obsoleto, "
                f"se usa '{model_name}'"
            )
        configured_base_url = runtime_app_config.get(
            provider.config_key("base_url"), ""
        )
        base_url = provider.resolve_base_url(configured_base_url)
        if configured_base_url and configured_base_url.strip().rstrip("/") in {
            url.rstrip("/") for url in provider.deprecated_base_urls
        }:
            logger.warning(
                f"la URL base '{configured_base_url}' de {llm_provider} está obsoleta, "
                f"se usa '{base_url}'"
            )
        adapter = provider.adapter
        api_version = ""

        # La dirección por defecto de Ollama depende de si se ejecuta en un contenedor,
        # así que no se puede guardar como valor estático del registro; el registro
        # sigue encargándose del modelo y de las reglas de campos obligatorios, y la
        # diferencia de entorno de ejecución se resuelve aquí.
        if llm_provider == "ollama":
            api_key = "ollama"
            if not base_url:
                base_url = config.get_default_ollama_base_url()

        if adapter == "azure":
            api_version = runtime_app_config.get(
                provider.config_key("api_version"), "2024-02-15-preview"
            )

        extra_values = {
            field.config_suffix: _resolve_provider_field_value(
                runtime_app_config.get(provider.config_key(field.config_suffix)),
                field.default_value,
            )
            for field in provider.extra_fields
        }

        if provider.requires_api_key and not api_key:
            raise ValueError(
                f"{llm_provider}: api_key no está configurada; configúrala en el archivo config.toml."
            )
        if provider.requires_model_name and not model_name:
            raise ValueError(
                f"{llm_provider}: model_name no está configurado; configúralo en el archivo config.toml."
            )
        if provider.requires_base_url and not base_url:
            raise ValueError(
                f"{llm_provider}: base_url no está configurada; configúrala en el archivo config.toml."
            )

        for field in provider.extra_fields:
            if field.required and not extra_values[field.config_suffix]:
                raise ValueError(
                    f"{llm_provider}: {field.config_suffix} no está configurado; "
                    "configúralo en el archivo config.toml."
                )

        if adapter == "qwen":
            import dashscope
            from dashscope.api_entities.dashscope_response import GenerationResponse

            response = trace_sdk_call(
                llm_provider, base_url, dashscope.Generation.call,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
            )
            if response:
                if isinstance(response, GenerationResponse):
                    status_code = response.status_code
                    if status_code != 200:
                        raise Exception(
                            f'[{llm_provider}] devolvió una respuesta de error: "{response}"'
                        )

                    return _extract_qwen_generation_text(response)
                else:
                    raise Exception(
                        f'[{llm_provider}] devolvió una respuesta no válida: "{response}"'
                    )
            else:
                raise Exception(f"[{llm_provider}] devolvió una respuesta vacía")

        if adapter == "gemini":
            from google import genai
            from google.genai import types

            http_options = types.HttpOptions(base_url=base_url) if base_url else None
            generation_config = types.GenerateContentConfig(
                temperature=0.5,
                top_p=1,
                top_k=1,
                max_output_tokens=2048,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                ],
            )

            try:
                # La versión nueva de google-genai expone el servicio de modelos a
                # través de un Client unificado. El gestor de contexto cierra la
                # conexión HTTP subyacente al terminar la solicitud, para no acumular
                # recursos de conexión al generar con frecuencia.
                with genai.Client(
                    api_key=api_key,
                    http_options=http_options,
                ) as client:
                    response = trace_sdk_call(
                        llm_provider, base_url, client.models.generate_content,
                        model=model_name,
                        contents=prompt,
                        config=generation_config,
                    )
                generated_text = response.text
            except (AttributeError, IndexError, ValueError) as e:
                logger.warning(f"Gemini devolvió contenido de respuesta no válido: {str(e)}")
                raise ValueError(f"[{llm_provider}] devolvió contenido de respuesta no válido")

            return _normalize_text_response(generated_text, llm_provider)

        if adapter == "cloudflare_ai_gateway":
            account_id = extra_values["account_id"]
            gateway_id = extra_values["gateway_id"]
            # La API REST de AI Gateway que recomienda actualmente Cloudflare es
            # compatible con el SDK de OpenAI. El Account ID sirve para construir el
            # endpoint unificado y el Gateway ID se elige mediante una cabecera; aquí ya
            # no se llama a la interfaz específica /ai/run/{model} de Workers AI.
            client = sdk_client = OpenAI(
                api_key=api_key,
                base_url=(
                    f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
                ),
                default_headers={"cf-aig-gateway-id": gateway_id},
                **_sdk_transport_options(request_timeout, max_retries),
            )
            response = trace_sdk_call(
                llm_provider, base_url, client.chat.completions.create,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "litellm":
            import litellm

            if not model_name:
                raise ValueError(
                    f"{llm_provider}: model_name no está configurado; configúralo en el archivo config.toml."
                )

            response = trace_sdk_call(
                llm_provider, base_url, litellm.completion,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                drop_params=True,
            )

            if not response:
                raise ValueError(f"[{llm_provider}] devolvió una respuesta vacía")
            if not getattr(response, "choices", None):
                raise ValueError(f"[{llm_provider}] devolvió una respuesta vacía")

            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "azure":
            # El SDK de Azure OpenAI genera la dirección de solicitud propia con
            # `azure_endpoint` y `api_version`, y no puede seguir reutilizando la lógica
            # de inicialización con `base_url` del caso normal compatible con OpenAI de
            # abajo. Aquí la solicitud se completa dentro de la rama de Azure y se
            # vuelve de inmediato, para que el cliente no lo sobrescriba un fallback
            # posterior y las credenciales de Azure configuradas por el usuario pasen la
            # validación pero no se usen en la solicitud real.
            logger.info(f"solicitando chat completion a Azure, modelo: {model_name}")
            client = sdk_client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=base_url,
                **_sdk_transport_options(request_timeout, max_retries),
            )
            response = trace_sdk_call(
                llm_provider, base_url, client.chat.completions.create,
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, ChatCompletion):
                    return _extract_chat_completion_text(response, llm_provider)
                else:
                    raise Exception(
                        f'[{llm_provider}] devolvió una respuesta no válida: "{response}"; revisa tu conexión '
                        f"de red e inténtalo de nuevo."
                    )
            else:
                raise Exception(
                    f"[{llm_provider}] devolvió una respuesta vacía; revisa tu conexión de red e inténtalo de nuevo."
                )

        if adapter == "claude_code":
            # La suscripción de Claude (Pro / Max / Team) no emite claves API, y sus
            # credenciales solo las puede usar el propio cliente oficial de Claude Code.
            # Aquí no se pide directamente a la API de Anthropic, sino que se llama en
            # modo headless a la CLI claude local con la sesión iniciada (`claude -p`);
            # la CLI se encarga de la autenticación y la generación de guiones solo
            # consume el texto que devuelve.
            configured_cli = (extra_values.get("cli_path") or "").strip() or "claude"
            cli_path = shutil.which(configured_cli)
            if not cli_path and os.path.isfile(configured_cli):
                cli_path = configured_cli
            if not cli_path:
                raise ValueError(
                    f"{llm_provider}: no se encontró la CLI de claude ('{configured_cli}'); "
                    f"instálala en el entorno de ejecución o configura "
                    f"{provider.config_key('cli_path')} en el archivo config.toml."
                )

            try:
                timeout_seconds = coerce_claude_code_timeout(
                    extra_values.get("timeout"), provider.config_key("timeout")
                )
            except ValueError as timeout_error:
                raise ValueError(f"{llm_provider}: {timeout_error}") from None

            # El prompt se pasa por stdin, no en la línea de comandos: en Windows, el
            # claude instalado con npm es claude.cmd, y cmd.exe corta los argumentos en
            # el primer salto de línea, así que se perderían el prompt de varias líneas
            # y los argumentos de aislamiento que vienen detrás.
            command = [
                cli_path,
                "-p",
                "--output-format",
                "json",
                "--system-prompt",
                CLAUDE_CODE_SYSTEM_PROMPT,
                # Desactiva todas las herramientas integradas, para que solo se genere
                # texto.
                "--tools",
                "",
                # Desactiva las personalizaciones del usuario como CLAUDE.md, skills,
                # hooks, plugins o MCP; la autenticación y la elección de modelo no se
                # ven afectadas (no se puede usar --bare, que desactiva OAuth).
                "--safe-mode",
            ]
            # Con el nombre del modelo vacío se usa el modelo por defecto de la propia
            # CLI, para que un ID de modelo fijado aquí no deje de funcionar cuando
            # cambien los modelos disponibles de la suscripción.
            if model_name:
                command += ["--model", model_name]

            cli_env, removed_env = build_claude_code_env()
            if removed_env:
                # Solo se registran los nombres de las variables, no sus valores, para
                # no escribir claves en el log.
                logger.warning(
                    f"{llm_provider}: se ignoran variables de entorno en conflicto "
                    f"para usar el inicio de sesión de la suscripción: {', '.join(removed_env)}"
                )

            logger.info(f"invocando la CLI de claude, modelo: {model_name or 'por defecto de la CLI'}")
            # La CLI lee el CLAUDE.md y los ajustes del proyecto del directorio de
            # trabajo, y eso contaminaría el resultado del texto, así que se ejecuta
            # siempre en un directorio temporal vacío.
            with tempfile.TemporaryDirectory() as work_dir:
                try:
                    completed = subprocess.run(
                        command,
                        input=prompt,
                        capture_output=True,
                        text=True,
                        # La CLI siempre emite UTF-8. Sin una codificación explícita,
                        # text=True decodifica con el locale del sistema (p. ej. cp1252
                        # en Windows que no está en inglés), así que todos los
                        # caracteres no ASCII llegan al guion como texto ilegible.
                        encoding="utf-8",
                        errors="replace",
                        timeout=timeout_seconds,
                        cwd=work_dir,
                        env=cli_env,
                    )
                except subprocess.TimeoutExpired:
                    raise Exception(
                        f"[{llm_provider}] la CLI de claude agotó el tiempo tras "
                        f"{timeout_seconds:.0f}s"
                    )

            # Los fallos como no tener sesión iniciada o haber agotado el uso también
            # devuelven JSON (`is_error` verdadero y `result` con el motivo legible),
            # solo que con un código de salida distinto de 0. Por eso se analiza primero
            # stdout, y solo si no se obtiene JSON se vuelve al código de salida y a
            # stderr.
            stdout = (completed.stdout or "").strip()
            try:
                payload = json.loads(stdout) if stdout else None
            except json.JSONDecodeError:
                payload = None

            if payload is None:
                detail = (completed.stderr or stdout or "").strip()
                if "unknown option" in detail.lower():
                    raise Exception(
                        f"[{llm_provider}] la CLI de claude instalada no admite "
                        f"las opciones de aislamiento necesarias; actualiza a "
                        f"{CLAUDE_CODE_MIN_CLI_VERSION} o posterior: {detail[:300]}"
                    )
                if completed.returncode != 0:
                    raise Exception(
                        f"[{llm_provider}] la CLI de claude terminó con el código "
                        f"{completed.returncode}: {detail[:500]}"
                    )
                raise Exception(
                    f'[{llm_provider}] devolvió una respuesta no válida: "{detail[:500]}"'
                )

            if payload.get("is_error") or completed.returncode != 0:
                reason = str(payload.get("result") or "").strip() or (
                    f"la CLI de claude terminó con el código {completed.returncode}"
                )
                # Dentro del contenedor no se puede ejecutar el /login interactivo, así
                # que aquí se da directamente una forma de autenticación utilizable.
                if "login" in reason.lower():
                    reason += (
                        " (ejecuta `claude setup-token` en el host y pasa el token "
                        "al contenedor como CLAUDE_CODE_OAUTH_TOKEN)"
                    )
                raise Exception(
                    f'[{llm_provider}] devolvió una respuesta de error: "{reason[:500]}"'
                )

            return _normalize_text_response(payload.get("result"), llm_provider)

        if adapter == "kimi_code":
            # La suscripción de Kimi (las ventajas de Kimi Code de Plus y superiores) no
            # emite claves API de pago por uso; la autenticación la hace la autorización
            # OAuth de dispositivo: la WebUI guía al usuario para que confirme con un
            # clic en el navegador y guarda en local el refresh token; el access token
            # caduca a los 15 minutos y kimi_code_oauth lo renueva de forma
            # transparente. Las solicitudes van al endpoint oficial compatible con
            # OpenAI, con una cuota independiente de la API de pago por uso.
            from app.services import kimi_code_oauth

            try:
                timeout_seconds = kimi_code_oauth.coerce_timeout(
                    extra_values.get("timeout")
                )
            except ValueError as timeout_error:
                raise ValueError(f"{llm_provider}: {timeout_error}") from None

            access_token = kimi_code_oauth.get_valid_access_token()
            # La región de autorización la determina la credencial: el token solo es
            # válido en la región que lo emitió. Si el usuario cambia de plataforma de
            # API tras autorizar y se pidiera según la región de la interfaz, un token
            # emitido en China iría al endpoint Global (y daría 401 seguro), así que la
            # dirección de la solicitud sigue primero a la región que emitió el token.
            authorized_region = kimi_code_oauth.authorized_region()
            if authorized_region:
                request_base_url = kimi_code_oauth.api_base_url(authorized_region)
                if base_url and authorized_region != kimi_code_oauth.resolve_region(
                    base_url
                ):
                    logger.warning(
                        f"kimi_code: la plataforma de API elegida no coincide con la "
                        f"región autorizada; se usa '{authorized_region}' para coincidir "
                        "con el token emitido"
                    )
            else:
                request_base_url = base_url or kimi_code_oauth.api_base_url(
                    kimi_code_oauth.resolve_region(base_url)
                )
            # Con el nombre del modelo vacío se usa por defecto kimi-for-coding: el
            # modelo de programación propio de las ventajas de la suscripción, para que
            # un id de modelo genérico fijado no deje de funcionar al cambiar las
            # ventajas.
            request_model = model_name or "kimi-for-coding"
            logger.info(f"solicitando chat completion a Kimi Code, modelo: {request_model}")
            # El tiempo de espera holgado del propio proveedor debe ajustarse al
            # presupuesto de la prueba en el escenario de prueba de conexión.
            effective_timeout = timeout_seconds
            if request_timeout is not None:
                effective_timeout = min(effective_timeout, request_timeout)
            client = sdk_client = OpenAI(
                api_key=access_token,
                base_url=request_base_url,
                timeout=effective_timeout,
                # El sistema antirrobots del servidor exige las mismas cabeceras de
                # identidad de dispositivo que la CLI.
                default_headers=kimi_code_oauth.api_headers(),
                **_sdk_transport_options(None, max_retries),
            )
            response = trace_sdk_call(
                llm_provider, base_url, client.chat.completions.create,
                model=request_model, messages=[{"role": "user", "content": prompt}]
            )
            if response and isinstance(response, ChatCompletion):
                return _extract_chat_completion_text(response, llm_provider)
            raise Exception(
                f'[{llm_provider}] devolvió una respuesta no válida: "{response}"'
            )

        if adapter == "modelscope":
            content = ""
            client = sdk_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                **_sdk_transport_options(request_timeout, max_retries),
            )
            response = sdk_stream = trace_sdk_call(
                llm_provider, base_url, client.chat.completions.create,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"enable_thinking": False},
                stream=True,
            )
            if response:
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content += delta.content

                if not content.strip():
                    raise ValueError("contenido vacío en la respuesta en streaming")

                return _normalize_text_response(content, llm_provider)
            else:
                raise Exception(f"[{llm_provider}] devolvió una respuesta vacía")

        client = sdk_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            **_sdk_transport_options(request_timeout, max_retries),
        )

        response = trace_sdk_call(
            llm_provider, base_url, client.chat.completions.create,
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )
        if response:
            if isinstance(response, ChatCompletion):
                return _extract_chat_completion_text(response, llm_provider)
            else:
                raise Exception(
                    f'[{llm_provider}] devolvió una respuesta no válida: "{response}"; revisa tu conexión '
                    f"de red e inténtalo de nuevo."
                )
        else:
            raise Exception(
                f"[{llm_provider}] devolvió una respuesta vacía; revisa tu conexión de red e inténtalo de nuevo."
            )

    except Exception as e:
        return f"Error: {_sanitize_error_message(e)}"

    finally:
        for resource in (sdk_stream, sdk_client):
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as cleanup_error:
                    logger.warning(
                        f"no se pudo cerrar el transporte del LLM: {type(cleanup_error).__name__}"
                    )


def test_connection() -> tuple[bool, str, float]:
    """
    Lanza una solicitud mínima con la configuración actual del proveedor para comprobar
    si la cadena de generación real funciona.

    La prueba de conexión reutiliza directamente `_generate_response()`, así que cubre
    la clave API, la URL base, el nombre del modelo y los campos propios del proveedor,
    pero no entra en la lógica de reintentos de la generación de guiones ni envía el
    tema o el texto del vídeo del usuario. La solicitud usa un tiempo de espera corto y
    acotado y desactiva los reintentos automáticos del SDK: si el endpoint no es
    accesible (fallo de proxy, cortafuegos o DNS) falla rápido y añade pistas para
    diagnosticar, en lugar de quedarse colgada varios minutos y dejar solo "Request
    timed out." (issue #1642). Devuelve, por orden, el estado de éxito, el mensaje de
    error y el tiempo que tardó la solicitud.
    """
    started_at = perf_counter()
    response = _generate_response(
        prompt="Reply with exactly: OK",
        request_timeout=LLM_CONNECTION_TEST_TIMEOUT_SECONDS,
        max_retries=LLM_CONNECTION_TEST_MAX_RETRIES,
    )
    elapsed = perf_counter() - started_at

    if not response:
        error_message = "el LLM devolvió una respuesta vacía"
        logger.warning(f"falló la prueba de conexión con el LLM: {error_message}")
        return False, error_message, elapsed

    if response.startswith("Error:"):
        error_message = _explain_connection_test_failure(
            response.removeprefix("Error:").strip()
        )
        logger.warning(f"falló la prueba de conexión con el LLM: {error_message}")
        return False, error_message, elapsed

    logger.info(f"prueba de conexión con el LLM correcta, tiempo: {elapsed:.2f}s")
    return True, "", elapsed


def _limit_script_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # La capa de API ya valida la longitud con Pydantic; aquí se sigue protegiendo para
    # que la WebUI o los servicios internos que llaman directamente a generate_script no
    # envíen al modelo prompts demasiado largos, evitando costes de tokens anómalos y
    # fallos de solicitud.
    logger.warning(
        f"{field_name} es demasiado largo y se truncará a {max_length} caracteres."
    )
    return value[:max_length]


def _normalize_script_paragraph_number(paragraph_number: int | None) -> int:
    try:
        value = int(paragraph_number or MIN_SCRIPT_PARAGRAPH_NUMBER)
    except (TypeError, ValueError):
        value = MIN_SCRIPT_PARAGRAPH_NUMBER

    if value < MIN_SCRIPT_PARAGRAPH_NUMBER or value > MAX_SCRIPT_PARAGRAPH_NUMBER:
        # Tanto la WebUI como la API limitan el rango; aquí se protegen las llamadas
        # internas, para que parámetros anómalos no disparen directamente el coste de
        # generación del LLM ni produzcan resultados vacíos.
        logger.warning(
            f"paragraph_number del guion está fuera de rango y se ajustará: {value}"
        )
        return max(MIN_SCRIPT_PARAGRAPH_NUMBER, min(value, MAX_SCRIPT_PARAGRAPH_NUMBER))

    return value


def build_script_prompt(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )

    # Las "reglas de generación del guion" y el "contexto en tiempo de ejecución" se
    # concatenan por separado. Así, aunque un usuario avanzado sobrescriba el system
    # prompt por defecto, no se pierden el tema del vídeo, el idioma ni el número de
    # párrafos, que deben acompañar a cada generación.
    prompt = custom_system_prompt or DEFAULT_SCRIPT_SYSTEM_PROMPT
    prompt += f"""

# Initialization:
- video subject: {video_subject}
- number of paragraphs: {paragraph_number}
""".rstrip()
    if language:
        prompt += f"\n- language: {language}"
    if video_script_prompt:
        prompt += f"""

# Additional User Requirements:
{video_script_prompt}
""".rstrip()

    return prompt


def generate_script(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
    app_config=None,
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )
    prompt = build_script_prompt(
        video_subject=video_subject,
        language=language,
        paragraph_number=paragraph_number,
        video_script_prompt=video_script_prompt,
        custom_system_prompt=custom_system_prompt,
    )
    final_script = ""
    logger.info(
        "generando el guion del vídeo: "
        f"subject={video_subject}, paragraph_number={paragraph_number}, "
        f"has_custom_prompt={bool(video_script_prompt.strip())}, "
        f"has_custom_system_prompt={bool(custom_system_prompt.strip())}"
    )

    def format_response(response):
        # Limpia el guion Elimina las marcas de énfasis y de encabezado sin cambiar
        # nombres en línea ni notas musicales como C#, F# y G#.
        response = response.replace("*", "")
        response = re.sub(
            r"(?m)^[ \t]{0,3}#{1,6}[ \t]+(.*?)[ \t]*$",
            lambda match: re.sub(r"[ \t]+#+[ \t]*$", "", match.group(1)),
            response,
        )

        # Las etiquetas de los enlaces son narración; se descartan los destinos antes de
        # quitar las acotaciones entre corchetes. Caben los componentes de URL
        # habituales entre paréntesis.
        response = re.sub(
            r"\[([^\]\n]+)\]\((?:[^()\n]|\([^()\n]*\))*\)",
            lambda match: match.group(1),
            response,
        )

        # Elimina la sintaxis markdown. Se usa el no codicioso .*? para que cada grupo
        # de corchetes/paréntesis se elimine de forma independiente; la forma codiciosa
        # se comería todo el texto entre el primer signo de apertura y el último de
        # cierre de la misma línea.
        response = re.sub(r"\[.*?\]", "", response)
        response = re.sub(r"\(.*?\)", "", response)

        # Divide el guion en párrafos
        paragraphs = response.split("\n\n")

        # Selecciona el número de párrafos indicado selected_paragraphs =
        # paragraphs[:paragraph_number]

        # Une los párrafos seleccionados en una sola cadena
        return "\n\n".join(paragraphs)

    for i in range(_max_retries):
        try:
            if app_config is None:
                response = _generate_response(prompt=prompt)
            else:
                response = _generate_response(prompt=prompt, app_config=app_config)
            if isinstance(response, str) and response.startswith("Error: "):
                # _generate_response devuelve los fallos del proveedor como texto. Pasar
                # ese texto haría que la tarea lo tratara como narración.
                raise ValueError(response)
            if response:
                candidate = format_response(response).strip()
            else:
                logging.error("GPT devolvió una respuesta vacía")
                candidate = ""

            # Algunos proveedores pueden devolver los errores de cuota como texto plano.
            if candidate and "当日额度已消耗完" in candidate:
                raise ValueError(candidate)

            if candidate:
                final_script = candidate
                break
        except Exception as e:
            logger.error(f"no se pudo generar el guion: {e}")

        if i < _max_retries - 1:
            logger.warning(f"no se pudo generar el guion del vídeo, reintentando... {i + 1}")
    if not final_script:
        logger.error("no se pudo generar el guion del vídeo tras los reintentos")
    else:
        logger.success(f"completado: \n{final_script}")
    return final_script.strip()


def _strip_code_fence(text: str) -> str:
    """Quita el bloque de código markdown que rodea una respuesta del LLM.

    Los proveedores que no son OpenAI (Claude, Gemini, …) suelen envolver la salida JSON
    en un bloque ```json … ``` aunque se les pida JSON en bruto. Quitarlo permite que el
    primer json.loads() tenga éxito en lugar de caer en la ruta de recuperación con
    regex (y registrar un aviso innecesario). Imita el tratamiento DOTALL que ya se usa
    en _parse_social_metadata().
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_embedded_json(text: str, opening: str):
    """Decodifica un valor JSON completo sin absorber texto o ejemplos posteriores."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text or ""):
        if character != opening:
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return value
    raise ValueError("la respuesta no contiene un valor JSON completo")


def generate_terms(
    video_subject: str,
    video_script: str,
    amount: int = 5,
    match_script_order: bool = False,
    app_config=None,
) -> List[str]:
    video_script = utils.remove_pause_tags(video_script or "").strip()
    if match_script_order:
        goal = (
            f"Generate {amount} chronological stock-video search terms that follow "
            "the order of topics in the video script."
        )
        ordering_rule = (
            "6. keep the terms in the same order as the script narration; "
            "earlier terms must describe earlier visual moments."
        )
        # En el modo de palabras clave ordenadas, el número de ejemplos debe coincidir
        # con amount, para que los 4 ejemplos fijos no confundan al modelo y un guion
        # largo devuelva solo unas pocas palabras clave, empeorando la cobertura del
        # material.
        example_terms = [
            "opening visual topic",
            *[f"script visual topic {index}" for index in range(2, max(amount, 1))],
            "final visual topic",
        ]
        output_example = json.dumps(example_terms[:amount], ensure_ascii=False)
    else:
        goal = (
            f"Generate {amount} search terms for stock videos, depending on the "
            "subject of a video."
        )
        ordering_rule = ""
        output_example = (
            '["search term 1", "search term 2", "search term 3",'
            '"search term 4", "search term 5"]'
        )

    prompt = f"""
# Role: Video Search Terms Generator

## Goals:
{goal}

## Constrains:
1. the search terms are to be returned as a json-array of strings.
2. each search term should consist of 1-3 words, always add the main subject of the video.
3. you must only return the json-array of strings. you must not return anything else. you must not return the script.
4. the search terms must be related to the subject of the video.
5. reply with english search terms only.
{ordering_rule}

## Output Example:
{output_example}

## Context:
### Video Subject
{video_subject}

### Video Script
{video_script}

Please note that you must use English for generating video search terms; Chinese is not accepted.
""".strip()

    logger.info(f"subject: {video_subject}, match_script_order: {match_script_order}")

    search_terms = []
    response = ""
    for i in range(_max_retries):
        search_terms = []
        try:
            if app_config is None:
                response = _generate_response(prompt)
            else:
                response = _generate_response(prompt, app_config=app_config)
            if response.startswith("Error: "):
                # El tipo de retorno público de generate_terms es List[str]. Si se
                # devolviera tal cual el texto de error del proveedor, como más adelante
                # solo se comprueba si está vacío, la cadena no vacía se tomaría por un
                # éxito, y el bucle de descarga de material recorrería el texto de error
                # carácter a carácter, generando solicitudes externas sin sentido. Aquí
                # se devuelve siempre una lista vacía, para que la capa de orquestación
                # de tareas termine la tarea de inmediato en el punto real del fallo.
                logger.error(f"no se pudieron generar los términos del vídeo: {response}")
                return []
            search_terms = json.loads(_strip_code_fence(response))
        except Exception as e:
            logger.warning(f"no se pudieron generar los términos del vídeo: {str(e)}")
            if response:
                try:
                    search_terms = _extract_embedded_json(response, "[")
                except ValueError as e:
                    logger.warning(f"no se pudieron generar los términos del vídeo: {str(e)}")

        # Aplica el mismo contrato al JSON directo y a la recuperación envuelta en
        # texto. Si no, un array no vacío de números u objetos llegaría a la búsqueda de
        # material.
        if not isinstance(search_terms, list) or not all(
            isinstance(term, str) for term in search_terms
        ):
            logger.error("la respuesta no es una lista de cadenas.")
            search_terms = []

        # El modelo puede ignorar la cantidad pedida o devolver cadenas en blanco. Se
        # hace cumplir el contrato del prompt antes de contactar con los proveedores de
        # material; los temas repetidos siguen teniendo sentido en el modo cronológico.
        search_terms = [term.strip() for term in search_terms if term.strip()][:amount]
        if search_terms:
            break
        if i < _max_retries - 1:
            logger.warning(f"no se pudieron generar los términos del vídeo, reintentando... {i + 1}")

    logger.success(f"completado: \n{search_terms}")
    return search_terms


# =============================================================================
# Metadatos para publicar en redes sociales
#
# Genera, a partir del tema y el guion del vídeo, el title, el caption y los hashtags
# que se suelen usar al publicar en plataformas de vídeo corto. Esta función solo
# reutiliza los proveedores LLM existentes, sin conectar con ningún servicio de
# publicación externo ni afectar a la cadena principal de generación de vídeo.
# =============================================================================

# Las plataformas prefieren longitudes de texto y números de hashtags distintos. Aquí se
# usan límites conservadores, para que quien llama no tenga que recortar después un
# contenido demasiado largo devuelto por el modelo.
SOCIAL_PLATFORMS = {
    "tiktok": {"title_max": 100, "caption_max": 2200, "hashtag_count": 5},
    "youtube_shorts": {"title_max": 100, "caption_max": 5000, "hashtag_count": 3},
    "instagram_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 8},
    "facebook_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 5},
}
DEFAULT_SOCIAL_PLATFORM = "tiktok"
DEFAULT_SOCIAL_LANGUAGE = "auto"
MAX_SOCIAL_SUBJECT_LENGTH = 500
MAX_SOCIAL_SCRIPT_LENGTH = 8000
MAX_SOCIAL_LANGUAGE_LENGTH = 64

SOCIAL_PLATFORM_LABELS = {
    "tiktok": "TikTok",
    "youtube_shorts": "YouTube Shorts",
    "instagram_reels": "Instagram Reels",
    "facebook_reels": "Facebook Reels",
}

# Etiquetas genéricas de respaldo cuando el LLM no está disponible. Aquí deliberadamente
# no se vinculan a un país o idioma concreto, para que la API devuelva una estructura
# utilizable en escenarios distintos como chino, inglés o vietnamita.
DEFAULT_SOCIAL_HASHTAGS = [
    "#shorts",
    "#viral",
    "#trending",
    "#fyp",
    "#video",
    "#reels",
    "#creator",
    "#content",
]


def _resolve_social_platform(platform: str | None) -> str:
    value = (platform or "").strip().lower()
    return value if value in SOCIAL_PLATFORMS else DEFAULT_SOCIAL_PLATFORM


def _normalize_social_language(language: str | None) -> str:
    value = (language or DEFAULT_SOCIAL_LANGUAGE).strip()
    if len(value) > MAX_SOCIAL_LANGUAGE_LENGTH:
        logger.warning(
            "el idioma de los metadatos sociales es demasiado largo y se truncará a "
            f"{MAX_SOCIAL_LANGUAGE_LENGTH} caracteres."
        )
        value = value[:MAX_SOCIAL_LANGUAGE_LENGTH]
    return value or DEFAULT_SOCIAL_LANGUAGE


def _limit_social_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # La capa de API limita la longitud; aquí se sigue protegiendo para que las llamadas
    # internas, o una futura llamada directa desde la WebUI, no envíen contenido
    # demasiado largo al modelo, evitando costes de tokens anómalos.
    logger.warning(
        f"{field_name} es demasiado largo y se truncará a {max_length} caracteres."
    )
    return value[:max_length]


def _social_language_instruction(language: str | None) -> str:
    language = _normalize_social_language(language)
    if language.lower() == DEFAULT_SOCIAL_LANGUAGE:
        return (
            "Use the same language as the video subject and script. If the subject "
            "and script use different languages, prefer the script language."
        )

    return f'Write "title" and "caption" in this language: {language}.'


def _clamp_text(text, max_length: int) -> str:
    value = ("" if text is None else str(text)).strip()
    if max_length and len(value) > max_length:
        return value[:max_length].rstrip()
    return value


def _normalize_hashtags(raw, count: int) -> List[str]:
    """
    Normaliza los hashtags devueltos por el LLM al formato `#tag`.

    El LLM puede devolver cadenas, arrays, grupos de palabras con espacios, etiquetas
    repetidas o contenido con puntuación. Limpiarlo todo aquí hace que la estructura de
    la respuesta de la interfaz sea estable, y evita etiquetas vacías, repetidas o
    hashtags con un formato poco habitual al publicar en las plataformas.
    """
    if isinstance(raw, str):
        # El texto CJK separa las etiquetas con ，、； y sin espacios; # empieza una
        # etiqueta nueva.
        candidates = re.split(r"[\s,，、;；#]+", raw)
    elif isinstance(raw, (list, tuple)):
        # Cada elemento del array se toma como una etiqueta completa, así que "du lich"
        # se convierte en "#dulich" y no se divide en dos etiquetas.
        candidates = [str(entry) for entry in raw]
    else:
        candidates = []

    seen = set()
    result: List[str] = []
    for item in candidates:
        # Idiomas como el hindi o el tailandés dependen de caracteres combinantes, que
        # no se pueden borrar sin más como haría \w. Pero los caracteres combinantes
        # deben ir unidos a un carácter conservado; si se borra un emoji, también hay
        # que descartar su selector de variante, o #☕️ dejaría una etiqueta con solo
        # caracteres invisibles y #coffee☕️ no podría deduplicarse con #coffee. Los
        # dígitos en sí se conservan, así que también hay que limpiar los selectores de
        # presentación de emoji y los combinantes de tecla, para que #1️⃣ coincida con
        # #1; los demás combinantes de idiomas y las variantes de glifos CJK se siguen
        # conservando según la regla de unión.
        characters = []
        can_attach_mark = False
        for character in item:
            if character in {"\ufe0e", "\ufe0f", "\u20e3"}:
                can_attach_mark = False
            elif character.isalnum() or character == "_":
                characters.append(character)
                can_attach_mark = True
            elif can_attach_mark and unicodedata.category(character).startswith("M"):
                characters.append(character)
            else:
                can_attach_mark = False
        tag = "".join(characters)
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(f"#{tag}")
        if count and len(result) >= count:
            break
    return result


def build_social_metadata_prompt(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> str:
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    platform = _resolve_social_platform(platform)
    spec = SOCIAL_PLATFORMS[platform]
    label = SOCIAL_PLATFORM_LABELS.get(platform, platform)
    language_instruction = _social_language_instruction(language)

    prompt = f"""
# Role: Short-Video Social Media Copywriter

## Goal
Write engaging publishing metadata for a short video that will be posted on {label}.

## Constraints
1. Respond ONLY with a single valid minified JSON object. No markdown, no code fences, no commentary.
2. The JSON must contain exactly these keys: "title", "caption", "hashtags".
3. "title": a catchy hook, at most {spec["title_max"]} characters.
4. "caption": an engaging description that ends with a call to action, at most {spec["caption_max"]} characters. Do not put hashtags inside the caption.
5. "hashtags": a JSON array of exactly {spec["hashtag_count"]} strings. Each must start with "#", contain no spaces, and be relevant to the topic and to {label}.
6. {language_instruction}

## Output Example
{{"title":"...","caption":"...","hashtags":["#example","#video"]}}

## Context
### Video Subject
{video_subject}

### Video Script
{video_script}
""".strip()
    return prompt


def _parse_social_metadata(response: str, platform: str) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]

    data = None
    try:
        data = json.loads(_strip_code_fence(response))
    except Exception:
        # Algunos modelos envuelven el JSON en un texto explicativo o un bloque de
        # código markdown. Quien llama a la API solo necesita una estructura estable,
        # así que aquí se intenta extraer el primer objeto JSON.
        data = _extract_embedded_json(response, "{")

    if not isinstance(data, dict):
        raise ValueError("la respuesta de metadatos sociales no es un objeto JSON")

    title = _clamp_text(data.get("title", ""), spec["title_max"])
    caption = _clamp_text(data.get("caption", ""), spec["caption_max"])
    hashtags = _normalize_hashtags(data.get("hashtags", []), spec["hashtag_count"])

    if not title and not caption:
        raise ValueError("a la respuesta de metadatos sociales le faltan tanto title como caption")

    return {"title": title, "caption": caption, "hashtags": hashtags}


def _fallback_social_metadata(
    video_subject: str, video_script: str, platform: str
) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]
    subject = (video_subject or "").strip()
    script = (video_script or "").strip()

    title = subject
    if not title and script:
        # En chino y japonés el final de frase normalmente no lleva espacio; en inglés
        # sigue exigiéndose un espacio en blanco, para no partir números como 2.0.
        title = re.split(r"(?<=[.!?])\s+|(?<=[。！？])\s*", script, maxsplit=1)[0]

    return {
        "title": _clamp_text(title, spec["title_max"]),
        "caption": _clamp_text(script or subject, spec["caption_max"]),
        "hashtags": _normalize_hashtags(DEFAULT_SOCIAL_HASHTAGS, spec["hashtag_count"]),
    }


def generate_social_metadata(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> dict:
    """
    Genera los metadatos de texto para publicar vídeos cortos.

    La estructura devuelta es siempre `{"title": str, "caption": str, "hashtags":
    List[str]}`. Si el LLM no está disponible o devuelve un formato anómalo, se degrada
    a un resultado heurístico genérico, para que quien llama a la API siempre obtenga
    una estructura de datos que se puede mostrar y editar antes de publicar.
    """
    platform = _resolve_social_platform(platform)
    language = _normalize_social_language(language)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    prompt = build_social_metadata_prompt(
        video_subject=video_subject,
        video_script=video_script,
        language=language,
        platform=platform,
    )
    logger.info(f"generando metadatos sociales: platform={platform}, language={language}")

    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if isinstance(response, str) and response.startswith("Error: "):
                logger.error(f"no se pudieron generar los metadatos sociales: {response}")
                break
            metadata = _parse_social_metadata(response, platform)
            logger.success(f"completado: \n{metadata}")
            return metadata
        except Exception as e:
            logger.warning(f"no se pudieron analizar los metadatos sociales: {str(e)}")

        if i < _max_retries - 1:
            logger.warning(
                f"no se pudieron generar los metadatos sociales, reintentando... {i + 1}"
            )

    logger.warning("se recurre a metadatos sociales heurísticos")
    return _fallback_social_metadata(video_subject, video_script, platform)


# El nicho y el tema son textos cortos de una sola línea. Se acotan para que un nicho
# pegado por error junto a un documento entero no genere un coste de tokens anómalo, y
# para que el tema devuelto siga cabiendo en video_subject.
MAX_NICHE_LENGTH = 500
MAX_GENERATED_SUBJECT_LENGTH = 200
# Número de temas recientes que se pasan al modelo como lista de exclusión. Más allá de
# esto el prompt crece sin que mejore la variedad.
MAX_RECENT_SUBJECTS = 30
_SUBJECT_LIST_MARKER_PATTERN = re.compile(r"^(?:[-*•>]+|\d+[.)])\s*")
_SUBJECT_WRAPPING_QUOTES = "\"'`“”‘’"
_SUBJECT_LABEL_PREFIXES = ("subject:", "tema:", "topic:", "título:", "titulo:")


def build_video_subject_prompt(
    niche: str,
    language: str = "",
    recent_subjects: List[str] | None = None,
) -> str:
    """Construye el prompt que convierte un nicho estable en el tema concreto del
    siguiente vídeo.
    """
    niche = _limit_social_text(niche, MAX_NICHE_LENGTH, "niche")
    language = _normalize_social_language(language)
    recent = [
        _clamp_text(subject, MAX_GENERATED_SUBJECT_LENGTH)
        for subject in (recent_subjects or [])
    ]
    recent = [subject for subject in dict.fromkeys(recent) if subject][
        -MAX_RECENT_SUBJECTS:
    ]

    if language.lower() == DEFAULT_SOCIAL_LANGUAGE:
        language_instruction = "Write the subject in the same language as the niche."
    else:
        language_instruction = f"Write the subject in this language: {language}."

    avoid_block = (
        "\n".join(f"- {subject}" for subject in recent)
        if recent
        else "(nothing published yet)"
    )

    return f"""
# Role: Short-Video Content Planner

## Goal
Pick the subject of the next short video for a channel that always posts about one niche.

## Constraints
1. Respond with the subject only: one single line of plain text, no quotes, no numbering, no commentary, no hashtags.
2. At most {MAX_GENERATED_SUBJECT_LENGTH} characters.
3. It must be a concrete, specific angle on the niche that a 30-60 second video can cover, not the niche restated.
4. It must be clearly different from every subject in "Already Published" below, in topic and not only in wording.
5. {language_instruction}

## Niche
{niche}

## Already Published
{avoid_block}
""".strip()


def _parse_video_subject(response: str) -> str:
    """Extrae un tema de una sola línea de la respuesta del modelo."""
    text = _strip_code_fence("" if response is None else str(response)).strip()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Algunos modelos prologan la respuesta o devuelven una lista numerada aunque se
        # haya pedido una sola línea. Se toma la primera línea útil y se le quitan los
        # adornos de lista, la etiqueta y las comillas envolventes.
        line = _SUBJECT_LIST_MARKER_PATTERN.sub("", line).strip()
        line = line.strip(_SUBJECT_WRAPPING_QUOTES).strip()
        lowered = line.lower()
        for prefix in _SUBJECT_LABEL_PREFIXES:
            if lowered.startswith(prefix):
                line = line.split(":", 1)[1].strip(_SUBJECT_WRAPPING_QUOTES + " ")
                break
        if line:
            return _clamp_text(line, MAX_GENERATED_SUBJECT_LENGTH)
    raise ValueError("la respuesta del tema del vídeo está vacía")


def generate_video_subject(
    niche: str,
    language: str = "",
    recent_subjects: List[str] | None = None,
) -> str:
    """
    Genera el tema del siguiente vídeo a partir de un nicho fijo.

    Lo usa la publicación automática diaria, que tiene el nicho configurado una sola vez
    y necesita un tema nuevo en cada ejecución sin que nadie lo escriba. Si el modelo no
    está disponible o responde en un formato que no se puede aprovechar se lanza
    ``ValueError``, para que quien planifica registre el fallo del hueco en lugar de
    publicar un vídeo sobre un tema vacío.
    """
    niche = _limit_social_text(niche, MAX_NICHE_LENGTH, "niche")
    if not niche:
        raise ValueError("el nicho no puede estar vacío")

    prompt = build_video_subject_prompt(
        niche=niche, language=language, recent_subjects=recent_subjects
    )
    logger.info(
        f"generando el tema del vídeo a partir del nicho: language={language or 'auto'}, "
        f"recent={len(recent_subjects or [])}"
    )

    last_error = "el modelo no devolvió ningún tema"
    for attempt in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if isinstance(response, str) and response.startswith("Error: "):
                # El marcador de error del proveedor no es un tema, y reintentarlo no lo
                # convierte en uno; se corta ya para no gastar más llamadas.
                last_error = response
                logger.error(f"no se pudo generar el tema del vídeo: {response}")
                break
            subject = _parse_video_subject(response)
            logger.success(f"tema del vídeo generado: {subject}")
            return subject
        except Exception as exc:
            last_error = str(exc)
            logger.warning(f"no se pudo analizar el tema del vídeo: {last_error}")

        if attempt < _max_retries - 1:
            logger.warning(
                f"no se pudo generar el tema del vídeo, reintentando... {attempt + 1}"
            )

    raise ValueError(
        f"no se pudo generar el tema del vídeo a partir del nicho: {last_error}"
    )


if __name__ == "__main__":
    video_subject = "¿Cuál es el sentido de la vida?"
    script = generate_script(
        video_subject=video_subject, language="zh-CN", paragraph_number=1
    )
    print("######################")
    print(script)
    search_terms = generate_terms(
        video_subject=video_subject, video_script=script, amount=5
    )
    print("######################")
    print(search_terms)
