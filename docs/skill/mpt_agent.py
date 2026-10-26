#!/usr/bin/env python3
"""Instalación multiplataforma y generación de vídeo para la Skill de MoneyPrinterTurbo."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.error
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ARCHIVE_URL = (
    "https://github.com/harry0703/MoneyPrinterTurbo/archive/refs/heads/main.zip"
)
DEFAULT_ROOT = Path.home() / "MoneyPrinterTurbo"
DEFAULT_VOICE_NAME = "es-ES-ElviraNeural-Female"
DEFAULT_UV_LOCK_TIMEOUT = "1200"
NEEDS_INPUT_EXIT_CODE = 10
SUPPORTED_SOURCES = {
    "pexels",
    "pixabay",
    "coverr",
    "wavespeed",
    "volcengine_seedance",
    "ofox",
    "metaso_minimax",
    "muapi",
    "infistar",
    # Mantén esta lista alineada con ``_CLI_VIDEO_SOURCES`` de cli.py. Una fuente que
    # acepta la CLI no debe rechazarse aquí como no admitida.
    "openai_image",
    "local",
}
VOLCENGINE_ARK_API_KEY_URL = (
    "https://console.volcengine.com/ark/region:ark+cn-beijing/apikey"
)
OFOX_API_KEY_URL = "https://ofox.ai"
MUAPI_API_KEY_URL = "https://muapi.ai"
INFISTAR_VIDEO_API_KEY_URL = "https://infistar.cc/keys"
PEXELS_API_KEY_URL = "https://www.pexels.com/api/"
PEXELS_VALIDATION_URL = "https://api.pexels.com/v1/collections?per_page=1"
PEXELS_API_KEY_HELP_URL = (
    "https://help.pexels.com/hc/en-us/articles/"
    "900004904026-How-do-I-get-an-API-key"
)

# La lista recomendada se centra en los proveedores más usados. Cuando falta una clave
# del LLM, el helper muestra todas las opciones a la vez para evitar turnos extra.
RECOMMENDED_LLM_PROVIDERS = {
    "moonshot": (
        "Kimi / Moonshot AI",
        "https://platform.kimi.com?track_id=track-6eec1e56a4494e52adcaebbcbbefce59&aff=moneyprinterturbo",
    ),
    "openai": ("OpenAI", "https://platform.openai.com/api-keys"),
    "gemini": ("Google Gemini", "https://aistudio.google.com/app/apikey"),
    "deepseek": ("DeepSeek", "https://platform.deepseek.com/api_keys"),
    "volcengine": (
        "ByteDance VolcEngine Ark / Doubao",
        "https://www.volcengine.com/activity/ai618?utm_source=MoneyPrinterTurbo",
    ),
    "minimax": ("MiniMax", "https://platform.minimax.io/"),
    "mimo": (
        "Xiaomi MiMo",
        "https://platform.xiaomimimo.com/docs/zh-CN/quick-start/first-api-call",
    ),
}
# Proveedores que generan sin una clave API guardada en config.toml: Ollama habla con un
# servidor local, LiteLLM resuelve las credenciales con su propio entorno,
# ``claude_code`` consume la suscripción de Claude a través de la CLI ``claude`` con la
# sesión iniciada en local, y ``kimi_code`` consume el plan Kimi Code a través de las
# credenciales OAuth que el inicio de sesión de la WebUI guarda en config.toml. Mantén
# este conjunto alineado con las entradas ``requires_api_key=False`` de
# ``app/models/llm_provider.py``; pedir al usuario una clave que el proveedor nunca lee
# deja atascada la Skill.
KEYLESS_LLM_PROVIDERS = {"ollama", "litellm", "claude_code", "kimi_code"}
CUSTOM_OPENAI_PROVIDER = "oneapi"

# Los proveedores ocultos como Qwen, Azure y Grok siguen siendo utilizables si ya están
# seleccionados, pero no son candidatos de respaldo automático. Un endpoint genérico
# compatible con OpenAI totalmente configurado se puede reutilizar con seguridad.
ADDITIONAL_REUSABLE_PROVIDERS = (CUSTOM_OPENAI_PROVIDER,)


class SkillError(RuntimeError):
    """Un error de la Skill accionable que se puede informar sin traza."""


def log(message: str) -> None:
    """Vuelca un progreso conciso para que el agente sepa que el trabajo largo ha empezado."""
    print(f"[MoneyPrinterTurbo] {message}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Instala MoneyPrinterTurbo y genera un vídeo final a partir de un tema."
    )
    parser.add_argument("--subject", required=True, help="tema del vídeo")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"directorio de instalación de MoneyPrinterTurbo (por defecto: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "cli_args",
        nargs=argparse.REMAINDER,
        help="argumentos adicionales de la CLI de MoneyPrinterTurbo colocados después de --",
    )
    args = parser.parse_args(argv)
    args.subject = args.subject.strip()
    if not args.subject:
        parser.error("--subject no puede estar vacío")
    if args.cli_args and args.cli_args[0] == "--":
        args.cli_args = args.cli_args[1:]
    return args


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    """Rechaza las entradas ZIP que escaparían del directorio temporal de extracción."""
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != destination and destination not in target.parents:
            raise SkillError(f"el archivo del proyecto contiene una ruta insegura: {member.filename}")
    archive.extractall(destination)


def ensure_project(root: Path) -> None:
    """Reutiliza un proyecto existente o lo instala desde el archivo oficial de GitHub."""
    root = root.expanduser().resolve()
    if (root / "cli.py").is_file() and (root / "config.example.toml").is_file():
        log(f"se usa el proyecto existente: {root}")
        return
    if root.exists() and any(root.iterdir()):
        raise SkillError(f"el directorio de instalación existe pero no es un proyecto válido: {root}")

    root.parent.mkdir(parents=True, exist_ok=True)
    log(f"primera instalación: descargando el proyecto oficial en {root}")
    with tempfile.TemporaryDirectory(prefix="mpt-install-") as temp_dir_value:
        temp_dir = Path(temp_dir_value)
        archive_path = temp_dir / "MoneyPrinterTurbo.zip"
        request = urllib.request.Request(
            PROJECT_ARCHIVE_URL,
            headers={"User-Agent": "MoneyPrinterTurbo-Agent-Skill"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            # Transmite el archivo por streaming para no tener en memoria una segunda
            # copia completa.
            with archive_path.open("wb") as archive_file:
                shutil.copyfileobj(response, archive_file)
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, temp_dir)

        candidates = [
            path
            for path in temp_dir.iterdir()
            if path.is_dir() and (path / "cli.py").is_file()
        ]
        if len(candidates) != 1:
            raise SkillError("la descarga terminó pero no se encontró ningún proyecto válido de MoneyPrinterTurbo")
        if root.exists():
            root.rmdir()
        shutil.move(str(candidates[0]), str(root))
    log("descarga del proyecto completada")


def ensure_config(root: Path) -> Path:
    """Crea la configuración inicial sin sobrescribir un archivo existente."""
    config_path = root / "config.toml"
    if not config_path.exists():
        shutil.copy2(root / "config.example.toml", config_path)
        log(f"archivo de configuración creado: {config_path}")
    return config_path


def _plain_config_value(text: str, key: str) -> str:
    """Lee los ajustes de la aplicación con la misma sintaxis TOML que el cargador del
    backend.
    """
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        # La configuración puede contener credenciales; no se repite su texto original.
        raise SkillError("la configuración no es un TOML válido") from exc
    settings = document.get("app", document)
    if not isinstance(settings, dict):
        raise SkillError("los ajustes app de la configuración deben ser una tabla TOML")
    value = settings.get(key, "")
    # Los consumidores existentes de readiness/listas esperan una cadena sin comillas o
    # un array JSON.
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _replace_config_value(text: str, key: str, value: object) -> str:
    """Sustituye un campo activo conservando la estructura de la configuración."""
    pattern = re.compile(rf"(?m)^({re.escape(key)}\s*=\s*).*$")
    match = pattern.search(text)
    if match is None:
        raise SkillError(f"no se encontró el campo de configuración en config.toml: {key}")
    value_start = match.start() + len(match.group(1))
    value_end = match.end()
    while True:
        try:
            tomllib.loads("replacement = " + text[value_start:value_end])
            break
        except tomllib.TOMLDecodeError as exc:
            next_line = text.find("\n", value_end + 1)
            if value_end == len(text):
                raise SkillError("el campo de configuración no es un TOML válido") from exc
            value_end = len(text) if next_line < 0 else next_line
    encoded = json.dumps(value, ensure_ascii=False)
    return text[:value_start] + encoded + text[value_end:]


def _has_configured_value(value: str) -> bool:
    """Trata las cadenas vacías y los arrays de claves con solo espacios como sin
    configurar.
    """
    if not value:
        return False
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return bool(value.strip())
    if isinstance(parsed, list):
        return any(str(item).strip() for item in parsed)
    return bool(str(parsed).strip())


def _parse_string_list(value: str) -> list[str]:
    """Analiza una lista de cadenas configurada quitando los vacíos y los duplicados."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in parsed if str(item).strip()))


def apply_environment_config(config_path: Path) -> None:
    """Escribe las credenciales proporcionadas registrando solo los nombres de los campos."""
    provider = os.environ.get("MPT_LLM_PROVIDER", "").strip().lower()
    if provider == "openai_compatible":
        provider = CUSTOM_OPENAI_PROVIDER
    llm_key = os.environ.get("MPT_LLM_API_KEY", "").strip()
    base_url = os.environ.get("MPT_LLM_BASE_URL", "").strip()
    model_name = os.environ.get("MPT_LLM_MODEL_NAME", "").strip()
    pexels_key = os.environ.get("MPT_PEXELS_API_KEY", "").strip()
    seedance_key = os.environ.get("MPT_VOLCENGINE_ARK_API_KEY", "").strip()
    ofox_key = os.environ.get("MPT_OFOX_API_KEY", "").strip()
    metaso_minimax_key = os.environ.get(
        "MPT_METASO_MINIMAX_API_KEY", ""
    ).strip()
    muapi_key = os.environ.get("MPT_MUAPI_API_KEY", "").strip()
    infistar_video_key = os.environ.get("MPT_INFISTAR_VIDEO_API_KEY", "").strip()
    if not any(
        (
            provider,
            llm_key,
            base_url,
            model_name,
            pexels_key,
            seedance_key,
            ofox_key,
            metaso_minimax_key,
            muapi_key,
            infistar_video_key,
        )
    ):
        return

    text = config_path.read_text(encoding="utf-8")
    current_provider = (_plain_config_value(text, "llm_provider") or "moonshot").lower()
    provider = provider or current_provider
    changes: list[str] = []
    if os.environ.get("MPT_LLM_PROVIDER", "").strip():
        text = _replace_config_value(text, "llm_provider", provider)
        changes.append("llm_provider")
    if llm_key:
        text = _replace_config_value(text, f"{provider}_api_key", llm_key)
        changes.append(f"{provider}_api_key")
    if base_url:
        text = _replace_config_value(text, f"{provider}_base_url", base_url)
        changes.append(f"{provider}_base_url")
    if model_name:
        text = _replace_config_value(text, f"{provider}_model_name", model_name)
        changes.append(f"{provider}_model_name")
    if pexels_key:
        text = _replace_config_value(text, "pexels_api_keys", [pexels_key])
        changes.append("pexels_api_keys")
    if seedance_key:
        text = _replace_config_value(
            text, "volcengine_seedance_api_key", seedance_key
        )
        changes.append("volcengine_seedance_api_key")
    if ofox_key:
        text = _replace_config_value(text, "ofox_api_key", ofox_key)
        changes.append("ofox_api_key")
    if metaso_minimax_key:
        text = _replace_config_value(
            text, "metaso_minimax_api_key", metaso_minimax_key
        )
        changes.append("metaso_minimax_api_key")
    if muapi_key:
        text = _replace_config_value(text, "muapi_api_key", muapi_key)
        changes.append("muapi_api_key")
    if infistar_video_key:
        text = _replace_config_value(text, "infistar_video_api_key", infistar_video_key)
        changes.append("infistar_video_api_key")
    config_path.write_text(text, encoding="utf-8")
    log("updated configuration fields: " + ", ".join(changes))


def _provider_is_ready(text: str, provider: str) -> bool:
    """Devuelve si un proveedor tiene configuración suficiente para generar."""
    if provider == "kimi_code":
        # No necesita clave API, pero sí las credenciales OAuth que escribe la WebUI
        # tras «Iniciar sesión con Kimi»; sin credenciales no puede considerarse listo,
        # o la generación solo informaría del error de autorización en tiempo de
        # ejecución.
        return any(
            _has_configured_value(_plain_config_value(text, field))
            for field in ("kimi_code_refresh_token", "kimi_code_access_token")
        )
    if provider in KEYLESS_LLM_PROVIDERS:
        return True
    if not _has_configured_value(
        _plain_config_value(text, f"{provider}_api_key")
    ):
        return False
    if provider == CUSTOM_OPENAI_PROVIDER:
        return all(
            _has_configured_value(_plain_config_value(text, f"{provider}_{suffix}"))
            for suffix in ("base_url", "model_name")
        )
    return True


def reuse_existing_llm_provider(config_path: Path) -> str:
    """
    Reutiliza las credenciales del LLM existentes antes de pedir otra clave al usuario.

    Mantiene el proveedor actual cuando está listo. Si no, recorre los proveedores
    recomendados configurados en el orden de la interfaz y actualiza ``llm_provider``.
    Los valores de las credenciales se inspeccionan en memoria y nunca se registran.
    """
    text = config_path.read_text(encoding="utf-8")
    current_provider = (_plain_config_value(text, "llm_provider") or "moonshot").lower()
    if _provider_is_ready(text, current_provider):
        return current_provider

    reusable_providers = (
        *RECOMMENDED_LLM_PROVIDERS,
        *ADDITIONAL_REUSABLE_PROVIDERS,
    )
    for provider in reusable_providers:
        if _provider_is_ready(text, provider):
            text = _replace_config_value(text, "llm_provider", provider)
            config_path.write_text(text, encoding="utf-8")
            log(f"reusing configured LLM provider: {provider}")
            return provider
    return current_provider


def selected_video_source(cli_args: list[str]) -> str:
    """Lee la fuente de material efectiva a partir de los argumentos de la CLI reenviados."""
    source = "pexels"
    for index, item in enumerate(cli_args):
        if item == "--video-source" and index + 1 < len(cli_args):
            source = cli_args[index + 1].strip().lower()
        if item.startswith("--video-source="):
            source = item.split("=", 1)[1].strip().lower()
    return source


def has_cli_option(cli_args: list[str], option: str) -> bool:
    """Devuelve si los argumentos reenviados fijan explícitamente una opción de la CLI."""
    return any(item == option or item.startswith(f"{option}=") for item in cli_args)


def _last_cli_value(cli_args: list[str], option: str) -> str:
    value = ""
    for index, item in enumerate(cli_args):
        if item == option and index + 1 < len(cli_args):
            value = cli_args[index + 1]
        elif item.startswith(f"{option}="):
            value = item.split("=", 1)[1]
    return value.strip()


def _requires_llm(cli_args: list[str]) -> bool:
    # Coincide con task.generate_script y con el atajo de términos del material local.
    return not _last_cli_value(cli_args, "--video-script") or (
        selected_video_source(cli_args) != "local"
        and not _last_cli_value(cli_args, "--video-terms")
    )


def missing_config(config_path: Path, cli_args: list[str]) -> tuple[str, list[str]]:
    """Devuelve el proveedor activo y solo los campos que necesita esta ejecución."""
    text = config_path.read_text(encoding="utf-8")
    provider = (_plain_config_value(text, "llm_provider") or "moonshot").lower()
    missing: list[str] = []
    needs_llm = _requires_llm(cli_args)
    if needs_llm and provider not in KEYLESS_LLM_PROVIDERS and not _has_configured_value(
        _plain_config_value(text, f"{provider}_api_key")
    ):
        missing.append(f"{provider}_api_key")
    if needs_llm and provider == "kimi_code" and not _provider_is_ready(text, provider):
        # No hay clave que rellenar; lo que falta es el inicio de sesión en la WebUI, no
        # un campo de configuración.
        missing.append("kimi_code_sign_in")
    if needs_llm and provider == CUSTOM_OPENAI_PROVIDER:
        for suffix in ("base_url", "model_name"):
            field = f"{provider}_{suffix}"
            if not _has_configured_value(_plain_config_value(text, field)):
                missing.append(field)

    source = selected_video_source(cli_args)
    if source not in SUPPORTED_SOURCES:
        raise SkillError(f"fuente de vídeo no admitida: {source}")
    if source == "local" and not has_cli_option(cli_args, "--video-materials"):
        raise SkillError(
            "La fuente de vídeo local necesita --video-materials apuntando a un "
            "directorio de archivos de vídeo"
        )
    if source == "volcengine_seedance":
        # Se mantiene exactamente la misma prioridad de credenciales que el proveedor en
        # tiempo de ejecución, para que no pase que la comprobación previa de la Skill
        # aprueba y luego el programa principal lee otra clave. ARK_API_KEY tiene un
        # significado demasiado amplio y ya no se admite explícitamente.
        value = (
            _plain_config_value(text, "volcengine_seedance_api_key").strip()
            or os.environ.get("VOLCENGINE_ARK_API_KEY", "").strip()
            or _plain_config_value(text, "volcengine_api_key").strip()
        )
        if not _has_configured_value(value):
            missing.append("volcengine_seedance_api_key")
        if not has_cli_option(cli_args, "--confirm-seedance-charge"):
            missing.append("confirm_seedance_charge")
    elif source == "wavespeed":
        # Las credenciales de WaveSpeed en tiempo de ejecución solo vienen de
        # wavespeed_api_keys, sin respaldo por variable de entorno, y aquí se mantiene
        # el mismo criterio. Las fuentes de generación de pago por uso también deben
        # confirmar el cobro explícitamente.
        if not _has_configured_value(
            _plain_config_value(text, "wavespeed_api_keys")
        ):
            missing.append("wavespeed_api_keys")
        if not has_cli_option(cli_args, "--confirm-wavespeed-charge"):
            missing.append("confirm_wavespeed_charge")
    elif source == "ofox":
        # Se mantiene exactamente la misma prioridad de credenciales que el proveedor en
        # tiempo de ejecución: primero la clave de configuración y después la variable
        # de entorno OFOX_API_KEY, de significado claro.
        value = (
            _plain_config_value(text, "ofox_api_key").strip()
            or os.environ.get("OFOX_API_KEY", "").strip()
        )
        if not _has_configured_value(value):
            missing.append("ofox_api_key")
        if not has_cli_option(cli_args, "--confirm-ofox-charge"):
            missing.append("confirm_ofox_charge")
    elif source == "metaso_minimax":
        # La clave de Metaso y la clave oficial del LLM de MiniMax no son
        # intercambiables; la Skill debe seguir la misma prioridad de configuración
        # independiente que el tiempo de ejecución, y no darla por disponible solo
        # porque ya esté configurado minimax_api_key.
        value = (
            _plain_config_value(text, "metaso_minimax_api_key").strip()
            or os.environ.get("METASO_MINIMAX_API_KEY", "").strip()
        )
        if not _has_configured_value(value):
            missing.append("metaso_minimax_api_key")
        if not has_cli_option(cli_args, "--confirm-metaso-minimax-charge"):
            missing.append("confirm_metaso_minimax_charge")
    elif source == "muapi":
        value = (
            _plain_config_value(text, "muapi_api_key").strip()
            or os.environ.get("MUAPI_API_KEY", "").strip()
        )
        if not _has_configured_value(value):
            missing.append("muapi_api_key")
        if not has_cli_option(cli_args, "--confirm-muapi-charge"):
            missing.append("confirm_muapi_charge")
    elif source == "infistar":
        # Las credenciales de vídeo son independientes del LLM de la misma marca, y la
        # comprobación previa debe coincidir con infistar_video.get_api_key.
        value = (
            _plain_config_value(text, "infistar_video_api_key").strip()
            or os.environ.get("INFISTAR_VIDEO_API_KEY", "").strip()
        )
        if not _has_configured_value(value):
            missing.append("infistar_video_api_key")
        if not has_cli_option(cli_args, "--confirm-infistar-charge"):
            missing.append("confirm_infistar_charge")
    elif source == "openai_image":
        # Coincide con is_openai_image_enabled() en tiempo de ejecución: la fuente de
        # material de texto a imagen solo exige el endpoint y el nombre del modelo. Un
        # gateway ComfyUI/SD totalmente local permite no configurar clave API, así que
        # aquí no se puede tratar openai_image_api_keys como obligatorio.
        for field in ("openai_image_base_url", "openai_image_model"):
            if not _has_configured_value(_plain_config_value(text, field)):
                missing.append(field)
    elif source != "local":
        value = _plain_config_value(text, f"{source}_api_keys")
        if not _has_configured_value(value):
            missing.append(f"{source}_api_keys")
    return provider, missing


def report_missing_config(provider: str, missing: list[str]) -> int:
    """Indica al agente exactamente qué credenciales hay que pedir."""
    print("MPT_NEEDS_INPUT")
    print(f"LLM_PROVIDER={provider}")
    for field in missing:
        print(f"MISSING={field}")
    if any(
        field.endswith("_api_key")
        and field not in {
            "volcengine_seedance_api_key",
            "ofox_api_key",
            "metaso_minimax_api_key",
            "muapi_api_key",
            "infistar_video_api_key",
        }
        for field in missing
    ):
        print("LLM_PROVIDER_OPTIONS_BEGIN")
        for provider_id, (label, url) in RECOMMENDED_LLM_PROVIDERS.items():
            print(f"LLM_PROVIDER_OPTION={provider_id}|{label}|{url}")
        print(
            "LLM_PROVIDER_OPTION=oneapi|Otro proveedor compatible con OpenAI|"
            "necesita una clave API, la URL base de la API y el nombre del modelo"
        )
        print("LLM_PROVIDER_OPTIONS_END")
    if any(field.startswith(f"{CUSTOM_OPENAI_PROVIDER}_") for field in missing):
        print(
            "OPENAI_COMPATIBLE_REQUIRED="
            "clave API, URL base de la API, nombre del modelo"
        )
    if any(field.startswith("openai_image_") for field in missing):
        print(
            "OPENAI_IMAGE_REQUIRED="
            "openai_image_base_url, openai_image_model (la clave API es opcional "
            "para gateways locales)"
        )
    if "pexels_api_keys" in missing:
        print(f"PEXELS_API_KEY_URL={PEXELS_API_KEY_URL}")
        print(f"PEXELS_API_KEY_HELP_URL={PEXELS_API_KEY_HELP_URL}")
    if "kimi_code_sign_in" in missing:
        print(
            "KIMI_CODE_SIGN_IN_REQUIRED="
            "abre Ajustes -> Configuración LLM e inicia sesión con Kimi"
        )
    if "volcengine_seedance_api_key" in missing:
        print(f"VOLCENGINE_ARK_API_KEY_URL={VOLCENGINE_ARK_API_KEY_URL}")
        print("VOLCENGINE_ARK_API_KEY_ENV=MPT_VOLCENGINE_ARK_API_KEY")
    if "confirm_seedance_charge" in missing:
        print("SEEDANCE_CHARGE_CONFIRMATION_REQUIRED=--confirm-seedance-charge")
    if "confirm_wavespeed_charge" in missing:
        print("WAVESPEED_CHARGE_CONFIRMATION_REQUIRED=--confirm-wavespeed-charge")
    if "ofox_api_key" in missing:
        print(f"OFOX_API_KEY_URL={OFOX_API_KEY_URL}")
        print("OFOX_API_KEY_ENV=MPT_OFOX_API_KEY")
    if "confirm_ofox_charge" in missing:
        print("OFOX_CHARGE_CONFIRMATION_REQUIRED=--confirm-ofox-charge")
    if "metaso_minimax_api_key" in missing:
        print("METASO_MINIMAX_API_KEY_ENV=MPT_METASO_MINIMAX_API_KEY")
    if "confirm_metaso_minimax_charge" in missing:
        print(
            "METASO_MINIMAX_CHARGE_CONFIRMATION_REQUIRED="
            "--confirm-metaso-minimax-charge"
        )
    if "muapi_api_key" in missing:
        print(f"MUAPI_API_KEY_URL={MUAPI_API_KEY_URL}")
        print("MUAPI_API_KEY_ENV=MPT_MUAPI_API_KEY")
    if "confirm_muapi_charge" in missing:
        print("MUAPI_CHARGE_CONFIRMATION_REQUIRED=--confirm-muapi-charge")
    if "infistar_video_api_key" in missing:
        print(f"INFISTAR_VIDEO_API_KEY_URL={INFISTAR_VIDEO_API_KEY_URL}")
        print("INFISTAR_VIDEO_API_KEY_ENV=MPT_INFISTAR_VIDEO_API_KEY")
    if "confirm_infistar_charge" in missing:
        print("INFISTAR_CHARGE_CONFIRMATION_REQUIRED=--confirm-infistar-charge")
    print("Pide solo los valores indicados, define las variables de entorno y vuelve a ejecutar el mismo comando.")
    return NEEDS_INPUT_EXIT_CODE


def report_invalid_pexels_config() -> int:
    """Solo pide una nueva clave de Pexels cuando se rechazan todas las claves
    configuradas.
    """
    print("MPT_NEEDS_INPUT")
    print("INVALID=pexels_api_keys")
    print(f"PEXELS_API_KEY_URL={PEXELS_API_KEY_URL}")
    print(f"PEXELS_API_KEY_HELP_URL={PEXELS_API_KEY_HELP_URL}")
    print("Se rechazaron todas las claves API de Pexels configuradas o no están disponibles. Proporciona una clave nueva.")
    return NEEDS_INPUT_EXIT_CODE


def _validate_pexels_key(api_key: str) -> str:
    """
    Devuelve ``valid``, ``rejected`` o ``unknown`` para una clave de Pexels.

    Las respuestas HTTP 401 y 403 rechazan las credenciales. Los límites de uso y los
    errores de red y del servidor devuelven unknown para que los fallos temporales no
    puedan eliminar una clave.
    """
    # Las búsquedas curated y popular pueden dar en una caché pública y devolver 200 sin
    # autorización válida. My Collections es propia de la cuenta, exige autenticación y
    # sigue devolviendo 200 con una lista de colecciones vacía.
    request = urllib.request.Request(
        PEXELS_VALIDATION_URL,
        headers={
            "Authorization": api_key,
            "User-Agent": "MoneyPrinterTurbo-Agent-Skill",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return "valid" if 200 <= response.status < 300 else "unknown"
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            return "rejected"
        return "unknown"
    except (TimeoutError, urllib.error.URLError):
        return "unknown"


def validate_pexels_config(config_path: Path, cli_args: list[str]) -> bool:
    """
    Valida todas las claves de Pexels que usa la fuente de material por defecto.

    El código posterior elige al azar entre las claves configuradas. Conservar claves
    rechazadas puede provocar respuestas 401 intermitentes y resultados de material que
    faltan. Si se verifica al menos una clave, solo se eliminan las rechazadas
    explícitamente. Si la validación es imposible por un fallo de red pasajero, se
    conserva la configuración original.
    """
    if selected_video_source(cli_args) != "pexels":
        return True

    text = config_path.read_text(encoding="utf-8")
    keys = _parse_string_list(_plain_config_value(text, "pexels_api_keys"))
    if not keys:
        return False

    valid_keys: list[str] = []
    retained_keys: list[str] = []
    rejected_count = 0
    unknown_count = 0
    for api_key in keys:
        status = _validate_pexels_key(api_key)
        if status == "valid":
            valid_keys.append(api_key)
            retained_keys.append(api_key)
        elif status == "rejected":
            rejected_count += 1
        else:
            unknown_count += 1
            retained_keys.append(api_key)

    if valid_keys:
        if retained_keys != keys:
            text = _replace_config_value(text, "pexels_api_keys", retained_keys)
            config_path.write_text(text, encoding="utf-8")
        log(
            "validación de claves de Pexels completada: "
            f"valid={len(valid_keys)}, rejected={rejected_count}, "
            f"unknown={unknown_count}"
        )
        return True
    if unknown_count:
        log("no se pudieron verificar las claves de Pexels por un error de red o del servicio; se mantiene la configuración existente")
        return True

    log(f"falló la validación de claves de Pexels: las {rejected_count} claves configuradas no se pueden usar")
    return False


def result_manifest_path(root: Path) -> Path:
    return root / ".agent-logs" / "moneyprinterturbo-video" / "latest-result.json"


def write_result_manifest(root: Path, payload: dict[str, object]) -> Path:
    """
    Escribe de forma atómica el archivo de resultado estable para los agentes que no
    pueden esperar.

    El archivo solo contiene el estado de la tarea y las rutas de los resultados, nunca
    el contenido de la configuración, credenciales ni logs completos.
    """
    result_path = result_manifest_path(root)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    unique_suffix = str(uuid.uuid4()).replace("-", "")
    temp_path = result_path.with_name(
        f".{result_path.name}.{os.getpid()}.{unique_suffix}.tmp"
    )
    temp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp_path.replace(result_path)
    return result_path.resolve()


def run_checked(command: list[str], *, cwd: Path) -> None:
    """Alarga la espera del bloqueo de caché de la primera instalación y, si falla,
    conserva el final del log y pistas para diagnosticar.
    """
    log("instalando o verificando las dependencias del proyecto con uv")
    # Varios procesos de instalación pueden compartir una caché de wheels grande; solo
    # se fija el valor por defecto en el subproceso, conservando las sobrescrituras del
    # usuario y el resto de variables de entorno, sin cambiar el entorno del proceso
    # actual ni del subproceso de generación de vídeo.
    sync_env = os.environ.copy()
    sync_env.setdefault("UV_LOCK_TIMEOUT", DEFAULT_UV_LOCK_TIMEOUT)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=sync_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        output_tail = (result.stdout or "").splitlines()[-30:]
        if output_tail:
            print("\n".join(output_tail), file=sys.stderr)
        output = (result.stdout or "").lower()
        if "lock" in output and ("timeout" in output or "timed out" in output):
            print(
                "uv agotó el tiempo esperando un bloqueo de la caché. Espera a que terminen otras instalaciones "
                "de uv y reinténtalo. Si hace falta, aumenta UV_LOCK_TIMEOUT "
                "(requiere uv 0.9.4 o superior).",
                file=sys.stderr,
            )
        raise SkillError(f"falló la instalación de dependencias con el código de salida {result.returncode}")


def generate_video(
    root: Path,
    subject: str,
    cli_args: list[str],
) -> tuple[list[Path], Path, Path, Path]:
    """Ejecuta una tarea de la CLI rastreable y devuelve solo sus archivos de vídeo
    finales.
    """
    uv = shutil.which("uv")
    if not uv:
        raise SkillError("no se encontró uv; vuelve a abrir el terminal o añade uv al PATH")
    run_checked([uv, "sync", "--frozen"], cwd=root)

    task_id = str(uuid.uuid4())
    task_dir = root / "storage" / "tasks" / task_id
    log_dir = root / ".agent-logs" / "moneyprinterturbo-video"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run-{task_id}.log"
    write_result_manifest(
        root,
        {
            "status": "running",
            "subject": subject,
            "task_id": task_id,
            "task_dir": str(task_dir.resolve()),
            "log_file": str(log_path.resolve()),
            "video_files": [],
        },
    )
    voice_args = (
        []
        if has_cli_option(cli_args, "--voice-name")
        else ["--voice-name", DEFAULT_VOICE_NAME]
    )
    command = [
        uv,
        "run",
        "python",
        "cli.py",
        *cli_args,
        "--video-subject",
        subject,
        "--task-id",
        task_id,
        # Las versiones antiguas de la CLI dejan voice_name vacío y fallan en Edge TTS
        # con ``Invalid voice ''``. Se proporciona una voz china estable salvo que el
        # usuario haya elegido explícitamente otra voz.
        *voice_args,
        # Una solicitud de la Skill debe producir un vídeo terminado. Se fuerza la etapa
        # final para que las opciones reenviadas no puedan detenerse en el guion, el
        # audio o el material.
        "--stop-at",
        "video",
    ]
    log(f"iniciando la generación del vídeo, ID de tarea: {task_id}")
    log(f"log completo de la generación: {log_path}")
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                command,
                cwd=root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    except OSError as exc:
        error = f"no se pudo iniciar la generación del vídeo: {exc}; log: {log_path}"
        write_result_manifest(
            root,
            {
                "status": "failed",
                "subject": subject,
                "task_id": task_id,
                "task_dir": str(task_dir.resolve()),
                "log_file": str(log_path.resolve()),
                "video_files": [],
                "error": error,
            },
        )
        raise SkillError(error) from exc
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        if tail:
            print("\n".join(tail), file=sys.stderr)
        error = (
            f"la generación del vídeo falló con el código de salida {result.returncode}; "
            f"log: {log_path}"
        )
        write_result_manifest(
            root,
            {
                "status": "failed",
                "subject": subject,
                "task_id": task_id,
                "task_dir": str(task_dir.resolve()),
                "log_file": str(log_path.resolve()),
                "video_files": [],
                "error": error,
            },
        )
        raise SkillError(error)

    videos = sorted(
        path.resolve()
        for path in task_dir.glob("final-*.mp4")
        if path.is_file() and path.stat().st_size > 0
    )
    if not videos:
        error = f"la generación terminó sin un MP4 final válido; log: {log_path}"
        write_result_manifest(
            root,
            {
                "status": "failed",
                "subject": subject,
                "task_id": task_id,
                "task_dir": str(task_dir.resolve()),
                "log_file": str(log_path.resolve()),
                "video_files": [],
                "error": error,
            },
        )
        raise SkillError(error)
    result_path = write_result_manifest(
        root,
        {
            "status": "completed",
            "subject": subject,
            "task_id": task_id,
            "task_dir": str(task_dir.resolve()),
            "log_file": str(log_path.resolve()),
            "video_files": [str(video) for video in videos],
        },
    )
    return videos, task_dir.resolve(), log_path.resolve(), result_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    try:
        ensure_project(root)
        config_path = ensure_config(root)
        apply_environment_config(config_path)
        reuse_existing_llm_provider(config_path)
        provider, missing = missing_config(config_path, args.cli_args)
        if missing:
            write_result_manifest(
                root,
                {
                    "status": "needs_input",
                    "subject": args.subject,
                    "missing": missing,
                },
            )
            return report_missing_config(provider, missing)
        if not validate_pexels_config(config_path, args.cli_args):
            write_result_manifest(
                root,
                {
                    "status": "needs_input",
                    "subject": args.subject,
                    "invalid": ["pexels_api_keys"],
                },
            )
            return report_invalid_pexels_config()
        videos, task_dir, log_path, result_path = generate_video(
            root, args.subject, args.cli_args
        )
    except (OSError, SkillError, urllib.error.URLError, zipfile.BadZipFile) as exc:
        print(f"MPT_ERROR={exc}", file=sys.stderr)
        return 1

    print("MPT_RESULT")
    for video in videos:
        print(f"VIDEO_FILE={video}")
    print(f"TASK_DIR={task_dir}")
    print(f"LOG_FILE={log_path}")
    print(f"RESULT_FILE={result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
