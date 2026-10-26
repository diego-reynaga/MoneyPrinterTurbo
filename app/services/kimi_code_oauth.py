"""
Soporte del flujo de dispositivo OAuth de Kimi Code (suscripción de Kimi).

Los suscriptores del plan Kimi Code no pagan por token: se autentican una vez en el
navegador y las aplicaciones de terceros llaman al endpoint de programación compatible
con OpenAI con un token Bearer. Este módulo implementa la concesión de autorización de
dispositivo que usa la CLI de Kimi Code:

  1. POST {auth}/api/oauth/device_authorization  -> device_code + user_code
  2. el usuario abre {site}/code/authorize_device?user_code=... y aprueba
  3. POST {auth}/api/oauth/token                 -> access_token (+ refresh_token)
  4. llamar a {api}/usages, {api}/messages con el access token

Se admiten ambas regiones: China (auth.kimi.com / api.kimi.com) y Global (auth.kimi.ai /
api.kimi.ai). Los paquetes de tokens se persisten en la configuración de la aplicación
para que un refresh sobreviva a los reinicios; los access tokens caducan a los 15
minutos y se renuevan de forma transparente con el refresh token guardado.

Las solicitudes se identifican de verdad como MoneyPrinterTurbo (cabeceras X-Msh-* y
User-Agent), siguiendo la guía de Kimi para terceros de mantener la identidad real del
cliente de la herramienta, con la misma forma que usa CLIProxyAPI en producción.
"""

import math
import threading
import time
import uuid

import requests
from loguru import logger

from app.utils.request_logging import trace_http_call

from app import __version__
from app.config import config

# ID de cliente OAuth público que incluye la CLI de Kimi Code. Hoy no existe registro de
# clientes de terceros, así que todas las integraciones con el flujo de dispositivo usan
# este id (CLIProxyAPI y muchas otras hacen lo mismo); lo que identifica a la aplicación
# son las cabeceras de identidad reales de abajo, no el client id. Más adelante, cuando
# se emita, se podrá configurar un client id propio de MoneyPrinterTurbo con el campo de
# configuración `kimi_code_client_id`.
DEFAULT_CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"

DEFAULT_TIMEOUT_SECONDS = 300.0

_REGIONS = {
    "china": {
        "auth_base": "https://auth.kimi.com",
        "api_base": "https://api.kimi.com/coding/v1",
        "site_base": "https://www.kimi.com",
    },
    "global": {
        "auth_base": "https://auth.kimi.ai",
        "api_base": "https://api.kimi.ai/coding/v1",
        "site_base": "https://www.kimi.ai",
    },
}

# claves de config.app usadas para persistir el estado OAuth
_K_REGION = "kimi_code_region"
_K_CLIENT_ID = "kimi_code_client_id"
_K_DEVICE_ID = "kimi_code_device_id"
_K_ACCESS = "kimi_code_access_token"
_K_REFRESH = "kimi_code_refresh_token"
_K_EXPIRES_AT = "kimi_code_token_expires_at"  # segundos unix (float)
_K_USER_CODE = "kimi_code_last_user_code"  # solo para diagnóstico

# Se refresca 60 segundos antes de la caducidad real para que una solicitud lenta nunca
# empiece con un token que muera a mitad de camino.
_EXPIRY_MARGIN_SECONDS = 60.0

# Kimi rota el refresh token, y un mismo token antiguo solo puede renovarse una vez con
# éxito. Las solicitudes concurrentes deben renovar en serie; si no, el segundo hilo
# usaría el token antiguo ya rotado y obtendría invalid_grant.
_refresh_lock = threading.Lock()
# Las solicitudes de red no tienen el bloqueo de credenciales; la verificación de a
# quién pertenece la respuesta y la escritura deben hacerse en la misma sección crítica.
_credentials_lock = threading.RLock()
_authorization_generation = 0


def authorization_generation() -> int:
    """Devuelve la versión de inicio de sesión/desconexión, para que las respuestas de
    autorización en curso comprueben si siguen perteneciendo al estado de cuenta actual.
    """
    with _credentials_lock:
        return _authorization_generation


def _cfg(key, default=""):
    # La ruta de lectura debe fusionar las actualizaciones no bloqueantes en cola:
    # mientras una tarea de vídeo tiene el bloqueo de configuración, las escrituras de
    # _set solo entran en la cola, y leer directamente config.app no vería el inicio de
    # sesión o el refresh recién completados, de modo que se tomaría por no autorizado
    # algo ya autorizado, o se renovaría otra vez con el refresh token antiguo ya
    # rotado.
    snapshot = config.snapshot_config_with_pending(config.app)
    value = snapshot.get(key, default)
    return value if value is not None else default


def _set(key, value):
    # Actualización no bloqueante en cola; se aplica de inmediato si ninguna tarea de
    # vídeo tiene el bloqueo de configuración y, si no, cuando la tarea en ejecución lo
    # libera.
    config.update_config_nonblocking(config.app, key, value)


def client_id() -> str:
    return str(_cfg(_K_CLIENT_ID, "") or DEFAULT_CLIENT_ID)


def device_id() -> str:
    existing = str(_cfg(_K_DEVICE_ID, "")).strip()
    if existing:
        return existing
    new_id = str(uuid.uuid4())
    _set(_K_DEVICE_ID, new_id)
    return new_id


def region_ids() -> list[str]:
    return list(_REGIONS.keys())


def resolve_region(base_url: str | None = None) -> str:
    """Convierte una URL base de API (o un valor vacío) de nuevo en un id de región."""
    url = (base_url or "").strip().lower()
    if "kimi.ai" in url:
        return "global"
    return "china"


def api_base_url(region: str) -> str:
    return _REGIONS[region]["api_base"]


def site_base_url(region: str) -> str:
    return _REGIONS[region]["site_base"]


def _identity_headers() -> dict:
    # Identidad real (siguiendo la guía de Kimi para terceros y la práctica de
    # CLIProxyAPI en producción): la plataforma y la versión son las de
    # MoneyPrinterTurbo, sin hacerse pasar por kimi_code_cli.
    return {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "User-Agent": f"MoneyPrinterTurbo/{__version__}",
        "X-Msh-Platform": "MoneyPrinterTurbo",
        "X-Msh-Version": __version__,
        "X-Msh-Device-Name": "MoneyPrinterTurbo",
        "X-Msh-Device-Model": "MoneyPrinterTurbo",
        "X-Msh-Os-Version": "any",
        "X-Msh-Device-Id": device_id(),
    }


def api_headers() -> dict:
    """Cabeceras de identidad para las llamadas al propio endpoint de programación."""
    headers = _identity_headers()
    headers.pop("Content-Type", None)
    return headers


def request_device_code(region: str) -> dict:
    """Inicia un flujo de autorización de dispositivo. Devuelve la carga útil del proveedor
    en bruto.
    """
    if region not in _REGIONS:
        raise ValueError(f"kimi_code: región desconocida '{region}'")
    url = f"{_REGIONS[region]['auth_base']}/api/oauth/device_authorization"
    response = trace_http_call(
        "POST", requests.post,
        url,
        headers=_identity_headers(),
        data={"client_id": client_id()},
        timeout=30,
    )
    if response.status_code != 200:
        raise ValueError(
            f"kimi_code: falló la solicitud del código de dispositivo "
            f"({response.status_code}): {response.text[:300]}"
        )
    data = _json_or_raise(response, "device authorization")
    for field in ("device_code", "user_code", "verification_uri_complete"):
        if not data.get(field):
            raise ValueError(f"kimi_code: a la autorización de dispositivo le falta '{field}'")
    _set(_K_USER_CODE, data["user_code"])
    return data


def _json_or_raise(response, what: str) -> dict:
    try:
        return response.json()
    except ValueError:
        raise ValueError(
            f"kimi_code: {what} devolvió una respuesta que no es JSON "
            f"(HTTP {response.status_code}): {response.text[:200]}"
        ) from None


def poll_token(region: str, device_code: str) -> dict:
    """
    Un sondeo del endpoint de token. Devuelve:
      - {"pending": True}                     — el usuario aún no ha aprobado
      - {"pending": True, "slow_down": True}  — el servidor pide sondear más despacio
      - {"expired": True}                     — el código de dispositivo caducó; volver
        a empezar
      - {"denied": True}                      — el usuario rechazó la solicitud
      - la carga útil del token               — aprobado
    """
    url = f"{_REGIONS[region]['auth_base']}/api/oauth/token"
    response = trace_http_call(
        "POST", requests.post,
        url,
        headers=_identity_headers(),
        data={
            "client_id": client_id(),
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        },
        timeout=30,
    )
    data = _json_or_raise(response, "token endpoint")
    if data.get("access_token"):
        return data
    error = data.get("error", "")
    if error == "authorization_pending":
        return {"pending": True}
    if error == "slow_down":
        # RFC 8628 §3.5: se sigue sondeando, pero se aumenta el intervalo; es una
        # indicación de ritmo, no un fallo.
        return {"pending": True, "slow_down": True}
    if error == "expired_token":
        return {"expired": True}
    if error == "access_denied":
        return {"denied": True}
    raise ValueError(
        f"kimi_code: falló la solicitud del token: "
        f"{data.get('error_description') or data.get('error') or data}"
    )


def _refresh(region: str, refresh_token: str) -> dict:
    url = f"{_REGIONS[region]['auth_base']}/api/oauth/token"
    response = trace_http_call(
        "POST", requests.post,
        url,
        headers=_identity_headers(),
        data={
            "client_id": client_id(),
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    data = _json_or_raise(response, "token refresh")
    if not data.get("access_token"):
        raise ValueError(
            f"kimi_code: falló la renovación del token: "
            f"{data.get('error_description') or data.get('error') or data}"
        )
    return data


def store_token_bundle(
    region: str,
    payload: dict,
    *,
    expected_generation: int | None = None,
    expected_refresh_token: str | None = None,
) -> None:
    """Comprueba de forma atómica a quién pertenece la respuesta y guarda las credenciales,
    para que una respuesta antigua no deshaga una desconexión ni sobrescriba un inicio
    de sesión nuevo.
    """
    global _authorization_generation
    access = payload["access_token"]
    expires_at = time.time() + float(payload.get("expires_in") or 900)
    with _credentials_lock:
        if (
            expected_generation is not None
            and expected_generation != _authorization_generation
        ) or (
            expected_refresh_token is not None
            and str(_cfg(_K_REFRESH, "")).strip() != expected_refresh_token
        ):
            raise ValueError(
                "kimi_code: las credenciales cambiaron durante la autorización o la renovación; "
                "vuelve a iniciar sesión si hace falta."
            )
        _set(_K_REGION, region)
        _set(_K_ACCESS, access)
        if payload.get("refresh_token"):
            _set(_K_REFRESH, payload["refresh_token"])
        _set(_K_EXPIRES_AT, expires_at)
        if expected_refresh_token is None:
            _authorization_generation += 1
        # El guardado no espera al bloqueo de configuración de la tarea de vídeo, y las
        # credenciales en cola se pueden leer de inmediato.
        config.try_save_config()


def clear_credentials() -> None:
    global _authorization_generation
    with _credentials_lock:
        _authorization_generation += 1
        for key in (_K_ACCESS, _K_REFRESH, _K_EXPIRES_AT, _K_USER_CODE):
            _set(key, "")
        config.try_save_config()


def is_authorized() -> bool:
    with _credentials_lock:
        return bool(
            str(_cfg(_K_REFRESH, "")).strip() or str(_cfg(_K_ACCESS, "")).strip()
        )


def authorized_region() -> str:
    with _credentials_lock:
        return str(_cfg(_K_REGION, "") or "")


def get_valid_access_token() -> str:
    """Devuelve un token válido; renueva en serie y rechaza las respuestas antiguas que
    lleguen después de un inicio de sesión/desconexión.
    """
    with _credentials_lock:
        access = str(_cfg(_K_ACCESS, "")).strip()
        expires_at = float(_cfg(_K_EXPIRES_AT, 0) or 0)
        if access and time.time() < expires_at - _EXPIRY_MARGIN_SECONDS:
            return access
    with _refresh_lock:
        # Durante la espera en cola puede que otra solicitud ya haya renovado, así que
        # se vuelve a leer el estado completo de las credenciales.
        with _credentials_lock:
            access = str(_cfg(_K_ACCESS, "")).strip()
            expires_at = float(_cfg(_K_EXPIRES_AT, 0) or 0)
            if access and time.time() < expires_at - _EXPIRY_MARGIN_SECONDS:
                return access
            refresh = str(_cfg(_K_REFRESH, "")).strip()
            if not refresh:
                raise ValueError(
                    "kimi_code: todavía no está autorizado; completa primero el inicio de sesión "
                    "en Kimi desde Ajustes."
                )
            region = str(_cfg(_K_REGION, "") or "china")
            generation = _authorization_generation
        logger.info("kimi_code: renovando el token de acceso")
        payload = _refresh(region, refresh)
        store_token_bundle(
            region,
            payload,
            expected_generation=generation,
            expected_refresh_token=refresh,
        )
        return payload["access_token"]


def coerce_timeout(value) -> float:
    if value in (None, ""):
        return DEFAULT_TIMEOUT_SECONDS
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"timeout debe ser un número, se recibió '{value}'") from None
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"timeout debe ser un número finito positivo, se recibió '{value}'")
    return parsed
