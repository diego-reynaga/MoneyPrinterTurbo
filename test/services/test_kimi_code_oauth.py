"""Tests unitarios del módulo OAuth de kimi_code.

Cubren la máquina de estados de autorización de dispositivo: solicitud del código de
dispositivo, sondeo pending / slow_down / expired / denied, persistencia del token y
rotación del refresh, renovación transparente con margen de caducidad, serialización de
renovaciones concurrentes, guardado de credenciales en disco y borrado de credenciales.
La capa de red (requests.post) y el tiempo (time.time) se sustituyen por stubs y no se
accede a endpoints reales; salvo en los tests específicos de guardado en disco, el
guardado de configuración se sustituye siempre, para no tocar el config.toml real.
"""

import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import toml

from app.config import config
from app.services import kimi_code_oauth as oauth


def _response(payload, status_code=200):
    class _Resp:
        def __init__(self, data, code):
            self._data = data
            self.status_code = code
            self.text = str(data)

        def json(self):
            if isinstance(self._data, Exception):
                raise self._data
            return self._data

    return _Resp(payload, status_code)


class TestKimiCodeOAuth(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app.clear()
        # Por defecto se sustituye el guardado de configuración para que los tests no
        # escriban el config.toml real; los tests de guardado en disco restauran
        # temporalmente la implementación real.
        self.real_try_save_config = config.try_save_config
        save_patcher = patch.object(config, "try_save_config", return_value=True)
        self.mock_try_save_config = save_patcher.start()
        self.addCleanup(save_patcher.stop)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    # ---- request_device_code ----

    def test_request_device_code_uses_regional_endpoint(self):
        payload = {
            "device_code": "dc-1",
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://www.kimi.com/code/authorize_device",
            "verification_uri_complete": "https://www.kimi.com/code/authorize_device?user_code=ABCD-EFGH",
            "expires_in": 1800,
            "interval": 5,
        }
        with patch.object(
            oauth.requests, "post", return_value=_response(payload)
        ) as post:
            result = oauth.request_device_code("china")
        self.assertEqual(result["device_code"], "dc-1")
        self.assertEqual(result["user_code"], "ABCD-EFGH")
        url = post.call_args.args[0]
        self.assertTrue(url.startswith("https://auth.kimi.com/"))
        self.assertEqual(
            post.call_args.kwargs["data"]["client_id"], oauth.DEFAULT_CLIENT_ID
        )

    def test_request_device_code_global_region(self):
        payload = {
            "device_code": "dc-2",
            "user_code": "WXYZ-1234",
            "verification_uri_complete": "https://www.kimi.ai/code/authorize_device?user_code=WXYZ-1234",
        }
        with patch.object(
            oauth.requests, "post", return_value=_response(payload)
        ) as post:
            oauth.request_device_code("global")
        self.assertTrue(post.call_args.args[0].startswith("https://auth.kimi.ai/"))

    def test_request_device_code_requires_fields(self):
        with patch.object(
            oauth.requests, "post", return_value=_response({"device_code": "dc"})
        ):
            with self.assertRaises(ValueError):
                oauth.request_device_code("china")

    def test_request_device_code_non_json_raises_diagnosable(self):
        with patch.object(
            oauth.requests,
            "post",
            return_value=_response(ValueError("No JSON"), status_code=200),
        ):
            with self.assertRaisesRegex(ValueError, "no es JSON"):
                oauth.request_device_code("china")

    # ---- poll_token ----

    def _poll(self, payload):
        with patch.object(
            oauth.requests, "post", return_value=_response(payload)
        ):
            return oauth.poll_token("china", "dc")

    def test_poll_pending(self):
        self.assertEqual(
            self._poll({"error": "authorization_pending"}), {"pending": True}
        )

    def test_poll_slow_down_stays_pending(self):
        self.assertEqual(
            self._poll({"error": "slow_down"}),
            {"pending": True, "slow_down": True},
        )

    def test_poll_expired(self):
        self.assertEqual(self._poll({"error": "expired_token"}), {"expired": True})

    def test_poll_denied(self):
        self.assertEqual(self._poll({"error": "access_denied"}), {"denied": True})

    def test_poll_success_returns_payload(self):
        payload = {"access_token": "at-1", "refresh_token": "rt-1", "expires_in": 900}
        self.assertEqual(self._poll(payload), payload)

    def test_poll_unknown_error_raises(self):
        with self.assertRaisesRegex(ValueError, "bad thing"):
            self._poll({"error": "unknown", "error_description": "bad thing"})

    def test_poll_non_json_raises_diagnosable(self):
        with patch.object(
            oauth.requests,
            "post",
            return_value=_response(ValueError("No JSON"), status_code=502),
        ):
            with self.assertRaisesRegex(ValueError, "no es JSON"):
                oauth.poll_token("china", "dc")

    # ---- store_token_bundle / rotación ----

    def test_store_token_bundle_persists_fields(self):
        oauth.store_token_bundle(
            "china", {"access_token": "at-1", "refresh_token": "rt-1", "expires_in": 900}
        )
        self.assertEqual(config.app["kimi_code_access_token"], "at-1")
        self.assertEqual(config.app["kimi_code_refresh_token"], "rt-1")
        self.assertEqual(config.app["kimi_code_region"], "china")
        self.assertGreater(config.app["kimi_code_token_expires_at"], 0)

    def test_store_token_bundle_rotates_refresh_only_when_issued(self):
        oauth.store_token_bundle(
            "china", {"access_token": "at-1", "refresh_token": "rt-old", "expires_in": 900}
        )
        # Si la nueva carga útil no trae refresh token, se conserva el existente en
        # lugar de sobrescribirlo con un vacío.
        oauth.store_token_bundle(
            "china", {"access_token": "at-2", "expires_in": 900}
        )
        self.assertEqual(config.app["kimi_code_access_token"], "at-2")
        self.assertEqual(config.app["kimi_code_refresh_token"], "rt-old")
        oauth.store_token_bundle(
            "china", {"access_token": "at-3", "refresh_token": "rt-new", "expires_in": 900}
        )
        self.assertEqual(config.app["kimi_code_refresh_token"], "rt-new")

    def test_store_token_bundle_requests_config_save(self):
        # La ruta de llamada pura a la API no tiene un guardado posterior de la WebUI,
        # así que al escribir las credenciales hay que pedir explícitamente que se
        # guarden en disco.
        oauth.store_token_bundle(
            "china", {"access_token": "at-1", "refresh_token": "rt-1", "expires_in": 900}
        )
        self.mock_try_save_config.assert_called()

    def test_store_token_bundle_saved_to_disk_and_survives_reload(self):
        """Tras rotar el refresh token debe guardarse en disco de forma fiable: simulando
        un reinicio, el estado en disco debe seguir permitiendo renovar.
        """
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            original_cfg = dict(config._cfg)
            try:
                with (
                    # El archivo temporal y el de destino están en el mismo directorio,
                    # para garantizar que os.replace sea una sustitución atómica en el
                    # mismo disco.
                    patch.object(config, "root_dir", temp_dir),
                    patch.object(config, "config_file", str(config_path)),
                    patch.object(
                        config, "try_save_config", self.real_try_save_config
                    ),
                ):
                    oauth.store_token_bundle(
                        "china",
                        {
                            "access_token": "at-1",
                            "refresh_token": "rt-new",
                            "expires_in": 900,
                        },
                    )
                    on_disk = toml.load(str(config_path))
            finally:
                # save_config reescribe _cfg a nivel de módulo; se restaura para no
                # afectar a otros tests.
                config._cfg.clear()
                config._cfg.update(original_cfg)
        self.assertEqual(on_disk["app"]["kimi_code_access_token"], "at-1")
        self.assertEqual(on_disk["app"]["kimi_code_refresh_token"], "rt-new")
        # Simula un reinicio: vacía la configuración en memoria y la restaura solo desde
        # el contenido en disco.
        config.app.clear()
        config.app.update(on_disk["app"])
        self.assertTrue(oauth.is_authorized())
        self.assertEqual(oauth.get_valid_access_token(), "at-1")

    # ---- get_valid_access_token ----

    def _prime_tokens(self, expires_in=900):
        oauth.store_token_bundle(
            "china", {"access_token": "at-1", "refresh_token": "rt-1", "expires_in": expires_in}
        )

    def test_get_valid_access_token_returns_cached(self):
        self._prime_tokens()
        self.assertEqual(oauth.get_valid_access_token(), "at-1")

    def test_get_valid_access_token_refreshes_near_expiry(self):
        self._prime_tokens(expires_in=10)  # margen inferior a 60 s, se dispara la renovación
        refreshed = {"access_token": "at-2", "refresh_token": "rt-2", "expires_in": 900}
        with patch.object(
            oauth.requests, "post", return_value=_response(refreshed)
        ) as post:
            token = oauth.get_valid_access_token()
        self.assertEqual(token, "at-2")
        # El resultado de la renovación debe escribirse en la configuración, para que la
        # siguiente llamada no vuelva a refrescar.
        self.assertEqual(config.app["kimi_code_access_token"], "at-2")
        self.assertEqual(config.app["kimi_code_refresh_token"], "rt-2")
        self.assertEqual(post.call_args.kwargs["data"]["grant_type"], "refresh_token")

    def test_get_valid_access_token_expired_refresh(self):
        self._prime_tokens(expires_in=-1)
        refreshed = {"access_token": "at-2", "refresh_token": "rt-2", "expires_in": 900}
        with patch.object(oauth.requests, "post", return_value=_response(refreshed)):
            self.assertEqual(oauth.get_valid_access_token(), "at-2")

    def test_get_valid_access_token_requires_authorization(self):
        with self.assertRaisesRegex(ValueError, "no está autorizado"):
            oauth.get_valid_access_token()

    def test_get_valid_access_token_refresh_failure_raises(self):
        self._prime_tokens(expires_in=-1)
        with patch.object(
            oauth.requests,
            "post",
            return_value=_response({"error": "invalid_grant"}),
        ):
            with self.assertRaisesRegex(ValueError, "falló la renovación"):
                oauth.get_valid_access_token()

    def test_concurrent_refresh_is_serialized(self):
        """Las renovaciones concurrentes deben serializarse: el refresh token antiguo solo
        se puede canjear una vez y el resto de hilos reutilizan el resultado.
        """
        self._prime_tokens(expires_in=-1)
        refresh_calls = []

        def fake_refresh(region, refresh_token):
            refresh_calls.append(refresh_token)
            time.sleep(0.05)  # amplía la ventana de la condición de carrera
            return {"access_token": "at-new", "refresh_token": "rt-new", "expires_in": 900}

        results = []
        errors = []

        def worker():
            try:
                results.append(oauth.get_valid_access_token())
            except Exception as exc:  # pragma: no cover - facilita el diagnóstico si falla
                errors.append(exc)

        with patch.object(oauth, "_refresh", side_effect=fake_refresh):
            threads = [threading.Thread(target=worker) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(refresh_calls), 1)
        self.assertEqual(sorted(results), ["at-new"] * 4)
        self.assertEqual(config.app["kimi_code_refresh_token"], "rt-new")

    # ---- visibilidad mientras se tiene el bloqueo de configuración (escenario de
    #  tareas de vídeo concurrentes) ----

    def test_credentials_visible_while_config_lock_held(self):
        """Mientras una tarea de vídeo tiene el bloqueo, _set solo encola: un inicio de
        sesión recién completado debe ser visible de inmediato para las lecturas.
        """
        with config.runtime_config_lock():
            oauth.store_token_bundle(
                "china",
                {"access_token": "at-1", "refresh_token": "rt-1", "expires_in": 900},
            )
            # Las credenciales en cola también cuentan: no deben tomarse por no
            # autorizadas.
            self.assertTrue(oauth.is_authorized())
            self.assertEqual(oauth.get_valid_access_token(), "at-1")

    def test_successive_calls_reuse_queued_refresh_while_lock_held(self):
        """Mientras se tiene el bloqueo, el resultado del refresh sigue en cola: las
        llamadas siguientes no deben reutilizar el token antiguo ya rotado.
        """
        redeemed = []

        def rotating_refresh(region, refresh_token):
            if refresh_token in redeemed:
                # Simula que el servidor rechaza la reutilización del token antiguo tras
                # la rotación
                raise ValueError("kimi_code: token refresh failed: invalid_grant")
            redeemed.append(refresh_token)
            return {
                "access_token": f"at-{len(redeemed)}",
                "refresh_token": f"rt-{len(redeemed)}",
                "expires_in": 900,
            }

        with config.runtime_config_lock(), patch.object(
            oauth, "_refresh", side_effect=rotating_refresh
        ):
            self._prime_tokens(expires_in=-1)  # at-1 ya ha caducado; rt-1 se puede canjear una vez
            first = oauth.get_valid_access_token()
            second = oauth.get_valid_access_token()

        self.assertEqual(first, second)
        self.assertEqual(redeemed, ["rt-1"])

    # ---- carrera entre un refresh en curso y el borrado/cambio de credenciales ----

    def test_in_flight_refresh_discarded_after_disconnect(self):
        """El usuario desconecta mientras la respuesta del refresh está en curso: el
        resultado debe descartarse y el estado de autorización no puede revivir.
        """
        self._prime_tokens(expires_in=-1)

        def refresh_then_disconnect(region, refresh_token):
            # Simula que, antes de que llegue la respuesta, el usuario pulsó Disconnect
            # en otra sesión
            oauth.clear_credentials()
            return {"access_token": "at-zombie", "refresh_token": "rt-zombie", "expires_in": 900}

        with patch.object(oauth, "_refresh", side_effect=refresh_then_disconnect):
            with self.assertRaisesRegex(ValueError, "credenciales cambiaron"):
                oauth.get_valid_access_token()
        self.assertFalse(oauth.is_authorized())
        self.assertEqual(config.app["kimi_code_access_token"], "")

    def test_in_flight_refresh_discarded_after_credential_replacement(self):
        """Durante un refresh en curso, un nuevo inicio de sesión sustituye las
        credenciales: el resultado antiguo no puede sobrescribir las nuevas.
        """
        self._prime_tokens(expires_in=-1)

        def refresh_then_replace(region, refresh_token):
            # Simula que, antes de que llegue la respuesta, el usuario completó un nuevo
            # inicio de sesión
            oauth.store_token_bundle(
                "china",
                {"access_token": "at-new-login", "refresh_token": "rt-new-login", "expires_in": 900},
            )
            return {"access_token": "at-stale", "refresh_token": "rt-stale", "expires_in": 900}

        with patch.object(oauth, "_refresh", side_effect=refresh_then_replace):
            with self.assertRaisesRegex(ValueError, "credenciales cambiaron"):
                oauth.get_valid_access_token()
        self.assertEqual(config.app["kimi_code_access_token"], "at-new-login")
        self.assertEqual(config.app["kimi_code_refresh_token"], "rt-new-login")

    # ---- cabeceras de identidad (identidad real de la aplicación) ----

    def test_identity_headers_use_genuine_application_identity(self):
        """Las cabeceras de identidad deben ser las propias de MoneyPrinterTurbo, sin
        hacerse pasar por kimi_code_cli.
        """
        headers = oauth._identity_headers()
        self.assertEqual(headers["X-Msh-Platform"], "MoneyPrinterTurbo")
        self.assertEqual(headers["X-Msh-Version"], oauth.__version__)
        self.assertTrue(headers["User-Agent"].startswith("MoneyPrinterTurbo/"))
        self.assertNotIn("kimi_code_cli", headers.values())

        payload = {
            "device_code": "dc-1",
            "user_code": "ABCD-EFGH",
            "verification_uri_complete": "https://www.kimi.com/code/authorize_device?user_code=ABCD-EFGH",
        }
        with patch.object(
            oauth.requests, "post", return_value=_response(payload)
        ) as post:
            oauth.request_device_code("china")
        sent = post.call_args.kwargs["headers"]
        self.assertEqual(sent["X-Msh-Platform"], "MoneyPrinterTurbo")
        self.assertTrue(sent["User-Agent"].startswith("MoneyPrinterTurbo/"))

    # ---- contrato de errores de red (del que depende la WebUI) ----

    def test_request_device_code_network_error_propagates(self):
        """Las excepciones de red se lanzan tal cual como RequestException, y la WebUI
        muestra con ellas un error reintentable.
        """
        with patch.object(
            oauth.requests,
            "post",
            side_effect=oauth.requests.exceptions.ConnectTimeout("boom"),
        ):
            with self.assertRaises(oauth.requests.exceptions.RequestException):
                oauth.request_device_code("china")

    def test_poll_token_network_error_propagates(self):
        with patch.object(
            oauth.requests,
            "post",
            side_effect=oauth.requests.exceptions.ReadTimeout("boom"),
        ):
            with self.assertRaises(oauth.requests.exceptions.RequestException):
                oauth.poll_token("china", "dc")

    # ---- borrado / funciones auxiliares ----

    def test_clear_credentials(self):
        self._prime_tokens()
        oauth.clear_credentials()
        self.assertFalse(oauth.is_authorized())
        self.assertEqual(config.app["kimi_code_access_token"], "")

    def test_is_authorized_variants(self):
        self.assertFalse(oauth.is_authorized())
        self._prime_tokens()
        self.assertTrue(oauth.is_authorized())

    def test_resolve_region(self):
        self.assertEqual(oauth.resolve_region("https://api.kimi.ai/coding/v1"), "global")
        self.assertEqual(oauth.resolve_region("https://api.kimi.com/coding/v1"), "china")
        self.assertEqual(oauth.resolve_region(""), "china")

    def test_coerce_timeout(self):
        self.assertEqual(oauth.coerce_timeout(None), oauth.DEFAULT_TIMEOUT_SECONDS)
        self.assertEqual(oauth.coerce_timeout("120"), 120.0)
        with self.assertRaises(ValueError):
            oauth.coerce_timeout("abc")
        with self.assertRaises(ValueError):
            oauth.coerce_timeout("0")
        with self.assertRaises(ValueError):
            oauth.coerce_timeout("nan")
        with self.assertRaises(ValueError):
            oauth.coerce_timeout("inf")


if __name__ == "__main__":
    unittest.main()
