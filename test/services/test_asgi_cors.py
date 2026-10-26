import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI, File, UploadFile
from fastapi.testclient import TestClient

from app import asgi


class TestASGICORS(unittest.TestCase):
    """Verifica los valores por defecto de peticiones entre orígenes del navegador y la
    configuración de compatibilidad explícita, para no volver a introducir un CORS
    abierto.
    """

    @staticmethod
    def _create_client(allowed_origins: list[str]) -> TestClient:
        """Construye una aplicación con solo la ruta de sonda, aislando las tareas de
        negocio y las llamadas a API externas.
        """

        application = FastAPI()

        @application.get("/probe")
        def probe():
            return {"status": "ok"}

        asgi.configure_browser_access(application, allowed_origins)
        return TestClient(application)

    def test_origin_parser_trims_values_and_ignores_empty_items(self):
        """Los espacios y las comas finales de la variable de entorno no deben romper la
        coincidencia de orígenes válidos.
        """

        origins = asgi.parse_cors_allowed_origins(
            " https://a.example,https://b.example, ,"
        )

        self.assertEqual(
            origins,
            ["https://a.example", "https://b.example"],
        )
        self.assertEqual(asgi.parse_cors_allowed_origins(""), [])
        self.assertEqual(asgi.parse_cors_allowed_origins(None), [])

    def test_origin_parser_normalizes_every_item_to_origin_header_shape(self):
        """Las opciones deben reducirse a la forma canónica de la cabecera Origin; si no,
        la lista blanca deja de funcionar en silencio.
        """

        origins = asgi.parse_cors_allowed_origins(
            "https://frontend.example/,"
            "HTTPS://Admin.Example,"
            "https://frontend.example/,"
            "localhost:3000,"
            "ftp://files.example,"
            "*"
        )

        # Las barras finales, las diferencias de mayúsculas y los duplicados se
        # normalizan al mismo origen; los nombres de host sin esquema y los esquemas que
        # no son http/https nunca pueden aparecer en una cabecera Origin, así que se
        # descartan.
        self.assertEqual(
            origins,
            ["https://frontend.example", "https://admin.example", "*"],
        )

    def test_malformed_entry_is_dropped_without_losing_valid_origins(self):
        """Una entrada mal formada solo puede descartarse a sí misma dejando un aviso; no
        debe interrumpir el análisis de toda la configuración.
        """

        with patch.object(asgi, "logger") as mocked_logger:
            origins = asgi.parse_cors_allowed_origins(
                "https://valid.example,"
                "https://[::1,"
                "https://frontend.example:bad,"
                "https://frontend.example:99999,"
                "https://second.example"
            )

        # Los literales IPv6 mal formados y los puertos no válidos nunca pueden aparecer
        # en una cabecera Origin. Sus excepciones de análisis deben contenerse aquí:
        # esta función se llama al importar el módulo, y si la excepción escapa toda la
        # API no arranca.
        self.assertEqual(
            origins,
            ["https://valid.example", "https://second.example"],
        )
        self.assertEqual(mocked_logger.warning.call_count, 3)

    def test_explicit_default_port_is_folded_into_the_origin(self):
        """Escribir explícitamente el puerto por defecto debe reducirse al Origin que envía
        realmente el navegador.
        """

        origins = asgi.parse_cors_allowed_origins(
            "https://secure.example:443,"
            "http://plain.example:80,"
            "https://padded.example:0443,"
            "https://custom.example:8443,"
            "https://[::1]:3000"
        )

        # :443 / :0443 / :80 equivalen al puerto por defecto del protocolo, que el
        # navegador omite al serializar el Origin; los puertos no predeterminados deben
        # conservarse, y los literales IPv6 deben conservarse con sus corchetes.
        self.assertEqual(
            origins,
            [
                "https://secure.example",
                "http://plain.example",
                "https://padded.example",
                "https://custom.example:8443",
                "https://[::1]:3000",
            ],
        )

    def test_address_bar_style_origin_admits_the_trusted_frontend(self):
        """La forma con barra final copiada de la barra de direcciones debe dar el mismo
        resultado de acceso del frontend que la forma canónica.
        """

        trusted_origin = "https://frontend.example"
        client = self._create_client(
            asgi.parse_cors_allowed_origins("https://frontend.example/")
        )

        response = client.get("/probe", headers={"Origin": trusted_origin})
        preflight = client.options(
            "/probe",
            headers={
                "Origin": trusted_origin,
                "Access-Control-Request-Method": "GET",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"], trusted_origin
        )
        self.assertEqual(preflight.status_code, 200)
        self.assertEqual(
            preflight.headers["access-control-allow-origin"], trusted_origin
        )

    def test_default_port_configuration_admits_the_trusted_frontend(self):
        """Si la configuración incluye el puerto por defecto, un Origin del navegador sin
        puerto debe seguir permitiéndose.
        """

        trusted_origin = "https://frontend.example"
        client = self._create_client(
            asgi.parse_cors_allowed_origins("https://frontend.example:443")
        )

        response = client.get("/probe", headers={"Origin": trusted_origin})
        preflight = client.options(
            "/probe",
            headers={
                "Origin": trusted_origin,
                "Access-Control-Request-Method": "GET",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"], trusted_origin
        )
        self.assertEqual(preflight.status_code, 200)
        self.assertEqual(
            preflight.headers["access-control-allow-origin"], trusted_origin
        )

    def test_empty_configuration_keeps_browser_same_origin_policy(self):
        """Sin lista blanca configurada, las páginas de terceros no pueden leer respuestas
        ni pasar el preflight.
        """

        client = self._create_client([])
        origin = "https://evil.attacker.example"

        response = client.get("/probe", headers={"Origin": origin})
        preflight = client.options(
            "/probe",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )

        self.assertEqual(response.status_code, 403)
        self.assertNotIn("access-control-allow-origin", response.headers)
        self.assertNotIn("access-control-allow-credentials", response.headers)
        self.assertEqual(preflight.status_code, 403)
        self.assertNotIn("access-control-allow-origin", preflight.headers)

    def test_same_origin_and_server_clients_remain_compatible(self):
        """Los navegadores del mismo origen y los clientes de servidor que no envían Origin
        deben seguir accediendo con normalidad.
        """

        client = self._create_client([])

        same_origin = client.get(
            "/probe",
            headers={"Origin": "http://testserver"},
        )
        server_client = client.get("/probe")

        self.assertEqual(same_origin.status_code, 200)
        self.assertEqual(server_client.status_code, 200)

    def test_explicit_origin_allows_only_the_trusted_frontend(self):
        """Un frontend web independiente configurado explícitamente puede acceder, y los
        demás orígenes deben seguir rechazándose.
        """

        trusted_origin = "https://frontend.example"
        untrusted_origin = "https://evil.attacker.example"
        client = self._create_client([trusted_origin])

        trusted = client.get("/probe", headers={"Origin": trusted_origin})
        untrusted = client.get("/probe", headers={"Origin": untrusted_origin})
        trusted_preflight = client.options(
            "/probe",
            headers={
                "Origin": trusted_origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        untrusted_preflight = client.options(
            "/probe",
            headers={
                "Origin": untrusted_origin,
                "Access-Control-Request-Method": "GET",
            },
        )

        self.assertEqual(trusted.headers["access-control-allow-origin"], trusted_origin)
        self.assertEqual(trusted.headers["access-control-allow-credentials"], "true")
        self.assertNotIn("access-control-allow-origin", untrusted.headers)
        self.assertEqual(trusted_preflight.status_code, 200)
        self.assertEqual(untrusted_preflight.status_code, 400)

    def test_trusted_origin_can_request_private_network_access(self):
        """La lista blanca exacta debe admitir el preflight adicional de una página remota
        que accede a una API local o de la red local.
        """

        trusted_origin = "https://frontend.example"
        client = self._create_client([trusted_origin])

        preflight = client.options(
            "/probe",
            headers={
                "Origin": trusted_origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Private-Network": "true",
            },
        )

        self.assertEqual(preflight.status_code, 200)
        self.assertEqual(
            preflight.headers["access-control-allow-private-network"],
            "true",
        )

    def test_explicit_wildcard_does_not_enable_credentials(self):
        """El comodín explícito mantiene la compatibilidad, pero no debe volver a formar
        una combinación que refleje el Origin.
        """

        client = self._create_client(["*"])
        origin = "https://frontend.example"

        response = client.get("/probe", headers={"Origin": origin})
        preflight = client.options(
            "/probe",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )

        self.assertEqual(response.headers["access-control-allow-origin"], "*")
        self.assertNotIn("access-control-allow-credentials", response.headers)
        self.assertEqual(preflight.status_code, 200)
        self.assertEqual(preflight.headers["access-control-allow-origin"], "*")
        self.assertNotIn("access-control-allow-credentials", preflight.headers)

    def test_untrusted_multipart_request_is_rejected_before_side_effect(self):
        """Las solicitudes multipart que no necesitan preflight también deben devolver 403
        antes de llegar a la función de subida.
        """

        application = FastAPI()
        save_upload = Mock(return_value="stored.mp3")

        @application.post("/upload")
        def upload(file: UploadFile = File(...)):
            return {"file": save_upload(file.filename)}

        asgi.configure_browser_access(application, [])
        client = TestClient(application)

        response = client.post(
            "/upload",
            headers={"Origin": "https://evil.attacker.example"},
            files={
                "file": (
                    "attack.mp3",
                    b"attacker-controlled",
                    "audio/mpeg",
                )
            },
        )

        self.assertEqual(response.status_code, 403)
        save_upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
