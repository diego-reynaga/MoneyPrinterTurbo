import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import asgi
from app.config import config


class TestAPIAuthenticationHTTP(unittest.TestCase):
    """Verifica desde la entrada ASGI real la autenticación opcional de la API V1,
    cubriendo los dos grupos de rutas de negocio.
    """

    def setUp(self):
        self.original_app_config = dict(config.app)
        self.client = TestClient(asgi.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_empty_key_preserves_existing_open_access(self):
        """La clave vacía por defecto no exige cabecera, para que los clientes antiguos y
        la WebUI local sigan funcionando.
        """

        config.app["api_key"] = ""

        response = self.client.get("/api/v1/tasks")

        self.assertEqual(response.status_code, 200)

    def test_video_routes_require_matching_key_when_configured(self):
        """Con la protección activada, las rutas de vídeo deben rechazar por igual las
        claves ausentes e incorrectas.
        """

        config.app["api_key"] = "video-secret"

        missing = self.client.get("/api/v1/tasks")
        wrong = self.client.get(
            "/api/v1/tasks",
            headers={"x-api-key": "wrong"},
        )
        accepted = self.client.get(
            "/api/v1/tasks",
            headers={"x-api-key": "video-secret"},
        )

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(accepted.status_code, 200)

    def test_llm_routes_authenticate_before_request_validation(self):
        """Las rutas de LLM deben autenticar primero; una solicitud no autenticada no debe
        llegar a la lógica de negocio que genera costes.
        """

        config.app["api_key"] = "llm-secret"

        # El modelo de solicitud tiene valores por defecto, así que una solicitud vacía
        # también puede llamar de verdad al modelo. Aquí se aíslan los servicios
        # externos y se comprueba el número de llamadas, para verificar el orden de
        # autenticación y no gastar la API del usuario en los tests.
        with patch(
            "app.controllers.v1.llm.llm.generate_script",
            return_value="mocked script",
        ) as generate_script:
            missing = self.client.post("/api/v1/scripts", json={})
            accepted = self.client.post(
                "/api/v1/scripts",
                json={},
                headers={"x-api-key": "llm-secret"},
            )

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(accepted.status_code, 200)
        generate_script.assert_called_once()

    def test_openapi_documents_api_key_header_for_v1_routes(self):
        """Swagger debe mostrar x-api-key, para que con la protección activada no haya que
        adivinar el formato de la solicitud.
        """

        schema = self.client.get("/openapi.json").json()
        parameters = schema["paths"]["/api/v1/tasks"]["get"]["parameters"]

        self.assertTrue(
            any(
                parameter["in"] == "header" and parameter["name"] == "x-api-key"
                for parameter in parameters
            )
        )

    def test_duplicate_api_key_headers_are_rejected(self):
        """La interpretación de credenciales repetidas puede variar según el proxy, así que
        deben rechazarse sea cual sea el orden.
        """

        config.app["api_key"] = "video-secret"

        correct_first = self.client.get(
            "/api/v1/tasks",
            headers=[
                ("x-api-key", "video-secret"),
                ("x-api-key", "wrong"),
            ],
        )
        wrong_first = self.client.get(
            "/api/v1/tasks",
            headers=[
                ("x-api-key", "wrong"),
                ("x-api-key", "video-secret"),
            ],
        )

        self.assertEqual(correct_first.status_code, 401)
        self.assertEqual(wrong_first.status_code, 401)


if __name__ == "__main__":
    unittest.main()
