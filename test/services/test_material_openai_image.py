# -*- coding: utf-8 -*-
import base64
import io
import os
import shutil
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import requests
from PIL import Image

from app.config import config
from app.services import material


def _png_bytes(width=64, height=96, color=(120, 40, 200)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _image_response(payload, status_code=200):
    return SimpleNamespace(json=lambda: payload, status_code=status_code)


def _download_response(content, status_code=200):
    return SimpleNamespace(
        status_code=status_code,
        content=content,
        headers={"Content-Length": str(len(content))},
        iter_content=lambda chunk_size: iter((content,)),
        close=lambda: None,
    )


class TestOpenAIImageProvider(unittest.TestCase):
    """
    Fuente de material de texto a imagen compatible con OpenAI. Igual que los tests de
    las demás fuentes de material, todo usa unittest.mock para sustituir requests y
    time.sleep, así que la CI no depende de la red, de claves API ni de facturación
    reales.
    """

    def setUp(self):
        self.original_app_config = dict(config.app)
        self.original_proxy_config = dict(config.proxy)
        # Las aserciones se hacen después de que vuelva la llamada de generación, así
        # que el directorio temporal no puede destruirse antes al salir del bloque with;
        # por eso su ciclo de vida se gestiona con mkdtemp + addCleanup.
        self.save_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.save_dir, ignore_errors=True)
        config.app["openai_image_base_url"] = "https://img.example.com/v1"
        config.app["openai_image_api_keys"] = ["sk-test-key"]
        config.app["openai_image_model"] = "test-image-model"
        # La plantilla de prompt y el tamaño personalizado están desactivados por
        # defecto, y los casos que necesitan cubrirlos los configuran por su cuenta,
        # para que los ajustes del config.toml local del desarrollador no afecten a las
        # aserciones de los escenarios por defecto.
        config.app.pop("openai_image_prompt_template", None)
        config.app.pop("openai_image_size", None)
        config.app.pop("tls_verify", None)
        config.proxy.clear()

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        config.proxy.clear()
        config.proxy.update(self.original_proxy_config)

    @staticmethod
    def _generated_item(term, image_path, duration=5):
        item = material.MaterialInfo()
        item.provider = "openai_image"
        item.url = image_path
        item.duration = duration
        item.source_info = {
            "provider": "openai_image",
            "search_term": term,
            "rendition": {"id": None, "width": 736, "height": 1312},
        }
        return item

    # ------------------------------------------------------------------
    # Ruta de éxito
    # ------------------------------------------------------------------

    def test_generate_images_openai_with_b64_json_response(self):
        """
        Una respuesta b64_json debe decodificarse y guardarse en disco como un PNG
        válido, y escribir la rendition con el tamaño real de la imagen (por si un
        servicio intermedio devuelve un tamaño distinto al pedido); duration registra la
        duración del clip de destino.
        """
        image_data = _png_bytes(width=736, height=1312)
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(image_data).decode("ascii")}]}
        )

        with patch(
            "app.services.material.requests.post", return_value=response
        ) as post:
            results = material.generate_images_openai(
                "sunrise over mountains",
                minimum_duration=5,
                video_aspect=material.VideoAspect.portrait,
                save_dir=self.save_dir,
            )

        self.assertEqual(len(results), 1)
        item = results[0]
        self.assertEqual(item.provider, "openai_image")
        self.assertEqual(item.duration, 5)
        # El size de la solicitud toma el tamaño compatible oficial de OpenAI según la
        # relación de aspecto, sin usar directamente la resolución del vídeo
        self.assertEqual(
            post.call_args.args[0],
            "https://img.example.com/v1/images/generations",
        )
        self.assertEqual(
            post.call_args.kwargs["json"],
            {
                "model": "test-image-model",
                "prompt": "sunrise over mountains",
                "n": 1,
                "size": "1024x1536",
            },
        )
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer sk-test-key",
        )
        # El archivo guardado es un PNG que se puede decodificar
        self.assertTrue(item.url.endswith(".png"))
        self.assertTrue(os.path.isfile(item.url))
        with Image.open(item.url) as saved:
            self.assertEqual(saved.size, (736, 1312))
        # La rendition registra el tamaño real de la imagen, sin depender de los
        # parámetros de la solicitud
        self.assertEqual(
            item.source_info["rendition"],
            {"id": None, "width": 736, "height": 1312},
        )
        self.assertEqual(item.source_info["search_term"], "sunrise over mountains")

    def test_generate_images_openai_with_url_response(self):
        """Una respuesta url debe descargar de inmediato la dirección temporal y guardarla
        en disco.
        """
        response = _image_response(
            {"data": [{"url": "https://cdn.example.com/generated/abc.png?sig=1"}]}
        )
        download = _download_response(_png_bytes(width=200, height=300))

        with (
            patch("app.services.material.requests.post", return_value=response),
            patch("app.services.material.requests.get", return_value=download) as get,
        ):
            results = material.generate_images_openai(
                "city at night", minimum_duration=3, save_dir=self.save_dir
            )

        self.assertEqual(len(results), 1)
        self.assertTrue(os.path.isfile(results[0].url))
        # La URL temporal se descarga tal cual; no se pueden quitar los parámetros de
        # consulta de la firma
        self.assertEqual(
            get.call_args.args[0],
            "https://cdn.example.com/generated/abc.png?sig=1",
        )

    def test_generate_images_openai_skips_b64_json_with_invalid_image(self):
        """
        Cuando la capa compatible devuelve 200 pero el cuerpo no es una imagen
        decodificable (por ejemplo, una página de error HTML disfrazada de JSON), debe
        devolver una lista vacía según el contrato de las fuentes de material para que
        la capa superior se salte esa palabra clave, en lugar de que la excepción de
        decodificación interrumpa toda la tarea.
        """
        fake_content = b"<html><body>gateway degraded</body></html>"
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(fake_content).decode("ascii")}]}
        )

        with patch("app.services.material.requests.post", return_value=response):
            results = material.generate_images_openai(
                "sunrise over mountains", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(results, [])
        self.assertEqual(os.listdir(self.save_dir), [])

    def test_generate_images_openai_skips_url_download_with_invalid_content(self):
        """Cuando la descarga de la URL temporal devuelve contenido 200 que no es imagen,
        también se sigue la ruta de omisión.
        """
        response = _image_response(
            {"data": [{"url": "https://cdn.example.com/generated/abc.png?sig=1"}]}
        )
        download = _download_response(b"\x89PNG\r\n\x1a\nnot-really-a-png")

        with (
            patch("app.services.material.requests.post", return_value=response),
            patch("app.services.material.requests.get", return_value=download),
        ):
            results = material.generate_images_openai(
                "city at night", minimum_duration=3, save_dir=self.save_dir
            )

        self.assertEqual(results, [])
        self.assertEqual(os.listdir(self.save_dir), [])

    def test_generate_images_openai_propagates_image_write_failure(self):
        """
        Cuando la imagen ya se ha decodificado bien pero falla la escritura del PNG, hay
        que interrumpir la tarea. Este tipo de fallo suele seguir afectando a las
        palabras clave siguientes; si se tomara por un problema de contenido de una sola
        imagen y se continuara, se harían solicitudes de pago que no se podrían guardar.
        """
        response = _image_response(
            {
                "data": [
                    {"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}
                ]
            }
        )

        with (
            patch("app.services.material.requests.post", return_value=response),
            patch.object(
                Image.Image,
                "save",
                side_effect=OSError("no space left on device"),
            ),
            self.assertRaisesRegex(OSError, "no space left on device"),
        ):
            material.generate_images_openai(
                "city at night", minimum_duration=3, save_dir=self.save_dir
            )

        self.assertEqual(os.listdir(self.save_dir), [])

    # ------------------------------------------------------------------
    # Reintentos con espera y rotación de claves
    # ------------------------------------------------------------------

    def test_generate_images_openai_retries_429_with_backoff(self):
        """Un 429 es un límite de uso temporal; hay que reintentar con espera en lugar de
        dar la tarea por perdida.
        """
        image_data = _png_bytes()
        responses = [
            _image_response({"error": {"message": "rate limited"}}, status_code=429),
            _image_response(
                {"data": [{"b64_json": base64.b64encode(image_data).decode("ascii")}]}
            ),
        ]

        with (
            tempfile.TemporaryDirectory() as save_dir,
            patch("app.services.material.requests.post", side_effect=responses) as post,
            patch("app.services.material.time.sleep") as sleep,
        ):
            results = material.generate_images_openai(
                "ocean waves", minimum_duration=5, save_dir=save_dir
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(post.call_count, 2)
        # Antes del primer reintento hay que esperar con un retroceso lineal, sin
        # saturar de inmediato la interfaz remota
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(
            sleep.call_args.args[0],
            material.OPENAI_IMAGE_RETRY_BACKOFF_SECONDS[0],
        )

    def test_generate_images_openai_rotates_key_on_401(self):
        """
        Un 401 indica que se ha rechazado la clave actual. Con varias claves
        configuradas, el reintento debe pasar a la siguiente clave mediante la rotación
        de get_api_key, en lugar de usar una y otra vez la misma clave rechazada.
        """
        config.app["openai_image_api_keys"] = ["sk-bad-key", "sk-good-key"]
        image_data = _png_bytes()

        responses = [
            _image_response({"error": {"message": "unauthorized"}}, status_code=401),
            _image_response(
                {"data": [{"b64_json": base64.b64encode(image_data).decode("ascii")}]}
            ),
        ]

        with (
            tempfile.TemporaryDirectory() as save_dir,
            patch("app.services.material.requests.post", side_effect=responses) as post,
            patch("app.services.material.time.sleep"),
        ):
            results = material.generate_images_openai(
                "forest fog", minimum_duration=5, save_dir=save_dir
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(post.call_count, 2)
        used_keys = [
            call.kwargs["headers"]["Authorization"] for call in post.call_args_list
        ]
        # Dos solicitudes seguidas deben usar claves distintas, y ambas de la lista
        # configurada
        self.assertNotEqual(used_keys[0], used_keys[1])
        for auth in used_keys:
            self.assertIn(auth.replace("Bearer ", ""), ["sk-bad-key", "sk-good-key"])

    def test_generate_images_openai_fails_fast_on_401_with_single_key(self):
        """Con una sola clave, reintentar un 401 no tiene sentido y debe fallar rápido
        devolviendo un resultado vacío.
        """
        response = _image_response(
            {"error": {"message": "unauthorized"}}, status_code=401
        )

        with (
            tempfile.TemporaryDirectory() as save_dir,
            patch("app.services.material.requests.post", return_value=response) as post,
            patch("app.services.material.time.sleep") as sleep,
        ):
            results = material.generate_images_openai(
                "desert dunes", minimum_duration=5, save_dir=save_dir
            )

        self.assertEqual(results, [])
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_generate_images_openai_returns_empty_after_retries_exhausted(self):
        """
        Tras agotar todos los reintentos se devuelve una lista vacía según el contrato
        de las fuentes de material, para que la capa superior se salte esa palabra
        clave; y no se deja ningún archivo residual en disco.
        """
        response = _image_response(
            {"error": {"message": "rate limited"}}, status_code=429
        )

        with (
            patch("app.services.material.requests.post", return_value=response) as post,
            patch("app.services.material.time.sleep") as sleep,
        ):
            results = material.generate_images_openai(
                "storm clouds", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(results, [])
        self.assertEqual(post.call_count, material.OPENAI_IMAGE_MAX_ATTEMPTS)
        self.assertEqual(sleep.call_count, material.OPENAI_IMAGE_MAX_ATTEMPTS - 1)
        # No se generó ningún archivo residual
        self.assertEqual(os.listdir(self.save_dir), [])

    def test_generate_images_openai_redacts_api_key_in_failure_detail(self):
        """Los detalles del fallo no pueden escribir la clave API en claro en el log."""
        config.app["openai_image_api_keys"] = ["sk-secret-123"]
        response = _image_response(
            {"error": {"message": "invalid key sk-secret-123 provided"}},
            status_code=401,
        )

        with (
            tempfile.TemporaryDirectory() as save_dir,
            patch("app.services.material.requests.post", return_value=response),
            patch("app.services.material.logger") as logger,
        ):
            results = material.generate_images_openai(
                "redacted term", minimum_duration=5, save_dir=save_dir
            )

        self.assertEqual(results, [])
        logged = [str(call) for call in logger.error.call_args_list]
        self.assertTrue(logged)
        for message in logged:
            self.assertNotIn("sk-secret-123", message)

    def test_generate_images_openai_retries_generated_image_download(self):
        """
        La imagen ya se ha cobrado por unidad; un fallo pasajero de descarga debe
        reintentar la misma URL y no volver a generar la misma imagen, lo que cobraría
        dos veces.
        """
        response = _image_response(
            {"data": [{"url": "https://cdn.example.com/generated/x.png"}]}
        )
        downloads = [
            _download_response(b"", status_code=502),
            _download_response(_png_bytes()),
        ]

        with (
            tempfile.TemporaryDirectory() as save_dir,
            patch("app.services.material.requests.post", return_value=response) as post,
            patch("app.services.material.requests.get", side_effect=downloads) as get,
            patch("app.services.material.time.sleep"),
        ):
            results = material.generate_images_openai(
                "aurora", minimum_duration=5, save_dir=save_dir
            )

        self.assertEqual(len(results), 1)
        # El reintento de descarga va a la misma dirección y no provoca una segunda
        # generación de pago
        self.assertEqual(post.call_count, 1)
        self.assertEqual(get.call_count, 2)
        for call in get.call_args_list:
            self.assertEqual(call.args[0], "https://cdn.example.com/generated/x.png")

    def test_generated_image_download_rejects_oversize_body(self):
        """Una URL de imagen generada no debe almacenar en búfer un cuerpo ilimitado del
        proveedor.
        """
        response = _download_response(b"too-large")
        response.headers = {}
        closed = []
        response.close = lambda: closed.append(True)
        with (
            patch("app.services.material.requests.get", return_value=response) as get,
            patch("app.services.material.OPENAI_IMAGE_MAX_BYTES", 5, create=True),
            patch("app.services.material.time.sleep"),
        ):
            image_bytes, error = material._openai_image_download_bytes(
                "https://cdn.example.com/generated/x.png", "test-key"
            )

        self.assertIsNone(image_bytes)
        self.assertIn("límite", error)
        self.assertTrue(all(call.kwargs.get("stream") is True for call in get.call_args_list))
        self.assertEqual(closed, [True])

    def test_generated_image_download_rejects_declared_oversize_body(self):
        """Una imagen anunciada como demasiado grande debe rechazarse antes de leer los
        bytes.
        """
        response = _download_response(b"small")
        response.headers = {"Content-Length": "100"}
        response.iter_content = lambda chunk_size: self.fail("body should not be read")
        with (
            patch("app.services.material.requests.get", return_value=response),
            patch("app.services.material.OPENAI_IMAGE_MAX_BYTES", 5),
        ):
            image_bytes, error = material._openai_image_download_bytes(
                "https://cdn.example.com/generated/x.png", "test-key"
            )

        self.assertIsNone(image_bytes)
        self.assertIn("límite", error)

    def test_generated_image_response_rejects_oversize_base64(self):
        """No se decodifica una imagen incrustada ilimitada de una respuesta del proveedor."""
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(b"too-large").decode()}]}
        )
        with patch("app.services.material.OPENAI_IMAGE_MAX_BYTES", 5):
            image_bytes, error = material._parse_openai_image_response(response, "test-key")

        self.assertIsNone(image_bytes)
        self.assertIn("límite", error)

    def test_generated_image_decode_rejects_pillow_bomb_warning(self):
        """Unos pocos bytes comprimidos aún pueden expandirse a demasiados píxeles de
        imagen.
        """
        image_bytes = _png_bytes(width=12, height=12)
        with (
            patch.object(Image, "MAX_IMAGE_PIXELS", 100),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with self.assertRaises(material._OpenAIImageDecodeError):
                material._save_openai_image_file(image_bytes, self.save_dir)

        self.assertEqual(os.listdir(self.save_dir), [])

    def test_generated_image_download_failure_does_not_buy_another_image(self):
        """Un resultado de pago confirmado sigue siendo un fallo de la tarea si no se puede
        descargar su URL.
        """
        signed_url = "https://cdn.example.com/generated/x.png?token=private"
        response = _image_response(
            {"data": [{"url": signed_url}]}
        )
        with (
            patch("app.services.material.requests.post", return_value=response) as post,
            patch(
                "app.services.material.requests.get",
                side_effect=requests.exceptions.ConnectionError(
                    f"download failed for {signed_url}"
                ),
            ) as get,
            patch("app.services.material.time.sleep"),
            patch("app.services.material.logger") as logger,
        ):
            with self.assertRaisesRegex(RuntimeError, "no se pudo descargar"):
                material.download_videos(
                    task_id="test-openai-image-download-failure",
                    search_terms=["first", "second"],
                    source="openai_image",
                    audio_duration=10,
                    max_clip_duration=5,
                )

        self.assertEqual(post.call_count, 1)
        self.assertEqual(get.call_count, material.OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS)
        self.assertNotIn("token=private", str(logger.warning.call_args_list))

    def test_generate_images_openai_returns_empty_on_rejected_request(self):
        """Un rechazo de negocio (por ejemplo, por la política de contenido) devuelve un
        resultado vacío sin reintentos con espera.
        """
        response = _image_response(
            {"error": {"message": "content policy violation"}}, status_code=400
        )

        with (
            tempfile.TemporaryDirectory() as save_dir,
            patch("app.services.material.requests.post", return_value=response) as post,
            patch("app.services.material.time.sleep") as sleep,
        ):
            results = material.generate_images_openai(
                "blocked term", minimum_duration=5, save_dir=save_dir
            )

        self.assertEqual(results, [])
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    # ------------------------------------------------------------------
    # Interruptores de configuración
    # ------------------------------------------------------------------

    def test_is_openai_image_enabled_requires_full_configuration(self):
        """
        base_url y model son imprescindibles; la clave API puede estar vacía, porque un
        gateway ComfyUI/SD totalmente local normalmente no necesita autenticación.
        """
        self.assertTrue(material.is_openai_image_enabled())

        config.app["openai_image_base_url"] = ""
        self.assertFalse(material.is_openai_image_enabled())
        config.app["openai_image_base_url"] = "https://img.example.com/v1"

        # Gateway local sin autenticación: sin clave también cuenta como activado
        config.app["openai_image_api_keys"] = []
        self.assertTrue(material.is_openai_image_enabled())
        config.app["openai_image_api_keys"] = ["sk-test-key"]

        config.app["openai_image_model"] = ""
        self.assertFalse(material.is_openai_image_enabled())

    def test_generate_images_openai_sends_no_authorization_without_key(self):
        """
        Sin clave API configurada debe generarse con normalidad y sin cabecera
        Authorization en la solicitud, para los gateways ComfyUI/SD locales sin
        autenticación.
        """
        config.app["openai_image_api_keys"] = []
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}]}
        )

        with (
            patch("app.services.material.requests.post", return_value=response) as post,
        ):
            results = material.generate_images_openai(
                "local gateway", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(len(results), 1)
        self.assertNotIn("Authorization", post.call_args.kwargs["headers"])

    def test_generate_images_openai_retries_connect_timeout(self):
        """
        Un tiempo de espera en la fase de conexión indica que la solicitud no llegó al
        servidor, así que no puede haber creado una tarea facturable y se permite
        reintentar con espera.
        """
        image_data = _png_bytes()
        responses = [
            requests.exceptions.ConnectTimeout("connect timed out"),
            _image_response(
                {"data": [{"b64_json": base64.b64encode(image_data).decode("ascii")}]}
            ),
        ]

        with (
            patch("app.services.material.requests.post", side_effect=responses) as post,
            patch("app.services.material.time.sleep"),
        ):
            results = material.generate_images_openai(
                "connect timeout term", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(post.call_count, 2)

    def test_generate_images_openai_does_not_retry_unconfirmed_errors(self):
        """
        Un tiempo de espera de lectura o un corte de conexión es un estado "sin
        confirmar": el servidor puede haber generado y cobrado ya, aunque la respuesta
        no haya llegado. Volver a enviar automáticamente generaría y cobraría dos veces,
        así que hay que fallar directamente y terminar la tarea, sin seguir enviando
        solicitudes de pago para las palabras clave siguientes.
        """
        for error in (
            requests.exceptions.ReadTimeout("read timed out"),
            requests.exceptions.ConnectionError("connection dropped"),
        ):
            with self.subTest(error=type(error).__name__):
                with (
                    patch(
                        "app.services.material.requests.post", side_effect=error
                    ) as post,
                    patch("app.services.material.time.sleep") as sleep,
                ):
                    with self.assertRaisesRegex(RuntimeError, "sin confirmar"):
                        material.generate_images_openai(
                            "unconfirmed term",
                            minimum_duration=5,
                            save_dir=self.save_dir,
                        )

                self.assertEqual(post.call_count, 1)
                sleep.assert_not_called()

    def test_download_videos_openai_image_stops_after_unconfirmed_paid_request(self):
        """Las imágenes anteriores no deben ocultar un envío de pago ambiguo posterior."""
        image_response = _image_response(
            {"data": [{"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}]}
        )
        config.app["material_directory"] = self.save_dir

        with (
            patch(
                "app.services.material.requests.post",
                side_effect=[
                    image_response,
                    requests.exceptions.ReadTimeout("response lost"),
                    image_response,
                ],
            ) as post,
            patch(
                "app.services.material._render_openai_image_video",
                return_value="/tmp/rendered.mp4",
            ),
            patch("app.services.material._persist_material_sources"),
        ):
            with self.assertRaisesRegex(RuntimeError, "sin confirmar"):
                material.download_videos(
                    task_id="test-openai-image-unconfirmed",
                    search_terms=["first", "uncertain", "third"],
                    source="openai_image",
                    audio_duration=20,
                    max_clip_duration=5,
                )

        self.assertEqual(post.call_count, 2)

    def test_generate_images_openai_size_defaults_and_override(self):
        """
        Por defecto se toma el tamaño compatible oficial de OpenAI según la relación de
        aspecto (portrait 1024x1536 / landscape 1536x1024); al configurar
        openai_image_size se sobrescribe por completo, para gateways locales que admiten
        cualquier resolución.
        """
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}]}
        )

        with patch(
            "app.services.material.requests.post", return_value=response
        ) as default_post:
            material.generate_images_openai(
                "landscape term",
                minimum_duration=5,
                video_aspect=material.VideoAspect.landscape,
                save_dir=self.save_dir,
            )
        # En horizontal se toma por defecto el tamaño compatible oficial de OpenAI
        self.assertEqual(default_post.call_args.kwargs["json"]["size"], "1536x1024")

        config.app["openai_image_size"] = "1080x1920"
        self.addCleanup(config.app.pop, "openai_image_size", None)

        with patch(
            "app.services.material.requests.post", return_value=response
        ) as post:
            material.generate_images_openai(
                "custom size term", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(
            post.call_args.kwargs["json"]["size"],
            "1080x1920",
        )

    def test_generate_images_openai_raises_without_base_url(self):
        """Al llamar directamente sin base_url configurada, debe lanzarse un error con
        instrucciones de configuración.
        """
        config.app["openai_image_base_url"] = ""
        with self.assertRaises(ValueError):
            material.generate_images_openai("term", minimum_duration=5)

    # ------------------------------------------------------------------
    # Plantilla de prompt
    # ------------------------------------------------------------------

    def test_generate_images_openai_applies_prompt_template(self):
        """
        Con una plantilla configurada que contenga el marcador {term}, el prompt de la
        solicitud debe ser el resultado de sustituir la plantilla, añadiendo de forma
        común modificadores de estilo que mejoran la correspondencia entre imagen y
        texto.
        """
        config.app["openai_image_prompt_template"] = (
            "cinematic photo of {term}, photorealistic, high detail"
        )
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}]}
        )

        with patch(
            "app.services.material.requests.post", return_value=response
        ) as post:
            material.generate_images_openai(
                "晨光中的玻璃杯", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(
            post.call_args.kwargs["json"]["prompt"],
            "cinematic photo of 晨光中的玻璃杯, photorealistic, high detail",
        )

    def test_generate_images_openai_sends_raw_term_without_template(self):
        """Sin plantilla configurada, el prompt debe ser la palabra clave original, con el
        mismo comportamiento que la versión antigua.
        """
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}]}
        )

        with patch(
            "app.services.material.requests.post", return_value=response
        ) as post:
            material.generate_images_openai(
                "raw term", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(post.call_args.kwargs["json"]["prompt"], "raw term")

    def test_generate_images_openai_falls_back_when_template_lacks_placeholder(self):
        """Si la plantilla no contiene el marcador {term} no se puede inyectar la palabra
        clave, así que hay que volver al texto original.
        """
        config.app["openai_image_prompt_template"] = "no placeholder here"
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}]}
        )

        with patch(
            "app.services.material.requests.post", return_value=response
        ) as post:
            material.generate_images_openai(
                "fallback term", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(post.call_args.kwargs["json"]["prompt"], "fallback term")

    # ------------------------------------------------------------------
    # Despacho de download_videos y generación bajo demanda
    # ------------------------------------------------------------------

    def test_download_videos_openai_image_generates_on_demand_and_stops(self):
        """
        El texto a imagen se cobra por imagen, así que no se puede generar primero para
        todas las palabras clave y elegir después. El material debe generarse imagen a
        imagen bajo demanda, y cuando la duración válida acumulada (limitada por la
        duración del clip) alcanza la duración de locución necesaria, las palabras clave
        siguientes ya no lanzan ninguna solicitud de pago.
        """
        generated = {
            "term-1": [self._generated_item("term-1", "/tmp/img-1.png")],
            "term-2": [self._generated_item("term-2", "/tmp/img-2.png")],
            "term-3": [self._generated_item("term-3", "/tmp/img-3.png")],
        }

        def fake_generate(search_term, minimum_duration, video_aspect, save_dir=""):
            return generated[search_term]

        def fake_render(image_path, clip_duration, image_motion="zoom"):
            return f"{image_path}.mp4"

        with (
            patch(
                "app.services.material.generate_images_openai",
                side_effect=fake_generate,
            ) as generate,
            patch(
                "app.services.material._render_openai_image_video",
                side_effect=fake_render,
            ) as render,
        ):
            result = material.download_videos(
                task_id="test-openai-image-lazy",
                search_terms=["term-1", "term-2", "term-3"],
                source="openai_image",
                audio_duration=8,
                max_clip_duration=5,
            )

        # 5 s + 5 s > 8 s, así que la tercera palabra clave ya no puede generar una
        # solicitud de pago
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(
            [call.kwargs["search_term"] for call in generate.call_args_list],
            ["term-1", "term-2"],
        )
        # Cada imagen solo cuenta en la duración después de renderizarse como clip mp4
        self.assertEqual(render.call_count, 2)
        self.assertEqual(result, ["/tmp/img-1.png.mp4", "/tmp/img-2.png.mp4"])

    def test_download_videos_openai_image_continues_after_invalid_image(self):
        """
        Cuando la respuesta de la primera interfaz compatible no se puede decodificar
        solo se salta su palabra clave, y la siguiente imagen válida aún puede guardarse
        y renderizarse, lo que verifica que la corrección cubre la cadena real de
        generación bajo demanda y no solo una función suelta.
        """
        invalid_response = _image_response(
            {
                "data": [
                    {
                        "b64_json": base64.b64encode(
                            b"<html>gateway degraded</html>"
                        ).decode("ascii")
                    }
                ]
            }
        )
        valid_response = _image_response(
            {
                "data": [
                    {"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}
                ]
            }
        )
        config.app["material_directory"] = self.save_dir

        with (
            patch(
                "app.services.material.requests.post",
                side_effect=[invalid_response, valid_response],
            ) as post,
            patch(
                "app.services.material._render_openai_image_video",
                return_value="/tmp/rendered-openai-image.mp4",
            ) as render,
            patch("app.services.material._persist_material_sources"),
        ):
            result = material.download_videos(
                task_id="test-openai-image-invalid-then-valid",
                search_terms=["invalid term", "valid term"],
                source="openai_image",
                audio_duration=5,
                max_clip_duration=5,
            )

        self.assertEqual(post.call_count, 2)
        self.assertEqual(render.call_count, 1)
        self.assertEqual(result, ["/tmp/rendered-openai-image.mp4"])

    def test_download_videos_openai_image_stops_when_duration_exactly_covered(self):
        """Regresión de límite: alcanzar exactamente la duración necesaria ya es
        suficiente, así que la condición de parada debe ser >= y no >.
        """
        generated = {
            "term-1": [self._generated_item("term-1", "/tmp/img-1.png")],
            "term-2": [self._generated_item("term-2", "/tmp/img-2.png")],
            "term-3": [self._generated_item("term-3", "/tmp/img-3.png")],
        }

        def fake_generate(search_term, minimum_duration, video_aspect, save_dir=""):
            return generated[search_term]

        with (
            patch(
                "app.services.material.generate_images_openai",
                side_effect=fake_generate,
            ) as generate,
            patch(
                "app.services.material._render_openai_image_video",
                return_value="/tmp/rendered.mp4",
            ),
        ):
            result = material.download_videos(
                task_id="test-openai-image-exact",
                search_terms=["term-1", "term-2", "term-3"],
                source="openai_image",
                audio_duration=10,
                max_clip_duration=5,
            )

        # 5 s + 5 s == 10 s, cubre justo, así que el tercer segmento nunca debe
        # generarse
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(len(result), 2)

    def test_download_videos_openai_image_bypasses_search_cache(self):
        """
        El resultado de la generación es un archivo de imagen de un solo uso y no entra
        en la caché de búsqueda de 24 horas: la caché haría que tareas distintas
        obtuvieran una y otra vez la misma imagen. download_videos debe ir directamente
        por la rama de generación bajo demanda.
        """
        with (
            patch(
                "app.services.material.generate_images_openai",
                return_value=[self._generated_item("sunrise", "/tmp/img-1.png")],
            ) as generate,
            patch("app.services.material._search_videos_with_cache") as cached_search,
            patch(
                "app.services.material._render_openai_image_video",
                return_value="/tmp/img-1.png.mp4",
            ),
        ):
            result = material.download_videos(
                task_id="test-openai-image-cache-bypass",
                search_terms=["sunrise"],
                source="openai_image",
                audio_duration=5,
                max_clip_duration=5,
            )

        self.assertEqual(generate.call_count, 1)
        cached_search.assert_not_called()
        self.assertEqual(result, ["/tmp/img-1.png.mp4"])

    def test_download_videos_openai_image_skips_rejected_segment(self):
        """
        Cuando un resultado de generación rechazado explícitamente está vacío se salta
        esa palabra clave y se sigue con el clip siguiente.
        """
        generated = {
            "term-1": [],  # La generación falla
            "term-2": [self._generated_item("term-2", "/tmp/img-2.png")],
            "term-3": [self._generated_item("term-3", "/tmp/img-3.png")],
        }

        def fake_generate(search_term, minimum_duration, video_aspect, save_dir=""):
            return generated[search_term]

        def fake_render(image_path, clip_duration, image_motion="zoom"):
            return f"{image_path}.mp4"

        with (
            patch(
                "app.services.material.generate_images_openai",
                side_effect=fake_generate,
            ) as generate,
            patch(
                "app.services.material._render_openai_image_video",
                side_effect=fake_render,
            ),
        ):
            result = material.download_videos(
                task_id="test-openai-image-skip",
                search_terms=["term-1", "term-2", "term-3"],
                source="openai_image",
                audio_duration=5,
                max_clip_duration=5,
            )

        self.assertEqual(generate.call_count, 2)
        self.assertEqual(result, ["/tmp/img-2.png.mp4"])

    def test_download_videos_openai_image_stops_after_render_failure(self):
        """Una imagen de pago guardada en local no debe provocar otra compra si falla el
        renderizado.
        """
        generated_item = self._generated_item("first", "/tmp/img-1.png")
        with (
            patch(
                "app.services.material.generate_images_openai",
                return_value=[generated_item],
            ) as generate,
            patch("app.services.material._render_openai_image_video", return_value=""),
        ):
            with self.assertRaisesRegex(RuntimeError, "no se pudo renderizar"):
                material.download_videos(
                    task_id="test-openai-image-render-failure",
                    search_terms=["first", "second"],
                    source="openai_image",
                    audio_duration=10,
                    max_clip_duration=5,
                )

        self.assertEqual(generate.call_count, 1)

    def test_download_videos_openai_image_skips_generation_without_audio(self):
        """Si la duración de la locución no es positiva se vuelve sin nada, sin pagar
        imágenes para una tarea que nunca se podría completar.
        """
        with patch("app.services.material.generate_images_openai") as generate:
            result = material.download_videos(
                task_id="test-openai-image-no-audio",
                search_terms=["term-1"],
                source="openai_image",
                audio_duration=0,
                max_clip_duration=5,
            )

        generate.assert_not_called()
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
