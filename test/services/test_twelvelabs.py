import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import twelvelabs

RUN_INTEGRATION_TESTS = os.environ.get("MPT_RUN_INTEGRATION_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}


class TestTwelveLabsService(unittest.TestCase):
    """
    La integración de TwelveLabs es totalmente opcional: si no se configura
    twelvelabs_api_keys, todas las funciones deben ser no-op sin efectos secundarios,
    con el mismo comportamiento que sin TwelveLabs. Todos estos casos sustituyen el
    cliente del SDK por un mock, así que la CI no depende de la red ni de una clave API
    reales.
    """

    def setUp(self):
        self.original_app_config = dict(config.app)
        twelvelabs._embed_text_cached.cache_clear()

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        twelvelabs._embed_text_cached.cache_clear()

    # ---------------- comportamiento desactivado / no-op ----------------

    def test_disabled_when_no_api_key(self):
        config.app.pop("twelvelabs_api_keys", None)
        self.assertFalse(twelvelabs.is_enabled())
        # rerank debe devolver la lista de entrada sin cambios
        terms = ["b", "a", "c"]
        self.assertEqual(
            twelvelabs.rerank_terms_by_subject("subject", terms), terms
        )
        # analyze debe ser un no-op que devuelve None
        self.assertIsNone(twelvelabs.analyze_clip("https://x/y.mp4"))

    def test_rerank_skipped_when_flag_off(self):
        config.app["twelvelabs_api_keys"] = ["tlk_test"]
        config.app["twelvelabs_rerank_terms"] = False
        terms = ["b", "a"]
        # Aunque esté activado, con la opción apagada no debemos tocar el orden ni
        # llamar a la API.
        with patch.object(twelvelabs, "_client") as client:
            result = twelvelabs.rerank_terms_by_subject("subject", terms)
        self.assertEqual(result, terms)
        client.assert_not_called()

    # ---------------- comportamiento de rerank activado ----------------

    def _client_returning(self, vectors_by_text):
        """Construye un cliente falso de TwelveLabs cuyo embed.create devuelve vectores
        predefinidos.
        """

        def fake_create(*, model_name, text):
            seg = MagicMock()
            seg.float_ = vectors_by_text[text]
            resp = MagicMock()
            resp.text_embedding.segments = [seg]
            return resp

        client = MagicMock()
        client.embed.create.side_effect = fake_create
        return client

    def test_rerank_orders_by_cosine_to_subject(self):
        config.app["twelvelabs_api_keys"] = ["tlk_test"]
        config.app["twelvelabs_rerank_terms"] = True

        # el tema está alineado con "city"; "kitten" es ortogonal.
        vectors = {
            "city skyline": [1.0, 0.0, 0.0],
            "downtown buildings": [0.9, 0.1, 0.0],  # cerca del tema
            "cute kitten": [0.0, 1.0, 0.0],  # lejos del tema
        }
        client = self._client_returning(vectors)

        with patch.object(twelvelabs, "_client", return_value=client):
            result = twelvelabs.rerank_terms_by_subject(
                "city skyline", ["cute kitten", "downtown buildings"]
            )

        # el término más relevante debe ir primero
        self.assertEqual(result, ["downtown buildings", "cute kitten"])

    def test_rerank_falls_back_on_embed_failure(self):
        config.app["twelvelabs_api_keys"] = ["tlk_test"]
        config.app["twelvelabs_rerank_terms"] = True

        client = MagicMock()
        client.embed.create.side_effect = RuntimeError("api down")

        terms = ["alpha", "beta"]
        with patch.object(twelvelabs, "_client", return_value=client):
            result = twelvelabs.rerank_terms_by_subject("subject", terms)

        # cualquier fallo debe conservar el orden original (nunca empeorar las cosas)
        self.assertEqual(result, terms)

    def test_rerank_noop_for_single_term(self):
        config.app["twelvelabs_api_keys"] = ["tlk_test"]
        config.app["twelvelabs_rerank_terms"] = True
        with patch.object(twelvelabs, "_client") as client:
            result = twelvelabs.rerank_terms_by_subject("subject", ["only"])
        self.assertEqual(result, ["only"])
        client.assert_not_called()

    # ---------------- analyze_clip ----------------

    def test_analyze_clip_returns_model_text(self):
        config.app["twelvelabs_api_keys"] = ["tlk_test"]

        # analyze_clip() importa de forma diferida `twelvelabs.types.VideoContext_Url`.
        # El SDK es un extra opcional, así que el test unitario determinista debe pasar
        # incluso sin `uv sync --extra twelvelabs`. Se inyectan módulos stub ligeros
        # para que se resuelva la importación interna; el _client simulado de abajo hace
        # el resto. (Cuando el SDK real *sí* está instalado, estos stubs se ignoran.)
        stub_types = type(sys)("twelvelabs.types")
        stub_types.VideoContext_Url = lambda *, url: {"url": url}
        stub_pkg = sys.modules.get("twelvelabs") or type(sys)("twelvelabs")
        with patch.dict(
            sys.modules, {"twelvelabs": stub_pkg, "twelvelabs.types": stub_types}
        ):
            self._run_analyze_clip_assertions()

    def _run_analyze_clip_assertions(self):
        resp = MagicMock()
        resp.data = "A city skyline at dusk."
        client = MagicMock()
        client.analyze.return_value = resp

        with patch.object(twelvelabs, "_client", return_value=client):
            out = twelvelabs.analyze_clip(
                "https://example.com/clip.mp4", prompt="describe"
            )

        self.assertEqual(out, "A city skyline at dusk.")
        # max_tokens debe limitarse al mínimo de Pegasus (>=512)
        self.assertGreaterEqual(client.analyze.call_args.kwargs["max_tokens"], 512)


@unittest.skipUnless(
    RUN_INTEGRATION_TESTS and os.getenv("TWELVELABS_API_KEY"),
    "live test: set MPT_RUN_INTEGRATION_TESTS=1 and TWELVELABS_API_KEY to run "
    "against the real TwelveLabs API",
)
class TestTwelveLabsLive(unittest.TestCase):
    """Comprobación del contrato en vivo: solo se ejecuta con MPT_RUN_INTEGRATION_TESTS=1 y
    una clave.
    """

    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app["twelvelabs_api_keys"] = [os.environ["TWELVELABS_API_KEY"]]
        config.app["twelvelabs_rerank_terms"] = True
        twelvelabs._embed_text_cached.cache_clear()

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        twelvelabs._embed_text_cached.cache_clear()

    def test_marengo_embedding_is_512_dim(self):
        vec = twelvelabs.embed_text("a city skyline at night")
        self.assertIsNotNone(vec)
        self.assertEqual(len(vec), 512)

    def test_rerank_puts_relevant_term_first(self):
        result = twelvelabs.rerank_terms_by_subject(
            "city skyline at night",
            ["cute kitten playing with yarn", "downtown buildings and traffic at dusk"],
        )
        self.assertEqual(result[0], "downtown buildings and traffic at dusk")


if __name__ == "__main__":
    unittest.main()
