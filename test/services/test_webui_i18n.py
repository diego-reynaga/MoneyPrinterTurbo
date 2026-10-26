import ast
import json
import re
import unittest
from pathlib import Path

from app.models.llm_provider import get_llm_provider
from app.utils import utils


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"
I18N_DIR = ROOT_DIR / "webui" / "i18n"
MARKDOWN_URL_PATTERN = re.compile(r"\[[^\]]+\]\((https?://[^)]+)\)")


class _TrKeyVisitor(ast.NodeVisitor):
    def __init__(self):
        self.keys = set()

    def visit_Call(self, node):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "tr"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            self.keys.add(node.args[0].value)
        self.generic_visit(node)


def _load_translation():
    data = json.loads((I18N_DIR / "es.json").read_text(encoding="utf-8"))
    return data.get("Translation", {})


def _duplicate_translation_keys(path):
    """Devuelve las claves definidas más de una vez en el texto original del locale; el
    análisis JSON solo conserva la última.
    """
    duplicates = []

    def collect(pairs):
        seen = set()
        for key, _ in pairs:
            if key in seen:
                duplicates.append(key)
            seen.add(key)
        return dict(pairs)

    json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=collect)
    return duplicates


def _markdown_urls(value):
    """Extrae los destinos de los enlaces Markdown, para no estropear las direcciones."""
    return set(MARKDOWN_URL_PATTERN.findall(value))


class TestWebuiI18n(unittest.TestCase):
    def test_spanish_is_the_only_interface_language(self):
        """La interfaz solo se distribuye en español."""
        self.assertEqual(sorted(path.name for path in I18N_DIR.glob("*.json")), ["es.json"])
        locales = utils.load_locales(str(I18N_DIR))
        self.assertEqual(list(locales), ["es"])
        self.assertEqual(locales["es"]["Language"], "Español")

    def test_any_saved_or_browser_language_resolves_to_spanish(self):
        """Los idiomas guardados por versiones antiguas o enviados por el navegador
        siempre acaban en español.
        """
        locales = utils.load_locales(str(I18N_DIR))
        for saved, browser in (("zh", "zh-CN"), ("en", "en-US"), ("", "es-MX"), ("", None)):
            with self.subTest(saved=saved, browser=browser):
                self.assertEqual(utils.resolve_ui_language(saved, browser, locales), "es")

    def test_saved_ui_language_takes_priority_over_browser_locale(self):
        language = utils.resolve_ui_language(
            saved_language="de",
            browser_locale="es-ES",
            supported_languages=["es", "de"],
        )

        self.assertEqual(language, "de")

    def test_browser_locale_is_normalized_to_supported_base_language(self):
        self.assertEqual(
            utils.resolve_ui_language("", "es-AR", ["es"]),
            "es",
        )
        self.assertEqual(
            utils.resolve_ui_language(None, "pt_BR", ["es", "pt"]),
            "pt",
        )

    def test_unsupported_browser_locale_falls_back_to_spanish(self):
        language = utils.resolve_ui_language(
            saved_language="",
            browser_locale="fr-FR",
            supported_languages=["es"],
        )

        self.assertEqual(language, "es")

    def test_spanish_locale_covers_static_webui_labels(self):
        tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
        visitor = _TrKeyVisitor()
        visitor.visit(tree)

        es_keys = set(_load_translation())

        self.assertEqual(sorted(visitor.keys - es_keys), [])

    def test_spanish_pause_examples_use_supported_syntax(self):
        """Las etiquetas de pausa de la ayuda deben poder analizarse; no se puede traducir
        la palabra clave a una sintaxis no admitida.
        """
        help_text = _load_translation()["Video Script Help"]
        examples = re.findall(r"\[(?:pausa|silencio): [^\]]+\]", help_text)
        self.assertTrue(examples)
        for example in examples:
            with self.subTest(example=example):
                self.assertTrue(utils.has_pause_tags(example))
                self.assertEqual(utils.remove_pause_tags(example).strip(), "")

    def test_shengsuanyun_provider_tips_keep_registration_and_model_links(self):
        """Las entradas de socios y los catálogos de modelos son configuración del
        producto; se evita borrar enlaces de seguimiento al cambiar textos.
        """
        expected_urls = {
            "https://www.shengsuanyun.com/?from=CH_XUQ4OTSK",
            "https://global.modelmesh.info/model",
        }

        tips = _load_translation()["llm_provider_tips.shengsuanyun"]
        provider = get_llm_provider("shengsuanyun")
        rendered = tips.format(
            api_key_url=provider.effective_api_key_url(),
            default_base_url=provider.effective_default_base_url,
            default_model=provider.default_model,
        )
        self.assertEqual(_markdown_urls(rendered), expected_urls)

    def test_metaso_api_key_label_keeps_mpt_referral_link(self):
        """La entrada para obtener la clave de Metaso debe conservar el parámetro de
        seguimiento MPT, para no romper el circuito de conversión del patrocinio.
        """
        expected_url = "https://metaso.cn/minimax-h3/?s=MPT"

        label = _load_translation()["Metaso MiniMax API Key"]
        self.assertEqual(_markdown_urls(label), {expected_url})

    def test_script_language_options_include_spanish_russian_and_catalan(self):
        tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
        support_locales = None

        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name) and target.id == "support_locales"
                for target in node.targets
            ):
                support_locales = ast.literal_eval(node.value)
                break

        self.assertIsNotNone(support_locales)
        self.assertIn("es-ES", support_locales)
        self.assertIn("ru-RU", support_locales)
        self.assertIn("ca-ES", support_locales)

    def test_locale_file_does_not_redefine_a_translation_key(self):
        """
        Cuando una clave aparece repetida en el mismo objeto JSON, el análisis solo
        conserva la última y la anterior se descarta en silencio. Aquí se revisa
        directamente el texto original del locale, para que este tipo de sobrescritura no
        se escape de los tests de cobertura de claves de tr().
        """

        for path in sorted(I18N_DIR.glob("*.json")):
            with self.subTest(locale=path.stem):
                self.assertEqual(_duplicate_translation_keys(path), [])
