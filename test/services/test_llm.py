import json
import os
import sys
import tempfile
import tomllib
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.llm_provider import (
    DEFAULT_LLM_PROVIDER_ID,
    LLM_PROVIDER_REGISTRY,
    LLM_PROVIDERS,
    get_llm_provider,
    normalize_provider_override,
)
from app.models.schema import VideoScriptRequest, VideoSocialMetadataRequest
from app.services import llm

RUN_INTEGRATION_TESTS = os.environ.get("MPT_RUN_INTEGRATION_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}


class TestScriptPromptOptions(unittest.TestCase):
    def test_normalize_text_response_preserves_internal_newlines(self):
        """
        La normalización solo limpia los espacios al principio y al final; no puede
        borrar los saltos de línea dentro del texto. El doble salto de línea separa los
        párrafos del guion, y un salto simple también puede ser una línea de subtítulo
        que devolvió el modelo según el sentido.
        """
        result = llm._normalize_text_response(
            "\n  第一行\n第二行\n\n第三段  \n",
            "openai",
        )

        self.assertEqual(result, "第一行\n第二行\n\n第三段")

    def test_normalize_text_response_removes_think_blocks(self):
        """
        Los modelos de razonamiento pueden devolver `<think>...</think>`. La cadena de
        generación de guiones debe conservar solo el texto final, para que el proceso de
        razonamiento no llegue a los subtítulos ni a la locución.
        """
        result = llm._normalize_text_response(
            "<think>\nI should reason here.\n</think>\n测试成功",
            "minimax",
        )

        self.assertEqual(result, "测试成功")

    def test_normalize_text_response_rejects_think_only_response(self):
        """
        Si el modelo solo devuelve el bloque de razonamiento sin respuesta final, debe
        tratarse como contenido vacío y provocar un reintento o un error claro.
        """
        with self.assertRaises(ValueError):
            llm._normalize_text_response("<think>hidden reasoning</think>", "minimax")

    def test_normalize_text_response_removes_unclosed_think_block(self):
        """
        Algunos gateways pueden devolver solo un `<think>` sin cerrar por un
        truncamiento. Ese contenido tampoco puede llegar al guion final; si tras
        limpiarlo no queda texto, debe tratarse como respuesta vacía.
        """
        with self.assertRaises(ValueError):
            llm._normalize_text_response("<think>hidden reasoning", "minimax")

    def test_normalize_text_response_removes_reasoning_before_unopened_close(self):
        """
        Cuando el `<think>` de apertura está en la plantilla de conversación, la salida
        del modelo solo contiene el razonamiento y la etiqueta de cierre. El
        razonamiento anterior a la etiqueta de cierre no puede entrar en el guion; si
        solo hay razonamiento, también se trata como respuesta vacía.
        """
        result = llm._normalize_text_response(
            "The user wants a sea script. Let me plan.\n</think>\n\n"
            "The ocean is vast.\n\nIt covers our planet.",
            "deepseek",
        )

        self.assertEqual(result, "The ocean is vast.\n\nIt covers our planet.")
        with self.assertRaises(ValueError):
            llm._normalize_text_response("hidden reasoning\n</THINK>\n", "ollama")

    def test_build_script_prompt_appends_advanced_requirements(self):
        """
        Los requisitos avanzados del guion solo se añaden como restricciones adicionales
        y no sustituyen al prompt del sistema por defecto. Así, los usuarios normales
        que no lo configuran siguen usando las reglas estables por defecto, y los
        usuarios avanzados pueden afinar el estilo.
        """
        prompt = llm.build_script_prompt(
            video_subject="咖啡",
            language="zh-CN",
            paragraph_number=3,
            video_script_prompt="语气轻松，面向程序员",
        )

        self.assertIn("# Role: Video Script Generator", prompt)
        self.assertIn("- video subject: 咖啡", prompt)
        self.assertIn("- number of paragraphs: 3", prompt)
        self.assertIn("- language: zh-CN", prompt)
        self.assertIn("# Additional User Requirements:", prompt)
        self.assertIn("语气轻松，面向程序员", prompt)

    def test_custom_system_prompt_keeps_runtime_context(self):
        """
        Un system prompt personalizado sustituye las reglas del guion por defecto, pero
        el tema del vídeo, el idioma y el número de párrafos los sigue añadiendo de
        forma común la capa de servicio, para que a los usuarios avanzados no se les
        olvide el contexto necesario.
        """
        prompt = llm.build_script_prompt(
            video_subject="露营",
            language="en",
            paragraph_number=2,
            custom_system_prompt="Only write cinematic narration.",
        )

        self.assertNotIn("# Role: Video Script Generator", prompt)
        self.assertIn("Only write cinematic narration.", prompt)
        self.assertIn("- video subject: 露营", prompt)
        self.assertIn("- number of paragraphs: 2", prompt)
        self.assertIn("- language: en", prompt)

    def test_generate_script_sends_custom_prompt_to_llm(self):
        captured = {}

        def fake_generate_response(prompt):
            captured["prompt"] = prompt
            return "第一段。\n\n第二段。"

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ):
            result = llm.generate_script(
                video_subject="咖啡",
                language="zh-CN",
                paragraph_number=2,
                video_script_prompt="开头更有悬念",
            )

        self.assertEqual(result, "第一段。\n\n第二段。")
        self.assertIn("- number of paragraphs: 2", captured["prompt"])
        self.assertIn("开头更有悬念", captured["prompt"])

    def test_generate_script_reuses_submitted_config_snapshot(self):
        """La configuración nueva que se aplica al terminar una tarea en segundo plano de
        la WebUI no puede cambiar una solicitud al modelo que se está reintentando.
        """
        captured = {}
        app_config = {
            "llm_provider": "openai",
            "openai_api_key": "snapshot-key",
            "openai_model_name": "snapshot-model",
        }

        def fake_generate_response(prompt, app_config=None):
            captured["prompt"] = prompt
            captured["app_config"] = app_config
            return "Snapshot response"

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ):
            result = llm.generate_script(
                video_subject="Snapshot test",
                app_config=app_config,
            )

        self.assertEqual(result, "Snapshot response")
        self.assertIs(captured["app_config"], app_config)
        self.assertEqual(captured["app_config"]["openai_api_key"], "snapshot-key")

    def test_generate_script_retries_provider_error_instead_of_using_it_as_narration(self):
        with patch.object(
            llm,
            "_generate_response",
            side_effect=["Error: temporary provider failure", "Real narration."],
        ) as generate_response:
            result = llm.generate_script(video_subject="Coffee")

        self.assertEqual(result, "Real narration.")
        self.assertEqual(generate_response.call_count, 2)

    def test_generate_script_returns_empty_when_provider_always_fails(self):
        with patch.object(
            llm, "_generate_response", return_value="Error: invalid API key"
        ):
            result = llm.generate_script(video_subject="Coffee")

        self.assertEqual(result, "")

    def test_generate_script_does_not_return_stale_quota_error_after_retries(self):
        responses = ["当日额度已消耗完"] + [
            RuntimeError("provider unavailable")
        ] * (llm._max_retries - 1)
        with patch.object(
            llm, "_generate_response", side_effect=responses
        ) as generate_response:
            result = llm.generate_script(video_subject="Coffee")

        self.assertEqual(result, "")
        self.assertEqual(generate_response.call_count, llm._max_retries)

    def test_generate_script_strips_each_bracket_group_independently(self):
        """
        format_response debe eliminar cada grupo [corchete] y (paréntesis) de forma
        aislada. La forma codiciosa [.*] coincide desde el primer signo de apertura
        hasta el *último* de cierre de la línea, borrando en silencio todo el texto
        intermedio.

        Ejemplo del error codicioso:
            "[Intro] Great content [end]"  →  "."     (se pierde todo el texto interior)
        Lo esperado con la corrección no codiciosa:
            "[Intro] Great content [end]"  →  " Great content "
        """

        def fake_generate_response(prompt):
            # Dos grupos de corchetes y dos de paréntesis en la misma línea.
            return (
                "[Scene: Beach] A beautiful day at the [location: ocean].\n\n"
                "Save (at least) 10% of your income (monthly)."
            )

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ):
            result = llm.generate_script(video_subject="savings tips", language="en-US")

        # Cada grupo de corchetes / paréntesis debe desaparecer, pero las palabras de
        # alrededor deben sobrevivir.
        self.assertNotIn("[", result)
        self.assertNotIn("]", result)
        self.assertNotIn("(", result)
        self.assertNotIn(")", result)
        self.assertIn("A beautiful day at the", result)
        self.assertIn("10% of your income", result)

    def test_generate_terms_can_request_script_ordered_keywords(self):
        """
        La correspondencia de material en el orden del guion depende de que el LLM
        devuelva palabras clave ordenadas. Aquí no se llama a un modelo real; solo se
        verifica que la capa de servicio escribe en el prompt la restricción de
        "devolverlas en el orden narrativo del guion", para que la descarga de material
        no quede ordenada mientras las palabras clave siguen siendo temas globales sin
        orden.
        """
        captured = {}

        def fake_generate_response(prompt):
            captured["prompt"] = prompt
            return '["opening city", "middle office", "final sunset"]'

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ):
            result = llm.generate_terms(
                video_subject="startup story",
                video_script="First city. Then office. Finally sunset.",
                amount=3,
                match_script_order=True,
            )

        self.assertEqual(result, ["opening city", "middle office", "final sunset"])
        self.assertIn("chronological stock-video search terms", captured["prompt"])
        self.assertIn("same order as the script narration", captured["prompt"])

    def test_generate_terms_returns_empty_list_on_provider_error(self):
        """
        Los errores del proveedor deben mantener el contrato de retorno List[str] de
        generate_terms.

        Una cadena ``Error: ...`` no vacía es un valor verdadero en Python; si se
        devolviera directamente, la capa de tareas la tomaría por palabras clave
        válidas, y la capa de descarga de material podría luego lanzar búsquedas
        carácter a carácter.
        """
        with patch.object(
            llm,
            "_generate_response",
            return_value="Error: invalid API key",
        ):
            result = llm.generate_terms(
                video_subject="startup story",
                video_script="A short startup story.",
            )

        self.assertEqual(result, [])
        self.assertIsInstance(result, list)

    def test_generate_terms_retries_non_string_items_in_recovered_json(self):
        """La ruta de recuperación de JSON envuelto en texto también debe garantizar
        List[str].
        """
        with patch.object(
            llm,
            "_generate_response",
            side_effect=[
                'Search terms: [123, {"query": "coffee"}]',
                'Search terms: ["coffee beans", "barista tools"]',
            ],
        ) as generate_response:
            result = llm.generate_terms(
                video_subject="Coffee",
                video_script="How to brew coffee.",
            )

        self.assertEqual(result, ["coffee beans", "barista tools"])
        self.assertEqual(generate_response.call_count, 2)

    def test_video_script_request_rejects_invalid_advanced_options(self):
        """
        El modelo de solicitud de la API debe limitar los parámetros de prompt
        avanzados, para que las llamadas externas no se salten la WebUI pasando números
        de párrafo anómalos o prompts demasiado largos que vuelvan incontrolables el
        coste y los resultados del modelo.
        """
        with self.assertRaises(ValidationError):
            VideoScriptRequest(video_subject="咖啡", paragraph_number=0)

        with self.assertRaises(ValidationError):
            VideoScriptRequest(
                video_subject="咖啡",
                video_script_prompt="x" * (llm.MAX_SCRIPT_PROMPT_LENGTH + 1),
            )


class TestLLMConnection(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_connection_sends_one_minimal_request(self):
        """La prueba de conexión solo envía una solicitud mínima fija, sin activar los
        reintentos de generación de guiones.
        """
        with (
            patch.object(llm, "_generate_response", return_value="OK") as generate,
            patch.object(llm, "perf_counter", side_effect=[10.0, 10.25]),
        ):
            result = llm.test_connection()

        generate.assert_called_once_with(
            prompt="Reply with exactly: OK",
            request_timeout=llm.LLM_CONNECTION_TEST_TIMEOUT_SECONDS,
            max_retries=llm.LLM_CONNECTION_TEST_MAX_RETRIES,
        )
        self.assertEqual(result, (True, "", 0.25))

    def test_connection_applies_bounded_timeout_and_disables_sdk_retries(self):
        """El tiempo de espera corto y la desactivación de reintentos de la prueba de
        conexión deben aplicarse al cliente real del SDK (issue #1642).
        """
        config.app.update(
            {
                "llm_provider": "openai",
                "openai_api_key": "test-key",
                "openai_model_name": "test-model",
                "openai_base_url": "https://example.test/v1",
            }
        )
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                chat=types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=lambda **kw: None)
                )
            )

        with patch.object(llm, "OpenAI", side_effect=create):
            llm.test_connection()

        self.assertEqual(
            captured.get("timeout"), llm.LLM_CONNECTION_TEST_TIMEOUT_SECONDS
        )
        self.assertEqual(
            captured.get("max_retries"), llm.LLM_CONNECTION_TEST_MAX_RETRIES
        )

    def test_connection_failure_hint_for_timeout_errors(self):
        """Los fallos por tiempo de espera o conexión deben incluir pistas para
        diagnosticar, en lugar de mostrar solo el texto original del SDK.
        """
        with (
            patch.object(
                llm, "_generate_response", return_value="Error: Request timed out."
            ),
            patch.object(llm, "perf_counter", side_effect=[0.0, 5.0]),
        ):
            ok, error, elapsed = llm.test_connection()

        self.assertFalse(ok)
        self.assertEqual(elapsed, 5.0)
        self.assertIn("Request timed out.", error)
        self.assertIn(f"{llm.LLM_CONNECTION_TEST_TIMEOUT_SECONDS:.0f}s", error)
        self.assertIn("accesible", error)
        self.assertIn("proxy", error)

    def test_connection_returns_provider_error(self):
        """Cuando el proveedor devuelve un error hay que conservar la información de
        diagnóstico e informar del tiempo que tardó la solicitud.
        """
        with (
            patch.object(
                llm,
                "_generate_response",
                return_value="Error: invalid API key",
            ),
            patch.object(llm, "perf_counter", side_effect=[20.0, 20.5]),
        ):
            result = llm.test_connection()

        self.assertEqual(result, (False, "invalid API key", 0.5))

    def test_connection_rejects_empty_response(self):
        """Una respuesta vacía en casos extremos debe mostrar un error claro, en lugar de
        informar por error de que la conexión funciona.
        """
        with (
            patch.object(llm, "_generate_response", return_value=""),
            patch.object(llm, "perf_counter", side_effect=[30.0, 31.0]),
        ):
            result = llm.test_connection()

        self.assertEqual(result, (False, "el LLM devolvió una respuesta vacía", 1.0))


class TestGenerateResponseTransportLimits(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    @staticmethod
    def _fake_client():
        """Devuelve un cliente mínimo funcional para que `_generate_response` recorra la
        rama de solicitud.
        """
        message = types.SimpleNamespace(content="ok")
        choice = types.SimpleNamespace(message=message)
        completion = types.SimpleNamespace(choices=[choice])
        return types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kwargs: completion)
            )
        )

    def test_transport_limits_forwarded_to_openai_compatible_clients(self):
        """Las restricciones de tiempo de espera/reintentos pasadas explícitamente deben
        aplicarse a todos los clientes compatibles con OpenAI.
        """
        cases = [
            (
                "openai",
                "OpenAI",
                {
                    "openai_api_key": "test-key",
                    "openai_model_name": "test-model",
                    "openai_base_url": "https://example.test/v1",
                },
            ),
            (
                "azure",
                "AzureOpenAI",
                {
                    "azure_api_key": "test-key",
                    "azure_model_name": "test-model",
                    "azure_base_url": "https://example.test",
                    "azure_api_version": "2024-02-15-preview",
                },
            ),
            (
                "cloudflare",
                "OpenAI",
                {
                    "cloudflare_api_key": "test-key",
                    "cloudflare_model_name": "test-model",
                    "cloudflare_account_id": "test-account",
                    "cloudflare_gateway_id": "test-gateway",
                },
            ),
        ]
        for provider, constructor, settings in cases:
            with self.subTest(provider=provider):
                config.app.update({"llm_provider": provider, **settings})
                with (
                    patch.object(
                        llm,
                        constructor,
                        return_value=self._fake_client(),
                    ) as client_constructor,
                    patch.object(llm, "ChatCompletion", types.SimpleNamespace),
                ):
                    result = llm._generate_response(
                        "hello",
                        request_timeout=12.5,
                        max_retries=0,
                    )

                self.assertEqual(result, "ok", result)
                kwargs = client_constructor.call_args.kwargs
                self.assertEqual(kwargs.get("timeout"), 12.5)
                self.assertEqual(kwargs.get("max_retries"), 0)

    def test_default_generation_keeps_sdk_defaults(self):
        """Sin restricciones, los parámetros de construcción del cliente se mantienen tal
        cual y el comportamiento de la cadena de generación no cambia.
        """
        config.app.update(
            {
                "llm_provider": "openai",
                "openai_api_key": "test-key",
                "openai_model_name": "test-model",
                "openai_base_url": "https://example.test/v1",
            }
        )
        with (
            patch.object(
                llm, "OpenAI", return_value=self._fake_client()
            ) as client_constructor,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("hello")

        self.assertEqual(result, "ok", result)
        client_constructor.assert_called_once_with(
            api_key="test-key",
            base_url="https://example.test/v1",
        )


class TestLiteLLMProvider(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_current_default_model_names(self):
        """La WebUI y la capa de servicio deben compartir el mismo conjunto de modelos por
        defecto, para que el valor mostrado y el valor pedido no se desvíen.
        """
        self.assertEqual(get_llm_provider("openai").default_model, "gpt-5.5")
        anthropic = get_llm_provider("anthropic")
        self.assertEqual(anthropic.default_model, "claude-sonnet-5")
        self.assertEqual(anthropic.default_base_url, "https://api.anthropic.com/v1/")
        self.assertEqual(anthropic.adapter, "openai_compatible")
        self.assertTrue(anthropic.requires_api_key)
        self.assertEqual(get_llm_provider("aimlapi").default_model, "openai/gpt-5-5")
        self.assertEqual(get_llm_provider("deepseek").default_model, "deepseek-v4-pro")
        self.assertEqual(
            get_llm_provider("modelscope").default_model, "ZhipuAI/GLM-5.2"
        )
        self.assertEqual(
            get_llm_provider("gemini").default_model, "gemini-3.1-pro-preview"
        )
        openrouter = get_llm_provider("openrouter")
        self.assertEqual(openrouter.default_model, "minimax/minimax-m3:free")
        self.assertEqual(openrouter.default_base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(openrouter.adapter, "openai_compatible")
        self.assertTrue(openrouter.requires_api_key)
        api_route = get_llm_provider("api_route")
        self.assertEqual(api_route.default_model, "gpt-5.4-mini")
        self.assertEqual(api_route.default_base_url, "https://www.api-route.com/v1")
        self.assertEqual(api_route.adapter, "openai_compatible")
        self.assertTrue(api_route.requires_api_key)
        cheaperinference = get_llm_provider("cheaperinference")
        self.assertEqual(cheaperinference.default_model, "gpt-5.4-mini")
        self.assertEqual(
            cheaperinference.default_base_url,
            "https://api.cheaperinference.com/v1",
        )
        self.assertEqual(cheaperinference.adapter, "openai_compatible")
        self.assertTrue(cheaperinference.requires_api_key)
        self.assertEqual(
            cheaperinference.api_key_url,
            "https://cheaperinference.com/signup",
        )
        self.assertEqual(
            cheaperinference.model_docs_url,
            "https://cheaperinference.com/#models",
        )
        requesty = get_llm_provider("requesty")
        self.assertEqual(requesty.default_model, "openai/gpt-5.4-mini")
        self.assertEqual(requesty.default_base_url, "https://router.requesty.ai/v1")
        self.assertEqual(requesty.adapter, "openai_compatible")
        self.assertTrue(requesty.requires_api_key)
        self.assertEqual(requesty.api_key_url, "https://app.requesty.ai/api-keys")
        self.assertEqual(requesty.model_docs_url, "https://www.requesty.ai/models")
        futureinfra = get_llm_provider("futureinfra")
        self.assertEqual(futureinfra.default_model, "openai/gpt-4o-mini")
        self.assertEqual(futureinfra.default_base_url, "https://futureinfra.ai/v1/ai")
        self.assertEqual(futureinfra.adapter, "openai_compatible")
        self.assertTrue(futureinfra.requires_api_key)
        self.assertEqual(
            futureinfra.api_key_url,
            "https://futureinfra.ai/console/?screen=ai-router",
        )
        self.assertEqual(futureinfra.model_docs_url, "https://futureinfra.ai/ai/")
        yapi = get_llm_provider("yapi")
        self.assertEqual(yapi.default_model, "deepseek/deepseek-v4-flash")
        self.assertEqual(yapi.default_base_url, "https://api.y-api.bestvirtualgoods.com/v1")
        self.assertEqual(yapi.adapter, "openai_compatible")
        self.assertTrue(yapi.requires_api_key)
        self.assertEqual(
            yapi.api_key_url,
            "https://y-api.bestvirtualgoods.com/app/keys",
        )
        self.assertEqual(yapi.model_docs_url, "https://y-api.bestvirtualgoods.com/models")
        opper = get_llm_provider("opper")
        self.assertEqual(opper.default_model, "gpt-5.4-mini")
        self.assertEqual(opper.default_base_url, "https://api.opper.ai/v3/compat")
        self.assertEqual(opper.adapter, "openai_compatible")
        self.assertTrue(opper.requires_api_key)
        self.assertEqual(opper.api_key_url, "https://platform.opper.ai")
        self.assertEqual(opper.model_docs_url, "https://opper.ai/models")
        onomeo = get_llm_provider("onomeo")
        self.assertEqual(onomeo.default_model, "auto")
        self.assertEqual(onomeo.default_base_url, "https://onomeo.com/v1")
        self.assertEqual(onomeo.adapter, "openai_compatible")
        self.assertTrue(onomeo.requires_api_key)
        self.assertEqual(onomeo.api_key_url, "https://onomeo.com/dashboard")
        self.assertEqual(onomeo.model_docs_url, "https://onomeo.com/models")
        empiriolabs = get_llm_provider("empiriolabs")
        self.assertEqual(empiriolabs.default_model, "qwen3-8-flash")
        self.assertEqual(empiriolabs.default_base_url, "https://api.empiriolabs.ai/v1")
        self.assertEqual(empiriolabs.adapter, "openai_compatible")
        self.assertTrue(empiriolabs.requires_api_key)
        self.assertEqual(
            empiriolabs.api_key_url,
            "https://platform.empiriolabs.ai/dashboard/api-keys",
        )
        self.assertEqual(
            empiriolabs.model_docs_url,
            "https://docs.empiriolabs.ai/models-pricing",
        )
        iflytek = get_llm_provider("iflytek")
        self.assertEqual(iflytek.default_model, "spark-x2.5")
        self.assertEqual(
            iflytek.default_base_url,
            "https://maas-api.cn-huabei-1.xf-yun.com/v2",
        )
        self.assertEqual(iflytek.adapter, "openai_compatible")
        self.assertTrue(iflytek.requires_api_key)
        self.assertEqual(iflytek.api_key_url, "https://maas.xfyun.cn/")
        self.assertEqual(iflytek.model_docs_url, "https://maas.xfyun.cn/modelSquare")
        pollinations = get_llm_provider("pollinations")
        self.assertEqual(pollinations.default_model, "openai-fast")
        self.assertEqual(
            pollinations.default_base_url,
            "https://gen.pollinations.ai/v1",
        )
        self.assertTrue(pollinations.requires_api_key)
        self.assertEqual(pollinations.adapter, "openai_compatible")

    def test_provider_defaults_are_not_persisted_as_user_overrides(self):
        """Los valores por defecto solo sirven para ejecutar y mostrar; solo los valores
        distintos deben escribirse en la configuración del usuario.
        """
        self.assertEqual(
            normalize_provider_override("gpt-5.5", "gpt-5.5"),
            "",
        )
        self.assertEqual(
            normalize_provider_override("  gpt-5.5  ", "gpt-5.5"),
            "",
        )
        self.assertEqual(
            normalize_provider_override("gpt-5.6-custom", "gpt-5.5"),
            "gpt-5.6-custom",
        )

    def test_provider_registry_has_unique_stable_ids(self):
        """El registro es la única fuente de datos de la lista de proveedores; los ID deben
        ser únicos y debe existir la opción por defecto.
        """
        provider_ids = [provider.provider_id for provider in LLM_PROVIDER_REGISTRY]

        self.assertEqual(len(provider_ids), len(set(provider_ids)))
        self.assertEqual(len(provider_ids), len(LLM_PROVIDERS))
        self.assertIn(DEFAULT_LLM_PROVIDER_ID, LLM_PROVIDERS)

    def test_provider_registry_preserves_product_group_order(self):
        """El orden del desplegable es: recomendados, fabricantes originales, plataformas
        de agregación, despliegue local y otros servicios.
        """
        self.assertEqual(
            [provider.provider_id for provider in LLM_PROVIDER_REGISTRY],
            [
                "moonshot",
                "kimi_code",
                "openai",
                "anthropic",
                "gemini",
                "muse_spark",
                "deepseek",
                "qwen",
                "azure",
                "volcengine",
                "grok",
                "minimax",
                "mimo",
                "iflytek",
                "infistar",
                "shengsuanyun",
                "apimart",
                "cloudflare",
                "modelscope",
                "aihubmix",
                "aimlapi",
                "evolink",
                "openrouter",
                "api_route",
                "fluxionai",
                "cheaperinference",
                "requesty",
                "futureinfra",
                "yapi",
                "opper",
                "onomeo",
                "empiriolabs",
                "ollama",
                "claude_code",
                "oneapi",
                "litellm",
                "groq",
                "pollinations",
            ],
        )
        self.assertEqual(
            get_llm_provider("gemini").default_label,
            "Google Gemini",
        )
        self.assertEqual(
            get_llm_provider("azure").default_label,
            "Microsoft Azure OpenAI",
        )
        shengsuanyun = get_llm_provider("shengsuanyun")
        self.assertEqual(
            shengsuanyun.api_key_url,
            "https://www.shengsuanyun.com/?from=CH_XUQ4OTSK",
        )
        self.assertEqual(
            shengsuanyun.default_model,
            "deepseek/deepseek-v4-flash",
        )
        apimart = get_llm_provider("apimart")
        self.assertEqual(
            apimart.api_key_url,
            "https://go.apimart.ai/gh-moneyprinterturbo",
        )
        self.assertEqual(apimart.default_model, "gpt-5.6-terra")
        self.assertEqual(apimart.default_base_url, "https://api.apimart.ai/v1")
        muse_spark = get_llm_provider("muse_spark")
        self.assertEqual(
            muse_spark.api_key_url,
            "https://dev.meta.ai/docs/authentication",
        )
        self.assertEqual(muse_spark.default_model, "muse-spark-1.3")
        self.assertEqual(muse_spark.default_base_url, "https://api.meta.ai/v1")
        self.assertEqual(muse_spark.adapter, "openai_compatible")
        openrouter = get_llm_provider("openrouter")
        self.assertEqual(
            openrouter.api_key_url,
            "https://openrouter.ai/settings/keys",
        )
        self.assertEqual(openrouter.default_model, "minimax/minimax-m3:free")
        self.assertEqual(openrouter.default_base_url, "https://openrouter.ai/api/v1")
        api_route = get_llm_provider("api_route")
        self.assertEqual(
            api_route.api_key_url,
            "https://www.api-route.com",
        )
        self.assertEqual(api_route.default_model, "gpt-5.4-mini")
        self.assertEqual(api_route.default_base_url, "https://www.api-route.com/v1")
        self.assertEqual(
            api_route.model_docs_url,
            "https://www.api-route.com/pricing",
        )

    def test_provider_registry_uses_conventional_locale_and_config_keys(self):
        """Una regla de nombres común evita que la WebUI añada mapeos fijos para cada
        proveedor.
        """
        for provider in LLM_PROVIDER_REGISTRY:
            self.assertEqual(
                provider.label_key,
                f"llm_provider_label.{provider.provider_id}",
            )
            self.assertEqual(
                provider.tips_key,
                f"llm_provider_tips.{provider.provider_id}",
            )
            self.assertEqual(
                provider.config_key("api_key"),
                f"{provider.provider_id}_api_key",
            )

    def test_registry_replaces_deprecated_provider_models(self):
        """Los modelos por defecto históricos deben migrarse automáticamente, para no
        seguir usando tras actualizar una semántica de acceso ya retirada.
        """
        cloudflare = get_llm_provider("cloudflare")
        gemini = get_llm_provider("gemini")

        self.assertEqual(
            cloudflare.resolve_model_name("@cf/meta/llama-3.1-8b-instruct"),
            "openai/gpt-4.1-mini",
        )
        self.assertEqual(
            gemini.resolve_model_name("gemini-pro"),
            "gemini-3.1-pro-preview",
        )
        self.assertEqual(
            cloudflare.resolve_model_name("anthropic/claude-sonnet-4-5"),
            "anthropic/claude-sonnet-4-5",
        )

        pollinations = get_llm_provider("pollinations")
        self.assertEqual(
            pollinations.resolve_model_name("default"),
            "openai-fast",
        )
        self.assertEqual(
            pollinations.resolve_base_url("https://text.pollinations.ai/openai"),
            "https://gen.pollinations.ai/v1",
        )
        self.assertEqual(
            pollinations.resolve_base_url("https://example.com/v1"),
            "https://example.com/v1",
        )

    def test_provider_tip_templates_accept_registry_defaults(self):
        """Las plantillas de avisos de proveedores deben poder inyectar con seguridad los
        valores por defecto del registro.
        """
        i18n_dir = Path(__file__).parent.parent.parent / "webui" / "i18n"
        for locale_file in i18n_dir.glob("*.json"):
            translations = json.loads(locale_file.read_text(encoding="utf-8"))[
                "Translation"
            ]
            for provider in LLM_PROVIDER_REGISTRY:
                tips = translations.get(provider.tips_key, "")
                if not tips:
                    continue
                default_endpoint = provider.default_service_endpoint
                rendered = tips.format(
                    api_key_url=provider.effective_api_key_url(),
                    default_model=provider.default_model,
                    default_base_url=provider.effective_default_base_url,
                    model_docs_url=(
                        default_endpoint.model_docs_url
                        if default_endpoint
                        else provider.effective_model_docs_url()
                    ),
                    docker_hint="",
                    **{
                        f"default_{field.config_suffix}": field.default_value
                        for field in provider.extra_fields
                    },
                )
                self.assertNotIn("{default_model}", rendered)
                self.assertNotIn("{default_base_url}", rendered)

    def test_primary_provider_tips_use_consistent_structure(self):
        """Las instrucciones de configuración muestran de forma uniforme la clave API, la
        URL base y el nombre del modelo.
        """
        i18n_dir = Path(__file__).parent.parent.parent / "webui" / "i18n"
        translations = json.loads(
            (i18n_dir / "es.json").read_text(encoding="utf-8")
        )["Translation"]
        for provider in LLM_PROVIDER_REGISTRY:
            tips = translations[provider.tips_key]
            self.assertTrue(tips.startswith("##### "), provider.provider_id)
            self.assertIn("**Clave API**", tips, provider.provider_id)
            self.assertIn("**URL base**", tips, provider.provider_id)
            self.assertIn("**Nombre del modelo**", tips, provider.provider_id)

        kimi_tips = translations["llm_provider_tips.moonshot"]
        self.assertIn("Por qué se recomienda:", kimi_tips)
        self.assertIn("Encaja en el flujo de creación de vídeos", kimi_tips)
        self.assertIn("la oferta termina el 31 de diciembre de 2026", kimi_tips)

    def test_required_api_key_providers_have_clickable_entry_points(self):
        """Los proveedores que necesitan clave deben dar una entrada común para
        solicitarla, para que la WebUI no dé solo texto.
        """
        i18n_dir = Path(__file__).parent.parent.parent / "webui" / "i18n"
        locale_translations = {
            locale_file.stem: json.loads(locale_file.read_text(encoding="utf-8"))[
                "Translation"
            ]
            for locale_file in i18n_dir.glob("*.json")
        }

        for provider in LLM_PROVIDER_REGISTRY:
            if provider.requires_api_key:
                api_key_url = provider.effective_api_key_url()
                self.assertTrue(api_key_url, provider.provider_id)
                self.assertTrue(
                    api_key_url.startswith("https://"),
                    provider.provider_id,
                )
                for language, translations in locale_translations.items():
                    tips_template = translations.get(provider.tips_key, "")
                    if not tips_template:
                        continue
                    default_endpoint = provider.default_service_endpoint
                    tips = tips_template.format(
                        api_key_url=api_key_url,
                        default_model=provider.default_model,
                        default_base_url=provider.effective_default_base_url,
                        model_docs_url=(
                            default_endpoint.model_docs_url
                            if default_endpoint
                            else provider.effective_model_docs_url()
                        ),
                        docker_hint="",
                        **{
                            f"default_{field.config_suffix}": field.default_value
                            for field in provider.extra_fields
                        },
                    )
                    api_key_line = next(
                        line for line in tips.splitlines() if "**Clave API**" in line
                    )
                    self.assertIn("](", api_key_line, provider.provider_id)
                    self.assertIn(
                        f"]({api_key_url})",
                        api_key_line,
                        f"{language}: {provider.provider_id}",
                    )

    def test_service_endpoint_registry_references_valid_stable_ids(self):
        """Las regiones de servicio deben asociarse por un ID único y estable, sin depender
        de enlaces ni de textos visibles.
        """
        for provider in LLM_PROVIDER_REGISTRY:
            endpoint_ids = [
                endpoint.endpoint_id for endpoint in provider.service_endpoints
            ]
            self.assertEqual(
                len(endpoint_ids),
                len(set(endpoint_ids)),
                provider.provider_id,
            )
            if not endpoint_ids:
                self.assertFalse(provider.default_service_endpoint_id)
                self.assertFalse(provider.international_service_endpoint_id)
                continue

            self.assertIn(provider.default_service_endpoint_id, endpoint_ids)
            if provider.international_service_endpoint_id:
                self.assertIn(provider.international_service_endpoint_id, endpoint_ids)

    def test_kimi_service_endpoint_selection_preserves_existing_configs(self):
        """Una configuración de Kimi existente no puede cambiarse en silencio a otro
        sistema de cuentas por cambiar el idioma de la interfaz.
        """
        provider = get_llm_provider("moonshot")

        china = provider.select_service_endpoint(
            "",
            has_api_key=True,
            prefer_international=True,
        )
        global_endpoint = provider.select_service_endpoint(
            "https://api.moonshot.ai/v1/",
            has_api_key=True,
            prefer_international=False,
        )

        self.assertEqual(china.endpoint_id, "china")
        self.assertEqual(global_endpoint.endpoint_id, "global")
        self.assertIsNone(
            provider.select_service_endpoint(
                "https://gateway.example.com/v1",
                has_api_key=True,
                prefer_international=True,
            )
        )

    def test_kimi_fresh_config_uses_interface_region(self):
        """Una configuración nueva recomienda un sitio, pero sigue siendo el usuario quien
        lo elige explícitamente en la WebUI.
        """
        provider = get_llm_provider("moonshot")

        china = provider.select_service_endpoint(
            "",
            has_api_key=False,
            prefer_international=False,
        )
        global_endpoint = provider.select_service_endpoint(
            "",
            has_api_key=False,
            prefer_international=True,
        )

        self.assertEqual(china.base_url, "https://api.moonshot.cn/v1")
        self.assertEqual(global_endpoint.base_url, "https://api.moonshot.ai/v1")
        self.assertIn("platform.kimi.ai", global_endpoint.api_key_url)

    def test_kimi_endpoint_selection_does_not_depend_on_marketing_url(self):
        """Actualizar los parámetros de promoción no puede cambiar el resultado de la
        elección de negocio del sitio internacional.
        """
        provider = get_llm_provider("moonshot")
        global_endpoint = replace(
            provider.international_service_endpoint,
            api_key_url="https://platform.kimi.ai/?new-tracking=1",
        )
        updated_provider = replace(
            provider,
            service_endpoints=tuple(
                global_endpoint if endpoint.endpoint_id == "global" else endpoint
                for endpoint in provider.service_endpoints
            ),
        )

        selected = updated_provider.select_service_endpoint(
            "",
            has_api_key=False,
            prefer_international=True,
        )

        self.assertEqual(selected.endpoint_id, "global")
        self.assertEqual(selected.api_key_url, global_endpoint.api_key_url)

    def test_example_config_does_not_duplicate_registry_defaults(self):
        """La configuración de ejemplo solo guarda los valores sobrescritos por el usuario;
        el modelo y la dirección por defecto los mantiene únicamente el registro.
        """
        config_path = Path(__file__).parent.parent.parent / "config.example.toml"
        app_config = tomllib.loads(config_path.read_text(encoding="utf-8"))["app"]

        for provider in LLM_PROVIDER_REGISTRY:
            if provider.default_model:
                self.assertEqual(
                    app_config.get(provider.config_key("model_name"), ""),
                    "",
                    provider.provider_id,
                )
            if provider.effective_default_base_url:
                self.assertEqual(
                    app_config.get(provider.config_key("base_url"), ""),
                    "",
                    provider.provider_id,
                )
            for field in provider.extra_fields:
                if field.default_value:
                    self.assertEqual(
                        app_config.get(provider.config_key(field.config_suffix), ""),
                        "",
                        provider.provider_id,
                    )

    def test_removed_ernie_provider_is_unsupported(self):
        """Tras eliminar ERNIE, una configuración residual debe devolver un error claro,
        sin volver a lanzar la antigua solicitud OAuth.
        """
        config.app["llm_provider"] = "ernie"

        with patch.object(llm, "OpenAI") as openai_client:
            result = llm._generate_response("test")

        openai_client.assert_not_called()
        self.assertIn("proveedor LLM no admitido", result)

    def test_pollinations_requires_api_key_before_request(self):
        """La nueva API unificada exige autenticación; si falta la clave no se pueden
        enviar solicitudes de generación anónimas.
        """
        config.app.update(
            {
                "llm_provider": "pollinations",
                "pollinations_api_key": "",
                "pollinations_base_url": "",
                "pollinations_model_name": "",
            }
        )

        with patch.object(llm, "OpenAI") as openai_client:
            result = llm._generate_response("test")

        openai_client.assert_not_called()
        self.assertIn("api_key no está configurada", result)

    def test_pollinations_uses_unified_openai_compatible_api(self):
        """Las direcciones y nombres de modelo históricos deben migrarse automáticamente y
        llamarse a través de la API unificada de Chat Completions.
        """
        config.app.update(
            {
                "llm_provider": "pollinations",
                "pollinations_api_key": "pollinations-test-key",
                "pollinations_base_url": "https://text.pollinations.ai/openai/",
                "pollinations_model_name": "default",
            }
        )

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\npollinations")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="pollinations-test-key",
            base_url="https://gen.pollinations.ai/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "openai-fast",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\npollinations")

    def test_anthropic_uses_openai_compatible_chat_completions(self):
        """Claude usa el endpoint compatible con OpenAI de Anthropic y no necesita una rama
        de adaptador adicional.
        """
        config.app.update(
            {
                "llm_provider": "anthropic",
                "anthropic_api_key": "anthropic-test-key",
                "anthropic_base_url": "",
                "anthropic_model_name": "",
            }
        )

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nclaude")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="anthropic-test-key",
            base_url="https://api.anthropic.com/v1/",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "claude-sonnet-5",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\nclaude")

    def test_gemini_uses_google_genai_client(self):
        """El adaptador de Gemini debe lanzar las solicitudes de generación de contenido a
        través del Client unificado del SDK nuevo.
        """
        config.app.update(
            {
                "llm_provider": "gemini",
                "gemini_api_key": "gemini-test-key",
                "gemini_base_url": "",
                "gemini_model_name": "gemini-test-model",
            }
        )
        captured = {}

        class FakeModels:
            def generate_content(self, **kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(text="hello\ngemini")

        class FakeClient:
            def __init__(self, **kwargs):
                captured["client_kwargs"] = kwargs
                self.models = FakeModels()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                captured["closed"] = True

        with patch("google.genai.Client", FakeClient):
            result = llm._generate_response("Say hello")

        self.assertEqual(result, "hello\ngemini")
        self.assertEqual(
            captured["client_kwargs"],
            {"api_key": "gemini-test-key", "http_options": None},
        )
        self.assertEqual(captured["model"], "gemini-test-model")
        self.assertEqual(captured["contents"], "Say hello")
        self.assertEqual(captured["config"].max_output_tokens, 2048)
        self.assertTrue(captured["closed"])

    def test_cloudflare_requires_account_id_before_request(self):
        """Si a Cloudflare le falta el Account ID, debe fallar en local sin enviar
        solicitudes no válidas.
        """
        config.app.update(
            {
                "llm_provider": "cloudflare",
                "cloudflare_api_key": "test-token",
                "cloudflare_account_id": "",
                "cloudflare_model_name": "",
            }
        )

        with patch.object(llm, "OpenAI") as openai_client:
            result = llm._generate_response("test")

        openai_client.assert_not_called()
        self.assertIn("account_id no está configurado", result)

    def test_cloudflare_uses_ai_gateway_openai_endpoint(self):
        """El proveedor de Cloudflare debe ir por AI Gateway y ya no llamar a la interfaz
        de Workers AI.
        """
        config.app.update(
            {
                "llm_provider": "cloudflare",
                "cloudflare_api_key": "cloudflare-token",
                "cloudflare_account_id": "account-123",
                "cloudflare_gateway_id": "",
                "cloudflare_model_name": "",
            }
        )

        fake_response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="gateway\nresponse")
                )
            ]
        )

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return fake_response

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="cloudflare-token",
            base_url=(
                "https://api.cloudflare.com/client/v4/accounts/account-123/ai/v1"
            ),
            default_headers={"cf-aig-gateway-id": "default"},
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "openai/gpt-4.1-mini",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "gateway\nresponse")

    def _use_litellm_provider(self, model_name="openai/gpt-4o-mini"):
        config.app["llm_provider"] = "litellm"
        config.app["litellm_model_name"] = model_name

    def test_litellm_provider_returns_normalized_text(self):
        """
        Verifica que la ruta principal del proveedor LiteLLM no depende de la red real
        ni de claves API privadas.

        Aquí se inyecta un módulo falso en `sys.modules` que sustituye directamente el
        `litellm.completion()` importado dinámicamente, para que el test cubra de forma
        estable la rama litellm de `_generate_response()`.
        """
        self._use_litellm_provider()

        fake_litellm = types.SimpleNamespace()

        def _completion(**kwargs):
            self.assertEqual(kwargs["model"], "openai/gpt-4o-mini")
            self.assertEqual(
                kwargs["messages"], [{"role": "user", "content": "Say hello"}]
            )
            self.assertTrue(kwargs["drop_params"])
            message = types.SimpleNamespace(content="hello\nworld")
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

        fake_litellm.completion = _completion

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = llm._generate_response("Say hello")

        self.assertEqual(result, "hello\nworld")

    def test_litellm_provider_uses_registry_default_model(self):
        self._use_litellm_provider(model_name="")

        fake_litellm = types.SimpleNamespace()

        def _completion(**kwargs):
            self.assertEqual(kwargs["model"], "openai/gpt-4o-mini")
            message = types.SimpleNamespace(content="default model")
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

        fake_litellm.completion = _completion

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = llm._generate_response("test")

        self.assertEqual(result, "default model")

    def test_litellm_provider_handles_empty_response(self):
        self._use_litellm_provider()

        fake_litellm = types.SimpleNamespace(
            completion=lambda **kwargs: types.SimpleNamespace(choices=[])
        )

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("devolvió una respuesta vacía", result)

    def test_litellm_provider_handles_empty_message(self):
        """
        Algunos gateways compatibles con OpenAI devuelven HTTP 200 al filtrar contenido
        o bloquear por seguridad, pero con `choices[0].message` a None. Aquí hay que
        devolver un error diagnosticable, en lugar de lanzar AttributeError.
        """
        self._use_litellm_provider()

        fake_litellm = types.SimpleNamespace(
            completion=lambda **kwargs: types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=None)]
            )
        )

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("devolvió un mensaje vacío", result)

    def test_sanitize_error_message_redacts_url_credentials_and_query_tokens(self):
        message = (
            "request failed for "
            "https://myuser:mypassword@proxy.example.com/v1/chat"
            "?api_key=secret-key&token=secret-token&safe=value"
        )

        result = llm._sanitize_error_message(message)

        self.assertIn("https://***:***@proxy.example.com", result)
        self.assertIn("api_key=***", result)
        self.assertIn("token=***", result)
        self.assertIn("safe=value", result)
        self.assertNotIn("myuser", result)
        self.assertNotIn("mypassword", result)
        self.assertNotIn("secret-key", result)
        self.assertNotIn("secret-token", result)

    def test_sanitize_error_message_redacts_token_only_url_userinfo(self):
        # Los gateways suelen aceptar el token de la API como usuario de la URL sin
        # contraseña, p. ej. https://<token>@gateway/v1.
        message = (
            "Connection error to https://sk-live-SECRET@gw.example.com/v1/chat; "
            "see https://gw.example.com/status?page=1 or ops@example.com"
        )

        result = llm._sanitize_error_message(message)

        self.assertIn("https://***:***@gw.example.com/v1/chat", result)
        self.assertNotIn("sk-live-SECRET", result)
        self.assertIn("https://gw.example.com/status?page=1", result)
        self.assertIn("ops@example.com", result)

    def test_openai_provider_error_redacts_embedded_base_url_credentials(self):
        """
        Una base_url personalizada compatible con OpenAI puede contener el user:pass de
        un gateway proxy. Cuando el SDK lanza un error suele devolver la URL en el
        mensaje de la excepción; aquí se verifica que el texto `Error:` que se devuelve
        finalmente a la WebUI/API no filtra esas credenciales.
        """
        config.app["llm_provider"] = "groq"
        config.app["groq_api_key"] = "groq-key"
        config.app["groq_model_name"] = "openai/gpt-oss-120b"
        config.app["groq_base_url"] = (
            "https://myuser:mypassword@proxy.example.com/openai/v1"
        )

        class FakeCompletions:
            def create(self, **kwargs):
                raise RuntimeError(
                    "connection failed: "
                    "https://myuser:mypassword@proxy.example.com/openai/v1"
                    "?access_token=secret-token"
                )

        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletions())
        )

        with patch.object(llm, "OpenAI", return_value=fake_client):
            result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("https://***:***@proxy.example.com", result)
        self.assertIn("access_token=***", result)
        self.assertNotIn("myuser", result)
        self.assertNotIn("mypassword", result)
        self.assertNotIn("secret-token", result)

    def test_openai_provider_still_uses_existing_path(self):
        config.app["llm_provider"] = "openai"
        config.app["openai_api_key"] = ""
        config.app["openai_base_url"] = "https://api.openai.com/v1"
        config.app["openai_model_name"] = "gpt-4o-mini"

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("api_key no está configurada", result)
        self.assertNotIn("litellm", result.lower())

    def _use_qwen_provider(self):
        config.app["llm_provider"] = "qwen"
        config.app["qwen_api_key"] = "qwen-key"
        config.app["qwen_model_name"] = "qwen-max"

    def _patch_dashscope_generation(self, response):
        class FakeGenerationResponse(dict):
            pass

        fake_response = FakeGenerationResponse(response)
        fake_response.status_code = response.get("status_code", 200)
        fake_dashscope = types.SimpleNamespace(
            api_key="",
            Generation=types.SimpleNamespace(call=lambda **kwargs: fake_response),
        )
        fake_dashscope_response = types.SimpleNamespace(
            GenerationResponse=FakeGenerationResponse
        )

        return patch.dict(
            sys.modules,
            {
                "dashscope": fake_dashscope,
                "dashscope.api_entities": types.SimpleNamespace(),
                "dashscope.api_entities.dashscope_response": fake_dashscope_response,
            },
        )

    def test_qwen_provider_reads_chat_choices_content(self):
        """
        El modo chat de DashScope pone el texto en `output.choices[0].message.content`.
        Aquí se cubre el caso `output.text is None` del issue #966, para que no vuelva a
        producirse `'NoneType' object has no attribute 'replace'`.
        """
        self._use_qwen_provider()
        response = {
            "output": {
                "text": None,
                "choices": [{"message": {"content": "你好\n世界"}}],
            }
        }

        with self._patch_dashscope_generation(response):
            result = llm._generate_response("Say hello")

        self.assertEqual(result, "你好\n世界")

    def test_qwen_provider_falls_back_to_output_text(self):
        """Mantiene la ruta de compatibilidad con la estructura de respuesta completion
        antigua de DashScope.
        """
        self._use_qwen_provider()
        response = {"output": {"text": "旧格式\n响应"}}

        with self._patch_dashscope_generation(response):
            result = llm._generate_response("Say hello")

        self.assertEqual(result, "旧格式\n响应")

    def test_qwen_provider_reports_empty_text(self):
        """Una respuesta vacía de Qwen debe devolver un error diagnosticable, en lugar del
        AttributeError subyacente.
        """
        self._use_qwen_provider()
        response = {
            "output": {"text": None, "choices": [{"message": {"content": None}}]}
        }

        with self._patch_dashscope_generation(response):
            result = llm._generate_response("Say hello")

        self.assertIn("Error:", result)
        self.assertIn("devolvió contenido de texto vacío", result)
        self.assertNotIn("NoneType", result)

    def test_qwen_provider_reports_empty_choices(self):
        """Cuando las choices de una respuesta chat de Qwen están vacías, debe devolverse
        un error claro.
        """
        self._use_qwen_provider()
        response = {"output": {"text": None, "choices": []}}

        with self._patch_dashscope_generation(response):
            result = llm._generate_response("Say hello")

        self.assertIn("Error:", result)
        self.assertIn("devolvió choices vacío", result)
        self.assertNotIn("NoneType", result)

    def test_qwen_concurrent_snapshots_keep_their_own_api_keys(self):
        class FakeGenerationResponse(dict):
            status_code = 200

        barrier = Barrier(2, timeout=5)
        calls = {}
        fake_dashscope = types.SimpleNamespace(api_key="unrelated-global-key")

        def call(**kwargs):
            barrier.wait()
            prompt = kwargs["messages"][0]["content"]
            calls[prompt] = kwargs.get("api_key")
            return FakeGenerationResponse({"output": {"text": prompt}})

        fake_dashscope.Generation = types.SimpleNamespace(call=call)
        modules = {
            "dashscope": fake_dashscope,
            "dashscope.api_entities": types.SimpleNamespace(),
            "dashscope.api_entities.dashscope_response": types.SimpleNamespace(
                GenerationResponse=FakeGenerationResponse
            ),
        }
        with patch.dict(sys.modules, modules), ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                llm._generate_response,
                "first prompt",
                {"llm_provider": "qwen", "qwen_api_key": "first-key"},
            )
            second = pool.submit(
                llm._generate_response,
                "second prompt",
                {"llm_provider": "qwen", "qwen_api_key": "second-key"},
            )
            self.assertEqual(first.result(), "first prompt")
            self.assertEqual(second.result(), "second prompt")

        self.assertEqual(
            calls, {"first prompt": "first-key", "second prompt": "second-key"}
        )
        self.assertEqual(fake_dashscope.api_key, "unrelated-global-key")

    def test_apimart_provider_uses_unwrapped_openai_compatible_endpoint(self):
        """
        La documentación de APIMart muestra a la vez las entradas `/api/v1` y `/v1`. Las
        respuestas de ejemplo de la primera llevan un envoltorio exterior code/data y el
        SDK de OpenAI no puede leer choices directamente del nivel superior; el
        proveedor LLM debe usar la dirección estándar `/v1` para reutilizar la cadena de
        análisis de respuestas existente.
        """
        config.app["llm_provider"] = "apimart"
        config.app["apimart_api_key"] = "apimart-key"
        config.app["apimart_base_url"] = ""
        config.app["apimart_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\napimart")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="apimart-key",
            base_url="https://api.apimart.ai/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "gpt-5.6-terra",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\napimart")

    def test_aihubmix_provider_uses_openai_compatible_client(self):
        """
        AIHubMix es un gateway compatible con OpenAI. Aquí se usa un cliente OpenAI
        falso para verificar que el proveedor independiente usa la dirección y el modelo
        por defecto del registro, para que la red real o una clave API privada no
        afecten a la estabilidad del test.
        """
        config.app["llm_provider"] = "aihubmix"
        config.app["aihubmix_api_key"] = "aihubmix-key"
        config.app["aihubmix_base_url"] = ""
        config.app["aihubmix_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\naihubmix")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="aihubmix-key",
            base_url="https://aihubmix.com/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "gpt-5.4-mini",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\naihubmix")

    def test_aimlapi_provider_uses_openai_compatible_client(self):
        config.app["llm_provider"] = "aimlapi"
        config.app["aimlapi_api_key"] = "aimlapi-key"
        config.app["aimlapi_base_url"] = ""
        config.app["aimlapi_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\naimlapi")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="aimlapi-key",
            base_url="https://api.aimlapi.com/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "openai/gpt-5-5",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\naimlapi")

    def test_evolink_provider_uses_openai_compatible_client(self):
        """
        EvoLink expone Chat Completions compatible con OpenAI en direct.evolink.ai. El
        proveedor debe mantener su propio endpoint y modelo por defecto en lugar de
        obligar a los usuarios a sobrecargar los ajustes genéricos de OpenAI.
        """
        config.app["llm_provider"] = "evolink"
        config.app["evolink_api_key"] = "evolink-key"
        config.app["evolink_base_url"] = ""
        config.app["evolink_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nevolink")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="evolink-key",
            base_url="https://direct.evolink.ai/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "gpt-5.5",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\nevolink")

    def test_iflytek_provider_uses_openai_compatible_client(self):
        """
        iFlytek Astron MaaS sirve Spark mediante Chat Completions compatible con OpenAI.
        El pago por uso es el valor por defecto; una URL base de Token Plan guardada por
        el usuario debe usarse tal cual, porque las claves de un plan no funcionan en el
        otro.
        """
        config.app["llm_provider"] = "iflytek"
        config.app["iflytek_api_key"] = "iflytek-key"
        config.app["iflytek_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\niflytek")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        for configured_base_url, expected_base_url in (
            ("", "https://maas-api.cn-huabei-1.xf-yun.com/v2"),
            (
                "https://maas-token-api.cn-huabei-1.xf-yun.com/v2",
                "https://maas-token-api.cn-huabei-1.xf-yun.com/v2",
            ),
        ):
            with self.subTest(base_url=configured_base_url):
                config.app["iflytek_base_url"] = configured_base_url
                fake_completions = FakeCompletions()
                fake_client = types.SimpleNamespace(
                    chat=types.SimpleNamespace(completions=fake_completions)
                )

                with (
                    patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
                    patch.object(llm, "ChatCompletion", types.SimpleNamespace),
                ):
                    result = llm._generate_response("Say hello")

                openai_client.assert_called_once_with(
                    api_key="iflytek-key",
                    base_url=expected_base_url,
                )
                self.assertEqual(
                    fake_completions.kwargs,
                    {
                        "model": "spark-x2.5",
                        "messages": [{"role": "user", "content": "Say hello"}],
                    },
                )
                self.assertEqual(result, "hello\niflytek")
    def test_openrouter_provider_uses_openai_compatible_client(self):
        """
        OpenRouter expone Chat Completions compatible con OpenAI a través de un endpoint
        unificado. El modelo por defecto se mantiene en un modelo de texto actualmente
        gratuito adecuado para generar guiones y palabras clave.
        """
        config.app["llm_provider"] = "openrouter"
        config.app["openrouter_api_key"] = "openrouter-key"
        config.app["openrouter_base_url"] = ""
        config.app["openrouter_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nopenrouter")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="openrouter-key",
            base_url="https://openrouter.ai/api/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "minimax/minimax-m3:free",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\nopenrouter")

    def test_api_route_provider_uses_openai_compatible_client(self):
        """
        API Route expone Chat Completions compatible con OpenAI a través de un endpoint
        unificado con enrutamiento inteligente y alta disponibilidad.
        """
        config.app["llm_provider"] = "api_route"
        config.app["api_route_api_key"] = "api-route-key"
        config.app["api_route_base_url"] = ""
        config.app["api_route_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\napi_route")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="api-route-key",
            base_url="https://www.api-route.com/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "gpt-5.4-mini",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\napi_route")

    def test_volcengine_provider_uses_openai_compatible_client(self):
        """
        VolcEngine Ark expone Chat Completions compatible con OpenAI. Aquí se usa un
        cliente OpenAI falso para cubrir la dirección y el modelo por defecto del
        proveedor, para que la red real o una clave API privada no afecten a la
        estabilidad del test.
        """
        config.app["llm_provider"] = "volcengine"
        config.app["volcengine_api_key"] = "volcengine-key"
        config.app["volcengine_base_url"] = ""
        config.app["volcengine_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nvolcengine")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="volcengine-key",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "doubao-seed-2-1-turbo-260628",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\nvolcengine")

    def test_grok_provider_still_uses_existing_path(self):
        config.app["llm_provider"] = "grok"
        config.app["grok_api_key"] = ""
        config.app["grok_base_url"] = "https://api.x.ai/v1"
        config.app["grok_model_name"] = "grok-4.3"

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("api_key no está configurada", result)
        self.assertNotIn("litellm", result.lower())

    def test_groq_provider_requires_api_key(self):
        config.app["llm_provider"] = "groq"
        config.app["groq_api_key"] = ""
        config.app["groq_base_url"] = "https://api.groq.com/openai/v1"
        config.app["groq_model_name"] = "openai/gpt-oss-120b"

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("api_key no está configurada", result)
        self.assertNotIn("litellm", result.lower())

    def test_groq_provider_uses_default_base_url(self):
        config.app["llm_provider"] = "groq"
        config.app["groq_api_key"] = "groq-test-key"
        config.app["groq_base_url"] = ""
        config.app["groq_model_name"] = "openai/gpt-oss-120b"

        fake_response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="hello\ngroq")
                )
            ]
        )
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kwargs: fake_response)
            )
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="groq-test-key",
            base_url="https://api.groq.com/openai/v1",
        )
        self.assertEqual(result, "hello\ngroq")

    def _use_ollama_provider(self, base_url=""):
        config.app["llm_provider"] = "ollama"
        config.app["ollama_api_key"] = ""
        config.app["ollama_base_url"] = base_url
        config.app["ollama_model_name"] = "llama3"

    def _assert_ollama_base_url(self, expected_base_url: str):
        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nollama")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="ollama",
            base_url=expected_base_url,
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "llama3",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\nollama")

    def test_ollama_default_base_url_uses_localhost_outside_container(self):
        """
        Al ejecutarse normalmente en el equipo, Ollama sigue usando localhost por
        defecto, para no afectar a los usuarios existentes.
        """
        self._use_ollama_provider()

        with patch.object(config, "is_running_in_container", return_value=False):
            self._assert_ollama_base_url("http://localhost:11434/v1")

    def test_ollama_default_base_url_uses_host_gateway_inside_container(self):
        """
        Dentro de un contenedor, localhost apunta al propio contenedor; el valor por
        defecto pasa a host.docker.internal, para que los usuarios de Docker Desktop
        puedan acceder al Ollama del host.
        """
        self._use_ollama_provider()

        with (
            patch.object(config, "is_running_in_container", return_value=True),
            patch.object(config, "_can_resolve_hostname", return_value=True),
        ):
            self._assert_ollama_base_url("http://host.docker.internal:11434/v1")

    def test_ollama_default_base_url_falls_back_to_container_gateway(self):
        """
        En el Docker nativo de Linux no siempre se puede resolver host.docker.internal.
        En ese caso, usar la puerta de enlace por defecto del contenedor como dirección
        de respaldo es más fiable que devolver un hostname que no se puede resolver.
        """
        self._use_ollama_provider()

        with (
            patch.object(config, "is_running_in_container", return_value=True),
            patch.object(config, "_can_resolve_hostname", return_value=False),
            patch.object(
                config, "get_container_default_gateway_ip", return_value="172.17.0.1"
            ),
        ):
            self._assert_ollama_base_url("http://172.17.0.1:11434/v1")

    def test_ollama_explicit_base_url_takes_precedence(self):
        """
        El ollama_base_url configurado a mano por el usuario tiene la máxima prioridad y
        no le afecta la detección de contenedor.
        """
        self._use_ollama_provider(base_url="http://ollama:11434/v1")

        with patch.object(config, "is_running_in_container", return_value=True):
            self._assert_ollama_base_url("http://ollama:11434/v1")

    def test_mimo_provider_uses_openai_compatible_client(self):
        """
        La interfaz oficial de MiMo es compatible con el protocolo Chat Completions de
        OpenAI. Aquí se usa un cliente OpenAI falso para verificar que el proveedor usa
        la configuración propia de MiMo y la base_url por defecto, sin depender de la
        red real ni de una clave API privada.
        """
        config.app["llm_provider"] = "mimo"
        config.app["mimo_api_key"] = "mimo-key"
        config.app["mimo_base_url"] = ""
        config.app["mimo_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nmimo")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="mimo-key",
            base_url="https://api.xiaomimimo.com/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "mimo-v2.5-pro",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\nmimo")

    def test_azure_provider_uses_azure_client_directly(self):
        """
        La autenticación, el endpoint y la api-version de Azure OpenAI los gestiona el
        cliente AzureOpenAI. Este test cubre el issue #892: la rama azure debe llamar
        directamente al cliente creado por AzureOpenAI, sin caer en la rama normal
        compatible con OpenAI, o se perdería la configuración de solicitud propia de
        Azure.
        """
        config.app["llm_provider"] = "azure"
        config.app["azure_api_key"] = "azure-key"
        config.app["azure_base_url"] = "https://example.openai.azure.com"
        config.app["azure_model_name"] = "gpt-4o-mini"
        config.app["azure_api_version"] = "2024-02-15-preview"

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nazure")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "AzureOpenAI", return_value=fake_client) as azure_client,
            patch.object(llm, "OpenAI") as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        azure_client.assert_called_once_with(
            api_key="azure-key",
            api_version="2024-02-15-preview",
            azure_endpoint="https://example.openai.azure.com",
        )
        openai_client.assert_not_called()
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\nazure")

    def test_unsupported_provider_returns_clear_error(self):
        config.app["llm_provider"] = "g" + "4f"

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("proveedor LLM no admitido", result)


class TestClaudeCodeProvider(unittest.TestCase):
    """El proveedor claude_code llama a la cuenta de suscripción a través de la CLI claude
    local, sin usar la API HTTP.
    """

    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app["llm_provider"] = "claude_code"
        config.app["claude_code_model_name"] = ""
        config.app["claude_code_cli_path"] = ""
        config.app["claude_code_timeout"] = ""

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    @staticmethod
    def _completed(stdout="", stderr="", returncode=0):
        return types.SimpleNamespace(
            stdout=stdout, stderr=stderr, returncode=returncode
        )

    @staticmethod
    def _cli_payload(result, is_error=False):
        return json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": is_error,
                "result": result,
            }
        )

    # ------------------------------------------------------------- éxito
    def test_successful_generation_returns_cli_result_text(self):
        """En el JSON que devuelve la CLI solo result es el texto; el resto de campos no
        debe filtrarse al guion.
        """
        with (
            patch.object(llm.shutil, "which", return_value="/usr/bin/claude"),
            patch.object(
                llm.subprocess,
                "run",
                return_value=self._completed(stdout=self._cli_payload("Hello world")),
            ) as run,
        ):
            self.assertEqual(llm._generate_response("write something"), "Hello world")

        command = run.call_args.args[0]
        self.assertEqual(command[0], "/usr/bin/claude")
        self.assertIn("-p", command)
        self.assertEqual(
            run.call_args.kwargs["timeout"], llm.CLAUDE_CODE_DEFAULT_TIMEOUT
        )

    def test_generation_disables_tools_and_user_customizations(self):
        """La generación de texto puro debe desactivar todas las herramientas y las
        personalizaciones del usuario, para no leer/escribir archivos ni cargar skills.
        """
        with (
            patch.object(llm.shutil, "which", return_value="/usr/bin/claude"),
            patch.object(
                llm.subprocess,
                "run",
                return_value=self._completed(stdout=self._cli_payload("ok")),
            ) as run,
        ):
            llm._generate_response("write something")

        command = run.call_args.args[0]
        self.assertIn("--tools", command)
        self.assertEqual(command[command.index("--tools") + 1], "")
        self.assertIn("--safe-mode", command)
        self.assertIn("--system-prompt", command)
        self.assertEqual(
            command[command.index("--system-prompt") + 1],
            llm.CLAUDE_CODE_SYSTEM_PROMPT,
        )

    def test_prompt_is_sent_through_stdin_not_argv(self):
        """En Windows, el claude instalado con npm es claude.cmd, y cmd.exe corta los
        argumentos en el primer salto de línea, así que se perderían el prompt de varias
        líneas y los argumentos de aislamiento que vienen detrás; por eso el prompt debe
        ir por stdin.
        """
        prompt = "# Role: Generator\n\n## Goals:\nwrite something"
        with (
            patch.object(llm.shutil, "which", return_value="/usr/bin/claude"),
            patch.object(
                llm.subprocess,
                "run",
                return_value=self._completed(stdout=self._cli_payload("ok")),
            ) as run,
        ):
            llm._generate_response(prompt)

        command = run.call_args.args[0]
        self.assertFalse(any("\n" in arg for arg in command))
        self.assertNotIn(prompt, command)
        self.assertEqual(run.call_args.kwargs["input"], prompt)

    def test_model_name_is_only_passed_when_configured(self):
        """Con el nombre del modelo vacío debe usarse el modelo por defecto de la CLI, en
        lugar de fijar un ID que puede dejar de funcionar.
        """
        with (
            patch.object(llm.shutil, "which", return_value="/usr/bin/claude"),
            patch.object(
                llm.subprocess,
                "run",
                return_value=self._completed(stdout=self._cli_payload("ok")),
            ) as run,
        ):
            llm._generate_response("write something")
            self.assertNotIn("--model", run.call_args.args[0])

            config.app["claude_code_model_name"] = "claude-opus-5"
            llm._generate_response("write something")
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--model") + 1], "claude-opus-5")

    # ------------------------------------------------- aislamiento de credenciales
    def test_conflicting_credentials_are_removed_from_subprocess_env(self):
        """Una clave API en el entorno haría que la CLI se saltara el inicio de sesión de
        la suscripción y generara facturación de API, así que hay que quitarla.
        """
        polluted = {
            "PATH": "/usr/bin",
            "ANTHROPIC_API_KEY": "sk-ant-api03-should-not-be-used",
            "ANTHROPIC_BASE_URL": "https://proxy.example.com",
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-subscription",
        }
        env, removed = llm.build_claude_code_env(polluted)

        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("ANTHROPIC_BASE_URL", env)
        self.assertNotIn("CLAUDE_CODE_USE_BEDROCK", env)
        # El token de suscripción es la única forma de autenticación dentro del
        # contenedor y debe conservarse.
        self.assertEqual(env["CLAUDE_CODE_OAUTH_TOKEN"], "sk-ant-oat01-subscription")
        self.assertEqual(env["PATH"], "/usr/bin")
        self.assertCountEqual(
            removed,
            ["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK"],
        )

    def test_every_cloud_provider_switch_is_removed(self):
        """Los interruptores de Bedrock / Vertex / Foundry / Mantle / Gateway, etc., hacen
        que se facture con el proveedor en la nube.
        """
        switches = {
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "CLAUDE_CODE_USE_VERTEX": "1",
            "CLAUDE_CODE_USE_FOUNDRY": "1",
            "CLAUDE_CODE_USE_MANTLE": "1",
            "CLAUDE_CODE_USE_GATEWAY": "1",
            "CLAUDE_CODE_USE_ANTHROPIC_AWS": "1",
            "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD": "1",
        }
        env, removed = llm.build_claude_code_env({"PATH": "/usr/bin", **switches})

        self.assertEqual(env, {"PATH": "/usr/bin"})
        self.assertCountEqual(removed, list(switches))

    def test_foundry_credentials_are_removed(self):
        """Las credenciales de Foundry harían que la CLI facturara en Azure, así que hay
        que quitar tanto las credenciales como el interruptor.
        """
        foundry = {
            "CLAUDE_CODE_USE_FOUNDRY": "1",
            "ANTHROPIC_FOUNDRY_API_KEY": "foundry-key",
            "ANTHROPIC_FOUNDRY_AUTH_TOKEN": "foundry-token",
            "ANTHROPIC_FOUNDRY_BASE_URL": "https://example.openai.azure.com",
            "ANTHROPIC_FOUNDRY_RESOURCE": "my-resource",
            "CLAUDE_CODE_SKIP_FOUNDRY_AUTH": "1",
        }
        env, removed = llm.build_claude_code_env(
            {"PATH": "/usr/bin", "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01", **foundry}
        )

        for name in foundry:
            self.assertNotIn(name, env, name)
        self.assertCountEqual(removed, list(foundry))
        self.assertEqual(env["CLAUDE_CODE_OAUTH_TOKEN"], "sk-ant-oat01")

    def test_auth_bypass_switches_are_removed(self):
        """CLAUDE_CODE_SKIP_*_AUTH se salta la autenticación del proveedor, así que tampoco
        puede pasarse al subproceso.
        """
        bypasses = {
            "CLAUDE_CODE_SKIP_BEDROCK_AUTH": "1",
            "CLAUDE_CODE_SKIP_VERTEX_AUTH": "1",
            "CLAUDE_CODE_SKIP_MANTLE_AUTH": "1",
            "CLAUDE_CODE_SKIP_ANTHROPIC_AWS_AUTH": "1",
        }
        env, removed = llm.build_claude_code_env({"PATH": "/usr/bin", **bypasses})
        self.assertEqual(env, {"PATH": "/usr/bin"})
        self.assertCountEqual(removed, list(bypasses))

    def test_credential_location_variables_are_preserved(self):
        """*_CONFIG_DIR solo indica dónde están las credenciales; quitarlo invalidaría la
         suscripción con la sesión ya iniciada.
        """
        preserved = {
            "ANTHROPIC_CONFIG_DIR": "/home/user/.config/anthropic",
            "CLAUDE_CONFIG_DIR": "/home/user/.claude",
            "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01",
        }
        env, removed = llm.build_claude_code_env(dict(preserved))
        self.assertEqual(env, preserved)
        self.assertEqual(removed, [])

    def test_unrelated_variables_are_never_removed(self):
        """El filtrado solo afecta a la autenticación y a los interruptores de proveedor,
        sin tocar variables normales como PATH o el proxy.
        """
        env, removed = llm.build_claude_code_env(
            {"PATH": "/usr/bin", "HTTPS_PROXY": "http://proxy:3128", "HOME": "/root"}
        )
        self.assertEqual(
            env,
            {"PATH": "/usr/bin", "HTTPS_PROXY": "http://proxy:3128", "HOME": "/root"},
        )
        self.assertEqual(removed, [])

    def test_clean_environment_is_left_unchanged(self):
        env, removed = llm.build_claude_code_env({"PATH": "/usr/bin"})
        self.assertEqual(env, {"PATH": "/usr/bin"})
        self.assertEqual(removed, [])

    def test_subprocess_receives_sanitized_environment(self):
        """El adaptador debe pasar de verdad el entorno limpio al subproceso, y no
        limitarse a calcularlo.
        """
        with (
            patch.object(llm.shutil, "which", return_value="/usr/bin/claude"),
            patch.dict(
                os.environ, {"ANTHROPIC_API_KEY": "sk-ant-api03-x"}, clear=False
            ),
            patch.object(
                llm.subprocess,
                "run",
                return_value=self._completed(stdout=self._cli_payload("ok")),
            ) as run,
        ):
            llm._generate_response("write something")

        self.assertNotIn("ANTHROPIC_API_KEY", run.call_args.kwargs["env"])

    # ------------------------------------------------------- configuración del tiempo
    #  de espera
    def test_timeout_accepts_numeric_and_string_values(self):
        """En TOML, 300 es int y "300" es str; ambas formas deben admitirse."""
        self.assertEqual(llm.coerce_claude_code_timeout(300), 300.0)
        self.assertEqual(llm.coerce_claude_code_timeout(300.5), 300.5)
        self.assertEqual(llm.coerce_claude_code_timeout("300"), 300.0)
        self.assertEqual(llm.coerce_claude_code_timeout("  300  "), 300.0)
        self.assertEqual(
            llm.coerce_claude_code_timeout(""), llm.CLAUDE_CODE_DEFAULT_TIMEOUT
        )
        self.assertEqual(
            llm.coerce_claude_code_timeout(None), llm.CLAUDE_CODE_DEFAULT_TIMEOUT
        )

    def test_timeout_rejects_non_finite_and_invalid_values(self):
        """nan / inf bloquearían subprocess para siempre, así que hay que rechazarlos ya en
        la fase de configuración.
        """
        for invalid in (
            float("nan"),
            float("inf"),
            float("-inf"),
            "nan",
            "inf",
            "-inf",
            0,
            -5,
            "abc",
            True,
            [300],
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    llm.coerce_claude_code_timeout(invalid)

    def test_integer_timeout_in_config_is_accepted(self):
        """Test de regresión: un tiempo de espera int provocaba 'int' object has no
        attribute 'strip'.
        """
        config.app["claude_code_timeout"] = 30
        with (
            patch.object(llm.shutil, "which", return_value="/usr/bin/claude"),
            patch.object(
                llm.subprocess,
                "run",
                return_value=self._completed(stdout=self._cli_payload("ok")),
            ) as run,
        ):
            self.assertEqual(llm._generate_response("write something"), "ok")
        self.assertEqual(run.call_args.kwargs["timeout"], 30.0)

    def test_zero_and_false_timeouts_are_rejected_not_defaulted(self):
        """0 / false son valores no válidos y deben dar error, en lugar de sustituirse en
        silencio por el valor por defecto de 300 segundos.
        """
        for invalid in (0, False, "0", 0.0):
            with self.subTest(invalid=invalid):
                config.app["claude_code_timeout"] = invalid
                with patch.object(llm.shutil, "which", return_value="/usr/bin/claude"):
                    response = llm._generate_response("write something")
                self.assertTrue(response.startswith("Error:"), response)
                self.assertIn("claude_code_timeout", response)

    def test_field_default_applies_only_to_empty_values(self):
        """El valor por defecto del registro solo tiene efecto cuando no hay configuración;
        los valores falsos válidos deben llegar tal cual a la validación.
        """
        self.assertEqual(llm._resolve_provider_field_value(None, "300"), "300")
        self.assertEqual(llm._resolve_provider_field_value("", "300"), "300")
        self.assertEqual(llm._resolve_provider_field_value("   ", "300"), "300")
        self.assertEqual(llm._resolve_provider_field_value(0, "300"), 0)
        self.assertEqual(llm._resolve_provider_field_value(False, "300"), False)
        self.assertEqual(llm._resolve_provider_field_value("60", "300"), "60")

    def test_invalid_timeout_reports_configuration_error(self):
        config.app["claude_code_timeout"] = "soon"
        with patch.object(llm.shutil, "which", return_value="/usr/bin/claude"):
            response = llm._generate_response("write something")
        self.assertIn("claude_code_timeout", response)
        self.assertTrue(response.startswith("Error:"), response)

    # -------------------------------------------------------- modos de fallo
    def test_missing_cli_reports_actionable_error(self):
        with (
            patch.object(llm.shutil, "which", return_value=None),
            patch.object(llm.os.path, "isfile", return_value=False),
        ):
            response = llm._generate_response("write something")
        self.assertIn("no se encontró la CLI de claude", response)

    def test_missing_login_reports_setup_token_hint(self):
        """Dentro del contenedor no se puede ejecutar el /login interactivo, así que el
        mensaje de error debe dar una alternativa utilizable.
        """
        payload = self._cli_payload("Not logged in · Please run /login", is_error=True)
        with (
            patch.object(llm.shutil, "which", return_value="/usr/bin/claude"),
            patch.object(
                llm.subprocess,
                "run",
                return_value=self._completed(stdout=payload, returncode=1),
            ),
        ):
            response = llm._generate_response("write something")
        self.assertIn("Not logged in", response)
        self.assertIn("setup-token", response)
        self.assertIn("CLAUDE_CODE_OAUTH_TOKEN", response)

    def test_exhausted_quota_surfaces_cli_message(self):
        """Agotar el uso también es is_error + código de salida distinto de cero, y hay que
        mostrar tal cual el motivo legible.
        """
        payload = self._cli_payload(
            "Claude usage limit reached. Your limit will reset at 5pm.", is_error=True
        )
        with (
            patch.object(llm.shutil, "which", return_value="/usr/bin/claude"),
            patch.object(
                llm.subprocess,
                "run",
                return_value=self._completed(stdout=payload, returncode=1),
            ),
        ):
            response = llm._generate_response("write something")
        self.assertIn("usage limit reached", response)

    def test_timeout_is_reported_with_configured_seconds(self):
        config.app["claude_code_timeout"] = 12
        with (
            patch.object(llm.shutil, "which", return_value="/usr/bin/claude"),
            patch.object(
                llm.subprocess,
                "run",
                side_effect=llm.subprocess.TimeoutExpired(cmd="claude", timeout=12),
            ),
        ):
            response = llm._generate_response("write something")
        self.assertIn("agotó el tiempo tras 12s", response)

    def test_malformed_output_is_reported_instead_of_crashing(self):
        with (
            patch.object(llm.shutil, "which", return_value="/usr/bin/claude"),
            patch.object(
                llm.subprocess,
                "run",
                return_value=self._completed(stdout="not json at all"),
            ),
        ):
            response = llm._generate_response("write something")
        self.assertIn("respuesta no válida", response)

    def test_empty_output_is_reported(self):
        with (
            patch.object(llm.shutil, "which", return_value="/usr/bin/claude"),
            patch.object(
                llm.subprocess, "run", return_value=self._completed(stdout="   ")
            ),
        ):
            response = llm._generate_response("write something")
        self.assertTrue(response.startswith("Error:"), response)

    def test_unsupported_cli_version_reports_upgrade_hint(self):
        """Las CLI antiguas no tienen --tools / --safe-mode; hay que pedir que se actualice
        en lugar de soltar el stderr en bruto.
        """
        with (
            patch.object(llm.shutil, "which", return_value="/usr/bin/claude"),
            patch.object(
                llm.subprocess,
                "run",
                return_value=self._completed(
                    stderr="error: unknown option '--safe-mode'", returncode=1
                ),
            ),
        ):
            response = llm._generate_response("write something")
        self.assertIn("actualiza", response.lower())
        self.assertIn(llm.CLAUDE_CODE_MIN_CLI_VERSION, response)


class TestRuntimeEnvironmentDetection(unittest.TestCase):
    def test_container_detection_ignores_plain_linux_cgroup_file(self):
        """
        Un Linux normal también tiene /proc/1/cgroup, así que no se puede concluir que
        es un contenedor solo porque exista el archivo.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            cgroup_path = Path(tmp_dir) / "cgroup"
            cgroup_path.write_text("0::/init.scope\n", encoding="utf-8")

            self.assertFalse(
                config.is_running_in_container(
                    dockerenv_path=str(Path(tmp_dir) / "missing-dockerenv"),
                    containerenv_path=str(Path(tmp_dir) / "missing-containerenv"),
                    cgroup_path=str(cgroup_path),
                )
            )

    def test_container_detection_accepts_dockerenv_marker(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dockerenv_path = Path(tmp_dir) / ".dockerenv"
            dockerenv_path.write_text("", encoding="utf-8")

            self.assertTrue(
                config.is_running_in_container(
                    dockerenv_path=str(dockerenv_path),
                    containerenv_path=str(Path(tmp_dir) / "missing-containerenv"),
                    cgroup_path=str(Path(tmp_dir) / "missing-cgroup"),
                )
            )

    def test_container_detection_accepts_cgroup_container_marker(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cgroup_path = Path(tmp_dir) / "cgroup"
            cgroup_path.write_text(
                "0::/system.slice/docker-abcdef.scope\n",
                encoding="utf-8",
            )

            self.assertTrue(
                config.is_running_in_container(
                    dockerenv_path=str(Path(tmp_dir) / "missing-dockerenv"),
                    containerenv_path=str(Path(tmp_dir) / "missing-containerenv"),
                    cgroup_path=str(cgroup_path),
                )
            )

    def test_container_gateway_ip_decodes_default_route(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            route_path = Path(tmp_dir) / "route"
            route_path.write_text(
                "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n"
                "eth0\t00000000\t010011AC\t0003\t0\t0\t0\t00000000\t0\t0\t0\n",
                encoding="utf-8",
            )

            self.assertEqual(
                config.get_container_default_gateway_ip(str(route_path)),
                "172.17.0.1",
            )

    def test_container_gateway_ip_ignores_missing_default_route(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            route_path = Path(tmp_dir) / "route"
            route_path.write_text(
                "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n"
                "eth0\t0011AC0A\t00000000\t0001\t0\t0\t0\t00FFFFFF\t0\t0\t0\n",
                encoding="utf-8",
            )

            self.assertEqual(
                config.get_container_default_gateway_ip(str(route_path)), ""
            )


class TestSocialMetadata(unittest.TestCase):
    """Generación común de metadatos de texto para publicar vídeos cortos."""

    def test_build_prompt_auto_language_uses_source_language(self):
        """
        Con language en auto por defecto no debe fijarse a un país o idioma concreto,
        sino dejar que el modelo siga el idioma del tema y del guion del vídeo, para
        ampliar el alcance de la API.
        """
        prompt = llm.build_social_metadata_prompt(
            video_subject="上海一日游",
            video_script="今天带你快速看完上海经典路线。",
            language="auto",
            platform="tiktok",
        )

        self.assertIn("TikTok", prompt)
        self.assertIn("Use the same language as the video subject and script", prompt)
        self.assertIn("上海一日游", prompt)
        self.assertIn("array of exactly 5 strings", prompt)

    def test_build_prompt_accepts_explicit_language(self):
        prompt = llm.build_social_metadata_prompt(
            video_subject="Coffee tips",
            language="en-US",
            platform="youtube_shorts",
        )

        self.assertIn("YouTube Shorts", prompt)
        self.assertIn('Write "title" and "caption" in this language: en-US', prompt)
        self.assertIn("array of exactly 3 strings", prompt)

    def test_unknown_platform_falls_back_to_tiktok(self):
        prompt = llm.build_social_metadata_prompt(
            video_subject="x",
            platform="unsupported-platform",
        )

        self.assertIn("TikTok", prompt)

    def test_normalize_hashtags_from_string_dedupes_and_clamps(self):
        tags = llm._normalize_hashtags("#fyp fyp, trending #Trending viral", count=2)

        self.assertEqual(tags, ["#fyp", "#trending"])

    def test_normalize_hashtags_from_string_splits_cjk_separators(self):
        tags = llm._normalize_hashtags("#旅行，#美食、#日常；#東京#グルメ", count=5)

        self.assertEqual(tags, ["#旅行", "#美食", "#日常", "#東京", "#グルメ"])

    def test_normalize_hashtags_from_list_keeps_unicode_letters(self):
        tags = llm._normalize_hashtags(
            ["上海 旅行", "#việt nam", "  ", "@bad!chars"], count=5
        )

        self.assertEqual(tags, ["#上海旅行", "#việtnam", "#badchars"])

    def test_normalize_hashtags_keeps_combining_marks(self):
        tags = llm._normalize_hashtags(["#भारत", "#বাংলা", "#தமிழ்", "#ท่องเที่ยว"], count=5)

        self.assertEqual(tags, ["#भारत", "#বাংলা", "#தமிழ்", "#ท่องเที่ยว"])

    def test_parse_social_metadata_recovers_embedded_json(self):
        raw = 'Sure: {"title":"T","caption":"C","hashtags":["#x"]} thanks'
        result = llm._parse_social_metadata(raw, "tiktok")

        self.assertEqual(result["title"], "T")
        self.assertEqual(result["caption"], "C")
        self.assertEqual(result["hashtags"], ["#x"])

    def test_parse_social_metadata_requires_title_or_caption(self):
        with self.assertRaises(ValueError):
            llm._parse_social_metadata('{"hashtags":["#x"]}', "tiktok")

    def test_generate_social_metadata_uses_llm_response(self):
        payload = (
            '{"title":"上海一日游","caption":"收藏这条路线，下次直接出发！",'
            '"hashtags":["#上海","#旅行","#shorts"]}'
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            result = llm.generate_social_metadata(
                video_subject="上海一日游",
                video_script="今天带你快速看完上海经典路线。",
                language="zh-CN",
                platform="tiktok",
            )

        self.assertEqual(result["title"], "上海一日游")
        self.assertEqual(result["caption"], "收藏这条路线，下次直接出发！")
        self.assertEqual(result["hashtags"], ["#上海", "#旅行", "#shorts"])

    def test_generate_social_metadata_falls_back_to_generic_hashtags(self):
        with patch.object(
            llm, "_generate_response", return_value="Error: api_key is not set"
        ):
            result = llm.generate_social_metadata(
                video_subject="Coffee tips",
                video_script="Save these three coffee tips.",
                platform="instagram_reels",
            )

        self.assertEqual(result["title"], "Coffee tips")
        self.assertEqual(result["caption"], "Save these three coffee tips.")
        self.assertEqual(len(result["hashtags"]), 8)
        self.assertEqual(result["hashtags"][0], "#shorts")

    def test_request_model_defaults_to_auto_language_tiktok(self):
        body = VideoSocialMetadataRequest(video_subject="Test")

        self.assertEqual(body.language, "auto")
        self.assertEqual(body.platform, "tiktok")

    def test_request_model_rejects_oversized_social_metadata_fields(self):
        """
        La API externa no puede aceptar guiones y parámetros de idioma de longitud
        ilimitada, o dispararía directamente el coste en tokens del LLM. La capa de
        schema intercepta primero y la capa de servicio vuelve a proteger las llamadas
        internas.
        """
        with self.assertRaises(ValidationError):
            VideoSocialMetadataRequest(video_subject="x" * 501)

        with self.assertRaises(ValidationError):
            VideoSocialMetadataRequest(video_subject="x", video_script="x" * 8001)

        with self.assertRaises(ValidationError):
            VideoSocialMetadataRequest(video_subject="x", language="x" * 65)

    def test_build_prompt_clamps_direct_service_inputs(self):
        prompt = llm.build_social_metadata_prompt(
            video_subject="x" * 600,
            video_script="y" * 9000,
            language="en",
        )

        self.assertIn("x" * llm.MAX_SOCIAL_SUBJECT_LENGTH, prompt)
        self.assertNotIn("x" * (llm.MAX_SOCIAL_SUBJECT_LENGTH + 1), prompt)
        self.assertIn("y" * llm.MAX_SOCIAL_SCRIPT_LENGTH, prompt)
        self.assertNotIn("y" * (llm.MAX_SOCIAL_SCRIPT_LENGTH + 1), prompt)

    def test_social_metadata_endpoint_response_shape(self):
        from fastapi.testclient import TestClient

        from app.asgi import app

        request_body = {
            "video_subject": "Tokyo coffee shops",
            "video_script": "Three quiet coffee shops for your next Tokyo morning.",
            "language": "en",
            "platform": "youtube_shorts",
        }
        llm_response = (
            '{"title":"3 Quiet Tokyo Coffee Shops",'
            '"caption":"Save these spots for your next Tokyo morning.",'
            '"hashtags":["#Tokyo","#Coffee","#Shorts"]}'
        )

        with patch.object(llm, "_generate_response", return_value=llm_response):
            response = TestClient(app).post(
                "/api/v1/social-metadata",
                json=request_body,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": 200,
                "message": "success",
                "data": {
                    "title": "3 Quiet Tokyo Coffee Shops",
                    "caption": "Save these spots for your next Tokyo morning.",
                    "hashtags": ["#Tokyo", "#Coffee", "#Shorts"],
                },
            },
        )


FOUNDRY_KEY = os.environ.get("ANTHROPIC_FOUNDRY_API_KEY", "")
FOUNDRY_BASE = "https://amanrai-test-resource.services.ai.azure.com/anthropic"
FOUNDRY_MODEL = "azure_ai/claude-sonnet-4-6"


@unittest.skipUnless(
    RUN_INTEGRATION_TESTS and FOUNDRY_KEY,
    "MPT_RUN_INTEGRATION_TESTS and ANTHROPIC_FOUNDRY_API_KEY not set",
)
class TestLiteLLMLiveIntegration(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app["llm_provider"] = "litellm"
        config.app["litellm_model_name"] = FOUNDRY_MODEL
        os.environ["AZURE_AI_API_KEY"] = FOUNDRY_KEY
        os.environ["AZURE_AI_API_BASE"] = FOUNDRY_BASE

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_live_litellm_completion(self):
        result = llm._generate_response("What is 2+2? Reply with just the number.")

        self.assertNotIn("Error:", result)
        self.assertIn("4", result)


class TestRetryWarningBoundary(unittest.TestCase):
    """'trying again' no debe registrarse en el último intento de reintento."""

    def _trying_again_count(self, mock_logger: object, fragment: str) -> int:
        return sum(
            1 for call in mock_logger.warning.call_args_list if fragment in str(call)
        )

    def test_generate_script_no_spurious_warning_on_last_attempt(self):
        with (
            patch.object(
                llm,
                "_generate_response",
                side_effect=RuntimeError("provider unavailable"),
            ),
            patch.object(llm, "logger") as mock_logger,
        ):
            llm.generate_script(video_subject="test subject")

        count = self._trying_again_count(mock_logger, "reintentando")
        self.assertEqual(
            count,
            llm._max_retries - 1,
            "El aviso no debe emitirse en el último intento: no habrá más reintentos",
        )

    def test_generate_terms_no_spurious_warning_on_last_attempt(self):
        with (
            patch.object(
                llm,
                "_generate_response",
                side_effect=RuntimeError("provider unavailable"),
            ),
            patch.object(llm, "logger") as mock_logger,
        ):
            llm.generate_terms(
                video_subject="test subject",
                video_script="some script text",
            )

        count = self._trying_again_count(mock_logger, "reintentando")
        self.assertEqual(
            count,
            llm._max_retries - 1,
            "El aviso no debe emitirse en el último intento: no habrá más reintentos",
        )


if __name__ == "__main__":
    unittest.main()
