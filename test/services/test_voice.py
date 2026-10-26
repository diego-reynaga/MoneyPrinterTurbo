import asyncio
import base64
import os
import shutil
import subprocess
import unittest
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# añade la raíz del proyecto al path de Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils import utils
from app.services import voice as vs
from app.services import task as task_service
from pydub import AudioSegment

temp_dir = utils.storage_dir("temp")

text_en = """
What is the meaning of life? 
This question has puzzled philosophers, scientists, and thinkers of all kinds for centuries. 
Throughout history, various cultures and individuals have come up with their interpretations and beliefs around the purpose of life. 
Some say it's to seek happiness and self-fulfillment, while others believe it's about contributing to the welfare of others and making a positive impact in the world. 
Despite the myriad of perspectives, one thing remains clear: the meaning of life is a deeply personal concept that varies from one person to another. 
It's an existential inquiry that encourages us to reflect on our values, desires, and the essence of our existence.
"""

text_zh = """
预计未来3天深圳冷空气活动频繁，未来两天持续阴天有小雨，出门带好雨具；
10-11日持续阴天有小雨，日温差小，气温在13-17℃之间，体感阴凉；
12日天气短暂好转，早晚清凉；
"""

voice_rate=1.0
voice_volume=1.0
RUN_INTEGRATION_TESTS = os.environ.get("MPT_RUN_INTEGRATION_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}
                    
class TestVoiceService(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        self.loop.close()

    def test_get_all_azure_voices(self):
        voices = vs.get_all_azure_voices()
        # Los datos se han migrado de cadenas en línea a azure_voices.json; se asegura
        # que siguen cargándose por completo
        self.assertEqual(len(voices), 331)
        # El resultado debe tener el formato "Name-Gender" y estar ordenado
        self.assertEqual(voices, sorted(voices))
        for v in voices:
            self.assertTrue(v.endswith("-Male") or v.endswith("-Female"))

    def test_get_all_azure_voices_filtered(self):
        filtered = vs.get_all_azure_voices(filter_locals=["zh-CN", "en-US"])
        self.assertTrue(len(filtered) > 0)
        self.assertTrue(
            all(v.startswith(("zh-CN", "en-US")) for v in filtered)
        )

    def test_get_gemini_voices_matches_documented_catalog(self):
        voices = vs.get_gemini_voices()

        self.assertEqual(len(voices), 30)
        self.assertEqual(
            [name for name, _style in vs.GEMINI_TTS_VOICES],
            [
                "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda",
                "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus",
                "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome",
                "Algenib", "Rasalgethi", "Laomedeia", "Achernar", "Alnilam",
                "Schedar", "Gacrux", "Pulcherrima", "Achird",
                "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager",
                "Sulafat",
            ],
        )
        self.assertIn("gemini:Achernar-Soft", voices)
        self.assertIn("gemini:Sulafat-Warm", voices)
        self.assertFalse(any("Atlas" in voice for voice in voices))

    def test_parse_gemini_voice_name_supports_new_and_legacy_labels(self):
        self.assertEqual(
            vs.parse_gemini_voice_name("gemini:Achernar-Soft"), "Achernar"
        )
        self.assertEqual(vs.parse_gemini_voice_name("gemini:Charon-Male"), "Charon")
        self.assertEqual(vs.parse_gemini_voice_name("Charon-Male"), "")

    def test_no_voice_tts_generates_silent_audio_and_subtitle_timeline(self):
        """
        El modo sin locución no llama a ningún proveedor TTS externo y solo genera audio
        silencioso como marcador de la línea de tiempo. Aquí se simula FFmpeg y se
        verifica que los parámetros de la solicitud, el archivo de salida y la
        estructura legacy de subtítulos cumplen lo que espera la cadena posterior de
        composición del vídeo.
        """

        def fake_run(command, capture_output, text, check, **kwargs):
            self.assertEqual(command[0], "/tmp/fake-ffmpeg")
            self.assertIn("anullsrc=r=44100:cl=mono", command)
            Path(command[-1]).write_bytes(b"fake-silent-mp3")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            vs.utils,
            "get_ffmpeg_binary",
            return_value="/tmp/fake-ffmpeg",
        ), patch.object(vs.subprocess, "run", side_effect=fake_run):
            voice_file = str(Path(tmp_dir) / "silent.mp3")
            sub_maker = vs.tts(
                text="第一句话。Second sentence.",
                voice_name=vs.NO_VOICE_NAME,
                voice_rate=1.0,
                voice_file=voice_file,
            )

            self.assertEqual(Path(voice_file).read_bytes(), b"fake-silent-mp3")

        self.assertIsNotNone(sub_maker)
        self.assertEqual(getattr(sub_maker, "subs", []), ["第一句话", "Second sentence"])
        self.assertEqual(len(getattr(sub_maker, "offset", [])), 2)
        self.assertGreater(vs.get_audio_duration(sub_maker), 0)

    def test_get_audio_duration_accepts_non_mp3_files(self):
        """
        El audio personalizado (custom_audio_file) suele estar en formatos distintos de
        mp3, como m4a/wav/aac. get_audio_duration no debe informar "Invalid target type"
        y devolver 0 solo porque la extensión no sea .mp3, sino dejar que moviepy
        (ffmpeg) lea la duración real.
        """
        for path in ("custom-audio.m4a", "voice.wav", "clip.aac"):
            with patch.object(vs.os.path, "exists", return_value=True), \
                    patch.object(vs, "AudioFileClip") as mock_afc:
                mock_afc.return_value.__enter__.return_value.duration = 28.89
                self.assertEqual(vs.get_audio_duration(path), 28.89)
                mock_afc.assert_called_once_with(path)

    def test_get_audio_duration_missing_file_returns_zero(self):
        """Si el archivo de audio no existe, devuelve 0 de forma segura en lugar de lanzar
        una excepción o fallar al leer.
        """
        with patch.object(vs.os.path, "exists", return_value=False):
            self.assertEqual(vs.get_audio_duration("does-not-exist.m4a"), 0.0)

    def test_no_voice_alias_none_is_supported_temporarily(self):
        """
        Se admite el centinela none que usó el PR #981, para que los pocos usuarios que
        llaman directamente a la API no dejen de funcionar de inmediato al actualizar.
        La nueva interfaz y el código nuevo siguen usando siempre no-voice.
        """
        self.assertTrue(vs.is_no_voice("none"))
        self.assertTrue(vs.is_no_voice(vs.NO_VOICE_NAME))
        self.assertFalse(vs.is_no_voice(""))

    def test_no_voice_duration_estimates_non_ascii_languages(self):
        """
        Sin locución no hay audio TTS real, así que el tiempo de lectura solo se puede
        estimar a partir del texto del guion. Los textos no ASCII como ruso, árabe, kana
        japonés o coreano también deben contar en la estimación, sin caer todos en el
        mínimo de 3 segundos.
        """
        russian_text = (
            "Это длинный тестовый сценарий без озвучки. "
            "Он должен получить достаточно времени для чтения субтитров."
        )
        arabic_text = "هذا اختبار طويل بدون تعليق صوتي، ويجب أن يحصل على وقت كاف لقراءة الترجمة."

        self.assertGreater(vs.estimate_no_voice_duration(russian_text), 8.0)
        self.assertGreater(vs.estimate_no_voice_duration(arabic_text), 8.0)

    def test_generate_silent_audio_rejects_missing_output_file(self):
        """
        Aunque el proceso de FFmpeg devuelva éxito, hay que confirmar que el archivo de
        salida existe de verdad y no está vacío. Así la anomalía se contiene en la fase
        TTS, en lugar de aparecer más tarde en la fase de composición del vídeo.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            vs.utils,
            "get_ffmpeg_binary",
            return_value="/tmp/fake-ffmpeg",
        ), patch.object(
            vs.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
        ):
            voice_file = str(Path(tmp_dir) / "missing-silent.mp3")

            self.assertFalse(vs.generate_silent_audio(3.0, voice_file))

    def test_no_voice_ffmpeg_timeout_preserves_previous_output(self):
        def timed_out(command, **kwargs):
            self.assertIn("-nostdin", command)
            self.assertGreater(kwargs.get("timeout", 0), 0)
            Path(command[-1]).write_bytes(b"partial-silence")
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "silent.mp3"
            output.write_bytes(b"previous-success")
            with patch.object(vs.subprocess, "run", side_effect=timed_out):
                self.assertFalse(vs.generate_silent_audio(3.0, str(output)))
            self.assertEqual(output.read_bytes(), b"previous-success")

    def test_empty_voice_name_does_not_enable_no_voice_mode(self):
        """
        Una voz vacía suele significar que falta configuración o que hay un error en los
        parámetros de la interfaz, y no se puede pasar automáticamente al modo sin
        locución. Si no, aunque el usuario configure mal el TTS obtendría un vídeo
        silencioso "correcto", y sería más difícil localizar el problema.
        """
        sentinel = object()

        with patch.object(vs, "azure_tts_v1", return_value=sentinel) as azure_tts_v1:
            result = vs.tts(
                text="empty voice should still use the default TTS path",
                voice_name="",
                voice_rate=1.0,
                voice_file="/tmp/empty-voice.mp3",
            )

        self.assertIs(result, sentinel)
        azure_tts_v1.assert_called_once()

    @unittest.skipUnless(
        RUN_INTEGRATION_TESTS,
        "MPT_RUN_INTEGRATION_TESTS not set",
    )
    def test_siliconflow(self):
        # La clave API de SiliconFlow está en [siliconflow].api_key, y el código en
        # tiempo de ejecución también la lee de config.siliconflow; aquí hay que usar la
        # misma fuente de configuración, para que el test no se omita por error aunque
        # las credenciales estén bien configuradas.
        if not vs.config.siliconflow.get("api_key"):
            self.skipTest("siliconflow_api_key is not configured")

        voice_name = "siliconflow:FunAudioLLM/CosyVoice2-0.5B:alex-Male"
        voice_name = vs.parse_voice_name(voice_name)
        
        async def _do():
            parts = voice_name.split(":")
            if len(parts) >= 3:
                model = parts[1]
                # Quita el sufijo de género, por ejemplo "alex-Male" -> "alex"
                voice_with_gender = parts[2]
                voice = voice_with_gender.split("-")[0]
                # Construye el parámetro voice completo, con el formato "model:voice"
                full_voice = f"{model}:{voice}"
                voice_file = f"{temp_dir}/tts-siliconflow-{voice}.mp3"
                subtitle_file = f"{temp_dir}/tts-siliconflow-{voice}.srt"
                sub_maker = vs.siliconflow_tts(
                    text=text_zh, model=model, voice=full_voice, voice_file=voice_file, voice_rate=voice_rate, voice_volume=voice_volume
                )
                if not sub_maker:
                    self.fail("siliconflow tts failed")
                vs.create_subtitle(sub_maker=sub_maker, text=text_zh, subtitle_file=subtitle_file)
                audio_duration = vs.get_audio_duration(sub_maker)
                print(f"voice: {voice_name}, audio duration: {audio_duration}s")
            else:
                self.fail("siliconflow invalid voice name")

        self.loop.run_until_complete(_do())
    
    @unittest.skipUnless(
        RUN_INTEGRATION_TESTS,
        "MPT_RUN_INTEGRATION_TESTS not set",
    )
    def test_azure_tts_v1(self):
        voice_name = "zh-CN-XiaoyiNeural-Female"
        voice_name = vs.parse_voice_name(voice_name)
        print(voice_name)
        
        voice_file = f"{temp_dir}/tts-azure-v1-{voice_name}.mp3"
        subtitle_file = f"{temp_dir}/tts-azure-v1-{voice_name}.srt"
        sub_maker = vs.azure_tts_v1(
            text=text_zh, voice_name=voice_name, voice_file=voice_file, voice_rate=voice_rate
        )
        if not sub_maker:
            self.fail("azure tts v1 failed")
        vs.create_subtitle(sub_maker=sub_maker, text=text_zh, subtitle_file=subtitle_file)
        audio_duration = vs.get_audio_duration(sub_maker)
        print(f"voice: {voice_name}, audio duration: {audio_duration}s")

    def test_azure_tts_v1_supports_legacy_edge_tts_without_boundary(self):
        """
        Verifica que Azure TTS V1 sigue funcionando cuando queda una dependencia
        edge_tts antigua.

        Este escenario de regresión corresponde al caso en que, tras fallar la
        actualización del paquete portátil de Windows, el entorno sigue con un edge_tts
        antiguo:
        1. `Communicate.__init__()` no acepta `boundary`
        2. solo existe el `stream()` asíncrono, sin `stream_sync()`
        """

        class _LegacyCommunicate:
            def __init__(self, text, voice, rate="+0%"):
                self.text = text
                self.voice = voice
                self.rate = rate

            async def stream(self):
                yield {"type": "audio", "data": b"legacy-audio"}
                yield {
                    "type": "WordBoundary",
                    "offset": 0,
                    "duration": 10000000,
                    "text": "legacy",
                }

        class _FakeSubMaker:
            def __init__(self):
                self.events = []

            def feed(self, chunk):
                self.events.append(chunk)

            def get_srt(self):
                if not self.events:
                    return ""
                return "1\n00:00:00,000 --> 00:00:01,000\nlegacy\n"

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            vs.edge_tts, "Communicate", _LegacyCommunicate
        ), patch.object(vs.edge_tts, "SubMaker", _FakeSubMaker):
            voice_file = str(Path(tmp_dir) / "legacy-edge-tts.mp3")
            sub_maker = vs.azure_tts_v1(
                text="legacy edge tts compatibility",
                voice_name="zh-CN-XiaoyiNeural-Female",
                voice_file=voice_file,
                voice_rate=1.0,
            )

            self.assertIsNotNone(sub_maker)
            self.assertEqual(Path(voice_file).read_bytes(), b"legacy-audio")
            self.assertEqual(len(sub_maker.events), 1)
            self.assertEqual(sub_maker.events[0]["type"], "WordBoundary")

    def test_azure_tts_v1_rejects_boundary_only_stream(self):
        """Los eventos de subtítulos sin audio no deben producir un resultado TTS correcto."""

        class _BoundaryOnlyCommunicate:
            def __init__(self, text, voice, rate="+0%", boundary=None):
                pass

            def stream_sync(self):
                yield {
                    "type": "WordBoundary",
                    "offset": 0,
                    "duration": 10000000,
                    "text": "hello",
                }

        class _FakeSubMaker:
            def __init__(self):
                self.events = []

            def feed(self, chunk):
                self.events.append(chunk)

            def get_srt(self):
                return "1\n00:00:00,000 --> 00:00:01,000\nhello\n" if self.events else ""

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            vs.edge_tts, "Communicate", _BoundaryOnlyCommunicate
        ), patch.object(vs.edge_tts, "SubMaker", _FakeSubMaker):
            voice_file = Path(tmp_dir) / "boundary-only.mp3"
            result = vs.azure_tts_v1(
                text="hello",
                voice_name="en-US-AriaNeural-Female",
                voice_file=str(voice_file),
                voice_rate=1.0,
            )

            self.assertIsNone(result)
            self.assertFalse(voice_file.exists())

    def test_azure_tts_v1_times_out_hanging_stream_sync(self):
        """
        Verifica que Azure TTS V1 falla rápido cuando el flujo síncrono de edge_tts se
        atasca.

        En la práctica, ante problemas de red, límites del servidor o una voz cuyo
        idioma no coincide con el texto, `stream_sync()` puede tardar mucho en volver, y
        la tarea de la WebUI se queda parada en `start, voice name...`. Aquí se
        reproduce ese caso con un stream falso que bloquea, confirmando que la
        protección por tiempo de espera hace que la función termine y devuelva None.
        """

        class _HangingCommunicate:
            def __init__(self, text, voice, rate="+0%", boundary=None):
                self.text = text
                self.voice = voice
                self.rate = rate
                self.boundary = boundary

            def stream_sync(self):
                time.sleep(10)
                yield {"type": "audio", "data": b"unreachable"}

        class _FakeSubMaker:
            def feed(self, chunk):
                return None

            def get_srt(self):
                return ""

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            vs.edge_tts, "Communicate", _HangingCommunicate
        ), patch.object(vs.edge_tts, "SubMaker", _FakeSubMaker), patch.object(
            vs.config,
            "app",
            dict(vs.config.app, edge_tts_timeout=0.05),
        ):
            voice_file = Path(tmp_dir) / "hanging-edge-tts.mp3"
            started_at = time.monotonic()
            sub_maker = vs.azure_tts_v1(
                text="帮我生成一个花开花落的视频",
                voice_name="en-AU-NatashaNeural-Female",
                voice_file=str(voice_file),
                voice_rate=1.0,
            )
            elapsed = time.monotonic() - started_at
            self.assertFalse(voice_file.exists())

        self.assertIsNone(sub_maker)
        self.assertLess(elapsed, 2)

    @unittest.skipUnless(
        RUN_INTEGRATION_TESTS,
        "MPT_RUN_INTEGRATION_TESTS not set",
    )
    def test_azure_tts_v2(self):
        if not vs.config.azure.get("speech_key") or not vs.config.azure.get("speech_region"):
            self.skipTest("Azure speech key or region is not configured")

        voice_name = "zh-CN-XiaoxiaoMultilingualNeural-V2-Female"
        voice_name = vs.parse_voice_name(voice_name)
        print(voice_name)

        async def _do():
            voice_file = f"{temp_dir}/tts-azure-v2-{voice_name}.mp3"
            subtitle_file = f"{temp_dir}/tts-azure-v2-{voice_name}.srt"
            sub_maker = vs.azure_tts_v2(
                text=text_zh,
                voice_name=voice_name,
                voice_file=voice_file,
                voice_rate=1.0,
            )
            if not sub_maker:
                self.fail("azure tts v2 failed")
            vs.create_subtitle(sub_maker=sub_maker, text=text_zh, subtitle_file=subtitle_file)
            audio_duration = vs.get_audio_duration(sub_maker)
            print(f"voice: {voice_name}, audio duration: {audio_duration}s")

        self.loop.run_until_complete(_do())

    def test_azure_tts_v2_ssml_applies_rate_and_escapes_text(self):
        """Azure V2 debe aplicar la velocidad mediante SSML, sin que el texto del usuario
        rompa el XML.
        """
        ssml = vs._build_azure_v2_ssml(
            text='A < B & "quoted"',
            voice_name="zh-CN-XiaoxiaoMultilingualNeural",
            voice_rate=1.8,
        )

        self.assertIn('xml:lang="zh-CN"', ssml)
        self.assertIn('rate="1.8"', ssml)
        self.assertIn("A &lt; B &amp; \"quoted\"", ssml)

    def test_tts_forwards_rate_to_azure_v2(self):
        """La entrada TTS común no puede perder voice_rate al despachar Azure V2."""
        voice_name = "zh-CN-XiaoxiaoMultilingualNeural-V2-Female"
        with patch.object(vs, "azure_tts_v2", return_value=object()) as mock_tts:
            result = vs.tts(
                text="语速测试",
                voice_name=voice_name,
                voice_rate=1.8,
                voice_file="/tmp/azure-v2-rate.mp3",
            )

        self.assertIsNotNone(result)
        mock_tts.assert_called_once_with(
            "语速测试",
            voice_name,
            "/tmp/azure-v2-rate.mp3",
            voice_rate=1.8,
        )

    def test_tts_strips_gemini_style_metadata_before_dispatch(self):
        """La descripción del estilo oficial del desplegable de Gemini no puede formar
        parte del voice_name de la API.
        """
        sentinel = object()

        with patch.object(vs, "gemini_tts", return_value=sentinel) as gemini_tts:
            result = vs.tts(
                text="Test the updated voice catalog.",
                voice_name="gemini:Achernar-Soft",
                voice_rate=1.0,
                voice_file="/tmp/gemini-achernar.mp3",
                voice_volume=1.0,
            )

        self.assertIs(result, sentinel)
        gemini_tts.assert_called_once_with(
            "Test the updated voice catalog.",
            "Achernar",
            1.0,
            "/tmp/gemini-achernar.mp3",
            1.0,
        )

    def test_gemini_tts_uses_google_genai_and_compatible_submaker_fields(self):
        """
        Verifica que Gemini TTS sigue devolviendo una estructura de subtítulos
        compatible con el proyecto en un entorno con edge_tts 7.x, y que la cadena de
        generación de subtítulos con `subtitle_provider=edge` puede consumirla
        directamente, para no volver a caer en Whisper. También se usa un directorio de
        salida anidado que no existe, cubriendo el caso límite de que la API o la CLI
        llamen directamente al servicio sin haber creado antes el directorio de la
        tarea.
        """

        class _InlineData:
            def __init__(self, data):
                self.data = data

        class _Part:
            def __init__(self, data):
                self.inline_data = _InlineData(data)

        class _Content:
            def __init__(self, data):
                self.parts = [_Part(data)]

        class _Candidate:
            def __init__(self, data):
                self.content = _Content(data)

        class _Response:
            def __init__(self, data):
                self.candidates = [_Candidate(data)]

        captured = {}

        class _FakeModels:
            def generate_content(self, **kwargs):
                captured.update(kwargs)
                tone = (
                    AudioSegment.silent(duration=1800)
                    .set_frame_rate(24000)
                    .set_channels(1)
                    .set_sample_width(2)
                )
                return _Response(tone.raw_data)

        class _FakeClient:
            def __init__(self, **kwargs):
                captured["client_kwargs"] = kwargs
                self.models = _FakeModels()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                captured["closed"] = True

        temp_root = Path(tempfile.mkdtemp(prefix="gemini-tts-output-"))
        self.addCleanup(shutil.rmtree, temp_root, True)
        output_dir = temp_root / "nested" / "audio"
        voice_file = str(output_dir / "tts-gemini-Zephyr.mp3")
        subtitle_file = str(output_dir / "tts-gemini-Zephyr.srt")
        text = "Gemini subtitle generation should work now. Testing multiple lines."

        self.assertFalse(output_dir.exists())

        with patch("google.genai.Client", _FakeClient), patch.object(
            vs.config,
            "app",
            dict(vs.config.app, gemini_api_key="test-key"),
        ):
            sub_maker = vs.gemini_tts(
                text=text,
                voice_name="Zephyr",
                voice_rate=1.0,
                voice_file=voice_file,
            )

        self.assertIsNotNone(sub_maker)
        self.assertTrue(Path(voice_file).is_file())
        self.assertEqual(
            getattr(sub_maker, "subs", []),
            ["Gemini subtitle generation should work now", "Testing multiple lines"],
        )
        self.assertEqual(len(getattr(sub_maker, "offset", [])), 2)
        self.assertEqual(sub_maker.offset[0][0], 0)
        self.assertLess(sub_maker.offset[0][1], sub_maker.offset[1][1])
        self.assertEqual(captured["client_kwargs"], {"api_key": "test-key"})
        self.assertEqual(captured["model"], "gemini-2.5-flash-preview-tts")
        self.assertEqual(captured["contents"], text)
        self.assertEqual(captured["config"].response_modalities, ["AUDIO"])
        voice_config = captured["config"].speech_config.voice_config
        self.assertEqual(
            voice_config.prebuilt_voice_config.voice_name,
            "Zephyr",
        )
        self.assertTrue(captured["closed"])

        vs.create_subtitle(sub_maker=sub_maker, text=text, subtitle_file=subtitle_file)
        subtitle_content = Path(subtitle_file).read_text(encoding="utf-8")
        self.assertIn("Gemini subtitle generation should work now", subtitle_content)
        self.assertIn("Testing multiple lines", subtitle_content)

    def test_mimo_tts_uses_openai_compatible_audio_response(self):
        """
        Verifica que Xiaomi MiMo TTS puede consumir la estructura de respuesta de audio
        compatible con OpenAI.

        Aquí se usan un cliente OpenAI falso y un AudioSegment falso para sustituir la
        red real y ffmpeg, confirmando que el código en tiempo de ejecución pone el
        texto a sintetizar en el mensaje assistant y exporta el audio WAV base64
        devuelto al archivo de audio que usa el resto del flujo del proyecto.
        """

        class _FakeAudio:
            def __init__(self):
                self.data = base64.b64encode(b"RIFF-fake-wav").decode("utf-8")

        class _FakeMessage:
            def __init__(self):
                self.audio = _FakeAudio()

        class _FakeChoice:
            def __init__(self):
                self.message = _FakeMessage()

        class _FakeCompletion:
            def __init__(self):
                self.choices = [_FakeChoice()]

        class _FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return _FakeCompletion()

        class _FakeAudioSegment:
            def __len__(self):
                return 1800

            def export(self, output_file, format):
                Path(output_file).write_bytes(b"fake-mp3")

        fake_completions = _FakeCompletions()
        class _FakeClient:
            chat = SimpleNamespace(completions=fake_completions)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.closed = True

        fake_client = _FakeClient()

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            vs,
            "OpenAI",
            return_value=fake_client,
        ) as openai_client, patch(
            "pydub.AudioSegment.from_file",
            return_value=_FakeAudioSegment(),
        ), patch.object(
            vs.config,
            "app",
            dict(
                vs.config.app,
                mimo_api_key="mimo-key",
                mimo_base_url="https://api.xiaomimimo.com/v1",
                mimo_tts_model_name="mimo-v2.5-tts",
                mimo_tts_style_prompt="用清晰的中文旁白朗读。",
            ),
        ):
            voice_file = str(Path(tmp_dir) / "mimo-tts.mp3")
            sub_maker = vs.mimo_tts(
                text="小米语音合成测试。第二句话。",
                voice_name="冰糖",
                voice_rate=1.0,
                voice_file=voice_file,
                voice_volume=1.0,
            )
            generated_audio = Path(voice_file).read_bytes()

        openai_client.assert_called_once_with(
            api_key="mimo-key",
            base_url="https://api.xiaomimimo.com/v1",
            max_retries=0,
            timeout=120.0,
        )
        self.assertTrue(fake_client.closed)
        self.assertEqual(fake_completions.kwargs["model"], "mimo-v2.5-tts")
        self.assertEqual(
            fake_completions.kwargs["messages"],
            [
                {"role": "user", "content": "用清晰的中文旁白朗读。"},
                {"role": "assistant", "content": "小米语音合成测试。第二句话。"},
            ],
        )
        self.assertEqual(
            fake_completions.kwargs["audio"],
            {"format": "wav", "voice": "冰糖"},
        )
        self.assertEqual(generated_audio, b"fake-mp3")
        self.assertIsNotNone(sub_maker)
        self.assertEqual(getattr(sub_maker, "subs", []), ["小米语音合成测试", "第二句话"])
        self.assertEqual(len(getattr(sub_maker, "offset", [])), 2)

    @patch("app.services.voice._validate_remote_tts_audio", new=lambda _path: None)
    def test_minimax_tts_uses_regional_endpoint_and_hex_audio(self):
        class _Response:
            status_code, text = 200, ""

            @staticmethod
            def json():
                return {"data": {"audio": b"audio".hex(), "status": 2}, "base_resp": {"status_code": 0}}

        class _Clip:
            duration = 2.5

            def close(self):
                pass

        captured = {}

        def _post(url, json=None, headers=None, timeout=None, allow_redirects=True):
            self.assertFalse(allow_redirects)
            captured.update(url=url, json=json, headers=headers, timeout=timeout)
            return _Response()

        settings = {
            "api_key": "test-key", "base_url": vs.MINIMAX_TTS_CN_URL,
            "model_id": "speech-2.8-turbo", "voice_id": "male-qn-qingse",
            "sample_rate": 32000, "bitrate": 128000, "audio_format": "mp3", "channel": 1,
        }
        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            vs.config, "minimax_tts", settings
        ), patch.object(vs.requests, "post", side_effect=_post), patch.object(
            vs, "AudioFileClip", return_value=_Clip()
        ):
            voice_file = str(Path(tmp_dir) / "minimax.mp3")
            result = vs.minimax_tts("Speech test.", "male-qn-qingse", 1.2, voice_file, 1.5)
            self.assertEqual(Path(voice_file).read_bytes(), b"audio")

        self.assertIsNotNone(result)
        self.assertEqual(captured["url"], "https://api.minimaxi.com/v1/t2a_v2")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(captured["json"]["model"], "speech-2.8-turbo")
        self.assertEqual(captured["json"]["text"], "Speech test.")
        self.assertEqual(captured["json"]["voice_setting"]["voice_id"], "male-qn-qingse")
        self.assertEqual(captured["json"]["audio_setting"]["format"], "mp3")

    @patch("app.services.voice._validate_remote_tts_audio", new=lambda _path: None)
    def test_minimax_tts_reuses_cn_llm_key_and_endpoint(self):
        """Si el TTS no se configura aparte, debe reutilizar las credenciales y la
        dirección del LLM de MiniMax de la misma región.
        """
        class _Response:
            status_code, text = 200, ""

            @staticmethod
            def json():
                return {"data": {"audio": b"audio".hex(), "status": 2}, "base_resp": {"status_code": 0}}

        class _Clip:
            duration = 1.25

            def close(self):
                pass

        captured = {}

        def _post(url, json=None, headers=None, timeout=None, allow_redirects=True):
            self.assertFalse(allow_redirects)
            captured.update(url=url, headers=headers)
            return _Response()

        settings = {
            "api_key": "", "base_url": vs.MINIMAX_TTS_GLOBAL_URL,
            "model_id": vs.MINIMAX_TTS_DEFAULT_MODEL, "audio_format": "mp3",
        }
        app_settings = {
            "minimax_api_key": "shared-cn-key",
            "minimax_base_url": "https://api.minimaxi.com/v1",
        }
        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            vs.config, "minimax_tts", settings
        ), patch.object(vs.config, "app", app_settings), patch.object(
            vs.requests, "post", side_effect=_post
        ), patch.object(vs, "AudioFileClip", return_value=_Clip()):
            voice_file = str(Path(tmp_dir) / "minimax.mp3")
            result = vs.minimax_tts("测试。", "male-qn-qingse", 1.0, voice_file)

        self.assertIsNotNone(result)
        self.assertEqual(captured["url"], vs.MINIMAX_TTS_CN_URL)
        self.assertEqual(captured["headers"]["Authorization"], "Bearer shared-cn-key")

    def test_get_minimax_voice_catalog_normalizes_all_voice_types(self):
        """La consulta de voces debe unificar las estructuras de respuesta de las distintas
        fuentes e ignorar los Voice ID repetidos o vacíos.
        """

        class _Response:
            status_code, text = 200, ""

            @staticmethod
            def json():
                return {
                    "system_voice": [
                        {"voice_id": "system-1", "voice_name": "系统音色"},
                        {"voice_id": "", "voice_name": "无效音色"},
                    ],
                    "voice_cloning": [
                        {"voice_id": "clone-1", "voice_name": "我的克隆音色"},
                        {"voice_id": "system-1", "voice_name": "重复音色"},
                    ],
                    "voice_generation": [{"voice_id": "generated-1"}],
                    "base_resp": {"status_code": "0"},
                }

        with patch.object(vs.requests, "post", return_value=_Response()) as post:
            catalog = vs.get_minimax_voice_catalog(
                api_key="test-key",
                endpoint=vs.MINIMAX_TTS_CN_URL,
            )

        post.assert_called_once_with(
            "https://api.minimaxi.com/v1/get_voice",
            json={"voice_type": "all"},
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        self.assertEqual(
            catalog,
            [
                {
                    "voice_id": "system-1",
                    "voice_name": "系统音色",
                    "voice_type": "system",
                },
                {
                    "voice_id": "clone-1",
                    "voice_name": "我的克隆音色",
                    "voice_type": "voice_cloning",
                },
                {
                    "voice_id": "generated-1",
                    "voice_name": "generated-1",
                    "voice_type": "voice_generation",
                },
            ],
        )

    def test_get_minimax_voice_catalog_exposes_provider_error(self):
        """Los errores de negocio remotos deben lanzarse claramente, sin disfrazarse de que
        la cuenta no tiene voces disponibles.
        """

        class _Response:
            status_code, text = 200, ""

            @staticmethod
            def json():
                return {
                    "base_resp": {
                        "status_code": 1004,
                        "status_msg": "invalid api key",
                    }
                }

        with patch.object(vs.requests, "post", return_value=_Response()):
            with self.assertRaisesRegex(RuntimeError, "invalid api key"):
                vs.get_minimax_voice_catalog(api_key="invalid-key")

    @patch("app.services.voice._validate_remote_tts_audio", new=lambda _path: None)
    def test_minimax_tts_does_not_leave_invalid_audio_output(self):
        """Si no se puede analizar el audio de la respuesta, no debe sobrescribirse el
        archivo existente ni dejar archivos temporales.
        """
        class _Response:
            status_code, text = 200, ""

            @staticmethod
            def json():
                return {"data": {"audio": b"invalid-audio".hex(), "status": 2}, "base_resp": {"status_code": 0}}

        settings = {
            "api_key": "test-key", "base_url": vs.MINIMAX_TTS_GLOBAL_URL,
            "model_id": vs.MINIMAX_TTS_DEFAULT_MODEL, "audio_format": "mp3",
        }
        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            vs.config, "minimax_tts", settings
        ), patch.object(vs.requests, "post", return_value=_Response()), patch.object(
            vs, "AudioFileClip", side_effect=OSError("invalid audio")
        ):
            voice_path = Path(tmp_dir) / "minimax.mp3"
            voice_path.write_bytes(b"existing-audio")
            result = vs.minimax_tts("Speech test.", "English_expressive_narrator", 1.0, str(voice_path))

            self.assertIsNone(result)
            self.assertEqual(voice_path.read_bytes(), b"existing-audio")
            self.assertEqual([path.name for path in Path(tmp_dir).iterdir()], ["minimax.mp3"])

    def test_minimax_voice_helpers_and_dispatch(self):
        with patch.object(vs.config, "minimax_tts", {"voice_id": "narrator"}):
            self.assertEqual(vs.get_minimax_voices(), ["minimax:narrator"])
        self.assertEqual(vs.get_minimax_voices("custom-voice"), ["minimax:custom-voice"])
        self.assertTrue(vs.is_minimax_voice("minimax:narrator"))
        sentinel = object()
        with patch.object(vs, "minimax_tts", return_value=sentinel) as implementation:
            result = vs.tts("test", "minimax:narrator", 1.0, "voice.mp3", 1.0)
        self.assertIs(result, sentinel)
        implementation.assert_called_once_with("test", "narrator", 1.0, "voice.mp3", 1.0)

    def test_chatterbox_voice_helpers(self):
        """Aspectos básicos y normalización de is_chatterbox_voice / get_chatterbox_voices."""
        self.assertTrue(vs.is_chatterbox_voice("chatterbox:default-Female"))
        self.assertFalse(vs.is_chatterbox_voice("elevenlabs:abc:Rachel"))
        self.assertFalse(vs.is_chatterbox_voice(""))
        self.assertFalse(vs.is_chatterbox_voice(None))

        # las entradas de la lista se normalizan al formato de despacho
        # chatterbox:<nombre>, y las que ya llevan el prefijo se dejan sin tocar
        with patch.object(
            vs.config,
            "chatterbox",
            {"voices": ["narrator-Male", "chatterbox:host"]},
        ):
            self.assertEqual(
                vs.get_chatterbox_voices(),
                ["chatterbox:narrator-Male", "chatterbox:host"],
            )

        # también se acepta una cadena separada por comas (cómoda en TOML)
        with patch.object(vs.config, "chatterbox", {"voices": "alpha, beta ,"}):
            self.assertEqual(
                vs.get_chatterbox_voices(),
                ["chatterbox:alpha", "chatterbox:beta"],
            )

        # sin nada configurado, el desplegable sigue teniendo un valor por defecto
        # utilizable
        with patch.object(vs.config, "chatterbox", {}):
            self.assertEqual(vs.get_chatterbox_voices(), ["chatterbox:default-Female"])

    @patch("app.services.voice._validate_remote_tts_audio", new=lambda _path: None)
    def test_chatterbox_tts_posts_to_openai_compatible_endpoint(self):
        """Ruta de éxito: POST /audio/speech, escribe el audio y devuelve un SubMaker
        legacy.
        """

        class _FakeResponse:
            status_code = 200
            content = b"RIFF-fake-wav"
            text = ""

        class _FakeClip:
            duration = 3.5

            def close(self):
                pass

        captured = {}

        def _fake_post(url, json=None, headers=None, timeout=None, allow_redirects=True):
            self.assertFalse(allow_redirects)
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _FakeResponse()

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            vs.config,
            "chatterbox",
            {
                "base_url": "http://localhost:4123/v1/",
                "api_key": "secret",
                "model_id": "chatterbox",
            },
        ), patch.object(
            vs.requests, "post", side_effect=_fake_post
        ) as post, patch.object(
            vs, "AudioFileClip", return_value=_FakeClip()
        ):
            voice_file = str(Path(tmp_dir) / "chatterbox.mp3")
            sub_maker = vs.chatterbox_tts(
                text="Hello world. Second sentence.",
                voice="default",
                voice_file=voice_file,
                voice_rate=1.2,
                voice_volume=1.0,
            )
            generated_audio = Path(voice_file).read_bytes()

        post.assert_called_once()
        # la barra final de base_url se quita antes de añadir /audio/speech
        self.assertEqual(captured["url"], "http://localhost:4123/v1/audio/speech")
        self.assertEqual(captured["json"]["model"], "chatterbox")
        self.assertEqual(captured["json"]["voice"], "default")
        self.assertEqual(captured["json"]["input"], "Hello world. Second sentence.")
        self.assertAlmostEqual(captured["json"]["speed"], 1.2)
        # api_key se reenvía como token bearer
        self.assertEqual(captured["headers"].get("Authorization"), "Bearer secret")
        # el volumen no forma parte intencionadamente de la carga útil de speech de
        # OpenAI
        self.assertNotIn("volume", captured["json"])
        self.assertEqual(generated_audio, b"RIFF-fake-wav")
        self.assertIsNotNone(sub_maker)
        self.assertTrue(getattr(sub_maker, "subs", []))

    def test_chatterbox_tts_requires_base_url(self):
        """Sin base_url se cortocircuita sin ninguna llamada de red."""
        with patch.object(
            vs.config, "chatterbox", {"base_url": ""}
        ), patch.object(vs.requests, "post") as post:
            result = vs.chatterbox_tts(
                text="hi", voice="default", voice_file="unused.mp3"
            )
        self.assertIsNone(result)
        post.assert_not_called()

    def test_chatterbox_tts_returns_none_on_http_error(self):
        """Una respuesta distinta de 200 se reintenta hasta 3 veces y luego falla
        devolviendo None.
        """

        class _FakeResponse:
            status_code = 500
            content = b""
            text = "boom"

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            vs.config, "chatterbox", {"base_url": "http://localhost:4123/v1"}
        ), patch.object(
            vs.requests, "post", return_value=_FakeResponse()
        ) as post:
            voice_file = str(Path(tmp_dir) / "chatterbox.mp3")
            result = vs.chatterbox_tts(
                text="hi", voice="default", voice_file=voice_file
            )
        self.assertIsNone(result)
        self.assertEqual(post.call_count, 3)

    def _make_broken_clip_class(self, close_calls: list):
        """Devuelve una clase de clip cuyo .duration lanza una excepción y cuyo .close()
        registra las llamadas.
        """

        class _BrokenClip:
            @property
            def duration(self):
                raise RuntimeError("FFmpeg probe failed")

            def close(self):
                close_calls.append(True)

        return _BrokenClip

    @patch("app.services.voice._validate_remote_tts_audio", new=lambda _path: None)
    def test_elevenlabs_tts_audio_clip_closed_on_duration_error(self):
        """AudioFileClip.close() debe llamarse aunque la lectura de .duration lance una
        excepción.
        """
        close_calls: list = []
        BrokenClip = self._make_broken_clip_class(close_calls)

        class _OkResponse:
            status_code = 200
            content = b"fake-mp3"
            text = ""

            def iter_content(self, chunk_size):
                yield self.content

            def close(self):
                pass

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            out = f.name
        try:
            with (
                patch.object(
                    vs.config,
                    "elevenlabs",
                    {"api_key": "test-key", "model_id": "eleven_multilingual_v2"},
                ),
                patch.object(vs.requests, "post", return_value=_OkResponse()),
                patch.object(vs, "AudioFileClip", side_effect=lambda _: BrokenClip()),
            ):
                result = vs.elevenlabs_tts("Hello world.", "voice-id", out)
        finally:
            if os.path.exists(out):
                os.remove(out)

        self.assertIsNone(result)
        self.assertTrue(close_calls, "AudioFileClip.close() was never called")

    @patch("app.services.voice._validate_remote_tts_audio", new=lambda _path: None)
    def test_chatterbox_tts_audio_clip_closed_on_duration_error(self):
        """AudioFileClip.close() debe llamarse aunque la lectura de .duration lance una
        excepción.
        """
        close_calls: list = []
        BrokenClip = self._make_broken_clip_class(close_calls)

        class _OkResponse:
            status_code = 200
            content = b"fake-mp3"
            text = ""

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            out = f.name
        try:
            with (
                patch.object(
                    vs.config,
                    "chatterbox",
                    {"base_url": "http://localhost:4123", "api_key": "", "model_id": "chatterbox"},
                ),
                patch.object(vs.requests, "post", return_value=_OkResponse()),
                patch.object(vs, "AudioFileClip", side_effect=lambda _: BrokenClip()),
            ):
                result = vs.chatterbox_tts("Hello world.", "default", out)
        finally:
            if os.path.exists(out):
                os.remove(out)

        self.assertIsNone(result)
        self.assertTrue(close_calls, "AudioFileClip.close() was never called")

    @patch("app.services.voice._validate_remote_tts_audio", new=lambda _path: None)
    def test_fish_audio_tts_audio_clip_closed_on_duration_error(self):
        """AudioFileClip.close() debe llamarse aunque la lectura de .duration lance una
        excepción.
        """
        close_calls: list = []
        BrokenClip = self._make_broken_clip_class(close_calls)

        class _OkResponse:
            status_code = 200
            content = b"x" * 200
            text = ""

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            out = f.name
        try:
            with (
                patch.object(
                    vs.config,
                    "fish_audio",
                    {"api_key": "test-key", "model": "s2.1-pro-free"},
                ),
                patch.object(vs.requests, "post", return_value=_OkResponse()),
                patch.object(vs, "AudioFileClip", side_effect=lambda _: BrokenClip()),
            ):
                result = vs.fish_audio_tts("Hello world.", out)
        finally:
            if os.path.exists(out):
                os.remove(out)

        self.assertIsNone(result)
        self.assertTrue(close_calls, "AudioFileClip.close() was never called")

    def test_generate_subtitle_keeps_edge_provider_for_gemini_legacy_submaker(self):
        """
        Verifica que la estructura legacy de subtítulos devuelta por Gemini TTS puede
        producir directamente un SRT con el proveedor edge, sin volver a Whisper por un
        fallo de correspondencia.
        """
        script = "Gemini subtitle generation should work now. Testing multiple lines."
        sub_maker = vs.populate_legacy_submaker_with_full_text(
            vs.ensure_legacy_submaker_fields(vs.SubMaker()),
            script,
            2.4,
        )

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            task_service.config,
            "app",
            dict(task_service.config.app, subtitle_provider="edge"),
        ), patch("app.services.subtitle.create") as whisper_create, patch(
            "app.utils.utils.task_dir",
            lambda tid="": str(Path(tmp_dir) / tid) if tid else str(Path(tmp_dir)),
        ):
            task_id = "gemini-subtitle-edge-task"
            Path(tmp_dir, task_id).mkdir(parents=True, exist_ok=True)
            subtitle_path = task_service.generate_subtitle(
                task_id=task_id,
                params=type("Params", (), {"subtitle_enabled": True})(),
                video_script=script,
                sub_maker=sub_maker,
                audio_file="",
            )

            self.assertTrue(subtitle_path.endswith("subtitle.srt"))
            self.assertTrue(Path(subtitle_path).exists())
            self.assertFalse(whisper_create.called)
            subtitle_content = Path(subtitle_path).read_text(encoding="utf-8")
            self.assertIn("Gemini subtitle generation should work now", subtitle_content)
            self.assertIn("Testing multiple lines", subtitle_content)

    def test_script_split_keeps_thousand_separator_comma(self):
        """
        Edge TTS devuelve "1,000 years" como texto continuo. Al dividir el guion en
        frases no se puede tratar la coma inglesa dentro del número como límite de
        frase; si no, la agregación de subtítulos daría, como en el issue #894, menos
        sub_items que script_lines y volvería por error a Whisper.
        """
        text = (
            "It takes about 1,000 years for a single drop of water to finish "
            "the whole trip!"
        )

        self.assertEqual(
            utils.split_string_by_punctuations(text),
            [
                (
                    "It takes about 1,000 years for a single drop of water to finish "
                    "the whole trip"
                )
            ],
        )

    def test_edge_cue_aggregation_handles_thousand_separator_comma(self):
        """
        Reproduce la forma clave del issue #894: en las cues de Edge, la última frase se
        devuelve como texto continuo e incluye `1,000 years`. La división del guion en
        frases debe coincidir con la agregación de cues, sin partirla en dos subtítulos.
        """
        text = (
            "The ocean isn't just sitting stil, it moves around the world like a massive "
            "amusement park ride! Cold water at the North and South Poles sinks to the "
            "bottom because it is heavy and salty. At the same time, warm water from the "
            "sunny equator flows along the top to take its place. This creates a giant "
            "underwater conveyor belt that travels all the way around the Earth. It takes "
            "about 1,000 years for a single drop of water to finish the whole trip!"
        )
        script_lines = utils.split_string_by_punctuations(text)
        cues = []
        for index, line in enumerate(script_lines):
            # El cue content de Edge a menudo no tiene la disposición de espacios y
            # puntuación del guion, así que aquí se quitan los espacios para simular un
            # escenario de correspondencia más estricto.
            cues.append(
                SimpleNamespace(
                    content=line.replace(" ", ""),
                    start=timedelta(seconds=index),
                    end=timedelta(seconds=index + 0.8),
                )
            )
        sub_maker = SimpleNamespace(cues=cues)

        sub_items = vs._build_subtitle_items_from_edge_cues(sub_maker, script_lines)

        self.assertEqual(len(sub_items), len(script_lines))
        self.assertIn("1,000 years", sub_items[-1])

    def test_script_split_keeps_clock_time_colon(self):
        """
        Edge TTS devuelve una hora como "10:30" como una sola palabra, así que la
        división del guion no debe cortar una línea en sus dos puntos.
        """
        text = "The class starts at 10:30 every morning. We film everything in 16:9."

        self.assertEqual(
            utils.split_string_by_punctuations(text),
            ["The class starts at 10:30 every morning", "We film everything in 16:9"],
        )

    def test_edge_cue_aggregation_handles_clock_time(self):
        """
        Los límites de palabra que devuelve edge-tts 7.3.1 para este guion con
        en-US-JennyNeural:
        "10:30" es una palabra y "16:9" son tres. Ambas líneas deben seguir teniendo
        cada una su subtítulo, sin volver a Whisper.
        """
        text = "We film everything in 16:9. The class starts at 10:30 every morning."
        script_lines = utils.split_string_by_punctuations(text)
        words = "We film everything in 16 : 9 The class starts at 10:30 every morning".split()
        cues = [
            SimpleNamespace(
                content=word,
                start=timedelta(seconds=index * 0.3),
                end=timedelta(seconds=index * 0.3 + 0.25),
            )
            for index, word in enumerate(words)
        ]

        sub_items = vs._build_subtitle_items_from_edge_cues(
            SimpleNamespace(cues=cues), script_lines
        )

        self.assertEqual(len(sub_items), len(script_lines))
        self.assertIn("10:30 every morning", sub_items[-1])

    def test_script_split_keeps_dots_inside_abbreviations_and_names(self):
        """Un punto entre dos letras pertenece a "U.S.", "e.g." o "example.com", no es un
        salto de línea.
        """
        text = (
            "The U.S. economy grew. Visit example.com for the details, e.g. the prices."
        )

        self.assertEqual(
            utils.split_string_by_punctuations(text),
            [
                "The U.S",
                "economy grew",
                "Visit example.com for the details",
                "e.g",
                "the prices",
            ],
        )

    def test_edge_cue_aggregation_handles_dotted_abbreviations(self):
        """
        Los límites de palabra que devuelve edge-tts 7.3.1 para cada una de estas frases
        con en-US-AriaNeural: "U.S.", "9 a.m." y "example.com" son una palabra cada uno.
        Cada línea debe seguir teniendo su propio subtítulo, sin volver a Whisper.
        """
        text = (
            "The U.S. economy grew last year. Dr. Smith arrives at 9 a.m. sharp. "
            "Visit example.com for the details."
        )
        script_lines = utils.split_string_by_punctuations(text)
        words = [
            *"The U.S. economy grew last year Dr. Smith arrives at".split(),
            "9 a.m.",
            *"sharp Visit example.com for the details".split(),
        ]
        cues = [
            SimpleNamespace(
                content=word,
                start=timedelta(seconds=index * 0.3),
                end=timedelta(seconds=index * 0.3 + 0.25),
            )
            for index, word in enumerate(words)
        ]

        sub_items = vs._build_subtitle_items_from_edge_cues(
            SimpleNamespace(cues=cues), script_lines
        )

        self.assertEqual(len(sub_items), len(script_lines))
        self.assertIn("Smith arrives at 9 a.m", sub_items[-3])
        self.assertIn("Visit example.com for the details", sub_items[-1])

    def test_script_split_supports_arabic_punctuation(self):
        """
        Los guiones en árabe suelen usar ، ؛ ؟ como puntuación natural de corte de
        frase. La fase de división debe reconocer esa puntuación; si no, los límites de
        pausa de las cues de edge-tts y los límites de línea del guion quedarían
        desalineados.
        """
        text = "مرحبا بالعالم، كيف حالك؟ هذا اختبار؛ يعمل بشكل جيد."

        self.assertEqual(
            utils.split_string_by_punctuations(text),
            [
                "مرحبا بالعالم",
                "كيف حالك",
                "هذا اختبار",
                "يعمل بشكل جيد",
            ],
        )

    def test_match_script_line_normalizes_arabic_letter_forms(self):
        """
        edge-tts puede normalizar distintas formas de letras del árabe, o devolver texto
        de cues con diacríticos o Tatweel. La correspondencia debe tolerarlo, pero el
        subtítulo final sigue conservando el texto original del guion.
        """
        script_lines = ["أهلاً وسهلاً بك في المدرسة"]

        matched = vs._match_script_line(
            script_lines,
            "اهلا وسهلا بك في المدرسه",
            0,
        )

        self.assertEqual(matched, script_lines[0])

    def test_edge_cue_aggregation_handles_arabic_variant_forms(self):
        """
        Reproduce la ruta principal del fallo de subtítulos en árabe: cuando el guion
        contiene formas de letras como أ/ة y la cue de edge devuelve formas normalizadas
        como ا/ه, la agregación debe seguir generando subtítulos completos, sin volver a
        Whisper.
        """
        text = "أهلاً وسهلاً بك في المدرسة؟ هذا اختبار رائع، شكراً لك."
        script_lines = utils.split_string_by_punctuations(text)
        cue_texts = [
            "اهلا وسهلا بك في المدرسه",
            "هذا اختبار رائع",
            "شكرا لك",
        ]
        sub_maker = SimpleNamespace(
            cues=[
                SimpleNamespace(
                    content=cue_text,
                    start=timedelta(seconds=index),
                    end=timedelta(seconds=index + 0.8),
                )
                for index, cue_text in enumerate(cue_texts)
            ]
        )

        sub_items = vs._build_subtitle_items_from_edge_cues(sub_maker, script_lines)

        self.assertEqual(len(sub_items), len(script_lines))
        self.assertIn("أهلاً وسهلاً بك في المدرسة", sub_items[0])
        self.assertIn("شكراً لك", sub_items[-1])

    def test_create_subtitle_ignores_markdown_separator_lines(self):
        """
        Un guion escrito a mano puede contener separadores Markdown como `---`. El TTS
        no lee estas líneas de símbolos, y la agregación de subtítulos tampoco debe
        tratarlas como líneas de subtítulo de destino; si no, los subtítulos reales
        siguientes se atascarían y volverían a Whisper.
        """
        text = "第一段\n---\n第二段"
        sub_maker = SimpleNamespace(
            cues=[
                SimpleNamespace(
                    content="第一段",
                    start=timedelta(seconds=0),
                    end=timedelta(seconds=0.8),
                ),
                SimpleNamespace(
                    content="第二段",
                    start=timedelta(seconds=1),
                    end=timedelta(seconds=1.8),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "subtitle.srt"
            vs.create_subtitle(
                sub_maker=sub_maker,
                text=text,
                subtitle_file=str(subtitle_file),
            )

            subtitle_content = subtitle_file.read_text(encoding="utf-8")

        self.assertIn("第一段", subtitle_content)
        self.assertIn("第二段", subtitle_content)
        self.assertNotIn("---", subtitle_content)
        self.assertNotIn("00:00:00,000 --> 00:00:00,000", subtitle_content)

    def test_create_subtitle_word_level_preserves_edge_cue_timing(self):
        """Las cues detalladas de Edge TTS deben escribirse una a una, sin volver a
        agruparse por puntuación.
        """
        sub_maker = SimpleNamespace(
            cues=[
                SimpleNamespace(
                    content="人工智能",
                    start=timedelta(seconds=0.1),
                    end=timedelta(seconds=0.8),
                ),
                SimpleNamespace(
                    content="正在",
                    start=timedelta(seconds=0.9),
                    end=timedelta(seconds=1.2),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "word-level.srt"
            vs.create_subtitle(
                sub_maker=sub_maker,
                text="人工智能正在发展。",
                subtitle_file=str(subtitle_file),
                word_level=True,
            )
            subtitle_content = subtitle_file.read_text(encoding="utf-8")

        self.assertIn("00:00:00,100 --> 00:00:00,800", subtitle_content)
        self.assertIn("人工智能", subtitle_content)
        self.assertIn("00:00:00,900 --> 00:00:01,200", subtitle_content)
        self.assertIn("正在", subtitle_content)

    def test_create_subtitle_word_level_falls_back_to_provider_granularity(self):
        """
        Cuando un SubMaker antiguo no tiene cues, debe conservarse la granularidad de
        tiempo original que devolvió el servicio de voz.

        Servicios como ElevenLabs o Fish Audio pueden devolver solo líneas de tiempo por
        frase o por oración completa. En ese caso no se puede dividir de forma uniforme
        por caracteres y fingir precisión palabra a palabra, o los subtítulos se irían
        desviando poco a poco de la voz real.
        """
        sub_maker = SimpleNamespace(
            cues=[],
            subs=["Hello world"],
            # El offset del SubMaker antiguo usa marcas de tiempo enteras en unidades de
            # 100 nanosegundos.
            offset=[(2_000_000, 11_000_000)],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "word-level-fallback.srt"
            vs.create_subtitle(
                sub_maker=sub_maker,
                text="Hello world",
                subtitle_file=str(subtitle_file),
                word_level=True,
            )
            subtitle_content = subtitle_file.read_text(encoding="utf-8")

        self.assertEqual(subtitle_content.count(" --> "), 1)
        self.assertIn("00:00:00,200 --> 00:00:01,100", subtitle_content)
        self.assertIn("Hello world", subtitle_content)

    def test_create_subtitle_ignores_markdown_underscore_marks(self):
        """
        `_` lo usan a menudo los usuarios como marca de énfasis de Markdown, pero las
        cues que devuelve el TTS normalmente no incluyen esos símbolos de formato. La
        correspondencia debe ignorar `_`, para no generar subtítulos vacíos ni volver a
        Whisper.
        """
        text = "这是_a_测试。"
        sub_maker = SimpleNamespace(
            cues=[
                SimpleNamespace(
                    content="这是a测试",
                    start=timedelta(seconds=0),
                    end=timedelta(seconds=0.8),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "subtitle.srt"
            vs.create_subtitle(
                sub_maker=sub_maker,
                text=text,
                subtitle_file=str(subtitle_file),
            )

            subtitle_content = subtitle_file.read_text(encoding="utf-8")

        self.assertIn("这是a测试", subtitle_content)
        self.assertNotIn("这是_a_测试", subtitle_content)
        self.assertNotIn("00:00:00,000 --> 00:00:00,000", subtitle_content)

    def test_convert_rate_to_percent_signs_zero_rate(self):
        # Las velocidades cercanas pero no exactamente iguales a 1.0 redondean a 0 por
        # ciento. edge-tts rechaza un "0%" sin signo (ValueError: Invalid rate '0%'),
        # así que el helper debe emitir "+0%" con prefijo de signo. Test de regresión de
        # ese fallo.
        self.assertEqual(vs.convert_rate_to_percent(1.0), "+0%")
        self.assertEqual(vs.convert_rate_to_percent(1.004), "+0%")
        self.assertEqual(vs.convert_rate_to_percent(0.997), "+0%")
        self.assertEqual(vs.convert_rate_to_percent(1.5), "+50%")
        self.assertEqual(vs.convert_rate_to_percent(0.8), "-20%")

    def test_convert_rate_to_percent_invalid_values_default_to_normal(self):
        # La API y los scripts por lotes pueden pasar una velocidad vacía como 0, None o
        # cadena vacía; nada de eso debe hacer que edge-tts reciba -100% ni provocar una
        # excepción, sino tratarse como velocidad normal.
        self.assertEqual(vs.convert_rate_to_percent(0), "+0%")
        self.assertEqual(vs.convert_rate_to_percent(0.0), "+0%")
        self.assertEqual(vs.convert_rate_to_percent(None), "+0%")
        self.assertEqual(vs.convert_rate_to_percent(""), "+0%")
        self.assertEqual(vs.convert_rate_to_percent(float("nan")), "+0%")
        self.assertEqual(vs.convert_rate_to_percent(float("inf")), "+0%")


def _write_test_wav(filepath: str, duration_seconds: float = 1.0, sample_rate: int = 24000) -> str:
    import wave
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(round(duration_seconds * sample_rate))
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_samples)
    return filepath


class TestElevenLabsVoice(unittest.TestCase):

    def test_is_elevenlabs_voice_true(self):
        self.assertTrue(vs.is_elevenlabs_voice("elevenlabs:pNInz6obpgDQGcFmaJgB:Adam"))

    def test_is_elevenlabs_voice_false_azure(self):
        self.assertFalse(vs.is_elevenlabs_voice("zh-CN-XiaoxiaoNeural-Female"))

    def test_is_elevenlabs_voice_false_siliconflow(self):
        self.assertFalse(vs.is_elevenlabs_voice("siliconflow:model:voice-Male"))

    def test_is_elevenlabs_voice_empty(self):
        self.assertFalse(vs.is_elevenlabs_voice(""))

    def test_is_elevenlabs_voice_none(self):
        self.assertFalse(vs.is_elevenlabs_voice(None))

    def test_get_elevenlabs_voices_empty_api_key(self):
        result = vs.get_elevenlabs_voices("")
        self.assertEqual(result, [])

    @patch("app.services.voice.requests.get")
    def test_get_elevenlabs_voices_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "voices": [
                {"voice_id": "abc123", "name": "Adam"},
                {"voice_id": "def456", "name": "Rachel"},
            ]
        }
        result = vs.get_elevenlabs_voices("fake-api-key")
        self.assertEqual(result, [
            "elevenlabs:abc123:Adam",
            "elevenlabs:def456:Rachel",
        ])
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        self.assertIn("xi-api-key", call_kwargs.kwargs.get("headers", {}))

    @patch("app.services.voice.requests.get")
    def test_get_elevenlabs_voices_http_error(self, mock_get):
        mock_get.return_value.status_code = 401
        mock_get.return_value.text = "Unauthorized"
        result = vs.get_elevenlabs_voices("bad-key")
        self.assertEqual(result, [])

    @patch("app.services.voice.requests.get")
    def test_get_elevenlabs_voices_does_not_forward_key_on_redirect(self, mock_get):
        mock_get.return_value.status_code = 302
        mock_get.return_value.text = "moved"

        self.assertEqual(vs.get_elevenlabs_voices("secret-key"), [])
        self.assertEqual(mock_get.call_args.kwargs["allow_redirects"], False)

    @patch("app.services.voice.requests.get")
    def test_get_elevenlabs_voices_network_error(self, mock_get):
        import requests as req_lib
        mock_get.side_effect = req_lib.exceptions.ConnectionError("timeout")
        result = vs.get_elevenlabs_voices("fake-key")
        self.assertEqual(result, [])

    @patch("app.services.voice._validate_remote_tts_audio", new=lambda _path: None)
    @patch("app.services.voice.requests.post")
    @patch("app.services.voice.AudioFileClip")
    @patch("app.services.voice.config")
    def test_elevenlabs_tts_success(self, mock_config, mock_clip_cls, mock_post):
        mock_config.elevenlabs.get.return_value = "fake-api-key"
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = b"fake-mp3-bytes"
        mock_post.return_value.iter_content.return_value = iter((b"fake-mp3-bytes",))
        mock_clip_cls.return_value.duration = 3.0
        mock_clip_cls.return_value.close = lambda: None

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            out_path = f.name

        try:
            result = vs.elevenlabs_tts("Hello world", "abc123", out_path)
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, "subs"))
            self.assertTrue(hasattr(result, "offset"))
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    @patch("app.services.voice._validate_remote_tts_audio", new=lambda _path: None)
    def test_elevenlabs_tts_rejects_oversize_audio_without_replacing_existing_file(self):
        """Una respuesta 200 de pago puede ser mucho más grande que una voz utilizable."""
        response = SimpleNamespace(
            status_code=200,
            content=b"oversize-audio",
            iter_content=lambda chunk_size: iter((b"oversize-audio",)),
            close=lambda: None,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "voice.mp3"
            output.write_bytes(b"previous-valid-audio")
            with (
                patch.object(vs, "get_elevenlabs_api_key", return_value="test-key"),
                patch.object(vs.requests, "post", return_value=response) as post,
                patch.object(vs, "AudioFileClip") as clip,
                patch.object(vs, "_ELEVENLABS_TTS_MAX_AUDIO_BYTES", 5, create=True),
            ):
                clip.return_value.duration = 1.0
                result = vs.elevenlabs_tts("Hello", "voice-id", str(output))

            self.assertIsNone(result)
            self.assertEqual(output.read_bytes(), b"previous-valid-audio")
            self.assertEqual(post.call_count, 1)
            self.assertEqual(sorted(path.name for path in Path(tmp_dir).iterdir()), ["voice.mp3"])

    @patch("app.services.voice._validate_remote_tts_audio", new=lambda _path: None)
    def test_elevenlabs_tts_decode_failure_preserves_existing_file(self):
        """Una respuesta correcta pero dañada no debe publicar un MP3 final parcial."""
        response = SimpleNamespace(
            status_code=200,
            content=b"corrupt-audio",
            iter_content=lambda chunk_size: iter((b"corrupt-audio",)),
            close=lambda: None,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "voice.mp3"
            output.write_bytes(b"previous-valid-audio")
            with (
                patch.object(vs, "get_elevenlabs_api_key", return_value="test-key"),
                patch.object(vs.requests, "post", return_value=response) as post,
                patch.object(vs, "AudioFileClip", side_effect=OSError("bad mp3")),
            ):
                result = vs.elevenlabs_tts("Hello", "voice-id", str(output))

            self.assertIsNone(result)
            self.assertEqual(output.read_bytes(), b"previous-valid-audio")
            self.assertEqual(post.call_count, 1)
            self.assertEqual(sorted(path.name for path in Path(tmp_dir).iterdir()), ["voice.mp3"])

    def test_elevenlabs_tts_does_not_follow_or_retry_redirect(self):
        response = SimpleNamespace(
            status_code=307,
            close=unittest.mock.Mock(),
            iter_content=unittest.mock.Mock(return_value=iter((b"moved",))),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "voice.mp3"
            with (
                patch.object(vs, "get_elevenlabs_api_key", return_value="secret-key"),
                patch.object(vs.requests, "post", return_value=response) as post,
            ):
                self.assertIsNone(vs.elevenlabs_tts("Hello", "voice-id", str(output)))

            self.assertEqual(post.call_count, 1)
            self.assertEqual(post.call_args.kwargs["allow_redirects"], False)
            self.assertFalse(output.exists())
            response.close.assert_called_once()

    @patch("app.services.voice.config")
    def test_elevenlabs_tts_no_api_key(self, mock_config):
        mock_config.elevenlabs.get.return_value = ""
        # La resolución de la clave incluye un respaldo por variable de entorno, así que
        # el test debe vaciar explícitamente el entorno del host, para que el equipo de
        # desarrollo o la CI no cambien la premisa de "sin configurar" por tener
        # definida ELEVENLABS_API_KEY.
        with patch.dict(os.environ, {}, clear=True):
            result = vs.elevenlabs_tts("Hello", "abc123", "/tmp/test.mp3")
        self.assertIsNone(result)

    @patch("app.services.voice.config")
    def test_elevenlabs_tts_empty_text(self, mock_config):
        mock_config.elevenlabs.get.return_value = "fake-key"
        result = vs.elevenlabs_tts("  ", "abc123", "/tmp/test.mp3")
        self.assertIsNone(result)

    def test_elevenlabs_api_key_prefers_config(self):
        with (
            patch.object(vs.config, "elevenlabs", {"api_key": "config-key"}),
            patch.dict(os.environ, {"ELEVENLABS_API_KEY": "env-key"}),
        ):
            self.assertEqual(vs.get_elevenlabs_api_key(), "config-key")

    def test_elevenlabs_api_key_falls_back_to_environment(self):
        with (
            patch.object(vs.config, "elevenlabs", {"api_key": ""}),
            patch.dict(os.environ, {"ELEVENLABS_API_KEY": " env-key "}),
        ):
            self.assertEqual(vs.get_elevenlabs_api_key(), "env-key")

    def test_elevenlabs_api_key_matches_music_service(self):
        """El TTS y la música comparten la misma configuración de cuenta, así que ambas
        cadenas de generación deben resolver la misma clave.
        """
        from app.services import elevenlabs_music

        for configured_key, env_key in (("config-key", "env-key"), ("", "env-key")):
            with self.subTest(configured_key=configured_key):
                with (
                    patch.object(
                        vs.config, "elevenlabs", {"api_key": configured_key}
                    ),
                    patch.object(
                        elevenlabs_music.config,
                        "elevenlabs",
                        {"api_key": configured_key},
                    ),
                    patch.dict(os.environ, {"ELEVENLABS_API_KEY": env_key}),
                ):
                    self.assertEqual(
                        vs.get_elevenlabs_api_key(),
                        elevenlabs_music.get_api_key(),
                    )


    @patch("app.services.voice._validate_remote_tts_audio", new=lambda _path: None)
    def test_siliconflow_subtitle_spans_full_audio_duration(self):
        """La última entrada del subtítulo debe terminar en el final real del audio, sin
        recortarse antes.

        El antiguo bucle ad hoc aplicaba la división entera por separado a cada frase,
        así que el truncamiento acumulado hacía que el último subtítulo siempre
        terminara unas unidades antes del final real del audio. Todos los demás
        proveedores TTS ya delegan en populate_legacy_submaker_with_full_text, que ancla
        la última entrada a la duración completa; este test verifica que siliconflow_tts
        ahora hace lo mismo.
        """
        audio_duration_seconds = 7.3
        expected_end_100ns = int(audio_duration_seconds * 10_000_000)

        fake_response = SimpleNamespace(status_code=200, content=b"fake-mp3")
        fake_clip = SimpleNamespace(
            duration=audio_duration_seconds, close=lambda: None
        )

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.object(vs.requests, "post", return_value=fake_response),
            patch.object(vs, "AudioFileClip", return_value=fake_clip),
            patch.object(vs.config, "siliconflow", {"api_key": "test-key"}),
        ):
            voice_file = str(Path(tmp_dir) / "test.mp3")
            sub_maker = vs.siliconflow_tts(
                text=(
                    "First sentence. Second sentence. "
                    "Third sentence. Fourth sentence."
                ),
                model="FunAudioLLM/CosyVoice2-0.5B",
                voice="FunAudioLLM/CosyVoice2-0.5B:alex",
                voice_rate=1.0,
                voice_file=voice_file,
            )

        self.assertIsNotNone(sub_maker)
        offsets = getattr(sub_maker, "offset", [])
        self.assertGreater(
            len(offsets), 1, "multi-sentence text must produce multiple subtitles"
        )
        last_end = offsets[-1][1]
        self.assertEqual(
            last_end,
            expected_end_100ns,
            f"last subtitle end ({last_end}) must equal the full audio duration "
            f"({expected_end_100ns} units = {audio_duration_seconds}s)",
        )

    def test_siliconflow_tts_bounds_each_network_attempt(self):
        """Un tiempo de espera previo a la conexión puede reintentarse sin enviar trabajo
        de pago.
        """
        timeouts = []

        def stalled_post(_url, **kwargs):
            timeouts.append(kwargs.get("timeout"))
            raise vs.requests.exceptions.ConnectTimeout("could not connect")

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(vs.requests, "post", side_effect=stalled_post),
            patch.object(vs.config, "siliconflow", {"api_key": "test-key"}),
        ):
            voice_file = str(Path(temp_dir) / "narration.mp3")
            result = vs.siliconflow_tts(
                text="An example narration",
                model="FunAudioLLM/CosyVoice2-0.5B",
                voice="FunAudioLLM/CosyVoice2-0.5B:alex",
                voice_rate=1.0,
                voice_file=voice_file,
            )
            self.assertFalse(Path(voice_file).exists())

        self.assertIsNone(result)
        self.assertEqual(timeouts, [(10, 300)] * 3)

    def test_paid_tts_does_not_repeat_ambiguous_read_timeout(self):
        """Una respuesta POST perdida puede haber creado ya voz facturable en remoto."""
        settings = {
            "api_key": "test-key",
            "model_id": vs.MINIMAX_TTS_DEFAULT_MODEL,
            "audio_format": "mp3",
        }
        cases = (
            ("elevenlabs", lambda path: vs.elevenlabs_tts("Hello", "voice", path)),
            ("fish", lambda path: vs.fish_audio_tts("Hello", path)),
            (
                "minimax",
                lambda path: vs.minimax_tts("Hello", "voice", 1.0, path),
            ),
            (
                "siliconflow",
                lambda path: vs.siliconflow_tts(
                    "Hello", "model", "voice", 1.0, path
                ),
            ),
        )
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.object(vs.requests, "post", side_effect=vs.requests.exceptions.ReadTimeout("response lost")) as post,
            patch.object(vs, "get_elevenlabs_api_key", return_value="test-key"),
            patch.object(vs, "get_fish_audio_api_key", return_value="test-key"),
            patch.object(vs, "get_minimax_tts_api_key", return_value="test-key"),
            patch.object(vs, "get_minimax_tts_endpoint", return_value="https://example.test/v1/t2a_v2"),
            patch.object(vs.config, "fish_audio", {"model": vs.FISH_AUDIO_DEFAULT_MODEL}),
            patch.object(vs.config, "minimax_tts", settings),
            patch.object(vs.config, "siliconflow", {"api_key": "test-key"}),
        ):
            for provider, synthesize in cases:
                with self.subTest(provider=provider):
                    post.reset_mock()
                    result = synthesize(str(Path(tmp_dir) / f"{provider}.mp3"))
                    self.assertIsNone(result)
                    post.assert_called_once()

    @patch("app.services.voice._validate_remote_tts_audio", new=lambda _path: None)
    def test_siliconflow_tts_rejects_invalid_success_audio(self):
        """Un HTTP 200 con audio dañado no debe convertirse en un falso éxito de 10
        segundos.
        """
        fake_response = SimpleNamespace(status_code=200, content=b"invalid mp3")
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(vs.requests, "post", return_value=fake_response) as post,
            patch.object(vs, "AudioFileClip", side_effect=OSError("invalid audio")),
            patch.object(vs.config, "siliconflow", {"api_key": "test-key"}),
        ):
            voice_file = str(Path(temp_dir) / "narration.mp3")
            result = vs.siliconflow_tts(
                text="An example narration",
                model="FunAudioLLM/CosyVoice2-0.5B",
                voice="FunAudioLLM/CosyVoice2-0.5B:alex",
                voice_rate=1.0,
                voice_file=voice_file,
            )
            self.assertFalse(Path(voice_file).exists())

        self.assertIsNone(result)
        post.assert_called_once()

    @patch("app.services.voice._validate_remote_tts_audio", new=lambda _path: None)
    def test_siliconflow_tts_invalid_audio_preserves_existing_narration(self):
        fake_response = SimpleNamespace(status_code=200, content=b"invalid mp3")
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(vs.requests, "post", return_value=fake_response),
            patch.object(vs, "AudioFileClip", side_effect=OSError("invalid audio")),
            patch.object(vs.config, "siliconflow", {"api_key": "test-key"}),
        ):
            voice_file = Path(temp_dir) / "narration.mp3"
            voice_file.write_bytes(b"previous valid narration")
            result = vs.siliconflow_tts(
                text="An example narration",
                model="FunAudioLLM/CosyVoice2-0.5B",
                voice="FunAudioLLM/CosyVoice2-0.5B:alex",
                voice_rate=1.0,
                voice_file=str(voice_file),
            )

            self.assertIsNone(result)
            self.assertEqual(voice_file.read_bytes(), b"previous valid narration")
            self.assertEqual(list(Path(temp_dir).iterdir()), [voice_file])

    @patch("app.services.voice._validate_remote_tts_audio", new=lambda _path: None)
    def test_siliconflow_tts_publishes_only_after_audio_validation(self):
        fake_response = SimpleNamespace(status_code=200, content=b"new mp3")
        with tempfile.TemporaryDirectory() as temp_dir:
            voice_file = Path(temp_dir) / "narration.mp3"
            voice_file.write_bytes(b"previous valid narration")

            def validate_audio(candidate):
                self.assertNotEqual(Path(candidate), voice_file)
                self.assertEqual(voice_file.read_bytes(), b"previous valid narration")
                self.assertEqual(Path(candidate).read_bytes(), b"new mp3")
                return SimpleNamespace(duration=2.5, close=lambda: None)

            with (
                patch.object(vs.requests, "post", return_value=fake_response),
                patch.object(vs, "AudioFileClip", side_effect=validate_audio),
                patch.object(vs.config, "siliconflow", {"api_key": "test-key"}),
            ):
                result = vs.siliconflow_tts(
                    text="An example narration",
                    model="FunAudioLLM/CosyVoice2-0.5B",
                    voice="FunAudioLLM/CosyVoice2-0.5B:alex",
                    voice_rate=1.0,
                    voice_file=str(voice_file),
                )

            self.assertIsNotNone(result)
            self.assertEqual(voice_file.read_bytes(), b"new mp3")
            self.assertEqual(list(Path(temp_dir).iterdir()), [voice_file])

    def test_pause_tag_detection_and_parsing(self):
        """Prueba la detección, el análisis y la limpieza de etiquetas de pausa en varios
        idiomas.
        """
        sample_script = (
            "Hola a todos. [pausa: 2s] "
            "Welcome back. [pause: 1.5s] "
            "今天天气很好。[停顿: 3秒] "
            "잠시 멈춤 [일시중지: 500ms] "
            "Final sentence."
        )

        self.assertTrue(utils.has_pause_tags(sample_script))
        self.assertFalse(utils.has_pause_tags("Plain script with no pause tags."))

        segments = utils.parse_script_with_pauses(sample_script)
        self.assertEqual(len(segments), 9)
        self.assertEqual(segments[0], ("speech", "Hola a todos."))
        self.assertEqual(segments[1], ("pause", 2.0))
        self.assertEqual(segments[2], ("speech", "Welcome back."))
        self.assertEqual(segments[3], ("pause", 1.5))
        self.assertEqual(segments[4], ("speech", "今天天气很好。"))
        self.assertEqual(segments[5], ("pause", 3.0))
        self.assertEqual(segments[6], ("speech", "잠시 멈춤"))
        self.assertEqual(segments[7], ("pause", 0.5))
        self.assertEqual(segments[8], ("speech", "Final sentence."))

        cleaned = utils.remove_pause_tags(sample_script)
        self.assertNotIn("[pausa", cleaned)
        self.assertNotIn("[pause", cleaned)
        self.assertNotIn("[停顿", cleaned)
        self.assertNotIn("[일시중지", cleaned)

        normalized = utils.normalize_script_for_subtitle_matching(sample_script)
        self.assertNotIn("[pausa", normalized)
        self.assertIn("Hola a todos", normalized)
        self.assertIn("Final sentence", normalized)

        # Tests de sintaxis flexible: sin dos puntos, con paréntesis y con duración por
        # defecto
        flexible_script = "Intro. [pausa 1s] Middle. (pausa: 2s) Next. [pause] End."
        self.assertTrue(utils.has_pause_tags(flexible_script))
        flex_segments = utils.parse_script_with_pauses(flexible_script)
        self.assertEqual(len(flex_segments), 7)
        self.assertEqual(flex_segments[0], ("speech", "Intro."))
        self.assertEqual(flex_segments[1], ("pause", 1.0))
        self.assertEqual(flex_segments[2], ("speech", "Middle."))
        self.assertEqual(flex_segments[3], ("pause", 2.0))
        self.assertEqual(flex_segments[4], ("speech", "Next."))
        self.assertEqual(flex_segments[5], ("pause", 1.0))
        self.assertEqual(flex_segments[6], ("speech", "End."))

        flex_cleaned = utils.remove_pause_tags(flexible_script)
        self.assertNotIn("[pausa", flex_cleaned)
        self.assertNotIn("(pausa", flex_cleaned)
        self.assertNotIn("[pause", flex_cleaned)

    def test_pause_tags_accept_duration_directly_after_cjk_keyword(self):
        """En chino y japonés se suele escribir sin dos puntos ni espacios, y `[停顿3秒]`
        también debe reconocerse como pausa y no leerse.
        """
        script = "大家好。[停顿3秒]今天聊聊理财。[ポーズ2秒]最後。[pause2s]End."

        self.assertTrue(utils.has_pause_tags("[暂停2秒]"))
        self.assertEqual(
            utils.parse_script_with_pauses(script),
            [
                ("speech", "大家好。"),
                ("pause", 3.0),
                ("speech", "今天聊聊理财。"),
                ("pause", 2.0),
                ("speech", "最後。"),
                ("pause", 2.0),
                ("speech", "End."),
            ],
        )
        self.assertNotIn("停顿", utils.remove_pause_tags(script))

        # Un texto normal en el que la palabra clave va seguida directamente de letras
        # sigue sin ser una etiqueta de pausa.
        for text in ("[paused]", "(pauses)", "[暂停一下]"):
            self.assertFalse(utils.has_pause_tags(text), text)

    def test_concat_audio_files_reads_pcm_in_bounded_chunks(self):
        """Una narración larga con pausas no debería cargar cada WAV en memoria."""
        real_wave_open = vs.wave.open
        read_sizes = []

        class RecordingReader:
            def __init__(self, reader):
                self.reader = reader

            def __enter__(self):
                self.reader.__enter__()
                return self

            def __exit__(self, *args):
                return self.reader.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self.reader, name)

            def readframes(self, frame_count):
                read_sizes.append(frame_count)
                return self.reader.readframes(frame_count)

        def recording_wave_open(file, mode):
            opened = real_wave_open(file, mode)
            return RecordingReader(opened) if mode == "rb" else opened

        with tempfile.TemporaryDirectory() as temp_dir:
            inputs = [str(Path(temp_dir) / f"input-{i}.wav") for i in range(2)]
            output = str(Path(temp_dir) / "joined.wav")
            for input_path in inputs:
                with real_wave_open(input_path, "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(24000)
                    writer.writeframes(b"\x01\x00" * 20000)

            with patch.object(vs.wave, "open", side_effect=recording_wave_open):
                self.assertTrue(vs._concat_audio_files(inputs, output))

            with real_wave_open(output, "rb") as result:
                self.assertEqual(result.getnframes(), 40000)

        self.assertTrue(read_sizes)
        self.assertLessEqual(max(read_sizes), 8192)

    def test_pause_audio_decode_timeout_fails_without_publishing(self):
        def timed_out(command, **kwargs):
            self.assertIn("-nostdin", command)
            self.assertGreater(kwargs.get("timeout", 0), 0)
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])

        with tempfile.TemporaryDirectory() as temp_dir:
            inputs = [str(Path(temp_dir) / f"chunk-{index}.mp3") for index in range(2)]
            for input_path in inputs:
                Path(input_path).write_bytes(b"compressed-audio")
            output = str(Path(temp_dir) / "voice.mp3")
            with patch.object(vs.subprocess, "run", side_effect=timed_out):
                self.assertFalse(vs._concat_audio_files(inputs, output))
            self.assertFalse(Path(output).exists())

    def test_pause_audio_encode_timeout_preserves_previous_output(self):
        def timed_out(command, **kwargs):
            self.assertIn("-nostdin", command)
            self.assertGreater(kwargs.get("timeout", 0), 0)
            Path(command[-1]).write_bytes(b"partial-output")
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])

        with tempfile.TemporaryDirectory() as temp_dir:
            inputs = [str(Path(temp_dir) / f"chunk-{index}.wav") for index in range(2)]
            for input_path in inputs:
                with vs.wave.open(input_path, "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(24000)
                    writer.writeframes(b"\0\0" * 20)
            output = Path(temp_dir) / "voice.mp3"
            output.write_bytes(b"previous-success")
            with patch.object(vs.subprocess, "run", side_effect=timed_out):
                self.assertFalse(vs._concat_audio_files(inputs, str(output)))
            self.assertEqual(output.read_bytes(), b"previous-success")

    def test_tts_with_pauses_shifts_submaker_timeline(self):
        """Prueba que, con etiquetas de pausa, la línea de tiempo del SubMaker y el montaje
        del audio se desplazan correctamente.
        """
        from edge_tts.srt_composer import Subtitle

        fake_sub1 = vs.ensure_legacy_submaker_fields(vs.SubMaker())
        fake_sub1.cues = [
            Subtitle(1, timedelta(seconds=0.1), timedelta(seconds=1.5), "Segment 1"),
        ]
        fake_sub1.subs = ["Segment 1"]
        fake_sub1.offset = [(1000000, 15000000)]

        fake_sub2 = vs.ensure_legacy_submaker_fields(vs.SubMaker())
        fake_sub2.cues = [
            Subtitle(1, timedelta(seconds=0.2), timedelta(seconds=1.8), "Segment 2"),
        ]
        fake_sub2.subs = ["Segment 2"]
        fake_sub2.offset = [(2000000, 18000000)]

        def fake_single_tts(text, voice_name, voice_rate, voice_file, voice_volume=1.0):
            if "Segment 1" in text:
                _write_test_wav(voice_file, 1.5)
                return fake_sub1
            _write_test_wav(voice_file, 1.8)
            return fake_sub2

        def fake_silence(duration, voice_file):
            _write_test_wav(voice_file, duration)
            return True

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.object(vs, "_single_tts", side_effect=fake_single_tts) as mock_single_tts,
            patch.object(vs, "generate_silent_audio", side_effect=fake_silence) as mock_silence,
            patch.object(vs, "_concat_audio_files", return_value=True) as mock_concat,
        ):
            out_file = str(Path(tmp_dir) / "combined.mp3")
            script = "Segment 1. [pausa: 2s] Segment 2."
            result_submaker = vs.tts(
                text=script,
                voice_name="zh-CN-XiaoxiaoNeural",
                voice_rate=1.0,
                voice_file=out_file,
            )

            self.assertIsNotNone(result_submaker)
            self.assertEqual(mock_single_tts.call_count, 2)
            mock_silence.assert_called_once()
            mock_concat.assert_called_once()

            # Verifica que las cues del segundo segmento se desplazan 1,5 s + 2,0 s =
            # 3,5 s
            self.assertEqual(len(result_submaker.cues), 2)
            self.assertAlmostEqual(result_submaker.cues[0].start.total_seconds(), 0.1, places=2)
            self.assertAlmostEqual(result_submaker.cues[0].end.total_seconds(), 1.5, places=2)
            self.assertAlmostEqual(result_submaker.cues[1].start.total_seconds(), 3.5 + 0.2, places=2)
            self.assertAlmostEqual(result_submaker.cues[1].end.total_seconds(), 3.5 + 1.8, places=2)

            # Verifica que el offset legacy también se desplaza correctamente
            self.assertEqual(len(result_submaker.offset), 2)
            self.assertEqual(result_submaker.offset[0], (1000000, 15000000))
            expected_ns_offset = int(3.5 * 10000000)
            self.assertEqual(
                result_submaker.offset[1],
                (2000000 + expected_ns_offset, 18000000 + expected_ns_offset),
            )


    def test_pause_invalid_and_excessive_durations(self):
        """Prueba que las duraciones no válidas (<= 0 s) se ignoran y que las duraciones
        demasiado largas se limitan al máximo.
        """
        # 1. Duración no válida o cero: no debe reconocerse como segmento de pausa
        zero_script = "Hello [pause: 0s] world. [pause: -2s] Bye."
        segments = utils.parse_script_with_pauses(zero_script)
        speech_only = [s for s in segments if s[0] == "speech"]
        pause_only = [s for s in segments if s[0] == "pause"]
        self.assertEqual(len(pause_only), 0)
        self.assertTrue(any("Hello" in s[1] for s in speech_only))
        self.assertTrue(any("world" in s[1] for s in speech_only))

        # 2. Pausa demasiado larga: lo que supera MAX_PAUSE_DURATION_SECONDS se limita
        long_script = "Hello [pause: 99s] world."
        segments_long = utils.parse_script_with_pauses(long_script)
        pauses = [s for s in segments_long if s[0] == "pause"]
        self.assertEqual(len(pauses), 1)
        self.assertEqual(pauses[0][1], utils.MAX_PAUSE_DURATION_SECONDS)

    def test_pause_consecutive_merging(self):
        """Prueba que las etiquetas de pausa consecutivas se fusionan automáticamente en un
        solo segmento de pausa, con la duración total protegida por el máximo.
        """
        # Dos pausas consecutivas se fusionan en 1 s + 2 s = 3 s
        script = "First part. [pause: 1s] [pause: 2s] Second part."
        segments = utils.parse_script_with_pauses(script)
        pauses = [s for s in segments if s[0] == "pause"]
        self.assertEqual(len(pauses), 1)
        self.assertEqual(pauses[0][1], 3.0)

        # Cuando la suma de varias pausas supera el máximo, tras fusionarse se recorta a
        # MAX_PAUSE_DURATION_SECONDS
        over_script = "Start. [pause: 7s] [pause: 8s] End."
        over_segments = utils.parse_script_with_pauses(over_script)
        over_pauses = [s for s in over_segments if s[0] == "pause"]
        self.assertEqual(len(over_pauses), 1)
        self.assertEqual(over_pauses[0][1], utils.MAX_PAUSE_DURATION_SECONDS)

    def test_pause_leading_and_trailing(self):
        """Prueba el desplazamiento de la pista de audio y de la línea de tiempo con una
        pausa al principio (leading) y al final (trailing).
        """
        from edge_tts.srt_composer import Subtitle

        fake_sub = vs.ensure_legacy_submaker_fields(vs.SubMaker())
        fake_sub.cues = [
            Subtitle(1, timedelta(seconds=0.1), timedelta(seconds=1.2), "Hello"),
        ]
        fake_sub.subs = ["Hello"]
        fake_sub.offset = [(1000000, 12000000)]

        def fake_single_tts(text, voice_name, voice_rate, voice_file, voice_volume=1.0):
            _write_test_wav(voice_file, 1.2)
            return fake_sub

        def fake_silence(duration, voice_file):
            _write_test_wav(voice_file, duration)
            return True

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.object(vs, "_single_tts", side_effect=fake_single_tts),
            patch.object(vs, "generate_silent_audio", side_effect=fake_silence) as mock_silence,
            patch.object(vs, "_concat_audio_files", return_value=True),
        ):
            # Pausa inicial: [pause: 1.5s] Hello
            out_file = str(Path(tmp_dir) / "leading.mp3")
            result = vs.tts(
                text="[pause: 1.5s] Hello",
                voice_name="zh-CN-XiaoxiaoNeural",
                voice_rate=1.0,
                voice_file=out_file,
            )
            self.assertIsNotNone(result)
            mock_silence.assert_called_with(1.5, mock_silence.call_args[0][1])
            # La cue del subtítulo debe empezar en 1,5 s + 0,1 s = 1,6 s
            self.assertAlmostEqual(result.cues[0].start.total_seconds(), 1.6, places=2)

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.object(vs, "_single_tts", side_effect=fake_single_tts),
            patch.object(vs, "generate_silent_audio", side_effect=fake_silence) as mock_silence,
            patch.object(vs, "_concat_audio_files", return_value=True),
        ):
            # Pausa final: Hello [pause: 2s]
            out_file = str(Path(tmp_dir) / "trailing.mp3")
            result = vs.tts(
                text="Hello [pause: 2s]",
                voice_name="zh-CN-XiaoxiaoNeural",
                voice_rate=1.0,
                voice_file=out_file,
            )
            self.assertIsNotNone(result)
            mock_silence.assert_called_with(2.0, mock_silence.call_args[0][1])
            # El subtítulo de la voz debe quedarse en su posición original, sin
            # desplazarse por el silencio final
            self.assertAlmostEqual(result.cues[0].start.total_seconds(), 0.1, places=2)
            self.assertAlmostEqual(result.cues[0].end.total_seconds(), 1.2, places=2)

    def test_pause_script_with_only_pauses(self):
        """Prueba la generación de silencio puro y su seguridad cuando el guion solo
        contiene etiquetas de pausa.
        """
        def fake_silence(duration, voice_file):
            _write_test_wav(voice_file, duration)
            return True

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.object(vs, "generate_silent_audio", side_effect=fake_silence) as mock_silence,
            patch.object(vs, "_single_tts") as mock_single_tts,
        ):
            out_file = str(Path(tmp_dir) / "only_pauses.mp3")
            result = vs.tts(
                text="[pause: 2.5s]",
                voice_name="zh-CN-XiaoxiaoNeural",
                voice_rate=1.0,
                voice_file=out_file,
            )
            self.assertIsNotNone(result)
            mock_single_tts.assert_not_called()
            mock_silence.assert_called_once_with(2.5, out_file)
            self.assertEqual(vs.get_audio_duration(result), 2.5)

            # La generación de subtítulos debe tratar con seguridad un guion sin texto,
            # sin lanzar excepciones
            srt_path = str(Path(tmp_dir) / "only_pauses.srt")
            vs.create_subtitle(result, "[pause: 2.5s]", srt_path, word_level=False)
            vs.create_subtitle(result, "[pause: 2.5s]", srt_path, word_level=True)

    def test_subtitle_sync_sentence_mode_with_pauses(self):
        """Prueba que, en el modo frase (sentence), la línea de tiempo y el contenido de
        los subtítulos con etiquetas de pausa quedan totalmente sincronizados.
        """
        from edge_tts.srt_composer import Subtitle

        fake_sub = vs.ensure_legacy_submaker_fields(vs.SubMaker())
        # La primera frase va de 0,1 s a 1,5 s, y la segunda va tras una pausa de 2 s
        # (de 3,5 s a 5,0 s)
        fake_sub.cues = [
            Subtitle(1, timedelta(seconds=0.1), timedelta(seconds=0.7), "Primera"),
            Subtitle(2, timedelta(seconds=0.8), timedelta(seconds=1.5), "frase."),
            Subtitle(3, timedelta(seconds=3.6), timedelta(seconds=4.2), "Segunda"),
            Subtitle(4, timedelta(seconds=4.3), timedelta(seconds=5.0), "frase."),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            srt_file = str(Path(tmp_dir) / "sentence_mode.srt")
            raw_script = "Primera frase. [pausa: 2s] Segunda frase."
            vs.create_subtitle(
                sub_maker=fake_sub,
                text=raw_script,
                subtitle_file=srt_file,
                word_level=False,
            )

            self.assertTrue(os.path.exists(srt_file))
            with open(srt_file, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("Primera frase", content)
            self.assertIn("Segunda frase", content)
            self.assertNotIn("[pausa", content)
            # La primera frase empieza en 0,1 s y la segunda en 3,6 s (reflejando la
            # pausa de 2 s)
            self.assertIn("00:00:00,100 --> 00:00:01,500", content)
            self.assertIn("00:00:03,600 --> 00:00:05,000", content)

    def test_subtitle_sync_word_mode_with_pauses(self):
        """Prueba que, en el modo palabra a palabra (word_by_word), la línea de tiempo y el
        contenido de los subtítulos con etiquetas de pausa quedan totalmente
        sincronizados.
        """
        from edge_tts.srt_composer import Subtitle

        fake_sub = vs.ensure_legacy_submaker_fields(vs.SubMaker())
        fake_sub.cues = [
            Subtitle(1, timedelta(seconds=0.1), timedelta(seconds=0.5), "Hello"),
            Subtitle(2, timedelta(seconds=0.6), timedelta(seconds=1.2), "world."),
            Subtitle(3, timedelta(seconds=3.3), timedelta(seconds=3.8), "Good"),
            Subtitle(4, timedelta(seconds=3.9), timedelta(seconds=4.5), "morning."),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            srt_file = str(Path(tmp_dir) / "word_mode.srt")
            raw_script = "Hello world. [pause: 2s] Good morning."
            vs.create_subtitle(
                sub_maker=fake_sub,
                text=raw_script,
                subtitle_file=srt_file,
                word_level=True,
            )

            self.assertTrue(os.path.exists(srt_file))
            with open(srt_file, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("Hello", content)
            self.assertIn("world.", content)
            self.assertIn("Good", content)
            self.assertIn("morning.", content)
            # Las entradas del segundo segmento se desplazan correctamente a 3,3 s y 3,9
            # s
            self.assertIn("00:00:00,100 --> 00:00:00,500", content)
            self.assertIn("00:00:03,300 --> 00:00:03,800", content)
            self.assertIn("00:00:03,900 --> 00:00:04,500", content)

    def test_tts_without_pauses_calls_single_tts_directly(self):
        """Verifica que sin etiquetas de pausa se llama directamente a _single_tts, sin
        cambiar en nada el comportamiento ni el rendimiento originales.
        """
        with (
            patch.object(vs, "_single_tts", return_value="normal_submaker") as mock_single_tts,
            patch.object(vs, "_tts_with_pauses") as mock_pauses,
        ):
            res = vs.tts(
                text="This is regular text without any pauses.",
                voice_name="zh-CN-XiaoxiaoNeural",
                voice_rate=1.0,
                voice_file="any.mp3",
            )
            self.assertEqual(res, "normal_submaker")
            mock_single_tts.assert_called_once()
            mock_pauses.assert_not_called()

    def test_pause_invalid_tags_rejected_and_cleaned(self):
        """Verifica que las etiquetas no válidas (como [pause: -2s], [pause: nope] o
        [pause: 0s]) se filtran por completo, sin leerse ni generar silencio.
        """
        # 1. utils.remove_pause_tags elimina por completo todas las etiquetas no válidas
        dirty_text = "Hello [pause: 0s] world. [pause: -2s] [pause: nope] Bye."
        cleaned = utils.remove_pause_tags(dirty_text)
        self.assertNotIn("[pause", cleaned)
        self.assertNotIn("-2s", cleaned)
        self.assertNotIn("nope", cleaned)
        self.assertIn("Hello world.", cleaned)
        self.assertIn("Bye.", cleaned)

        # 2. parse_script_with_pauses ignora las etiquetas no válidas, y el texto no las
        #    contiene
        segments = utils.parse_script_with_pauses(dirty_text)
        pauses = [s for s in segments if s[0] == "pause"]
        speech = [s for s in segments if s[0] == "speech"]
        self.assertEqual(len(pauses), 0)
        self.assertEqual(len(speech), 3)
        for _, text in speech:
            self.assertNotIn("[pause", text)
            self.assertNotIn("-2s", text)
            self.assertNotIn("nope", text)

        # 3. Cuando el guion solo tiene etiquetas no válidas, tts vuelve a _single_tts y
        #    pasa el texto limpio en lugar del texto sucio original
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.object(vs, "_single_tts", return_value="submaker_ok") as mock_single,
        ):
            out_file = str(Path(tmp_dir) / "cleaned.mp3")
            res = vs.tts(
                text="Hello [pause: 0s] world [pause: -2s] [pause: nope]",
                voice_name="zh-CN-XiaoxiaoNeural",
                voice_rate=1.0,
                voice_file=out_file,
            )
            self.assertEqual(res, "submaker_ok")
            mock_single.assert_called_once()
            called_text = mock_single.call_args[1].get("text") or mock_single.call_args[0][0]
            self.assertNotIn("[pause", called_text)
            self.assertNotIn("-2s", called_text)
            self.assertNotIn("nope", called_text)
            self.assertIn("Hello world", called_text)

    def test_pause_minimum_duration_validation(self):
        """Verifica que las pausas diminutas (como 1 ms) se validan y se limitan al umbral
        válido mínimo MIN_PAUSE_DURATION_SECONDS (0,1 s/100 ms).
        """
        script = "Start [pause: 1ms] End"
        segments = utils.parse_script_with_pauses(script)
        pauses = [s for s in segments if s[0] == "pause"]
        self.assertEqual(len(pauses), 1)
        self.assertEqual(pauses[0][1], utils.MIN_PAUSE_DURATION_SECONDS)
        self.assertEqual(utils.MIN_PAUSE_DURATION_SECONDS, 0.1)

    def test_tts_provider_limitation_to_azure_v1(self):
        """Verifica que solo Azure TTS v1 (Edge TTS) entra en la cadena por segmentos, y
        que Gemini/Fish Audio/SiliconFlow/Kokoro mantienen una sola solicitud.
        """
        # 1. Comprobación del proveedor de voz
        self.assertTrue(vs.is_azure_v1_voice("zh-CN-XiaoxiaoNeural"))
        self.assertTrue(vs.is_azure_v1_voice("es-ES-AlvaroNeural"))
        self.assertFalse(vs.is_azure_v1_voice("gemini:Puck-Male"))
        self.assertFalse(vs.is_azure_v1_voice("fish_audio:default"))
        self.assertFalse(vs.is_azure_v1_voice("siliconflow:fishaudio/fish-speech-1.5:alex-Male"))
        self.assertFalse(vs.is_azure_v1_voice("kokoro:af_bella"))
        self.assertFalse(vs.is_azure_v1_voice("elevenlabs:voice-id:voice-name"))

        # 2. Cuando el guion de otros proveedores tiene etiquetas de pausa, hay que
        #    eliminarlas primero y llamar a una síntesis única, sin llamar a
        #    _tts_with_pauses
        non_azure_voices = [
            "gemini:Puck-Male",
            "fish_audio:default",
            "siliconflow:fishaudio/fish-speech-1.5:alex-Male",
            "kokoro:af_bella",
        ]
        script_with_pause = "Part 1. [pause: 2s] Part 2."

        for voice in non_azure_voices:
            with (
                patch.object(vs, "_single_tts", return_value="mock_sub") as mock_single,
                patch.object(vs, "_tts_with_pauses") as mock_split,
            ):
                res = vs.tts(
                    text=script_with_pause,
                    voice_name=voice,
                    voice_rate=1.0,
                    voice_file="dummy.mp3",
                )
                self.assertEqual(res, "mock_sub")
                mock_split.assert_not_called()
                mock_single.assert_called_once()
                called_text = mock_single.call_args[1].get("text") or mock_single.call_args[0][0]
                self.assertNotIn("[pause", called_text)
                self.assertIn("Part 1. Part 2.", called_text)

    def test_real_multi_segment_concatenation_no_drift(self):
        """Test de montaje de audio real por segmentos: 12 audios de 1 s y 11 pausas de 0,5
        s, verificando que el desplazamiento de los subtítulos coincide exactamente con
        las muestras reales decodificadas, sin deriva acumulada.
        """
        import wave
        import struct
        import math
        import subprocess
        from edge_tts.srt_composer import Subtitle

        sr = 24000
        # Genera un WAV PCM estándar de 16 bits mono con una onda senoidal de 1 segundo
        speech_pcm = bytearray()
        for i in range(sr):
            val = int(32767.0 * 0.3 * math.sin(2.0 * math.pi * 440.0 * i / sr))
            speech_pcm.extend(struct.pack("<h", val))

        def make_fake_speech_submaker(idx):
            sub = vs.ensure_legacy_submaker_fields(vs.SubMaker())
            # La cue de cada fragmento de texto va de 0,0 s a 1,0 s dentro de su propio
            # segmento
            sub.cues = [
                Subtitle(1, timedelta(seconds=0.0), timedelta(seconds=1.0), f"Word_{idx}"),
            ]
            sub.subs = [f"Word_{idx}"]
            sub.offset = [(0, 10000000)]
            sub.duration = 1.0
            return sub

        with tempfile.TemporaryDirectory() as tmp_dir:
            def real_single_tts_wav(text, voice_name, voice_rate, voice_file, voice_volume=1.0):
                # Escribe datos de audio WAV reales de 1 segundo
                with wave.open(voice_file, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(speech_pcm)
                idx = int(text.split()[-1]) if text.split()[-1].isdigit() else 0
                return make_fake_speech_submaker(idx)

            # Construye un guion con 12 fragmentos de texto y 11 pausas de 0,5 s
            script_parts = []
            for i in range(12):
                script_parts.append(f"Word {i}")
                if i < 11:
                    script_parts.append("[pause: 0.5s]")
            full_script = " ".join(script_parts)

            out_mp3 = str(Path(tmp_dir) / "output.mp3")

            # Ejecuta la cadena real por segmentos sin simular _concat_audio_files ni
            # get_audio_duration
            with patch.object(vs, "_single_tts", side_effect=real_single_tts_wav):
                result_submaker = vs._tts_with_pauses(
                    text=full_script,
                    voice_name="zh-CN-XiaoxiaoNeural",
                    voice_rate=1.0,
                    voice_file=out_mp3,
                )

            self.assertIsNotNone(result_submaker)
            self.assertTrue(os.path.exists(out_mp3))
            self.assertGreater(os.path.getsize(out_mp3), 0)

            # Verifica que el número de cues de subtítulos es 12
            self.assertEqual(len(result_submaker.cues), 12)

            # Verifica el segmento 12 (el último fragmento de texto):
            # antes ha habido 11 voces de 1,0 s + 11 pausas de 0,5 s = 11,0 s + 5,5 s =
            # 16,50 s
            last_cue = result_submaker.cues[-1]
            # Aserción estricta: el tiempo de inicio debe ser 16,50 s, ¡nunca puede
            # derivar a 17,71 s!
            self.assertAlmostEqual(last_cue.start.total_seconds(), 16.50, places=2)
            self.assertAlmostEqual(last_cue.end.total_seconds(), 17.50, places=2)

            # Audio MP3 de salida decodificado de verdad; verifica que la duración total
            # de las muestras decodificadas es de 17,50 s
            decoded_wav = str(Path(tmp_dir) / "decoded.wav")
            ffmpeg_binary = utils.get_ffmpeg_binary()
            subprocess.run(
                [ffmpeg_binary, "-y", "-i", out_mp3, decoded_wav],
                capture_output=True,
                check=True,
            )
            with wave.open(decoded_wav, "rb") as wf:
                total_frames = wf.getnframes()
                total_sr = wf.getframerate()
                decoded_duration = total_frames / float(total_sr)

            # Verifica que la duración final decodificada coincide exactamente con el
            # final de los subtítulos (17,50 s)
            self.assertAlmostEqual(decoded_duration, 17.50, delta=0.06)

    def test_tts_with_pauses_fails_on_empty_chunk_audio(self):
        """Test de regresión: cuando la síntesis de un fragmento de voz genera un archivo
        vacío (0 bytes) o el archivo falta, _tts_with_pauses debe fallar y devolver
        None, sin volver nunca a generar silencio que oculte el error.
        """
        from edge_tts.srt_composer import Subtitle

        fake_sub = vs.ensure_legacy_submaker_fields(vs.SubMaker())
        fake_sub.cues = [
            Subtitle(1, timedelta(seconds=0.0), timedelta(seconds=1.0), "Hello"),
        ]

        def fake_single_tts_empty(text, voice_name, voice_rate, voice_file, voice_volume=1.0):
            Path(voice_file).touch()  # archivo vacío de 0 bytes
            return fake_sub

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = str(Path(tmp_dir) / "out.mp3")
            with patch.object(vs, "_single_tts", side_effect=fake_single_tts_empty):
                result = vs.tts(
                    text="Hello [pause: 1s] World",
                    voice_name="zh-CN-XiaoxiaoNeural",
                    voice_rate=1.0,
                    voice_file=out_file,
                )
            self.assertIsNone(result)

    def test_tts_with_pauses_fails_on_corrupted_chunk_audio(self):
        """Test de regresión: cuando el audio de un fragmento de voz está dañado y no se
        puede decodificar a PCM, _tts_with_pauses debe dar error y devolver None, en
        lugar de sustituir la narración por silencio y seguir.
        """
        from edge_tts.srt_composer import Subtitle

        fake_sub = vs.ensure_legacy_submaker_fields(vs.SubMaker())
        fake_sub.cues = [
            Subtitle(1, timedelta(seconds=0.0), timedelta(seconds=1.0), "Hello"),
        ]

        def fake_single_tts_corrupted(text, voice_name, voice_rate, voice_file, voice_volume=1.0):
            with open(voice_file, "wb") as f:
                f.write(b"NOT_A_VALID_AUDIO_FILE_DATA_CORRUPTED_1234567890")
            return fake_sub

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = str(Path(tmp_dir) / "out.mp3")
            with patch.object(vs, "_single_tts", side_effect=fake_single_tts_corrupted):
                result = vs.tts(
                    text="Hello [pause: 1s] World",
                    voice_name="zh-CN-XiaoxiaoNeural",
                    voice_rate=1.0,
                    voice_file=out_file,
                )
            self.assertIsNone(result)

    def test_tts_with_pauses_decode_timeout_returns_failure(self):
        def fake_single_tts(text, voice_name, voice_rate, voice_file, voice_volume=1.0):
            Path(voice_file).write_bytes(b"synthesized-audio")
            return vs.SubMaker()

        def timed_out(command, **kwargs):
            self.assertIn("-nostdin", command)
            self.assertGreater(kwargs.get("timeout", 0), 0)
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "narration.mp3"
            with (
                patch.object(vs, "_single_tts", side_effect=fake_single_tts),
                patch.object(vs.subprocess, "run", side_effect=timed_out),
            ):
                result = vs._tts_with_pauses(
                    text="Hello [pause: 1s] world",
                    voice_name="zh-CN-XiaoxiaoNeural",
                    voice_rate=1.0,
                    voice_file=str(output),
                )
            self.assertIsNone(result)
            self.assertFalse(output.exists())

    def test_tts_passes_original_text_unchanged_without_pauses(self):
        """Prueba que sin etiquetas de pausa, tts pasa el texto original tal cual a
        _single_tts, sin aplicar sustituciones de regex ni limpieza.
        """
        original_text = "  Leading and trailing spaces, [regular bracket] and punctuation!  \nNew line here.  "
        with patch.object(vs, "_single_tts", return_value="dummy_submaker") as mock_single:
            result = vs.tts(
                text=original_text,
                voice_name="zh-CN-XiaoxiaoNeural",
                voice_rate=1.0,
                voice_file="out.mp3",
            )
            self.assertEqual(result, "dummy_submaker")
            mock_single.assert_called_once()
            called_text = mock_single.call_args[1].get("text") or mock_single.call_args[0][0]
            self.assertEqual(called_text, original_text)


if __name__ == "__main__":
    # python -m unittest test.services.test_voice.TestVoiceService.test_azure_tts_v1
    # python -m unittest test.services.test_voice.TestVoiceService.test_azure_tts_v2
    unittest.main()
