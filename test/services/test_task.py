import unittest
import os
import shutil
import sys
import tempfile
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from uuid import uuid4

# añade la raíz del proyecto al path de Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import task as tm
from app.models.schema import MaterialInfo, VideoParams
from app.services.state import MemoryState, RedisState
from app.utils import utils

resources_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")
RUN_INTEGRATION_TESTS = os.environ.get("MPT_RUN_INTEGRATION_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}


class TestTaskService(unittest.TestCase):
    def setUp(self):
        # El registro de Futures de publicación es un estado global del proceso.
        # Limpiarlo entre tests evita que un Future simulado afecte a los tests de
        # recuperación posteriores, sin tocar las tareas de producción del pool de hilos
        # real.
        with tm._cross_post_registry_lock:
            tm._cross_post_futures.clear()

    def tearDown(self):
        with tm._cross_post_registry_lock:
            tm._cross_post_futures.clear()

    def test_is_task_busy_covers_generation_and_cross_posting(self):
        """La entrada de borrado debe reconocer a la vez el estado activo de la generación
        de vídeo y de la publicación multiplataforma.
        """
        busy_tasks = (
            {"state": tm.const.TASK_STATE_PROCESSING},
            {
                "state": tm.const.TASK_STATE_COMPLETE,
                "cross_post_state": tm.const.CROSS_POST_STATE_PENDING,
            },
            {
                "state": tm.const.TASK_STATE_COMPLETE,
                "cross_post_state": tm.const.CROSS_POST_STATE_PROCESSING,
            },
        )
        for task in busy_tasks:
            with self.subTest(task=task):
                self.assertTrue(tm.is_task_busy(task))

        self.assertFalse(
            tm.is_task_busy(
                {
                    "state": tm.const.TASK_STATE_COMPLETE,
                    "cross_post_state": tm.const.CROSS_POST_STATE_COMPLETE,
                }
            )
        )
        self.assertFalse(tm.is_task_busy(None))

    def test_generate_script_forwards_advanced_prompt_options(self):
        """
        La entrada de generación de tareas y la WebUI/API comparten VideoParams. Aquí se
        verifica que, al generar el guion automáticamente, los parámetros de prompt
        avanzado siguen llegando a la capa de servicio del LLM, para que no solo
        funcionen en la interfaz /scripts.
        """
        params = VideoParams(
            video_subject="咖啡",
            video_script="",
            video_language="zh-CN",
            paragraph_number=2,
            video_script_prompt="语气轻松",
            custom_system_prompt="Only write short narration.",
        )

        with patch.object(
            tm.llm, "generate_script", return_value="生成的文案"
        ) as generate:
            result = tm.generate_script("task-id", params)

        self.assertEqual(result, "生成的文案")
        generate.assert_called_once_with(
            video_subject="咖啡",
            language="zh-CN",
            paragraph_number=2,
            video_script_prompt="语气轻松",
            custom_system_prompt="Only write short narration.",
        )

    def test_generate_final_videos_forwards_clip_speed_and_fit_mode(self):
        """La capa de orquestación de tareas debe pasar la velocidad de imagen y el modo de
        ajuste al servicio de composición de vídeo.
        """
        params = VideoParams(
            video_subject="test",
            video_count=1,
            video_clip_speed=1.25,
            video_fit_mode="contain",
        )

        with (
            patch.object(tm.video, "combine_videos") as combine_videos,
            patch.object(tm.video, "generate_video"),
            patch.object(tm.sm.state, "update_task"),
        ):
            tm.generate_final_videos(
                task_id="clip-speed-task",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(combine_videos.call_args.kwargs["clip_speed"], 1.25)
        self.assertEqual(
            combine_videos.call_args.kwargs["video_fit_mode"],
            params.video_fit_mode,
        )

    def test_stage_progress_reporter_maps_fraction_into_stage_range(self):
        """
        La proporción de avance 0~1 dentro de una etapa debe convertirse al tramo de
        progreso que ocupa esa etapa, conservando los demás campos de la tarea. Una
        proporción fuera de rango o que no se puede analizar no puede llevar el progreso
        fuera del tramo.
        """
        state = MemoryState()
        state.update_task("stage-progress", progress=40, video_subject="咖啡")

        with patch.object(tm.sm, "state", state):
            report = tm._stage_progress_reporter("stage-progress", 40, 50)
            for fraction, expected in (
                (0.0, 40),
                (0.55, 45),
                (1.0, 50),
                (3.0, 50),
                (-1.0, 40),
            ):
                with self.subTest(fraction=fraction):
                    report(fraction)
                    self.assertEqual(
                        state.get_task("stage-progress")["progress"], expected
                    )

            report(0.5)
            report("not a number")
            task = state.get_task("stage-progress")

        self.assertEqual(task["progress"], 45)
        self.assertEqual(task["state"], tm.const.TASK_STATE_PROCESSING)
        self.assertEqual(task["video_subject"], "咖啡")

    def test_stage_progress_reporter_survives_state_backend_failure(self):
        """El progreso es solo información para mostrar; si el backend de estado no está
        disponible temporalmente, no puede hacer fallar la descarga ni la composición.
        """
        with (
            patch.object(
                tm.sm.state, "update_task", side_effect=RuntimeError("redis down")
            ),
            patch.object(tm.logger, "warning") as warning,
        ):
            tm._stage_progress_reporter("stage-progress", 40, 50)(0.5)

        warning.assert_called_once()
        self.assertIn("redis down", str(warning.call_args.args[0]))

    def test_material_download_moves_progress_between_40_and_50(self):
        """
        La descarga de material es la fase que más tarda con redes lentas; antes el
        progreso se quedaba en el 40 % y saltaba al 50 % solo cuando terminaba todo. La
        proporción de avance informada durante la descarga debe reflejarse en el
        progreso de la tarea.
        """
        params = VideoParams(video_subject="test", video_source="pexels")
        state = MemoryState()
        state.update_task("download-progress", progress=40)
        observed = []

        def fake_download_videos(**kwargs):
            for fraction in (0.25, 0.5, 1.0):
                kwargs["progress_callback"](fraction)
                observed.append(state.get_task("download-progress")["progress"])
            return ["a.mp4"]

        with (
            patch.object(tm.sm, "state", state),
            patch.object(
                tm.material, "download_videos", side_effect=fake_download_videos
            ),
        ):
            result = tm.get_video_materials(
                "download-progress", params, ["scene"], 10
            )

        self.assertEqual(result, ["a.mp4"])
        self.assertEqual(observed, [42, 45, 50])

    def test_clip_processing_moves_progress_through_the_combine_half(self):
        """
        Antes, durante el procesamiento de los clips, el progreso se quedaba fijo en el
        50 %. La composición ocupa la primera mitad de la parte del progreso de cada
        vídeo: con un vídeo va del 50 % al 75 %, y con dos el segundo empieza en el 75
        %.
        """
        for video_count, expected in (
            (1, [[62, 75]]),
            (2, [[56, 62], [81, 87]]),
        ):
            with self.subTest(video_count=video_count):
                params = VideoParams(video_subject="test", video_count=video_count)
                state = MemoryState()
                state.update_task("combine-progress", progress=50)
                observed = []

                def fake_combine_videos(**kwargs):
                    seen = []
                    for fraction in (0.5, 1.0):
                        kwargs["progress_callback"](fraction)
                        seen.append(
                            state.get_task("combine-progress")["progress"]
                        )
                    observed.append(seen)

                with (
                    patch.object(tm.sm, "state", state),
                    patch.object(
                        tm.video, "combine_videos", side_effect=fake_combine_videos
                    ),
                    patch.object(tm.video, "generate_video"),
                    patch.object(tm.task_artifacts, "patch_script_data"),
                ):
                    tm.generate_final_videos(
                        task_id="combine-progress",
                        params=params,
                        downloaded_videos=["material.mp4"],
                        audio_file="audio.mp3",
                        subtitle_path="",
                        audio_duration=5,
                    )

                self.assertEqual(observed, expected)
                self.assertEqual(
                    state.get_task("combine-progress")["progress"], 100
                )

    def test_generate_final_videos_uses_generated_sonilo_music(self):
        """Sonilo debe generar la música para cada vídeo montado y pasarla a la mezcla
        final.
        """
        params = VideoParams(
            video_subject="test",
            video_count=1,
            bgm_type="sonilo",
            sonilo_bgm_prompt="warm acoustic",
        )

        with (
            patch.object(tm.video, "combine_videos"),
            patch.object(
                tm.sonilo,
                "generate_bgm",
                side_effect=lambda **kwargs: kwargs["output_path"],
            ) as generate_bgm,
            patch.object(tm.video, "generate_video") as generate_video,
            patch.object(tm.sm.state, "update_task"),
        ):
            _, _, warnings = tm.generate_final_videos(
                task_id="sonilo-task",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(warnings, [])
        self.assertEqual(generate_bgm.call_args.kwargs["video_duration"], 5)
        self.assertEqual(generate_bgm.call_args.kwargs["prompt"], "warm acoustic")
        self.assertTrue(
            generate_video.call_args.kwargs["bgm_file_override"].endswith(
                "sonilo-bgm-1.m4a"
            )
        )

    def test_generate_final_videos_uses_generated_elevenlabs_music(self):
        """ElevenLabs debe reutilizar la orquestación de música del vídeo y usar el prompt
        de estilo común.
        """
        params = VideoParams(
            video_subject="test",
            video_count=1,
            bgm_type="elevenlabs",
            video_music_prompt="gentle documentary",
        )

        with (
            patch.object(tm.video, "combine_videos"),
            patch.object(
                tm.elevenlabs_music,
                "generate_bgm",
                side_effect=lambda **kwargs: kwargs["output_path"],
            ) as generate_bgm,
            patch.object(tm.video, "generate_video") as generate_video,
            patch.object(tm.sm.state, "update_task"),
        ):
            _, _, warnings = tm.generate_final_videos(
                task_id="elevenlabs-task",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(warnings, [])
        self.assertEqual(generate_bgm.call_args.kwargs["video_duration"], 5)
        self.assertEqual(generate_bgm.call_args.kwargs["prompt"], "gentle documentary")
        self.assertTrue(
            generate_video.call_args.kwargs["bgm_file_override"].endswith(
                "elevenlabs-bgm-1.mp3"
            )
        )

    def test_generate_final_videos_falls_back_on_elevenlabs_failure(self):
        """Cuando ElevenLabs falla temporalmente hay que conservar el vídeo sin música y un
        aviso estructurado.
        """
        params = VideoParams(video_subject="test", bgm_type="elevenlabs")

        with (
            patch.object(tm.video, "combine_videos"),
            patch.object(
                tm.elevenlabs_music,
                "generate_bgm",
                side_effect=tm.elevenlabs_music.ElevenLabsMusicError(
                    "temporary outage"
                ),
            ),
            patch.object(tm.video, "generate_video") as generate_video,
            patch.object(tm.sm.state, "update_task"),
        ):
            final_paths, _, warnings = tm.generate_final_videos(
                task_id="elevenlabs-fallback",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(len(final_paths), 1)
        self.assertEqual(
            warnings,
            [{"code": "elevenlabs_bgm_failed", "video_index": 1}],
        )
        self.assertEqual(generate_video.call_args.kwargs["bgm_file_override"], "")

    def test_generate_final_videos_falls_back_without_bgm_on_sonilo_failure(self):
        """Cuando falla la música de un tercero hay que completar el vídeo y devolver un
        aviso visible, en lugar de descartar todos los productos.
        """
        params = VideoParams(video_subject="test", bgm_type="sonilo")

        with (
            patch.object(tm.video, "combine_videos"),
            patch.object(
                tm.sonilo,
                "generate_bgm",
                side_effect=tm.sonilo.SoniloError("temporary outage"),
            ),
            patch.object(tm.video, "generate_video") as generate_video,
            patch.object(tm.sm.state, "update_task"),
        ):
            final_paths, _, warnings = tm.generate_final_videos(
                task_id="sonilo-fallback",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(len(final_paths), 1)
        self.assertEqual(warnings, [{"code": "sonilo_bgm_failed", "video_index": 1}])
        self.assertEqual(generate_video.call_args.kwargs["bgm_file_override"], "")

    def test_generate_final_videos_skips_sonilo_when_volume_is_zero(self):
        """Con volumen 0 hay que saltarse por completo la generación de Sonilo y desactivar
        explícitamente la música de fondo residual.
        """
        params = VideoParams(
            video_subject="test",
            bgm_type="sonilo",
            bgm_volume=0.0,
            bgm_file="stale-custom-bgm.mp3",
        )

        with (
            patch.object(tm.video, "combine_videos"),
            patch.object(tm.sonilo, "generate_bgm") as generate_bgm,
            patch.object(tm.video, "generate_video", return_value=True) as generate,
            patch.object(tm.sm.state, "update_task"),
        ):
            final_paths, _, warnings = tm.generate_final_videos(
                task_id="sonilo-zero-volume",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(len(final_paths), 1)
        self.assertEqual(warnings, [])
        generate_bgm.assert_not_called()
        self.assertEqual(generate.call_args.kwargs["bgm_file_override"], "")

    def test_generate_final_videos_warns_when_sonilo_mix_fails(self):
        """Si Sonilo genera bien pero la mezcla final falla, la tarea debe conservar el
        vídeo y devolver un aviso.
        """
        params = VideoParams(video_subject="test", bgm_type="sonilo")

        with (
            patch.object(tm.video, "combine_videos"),
            patch.object(
                tm.sonilo,
                "generate_bgm",
                side_effect=lambda **kwargs: kwargs["output_path"],
            ),
            patch.object(tm.video, "generate_video", return_value=False) as generate,
            patch.object(tm.sm.state, "update_task"),
        ):
            final_paths, _, warnings = tm.generate_final_videos(
                task_id="sonilo-mix-fallback",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(len(final_paths), 1)
        self.assertEqual(warnings, [{"code": "sonilo_bgm_failed", "video_index": 1}])
        self.assertTrue(generate.call_args.kwargs["bgm_file_override"].endswith(".m4a"))

    def test_run_pipeline_fails_fast_when_ffmpeg_is_not_ready(self):
        """El flujo de vídeo completo debe confirmar que FFmpeg está disponible antes de
        los servicios de LLM/TTS/material.
        """
        params = VideoParams(video_subject="test")
        state = MemoryState()
        with (
            patch.object(tm.utils, "check_ffmpeg_ready", return_value=False),
            patch.object(tm, "generate_script") as generate_script,
            patch.object(tm, "generate_audio") as generate_audio,
            patch.object(tm, "get_video_materials") as get_materials,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("ffmpeg-missing", params)

        generate_script.assert_not_called()
        generate_audio.assert_not_called()
        get_materials.assert_not_called()
        self.assertEqual(result["state"], tm.const.TASK_STATE_FAILED)
        self.assertEqual(result["failed_stage"], "preflight")
        self.assertIn("ffmpeg", result["error"])

    def test_run_pipeline_skips_ffmpeg_check_for_script_stage(self):
        """La fase del guion no implica composición de audio/vídeo y no debe rechazarse
        porque falte FFmpeg.
        """
        params = VideoParams(video_subject="test")
        state = MemoryState()
        with (
            patch.object(tm.utils, "check_ffmpeg_ready", return_value=False) as check,
            patch.object(tm, "generate_script", return_value="脚本") as generate_script,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("ffmpeg-missing-script-stage", params, stop_at="script")

        check.assert_not_called()
        generate_script.assert_called_once()
        self.assertEqual(result, {"script": "脚本"})

    def test_custom_script_keeps_literal_error_text(self):
        """Una narración del usuario que habla de errores no es un fallo del proveedor LLM."""
        scripts = (
            "The server logged Error: 404 before the page loaded.",
            "Error: 404 is the status shown when a page is missing.",
        )
        for index, script in enumerate(scripts):
            with self.subTest(script=script):
                task_id = f"literal-error-script-{index}"
                state = MemoryState()
                params = VideoParams(video_subject="debugging", video_script=script)
                with patch.object(tm.sm, "state", state):
                    result = tm.start(task_id, params, stop_at="script")

                self.assertEqual(result, {"script": script})
                self.assertEqual(
                    state.get_task(task_id)["state"], tm.const.TASK_STATE_COMPLETE
                )

    def test_generated_script_still_rejects_provider_error_prefix(self):
        """El centinela de error del proveedor debe seguir deteniendo los guiones
        generados.
        """
        state = MemoryState()
        params = VideoParams(video_subject="debugging")
        with (
            patch.object(tm, "generate_script", return_value="Error: invalid API key"),
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("generated-script-error", params, stop_at="script")

        self.assertEqual(result["state"], tm.const.TASK_STATE_FAILED)
        self.assertEqual(result["failed_stage"], "script")
        self.assertEqual(result["error"], "invalid API key")

    def test_run_pipeline_skips_ffmpeg_check_for_terms_stage(self):
        """La fase de términos de búsqueda tampoco necesita FFmpeg y no debe lanzar la
        comprobación.
        """
        params = VideoParams(video_subject="test")
        state = MemoryState()
        with (
            patch.object(tm.utils, "check_ffmpeg_ready", return_value=False) as check,
            patch.object(tm, "generate_script", return_value="脚本"),
            patch.object(tm, "generate_terms", return_value=["term"]),
            patch.object(tm, "save_script_data"),
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("ffmpeg-missing-terms-stage", params, stop_at="terms")

        check.assert_not_called()
        self.assertEqual(result, {"script": "脚本", "terms": ["term"]})

    def test_run_pipeline_proceeds_past_ffmpeg_preflight_when_ready(self):
        """Cuando FFmpeg está disponible, la comprobación no debe bloquear la generación
        del guion posterior.
        """
        params = VideoParams(video_subject="test")
        state = MemoryState()
        with (
            patch.object(tm.utils, "check_ffmpeg_ready", return_value=True) as check,
            patch.object(tm, "generate_script", return_value="脚本") as generate_script,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("ffmpeg-ready", params, stop_at="script")

        # Aunque la fase script no exija FFmpeg, aquí se verifica que se omite la
        # llamada a la función de comprobación, coherente con el acuerdo de "solo
        # comprobar en las fases distintas de script/terms".
        check.assert_not_called()
        generate_script.assert_called_once()
        self.assertEqual(result, {"script": "脚本"})

    def test_start_rejects_missing_sonilo_key_before_costly_pipeline_steps(self):
        """Si a una tarea completa le falta la clave de Sonilo, no se puede llamar antes al
        LLM, al TTS ni a los servicios de material.
        """
        params = VideoParams(video_subject="test", bgm_type="sonilo")
        state = MemoryState()
        with (
            patch.object(tm.sonilo, "is_enabled", return_value=False),
            patch.object(tm, "generate_script") as generate_script,
            patch.object(tm, "generate_audio") as generate_audio,
            patch.object(tm, "get_video_materials") as get_materials,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("missing-sonilo-key", params)

        generate_script.assert_not_called()
        generate_audio.assert_not_called()
        get_materials.assert_not_called()
        failed_task = state.get_task("missing-sonilo-key")
        self.assertEqual(result, failed_task)
        self.assertEqual(failed_task["state"], tm.const.TASK_STATE_FAILED)
        self.assertEqual(failed_task["failed_stage"], "preflight")
        self.assertIn("API Key", failed_task["error"])

    def test_start_does_not_require_sonilo_key_when_volume_is_zero(self):
        """Con volumen 0 no se usa Sonilo, así que aunque falte la clave la tarea debe
        entrar en el flujo normal.
        """
        params = VideoParams(
            video_subject="test",
            bgm_type="sonilo",
            bgm_volume=0.0,
        )
        state = MemoryState()
        with (
            patch.object(tm.sonilo, "is_enabled", return_value=False),
            patch.object(tm, "generate_script", return_value="") as generate_script,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("zero-volume-without-key", params)

        generate_script.assert_called_once_with("zero-volume-without-key", params)
        self.assertEqual(result["failed_stage"], "script")

    def test_loomloom_material_failure_keeps_remote_run_id(self):
        """Si falla después de crearse la ejecución remota, el estado de la tarea debe
        conservar el run ID de LoomLoom.
        """
        params = VideoParams(video_subject="AI 办公", video_source="loomloom")
        settings = tm.loomloom.LoomLoomSettings(
            base_url="https://example.test/loom/v1",
            api_token="test-token",
            market_listing_id=tm.loomloom.DEFAULT_SCRIPT_MARKET_LISTING_ID,
        )
        batch = tm.loomloom.LoomLoomVideoBatch(
            input_rows=(
                {
                    "scenePrompt": "office worker",
                    "aspectRatio": "9:16",
                    "sceneIndex": "1",
                },
            ),
        )
        request = tm.loomloom.LoomLoomConfirmedVideoRequest(
            settings=settings,
            batch=batch,
            listing_version_id="version-1",
            client_request_id="mpt-video-1",
        )
        backend = MagicMock()
        backend.execute.return_value = tm.loomloom.LoomLoomExecution(
            run_id="run-1",
            transaction_id="transaction-1",
            transaction_status="running",
            listing_version_id="version-1",
        )
        backend.wait_for_run.side_effect = tm.loomloom.LoomLoomRunError(
            "remote run timeout"
        )
        state = MemoryState()
        state.update_task(
            "loomloom-material-timeout",
            state=tm.const.TASK_STATE_PROCESSING,
            progress=40,
        )

        with (
            patch.object(tm.sm, "state", state),
            patch.object(
                tm.loomloom,
                "LoomLoomVideoBackend",
                return_value=backend,
            ),
        ):
            result = tm.get_video_materials(
                "loomloom-material-timeout",
                params,
                ["office worker"],
                10,
                loomloom_video_request=request,
            )

        self.assertIsNone(result)
        failed_task = state.get_task("loomloom-material-timeout")
        self.assertEqual(failed_task["state"], tm.const.TASK_STATE_FAILED)
        self.assertEqual(failed_task["failed_stage"], "materials")
        self.assertEqual(failed_task["loomloom_run_id"], "run-1")
        self.assertEqual(failed_task["loomloom_listing_version_id"], "version-1")

    def test_wavespeed_paid_material_failure_keeps_prediction_id(self):
        """Tanto las tareas sin confirmar como las ya pagadas cuya descarga falló deben
        conservar en el estado el ID para recuperarlas.
        """
        params = VideoParams(video_subject="test", video_source="wavespeed")
        for error_type in (
            tm.material.WaveSpeedUnconfirmedTaskError,
            tm.material.WaveSpeedDownloadError,
        ):
            with self.subTest(error_type=error_type):
                state = MemoryState()
                state.update_task("wavespeed-paid-failure", progress=40)
                with (
                    patch.object(tm.sm, "state", state),
                    patch.object(
                        tm.material,
                        "download_videos",
                        side_effect=error_type(
                            "paid run unavailable", prediction_id="pred-123"
                        ),
                    ),
                ):
                    result = tm.get_video_materials(
                        "wavespeed-paid-failure", params, ["scene"], 10
                    )

                self.assertIsNone(result)
                failed_task = state.get_task("wavespeed-paid-failure")
                self.assertEqual(failed_task["state"], tm.const.TASK_STATE_FAILED)
                self.assertEqual(failed_task["failed_stage"], "materials")
                self.assertEqual(failed_task["wavespeed_prediction_id"], "pred-123")

    def test_paid_openai_image_failure_stops_at_material_stage(self):
        """Los resultados de imagen de pago ambiguos o inutilizables deben seguir viéndose
        como fallos.
        """
        params = VideoParams(video_subject="test", video_source="openai_image")
        for error_type in (
            tm.material.OpenAIImageUnconfirmedError,
            tm.material.OpenAIImagePaidResultError,
        ):
            with self.subTest(error_type=error_type):
                state = MemoryState()
                state.update_task("openai-image-paid-failure", progress=40)
                with (
                    patch.object(tm.sm, "state", state),
                    patch.object(
                        tm.material,
                        "download_videos",
                        side_effect=error_type("paid image unavailable"),
                    ),
                ):
                    result = tm.get_video_materials(
                        "openai-image-paid-failure", params, ["scene"], 10
                    )

                self.assertIsNone(result)
                failed_task = state.get_task("openai-image-paid-failure")
                self.assertEqual(failed_task["state"], tm.const.TASK_STATE_FAILED)
                self.assertEqual(failed_task["failed_stage"], "materials")
                self.assertIn("paid image unavailable", failed_task["error"])

    def test_loomloom_state_failure_does_not_abandon_paid_remote_run(self):
        """Aunque el backend de estado no esté disponible, hay que esperar y descargar la
        tarea remota que ya ha empezado a facturarse.
        """
        params = VideoParams(video_subject="AI 办公", video_source="loomloom")
        settings = tm.loomloom.LoomLoomSettings(
            base_url="https://example.test/loom/v1",
            api_token="test-token",
            market_listing_id=tm.loomloom.DEFAULT_VIDEO_MARKET_LISTING_ID,
        )
        request = tm.loomloom.LoomLoomConfirmedVideoRequest(
            settings=settings,
            batch=tm.loomloom.LoomLoomVideoBatch(
                input_rows=(
                    {
                        "scenePrompt": "office worker",
                        "aspectRatio": "9:16",
                        "sceneIndex": "1",
                    },
                )
            ),
            listing_version_id="version-1",
            client_request_id="mpt-video-state-failure",
        )
        backend = MagicMock()
        backend.execute.return_value = tm.loomloom.LoomLoomExecution(
            run_id="paid-run-1",
            transaction_id="transaction-1",
            transaction_status="running",
            listing_version_id="version-1",
        )
        backend.download_video_results.return_value = ("clip.mp4",)
        unavailable_state = MagicMock()
        unavailable_state.patch_task.side_effect = RuntimeError("Redis unavailable")

        with (
            patch.object(tm.sm, "state", unavailable_state),
            patch.object(
                tm.loomloom,
                "LoomLoomVideoBackend",
                return_value=backend,
            ),
            patch.object(tm.time, "sleep") as sleep,
        ):
            result = tm.get_video_materials(
                "loomloom-state-failure",
                params,
                ["office worker"],
                10,
                loomloom_video_request=request,
            )

        self.assertEqual(result, ["clip.mp4"])
        self.assertEqual(
            unavailable_state.patch_task.call_count,
            tm._LOOMLOOM_STATE_WRITE_ATTEMPTS,
        )
        self.assertEqual(
            sleep.call_count,
            tm._LOOMLOOM_STATE_WRITE_ATTEMPTS - 1,
        )
        backend.wait_for_run.assert_called_once_with("paid-run-1")
        backend.download_video_results.assert_called_once()

    def test_mark_task_failed_preserves_a_specific_service_failure(self):
        """Cuando la capa de servicio ya ha registrado el error concreto, la capa de
        orquestación no puede sobrescribirlo con un error genérico.
        """
        state = MemoryState()
        state.update_task(
            "specific-service-failure",
            state=tm.const.TASK_STATE_FAILED,
            progress=40,
            failed_stage="materials",
            error="remote run timed out",
            loomloom_run_id="run-1",
        )

        with patch.object(tm.sm, "state", state):
            result = tm._mark_task_failed(
                "specific-service-failure",
                "materials",
                "failed to prepare video materials",
            )

        self.assertEqual(result["error"], "remote run timed out")
        self.assertEqual(result["loomloom_run_id"], "run-1")

    def test_start_rejects_missing_elevenlabs_key_before_pipeline_steps(self):
        """Si a una tarea completa le falta la clave de ElevenLabs, debe fallar antes de
        cualquier paso de pago.
        """
        params = VideoParams(video_subject="test", bgm_type="elevenlabs")
        state = MemoryState()
        with (
            patch.object(tm.elevenlabs_music, "is_enabled", return_value=False),
            patch.object(tm, "generate_script") as generate_script,
            patch.object(tm, "generate_audio") as generate_audio,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("missing-elevenlabs-key", params)

        generate_script.assert_not_called()
        generate_audio.assert_not_called()
        self.assertEqual(result["state"], tm.const.TASK_STATE_FAILED)
        self.assertEqual(result["failed_stage"], "preflight")
        self.assertIn("ElevenLabs", result["error"])

    def test_start_rejects_free_elevenlabs_plan_before_pipeline_steps(self):
        """Un plan gratuito confirmado no puede gastar antes el saldo del LLM, del TTS ni
        de los servicios de material.
        """
        params = VideoParams(video_subject="test", bgm_type="elevenlabs")
        state = MemoryState()
        with (
            patch.object(tm.elevenlabs_music, "is_enabled", return_value=True),
            patch.object(
                tm.elevenlabs_music,
                "validate_generation_access",
                side_effect=(
                    tm.elevenlabs_music.ElevenLabsPaidPlanRequiredError(
                        "ElevenLabs Music API requires a paid plan"
                    )
                ),
            ) as validate_access,
            patch.object(tm, "generate_script") as generate_script,
            patch.object(tm, "generate_audio") as generate_audio,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("free-elevenlabs-plan", params)

        validate_access.assert_called_once_with()
        generate_script.assert_not_called()
        generate_audio.assert_not_called()
        self.assertEqual(result["failed_stage"], "preflight")
        self.assertIn("paid plan", result["error"])

    def test_start_rejects_oversized_elevenlabs_prompt_before_account_check(self):
        """Cuando la API/CLI se saltan la WebUI, los prompts demasiado largos también deben
        rechazarse antes de los pasos caros.
        """
        params = VideoParams(
            video_subject="test",
            bgm_type="elevenlabs",
            video_music_prompt="x" * 1001,
        )
        state = MemoryState()
        with (
            patch.object(tm.elevenlabs_music, "is_enabled", return_value=True),
            patch.object(
                tm.elevenlabs_music, "validate_generation_access"
            ) as validate_access,
            patch.object(tm, "generate_script") as generate_script,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("oversized-elevenlabs-prompt", params)

        validate_access.assert_not_called()
        generate_script.assert_not_called()
        self.assertEqual(result["failed_stage"], "preflight")
        self.assertIn("1000", result["error"])

    def test_generate_terms_uses_script_order_mode_when_enabled(self):
        """
        El modo por defecto no cambia; solo cuando el usuario activa explícitamente que
        el material siga el orden del guion, la capa de tareas pide al LLM palabras
        clave ordenadas y aumenta un poco su número para cubrir más fragmentos del
        guion.
        """
        params = VideoParams(
            video_subject="城市通勤",
            video_script="",
            match_materials_to_script=True,
        )

        with patch.object(
            tm.llm, "generate_terms", return_value=["city", "train"]
        ) as generate:
            result = tm.generate_terms("task-id", params, "先城市，再地铁")

        self.assertEqual(result, ["city", "train"])
        generate.assert_called_once_with(
            video_subject="城市通勤",
            video_script="先城市，再地铁",
            amount=8,
            match_script_order=True,
        )

    def test_start_stops_before_materials_when_term_provider_fails(self):
        """
        Cuando falla el proveedor de palabras clave, la tarea debe terminar de
        inmediato, sin seguir generando audio ni descargando material.

        Aquí se cubre toda la ruta de propagación del error desde la entrada de la
        tarea, para que en el futuro no se corrija solo el tipo de retorno de la capa de
        servicio y la capa de orquestación convierta la lista vacía en otro valor
        verdadero y siga haciendo solicitudes externas.
        """
        params = VideoParams(
            video_subject="startup story",
            video_script="A short startup story.",
        )
        state = MemoryState()

        with (
            patch.object(
                tm.llm,
                "_generate_response",
                return_value="Error: invalid API key",
            ),
            patch.object(tm, "generate_audio") as generate_audio,
            patch.object(tm, "get_video_materials") as get_video_materials,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("term-provider-error", params)

        generate_audio.assert_not_called()
        get_video_materials.assert_not_called()
        failed_task = state.get_task("term-provider-error")
        self.assertEqual(result, failed_task)
        self.assertEqual(failed_task["state"], tm.const.TASK_STATE_FAILED)
        self.assertEqual(failed_task["failed_stage"], "terms")
        self.assertTrue(failed_task["error"])

    def test_generate_audio_uses_custom_file_inside_task_directory(self):
        task_id = "test-custom-audio-safe"
        task_dir = utils.task_dir(task_id)
        custom_audio_file = os.path.join(task_dir, "custom-audio.mp3")
        with open(custom_audio_file, "wb") as audio:
            audio.write(b"fake audio")

        params = VideoParams(
            video_subject="custom audio",
            video_script="",
            custom_audio_file=custom_audio_file,
            voice_name="test-voice",
        )

        try:
            with (
                patch.object(tm.voice, "tts") as tts,
                patch.object(tm.voice, "get_audio_duration", return_value=7),
            ):
                audio_file, audio_duration, sub_maker = tm.generate_audio(
                    task_id, params, "script"
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertEqual(audio_file, os.path.realpath(custom_audio_file))
        self.assertEqual(audio_duration, 7)
        self.assertIsNone(sub_maker)
        tts.assert_not_called()

    def test_generate_audio_rejects_server_side_custom_file_by_default(self):
        task_id = "test-custom-audio-untrusted-server-side"
        task_dir = utils.task_dir(task_id)
        state = MemoryState()

        with tempfile.NamedTemporaryFile(suffix=".mp3") as server_audio:
            server_audio.write(b"fake audio")
            server_audio.flush()
            params = VideoParams(
                video_subject="custom audio",
                video_script="",
                custom_audio_file=server_audio.name,
                voice_name="test-voice",
            )

            try:
                with (
                    patch.object(tm.voice, "tts") as tts,
                    patch.object(tm.voice, "get_audio_duration") as get_duration,
                    patch.object(tm.sm, "state", state),
                ):
                    audio_file, audio_duration, result_sub_maker = tm.generate_audio(
                        task_id, params, "script"
                    )
            finally:
                shutil.rmtree(task_dir, ignore_errors=True)

        self.assertIsNone(audio_file)
        self.assertIsNone(audio_duration)
        self.assertIsNone(result_sub_maker)
        tts.assert_not_called()
        get_duration.assert_not_called()
        failed_task = state.get_task(task_id)
        self.assertEqual(failed_task["failed_stage"], "audio")
        self.assertIn("directorio de la tarea actual", failed_task["error"])

    def test_external_custom_audio_error_does_not_reveal_file_existence(self):
        task_id = "test-custom-audio-existence-oracle"
        task_dir = utils.task_dir(task_id)

        with tempfile.NamedTemporaryFile(suffix=".mp3") as server_audio:
            external_paths = [server_audio.name, f"{server_audio.name}.missing"]
            errors = []
            try:
                for external_path in external_paths:
                    with self.assertRaises(ValueError) as raised:
                        tm.resolve_custom_audio_file(task_id, external_path)
                    errors.append(str(raised.exception))
            finally:
                shutil.rmtree(task_dir, ignore_errors=True)

        self.assertEqual(errors[0], errors[1])
        self.assertIn("directorio de la tarea actual", errors[0])

    def test_generate_audio_accepts_server_side_custom_file_for_trusted_cli(self):
        task_id = "test-custom-audio-server-side"
        task_dir = utils.task_dir(task_id)

        with tempfile.NamedTemporaryFile(suffix=".mp3") as server_audio:
            server_audio.write(b"fake audio")
            server_audio.flush()
            params = VideoParams(
                video_subject="custom audio",
                video_script="",
                custom_audio_file=server_audio.name,
                voice_name="test-voice",
            )

            try:
                with (
                    patch.object(tm.voice, "tts") as tts,
                    patch.object(tm.voice, "get_audio_duration", return_value=6),
                ):
                    audio_file, audio_duration, result_sub_maker = tm.generate_audio(
                        task_id,
                        params,
                        "script",
                        allow_server_file_input=True,
                    )
            finally:
                shutil.rmtree(task_dir, ignore_errors=True)

        self.assertEqual(audio_file, os.path.realpath(server_audio.name))
        self.assertEqual(audio_duration, 6)
        self.assertIsNone(result_sub_maker)
        tts.assert_not_called()

    def test_generate_audio_rejects_missing_custom_file_without_tts(self):
        task_id = "test-custom-audio-missing"
        task_dir = utils.task_dir(task_id)
        missing_audio_file = os.path.join(task_dir, "missing.mp3")
        params = VideoParams(
            video_subject="custom audio",
            video_script="",
            custom_audio_file=missing_audio_file,
            voice_name="test-voice",
        )
        state = MemoryState()

        try:
            with (
                patch.object(tm.voice, "tts") as tts,
                patch.object(tm.sm, "state", state),
            ):
                audio_file, audio_duration, result_sub_maker = tm.generate_audio(
                    task_id, params, "script"
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertIsNone(audio_file)
        self.assertIsNone(audio_duration)
        self.assertIsNone(result_sub_maker)
        tts.assert_not_called()
        failed_task = state.get_task(task_id)
        self.assertEqual(failed_task["failed_stage"], "audio")
        self.assertIn("no existe", failed_task["error"])

    def test_generate_audio_prefers_file_duration_over_sub_maker(self):
        # Cada fixture hace deliberadamente que la duración del archivo y la duración de
        # SubMaker redondeen hacia arriba a enteros DISTINTOS. Si alguien los
        # "simplifica" a valores que redondean igual, este test ya no puede saber qué
        # fuente usó la implementación: deja de discriminar, en silencio.
        cases = (
            # Los números de reproducción del propio mantenedor de la discusión del PR.
            (8.4, 7.8375, 9),
            # Una duración de archivo entera exacta: demuestra que se usa de verdad
            # math.ceil() y descarta código del estilo int()+1 que añade un segundo
            # espurio. El valor de SubMaker redondea hacia arriba a 7, así que 8 solo
            # puede venir del archivo.
            (8.0, 6.2, 8),
            # Duración del archivo menor que el valor de SubMaker: el único caso en que
            # este cambio hace audio_duration menor que antes (el código antiguo
            # devolvía 8). El contrato es "gana el archivo", no "gana el valor mayor".
            (5.0, 7.8375, 5),
        )

        for file_duration, sub_maker_duration, expected in cases:
            with self.subTest(file_duration=file_duration):
                task_id = f"test-tts-audio-priority-{uuid4().hex}"
                task_dir = utils.task_dir(task_id)
                audio_path = os.path.join(task_dir, "audio.mp3")
                params = VideoParams(
                    video_subject="tts audio",
                    video_script="",
                    voice_name="test-voice",
                )
                sub_maker = MagicMock()

                def fake_duration(target, _file=file_duration, _sub=sub_maker_duration):
                    # Se despacha por el tipo de argumento, nunca por el orden de
                    # llamada: un side_effect en secuencia seguiría pasando contra una
                    # implementación que midiera primero el SubMaker, que es justo la
                    # regresión que este test existe para detectar.
                    return _file if isinstance(target, str) else _sub

                try:
                    with (
                        patch.object(tm.voice, "tts", return_value=sub_maker) as tts,
                        patch.object(
                            tm.voice, "get_audio_duration", side_effect=fake_duration
                        ) as get_duration,
                    ):
                        audio_file, audio_duration, result_sub_maker = tm.generate_audio(
                            task_id, params, "script"
                        )
                finally:
                    shutil.rmtree(task_dir, ignore_errors=True)

                self.assertEqual(audio_file, audio_path)
                self.assertEqual(audio_duration, expected)
                # Comprobar solo el valor seguiría dejando pasar una implementación que
                # devolviera 9.0; la aserción de tipo fija el otro lado del contrato de
                # redondeo, para que una refactorización no pueda quitar math.ceil() y
                # pasar el float tal cual.
                self.assertIsInstance(audio_duration, int)
                self.assertIs(result_sub_maker, sub_maker)
                tts.assert_called_once()
                # Cuando la medición del archivo tiene éxito, el SubMaker no debe
                # medirse en absoluto: exactamente una llamada, y el argumento de esa
                # llamada es la ruta del archivo de audio. Las dos aserciones juntas son
                # las que demuestran el orden de prioridad.
                self.assertEqual(len(get_duration.call_args_list), 1)
                self.assertEqual(get_duration.call_args_list[0].args[0], audio_path)

    def test_generate_audio_falls_back_to_sub_maker_when_file_duration_is_zero(self):
        task_id = "test-tts-audio-fallback"
        task_dir = utils.task_dir(task_id)
        audio_path = os.path.join(task_dir, "audio.mp3")
        params = VideoParams(
            video_subject="tts audio",
            video_script="",
            voice_name="test-voice",
        )
        sub_maker = MagicMock()

        def fake_duration(target):
            # voice.get_audio_duration() devuelve 0.0 cuando falla la medición del
            # archivo (archivo ausente o error de decodificación); solo entonces puede
            # usarse la duración por límites de palabra del SubMaker.
            return 0.0 if isinstance(target, str) else 7.8375

        try:
            with (
                patch.object(tm.voice, "tts", return_value=sub_maker),
                patch.object(
                    tm.voice, "get_audio_duration", side_effect=fake_duration
                ) as get_duration,
            ):
                audio_file, audio_duration, result_sub_maker = tm.generate_audio(
                    task_id, params, "script"
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertEqual(audio_file, audio_path)
        self.assertEqual(audio_duration, 8)
        self.assertIsInstance(audio_duration, int)
        self.assertIs(result_sub_maker, sub_maker)
        self.assertEqual(len(get_duration.call_args_list), 2)
        self.assertEqual(get_duration.call_args_list[0].args[0], audio_path)
        self.assertIs(get_duration.call_args_list[1].args[0], sub_maker)

    def test_generate_audio_fails_when_file_and_sub_maker_durations_are_zero(self):
        # Este cambio sustituye la fuente de audio_duration, así que hay que demostrar
        # que la protección existente de duración cero sigue saltando en lugar de quedar
        # anulada por la nueva rama de medición del archivo.
        task_id = "test-tts-audio-zero-duration"
        task_dir = utils.task_dir(task_id)
        params = VideoParams(
            video_subject="tts audio",
            video_script="",
            voice_name="test-voice",
        )
        sub_maker = MagicMock()

        try:
            with (
                patch.object(tm.voice, "tts", return_value=sub_maker),
                patch.object(tm.voice, "get_audio_duration", return_value=0.0),
                patch.object(tm, "_mark_task_failed") as mark_task_failed,
            ):
                audio_file, audio_duration, result_sub_maker = tm.generate_audio(
                    task_id, params, "script"
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertIsNone(audio_file)
        self.assertIsNone(audio_duration)
        self.assertIsNone(result_sub_maker)
        mark_task_failed.assert_called_once_with(
            task_id, "audio", "la duración del audio generado es cero"
        )

    def test_generate_subtitle_uses_whisper_for_custom_audio_without_sub_maker(self):
        """
        El audio personalizado no pasa por el TTS, así que no hay sub_maker. Whisper
        puede transcribir directamente desde el archivo de audio, y en ese caso la
        protección de sub_maker vacío no debe saltárselo antes.
        """
        task_id = "test-custom-audio-whisper-subtitle"
        task_dir = utils.task_dir(task_id)
        audio_file = os.path.join(task_dir, "custom-audio.mp3")
        Path(audio_file).write_bytes(b"fake audio")
        params = VideoParams(
            video_subject="custom audio",
            video_script="Hello world.",
            subtitle_enabled=True,
            # Al probar la ruta de corrección por frases hay que indicar el modo
            # explícitamente, sin heredar las preferencias de la WebUI del equipo de
            # desarrollo.
            subtitle_display_mode="sentence",
        )

        def fake_whisper_create(audio_file, subtitle_file, word_level=False):
            self.assertFalse(word_level)
            Path(subtitle_file).write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nHello world.\n\n",
                encoding="utf-8",
            )

        try:
            with (
                patch.object(
                    tm.config,
                    "app",
                    dict(tm.config.app, subtitle_provider="whisper"),
                ),
                patch.object(
                    tm.subtitle, "create", side_effect=fake_whisper_create
                ) as create,
                patch.object(tm.subtitle, "correct") as correct,
            ):
                subtitle_path = tm.generate_subtitle(
                    task_id=task_id,
                    params=params,
                    video_script="Hello world.",
                    sub_maker=None,
                    audio_file=audio_file,
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertTrue(subtitle_path.endswith("subtitle.srt"))
        created_path = create.call_args.kwargs["subtitle_file"]
        self.assertNotEqual(created_path, subtitle_path)
        self.assertEqual(Path(created_path).parent, Path(subtitle_path).parent)
        create.assert_called_once_with(
            audio_file=audio_file,
            subtitle_file=created_path,
            word_level=False,
        )
        correct.assert_called_once_with(
            subtitle_file=created_path, video_script="Hello world."
        )

    def test_generate_subtitle_uses_whisper_word_timing_without_correction(self):
        """
        El modo palabra a palabra debe pasar word_level a Whisper y saltarse la
        corrección por el texto de las frases.

        Si se siguiera ejecutando correct(), las entradas palabra a palabra recién
        generadas volverían a agruparse, y aunque la interfaz eligiera mostrar palabra a
        palabra, el vídeo final seguiría mostrándose por frases.
        """
        task_id = "test-custom-audio-whisper-word-subtitle"
        task_dir = utils.task_dir(task_id)
        audio_file = os.path.join(task_dir, "custom-audio.mp3")
        Path(audio_file).write_bytes(b"fake audio")
        params = VideoParams(
            video_subject="custom audio",
            video_script="Hello world.",
            subtitle_enabled=True,
            subtitle_display_mode="word_by_word",
        )

        def fake_whisper_create(audio_file, subtitle_file, word_level=False):
            self.assertTrue(word_level)
            Path(subtitle_file).write_text(
                "1\n00:00:00,000 --> 00:00:00,500\nHello\n\n",
                encoding="utf-8",
            )

        try:
            with (
                patch.object(
                    tm.config,
                    "app",
                    dict(tm.config.app, subtitle_provider="whisper"),
                ),
                patch.object(
                    tm.subtitle, "create", side_effect=fake_whisper_create
                ) as create,
                patch.object(tm.subtitle, "correct") as correct,
            ):
                subtitle_path = tm.generate_subtitle(
                    task_id=task_id,
                    params=params,
                    video_script="Hello world.",
                    sub_maker=None,
                    audio_file=audio_file,
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertTrue(subtitle_path.endswith("subtitle.srt"))
        created_path = create.call_args.kwargs["subtitle_file"]
        self.assertNotEqual(created_path, subtitle_path)
        self.assertEqual(Path(created_path).parent, Path(subtitle_path).parent)
        create.assert_called_once_with(
            audio_file=audio_file,
            subtitle_file=created_path,
            word_level=True,
        )
        correct.assert_not_called()

    def test_generate_subtitle_skips_edge_provider_without_sub_maker(self):
        """
        Los subtítulos de Edge dependen de la línea de tiempo del sub_maker que devuelve
        el TTS. Si al audio personalizado le falta ese objeto, debe seguir omitiéndose,
        para no producir una línea de tiempo de subtítulos poco fiable.
        """
        task_id = "test-custom-audio-edge-no-submaker"
        task_dir = utils.task_dir(task_id)
        audio_file = os.path.join(task_dir, "custom-audio.mp3")
        Path(audio_file).write_bytes(b"fake audio")
        params = VideoParams(
            video_subject="custom audio",
            video_script="Hello world.",
            subtitle_enabled=True,
        )

        try:
            with (
                patch.object(
                    tm.config,
                    "app",
                    dict(tm.config.app, subtitle_provider="edge"),
                ),
                patch.object(tm.voice, "create_subtitle") as create_subtitle,
                patch.object(tm.subtitle, "create") as whisper_create,
            ):
                subtitle_path = tm.generate_subtitle(
                    task_id=task_id,
                    params=params,
                    video_script="Hello world.",
                    sub_maker=None,
                    audio_file=audio_file,
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertEqual(subtitle_path, "")
        create_subtitle.assert_not_called()
        whisper_create.assert_not_called()

    def test_generate_subtitle_does_not_fallback_to_whisper_when_edge_fails(self):
        """
        Si Edge no genera el archivo de subtítulos, debe conservarse el resultado sin
        subtítulos, sin descargar automáticamente el modelo de Whisper.

        Este caso puede darse cuando la línea de tiempo del TTS no se puede hacer
        coincidir con el texto original. Un respaldo automático haría que usuarios que
        no eligieron Whisper descargaran sin querer un modelo de varios GB, así que hay
        que verificar que Whisper no se llama en absoluto.
        """
        task_id = "test-edge-subtitle-without-output"
        task_dir = utils.task_dir(task_id)
        params = VideoParams(
            video_subject="edge subtitle",
            video_script="Hello world.",
            subtitle_enabled=True,
        )
        sub_maker = object()

        try:
            with (
                patch.object(
                    tm.config,
                    "app",
                    dict(tm.config.app, subtitle_provider="edge"),
                ),
                patch.object(tm.voice, "create_subtitle") as create_subtitle,
                patch.object(tm.subtitle, "create") as whisper_create,
                patch.object(tm.subtitle, "correct") as whisper_correct,
            ):
                subtitle_path = tm.generate_subtitle(
                    task_id=task_id,
                    params=params,
                    video_script="Hello world.",
                    sub_maker=sub_maker,
                    audio_file=os.path.join(task_dir, "audio.mp3"),
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertEqual(subtitle_path, "")
        create_subtitle.assert_called_once()
        whisper_create.assert_not_called()
        whisper_correct.assert_not_called()

    def test_start_returns_each_intermediate_result(self):
        """
        Los modos script, terms, audio, subtitle y materials de la API comparten el
        mismo flujo de tareas. Cada punto de parada anticipada debe devolver su producto
        correspondiente sin ejecutar por error las fases posteriores.
        """
        expected_results = {
            "script": {"script": "generated script"},
            "terms": {
                "script": "generated script",
                "terms": ["coffee", "morning"],
            },
            "audio": {"audio_file": "audio.mp3", "audio_duration": 5},
            "subtitle": {"subtitle_path": "subtitle.srt"},
            "materials": {"materials": ["clip.mp4"]},
        }

        for stop_at, expected in expected_results.items():
            with self.subTest(stop_at=stop_at):
                params = VideoParams(video_subject="Coffee")
                with (
                    patch.object(
                        tm, "generate_script", return_value="generated script"
                    ),
                    patch.object(
                        tm,
                        "generate_terms",
                        return_value=["coffee", "morning"],
                    ),
                    patch.object(tm, "save_script_data"),
                    patch.object(
                        tm,
                        "generate_audio",
                        return_value=("audio.mp3", 5, object()),
                    ),
                    patch.object(
                        tm,
                        "generate_subtitle",
                        return_value="subtitle.srt",
                    ),
                    patch.object(
                        tm,
                        "get_video_materials",
                        return_value=["clip.mp4"],
                    ),
                    patch.object(tm, "generate_final_videos") as generate_final,
                    patch.object(tm.sm.state, "update_task"),
                ):
                    result = tm.start(
                        f"intermediate-{stop_at}", params, stop_at=stop_at
                    )

                self.assertEqual(result, expected)
                generate_final.assert_not_called()

    def test_start_forwards_trusted_server_file_flag_to_audio_stage(self):
        params = VideoParams(video_subject="CLI custom audio")

        with (
            patch.object(tm.utils, "check_ffmpeg_ready", return_value=True),
            patch.object(tm, "generate_script", return_value="generated script"),
            patch.object(tm, "generate_terms", return_value=["audio"]),
            patch.object(tm, "save_script_data"),
            patch.object(
                tm,
                "generate_audio",
                return_value=("audio.mp3", 5, None),
            ) as generate_audio,
            patch.object(tm.sm.state, "update_task"),
        ):
            result = tm.start(
                "trusted-cli-audio",
                params,
                stop_at="audio",
                allow_server_file_input=True,
            )

        self.assertEqual(result, {"audio_file": "audio.mp3", "audio_duration": 5})
        generate_audio.assert_called_once_with(
            "trusted-cli-audio",
            params,
            "generated script",
            voice_preview=None,
            allow_server_file_input=True,
        )

    def test_start_completes_video_without_cross_posting(self):
        """
        Una tarea completa debe terminar de forma estable aunque la publicación
        automática no esté configurada, escribiendo todos los productos intermedios en
        el estado final. Aquí también se cubre la conversión de compatibilidad del modo
        de montaje que la API puede pasar como cadena.
        """
        params = VideoParams(video_subject="Coffee")
        params.video_concat_mode = "sequential"

        with (
            patch.object(tm, "generate_script", return_value="generated script"),
            patch.object(tm, "generate_terms", return_value=["coffee"]),
            patch.object(tm, "save_script_data"),
            patch.object(
                tm,
                "generate_audio",
                return_value=("audio.mp3", 5, object()),
            ),
            patch.object(tm, "generate_subtitle", return_value="subtitle.srt"),
            patch.object(
                tm,
                "get_video_materials",
                return_value=["clip.mp4"],
            ),
            patch.object(
                tm,
                "generate_final_videos",
                return_value=(["final.mp4"], ["combined.mp4"], []),
            ),
            patch.object(
                tm.upload_post.upload_post_service,
                "is_configured",
                return_value=False,
            ),
            patch.object(tm.upload_post, "cross_post_video") as cross_post,
            patch.object(tm.sm.state, "update_task") as update_task,
        ):
            result = tm.start("complete-video", params)

        self.assertEqual(result["videos"], ["final.mp4"])
        self.assertEqual(result["combined_videos"], ["combined.mp4"])
        self.assertEqual(result["cross_post_results"], None)
        self.assertEqual(params.video_concat_mode, tm.VideoConcatMode.sequential)
        cross_post.assert_not_called()
        update_task.assert_called_with(
            "complete-video",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            **result,
        )

    def test_start_marks_pipeline_failures(self):
        """
        Si falta cualquiera de los productos clave (audio, material o vídeo final), la
        tarea debe pasar a estado de fallo; no se puede informar como completada una
        tarea incompleta. Los tres escenarios reutilizan los mismos mocks y solo cambian
        la fase que falla.
        """
        failure_cases = {
            "audio": (
                (None, None, None),
                ["clip.mp4"],
                (["final.mp4"], ["combined.mp4"], []),
            ),
            "materials": (
                ("audio.mp3", 5, object()),
                None,
                (["final.mp4"], ["combined.mp4"], []),
            ),
            "video": (("audio.mp3", 5, object()), ["clip.mp4"], ([], [], [])),
        }

        for stage, failure_results in failure_cases.items():
            with self.subTest(stage=stage):
                audio_result, materials_result, videos_result = failure_results
                params = VideoParams(video_subject="Coffee")
                state = MemoryState()
                with (
                    patch.object(
                        tm, "generate_script", return_value="generated script"
                    ),
                    patch.object(tm, "generate_terms", return_value=["coffee"]),
                    patch.object(tm, "save_script_data"),
                    patch.object(tm, "generate_audio", return_value=audio_result),
                    patch.object(tm, "generate_subtitle", return_value="subtitle.srt"),
                    patch.object(
                        tm,
                        "get_video_materials",
                        return_value=materials_result,
                    ),
                    patch.object(
                        tm,
                        "generate_final_videos",
                        return_value=videos_result,
                    ),
                    patch.object(tm.sm, "state", state),
                ):
                    result = tm.start(f"failed-{stage}", params)

                failed_task = state.get_task(f"failed-{stage}")
                self.assertEqual(result, failed_task)
                self.assertEqual(failed_task["state"], tm.const.TASK_STATE_FAILED)
                self.assertEqual(failed_task["failed_stage"], stage)
                self.assertTrue(failed_task["error"])

    def test_start_records_unexpected_pipeline_exception(self):
        """Una excepción inesperada también debe terminar la tarea y exponer a la API el
        tipo y el mensaje originales de la excepción.
        """
        params = VideoParams(video_subject="Coffee")
        state = MemoryState()

        with (
            patch.object(
                tm,
                "generate_script",
                side_effect=RuntimeError("provider connection reset"),
            ),
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("unexpected-failure", params)

        failed_task = state.get_task("unexpected-failure")
        self.assertEqual(result, failed_task)
        self.assertEqual(failed_task["state"], tm.const.TASK_STATE_FAILED)
        self.assertEqual(failed_task["failed_stage"], "pipeline")
        self.assertEqual(
            failed_task["error"],
            "RuntimeError: provider connection reset",
        )

    def test_start_generates_youtube_metadata_for_each_cross_post(self):
        """
        Al publicar automáticamente en YouTube los metadatos solo se generan una vez,
        pero los mismos campos deben pasarse a cada vídeo final, y el resultado de la
        tarea debe conservar el resultado independiente de cada subida, correcta o
        fallida.
        """
        params = VideoParams(
            video_subject="Coffee",
            video_language="en",
        )
        metadata = {
            "title": "Morning Coffee",
            "caption": "A better morning.",
            "hashtags": ["coffee", "shorts"],
        }
        service = tm.upload_post.upload_post_service
        state = MemoryState()

        def run_immediately(function, *args):
            future = Future()
            try:
                function(*args)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(None)
            return future

        with (
            patch.object(tm, "generate_script", return_value="generated script"),
            patch.object(tm, "generate_terms", return_value=["coffee"]),
            patch.object(tm, "save_script_data"),
            patch.object(
                tm,
                "generate_audio",
                return_value=("audio.mp3", 5, object()),
            ),
            patch.object(tm, "generate_subtitle", return_value="subtitle.srt"),
            patch.object(
                tm,
                "get_video_materials",
                return_value=["clip.mp4"],
            ),
            patch.object(
                tm,
                "generate_final_videos",
                return_value=(
                    ["final-1.mp4", "final-2.mp4"],
                    ["combined-1.mp4", "combined-2.mp4"],
                    [],
                ),
            ),
            patch.object(service, "is_configured", return_value=True),
            patch.object(type(service), "auto_upload", new_callable=PropertyMock, return_value=True),
            patch.object(type(service), "platforms", new_callable=PropertyMock, return_value=["youtube"]),
            patch.object(type(service), "youtube_privacy_status", new_callable=PropertyMock, return_value="unlisted"),
            patch.object(type(service), "youtube_made_for_kids", new_callable=PropertyMock, return_value=True),
            patch.object(
                tm.llm,
                "generate_social_metadata",
                return_value=metadata,
            ) as generate_metadata,
            patch.object(
                tm.upload_post,
                "cross_post_video",
                side_effect=[
                    {"success": True},
                    {"success": False, "error": "upload failed"},
                ],
            ) as cross_post,
            patch.object(tm.sm, "state", state),
            patch.object(
                tm._cross_post_executor,
                "submit",
                side_effect=run_immediately,
            ),
        ):
            result = tm.start("youtube-cross-post", params)

        generate_metadata.assert_called_once_with(
            video_subject="Coffee",
            video_script="generated script",
            language="en",
            platform="youtube_shorts",
        )
        expected_extra = {
            "youtube_title": "Morning Coffee",
            "youtube_description": "A better morning.",
            "tags": ["coffee", "shorts"],
            "privacyStatus": "unlisted",
            "selfDeclaredMadeForKids": True,
            "containsSyntheticMedia": True,
        }
        self.assertEqual(cross_post.call_count, 2)
        for call in cross_post.call_args_list:
            self.assertEqual(call.kwargs["youtube_extra"], expected_extra)
            self.assertEqual(call.kwargs["platforms"], ["youtube"])

        # start() devuelve una instantánea estable del momento en que termina el vídeo;
        # el resultado de la publicación en segundo plano se obtiene consultando la
        # tarea.
        self.assertEqual(result["cross_post_state"], tm.const.CROSS_POST_STATE_PENDING)
        self.assertIsNone(result["cross_post_results"])
        published_task = state.get_task("youtube-cross-post")
        self.assertEqual(published_task["state"], tm.const.TASK_STATE_COMPLETE)
        self.assertEqual(
            published_task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED
        )
        self.assertEqual(
            published_task["cross_post_results"],
            [
                {"success": True},
                {"success": False, "error": "upload failed"},
            ],
        )
        self.assertEqual(published_task["cross_post_error"], "upload failed")

    def test_start_returns_before_cross_post_worker_runs(self):
        """Al terminar la tarea de vídeo solo se envía el trabajo de publicación; no se
        puede subir de forma síncrona en el hilo de generación.
        """
        params = VideoParams(video_subject="Coffee")
        service = tm.upload_post.upload_post_service
        state = MemoryState()
        submitted = []

        def capture_submission(function, *args):
            submitted.append((function, args))
            return MagicMock(spec=Future)

        with (
            patch.object(tm, "generate_script", return_value="generated script"),
            patch.object(tm, "generate_terms", return_value=["coffee"]),
            patch.object(tm, "save_script_data"),
            patch.object(
                tm,
                "generate_audio",
                return_value=("audio.mp3", 5, object()),
            ),
            patch.object(tm, "generate_subtitle", return_value="subtitle.srt"),
            patch.object(tm, "get_video_materials", return_value=["clip.mp4"]),
            patch.object(
                tm,
                "generate_final_videos",
                return_value=(["final.mp4"], ["combined.mp4"], []),
            ),
            patch.object(service, "is_configured", return_value=True),
            patch.object(type(service), "auto_upload", new_callable=PropertyMock, return_value=True),
            patch.object(type(service), "platforms", new_callable=PropertyMock, return_value=["tiktok"]),
            patch.object(type(service), "youtube_privacy_status", new_callable=PropertyMock, return_value="private"),
            patch.object(tm.upload_post, "cross_post_video") as cross_post,
            patch.object(tm.sm, "state", state),
            patch.object(
                tm._cross_post_executor,
                "submit",
                side_effect=capture_submission,
            ) as submit,
        ):
            result = tm.start("deferred-cross-post", params)

        submit.assert_called_once()
        cross_post.assert_not_called()
        self.assertEqual(result["videos"], ["final.mp4"])
        self.assertEqual(result["cross_post_state"], tm.const.CROSS_POST_STATE_PENDING)
        completed_task = state.get_task("deferred-cross-post")
        self.assertEqual(completed_task["state"], tm.const.TASK_STATE_COMPLETE)
        self.assertEqual(completed_task["progress"], 100)

        worker, worker_args = submitted[0]
        with (
            patch.object(tm.sm, "state", state),
            patch.object(
                tm.upload_post,
                "cross_post_video",
                return_value={"success": True, "request_id": "upload-1"},
            ),
        ):
            worker(*worker_args)

        published_task = state.get_task("deferred-cross-post")
        self.assertEqual(published_task["videos"], ["final.mp4"])
        self.assertEqual(
            published_task["cross_post_state"], tm.const.CROSS_POST_STATE_COMPLETE
        )

    def test_cross_post_worker_failure_does_not_change_video_completion(self):
        """Una excepción del hilo de publicación solo puede actualizar el estado de
        publicación, sin romper el resultado de vídeo ya completado.
        """
        state = MemoryState()
        state.update_task(
            "cross-post-worker-failure",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )

        with (
            patch.object(tm.sm, "state", state),
            patch.object(
                tm.llm,
                "generate_social_metadata",
                side_effect=RuntimeError("metadata provider unavailable"),
            ),
            patch.object(tm.upload_post, "cross_post_video") as cross_post,
        ):
            tm._run_cross_post(
                "cross-post-worker-failure",
                ("final.mp4",),
                "Coffee",
                "A short coffee story.",
                "en",
                ("youtube",),
                "private",
            )

        cross_post.assert_not_called()
        task = state.get_task("cross-post-worker-failure")
        self.assertEqual(task["state"], tm.const.TASK_STATE_COMPLETE)
        self.assertEqual(task["videos"], ["final.mp4"])
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("metadata provider unavailable", task["cross_post_error"])

    def test_background_upload_request_id_is_saved_while_polling(self):
        """Un worker interrumpido debe dejar en el estado de la tarea el ID de la subida
        remota.
        """
        state = MemoryState()
        state.update_task(
            "background-upload",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )
        observed = {}

        def upload_with_background_start(**kwargs):
            kwargs["on_background_start"]("request-42")
            observed.update(state.get_task("background-upload"))
            return {"success": True, "request_id": "request-42"}

        with (
            patch.object(tm.sm, "state", state),
            patch.object(tm.llm, "generate_social_metadata", return_value={}),
            patch.object(
                tm.upload_post,
                "cross_post_video",
                side_effect=upload_with_background_start,
            ),
        ):
            tm._run_cross_post(
                "background-upload",
                ("final.mp4",),
                "Coffee",
                "Coffee script",
                "en",
                ("tiktok",),
                "public",
            )

        self.assertEqual(
            observed["cross_post_state"], tm.const.CROSS_POST_STATE_PROCESSING
        )
        self.assertEqual(
            observed["cross_post_results"][0]["request_id"], "request-42"
        )
        finished = state.get_task("background-upload")
        self.assertEqual(finished["cross_post_state"], tm.const.CROSS_POST_STATE_COMPLETE)
        self.assertEqual(finished["cross_post_results"], [{"success": True, "request_id": "request-42"}])

    def test_background_upload_request_id_survives_worker_error(self):
        """Una excepción durante el sondeo no debe borrar el único identificador de
        recuperación remota.
        """
        state = MemoryState()
        state.update_task(
            "background-upload-error",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )

        def interrupted_upload(**kwargs):
            kwargs["on_background_start"]("request-lost")
            raise RuntimeError("worker stopped during status polling")

        with (
            patch.object(tm.sm, "state", state),
            patch.object(tm.llm, "generate_social_metadata", return_value={}),
            patch.object(
                tm.upload_post,
                "cross_post_video",
                side_effect=interrupted_upload,
            ),
        ):
            tm._run_cross_post(
                "background-upload-error",
                ("final.mp4",),
                "Coffee",
                "Coffee script",
                "en",
                ("tiktok",),
                "public",
            )

        failed = state.get_task("background-upload-error")
        self.assertEqual(failed["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertEqual(
            failed["cross_post_results"][0]["request_id"], "request-lost"
        )

    def test_start_returns_cross_post_scheduling_failure(self):
        """Un fallo de planificación síncrono debe reflejarse a la vez en el estado de la
        tarea y en la instantánea que devuelve start().
        """
        params = VideoParams(video_subject="Coffee")
        service = tm.upload_post.upload_post_service
        state = MemoryState()

        with (
            patch.object(tm, "generate_script", return_value="generated script"),
            patch.object(tm, "generate_terms", return_value=["coffee"]),
            patch.object(tm, "save_script_data"),
            patch.object(
                tm,
                "generate_audio",
                return_value=("audio.mp3", 5, object()),
            ),
            patch.object(tm, "generate_subtitle", return_value="subtitle.srt"),
            patch.object(tm, "get_video_materials", return_value=["clip.mp4"]),
            patch.object(
                tm,
                "generate_final_videos",
                return_value=(["final.mp4"], ["combined.mp4"], []),
            ),
            patch.object(service, "is_configured", return_value=True),
            patch.object(type(service), "auto_upload", new_callable=PropertyMock, return_value=True),
            patch.object(type(service), "platforms", new_callable=PropertyMock, return_value=["tiktok"]),
            patch.object(type(service), "youtube_privacy_status", new_callable=PropertyMock, return_value="private"),
            patch.object(tm.sm, "state", state),
            patch.object(tm._cross_post_slots, "acquire", return_value=False),
            patch.object(tm._cross_post_executor, "submit") as submit,
        ):
            result = tm.start("cross-post-queue-full-result", params)

        submit.assert_not_called()
        self.assertEqual(result["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("cola de publicación multiplataforma está llena", result["cross_post_error"])
        persisted_task = state.get_task("cross-post-queue-full-result")
        self.assertEqual(
            persisted_task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED
        )
        self.assertEqual(
            persisted_task["cross_post_error"],
            result["cross_post_error"],
        )

    def test_cross_post_schedule_failure_is_recorded_separately(self):
        """Cuando el pool de hilos rechaza tareas nuevas, hay que conservar el vídeo final
        y ofrecer un error de publicación consultable.
        """
        state = MemoryState()
        slots = MagicMock()
        slots.acquire.return_value = True
        state.update_task(
            "cross-post-schedule-failure",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )

        with (
            patch.object(tm.sm, "state", state),
            patch.object(tm, "_cross_post_slots", slots),
            patch.object(
                tm._cross_post_executor,
                "submit",
                side_effect=RuntimeError("executor is shutting down"),
            ),
        ):
            scheduling_error = tm._schedule_cross_post(
                task_id="cross-post-schedule-failure",
                video_paths=["final.mp4"],
                params=VideoParams(video_subject="Coffee"),
                video_script="A short coffee story.",
                platforms=["tiktok"],
                youtube_privacy_status="private",
            )

        slots.release.assert_called_once_with()
        self.assertIn("executor is shutting down", scheduling_error)
        task = state.get_task("cross-post-schedule-failure")
        self.assertEqual(task["state"], tm.const.TASK_STATE_COMPLETE)
        self.assertEqual(task["videos"], ["final.mp4"])
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("executor is shutting down", task["cross_post_error"])

    def test_cross_post_worker_always_releases_queue_slot(self):
        """Aunque el trabajo de publicación salga con una excepción, debe devolver la
        capacidad, para que las publicaciones posteriores no se rechacen para siempre.
        """
        slots = MagicMock()
        state = MemoryState()
        state.update_task(
            "task-id",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )

        with (
            patch.object(tm, "_cross_post_slots", slots),
            patch.object(tm.sm, "state", state),
            patch.object(
                tm,
                "_run_cross_post",
                side_effect=RuntimeError("worker crashed"),
            ),
        ):
            tm._run_cross_post_with_slot("task-id")

        slots.release.assert_called_once_with()
        task = state.get_task("task-id")
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("worker crashed", task["cross_post_error"])

    def test_cross_post_state_backend_failure_is_logged_and_skips_upload(self):
        """Si falla la primera escritura de estado no se puede salir en silencio ni seguir
        gastando cuota de publicación.
        """
        state = MagicMock()
        state.patch_task.side_effect = RuntimeError("redis unavailable")

        with (
            patch.object(tm.sm, "state", state),
            patch.object(tm.upload_post, "cross_post_video") as cross_post,
            patch.object(tm.logger, "exception") as log_exception,
            patch.object(tm.time, "sleep") as sleep,
        ):
            tm._run_cross_post(
                "state-backend-failure",
                ("final.mp4",),
                "Coffee",
                "A short coffee story.",
                "en",
                ("tiktok",),
                "private",
            )

        cross_post.assert_not_called()
        self.assertEqual(state.patch_task.call_count, 6)
        self.assertEqual(sleep.call_count, 4)
        self.assertEqual(log_exception.call_count, 2)
        self.assertTrue(
            all(
                "redis unavailable" in call.args[0]
                for call in log_exception.call_args_list
            )
        )

    def test_cross_post_state_update_retries_transient_backend_failure(self):
        """Tras un fallo breve y único del backend de estado debe seguir publicando y
        acabar guardando el estado completado.
        """

        class FlakyMemoryState(MemoryState):
            def __init__(self):
                super().__init__()
                self.patch_calls = 0

            def patch_task(self, task_id, **kwargs):
                self.patch_calls += 1
                if self.patch_calls == 1:
                    raise RuntimeError("temporary redis outage")
                return super().patch_task(task_id, **kwargs)

        state = FlakyMemoryState()
        state.update_task(
            "transient-state-failure",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )

        with (
            patch.object(tm.sm, "state", state),
            patch.object(
                tm.upload_post,
                "cross_post_video",
                return_value={"success": True, "request_id": "upload-1"},
            ) as cross_post,
            patch.object(tm.time, "sleep") as sleep,
        ):
            tm._run_cross_post(
                "transient-state-failure",
                ("final.mp4",),
                "Coffee",
                "A short coffee story.",
                "en",
                ("tiktok",),
                "private",
            )

        sleep.assert_called_once_with(tm._CROSS_POST_STATE_RETRY_DELAY_SECONDS)
        cross_post.assert_called_once()
        task = state.get_task("transient-state-failure")
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_COMPLETE)
        self.assertIsNone(task["cross_post_error"])

    def test_cross_post_generates_caption_for_non_youtube_platforms(self):
        """
        La publicación en TikTok/Instagram también debe generar una vez el texto para
        redes sociales y usar el caption como título de publicación compartido por todos
        los vídeos finales, en lugar de enviar directamente el tema original.
        """
        metadata = {
            "title": "Coffee Hook",
            "caption": "Watch this coffee ritual.",
            "hashtags": ["#coffee"],
        }
        state = MemoryState()
        cases = {
            "tiktok-first": (("tiktok", "instagram"), "tiktok"),
            "instagram-first": (("instagram", "tiktok"), "instagram_reels"),
        }

        for case_name, (platforms, expected_platform) in cases.items():
            with self.subTest(case=case_name):
                task_id = f"caption-{case_name}"
                state.update_task(
                    task_id,
                    state=tm.const.TASK_STATE_COMPLETE,
                    progress=100,
                    videos=["final.mp4"],
                    cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
                )
                with (
                    patch.object(tm.sm, "state", state),
                    patch.object(
                        tm.llm,
                        "generate_social_metadata",
                        return_value=metadata,
                    ) as generate_metadata,
                    patch.object(
                        tm.upload_post,
                        "cross_post_video",
                        return_value={"success": True},
                    ) as cross_post,
                ):
                    tm._run_cross_post(
                        task_id,
                        ("final.mp4",),
                        "Coffee",
                        "A short coffee story.",
                        "en",
                        platforms,
                        "private",
                    )

                generate_metadata.assert_called_once_with(
                    video_subject="Coffee",
                    video_script="A short coffee story.",
                    language="en",
                    platform=expected_platform,
                )
                cross_post.assert_called_once()
                call = cross_post.call_args
                self.assertEqual(call.kwargs["title"], "Watch this coffee ritual.")
                self.assertEqual(call.kwargs["platforms"], list(platforms))
                self.assertIsNone(call.kwargs["youtube_extra"])
                task = state.get_task(task_id)
                self.assertEqual(
                    task["cross_post_state"], tm.const.CROSS_POST_STATE_COMPLETE
                )

    def test_cross_post_shares_metadata_between_youtube_fields_and_title(self):
        """Los campos propios de YouTube y el título de publicación compartido deben venir
        de la misma llamada de metadatos.
        """
        metadata = {
            "title": "Morning Coffee",
            "caption": "A better morning.",
            "hashtags": ["#coffee", "#shorts"],
        }
        state = MemoryState()
        state.update_task(
            "shared-youtube-metadata",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final-1.mp4", "final-2.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )

        with (
            patch.object(tm.sm, "state", state),
            patch.object(
                tm.llm,
                "generate_social_metadata",
                return_value=metadata,
            ) as generate_metadata,
            patch.object(
                tm.upload_post,
                "cross_post_video",
                return_value={"success": True},
            ) as cross_post,
        ):
            tm._run_cross_post(
                "shared-youtube-metadata",
                ("final-1.mp4", "final-2.mp4"),
                "Coffee",
                "A short coffee story.",
                "en",
                ("youtube",),
                "unlisted",
            )

        generate_metadata.assert_called_once_with(
            video_subject="Coffee",
            video_script="A short coffee story.",
            language="en",
            platform="youtube_shorts",
        )
        expected_extra = {
            "youtube_title": "Morning Coffee",
            "youtube_description": "A better morning.",
            "tags": ["#coffee", "#shorts"],
            "privacyStatus": "unlisted",
            "selfDeclaredMadeForKids": False,
            "containsSyntheticMedia": True,
        }
        self.assertEqual(cross_post.call_count, 2)
        for call in cross_post.call_args_list:
            self.assertEqual(call.kwargs["title"], "A better morning.")
            self.assertEqual(call.kwargs["youtube_extra"], expected_extra)

    def test_cross_post_empty_metadata_degrades_to_fallback_title(self):
        """Si los metadatos faltan o están vacíos se retrocede nivel a nivel, conservando
        al final el antiguo título genérico de respaldo.
        """
        state = MemoryState()
        cases = {
            "legacy-string": ({}, "", "¡Mira este vídeo! #shorts #viral"),
            "title-over-subject": (
                {"title": "Fallback Title", "caption": ""},
                "Coffee",
                "Fallback Title",
            ),
        }

        for case_name, (metadata, subject, expected_title) in cases.items():
            with self.subTest(case=case_name):
                task_id = f"fallback-title-{case_name}"
                state.update_task(
                    task_id,
                    state=tm.const.TASK_STATE_COMPLETE,
                    progress=100,
                    videos=["final.mp4"],
                    cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
                )
                with (
                    patch.object(tm.sm, "state", state),
                    patch.object(
                        tm.llm,
                        "generate_social_metadata",
                        return_value=metadata,
                    ) as generate_metadata,
                    patch.object(
                        tm.upload_post,
                        "cross_post_video",
                        return_value={"success": True},
                    ) as cross_post,
                ):
                    tm._run_cross_post(
                        task_id,
                        ("final.mp4",),
                        subject,
                        "",
                        "",
                        ("tiktok",),
                        "private",
                    )

                generate_metadata.assert_called_once()
                cross_post.assert_called_once()
                self.assertEqual(cross_post.call_args.kwargs["title"], expected_title)

    def test_recover_interrupted_cross_posts_preserves_active_future(self):
        """La recuperación al arrancar solo trata estados residuales; no puede afectar por
        error a las tareas de publicación que aún tiene el proceso actual.
        """
        state = MemoryState()
        for task_id in (
            "stale-pending",
            "active-processing",
            "inactive-current-owner",
            "remote-processing",
            "already-complete",
        ):
            cross_post_state = {
                "stale-pending": tm.const.CROSS_POST_STATE_PENDING,
                "active-processing": tm.const.CROSS_POST_STATE_PROCESSING,
                "inactive-current-owner": tm.const.CROSS_POST_STATE_PROCESSING,
                "remote-processing": tm.const.CROSS_POST_STATE_PROCESSING,
                "already-complete": tm.const.CROSS_POST_STATE_COMPLETE,
            }[task_id]
            state.update_task(
                task_id,
                state=tm.const.TASK_STATE_COMPLETE,
                progress=100,
                videos=["final.mp4"],
                cross_post_state=cross_post_state,
                cross_post_owner=(
                    "another-host:123:remote"
                    if task_id == "remote-processing"
                    else (
                        tm._cross_post_process_owner
                        if task_id == "inactive-current-owner"
                        else None
                    )
                ),
            )

        active_future = Future()
        tm._register_cross_post_future("active-processing", active_future)
        with patch.object(tm.sm, "state", state):
            recovered = tm.recover_interrupted_cross_posts(page_size=1)

        self.assertEqual(recovered, 2)
        stale_task = state.get_task("stale-pending")
        self.assertEqual(
            stale_task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED
        )
        self.assertEqual(
            stale_task["cross_post_error"], tm._INTERRUPTED_CROSS_POST_ERROR
        )
        self.assertEqual(
            state.get_task("active-processing")["cross_post_state"],
            tm.const.CROSS_POST_STATE_PROCESSING,
        )
        self.assertEqual(
            state.get_task("inactive-current-owner")["cross_post_state"],
            tm.const.CROSS_POST_STATE_FAILED,
        )
        self.assertEqual(
            state.get_task("remote-processing")["cross_post_state"],
            tm.const.CROSS_POST_STATE_PROCESSING,
        )
        self.assertEqual(
            state.get_task("already-complete")["cross_post_state"],
            tm.const.CROSS_POST_STATE_COMPLETE,
        )
        active_future.set_result(None)

    def test_recover_interrupted_cross_posts_scans_ids_only_once(self):
        """La recuperación al arrancar no puede depender del orden de paginación de varios
        escaneos independientes.
        """
        state = MemoryState()
        for task_id, cross_post_state in (
            ("stale-pending", tm.const.CROSS_POST_STATE_PENDING),
            ("stale-processing", tm.const.CROSS_POST_STATE_PROCESSING),
            ("already-complete", tm.const.CROSS_POST_STATE_COMPLETE),
        ):
            state.update_task(
                task_id,
                state=tm.const.TASK_STATE_COMPLETE,
                progress=100,
                videos=["final.mp4"],
                cross_post_state=cross_post_state,
            )

        with (
            patch.object(tm.sm, "state", state),
            patch.object(state, "list_task_ids", wraps=state.list_task_ids) as list_ids,
            patch.object(
                state, "get_all_tasks", side_effect=AssertionError("pagination used")
            ) as get_all_tasks,
        ):
            recovered = tm.recover_interrupted_cross_posts(page_size=1)

        self.assertEqual(recovered, 2)
        list_ids.assert_called_once_with(scan_count=1)
        get_all_tasks.assert_not_called()
        for task_id in ("stale-pending", "stale-processing"):
            task = state.get_task(task_id)
            self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
            self.assertEqual(task["videos"], ["final.mp4"])
        self.assertEqual(
            state.get_task("already-complete")["cross_post_state"],
            tm.const.CROSS_POST_STATE_COMPLETE,
        )

    def test_cross_post_owner_uses_future_registry_for_current_process(self):
        """Cuando el proceso actual no tiene Futures activos, tanto los owners nuevos como
        los antiguos con el mismo PID deben considerarse interrumpidos.
        """
        stale_owner = f"{tm.socket.gethostname()}:{tm.os.getpid()}:old-instance"

        self.assertFalse(tm._is_cross_post_owner_alive(stale_owner))
        self.assertFalse(tm._is_cross_post_owner_alive(tm._cross_post_process_owner))

    def test_cross_post_owner_detection_handles_process_boundaries(self):
        """La detección del propietario debe cubrir registros antiguos, otros hosts y los
        casos límite de procesos anómalos en este equipo.
        """
        hostname = tm.socket.gethostname()

        self.assertFalse(tm._is_cross_post_owner_alive(None))
        self.assertFalse(tm._is_cross_post_owner_alive("invalid-owner"))
        self.assertTrue(tm._is_cross_post_owner_alive("another-host:123:instance"))

        with (
            patch.object(tm.os, "name", "posix"),
            patch.object(tm.os, "kill", side_effect=ProcessLookupError),
        ):
            self.assertFalse(
                tm._is_cross_post_owner_alive(f"{hostname}:987654:dead-instance")
            )
        with (
            patch.object(tm.os, "name", "posix"),
            patch.object(tm.os, "kill", side_effect=PermissionError),
        ):
            self.assertTrue(
                tm._is_cross_post_owner_alive(f"{hostname}:987654:restricted")
            )
        with (
            patch.object(tm.os, "name", "posix"),
            patch.object(tm.os, "kill", side_effect=OSError("inspection failed")),
            patch.object(tm.logger, "warning") as log_warning,
        ):
            self.assertTrue(tm._is_cross_post_owner_alive(f"{hostname}:987654:unknown"))
        self.assertIn("inspection failed", log_warning.call_args.args[0])

        with (
            patch.object(tm.os, "name", "nt"),
            patch.object(tm, "_is_windows_process_alive", return_value=True) as probe,
        ):
            self.assertTrue(tm._is_cross_post_owner_alive(f"{hostname}:987654:windows"))
        probe.assert_called_once_with(987654)

    @unittest.skipUnless(os.name == "nt", "Windows process API test")
    def test_windows_process_probe_is_read_only_and_detects_liveness(self):
        """La CI de Windows debe verificar de verdad la detección de procesos de solo
        lectura, sin volver a os.kill.
        """
        self.assertTrue(tm._is_windows_process_alive(os.getpid()))
        self.assertFalse(tm._is_windows_process_alive(2_147_483_647))

    def test_cross_post_terminal_check_converts_active_state_to_failure(self):
        """Si el worker ha terminado pero el estado sigue activo, el callback final debe
        escribir el estado final de fallo.
        """
        state = MemoryState()
        state.update_task(
            "unfinished-cross-post",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PROCESSING,
        )

        with patch.object(tm.sm, "state", state):
            tm._ensure_cross_post_terminal_state("unfinished-cross-post")

        task = state.get_task("unfinished-cross-post")
        self.assertEqual(task["videos"], ["final.mp4"])
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("sin guardar un estado final", task["cross_post_error"])

    def test_cross_post_recovery_reports_state_backend_failure(self):
        """Si la recuperación al arrancar no puede leer el estado debe devolver None, para
        que un rerun posterior de la WebUI lo reintente.
        """
        state = MagicMock()
        state.list_task_ids.side_effect = RuntimeError("redis unavailable")

        with (
            patch.object(tm.sm, "state", state),
            patch.object(tm.logger, "exception") as log_exception,
        ):
            recovered = tm.recover_interrupted_cross_posts()

        self.assertIsNone(recovered)
        self.assertIn("redis unavailable", log_exception.call_args.args[0])

    def test_cancelled_cross_post_future_releases_slot_and_records_failure(self):
        """Cuando se cancela un Future en cola también hay que liberar la capacidad y
        escribir el estado final de fallo.
        """
        state = MemoryState()
        state.update_task(
            "cancelled-cross-post",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )
        slots = MagicMock()
        future = Future()
        tm._register_cross_post_future("cancelled-cross-post", future)
        self.assertTrue(future.cancel())

        with (
            patch.object(tm.sm, "state", state),
            patch.object(tm, "_cross_post_slots", slots),
        ):
            tm._finalize_cross_post_future("cancelled-cross-post", future)

        slots.release.assert_called_once_with()
        self.assertFalse(tm._is_cross_post_active_in_process("cancelled-cross-post"))
        task = state.get_task("cancelled-cross-post")
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("canceló", task["cross_post_error"])

    @unittest.skipUnless(
        os.getenv("MPT_TEST_REDIS_HOST"),
        "MPT_TEST_REDIS_HOST not set",
    )
    def test_real_redis_recovers_interrupted_cross_post_state(self):
        """Un escaneo en varios lotes de un Redis real debe recuperar todos los estados de
        publicación residuales y conservar los vídeos.
        """
        state = RedisState(
            host=os.environ["MPT_TEST_REDIS_HOST"],
            port=int(os.getenv("MPT_TEST_REDIS_PORT", "6379")),
            db=int(os.getenv("MPT_TEST_REDIS_DB", "15")),
        )
        task_ids = [f"ci-cross-post-recovery-{uuid4()}" for _ in range(3)]
        for task_id, cross_post_state in zip(
            task_ids,
            (
                tm.const.CROSS_POST_STATE_PENDING,
                tm.const.CROSS_POST_STATE_PROCESSING,
                tm.const.CROSS_POST_STATE_COMPLETE,
            ),
        ):
            state.update_task(
                task_id,
                state=tm.const.TASK_STATE_COMPLETE,
                progress=100,
                videos=["final.mp4"],
                cross_post_state=cross_post_state,
                cross_post_owner="",
            )

        try:
            with patch.object(tm.sm, "state", state):
                recovered = tm.recover_interrupted_cross_posts(page_size=1)

            self.assertGreaterEqual(recovered, 2)
            for task_id in task_ids[:2]:
                task = state.get_task(task_id)
                self.assertEqual(task["videos"], ["final.mp4"])
                self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
                self.assertEqual(
                    task["cross_post_error"], tm._INTERRUPTED_CROSS_POST_ERROR
                )
            self.assertEqual(
                state.get_task(task_ids[2])["cross_post_state"],
                tm.const.CROSS_POST_STATE_COMPLETE,
            )
        finally:
            for task_id in task_ids:
                state.delete_task(task_id)

    def test_cross_post_future_exception_is_observed(self):
        """Las excepciones que lanza el propio pool de hilos deben llegar al log y no
        quedarse en un Future que nadie lee.
        """
        future = Future()
        future.set_exception(RuntimeError("executor worker failed"))

        with patch.object(tm.logger, "error") as log_error:
            tm._finalize_cross_post_future("future-failure", future)

        log_error.assert_called_once()
        self.assertIn("executor worker failed", log_error.call_args.args[0])

    def test_cross_post_queue_full_rejects_only_publishing(self):
        """Con la cola de publicación llena hay que conservar el vídeo final y no seguir
        enviando tareas al pool de hilos.
        """
        state = MemoryState()
        state.update_task(
            "cross-post-queue-full",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )

        with (
            patch.object(tm.sm, "state", state),
            patch.object(
                tm._cross_post_slots,
                "acquire",
                return_value=False,
            ),
            patch.object(tm._cross_post_executor, "submit") as submit,
        ):
            scheduling_error = tm._schedule_cross_post(
                task_id="cross-post-queue-full",
                video_paths=["final.mp4"],
                params=VideoParams(video_subject="Coffee"),
                video_script="A short coffee story.",
                platforms=["tiktok"],
                youtube_privacy_status="private",
            )

        submit.assert_not_called()
        self.assertIn("cola de publicación multiplataforma está llena", scheduling_error)
        task = state.get_task("cross-post-queue-full")
        self.assertEqual(task["state"], tm.const.TASK_STATE_COMPLETE)
        self.assertEqual(task["videos"], ["final.mp4"])
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("cola de publicación multiplataforma está llena", task["cross_post_error"])

    @unittest.skipUnless(
        RUN_INTEGRATION_TESTS,
        "MPT_RUN_INTEGRATION_TESTS not set",
    )
    def test_task_local_materials(self):
        task_id = "00000000-0000-0000-0000-000000000000"
        video_materials = []
        for i in range(1, 4):
            video_materials.append(
                MaterialInfo(
                    provider="local",
                    url=os.path.join(resources_dir, f"{i}.png"),
                    duration=0,
                )
            )

        params = VideoParams(
            video_subject="金钱的作用",
            video_script="金钱不仅是交换媒介，更是社会资源的分配工具。它能满足基本生存需求，如食物和住房，也能提供教育、医疗等提升生活品质的机会。拥有足够的金钱意味着更多选择权，比如职业自由或创业可能。但金钱的作用也有边界，它无法直接购买幸福、健康或真诚的人际关系。过度追逐财富可能导致价值观扭曲，忽视精神层面的需求。理想的状态是理性看待金钱，将其作为实现目标的工具而非终极目的。",
            video_terms="money importance, wealth and society, financial freedom, money and happiness, role of money",
            video_aspect="9:16",
            video_concat_mode="random",
            video_transition_mode="None",
            video_clip_duration=3,
            video_count=1,
            video_source="local",
            video_materials=video_materials,
            video_language="",
            voice_name="zh-CN-XiaoxiaoNeural-Female",
            voice_volume=1.0,
            voice_rate=1.0,
            bgm_type="random",
            bgm_file="",
            bgm_volume=0.2,
            subtitle_enabled=True,
            subtitle_position="bottom",
            custom_position=70.0,
            font_name="MicrosoftYaHeiBold.ttc",
            text_fore_color="#FFFFFF",
            text_background_color=True,
            font_size=60,
            stroke_color="#000000",
            stroke_width=1.5,
            n_threads=2,
            paragraph_number=1,
        )
        result = tm.start(task_id=task_id, params=params)
        print(result)


if __name__ == "__main__":
    unittest.main()
