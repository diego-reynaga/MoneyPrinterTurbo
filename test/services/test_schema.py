import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models import schema
from app.models.schema import (
    SubtitleRequest,
    TaskVideoRequest,
    VideoAspect,
    VideoFitMode,
    VideoParams,
)


class TestVideoAspect(unittest.TestCase):
    def test_to_resolution_known_aspects(self):
        self.assertEqual(VideoAspect.landscape.to_resolution(), (1920, 1080))
        self.assertEqual(VideoAspect.portrait.to_resolution(), (1080, 1920))
        self.assertEqual(VideoAspect.square.to_resolution(), (1080, 1080))

    def test_to_resolution_rejects_unsupported_value(self):
        with self.assertRaises(ValueError):
            VideoAspect.to_resolution("4:5")


class TestVideoParams(unittest.TestCase):
    def test_video_fit_mode_defaults_to_cover_and_validates_values(self):
        self.assertEqual(
            VideoParams(video_subject="Coffee").video_fit_mode,
            VideoFitMode.cover,
        )
        self.assertEqual(
            VideoParams(
                video_subject="Coffee", video_fit_mode="contain"
            ).video_fit_mode,
            VideoFitMode.contain,
        )
        with self.assertRaises(ValidationError):
            VideoParams(video_subject="Coffee", video_fit_mode="stretch")

    def test_rejects_non_positive_generation_counts(self):
        for field_name in ("video_clip_duration", "video_count"):
            for value in (0, -1, None):
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaises(ValidationError):
                        VideoParams(video_subject="Coffee", **{field_name: value})

    def test_accepts_positive_generation_counts(self):
        params = VideoParams(
            video_subject="Coffee", video_clip_duration=1, video_count=1
        )

        self.assertEqual(params.video_clip_duration, 1)
        self.assertEqual(params.video_count, 1)

    def test_api_video_count_is_bounded_without_limiting_cli_params(self):
        """Una tarea HTTP no debe programar un lote de renderizado ilimitado."""
        self.assertEqual(
            TaskVideoRequest(video_subject="Coffee", video_count=5).video_count, 5
        )
        with self.assertRaises(ValidationError):
            TaskVideoRequest(video_subject="Coffee", video_count=6)

        # El modelo compartido también alimenta la CLI, que documenta cualquier cantidad
        # positiva, así que su comportamiento actual debe seguir disponible allí.
        self.assertEqual(
            VideoParams(video_subject="Coffee", video_count=6).video_count, 6
        )

    def test_api_render_threads_are_bounded_without_limiting_cli_params(self):
        """Quien llama por HTTP no puede pasar cantidades arbitrarias de hilos de FFmpeg."""
        self.assertEqual(
            TaskVideoRequest(video_subject="Coffee", n_threads=16).n_threads, 16
        )
        for value in (0, -1, 17):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    TaskVideoRequest(video_subject="Coffee", n_threads=value)

        self.assertEqual(
            VideoParams(video_subject="Coffee", n_threads=17).n_threads, 17
        )

    def test_api_clip_duration_is_bounded_without_limiting_cli_params(self):
        """La API mantiene cada clip dentro del rango de duración de proveedor de la WebUI."""
        self.assertEqual(
            TaskVideoRequest(video_subject="Coffee", video_clip_duration=15).video_clip_duration,
            15,
        )
        with self.assertRaises(ValidationError):
            TaskVideoRequest(video_subject="Coffee", video_clip_duration=16)

        self.assertEqual(
            VideoParams(video_subject="Coffee", video_clip_duration=16).video_clip_duration,
            16,
        )

    def test_subtitle_modes_accept_only_supported_api_values(self):
        """Los nuevos parámetros de subtítulos deben rechazar errores de escritura, para
        que una solicitud correcta no degrade en silencio.
        """
        params = VideoParams(
            video_subject="Coffee",
            subtitle_display_mode="word_by_word",
            subtitle_animation="pop_spring",
        )
        request = SubtitleRequest(
            video_script="Coffee",
            subtitle_display_mode="word_by_word",
            subtitle_animation="pop_spring",
        )

        self.assertEqual(params.subtitle_display_mode, "word_by_word")
        self.assertEqual(params.subtitle_animation, "pop_spring")
        self.assertEqual(request.subtitle_display_mode, "word_by_word")
        self.assertEqual(request.subtitle_animation, "pop_spring")

        invalid_cases = (
            ("subtitle_display_mode", "word-by-word"),
            ("subtitle_display_mode", "invalid"),
            ("subtitle_display_mode", None),
            ("subtitle_animation", "pop-spring"),
            ("subtitle_animation", "invalid"),
            ("subtitle_animation", None),
        )
        for field_name, value in invalid_cases:
            with self.subTest(model="VideoParams", field=field_name, value=value):
                with self.assertRaises(ValidationError):
                    VideoParams(video_subject="Coffee", **{field_name: value})
            with self.subTest(model="SubtitleRequest", field=field_name, value=value):
                with self.assertRaises(ValidationError):
                    SubtitleRequest(video_script="Coffee", **{field_name: value})

    def test_subtitle_request_normalizes_enabled_flag_to_bool(self):
        for value in (False, "false"):
            with self.subTest(value=value):
                request = SubtitleRequest(
                    video_script="Coffee", subtitle_enabled=value
                )

                self.assertIs(request.subtitle_enabled, False)

    def test_invalid_saved_subtitle_mode_falls_back_during_upgrade(self):
        """Si la configuración antigua contiene valores no válidos, debe volver a los
        valores por defecto en lugar de impedir que arranque el servicio.
        """
        with patch.object(
            schema.config,
            "ui",
            {
                "subtitle_display_mode": "word-by-word",
                "subtitle_animation": "pop-spring",
            },
        ):
            self.assertEqual(
                schema._get_valid_ui_choice(
                    "subtitle_display_mode",
                    schema._SUBTITLE_DISPLAY_MODES,
                    "sentence",
                ),
                "sentence",
            )
            self.assertEqual(
                schema._get_valid_ui_choice(
                    "subtitle_animation",
                    schema._SUBTITLE_ANIMATIONS,
                    "none",
                ),
                "none",
            )


if __name__ == "__main__":
    unittest.main()
