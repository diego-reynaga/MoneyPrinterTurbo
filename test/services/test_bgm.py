import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

from fastapi import UploadFile

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import bgm
from app.controllers.v1 import video as video_controller
from app.models.exception import HttpException


class _FakeRequest:
    def __init__(self):
        self.headers = {"x-task-id": "bgm-upload-test"}


class _UnseekableUpload(io.BytesIO):
    def seek(self, *args, **kwargs):
        raise OSError("stream is not seekable")


class _TextUpload(io.BytesIO):
    def read(self, *args, **kwargs):
        return "not binary"


class TestBackgroundMusicService(unittest.TestCase):
    def test_should_use_bgm_is_provider_independent(self):
        """El interruptor general debe cubrir la fuente actual y los proveedores futuros,
        sin volver a escribir casos especiales por proveedor.
        """
        test_cases = [
            ("random", 0.2, True),
            ("custom", 0.2, True),
            ("sonilo", 0.2, True),
            ("elevenlabs", 0.2, True),
            ("future_provider", 0.2, True),
            ("random", 0.0, False),
            ("custom", -0.1, False),
            ("sonilo", None, False),
            ("future_provider", float("nan"), False),
            ("future_provider", "invalid", False),
            ("", 0.2, False),
            (None, 0.2, False),
        ]
        for bgm_type, bgm_volume, expected in test_cases:
            with self.subTest(bgm_type=bgm_type, bgm_volume=bgm_volume):
                self.assertEqual(
                    bgm.should_use_bgm(bgm_type, bgm_volume), expected
                )

    def test_sanitize_upload_filename_strips_directory_components(self):
        self.assertEqual(
            bgm.sanitize_upload_filename("../../用户音乐.mp3"), "用户音乐.mp3"
        )
        self.assertEqual(
            bgm.sanitize_upload_filename("windows\\path\\song.MP3"), "song.MP3"
        )

    def test_sanitize_upload_filename_accepts_common_audio_formats(self):
        # La WebUI y la API comparten la misma lista blanca de formatos; aquí se cubren
        # mayúsculas y contenedores habituales, para que en el futuro un cambio del
        # control de subida no haga que la interfaz los acepte y la capa de servicio
        # rechace el archivo.
        for extension in bgm.SUPPORTED_BGM_EXTENSIONS:
            filename = f"background{extension.upper()}"
            with self.subTest(filename=filename):
                self.assertEqual(bgm.sanitize_upload_filename(filename), filename)

    def test_sanitize_upload_filename_rejects_invalid_names_and_extensions(self):
        for filename in (
            "",
            "   ",
            ".",
            "..",
            "song.mp4",
            "song.mp3\x00",
            "CON.mp3",
            "lpt1.wav",
            "COM¹.mp3",
            "lpt².wav",
            "com³.flac",
            "bad:name.mp3",
            "bad?.flac",
            ".bgm-upload-user.m4a",
            "song.mp3.",
            "aux.extra.ogg",
        ):
            with self.subTest(filename=filename):
                with self.assertRaises(bgm.BgmUploadError):
                    bgm.sanitize_upload_filename(filename)

        # Win32 solo trata como nombre de dispositivo el primer segmento antes de la
        # extensión; los dígitos en superíndice en otras posiciones no afectan.
        self.assertEqual(bgm.sanitize_upload_filename("song¹.mp3"), "song¹.mp3")

    def test_sanitize_upload_filename_rejects_control_and_bidi_characters(self):
        """Los caracteres de control y los de control bidireccional estropean cómo se
        muestra el nombre del archivo en los logs y en la interfaz; deben rechazarse.
        """
        for filename in (
            # Caracteres de control C0 (comportamiento original; aquí solo como
            # protección de regresión)
            "song\x00.mp3",
            "song\n.mp3",
            "song\t.mp3",
            # Caracteres de control C1: algunos visores de logs muestran U+0085 como
            # salto de línea
            "song\x7f.mp3",
            "song\x85.mp3",
            "song\x9b.mp3",
            # Caracteres de control de texto bidireccional: el texto tras U+202E se
            # muestra invertido
            "photo\u202egnp.mp3",
            "song\u200e.mp3",
            "song\u2069.mp3",
            # Separadores Unicode de línea/párrafo
            "song\u2028.mp3",
            "song\u2029.mp3",
        ):
            with self.subTest(filename=filename):
                with self.assertRaises(bgm.BgmUploadError):
                    bgm.sanitize_upload_filename(filename)

    def test_sanitize_upload_filename_keeps_printable_unicode_names(self):
        """Solo se bloquean los caracteres de control y los de control bidireccional, sin
        afectar a nombres normales con chino, caracteres combinantes, etc.
        """
        for filename in (
            "用户音乐.mp3",
            "backing track.mp3",
            "song\u00a0.mp3",  # Espacio de no separación: se puede escribir directamente y no cambia el orden de visualización
            "e\u0301tude.mp3",  # Acento combinante
            "let\u2019s-go.mp3",  # Comillas tipográficas
            "song\ufe0f.mp3",  # Selector de variante (Cf, pero no afecta al orden de visualización)
            "family\u200d.mp3",  # ZWJ (Cf; las secuencias de emoji dependen de él)
        ):
            with self.subTest(filename=filename):
                self.assertEqual(bgm.sanitize_upload_filename(filename), filename)

    def test_save_bgm_upload_uses_atomic_storage_directory_write(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = io.BytesIO(b"valid-mp3-placeholder")
            with (
                patch.object(bgm, "uploaded_bgm_dir", return_value=temp_dir),
                patch.object(bgm, "_validate_audio", return_value=None),
                patch.object(
                    bgm,
                    "uuid4",
                    return_value=UUID("4fca18fc-e734-4f3a-a824-777a40d45c8c"),
                ),
            ):
                saved_name = bgm.save_bgm_upload("music.mp3", source)

            self.assertEqual(saved_name, "4fca18fce7344f3aa824777a40d45c8c.mp3")
            self.assertEqual(
                Path(temp_dir, saved_name).read_bytes(), b"valid-mp3-placeholder"
            )
            self.assertEqual(
                [name for name in os.listdir(temp_dir) if name.startswith(".bgm-upload-")],
                [],
            )
            # El servicio devuelve el puntero del archivo al principio, así que el mismo
            # UploadedFile sigue sirviendo para la vista previa de Streamlit o para un
            # rerun posterior, y no se queda vacío por haberlo guardado.
            self.assertEqual(source.tell(), 0)

    def test_validate_bgm_upload_checks_audio_without_persisting_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = io.BytesIO(b"valid-audio-placeholder")
            with (
                patch.object(bgm, "uploaded_bgm_dir", return_value=temp_dir),
                patch.object(bgm, "_validate_audio", return_value=None),
            ):
                validated_name = bgm.validate_bgm_upload("preview.m4a", source)

            self.assertEqual(validated_name, "preview.m4a")
            # La validación previa de la WebUI solo permite usar un archivo temporal
            # brevemente; si el usuario aún no ha pulsado generar, no se puede dejar el
            # archivo en el directorio persistente de BGM ni cambiar el puntero del
            # archivo que necesita la vista previa posterior.
            self.assertEqual(os.listdir(temp_dir), [])

    def test_staging_rejects_empty_unseekable_and_non_binary_uploads(self):
        invalid_sources = (
            (io.BytesIO(b""), "el archivo de música de fondo está vacío"),
            (_UnseekableUpload(b"audio"), "acceso aleatorio"),
            (_TextUpload(b"audio"), "debe ser binaria"),
        )
        for source, expected_error in invalid_sources:
            with self.subTest(expected_error=expected_error):
                with tempfile.TemporaryDirectory() as temp_dir:
                    with patch.object(
                        bgm, "uploaded_bgm_dir", return_value=temp_dir
                    ):
                        with self.assertRaisesRegex(
                            bgm.BgmUploadError, expected_error
                        ):
                            bgm.save_bgm_upload("invalid.mp3", source)
                    self.assertEqual(os.listdir(temp_dir), [])
            self.assertEqual(source.tell(), 0)

    def test_validate_bgm_upload_rejects_invalid_audio_and_cleans_temp_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(bgm, "uploaded_bgm_dir", return_value=temp_dir),
                patch.object(
                    bgm,
                    "_validate_audio",
                    side_effect=bgm.BgmUploadError("invalid audio"),
                ),
            ):
                with self.assertRaises(bgm.BgmUploadError):
                    bgm.validate_bgm_upload("invalid.m4a", io.BytesIO(b"invalid"))

            # Una validación previa fallida tampoco puede dejar audio temporal en el
            # directorio persistente, para que no lo elija después la lógica de BGM
            # aleatoria ni se acumulen archivos no válidos con el tiempo.
            self.assertEqual(os.listdir(temp_dir), [])

    def test_save_bgm_upload_rejects_oversized_file_and_cleans_temp_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(bgm, "uploaded_bgm_dir", return_value=temp_dir),
                patch.object(bgm, "MAX_BGM_UPLOAD_BYTES", 4),
            ):
                with self.assertRaises(bgm.BgmUploadError):
                    bgm.save_bgm_upload("large.mp3", io.BytesIO(b"12345"))

            self.assertEqual(os.listdir(temp_dir), [])

    def test_save_bgm_upload_rejects_invalid_audio_before_replacing_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir, "music.mp3")
            target.write_bytes(b"existing-valid-file")
            with (
                patch.object(bgm, "uploaded_bgm_dir", return_value=temp_dir),
                patch.object(
                    bgm,
                    "_validate_audio",
                    side_effect=bgm.BgmUploadError("invalid audio"),
                ),
            ):
                with self.assertRaises(bgm.BgmUploadError):
                    bgm.save_bgm_upload("music.mp3", io.BytesIO(b"broken-file"))

            # El fallo de validación ocurre antes de la sustitución atómica; el archivo
            # existente del usuario no puede dañarse.
            self.assertEqual(target.read_bytes(), b"existing-valid-file")
            self.assertEqual(os.listdir(temp_dir), ["music.mp3"])

    def test_same_original_filename_creates_immutable_storage_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            generated_uuids = [
                UUID("11111111-1111-4111-8111-111111111111"),
                UUID("22222222-2222-4222-8222-222222222222"),
            ]
            with (
                patch.object(bgm, "uploaded_bgm_dir", return_value=temp_dir),
                patch.object(bgm, "_validate_audio", return_value=None),
                patch.object(bgm, "uuid4", side_effect=generated_uuids),
            ):
                first_name = bgm.save_bgm_upload(
                    "shared.mp3", io.BytesIO(b"first-user-audio")
                )
                second_name = bgm.save_bgm_upload(
                    "shared.mp3", io.BytesIO(b"second-user-audio")
                )

            self.assertNotEqual(first_name, second_name)
            self.assertEqual(Path(temp_dir, first_name).read_bytes(), b"first-user-audio")
            self.assertEqual(
                Path(temp_dir, second_name).read_bytes(), b"second-user-audio"
            )

    def test_audio_validation_uses_only_configured_ffmpeg(self):
        completed = SimpleNamespace(
            returncode=0,
            stdout=b"#tb 0: 1/16000\n0,          0,          0,     1600,     3200, 0x00000000\n",
        )
        with (
            patch.object(
                bgm.utils,
                "get_ffmpeg_binary",
                return_value="/portable/imageio/ffmpeg",
            ),
            patch.object(bgm.subprocess, "run", return_value=completed) as run,
        ):
            bgm._validate_audio("/tmp/background.m4a")

        command = run.call_args.args[0]
        self.assertEqual(command[0], "/portable/imageio/ffmpeg")
        self.assertNotIn("ffprobe", " ".join(command).lower())
        self.assertIn("0:a:0", command)

    def test_audio_validation_distinguishes_service_failure_from_invalid_audio(self):
        with patch.object(
            bgm.subprocess,
            "run",
            side_effect=OSError("ffmpeg missing"),
        ):
            with self.assertRaises(bgm.BgmServiceError):
                bgm._validate_audio("/tmp/background.mp3")

        with patch.object(
            bgm.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=1),
        ):
            with self.assertRaises(bgm.BgmUploadError):
                bgm._validate_audio("/tmp/background.mp3")

        with patch.object(
            bgm.subprocess,
            "run",
            side_effect=bgm.subprocess.TimeoutExpired("ffmpeg", 30),
        ):
            with self.assertRaises(bgm.BgmServiceError):
                bgm._validate_audio("/tmp/background.mp3")

    def test_save_normalizes_extension_and_maps_storage_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(bgm, "uploaded_bgm_dir", return_value=temp_dir),
                patch.object(bgm, "_validate_audio", return_value=None),
                patch.object(
                    bgm,
                    "uuid4",
                    return_value=UUID("4fca18fc-e734-4f3a-a824-777a40d45c8c"),
                ),
            ):
                saved_name = bgm.save_bgm_upload(
                    "UPPER.FLAC", io.BytesIO(b"valid-audio")
                )
            self.assertTrue(saved_name.endswith(".flac"))

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(bgm, "uploaded_bgm_dir", return_value=temp_dir),
                patch.object(bgm, "_validate_audio", return_value=None),
                patch.object(bgm.os, "replace", side_effect=OSError("disk full")),
            ):
                with self.assertRaises(bgm.BgmServiceError):
                    bgm.save_bgm_upload("music.mp3", io.BytesIO(b"valid-audio"))
            self.assertEqual(os.listdir(temp_dir), [])

    def test_upload_controller_returns_uuid_and_maps_errors(self):
        request = _FakeRequest()
        upload = UploadFile(filename="music.mp3", file=io.BytesIO(b"audio"))
        with patch.object(
            bgm,
            "save_bgm_upload",
            return_value="4fca18fce7344f3aa824777a40d45c8c.mp3",
        ):
            response = video_controller.upload_bgm_file(request, upload)
        self.assertEqual(response["status"], 200)
        self.assertEqual(
            response["data"]["file"],
            "4fca18fce7344f3aa824777a40d45c8c.mp3",
        )

        for service_error, expected_status in (
            (bgm.BgmUploadError("invalid audio"), 400),
            (bgm.BgmServiceError("FFmpeg unavailable"), 500),
        ):
            with self.subTest(expected_status=expected_status):
                with patch.object(
                    bgm, "save_bgm_upload", side_effect=service_error
                ):
                    with self.assertRaises(HttpException) as context:
                        video_controller.upload_bgm_file(request, upload)
                self.assertEqual(context.exception.status_code, expected_status)

    def test_resolve_and_list_prefer_uploaded_file_with_same_name(self):
        with (
            tempfile.TemporaryDirectory() as builtin_dir,
            tempfile.TemporaryDirectory() as uploaded_dir,
        ):
            Path(builtin_dir, "same.mp3").write_bytes(b"builtin")
            Path(builtin_dir, "builtin.mp3").write_bytes(b"builtin")
            Path(uploaded_dir, "same.mp3").write_bytes(b"uploaded")
            Path(uploaded_dir, "user.flac").write_bytes(b"uploaded")
            Path(uploaded_dir, ".bgm-upload-pending.m4a").write_bytes(b"pending")

            with (
                patch.object(bgm, "uploaded_bgm_dir", return_value=uploaded_dir),
                patch.object(bgm.utils, "song_dir", return_value=builtin_dir),
            ):
                resolved = bgm.resolve_bgm_file("same.mp3")
                listed = bgm.list_bgm_files()
                builtin_listed = bgm.list_builtin_bgm_files()

            self.assertEqual(
                resolved, os.path.realpath(os.path.join(uploaded_dir, "same.mp3"))
            )
            self.assertEqual(
                [os.path.basename(file_path) for file_path in listed],
                ["builtin.mp3", "same.mp3", "user.flac"],
            )
            same_file = next(path for path in listed if path.endswith("same.mp3"))
            self.assertEqual(
                same_file, os.path.realpath(os.path.join(uploaded_dir, "same.mp3"))
            )
            self.assertEqual(
                [os.path.basename(file_path) for file_path in builtin_listed],
                ["builtin.mp3", "same.mp3"],
            )
            with patch.object(bgm.utils, "song_dir", return_value=builtin_dir):
                self.assertEqual(
                    bgm.resolve_builtin_bgm_file("same.mp3"),
                    os.path.realpath(os.path.join(builtin_dir, "same.mp3")),
                )

                for invalid_path in (
                    "../same.mp3",
                    str(Path(uploaded_dir, "user.flac")),
                ):
                    with self.subTest(invalid_path=invalid_path):
                        with self.assertRaises(ValueError):
                            bgm.resolve_builtin_bgm_file(invalid_path)

    def test_resolve_rejects_staged_upload_left_by_an_interrupted_write(self):
        """Un archivo intermedio temporal que quede de una escritura interrumpida no puede
        interpretarse como música de fondo válida.

        ``_list_bgm_files`` ya excluye de la BGM aleatoria estos archivos intermedios
        aún no validados, pero el ``bgm_file`` que pasan la API y la CLI pasa por
        ``resolve_bgm_file``: el mismo archivo debe dar la misma conclusión en ambas
        entradas, o un cliente podría hacer que una tarea use un archivo intermedio
        incompleto que no se puede decodificar.
        """

        with tempfile.TemporaryDirectory() as uploaded_dir:
            staged_name = f"{bgm._INTERNAL_UPLOAD_PREFIX}pending.m4a"
            Path(uploaded_dir, staged_name).write_bytes(b"partial")
            Path(uploaded_dir, "user.flac").write_bytes(b"uploaded")

            with patch.object(bgm, "uploaded_bgm_dir", return_value=uploaded_dir):
                for candidate in (
                    staged_name,
                    os.path.join(uploaded_dir, staged_name),
                    # Los sistemas de archivos de Windows y macOS no distinguen
                    # mayúsculas, así que la comprobación del prefijo temporal tampoco
                    # debe hacerlo; si no, el mismo archivo resucitaría al escribirlo
                    # con otras mayúsculas.
                    staged_name.upper(),
                ):
                    with self.subTest(candidate=candidate):
                        with self.assertRaises(ValueError):
                            bgm.resolve_bgm_file(candidate)

                # Un archivo subido normal se sigue resolviendo correctamente (control
                # inverso).
                self.assertEqual(
                    bgm.resolve_bgm_file("user.flac"),
                    os.path.realpath(os.path.join(uploaded_dir, "user.flac")),
                )


if __name__ == "__main__":
    unittest.main()
