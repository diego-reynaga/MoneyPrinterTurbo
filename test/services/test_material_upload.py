import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

from PIL import Image, UnidentifiedImageError

from app.models import const
from app.services import material_upload


class _UnseekableUpload(io.BytesIO):
    def seek(self, *args, **kwargs):
        raise OSError("acceso aleatorio")


class _TextUpload(io.BytesIO):
    def read(self, *args, **kwargs):
        return "not binary"


def _image_bytes(image_format: str = "PNG") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (2, 2), color="red").save(output, format=image_format)
    return output.getvalue()


class TestMaterialUploadService(unittest.TestCase):
    def test_sanitize_filename_strips_paths_and_validates_complete_extension(self):
        self.assertEqual(
            material_upload.sanitize_material_filename(r"C:\videos\clip.MOV"),
            "clip.MOV",
        )
        self.assertEqual(
            material_upload.sanitize_material_filename("../../images/photo.png"),
            "photo.png",
        )

        for filename in ("", ".", "..", "photojpg", "clip.mp4\x00", "clip.exe"):
            with self.subTest(filename=filename):
                with self.assertRaises(material_upload.MaterialUploadError):
                    material_upload.sanitize_material_filename(filename)

    def test_sanitize_filename_rejects_windows_invalid_and_reserved_names(self):
        # bgm.sanitize_upload_filename ya aplica estas reglas de Windows a la subida de
        # música de fondo. La subida de material local también guarda su archivo con un
        # UUID, así que ambos endpoints deben coincidir en qué nombres del cliente
        # aceptan en lugar de diferir según la plataforma.
        for filename in (
            "CON.mp4",
            "con.mp4",
            "lpt1.webm",
            "COM¹.mp4",
            "lpt².webm",
            "com³.png",
            "aux.extra.mp4",
            "bad:name.mp4",
            "bad?.mp4",
            "bad<1>.mp4",
            'quote".mp4',
            "pipe|name.mp4",
            "star*.mp4",
        ):
            with self.subTest(filename=filename):
                with self.assertRaises(material_upload.MaterialUploadError):
                    material_upload.sanitize_material_filename(filename)

        # Los nombres que solo contienen una palabra reservada, o un carácter no válido
        # en la raíz sin extensión, siguen siendo subidas normales.
        self.assertEqual(
            material_upload.sanitize_material_filename("console.mp4"), "console.mp4"
        )
        self.assertEqual(
            material_upload.sanitize_material_filename(r"C:\videos\aux-extra.png"),
            "aux-extra.png",
        )

    def test_sanitize_filename_rejects_control_and_bidi_characters(self):
        """Los caracteres de control y los de control bidireccional estropean cómo se
        muestra el nombre del archivo en los logs y en la interfaz; deben rechazarse.
        """
        for filename in (
            # Caracteres de control C0 (comportamiento original; aquí solo como
            # protección de regresión)
            "clip\x00.mp4",
            "clip\n.mp4",
            "clip\t.mp4",
            # Caracteres de control C1: algunos visores de logs muestran U+0085 como
            # salto de línea
            "clip\x7f.mp4",
            "clip\x85.mp4",
            "clip\x9b.mp4",
            # Caracteres de control de texto bidireccional: el texto tras U+202E se
            # muestra invertido
            "photo\u202egnp.mp4",
            "clip\u200e.mp4",
            "clip\u2069.mp4",
            # Separadores Unicode de línea/párrafo
            "clip\u2028.mp4",
            "clip\u2029.mp4",
        ):
            with self.subTest(filename=filename):
                with self.assertRaises(material_upload.MaterialUploadError):
                    material_upload.sanitize_material_filename(filename)

    def test_sanitize_filename_keeps_printable_unicode_names(self):
        """Solo se bloquean los caracteres de control y los de control bidireccional, sin
        afectar a nombres normales con chino, caracteres combinantes, etc.
        """
        for filename in (
            "用户素材.mp4",
            "my clip.mp4",
            "clip\u00a0.mp4",  # Espacio de no separación: se puede escribir directamente y no cambia el orden de visualización
            "e\u0301tude.mp4",  # Acento combinante
            "clip\u2019s.mp4",  # Comillas tipográficas
            "clip\ufe0f.mp4",  # Selector de variante (Cf, pero no afecta al orden de visualización)
            "family\u200d.mp4",  # ZWJ (Cf; las secuencias de emoji dependen de él)
        ):
            with self.subTest(filename=filename):
                self.assertEqual(
                    material_upload.sanitize_material_filename(filename), filename
                )

    def test_video_upload_is_chunked_validated_and_atomically_persisted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = io.BytesIO(b"decodable-video-placeholder")
            with (
                patch.object(
                    material_upload, "uploaded_material_dir", return_value=temp_dir
                ),
                patch.object(material_upload, "_validate_video") as validate_video,
                patch.object(
                    material_upload,
                    "uuid4",
                    return_value=UUID("4fca18fc-e734-4f3a-a824-777a40d45c8c"),
                ),
                patch.object(
                    material_upload.os, "replace", wraps=os.replace
                ) as replace,
            ):
                stored_name = material_upload.save_material_upload("clip.MOV", source)

            self.assertEqual(stored_name, "4fca18fce7344f3aa824777a40d45c8c.mov")
            self.assertEqual(
                Path(temp_dir, stored_name).read_bytes(),
                b"decodable-video-placeholder",
            )
            validate_video.assert_called_once()
            replace.assert_called_once()
            self.assertEqual(source.tell(), 0)
            self.assertFalse(
                any(
                    name.startswith(".material-upload-")
                    for name in os.listdir(temp_dir)
                )
            )

    def test_same_original_name_creates_immutable_storage_keys(self):
        generated_uuids = [
            UUID("11111111-1111-4111-8111-111111111111"),
            UUID("22222222-2222-4222-8222-222222222222"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    material_upload, "uploaded_material_dir", return_value=temp_dir
                ),
                patch.object(material_upload, "_validate_video"),
                patch.object(material_upload, "uuid4", side_effect=generated_uuids),
            ):
                first = material_upload.save_material_upload(
                    "shared.mp4", io.BytesIO(b"first")
                )
                second = material_upload.save_material_upload(
                    "shared.mp4", io.BytesIO(b"second")
                )

            self.assertNotEqual(first, second)
            self.assertEqual(Path(temp_dir, first).read_bytes(), b"first")
            self.assertEqual(Path(temp_dir, second).read_bytes(), b"second")

    def test_image_upload_validates_content_and_extension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                material_upload, "uploaded_material_dir", return_value=temp_dir
            ):
                stored_name = material_upload.save_material_upload(
                    "photo.PNG", io.BytesIO(_image_bytes("PNG"))
                )
                with self.assertRaisesRegex(
                    material_upload.MaterialUploadError, "no coincide"
                ):
                    material_upload.save_material_upload(
                        "renamed.png", io.BytesIO(_image_bytes("JPEG"))
                    )
                with self.assertRaisesRegex(
                    material_upload.MaterialUploadError, "JPEG, PNG o BMP válida"
                ):
                    material_upload.save_material_upload(
                        "broken.jpg", io.BytesIO(b"not-an-image")
                    )

            self.assertTrue(stored_name.endswith(".png"))
            self.assertEqual(len(os.listdir(temp_dir)), 1)

    def test_truncated_jpeg_is_rejected_before_publishing(self):
        # verify() de Pillow acepta este marcador final ausente, pero decodificar los
        # píxeles después lanza OSError y haría fallar una tarea que por lo demás es
        # válida.
        truncated_jpeg = _image_bytes("JPEG")[:-2]
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                material_upload, "uploaded_material_dir", return_value=temp_dir
            ):
                with self.assertRaisesRegex(
                    material_upload.MaterialUploadError, "JPEG, PNG o BMP válida"
                ):
                    material_upload.save_material_upload(
                        "truncated.jpg", io.BytesIO(truncated_jpeg)
                    )
            self.assertEqual(os.listdir(temp_dir), [])

    def test_multi_picture_camera_jpeg_is_accepted(self):
        # Los móviles y las cámaras de doble lente incrustan un índice MPF en fotos .jpg
        # normales; Pillow informa de esos archivos como MPO en lugar de JPEG.
        output = io.BytesIO()
        Image.new("RGB", (4, 4), color="red").save(
            output,
            format="MPO",
            save_all=True,
            append_images=[Image.new("RGB", (2, 2), color="blue")],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                material_upload, "uploaded_material_dir", return_value=temp_dir
            ):
                stored_name = material_upload.save_material_upload(
                    "IMG_0001.JPG", io.BytesIO(output.getvalue())
                )
                with self.assertRaisesRegex(
                    material_upload.MaterialUploadError, "no coincide"
                ):
                    material_upload.save_material_upload(
                        "renamed.png", io.BytesIO(output.getvalue())
                    )

            self.assertTrue(stored_name.endswith(".jpg"))
            with Image.open(os.path.join(temp_dir, stored_name)) as image:
                self.assertEqual(image.format, "MPO")
                self.assertEqual(image.size, (4, 4))

    def test_webm_and_bmp_materials_are_accepted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    material_upload, "uploaded_material_dir", return_value=temp_dir
                ),
                patch.object(material_upload, "_validate_video") as validate_video,
            ):
                video_name = material_upload.save_material_upload(
                    "clip.webm", io.BytesIO(b"decodable-video-placeholder")
                )
                image_name = material_upload.save_material_upload(
                    "photo.bmp", io.BytesIO(_image_bytes("BMP"))
                )

                # La extensión declarada sigue teniendo que coincidir con el formato
                # real de la imagen.
                with self.assertRaisesRegex(
                    material_upload.MaterialUploadError, "no coincide"
                ):
                    material_upload.save_material_upload(
                        "renamed.bmp", io.BytesIO(_image_bytes("PNG"))
                    )

            self.assertTrue(video_name.endswith(".webm"))
            self.assertTrue(image_name.endswith(".bmp"))
            validate_video.assert_called_once()
            self.assertEqual(len(os.listdir(temp_dir)), 2)

    def test_supported_extensions_cover_cli_accepted_formats(self):
        cli_accepted_formats = {
            *(f".{extension}" for extension in const.FILE_TYPE_VIDEOS),
            *(f".{extension}" for extension in const.FILE_TYPE_IMAGES),
            ".avi",
            ".flv",
        }
        supported = set(material_upload.SUPPORTED_MATERIAL_EXTENSIONS)

        self.assertTrue(
            cli_accepted_formats.issubset(supported),
            "local material upload must accept every format the CLI accepts; "
            f"missing: {sorted(cli_accepted_formats - supported)}",
        )
        # Cada extensión de imagen aceptada necesita un formato de Pillow declarado; si
        # no, la validación de imagen lanza una excepción en lugar de devolver
        # MaterialUploadError.
        self.assertEqual(
            set(material_upload.SUPPORTED_IMAGE_EXTENSIONS),
            set(material_upload._IMAGE_FORMATS_BY_EXTENSION),
        )

    def test_renamed_image_is_not_accepted_as_video(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    material_upload, "uploaded_material_dir", return_value=temp_dir
                ),
                patch.object(material_upload.subprocess, "run") as run,
            ):
                with self.assertRaisesRegex(
                    material_upload.MaterialUploadError, "no una imagen"
                ):
                    material_upload.save_material_upload(
                        "renamed.mp4", io.BytesIO(_image_bytes("JPEG"))
                    )

            run.assert_not_called()
            self.assertEqual(os.listdir(temp_dir), [])

    def test_empty_unseekable_and_non_binary_uploads_are_cleaned_up(self):
        invalid_sources = (
            (io.BytesIO(b""), "está vacío"),
            (_UnseekableUpload(b"video"), "acceso aleatorio"),
            (_TextUpload(b"video"), "debe ser binaria"),
        )
        for source, expected_error in invalid_sources:
            with self.subTest(expected_error=expected_error):
                with tempfile.TemporaryDirectory() as temp_dir:
                    with patch.object(
                        material_upload,
                        "uploaded_material_dir",
                        return_value=temp_dir,
                    ):
                        with self.assertRaisesRegex(
                            material_upload.MaterialUploadError, expected_error
                        ):
                            material_upload.save_material_upload("clip.mp4", source)
                    self.assertEqual(os.listdir(temp_dir), [])
                self.assertEqual(source.tell(), 0)

    def test_per_media_size_limits_are_enforced_and_temp_files_removed(self):
        cases = (
            ("clip.mp4", "MAX_VIDEO_MATERIAL_UPLOAD_BYTES"),
            ("photo.png", "MAX_IMAGE_MATERIAL_UPLOAD_BYTES"),
        )
        for filename, limit_name in cases:
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as temp_dir:
                    with (
                        patch.object(
                            material_upload,
                            "uploaded_material_dir",
                            return_value=temp_dir,
                        ),
                        patch.object(material_upload, limit_name, 4),
                    ):
                        with self.assertRaisesRegex(
                            material_upload.MaterialUploadError, "supera"
                        ):
                            material_upload.save_material_upload(
                                filename, io.BytesIO(b"12345")
                            )
                    self.assertEqual(os.listdir(temp_dir), [])

    def test_validation_failure_and_storage_failure_remove_temp_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    material_upload, "uploaded_material_dir", return_value=temp_dir
                ),
                patch.object(
                    material_upload,
                    "_validate_video",
                    side_effect=material_upload.MaterialUploadError("invalid video"),
                ),
            ):
                with self.assertRaises(material_upload.MaterialUploadError):
                    material_upload.save_material_upload(
                        "clip.mp4", io.BytesIO(b"broken")
                    )
            self.assertEqual(os.listdir(temp_dir), [])

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    material_upload, "uploaded_material_dir", return_value=temp_dir
                ),
                patch.object(material_upload, "_validate_video"),
                patch.object(
                    material_upload.os,
                    "replace",
                    side_effect=OSError("disk full"),
                ),
            ):
                with self.assertRaises(material_upload.MaterialServiceError):
                    material_upload.save_material_upload(
                        "clip.mp4", io.BytesIO(b"video")
                    )
            self.assertEqual(os.listdir(temp_dir), [])

    def test_video_validation_uses_configured_ffmpeg_and_video_stream(self):
        completed = SimpleNamespace(returncode=0)
        with (
            patch.object(
                material_upload.Image,
                "open",
                side_effect=UnidentifiedImageError("no una imagen"),
            ),
            patch.object(
                material_upload.utils,
                "get_ffmpeg_binary",
                return_value="/portable/imageio/ffmpeg",
            ),
            patch.object(
                material_upload.subprocess, "run", return_value=completed
            ) as run,
        ):
            material_upload._validate_video("/tmp/clip.mp4")

        command = run.call_args.args[0]
        self.assertEqual(command[0], "/portable/imageio/ffmpeg")
        self.assertIn("0:V:0", command)
        self.assertIn("-xerror", command)
        self.assertNotIn("ffprobe", " ".join(command).lower())

    def test_video_validation_distinguishes_invalid_media_from_tool_failure(self):
        image_error = UnidentifiedImageError("no una imagen")
        with patch.object(material_upload.Image, "open", side_effect=image_error):
            with patch.object(
                material_upload.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=1),
            ):
                with self.assertRaises(material_upload.MaterialUploadError):
                    material_upload._validate_video("/tmp/clip.mp4")

            with patch.object(
                material_upload.subprocess,
                "run",
                side_effect=OSError("ffmpeg missing"),
            ):
                with self.assertRaises(material_upload.MaterialServiceError):
                    material_upload._validate_video("/tmp/clip.mp4")

            with patch.object(
                material_upload.subprocess,
                "run",
                side_effect=material_upload.subprocess.TimeoutExpired("ffmpeg", 120),
            ):
                with self.assertRaises(material_upload.MaterialServiceError):
                    material_upload._validate_video("/tmp/clip.mp4")


if __name__ == "__main__":
    unittest.main()
