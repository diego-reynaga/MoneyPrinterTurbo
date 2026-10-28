import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# añade la raíz del proyecto al path de Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils import utils


class TestCheckFfmpegReady(unittest.TestCase):
    """Cubre los cuatro resultados de detección de utils.check_ffmpeg_ready()."""

    def test_returns_true_when_ffmpeg_probe_succeeds(self):
        completed = subprocess.CompletedProcess(args=["ffmpeg", "-version"], returncode=0)
        with (
            patch.object(utils, "get_ffmpeg_binary", return_value="/usr/bin/ffmpeg"),
            patch.object(utils.subprocess, "run", return_value=completed) as run,
        ):
            self.assertTrue(utils.check_ffmpeg_ready())

        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["/usr/bin/ffmpeg", "-version"])

    def test_returns_false_when_executable_is_missing(self):
        with (
            patch.object(utils, "get_ffmpeg_binary", return_value="ffmpeg"),
            patch.object(utils.subprocess, "run", side_effect=FileNotFoundError()),
            patch.object(utils.logger, "warning") as warning,
        ):
            self.assertFalse(utils.check_ffmpeg_ready())

        warning.assert_called_once()
        self.assertIn("no se encontró ningún ejecutable de ffmpeg utilizable", warning.call_args.args[0])

    def test_returns_false_on_non_zero_exit_code(self):
        completed = subprocess.CompletedProcess(args=["ffmpeg", "-version"], returncode=1)
        with (
            patch.object(utils, "get_ffmpeg_binary", return_value="/usr/bin/ffmpeg"),
            patch.object(utils.subprocess, "run", return_value=completed),
            patch.object(utils.logger, "warning") as warning,
        ):
            self.assertFalse(utils.check_ffmpeg_ready())

        warning.assert_called_once()
        self.assertIn("terminó con el estado 1", warning.call_args.args[0])

    def test_returns_false_on_timeout(self):
        with (
            patch.object(utils, "get_ffmpeg_binary", return_value="/usr/bin/ffmpeg"),
            patch.object(
                utils.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=10),
            ),
            patch.object(utils.logger, "warning") as warning,
        ):
            self.assertFalse(utils.check_ffmpeg_ready(timeout=10))

        warning.assert_called_once()
        self.assertIn("no se pudo comprobar ffmpeg", warning.call_args.args[0])


class TestFfmpegDecodeFailed(unittest.TestCase):
    def _completed(self, returncode, stderr=b""):
        return subprocess.CompletedProcess(args=["ffmpeg"], returncode=returncode, stderr=stderr)

    def test_non_zero_exit_code_fails(self):
        self.assertTrue(utils.ffmpeg_decode_failed(self._completed(183)))

    def test_clean_decode_passes(self):
        self.assertFalse(utils.ffmpeg_decode_failed(self._completed(0)))

    def test_ffmpeg_61_aborted_decoder_with_zero_exit_fails(self):
        # FFmpeg 6.1 detiene el decodificador con -xerror pero sale con 0.
        stderr = (
            b"[mp3float @ 0x1] Header missing\n"
            b"[aist#0:0/mp3 @ 0x2] Decoder thread returned error: Invalid data found\n"
        )
        self.assertTrue(utils.ffmpeg_decode_failed(self._completed(0, stderr)))

    def test_benign_decoder_messages_pass(self):
        stderr = b"[h264 @ 0x1] mmco: unref short failure\n"
        self.assertFalse(utils.ffmpeg_decode_failed(self._completed(0, stderr)))

    def test_missing_stderr_attribute_passes(self):
        from types import SimpleNamespace

        self.assertFalse(utils.ffmpeg_decode_failed(SimpleNamespace(returncode=0)))


if __name__ == "__main__":
    unittest.main()
