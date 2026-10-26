import io
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from loguru import logger
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import bgm, elevenlabs_music, sonilo, voice


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"
I18N_DIR = ROOT_DIR / "webui" / "i18n"
TEST_LOCALES = ("es",)


def _valid_wav_bytes() -> bytes:
    """Genera un WAV estándar muy corto, para que el test no dependa de audio externo al
    repositorio ni de grabaciones del sistema.
    """
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * 800)
    return output.getvalue()


class TestWebuiBackgroundMusic(unittest.TestCase):
    @staticmethod
    def _translation(locale, key):
        """Lee el texto esperado según el idioma del test, para que la aserción no dependa
        a su vez de un idioma de visualización concreto.
        """
        locale_data = json.loads(
            (I18N_DIR / f"{locale}.json").read_text(encoding="utf-8")
        )
        return locale_data["Translation"][key]

    def _widget_by_key(self, elements, key_prefix):
        """Busca el control por una clave de negocio estable, para encontrar el mismo
        control aunque se traduzca la etiqueta visible.
        """
        widget = next(
            (
                item
                for item in elements
                if str(getattr(item, "key", "")) == key_prefix
                or str(getattr(item, "key", "")).startswith(f"{key_prefix}_")
            ),
            None,
        )
        self.assertIsNotNone(widget, f"widget not found: {key_prefix}")
        return widget

    def _open_custom_bgm_panel(self, locale):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        # La CI no tiene el idioma guardado en el config.toml local. Se sobrescribe
        # explícitamente el locale de la sesión para que el test sea determinista con
        # independencia de la configuración del desarrollador.
        app.session_state["ui_language"] = locale
        app.run()
        source_select = self._widget_by_key(app.selectbox, "bgm_type_select")
        # Las opciones reales de stable_selectbox son valores de negocio; solo el texto
        # visible cambia con el locale.
        source_select.set_value("custom").run()
        return app

    def _open_sonilo_bgm_panel(self, locale):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = locale
        app.run()
        source_select = self._widget_by_key(app.selectbox, "bgm_type_select")
        source_select.set_value("sonilo").run()
        return app

    def _open_elevenlabs_bgm_panel(self, locale):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = locale
        app.run()
        source_select = self._widget_by_key(app.selectbox, "bgm_type_select")
        source_select.set_value("elevenlabs").run()
        return app

    def _open_preset_bgm_panel(self, locale):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = locale
        app.run()
        source_select = self._widget_by_key(app.selectbox, "bgm_type_select")
        source_select.set_value("preset").run()
        return app

    def _uploader(self, app):
        return self._widget_by_key(app.file_uploader, "custom_bgm_uploader")

    def _volume_select(self, app):
        return self._widget_by_key(app.selectbox, "bgm_volume_select")

    def test_preset_song_selection_is_previewed_and_persisted(self):
        """Al cambiar de canción predefinida el reproductor debe actualizarse de inmediato,
        conservando la configuración estable del nombre de archivo.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            first_song = Path(temp_dir) / "first.wav"
            second_song = Path(temp_dir) / "second.wav"
            first_song.write_bytes(_valid_wav_bytes())
            second_song.write_bytes(_valid_wav_bytes())
            test_ui = dict(
                config.ui,
                language="es",
                bgm_type="random",
                preset_song=first_song.name,
            )

            with (
                patch.object(config, "ui", test_ui),
                patch.object(config, "try_save_config", return_value=True),
                patch.object(
                    bgm,
                    "list_builtin_bgm_files",
                    return_value=[str(first_song), str(second_song)],
                ),
            ):
                app = self._open_preset_bgm_panel("es")
                preset_select = self._widget_by_key(
                    app.selectbox, "preset_song_select"
                )
                self.assertEqual(preset_select.value, first_song.name)
                self.assertEqual(len(app.get("audio")), 1)

                preset_select.set_value(second_song.name).run()

            updated_select = self._widget_by_key(
                app.selectbox, "preset_song_select"
            )
            self.assertEqual(updated_select.value, second_song.name)
            self.assertEqual(test_ui["preset_song"], second_song.name)
            self.assertEqual(len(app.get("audio")), 1)
            self.assertEqual([str(item.value) for item in app.exception], [])

    def test_empty_preset_song_list_shows_localized_warning(self):
        """Si no hay canciones disponibles debe mostrarse un aviso en el idioma actual, en
        lugar de un selector no válido.
        """
        for locale in TEST_LOCALES:
            with self.subTest(locale=locale):
                test_ui = dict(config.ui, language=locale, bgm_type="random")
                with (
                    patch.object(config, "ui", test_ui),
                    patch.object(config, "try_save_config", return_value=True),
                    patch.object(bgm, "list_builtin_bgm_files", return_value=[]),
                ):
                    app = self._open_preset_bgm_panel(locale)

                self.assertTrue(
                    any(
                        item.value
                        == self._translation(locale, "No Background Music Available")
                        for item in app.warning
                    )
                )
                self.assertFalse(
                    any(
                        str(getattr(item, "key", "")).startswith(
                            "preset_song_select"
                        )
                        for item in app.selectbox
                    )
                )
                self.assertEqual([str(item.value) for item in app.exception], [])

    def test_task_restore_selects_the_original_preset_song(self):
        """Al restaurar una tarea histórica no debe sobrescribirla la canción predefinida
        guardada globalmente.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_song = Path(temp_dir) / "saved.wav"
            restored_song = Path(temp_dir) / "restored.wav"
            saved_song.write_bytes(_valid_wav_bytes())
            restored_song.write_bytes(_valid_wav_bytes())
            test_ui = dict(
                config.ui,
                language="es",
                bgm_type="random",
                preset_song=saved_song.name,
            )

            with (
                patch.object(config, "ui", test_ui),
                patch.object(config, "try_save_config", return_value=True),
                patch.object(
                    bgm,
                    "list_builtin_bgm_files",
                    return_value=[str(saved_song), str(restored_song)],
                ),
            ):
                app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
                app.session_state["ui_language"] = "es"
                app.session_state["task_restore_payload"] = {
                    "task_id": "preset-bgm-restore-test",
                    "params": {
                        "bgm_type": "preset",
                        "bgm_file": str(restored_song),
                    },
                }
                app.run()

            preset_select = self._widget_by_key(
                app.selectbox, "preset_song_select"
            )
            self.assertEqual(preset_select.value, restored_song.name)
            self.assertEqual(test_ui["preset_song"], restored_song.name)
            self.assertEqual(len(app.get("audio")), 1)
            self.assertEqual([str(item.value) for item in app.exception], [])

    def test_missing_preset_song_does_not_interrupt_the_page(self):
        """Si un archivo deja de ser válido tras enumerarlo, debe mostrarse un aviso y no
        crearse un reproductor roto.
        """
        missing_song = Path(tempfile.gettempdir()) / "mpt-missing-preset.wav"
        missing_song.unlink(missing_ok=True)
        test_ui = dict(config.ui, language="es", bgm_type="random")

        with (
            patch.object(config, "ui", test_ui),
            patch.object(config, "try_save_config", return_value=True),
            patch.object(
                bgm,
                "list_builtin_bgm_files",
                return_value=[str(missing_song)],
            ),
        ):
            app = self._open_preset_bgm_panel("es")

        self.assertIn(
            self._translation("es", "Background Music Preview Failed"),
            [item.value for item in app.warning],
        )
        self.assertEqual(len(app.get("audio")), 0)
        self.assertEqual([str(item.value) for item in app.exception], [])

    def test_invalid_audio_shows_error_without_ready_state_or_player(self):
        for locale in TEST_LOCALES:
            with self.subTest(locale=locale):
                app = self._open_custom_bgm_panel(locale)
                with patch.object(logger, "warning") as warning:
                    self._uploader(app).set_value(
                        (
                            "invalid.m4a",
                            b"not-a-decodable-audio-file",
                            "audio/mp4",
                        )
                    ).run()
                    # Cuando un archivo no válido sigue en el control de subida, ajustar
                    # el volumen provoca un rerun de Streamlit. Un acierto de caché solo
                    # puede volver a dibujar el error, sin validar otra vez ni registrar
                    # otra vez el warning.
                    self._volume_select(app).set_value(0.4).run()

                rejection_logs = [
                    call
                    for call in warning.call_args_list
                    if "WebUI background music validation rejected" in str(call)
                ]
                self.assertEqual([str(item.value) for item in app.exception], [])
                self.assertEqual(
                    [item.value for item in app.error],
                    [self._translation(locale, "Invalid Background Music")],
                )
                self.assertFalse(
                    any("invalid.m4a" in item.value for item in app.info)
                )
                self.assertEqual(len(app.get("audio")), 0)
                self.assertEqual(len(rejection_logs), 1)

    def test_valid_audio_shows_ready_state_and_reuses_validation_cache(self):
        for locale in TEST_LOCALES:
            with self.subTest(locale=locale):
                app = self._open_custom_bgm_panel(locale)
                self._uploader(app).set_value(
                    ("valid.wav", _valid_wav_bytes(), "audio/wav")
                ).run()

                # Tras pasar la primera validación, la función del servicio se cambia
                # para que falle explícitamente; si el rerun de volumen llamara otra vez
                # a FFmpeg por error, AppTest recibiría un AssertionError.
                with patch.object(
                    bgm,
                    "validate_bgm_upload",
                    side_effect=AssertionError(
                        "validation repeated during rerun"
                    ),
                ):
                    self._volume_select(app).set_value(0.4).run()

                self.assertEqual([str(item.value) for item in app.exception], [])
                self.assertEqual([item.value for item in app.error], [])
                self.assertEqual(
                    [item.value for item in app.info if "valid.wav" in item.value],
                    [
                        f"{self._translation(locale, 'Background Music Ready')}: "
                        "valid.wav"
                    ],
                )
                self.assertEqual(len(app.get("audio")), 1)

    def test_zero_volume_defers_custom_upload_validation_until_enabled(self):
        """Con volumen 0 se conserva la selección subida, pero solo se valida y se
        previsualiza al volver a activar la BGM.
        """
        app = self._open_custom_bgm_panel("es")
        self._volume_select(app).set_value(0.0).run()

        with patch.object(bgm, "validate_bgm_upload") as validation:
            self._uploader(app).set_value(
                ("deferred.wav", _valid_wav_bytes(), "audio/wav")
            ).run()

        validation.assert_not_called()
        self.assertEqual([str(item.value) for item in app.exception], [])
        self.assertEqual([item.value for item in app.error], [])
        self.assertFalse(any("deferred.wav" in item.value for item in app.info))
        self.assertEqual(len(app.get("audio")), 0)

        # El archivo sigue en la sesión de Streamlit. Cuando el usuario sube el volumen,
        # ese mismo rerun debe completar automáticamente la validación y mostrar el
        # reproductor, sin tener que elegir de nuevo el archivo.
        with patch.object(bgm, "validate_bgm_upload") as validation:
            self._volume_select(app).set_value(0.2).run()

        validation.assert_called_once()
        self.assertEqual([str(item.value) for item in app.exception], [])
        self.assertEqual([item.value for item in app.error], [])
        self.assertTrue(any("deferred.wav" in item.value for item in app.info))
        self.assertEqual(len(app.get("audio")), 1)

    def test_service_failure_is_not_reported_as_invalid_user_audio(self):
        for locale in TEST_LOCALES:
            with self.subTest(locale=locale):
                app = self._open_custom_bgm_panel(locale)
                with patch.object(
                    bgm,
                    "validate_bgm_upload",
                    side_effect=bgm.BgmServiceError("FFmpeg unavailable"),
                ):
                    self._uploader(app).set_value(
                        ("valid.wav", _valid_wav_bytes(), "audio/wav")
                    ).run()

                self.assertEqual([str(item.value) for item in app.exception], [])
                self.assertEqual(
                    [item.value for item in app.error],
                    [
                        self._translation(
                            locale, "Background Music Validation Failed"
                        )
                    ],
                )
                self.assertEqual(len(app.get("audio")), 0)

    def test_sonilo_source_shows_masked_prefilled_key_and_optional_prompt(self):
        """Al elegir Sonilo debe rellenarse la clave local, manteniendo el modo de
        contraseña.
        """
        for locale in TEST_LOCALES:
            with self.subTest(locale=locale):
                test_config = dict(config.app, sonilo_api_key="saved-test-key")
                with (
                    patch.object(config, "app", test_config),
                    patch.object(config, "save_config"),
                ):
                    app = self._open_sonilo_bgm_panel(locale)

                api_key_input = self._widget_by_key(
                    app.text_input, "sonilo_api_key_input"
                )
                prompt_input = self._widget_by_key(
                    app.text_input, "sonilo_bgm_prompt_input"
                )
                self.assertEqual(api_key_input.value, "saved-test-key")
                self.assertEqual(
                    api_key_input.label,
                    self._translation(locale, "Sonilo API Key"),
                )
                self.assertIn("platform.sonilo.com", api_key_input.label)
                # element.type de AppTest indica el tipo de control (text_input); el
                # modo contraseña se guarda en un enum del protobuf subyacente, y hay
                # que comprobar ese campo para verificar el renderizado real.
                self.assertEqual(
                    api_key_input.proto.type, api_key_input.proto.PASSWORD
                )
                self.assertFalse(getattr(api_key_input.proto, "help", ""))
                self.assertEqual(prompt_input.value, "")
                self.assertEqual([str(item.value) for item in app.exception], [])

    def test_sonilo_connection_button_reports_success(self):
        test_config = dict(config.app, sonilo_api_key="saved-test-key")
        with (
            patch.object(config, "app", test_config),
            patch.object(config, "save_config"),
            patch.object(sonilo, "test_connection", return_value={}) as connection,
        ):
            app = self._open_sonilo_bgm_panel("es")
            button = self._widget_by_key(
                app.button, "test_sonilo_connection_button"
            )
            button.click().run()

        connection.assert_called_once_with()
        self.assertIn(
            self._translation("es", "Sonilo Connection Test Succeeded"),
            [item.value for item in app.success],
        )

    def test_zero_volume_does_not_require_sonilo_key(self):
        """Con el volumen de Sonilo a 0, la WebUI no debe seguir mostrando el aviso de
        clave API obligatoria.
        """
        test_config = dict(config.app, sonilo_api_key="")
        # El volumen de la BGM es ahora una preferencia persistente del usuario. Se fija
        # explícitamente una condición inicial distinta de cero para este test, para que
        # los valores guardados por otras sesiones de AppTest no afecten a la aserción
        # previa.
        test_ui = dict(config.ui, bgm_volume=0.2)
        required_warning = self._translation("es", "Sonilo API Key Required")
        with (
            patch.object(config, "app", test_config),
            patch.object(config, "ui", test_ui),
            patch.object(config, "save_config"),
            patch.object(sonilo, "is_enabled", return_value=False),
        ):
            app = self._open_sonilo_bgm_panel("es")
            self.assertIn(required_warning, [item.value for item in app.warning])
            self._volume_select(app).set_value(0.0).run()

        self.assertNotIn(required_warning, [item.value for item in app.warning])
        self.assertEqual([str(item.value) for item in app.exception], [])

    def test_elevenlabs_source_reuses_masked_tts_key_and_shows_prompt(self):
        """La música y el TTS deben compartir la clave, manteniendo la entrada de
        contraseña y una configuración de modelo musical independiente.
        """
        for locale in TEST_LOCALES:
            with self.subTest(locale=locale):
                test_config = dict(
                    config.elevenlabs,
                    api_key="saved-elevenlabs-key",
                    model_id="eleven_multilingual_v2",
                    music_model_id="music_v2",
                )
                with (
                    patch.object(config, "elevenlabs", test_config),
                    patch.object(config, "save_config"),
                ):
                    app = self._open_elevenlabs_bgm_panel(locale)

                api_key_input = self._widget_by_key(
                    app.text_input, "elevenlabs_api_key_input"
                )
                prompt_input = self._widget_by_key(
                    app.text_input, "elevenlabs_music_prompt_input"
                )
                self.assertEqual(api_key_input.value, "saved-elevenlabs-key")
                self.assertEqual(
                    api_key_input.label,
                    self._translation(locale, "ElevenLabs Music API Key"),
                )
                self.assertIn(
                    "elevenlabs.io/app/settings/api-keys",
                    api_key_input.label,
                )
                self.assertEqual(
                    api_key_input.proto.type, api_key_input.proto.PASSWORD
                )
                self.assertFalse(getattr(api_key_input.proto, "help", ""))
                self.assertEqual(prompt_input.value, "")
                self.assertEqual(
                    test_config["model_id"], "eleven_multilingual_v2"
                )
                self.assertEqual([str(item.value) for item in app.exception], [])

    def test_elevenlabs_tts_and_music_share_one_api_key_widget(self):
        """Con locución y música activadas a la vez solo puede haber un estado de clave, y
        tras modificarlo no debe sobrescribirlo el valor antiguo.
        """
        test_config = dict(config.elevenlabs, api_key="key-A")
        test_ui = dict(config.ui, voice_mode="tts")
        with (
            patch.object(config, "elevenlabs", test_config),
            patch.object(config, "ui", test_ui),
            patch.object(config, "save_config"),
            patch.object(voice, "get_elevenlabs_voices", return_value=[]),
        ):
            app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
            app.session_state["ui_language"] = "es"
            app.run()
            self._widget_by_key(
                app.selectbox, "tts_server_select"
            ).set_value("elevenlabs").run()
            self._widget_by_key(
                app.selectbox, "bgm_type_select"
            ).set_value("elevenlabs").run()

            shared_inputs = [
                item
                for item in app.text_input
                if str(getattr(item, "key", "")).startswith(
                    "elevenlabs_api_key_input"
                )
            ]
            self.assertEqual(len(shared_inputs), 1)
            self.assertEqual(shared_inputs[0].value, "key-A")
            self.assertFalse(
                any(
                    str(getattr(item, "key", "")).startswith(
                        "elevenlabs_music_api_key_input"
                    )
                    for item in app.text_input
                )
            )

            shared_inputs[0].set_value("key-B").run()
            updated_input = self._widget_by_key(
                app.text_input, "elevenlabs_api_key_input"
            )
            self.assertEqual(updated_input.value, "key-B")
            self.assertEqual(test_config["api_key"], "key-B")

        self.assertEqual([str(item.value) for item in app.exception], [])

    def test_elevenlabs_connection_button_reports_success(self):
        test_config = dict(config.elevenlabs, api_key="saved-test-key")
        with (
            patch.object(config, "elevenlabs", test_config),
            patch.object(config, "save_config"),
            patch.object(
                elevenlabs_music, "test_connection", return_value={}
            ) as connection,
        ):
            app = self._open_elevenlabs_bgm_panel("es")
            button = self._widget_by_key(
                app.button, "test_elevenlabs_music_connection_button"
            )
            button.click().run()

        connection.assert_called_once_with()
        self.assertIn(
            self._translation(
                "es", "ElevenLabs Connection Test Succeeded"
            ),
            [item.value for item in app.success],
        )

    def test_elevenlabs_connection_reports_paid_plan_requirement(self):
        """El error del plan gratuito debe usar lenguaje natural en el idioma de la
        interfaz, en lugar de mostrar directamente la excepción en inglés.
        """
        for locale in TEST_LOCALES:
            with self.subTest(locale=locale):
                test_config = dict(
                    config.elevenlabs, api_key="saved-test-key"
                )
                with (
                    patch.object(config, "elevenlabs", test_config),
                    patch.object(config, "save_config"),
                    patch.object(
                        elevenlabs_music,
                        "test_connection",
                        side_effect=(
                            elevenlabs_music.ElevenLabsPaidPlanRequiredError(
                                "paid plan required"
                            )
                        ),
                    ),
                ):
                    app = self._open_elevenlabs_bgm_panel(locale)
                    button = self._widget_by_key(
                        app.button,
                        "test_elevenlabs_music_connection_button",
                    )
                    button.click().run()

                self.assertIn(
                    self._translation(
                        locale, "ElevenLabs Paid Plan Required"
                    ),
                    [item.value for item in app.error],
                )

    def test_zero_volume_does_not_require_elevenlabs_key(self):
        """Con el volumen de ElevenLabs a 0 tampoco debe exigirse la clave ni llamarse al
        servicio de pago.
        """
        test_config = dict(config.elevenlabs, api_key="")
        required_warning = self._translation(
            "es", "ElevenLabs API Key Required"
        )
        with (
            patch.object(config, "elevenlabs", test_config),
            patch.object(config, "save_config"),
            patch.object(
                elevenlabs_music, "is_enabled", return_value=False
            ),
        ):
            app = self._open_elevenlabs_bgm_panel("es")
            self.assertIn(required_warning, [item.value for item in app.warning])
            self._volume_select(app).set_value(0.0).run()

        self.assertNotIn(required_warning, [item.value for item in app.warning])
        self.assertEqual([str(item.value) for item in app.exception], [])


if __name__ == "__main__":
    unittest.main()
