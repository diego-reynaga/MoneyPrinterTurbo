import errno
import re
import threading
import time
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import toml
import pytest

from app.config import config
from app.models.llm_provider import LLM_PROVIDER_REGISTRY, get_llm_provider


class TestConfigPersistence:
    @staticmethod
    def _wait_for_deferred_flush(timeout=1):
        """Espera a que termine el hilo de refresco de configuración, para que los tests
        concurrentes no compartan estado en segundo plano.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with config._pending_config_lock:
                if not config._pending_config_flush_scheduled:
                    return
            time.sleep(0.005)
        raise AssertionError("deferred config flush did not finish")

    @staticmethod
    def _load_example_config():
        config_path = Path(__file__).resolve().parents[2] / "config.example.toml"
        return tomllib.loads(config_path.read_text(encoding="utf-8"))

    def test_example_config_documents_runtime_settings(self):
        """La configuración de ejemplo debe mostrar los servicios, el material y los
        parámetros avanzados de ejecución que el usuario mantiene a mano.
        """
        example_config = self._load_example_config()
        app_config = example_config["app"]

        assert example_config["listen_host"] == "0.0.0.0"
        assert example_config["listen_port"] == 8080
        assert example_config["log_level"] == "DEBUG"
        assert app_config["video_source"] in {
            "pexels",
            "pixabay",
            "coverr",
            "volcengine_seedance",
            "ofox",
            "metaso_minimax",
            "muapi",
            "loomloom",
            "local",
        }
        assert "match_materials_to_script" in app_config
        assert app_config["script_generation_backend"] == "local"
        assert app_config["loomloom_api_token"] == ""
        assert app_config["loomloom_video_run_timeout_seconds"] == 1800
        assert app_config["volcengine_seedance_api_key"] == ""
        assert app_config["volcengine_seedance_base_url"].startswith("https://")
        assert app_config["volcengine_seedance_model"]
        assert app_config["ofox_api_key"] == ""
        assert app_config["ofox_base_url"].startswith("https://")
        assert app_config["ofox_text_to_video_model"]
        assert app_config["metaso_minimax_api_key"] == ""
        assert app_config["metaso_minimax_base_url"].startswith("https://")
        assert app_config["metaso_minimax_resolution"] in {"768P", "2K"}
        assert app_config["muapi_api_key"] == ""
        assert app_config["muapi_base_url"].startswith("https://")
        assert app_config["muapi_video_endpoint"]
        assert app_config["muapi_resolution"]
        assert app_config["muapi_min_duration"] == 3
        assert app_config["muapi_max_duration"] == 12
        assert "loomloom_market_listing_id" not in app_config
        assert "loomloom_video_market_listing_id" not in app_config
        assert app_config["shengsuanyun_api_key"] == ""
        assert example_config["voxcpm"] == {
            "base_url": "https://api.modelbest.cn/v1",
            "api_key": "",
            "model_id": "",
            "voice_id": "default",
        }
        # Video-to-Music lee music_base_url de [elevenlabs] y, a diferencia de la clave
        # API, no tiene respaldo por variable de entorno: si el ejemplo lo omite no hay
        # forma documentada de enrutar el endpoint a través de un proxy.
        assert example_config["elevenlabs"] == {
            "api_key": "",
            "model_id": "eleven_multilingual_v2",
            "music_base_url": "https://api.elevenlabs.io",
            "music_model_id": "music_v2",
            "music_timeout": 600,
        }
        assert example_config["whisper"]["device"] == "cpu"

    def test_example_config_covers_llm_provider_registry(self):
        """Los campos configurables de proveedores del registro deben poder descubrirse en
        el archivo de ejemplo.
        """
        app_config = self._load_example_config()["app"]

        for provider in LLM_PROVIDER_REGISTRY:
            if provider.show_api_key:
                assert provider.config_key("api_key") in app_config
            if provider.show_base_url:
                assert provider.config_key("base_url") in app_config
            if provider.requires_model_name:
                assert provider.config_key("model_name") in app_config
            for field in provider.extra_fields:
                assert provider.config_key(field.config_suffix) in app_config

    def test_example_ui_block_documents_every_pinnable_default(self):
        """El bloque de comentarios de [ui] debe enumerar todas las claves que el código
        lee realmente de [ui].

        Ese bloque es la única forma de que el usuario fije a mano los valores
        iniciales, así que enumerar solo algunas claves haría que los ajustes dejen de
        funcionar en silencio: el usuario escribió [ui].subtitle_display_mode siguiendo
        el ejemplo, pero como el ejemplo no lo mencionaba ni sabía que existía. La
        aserción se deriva del código fuente, así que fallará primero al añadir nuevos
        campos fijables.
        """
        project_root = Path(__file__).resolve().parents[2]
        ui_block = (project_root / "config.example.toml").read_text(
            encoding="utf-8"
        ).split("[ui]", 1)[1]
        documented = set(re.findall(r"^#\s*([a-z_]+)\s*=", ui_block, re.MULTILINE))

        schema_text = (project_root / "app/models/schema.py").read_text(
            encoding="utf-8"
        )
        pinnable = set(re.findall(r'config\.ui\.get\(\s*"([a-z_]+)"', schema_text))
        pinnable |= set(
            re.findall(r'_get_valid_ui_choice\(\s*"([a-z_]+)"', schema_text)
        )
        cli_text = (project_root / "cli.py").read_text(encoding="utf-8")
        pinnable |= set(
            re.findall(r'_ui_config_value\(\s*\w+\s*,\s*"([a-z_]+)"', cli_text)
        )

        assert pinnable <= documented, sorted(pinnable - documented)

    def test_load_config_accepts_repeated_utf8_bom_without_rewriting_file(self):
        """Un BOM repetido no debe impedir arrancar a los usuarios de Windows ni reescribir
        la configuración existente.
        """
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            original_content = b"\xef\xbb\xbf\xef\xbb\xbf[app]\nvideo_source = \"pexels\"\n"
            config_path.write_bytes(original_content)

            with patch.object(config, "config_file", str(config_path)):
                loaded_config = config.load_config()

            assert loaded_config["app"]["video_source"] == "pexels"
            assert config_path.read_bytes() == original_content

    def test_load_config_still_rejects_invalid_toml_after_bom_normalization(self):
        """La compatibilidad con BOM no debe ocultar errores de sintaxis reales; al fallar
        debe quedar un log de diagnóstico claro.
        """
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text("[app\nvideo_source = \"pexels\"\n", encoding="utf-8")

            with (
                patch.object(config, "config_file", str(config_path)),
                patch.object(config.logger, "error") as error_mock,
            ):
                try:
                    config.load_config()
                except toml.TomlDecodeError:
                    pass
                else:
                    raise AssertionError("expected invalid TOML to be rejected")

            error_message = str(error_mock.call_args.args[0])
            assert str(config_path) in error_message
            assert "TomlDecodeError" in error_message

    def test_load_config_recovers_empty_config_directory(self):
        """Docker puede crear un directorio vacío en la ruta de un bind mount de archivo
        que falta.
        """
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.toml"
            config_path.mkdir()
            (root / "config.example.toml").write_text(
                '[app]\nvideo_source = "pexels"\n', encoding="utf-8"
            )

            with (
                patch.object(config, "root_dir", temp_dir),
                patch.object(config, "config_file", str(config_path)),
            ):
                loaded_config = config.load_config()

            assert loaded_config["app"]["video_source"] == "pexels"
            assert config_path.is_file()

    def test_load_config_preserves_nonempty_config_directory(self):
        """Un config.toml que por error es un directorio no debe borrarse de forma
        recursiva.
        """
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.toml"
            config_path.mkdir()
            sentinel = config_path / "user-data.txt"
            sentinel.write_text("keep this file", encoding="utf-8")
            (root / "config.example.toml").write_text(
                '[app]\nvideo_source = "pexels"\n', encoding="utf-8"
            )

            with (
                patch.object(config, "root_dir", temp_dir),
                patch.object(config, "config_file", str(config_path)),
            ):
                with pytest.raises(IsADirectoryError, match="config.toml"):
                    config.load_config()

            assert sentinel.read_text(encoding="utf-8") == "keep this file"

    def test_kimi_uses_current_default_model(self):
        """Si Kimi no tiene configurado un modelo propio, debe usarse el modelo por defecto
        de la versión publicada actual.
        """
        provider = get_llm_provider("moonshot")

        assert provider is not None
        assert provider.resolve_model_name("") == "kimi-k3"

    def test_upload_post_settings_belong_to_app_section(self):
        """La configuración de publicación debe estar en el nodo app, para que el archivo
        de ejemplo coincida con la ruta que se lee en tiempo de ejecución.
        """
        example_config = self._load_example_config()
        upload_post_keys = {
            "upload_post_enabled",
            "upload_post_api_key",
            "upload_post_username",
            "upload_post_platforms",
            "upload_post_auto_upload",
            "upload_post_youtube_privacy_status",
            "upload_post_max_pending_tasks",
        }

        assert upload_post_keys <= example_config["app"].keys()
        assert upload_post_keys.isdisjoint(example_config.get("ui", {}).keys())

    def test_save_config_uses_parseable_atomic_output(self):
        """
        El guardado de la configuración escribe primero un archivo temporal y luego lo
        sustituye de forma atómica. El test también confirma que la salida sigue siendo
        TOML válido y que tras guardar correctamente no queda ningún temporal en el
        directorio de configuración.
        """
        original_cfg = dict(config._cfg)
        original_app = dict(config.app)
        try:
            with TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "config.toml"
                config.app["atomic_save_test"] = "ok"
                with (
                    patch.object(config, "root_dir", temp_dir),
                    patch.object(config, "config_file", str(config_path)),
                ):
                    config.save_config()

                saved_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
                assert saved_config["app"]["atomic_save_test"] == "ok"
                assert list(Path(temp_dir).glob(".config-*.toml.tmp")) == []
        finally:
            config.app.clear()
            config.app.update(original_app)
            config._cfg.clear()
            config._cfg.update(original_cfg)

    def test_save_config_falls_back_for_bind_mounted_file(self):
        """
        El punto de montaje de un solo archivo de Docker Desktop no puede sustituirse
        con os.replace. Ante EBUSY debe sobrescribirse en el sitio dentro del bloqueo,
        asegurando que el contenido final esté completo, se pueda analizar y no deje
        temporales.
        """
        original_cfg = dict(config._cfg)
        original_app = dict(config.app)
        try:
            with TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "config.toml"
                config_path.write_text("[app]\nold_value = true\n", encoding="utf-8")
                config.app["bind_mount_save_test"] = "ok"

                with (
                    patch.object(config, "root_dir", temp_dir),
                    patch.object(config, "config_file", str(config_path)),
                    patch.object(
                        config.os,
                        "replace",
                        side_effect=OSError(
                            errno.EBUSY,
                            "Device or resource busy",
                        ),
                    ),
                    patch.object(config.logger, "warning") as warning_mock,
                ):
                    config.save_config()

                saved_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
                assert saved_config["app"]["bind_mount_save_test"] == "ok"
                assert list(Path(temp_dir).glob(".config-*.toml.tmp")) == []
                warning_mock.assert_called_once()
        finally:
            config.app.clear()
            config.app.update(original_app)
            config._cfg.clear()
            config._cfg.update(original_cfg)

    def test_save_config_does_not_hide_other_replace_errors(self):
        """Los errores que no son EBUSY deben seguir lanzándose; no se puede disfrazar un
        fallo de permisos o de disco de guardado correcto.
        """
        original_cfg = dict(config._cfg)
        original_app = dict(config.app)
        try:
            with TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "config.toml"
                config_path.write_text("[app]\nold_value = true\n", encoding="utf-8")
                config.app["replace_error_test"] = "not-saved"

                with (
                    patch.object(config, "root_dir", temp_dir),
                    patch.object(config, "config_file", str(config_path)),
                    patch.object(
                        config.os,
                        "replace",
                        side_effect=OSError(errno.EACCES, "Permission denied"),
                    ),
                ):
                    try:
                        config.save_config()
                    except OSError as exc:
                        assert exc.errno == errno.EACCES
                    else:
                        raise AssertionError("expected config save to fail")

                saved_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
                assert saved_config["app"]["old_value"] is True
                assert list(Path(temp_dir).glob(".config-*.toml.tmp")) == []
        finally:
            config.app.clear()
            config.app.update(original_app)
            config._cfg.clear()
            config._cfg.update(original_cfg)

    def test_runtime_config_lock_blocks_concurrent_config_writes(self):
        """Cuando una tarea larga tiene el bloqueo de ejecución, otras sesiones no pueden
        reescribir la configuración global a mitad de la tarea.
        """
        write_started = threading.Event()
        write_finished = threading.Event()

        def update_config():
            write_started.set()
            config.app["runtime_lock_test"] = "updated"
            write_finished.set()

        config.app.pop("runtime_lock_test", None)
        with config.runtime_config_lock():
            worker = threading.Thread(target=update_config)
            worker.start()
            assert write_started.wait(timeout=1)
            assert not write_finished.wait(timeout=0.05)

        worker.join(timeout=1)
        assert write_finished.is_set()
        config.app.pop("runtime_lock_test", None)

    def test_runtime_config_lock_allows_idempotent_page_writes(self):
        """Al refrescar la página durante la generación, escribir de nuevo el mismo valor
        de un control no debe bloquear el renderizado de toda la página.
        """
        key = "runtime_lock_idempotent_test"
        config.app[key] = "unchanged"
        write_finished = threading.Event()

        def write_same_value():
            config.app[key] = "unchanged"
            assert config.app.setdefault(key, "other") == "unchanged"
            config.app.update({key: "unchanged"})
            assert config.app.pop("runtime_lock_missing_key", None) is None
            write_finished.set()

        with config.runtime_config_lock():
            worker = threading.Thread(target=write_same_value)
            worker.start()
            assert write_finished.wait(timeout=0.2)

        worker.join(timeout=1)
        assert config.app[key] == "unchanged"
        config.app.pop(key, None)

    def test_try_runtime_config_lock_returns_immediately_when_busy(self):
        """El bloqueo de vista previa no puede esperar a que una tarea larga libere la
        configuración global; si está ocupado, la interfaz debe pedir de inmediato que
        se reintente.
        """
        attempted = threading.Event()
        result = []

        def try_lock():
            with config.try_runtime_config_lock() as acquired:
                result.append(acquired)
            attempted.set()

        with config.runtime_config_lock():
            worker = threading.Thread(target=try_lock)
            worker.start()
            assert attempted.wait(timeout=0.2)

        worker.join(timeout=1)
        assert result == [False]

        with config.try_runtime_config_lock() as acquired:
            assert acquired is True

    def test_nonblocking_update_is_applied_after_runtime_task_finishes(self):
        """Los cambios de la WebUI no pueden esperar a las tareas largas, y al terminar la
        tarea deben aplicarse y guardarse los valores más recientes.
        """
        key = "nonblocking_runtime_update_test"
        original_value = config.app.get(key, config._MISSING)
        update_finished = threading.Event()
        update_result = []

        def update_config():
            update_result.append(
                config.update_config_nonblocking(config.app, key, "updated")
            )
            update_finished.set()

        try:
            with patch.object(config, "save_config") as save_config:
                with config.runtime_config_lock():
                    worker = threading.Thread(target=update_config)
                    worker.start()
                    assert update_finished.wait(timeout=0.2)
                    assert update_result == [False]
                    assert config.app.get(key) != "updated"

                worker.join(timeout=1)
                assert config.app[key] == "updated"
                save_config.assert_called_once()
        finally:
            if original_value is config._MISSING:
                config.app.pop(key, None)
            else:
                config.app[key] = original_value

    def test_nonblocking_update_keeps_only_latest_value(self):
        """Si el mismo control se cambia varias veces durante la tarea, solo se aplica la
        última elección.
        """
        key = "nonblocking_latest_value_test"
        original_value = config.app.get(key, config._MISSING)
        updates_finished = threading.Event()

        def update_config():
            assert not config.update_config_nonblocking(config.app, key, "first")
            assert not config.update_config_nonblocking(config.app, key, "latest")
            updates_finished.set()

        try:
            with patch.object(config, "save_config"):
                with config.runtime_config_lock():
                    worker = threading.Thread(target=update_config)
                    worker.start()
                    assert updates_finished.wait(timeout=0.2)

                worker.join(timeout=1)
                assert config.app[key] == "latest"
        finally:
            if original_value is config._MISSING:
                config.app.pop(key, None)
            else:
                config.app[key] = original_value

    def test_nonblocking_delete_is_applied_after_runtime_task_finishes(self):
        """Al volver a la opción por defecto, borrar la configuración tampoco debe bloquear
        la tarea de vídeo en ejecución.
        """
        key = "nonblocking_runtime_delete_test"
        config.app[key] = "custom"
        delete_finished = threading.Event()
        delete_result = []

        def delete_config():
            delete_result.append(config.delete_config_nonblocking(config.app, key))
            delete_finished.set()

        try:
            with patch.object(config, "save_config") as save_config:
                with config.runtime_config_lock():
                    worker = threading.Thread(target=delete_config)
                    worker.start()
                    assert delete_finished.wait(timeout=0.2)
                    assert delete_result == [False]
                    assert config.app[key] == "custom"

                worker.join(timeout=1)
                assert key not in config.app
                save_config.assert_called_once()
        finally:
            config.app.pop(key, None)

    def test_try_save_config_returns_immediately_while_runtime_task_is_active(self):
        """Cuando un rerun de la página pide guardar, no puede esperar a que la tarea de
        vídeo libere el bloqueo de configuración.
        """
        save_finished = threading.Event()
        save_result = []

        def save_config():
            save_result.append(config.try_save_config())
            save_finished.set()

        with patch.object(config, "save_config") as blocking_save:
            with config.runtime_config_lock():
                worker = threading.Thread(target=save_config)
                worker.start()
                assert save_finished.wait(timeout=0.2)
                assert save_result == [False]

            worker.join(timeout=1)
            blocking_save.assert_called_once()

        self._wait_for_deferred_flush()

    def test_try_runtime_lock_flushes_updates_queued_during_operation(self):
        """Cuando una operación corta libera el bloqueo de configuración, también debe
        aplicar y guardar los cambios de la página que llegaron mientras tanto.
        """
        key = "try_runtime_queued_update_test"
        original_value = config.app.get(key, config._MISSING)
        update_finished = threading.Event()

        def queue_update():
            assert not config.update_config_nonblocking(config.app, key, "updated")
            update_finished.set()

        try:
            with patch.object(config, "save_config") as save_config:
                with config.try_runtime_config_lock() as acquired:
                    assert acquired is True
                    worker = threading.Thread(target=queue_update)
                    worker.start()
                    assert update_finished.wait(timeout=0.2)
                    assert config.app.get(key) != "updated"

                worker.join(timeout=1)
                assert config.app[key] == "updated"
                save_config.assert_called_once()

            self._wait_for_deferred_flush()
        finally:
            if original_value is config._MISSING:
                config.app.pop(key, None)
            else:
                config.app[key] = original_value

    def test_update_queued_during_save_is_flushed_after_lock_release(self):
        """Los cambios nuevos que llegan durante el guardado de salida no pueden quedarse
        en la cola ni ser sobrescritos por valores anteriores.
        """
        key = "late_runtime_update_test"
        original_value = config.app.get(key, config._MISSING)
        runtime_entered = threading.Event()
        release_runtime = threading.Event()
        first_save_started = threading.Event()
        release_first_save = threading.Event()
        second_save_finished = threading.Event()
        save_count = 0
        save_count_lock = threading.Lock()

        def blocking_save():
            nonlocal save_count
            with save_count_lock:
                save_count += 1
                current_save = save_count
            if current_save == 1:
                first_save_started.set()
                assert release_first_save.wait(timeout=1)
            elif current_save == 2:
                second_save_finished.set()

        def hold_runtime_lock():
            with config.runtime_config_lock():
                runtime_entered.set()
                assert release_runtime.wait(timeout=1)

        try:
            with patch.object(config, "save_config", side_effect=blocking_save):
                runtime_worker = threading.Thread(target=hold_runtime_lock)
                runtime_worker.start()
                assert runtime_entered.wait(timeout=1)

                assert not config.update_config_nonblocking(config.app, key, "first")
                release_runtime.set()
                assert first_save_started.wait(timeout=1)

                # La primera ronda de guardado ya ha tomado la instantánea de la
                # configuración; un valor que llegue entonces debe volver a aplicarlo y
                # guardarlo el hilo de refresco en segundo plano cuando se libere el
                # bloqueo, y el resultado final debe ser ese valor.
                assert not config.update_config_nonblocking(config.app, key, "latest")
                release_first_save.set()

                runtime_worker.join(timeout=1)
                assert not runtime_worker.is_alive()
                assert second_save_finished.wait(timeout=1)
                assert config.app[key] == "latest"
                assert save_count == 2

            self._wait_for_deferred_flush()
        finally:
            release_runtime.set()
            release_first_save.set()
            if original_value is config._MISSING:
                config.app.pop(key, None)
            else:
                config.app[key] = original_value

    def test_config_snapshot_includes_pending_updates(self):
        """Mientras la generación de vídeo tiene el bloqueo, una nueva solicitud al LLM
        debe ver la última elección de la interfaz y no la configuración antigua.
        """
        keys = {
            "llm_provider": "pending-provider",
            "pending-provider_api_key": "pending-key",
            "pending-provider_model_name": "pending-model",
        }
        original_values = {key: config.app.get(key, config._MISSING) for key in keys}
        updates_finished = threading.Event()

        def queue_updates():
            for key, value in keys.items():
                assert not config.update_config_nonblocking(config.app, key, value)
            updates_finished.set()

        try:
            with patch.object(config, "save_config"):
                with config.runtime_config_lock():
                    worker = threading.Thread(target=queue_updates)
                    worker.start()
                    assert updates_finished.wait(timeout=0.2)

                    snapshot = config.snapshot_config_with_pending(config.app)
                    assert all(snapshot[key] == value for key, value in keys.items())
                    assert config.app.get("llm_provider") != "pending-provider"

                worker.join(timeout=1)

            self._wait_for_deferred_flush()
        finally:
            for key, original_value in original_values.items():
                if original_value is config._MISSING:
                    config.app.pop(key, None)
                else:
                    config.app[key] = original_value
