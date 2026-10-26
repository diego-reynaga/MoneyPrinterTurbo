import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.models.schema import MaterialInfo, VideoAspect
from app.services import material, material_cache


class TestMaterialSearchCache(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir_patch = patch(
            "app.services.material_cache.utils.storage_dir",
            return_value=self.temp_dir.name,
        )
        self.cache_dir_patch.start()

    def tearDown(self):
        self.cache_dir_patch.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def _item(url: str = "https://example.com/video.mp4") -> MaterialInfo:
        return MaterialInfo(
            provider="pixabay",
            url=url,
            duration=12,
            source_info={
                "provider": "pixabay",
                "search_term": "nature",
                "asset_id": "123",
                "source_page": "https://pixabay.com/videos/example-123/",
                "creator": {
                    "id": "456",
                    "name": "Creator",
                    "profile_page": "https://pixabay.com/users/creator-456/",
                },
                "rendition": {
                    "id": "large",
                    "width": 1080,
                    "height": 1920,
                },
            },
        )

    def _cache_path(self) -> Path:
        return material_cache._cache_path(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )

    def test_cache_round_trip_preserves_material_fields(self):
        """
        La caché en disco debe poder recuperar entre procesos todos los campos que
        necesita MaterialInfo; no puede guardar solo la URL y perder provider o
        duration, porque cambiaría el comportamiento posterior de las descargas y del
        cálculo de duración.
        """
        saved = material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[self._item()],
        )
        loaded = material_cache.load_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )

        self.assertTrue(saved)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].provider, "pixabay")
        self.assertEqual(loaded[0].url, "https://example.com/video.mp4")
        self.assertEqual(loaded[0].duration, 12)
        self.assertEqual(loaded[0].source_info["search_term"], "nature")
        self.assertEqual(loaded[0].source_info["asset_id"], "123")
        self.assertEqual(
            loaded[0].source_info["source_page"],
            "https://pixabay.com/videos/example-123/",
        )
        self.assertEqual(
            loaded[0].source_info["creator"]["profile_page"],
            "https://pixabay.com/users/creator-456/",
        )

    def test_expired_cache_is_removed_and_treated_as_miss(self):
        """
        Pixabay exige reutilizar los resultados de búsqueda como máximo 24 horas. Los
        archivos caducados deben invalidarse y borrarse de inmediato, para que las URL
        de material antiguas no se reutilicen sin límite y el directorio de caché no
        acumule JSON no válidos.
        """
        material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[self._item()],
        )
        cache_path = self._cache_path()
        now = 2_000_000_000.0
        expired_mtime = now - material_cache.MATERIAL_SEARCH_CACHE_TTL_SECONDS - 1
        os.utime(cache_path, (expired_mtime, expired_mtime))

        loaded = material_cache.load_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            now=now,
        )

        self.assertIsNone(loaded)
        self.assertFalse(cache_path.exists())

    def test_future_dated_cache_is_removed_and_treated_as_miss(self):
        """Si la hora del sistema es anómala, una marca de tiempo futura no puede saltarse
        la validez de 24 horas.
        """
        material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[self._item()],
        )
        cache_path = self._cache_path()
        now = 2_000_000_000.0
        future_mtime = now + 60
        os.utime(cache_path, (future_mtime, future_mtime))

        loaded = material_cache.load_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            now=now,
        )

        self.assertIsNone(loaded)
        self.assertFalse(cache_path.exists())

    def test_corrupted_cache_is_removed_without_breaking_search(self):
        """
        Una salida anómala del proceso, un fallo del disco o una edición manual del
        usuario pueden dejar archivos dañados. Si falla la lectura hay que volver a la
        búsqueda remota y limpiar el archivo dañado; una caché no puede bloquear para
        siempre la generación de material.
        """
        cache_path = self._cache_path()
        cache_path.write_text("{invalid-json", encoding="utf-8")

        with patch("app.services.material_cache.logger.warning") as warning:
            loaded = material_cache.load_material_search_cache(
                provider="pixabay",
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            )

        self.assertIsNone(loaded)
        self.assertFalse(cache_path.exists())
        self.assertTrue(warning.called)

    def test_empty_results_are_not_cached(self):
        """
        La interfaz actual de proveedores usa [] tanto para "sin resultados" como para
        "la solicitud falló". Guardar en caché una lista vacía fijaría durante 24 horas
        un bloqueo de Cloudflare o un fallo de red pasajero, así que solo se guardan
        resultados correctos no vacíos.
        """
        saved = material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[],
        )

        self.assertFalse(saved)
        self.assertEqual(list(Path(self.temp_dir.name).iterdir()), [])

    def test_cache_file_does_not_contain_search_parameters_or_credentials(self):
        """
        El nombre del archivo de caché usa un resumen y el contenido solo guarda los
        campos del material. Aunque el usuario comparta el directorio storage, el
        archivo no debe contener palabras clave, claves API ni otra configuración de la
        solicitud.
        """
        item = self._item()
        item.source_info["source_page"] += "?token=drop"
        item.source_info["creator"]["profile_page"] += "?key=drop"
        material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="private search term",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[item],
        )
        cache_files = list(Path(self.temp_dir.name).glob("*.json"))

        self.assertEqual(len(cache_files), 1)
        self.assertNotIn("private search term", cache_files[0].name)
        raw_payload = cache_files[0].read_text(encoding="utf-8")
        payload = json.loads(raw_payload)
        self.assertEqual(set(payload), {"version", "items"})
        self.assertNotIn("private search term", raw_payload)
        self.assertNotIn("token=drop", raw_payload)

    def test_coverr_signed_urls_are_never_cached(self):
        """Las direcciones de descarga de Coverr contienen un JWT firmado y no pueden
        entrar en la caché en disco de larga duración.
        """
        item = self._item(
            "https://storage.coverr.co/video/download?token=signed-jwt"
        )
        item.provider = "coverr"
        item.source_info["provider"] = "coverr"

        saved = material_cache.save_material_search_cache(
            provider="coverr",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[item],
        )

        self.assertFalse(saved)
        self.assertEqual(list(Path(self.temp_dir.name).glob("*.json")), [])

    def test_coverr_cache_load_removes_legacy_signed_url(self):
        """Al acceder a Coverr hay que limpiar la caché de direcciones de descarga firmadas
        que pudieran dejar versiones antiguas.
        """
        cache_path = material_cache._cache_path(
            provider="coverr",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )
        cache_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "items": [
                        {
                            "provider": "coverr",
                            "url": "https://storage.coverr.co/video?token=signed-jwt",
                            "duration": 12,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        loaded = material_cache.load_material_search_cache(
            provider="coverr",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )

        self.assertIsNone(loaded)
        self.assertFalse(cache_path.exists())

    def test_version_one_cache_is_invalidated(self):
        """A la caché antigua le falta la información de origen; tras actualizar hay que
        volver a consultar en lugar de generar registros de tarea incompletos.
        """
        cache_path = self._cache_path()
        cache_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "items": [
                        {
                            "provider": "pixabay",
                            "url": "https://example.com/old.mp4",
                            "duration": 12,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        loaded = material_cache.load_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )

        self.assertIsNone(loaded)
        self.assertFalse(cache_path.exists())

    def test_cache_key_separates_provider_duration_and_aspect(self):
        """
        La fuente de material, la duración mínima y la relación de aspecto cambian los
        resultados de la búsqueda remota; si cambia cualquiera de ellos hay que usar una
        caché distinta, para no devolver al flujo de generación de vídeo material que no
        cumple los requisitos de la tarea actual.
        """
        base_path = material_cache._cache_path(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )
        paths = {
            base_path,
            material_cache._cache_path(
                provider="pexels",
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            ),
            material_cache._cache_path(
                provider="pixabay",
                search_term="nature",
                minimum_duration=10,
                video_aspect=VideoAspect.portrait,
            ),
            material_cache._cache_path(
                provider="pixabay",
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.landscape,
            ),
        }

        self.assertEqual(len(paths), 4)

    def test_search_wrapper_reuses_cache_across_calls(self):
        """
        La primera llamada consulta la búsqueda remota y escribe la caché; la segunda
        con los mismos parámetros debe reutilizar directamente el resultado en disco. Es
        el comportamiento clave para reducir las llamadas a la API de Pixabay y la
        probabilidad de activar el control de riesgos de Cloudflare.
        """
        remote_search = Mock(return_value=[self._item()])

        first = material._search_videos_with_cache(
            provider="pixabay",
            search_videos=remote_search,
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )
        second = material._search_videos_with_cache(
            provider="pixabay",
            search_videos=remote_search,
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )

        self.assertEqual(remote_search.call_count, 1)
        self.assertEqual(first, second)

    def test_search_wrapper_refreshes_mixed_orientation_cache(self):
        """
        La caché anterior a la actualización puede mezclar material de otra orientación.
        Devolver solo las pocas entradas filtradas reduciría la variedad del material,
        así que si cualquier orientación no coincide hay que volver a pedir y sustituir
        todo el conjunto de candidatos.
        """
        portrait_item = self._item("https://example.com/old-portrait.mp4")
        landscape_item = self._item("https://example.com/old-landscape.mp4")
        landscape_item.source_info["rendition"] = {
            "id": "large",
            "width": 1920,
            "height": 1080,
        }
        material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[portrait_item, landscape_item],
        )

        refreshed_item = self._item("https://example.com/refreshed-portrait.mp4")
        remote_search = Mock(return_value=[refreshed_item])
        results = material._search_videos_with_cache(
            provider="pixabay",
            search_videos=remote_search,
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )

        self.assertEqual(remote_search.call_count, 1)
        self.assertEqual(
            [item.url for item in results],
            ["https://example.com/refreshed-portrait.mp4"],
        )
        cached_items = material_cache.load_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )
        self.assertEqual(
            [item.url for item in cached_items],
            ["https://example.com/refreshed-portrait.mp4"],
        )

    def test_square_search_reuses_crop_compatible_cache(self):
        """Las tareas cuadradas deben seguir reutilizando la caché de material recortable,
        sin pedir una y otra vez al servidor porque la orientación original sea
        distinta.
        """
        landscape_item = self._item("https://example.com/landscape.mp4")
        landscape_item.source_info["rendition"] = {
            "id": "large",
            "width": 1920,
            "height": 1080,
        }
        material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.square,
            items=[landscape_item],
        )
        remote_search = Mock(return_value=[])

        results = material._search_videos_with_cache(
            provider="pixabay",
            search_videos=remote_search,
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.square,
        )

        self.assertEqual(remote_search.call_count, 0)
        self.assertEqual(
            [item.url for item in results],
            ["https://example.com/landscape.mp4"],
        )

    def test_search_wrapper_retries_after_empty_result(self):
        """Los resultados vacíos no se guardan en caché, así que la siguiente llamada debe
        volver al servidor y reintentarse sola cuando se recupere un fallo temporal.
        """
        remote_search = Mock(return_value=[])

        for _ in range(2):
            results = material._search_videos_with_cache(
                provider="pixabay",
                search_videos=remote_search,
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            )
            self.assertEqual(results, [])

        self.assertEqual(remote_search.call_count, 2)

    def test_cache_read_failure_falls_back_to_remote_search(self):
        """Una excepción al leer la caché solo puede degradarse a fallo de caché; no puede
        bloquear la búsqueda de material remota.
        """
        remote_items = [self._item()]
        remote_search = Mock(return_value=remote_items)

        with patch.object(
            material_cache,
            "load_material_search_cache",
            side_effect=RuntimeError("cache read failed"),
        ), patch.object(material_cache.logger, "warning") as warning:
            results = material._search_videos_with_cache(
                provider="pixabay",
                search_videos=remote_search,
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            )

        self.assertEqual(results, remote_items)
        self.assertEqual(remote_search.call_count, 1)
        self.assertTrue(warning.called)

    def test_cache_write_failure_keeps_remote_results(self):
        """Si la búsqueda remota tiene éxito, aunque falle la escritura de la caché hay que
        seguir devolviendo el material utilizable.
        """
        remote_items = [self._item()]
        remote_search = Mock(return_value=remote_items)

        with patch.object(
            material_cache,
            "load_material_search_cache",
            return_value=None,
        ), patch.object(
            material_cache,
            "save_material_search_cache",
            side_effect=RuntimeError("cache write failed"),
        ), patch.object(material_cache.logger, "warning") as warning:
            results = material._search_videos_with_cache(
                provider="pixabay",
                search_videos=remote_search,
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            )

        self.assertEqual(results, remote_items)
        self.assertEqual(remote_search.call_count, 1)
        self.assertTrue(warning.called)

    def test_invalid_cache_item_does_not_raise(self):
        """Un objeto de material anómalo no debe permitir que la escritura opcional de la
        caché rompa el flujo principal de quien llama.
        """
        with patch.object(material_cache.logger, "warning") as warning:
            saved = material_cache.save_material_search_cache(
                provider="pixabay",
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
                items=[None],
            )

        self.assertFalse(saved)
        self.assertTrue(warning.called)

    def test_concurrent_identical_searches_share_remote_request(self):
        """
        El servicio API permite varias tareas concurrentes. En la primera búsqueda con
        las mismas condiciones, los hilos que llegan después deben esperar a que el
        primero escriba la caché, en lugar de volver a gastar saldo de la interfaz de
        terceros.
        """
        remote_started = threading.Event()
        allow_remote_finish = threading.Event()
        remote_call_lock = threading.Lock()
        remote_call_count = 0
        results = []

        def remote_search(**_kwargs):
            nonlocal remote_call_count
            with remote_call_lock:
                remote_call_count += 1
            remote_started.set()
            self.assertTrue(allow_remote_finish.wait(timeout=2))
            return [self._item()]

        def run_search():
            results.append(
                material._search_videos_with_cache(
                    provider="pixabay",
                    search_videos=remote_search,
                    search_term="shared nature",
                    minimum_duration=5,
                    video_aspect=VideoAspect.portrait,
                )
            )

        first_thread = threading.Thread(target=run_search)
        second_thread = threading.Thread(target=run_search)
        first_thread.start()
        self.assertTrue(remote_started.wait(timeout=2))
        second_thread.start()
        # Da tiempo al segundo hilo para entrar en la zona de espera del bloqueo de
        # caché, para que el test cubra un fallo de caché concurrente real.
        time.sleep(0.05)
        allow_remote_finish.set()
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)

        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())
        self.assertEqual(remote_call_count, 1)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], results[1])

    def test_cleanup_removes_expired_entries_only(self):
        """La limpieza poco frecuente solo borra caché caducada, sin afectar a la caché
        válida ni a otros archivos del usuario.
        """
        stale_path = self._cache_path()
        stale_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "items": [
                        {
                            "provider": "pixabay",
                            "url": "https://example.com/stale.mp4",
                            "duration": 12,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        fresh_path = material_cache._cache_path(
            provider="pexels",
            search_term="fresh",
            minimum_duration=5,
            video_aspect=VideoAspect.landscape,
        )
        fresh_path.write_text("{}", encoding="utf-8")
        unrelated_path = Path(self.temp_dir.name) / "notes.json"
        unrelated_path.write_text("keep", encoding="utf-8")

        now = 2_000_000_000.0
        stale_mtime = now - material_cache.MATERIAL_SEARCH_CACHE_TTL_SECONDS - 1
        os.utime(stale_path, (stale_mtime, stale_mtime))
        os.utime(fresh_path, (now - 60, now - 60))

        deleted = material_cache.cleanup_expired_material_search_cache(
            now=now,
            force=True,
        )

        self.assertEqual(deleted, 1)
        self.assertFalse(stale_path.exists())
        self.assertTrue(fresh_path.exists())
        self.assertTrue(unrelated_path.exists())

    def test_cleanup_removes_orphaned_temp_files(self):
        """
        Cuando el proceso se termina a la fuerza, el archivo intermedio que deja
        NamedTemporaryFile(delete=False) no llega a consumirlo os.replace ni lo borra el
        respaldo de excepciones. Debe recuperarse junto con la caché caducada; si no,
        cada interrupción anómala dejaría para siempre un archivo residual en el
        directorio de caché.
        """
        orphan_path = Path(self.temp_dir.name) / (
            f".{self._cache_path().stem}-m429jzwe.tmp"
        )
        orphan_path.write_text('{"version":2,"items":[]}', encoding="utf-8")
        unrelated_path = Path(self.temp_dir.name) / "notes.tmp"
        unrelated_path.write_text("keep", encoding="utf-8")

        now = 2_000_000_000.0
        stale_mtime = now - material_cache.MATERIAL_SEARCH_CACHE_TTL_SECONDS - 1
        os.utime(orphan_path, (stale_mtime, stale_mtime))
        os.utime(unrelated_path, (stale_mtime, stale_mtime))

        deleted = material_cache.cleanup_expired_material_search_cache(
            now=now,
            force=True,
        )

        self.assertEqual(deleted, 1)
        self.assertFalse(orphan_path.exists())
        # Los demás archivos cuyo prefijo no coincide son del usuario y la lógica de
        # limpieza no puede borrarlos.
        self.assertTrue(unrelated_path.exists())

    def test_cleanup_keeps_recent_temp_files(self):
        """
        Durante una búsqueda concurrente, otro proceso puede estar escribiendo un
        archivo temporal. La limpieza debe reutilizar el criterio de caducidad de la
        caché, para dejar intactos los archivos intermedios que aún no han caducado y no
        borrar archivos que esperan a os.replace.
        """
        in_flight_path = Path(self.temp_dir.name) / (
            f".{self._cache_path().stem}-m429jzwe.tmp"
        )
        in_flight_path.write_text('{"version":2,"items":', encoding="utf-8")

        now = 2_000_000_000.0
        os.utime(in_flight_path, (now - 1, now - 1))

        deleted = material_cache.cleanup_expired_material_search_cache(
            now=now,
            force=True,
        )

        self.assertEqual(deleted, 0)
        self.assertTrue(in_flight_path.exists())


if __name__ == "__main__":
    unittest.main()
