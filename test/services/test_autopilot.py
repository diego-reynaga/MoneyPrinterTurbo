"""Tests de la publicación automática diaria: plan, persistencia y ejecución de huecos."""

import json
import random
import sys
import threading
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models import const
from app.services import autopilot as ap


def _settings(**overrides) -> ap.AutopilotSettings:
    base = {
        "enabled": True,
        "niche": "finanzas personales",
        "videos_per_day": 3,
        "window_start_hour": 10,
        "window_end_hour": 22,
        "min_gap_minutes": 45,
        "retry_delay_minutes": 15,
        "max_attempts": 2,
    }
    base.update(overrides)
    return ap.AutopilotSettings(**base)


class TestSettings(unittest.TestCase):
    def test_defaults_match_three_videos_between_ten_and_ten(self):
        with patch.dict(ap.config.autopilot, {}, clear=True):
            settings = ap.load_settings()
        self.assertFalse(settings.enabled)
        self.assertEqual(settings.videos_per_day, 3)
        self.assertEqual(settings.window_start_hour, 10)
        self.assertEqual(settings.window_end_hour, 22)
        self.assertEqual(ap.describe_window(settings), "10:00-22:00")

    def test_example_config_section_loads(self):
        import toml

        example = toml.load(Path(__file__).parent.parent.parent / "config.example.toml")
        section = example["autopilot"]
        with patch.dict(ap.config.autopilot, section, clear=True):
            settings = ap.load_settings()
        self.assertFalse(settings.enabled)
        self.assertEqual(settings.videos_per_day, 3)
        self.assertEqual(settings.window_start_hour, 10)
        self.assertEqual(settings.window_end_hour, 22)

    def test_string_booleans_are_rejected(self):
        with patch.dict(ap.config.autopilot, {"enabled": "false"}, clear=True):
            with self.assertRaises(ap.AutopilotConfigError) as error:
                ap.load_settings()
        self.assertIn("autopilot.enabled", str(error.exception))

    def test_subjects_must_be_a_list(self):
        with patch.dict(ap.config.autopilot, {"subjects": "un tema"}, clear=True):
            with self.assertRaises(ap.AutopilotConfigError):
                ap.load_settings()

    def test_out_of_range_values_are_rejected(self):
        with patch.dict(ap.config.autopilot, {"videos_per_day": 0}, clear=True):
            with self.assertRaises(ap.AutopilotConfigError):
                ap.load_settings()

    def test_window_must_be_ordered(self):
        with patch.dict(
            ap.config.autopilot,
            {"window_start_hour": 22, "window_end_hour": 10},
            clear=True,
        ):
            with self.assertRaises(ap.AutopilotConfigError):
                ap.load_settings()

    def test_overrides_apply_after_the_file(self):
        with patch.dict(ap.config.autopilot, {"videos_per_day": 3}, clear=True):
            settings = ap.load_settings(overrides={"videos_per_day": 5, "enabled": True})
        self.assertEqual(settings.videos_per_day, 5)
        self.assertTrue(settings.enabled)

    def test_unknown_overrides_are_rejected(self):
        with patch.dict(ap.config.autopilot, {}, clear=True):
            with self.assertRaises(ap.AutopilotConfigError):
                ap.load_settings(overrides={"videos": 2})

    def test_preflight_lists_every_missing_piece(self):
        service = MagicMock()
        service.enabled = False
        service.api_key = ""
        service.username = ""
        service.platforms = []
        with patch("app.services.upload_post.upload_post_service", service):
            problems = ap.preflight(_settings(enabled=False, niche="", subjects=()))
        self.assertEqual(len(problems), 6)
        self.assertTrue(any("autopilot.enabled" in problem for problem in problems))
        self.assertTrue(any("nicho" in problem for problem in problems))

    def test_preflight_rejects_cli_options_the_video_cli_does_not_accept(self):
        service = MagicMock()
        service.enabled = True
        service.api_key = "key"
        service.username = "usuario"
        service.platforms = ["tiktok"]
        with patch("app.services.upload_post.upload_post_service", service):
            problems = ap.preflight(_settings(cli_options=("--no-existe",)))
        self.assertEqual(len(problems), 1)
        self.assertIn("cli_options", problems[0])

    def test_preflight_accepts_valid_cli_options(self):
        service = MagicMock()
        service.enabled = True
        service.api_key = "key"
        service.username = "usuario"
        service.platforms = ["tiktok"]
        with patch("app.services.upload_post.upload_post_service", service):
            problems = ap.preflight(
                _settings(cli_options=("--video-source", "pexels"))
            )
        self.assertEqual(problems, [])

    def test_preflight_passes_with_credentials_and_niche(self):
        service = MagicMock()
        service.enabled = True
        service.api_key = "key"
        service.username = "usuario"
        service.platforms = ["tiktok"]
        with patch("app.services.upload_post.upload_post_service", service):
            self.assertEqual(ap.preflight(_settings()), [])


class TestSchedulePlanning(unittest.TestCase):
    def test_times_stay_inside_the_window_and_respect_the_gap(self):
        settings = _settings()
        rng = random.Random(1234)
        for day_offset in range(60):
            day = date(2026, 1, 1) + timedelta(days=day_offset)
            times = ap.plan_slot_times(day, settings, rng=rng)
            start, end = ap.window_bounds(day, settings)
            self.assertEqual(len(times), 3)
            self.assertEqual(times, sorted(times))
            for moment in times:
                self.assertGreaterEqual(moment, start)
                self.assertLessEqual(moment, end)
            for earlier, later in zip(times, times[1:]):
                self.assertGreaterEqual(
                    (later - earlier).total_seconds() / 60, settings.min_gap_minutes
                )

    def test_times_differ_from_day_to_day(self):
        settings = _settings()
        rng = random.Random(7)
        plans = {
            tuple(
                moment.strftime("%H:%M")
                for moment in ap.plan_slot_times(
                    date(2026, 3, 1) + timedelta(days=offset), settings, rng=rng
                )
            )
            for offset in range(14)
        }
        self.assertGreater(len(plans), 10)

    def test_times_spread_over_the_whole_window(self):
        settings = _settings()
        rng = random.Random(99)
        hours = {
            moment.hour
            for offset in range(80)
            for moment in ap.plan_slot_times(
                date(2026, 5, 1) + timedelta(days=offset), settings, rng=rng
            )
        }
        self.assertLessEqual(hours, set(range(10, 23)))
        self.assertGreaterEqual(len(hours), 10)

    def test_gap_shrinks_when_it_does_not_fit(self):
        settings = _settings(videos_per_day=12, min_gap_minutes=120)
        times = ap.plan_slot_times(date(2026, 2, 2), settings, rng=random.Random(3))
        self.assertEqual(len(times), 12)
        _, end = ap.window_bounds(date(2026, 2, 2), settings)
        self.assertLessEqual(times[-1], end)

    def test_reserve_minutes_pulls_the_last_start_earlier(self):
        settings = _settings(reserve_minutes=60)
        times = ap.plan_slot_times(date(2026, 2, 2), settings, rng=random.Random(5))
        _, end = ap.window_bounds(date(2026, 2, 2), settings)
        self.assertLessEqual(times[-1], end - timedelta(minutes=60))

    def test_planning_mid_window_only_uses_the_remaining_hours(self):
        settings = _settings()
        day = date(2026, 4, 10)
        times = ap.plan_slot_times(
            day,
            settings,
            rng=random.Random(11),
            not_before=datetime(2026, 4, 10, 15, 30),
        )
        self.assertEqual(len(times), 3)
        for moment in times:
            self.assertGreaterEqual(moment, datetime(2026, 4, 10, 15, 30))

    def test_no_slots_when_the_window_is_already_over(self):
        settings = _settings()
        times = ap.plan_slot_times(
            date(2026, 4, 10),
            settings,
            rng=random.Random(11),
            not_before=datetime(2026, 4, 10, 23, 30),
        )
        self.assertEqual(times, [])

    def test_single_video_per_day_is_supported(self):
        settings = _settings(videos_per_day=1)
        times = ap.plan_slot_times(date(2026, 6, 6), settings, rng=random.Random(2))
        self.assertEqual(len(times), 1)


class TestStateFile(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._directory = tempfile.TemporaryDirectory()
        self.path = str(Path(self._directory.name) / "state.json")
        self.addCleanup(self._directory.cleanup)

    def test_missing_file_starts_empty(self):
        state = ap.load_state(self.path)
        self.assertEqual(state["slots"], [])
        self.assertEqual(state["date"], "")

    def test_round_trip(self):
        state = ap.empty_state()
        state["date"] = "2026-10-26"
        state["slots"] = [ap._new_slot(1, datetime(2026, 10, 26, 11, 0))]
        ap.save_state(self.path, state)
        self.assertEqual(ap.load_state(self.path)["slots"][0]["at"], "2026-10-26T11:00:00")

    def test_corrupt_file_falls_back_to_a_fresh_plan(self):
        Path(self.path).write_text("{ esto no es json", encoding="utf-8")
        state = ap.load_state(self.path)
        self.assertEqual(state["slots"], [])

    def test_unexpected_shape_falls_back_to_a_fresh_plan(self):
        Path(self.path).write_text(json.dumps({"slots": "no es una lista"}), encoding="utf-8")
        self.assertEqual(ap.load_state(self.path)["slots"], [])

    def test_save_leaves_no_temporary_files(self):
        ap.save_state(self.path, ap.empty_state())
        self.assertEqual(list(Path(self._directory.name).glob(".autopilot-*")), [])

    def test_plan_day_is_idempotent(self):
        settings = _settings()
        state = ap.empty_state()
        now = datetime(2026, 10, 26, 9, 0)
        ap.plan_day(settings, state, now.date(), rng=random.Random(1), now=now)
        first = [slot["at"] for slot in state["slots"]]
        ap.plan_day(settings, state, now.date(), rng=random.Random(2), now=now)
        self.assertEqual([slot["at"] for slot in state["slots"]], first)

    def test_a_new_day_gets_a_new_plan(self):
        settings = _settings()
        state = ap.empty_state()
        ap.plan_day(
            settings,
            state,
            date(2026, 10, 26),
            rng=random.Random(1),
            now=datetime(2026, 10, 26, 9, 0),
        )
        state["slots"][0]["status"] = ap.SLOT_STATE_DONE
        ap.plan_day(
            settings,
            state,
            date(2026, 10, 27),
            rng=random.Random(1),
            now=datetime(2026, 10, 27, 9, 0),
        )
        self.assertEqual(state["date"], "2026-10-27")
        self.assertTrue(
            all(slot["status"] == ap.SLOT_STATE_PENDING for slot in state["slots"])
        )

    def test_a_day_without_slots_is_not_replanned(self):
        settings = _settings()
        state = ap.empty_state()
        late = datetime(2026, 10, 26, 23, 50)
        ap.plan_day(settings, state, late.date(), rng=random.Random(1), now=late)
        self.assertEqual(state["slots"], [])
        self.assertEqual(state["date"], "2026-10-26")
        ap.plan_day(settings, state, late.date(), rng=random.Random(1), now=late)
        self.assertEqual(state["planned_at"], late.isoformat(timespec="seconds"))


class TestSlotSelection(unittest.TestCase):
    def _state(self, *times, status=ap.SLOT_STATE_PENDING):
        state = ap.empty_state()
        state["date"] = times[0].date().isoformat()
        state["slots"] = [ap._new_slot(index, at) for index, at in enumerate(times, 1)]
        for slot in state["slots"]:
            slot["status"] = status
        return state

    def test_only_slots_that_are_due_run(self):
        state = self._state(
            datetime(2026, 10, 26, 11, 0),
            datetime(2026, 10, 26, 17, 0),
        )
        due = ap.next_due_slot(state, datetime(2026, 10, 26, 12, 0))
        self.assertEqual(due["index"], 1)
        self.assertIsNone(ap.next_due_slot(state, datetime(2026, 10, 26, 10, 0)))

    def test_missed_slots_are_caught_up_inside_the_window(self):
        settings = _settings(catch_up=True)
        state = self._state(datetime(2026, 10, 26, 11, 0))
        ap.expire_slots(settings, state, datetime(2026, 10, 26, 20, 0))
        self.assertEqual(state["slots"][0]["status"], ap.SLOT_STATE_PENDING)
        self.assertIsNotNone(ap.next_due_slot(state, datetime(2026, 10, 26, 20, 0)))

    def test_missed_slots_outside_the_window_are_skipped(self):
        settings = _settings(catch_up=True)
        state = self._state(datetime(2026, 10, 26, 11, 0))
        ap.expire_slots(settings, state, datetime(2026, 10, 27, 1, 0))
        self.assertEqual(state["slots"][0]["status"], ap.SLOT_STATE_SKIPPED)
        self.assertIn("ventana", state["slots"][0]["error"])

    def test_grace_minutes_allow_a_late_catch_up(self):
        settings = _settings(catch_up=True, catch_up_grace_minutes=90)
        state = self._state(datetime(2026, 10, 26, 11, 0))
        ap.expire_slots(settings, state, datetime(2026, 10, 26, 23, 0))
        self.assertEqual(state["slots"][0]["status"], ap.SLOT_STATE_PENDING)

    def test_without_catch_up_a_missed_slot_is_skipped(self):
        settings = _settings(catch_up=False)
        state = self._state(datetime(2026, 10, 26, 11, 0))
        ap.expire_slots(settings, state, datetime(2026, 10, 26, 12, 0))
        self.assertEqual(state["slots"][0]["status"], ap.SLOT_STATE_SKIPPED)

    def test_without_catch_up_the_poll_delay_is_tolerated(self):
        settings = _settings(catch_up=False)
        state = self._state(datetime(2026, 10, 26, 11, 0))
        ap.expire_slots(settings, state, datetime(2026, 10, 26, 11, 2))
        self.assertEqual(state["slots"][0]["status"], ap.SLOT_STATE_PENDING)

    def test_a_slot_interrupted_mid_upload_is_never_retried(self):
        settings = _settings()
        state = self._state(
            datetime(2026, 10, 26, 11, 0), status=ap.SLOT_STATE_RUNNING
        )
        ap.expire_slots(settings, state, datetime(2026, 10, 26, 11, 30))
        self.assertEqual(state["slots"][0]["status"], ap.SLOT_STATE_FAILED)
        self.assertIn("terminó", state["slots"][0]["error"])

    def test_published_slots_are_left_alone(self):
        settings = _settings()
        state = self._state(datetime(2026, 10, 26, 11, 0), status=ap.SLOT_STATE_DONE)
        ap.expire_slots(settings, state, datetime(2026, 10, 27, 5, 0))
        self.assertEqual(state["slots"][0]["status"], ap.SLOT_STATE_DONE)

    def test_next_wake_time_prefers_the_next_slot(self):
        state = self._state(
            datetime(2026, 10, 26, 11, 0), datetime(2026, 10, 26, 17, 0)
        )
        wake = ap.next_wake_time(_settings(), state, datetime(2026, 10, 26, 12, 0))
        self.assertEqual(wake, datetime(2026, 10, 26, 17, 0))

    def test_next_wake_time_falls_back_to_tomorrow(self):
        state = self._state(datetime(2026, 10, 26, 11, 0), status=ap.SLOT_STATE_DONE)
        wake = ap.next_wake_time(_settings(), state, datetime(2026, 10, 26, 21, 0))
        self.assertEqual(wake, datetime(2026, 10, 27, 0, 0))


class TestSubjects(unittest.TestCase):
    def test_a_subject_list_rotates_and_keeps_the_cursor(self):
        settings = _settings(subjects=("uno", "dos", "tres"))
        state = ap.empty_state()
        picked = [ap.resolve_subject(settings, state) for _ in range(4)]
        self.assertEqual(picked, ["uno", "dos", "tres", "uno"])
        self.assertEqual(state["subject_cursor"], 1)

    def test_the_niche_asks_the_model_for_a_fresh_subject(self):
        settings = _settings(niche="finanzas personales", video_language="es")
        state = ap.empty_state()
        state["recent_subjects"] = ["Interés compuesto"]
        with patch("app.services.llm.generate_video_subject", return_value="Fondo de emergencia") as generate:
            self.assertEqual(ap.resolve_subject(settings, state), "Fondo de emergencia")
        generate.assert_called_once_with(
            niche="finanzas personales",
            language="es",
            recent_subjects=["Interés compuesto"],
        )

    def test_without_niche_or_subjects_it_refuses(self):
        with self.assertRaises(ap.AutopilotConfigError):
            ap.resolve_subject(_settings(niche="", subjects=()), ap.empty_state())

    def test_remembered_subjects_are_deduplicated_and_capped(self):
        state = ap.empty_state()
        for index in range(ap.MAX_RECENT_SUBJECTS + 10):
            ap.remember_subject(state, f"tema {index}")
        ap.remember_subject(state, "tema 5")
        self.assertEqual(len(state["recent_subjects"]), ap.MAX_RECENT_SUBJECTS)
        self.assertEqual(state["recent_subjects"][-1], "tema 5")
        self.assertEqual(state["recent_subjects"].count("tema 5"), 1)


class TestAwaitCrossPost(unittest.TestCase):
    def test_a_completed_upload_succeeds(self):
        task = {"cross_post_state": const.CROSS_POST_STATE_COMPLETE, "cross_post_results": [{"ok": True}]}
        with patch.object(ap.sm.state, "get_task", return_value=task):
            result = ap.await_cross_post("task", timeout_seconds=5)
        self.assertTrue(result["success"])
        self.assertEqual(result["results"], [{"ok": True}])

    def test_a_failed_upload_reports_the_reason(self):
        task = {
            "cross_post_state": const.CROSS_POST_STATE_FAILED,
            "cross_post_error": "tiktok rechazó el vídeo",
        }
        with patch.object(ap.sm.state, "get_task", return_value=task):
            result = ap.await_cross_post("task", timeout_seconds=5)
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "tiktok rechazó el vídeo")

    def test_a_task_without_upload_is_not_waited_for(self):
        with patch.object(ap.sm.state, "get_task", return_value={"cross_post_state": None}):
            result = ap.await_cross_post("task", timeout_seconds=5)
        self.assertFalse(result["success"])

    def test_the_wait_gives_up_at_the_timeout(self):
        task = {"cross_post_state": const.CROSS_POST_STATE_PROCESSING}
        stop = threading.Event()
        with patch.object(ap.sm.state, "get_task", return_value=task):
            result = ap.await_cross_post(
                "task", timeout_seconds=0, poll_interval_seconds=0, stop_event=stop
            )
        self.assertFalse(result["success"])
        self.assertIn("no terminó", result["error"])

    def test_stopping_interrupts_the_wait(self):
        task = {"cross_post_state": const.CROSS_POST_STATE_PROCESSING}
        stop = threading.Event()
        stop.set()
        with patch.object(ap.sm.state, "get_task", return_value=task):
            result = ap.await_cross_post(
                "task", timeout_seconds=30, poll_interval_seconds=0, stop_event=stop
            )
        self.assertFalse(result["success"])
        self.assertIn("detuvo", result["error"])


class TestPublishOneVideo(unittest.TestCase):
    def setUp(self):
        self.params = MagicMock(name="VideoParams")
        self.build = patch.object(ap, "build_video_params", return_value=self.params)
        self.build.start()
        self.addCleanup(self.build.stop)

    def _run(self, start_result, schedule_error=None, upload=None):
        upload = upload or {"success": True, "cross_post_state": const.CROSS_POST_STATE_COMPLETE}
        with (
            patch("app.services.task.start", return_value=start_result) as start,
            patch(
                "app.services.task.schedule_cross_post_for_task",
                return_value=schedule_error,
            ) as schedule,
            patch.object(ap, "await_cross_post", return_value=upload) as await_upload,
        ):
            outcome = ap.publish_one_video(_settings(), ap.empty_state(), "un tema")
        return outcome, start, schedule, await_upload

    def test_a_generated_video_is_queued_and_awaited(self):
        outcome, start, schedule, await_upload = self._run(
            {"videos": ["/videos/uno.mp4"], "script": "guion", "cross_post_state": None}
        )
        self.assertTrue(outcome["success"])
        self.assertEqual(outcome["videos"], ["/videos/uno.mp4"])
        self.assertEqual(start.call_args.kwargs["stop_at"], "video")
        self.assertEqual(schedule.call_args.kwargs["video_paths"], ["/videos/uno.mp4"])
        await_upload.assert_called_once()

    def test_an_upload_already_queued_by_the_pipeline_is_not_queued_twice(self):
        outcome, _start, schedule, await_upload = self._run(
            {
                "videos": ["/videos/uno.mp4"],
                "cross_post_state": const.CROSS_POST_STATE_PENDING,
            }
        )
        self.assertTrue(outcome["success"])
        schedule.assert_not_called()
        await_upload.assert_called_once()

    def test_a_failed_generation_is_reported(self):
        outcome, _start, schedule, _await = self._run(
            {"state": const.TASK_STATE_FAILED, "failed_stage": "audio", "error": "sin TTS"}
        )
        self.assertFalse(outcome["success"])
        self.assertIn("audio", outcome["error"])
        schedule.assert_not_called()

    def test_a_generation_without_files_is_reported(self):
        outcome, _start, _schedule, _await = self._run({"videos": []})
        self.assertFalse(outcome["success"])
        self.assertIn("ningún archivo", outcome["error"])

    def test_a_queueing_failure_is_reported(self):
        outcome, _start, _schedule, await_upload = self._run(
            {"videos": ["/videos/uno.mp4"], "cross_post_state": None},
            schedule_error="la cola de publicación multiplataforma está llena",
        )
        self.assertFalse(outcome["success"])
        self.assertIn("cola", outcome["error"])
        await_upload.assert_not_called()

    def test_a_broken_wait_is_reported_as_in_flight(self):
        """Si falla la espera, la subida sigue su curso; el hueco no se puede reintentar."""
        with (
            patch("app.services.task.start", return_value={
                "videos": ["/videos/uno.mp4"], "cross_post_state": None,
            }),
            patch("app.services.task.schedule_cross_post_for_task", return_value=None),
            patch.object(ap, "await_cross_post", side_effect=RuntimeError("Redis cayó")),
        ):
            outcome = ap.publish_one_video(_settings(), ap.empty_state(), "un tema")
        self.assertFalse(outcome["success"])
        self.assertEqual(
            outcome["cross_post_state"], const.CROSS_POST_STATE_PROCESSING
        )
        self.assertIn("Redis cayó", outcome["error"])

    def test_a_broken_queueing_call_is_reported_as_in_flight(self):
        with (
            patch("app.services.task.start", return_value={
                "videos": ["/videos/uno.mp4"], "cross_post_state": None,
            }),
            patch(
                "app.services.task.schedule_cross_post_for_task",
                side_effect=RuntimeError("el pool está cerrado"),
            ),
            patch.object(ap, "await_cross_post") as await_upload,
        ):
            outcome = ap.publish_one_video(_settings(), ap.empty_state(), "un tema")
        self.assertFalse(outcome["success"])
        self.assertEqual(outcome["cross_post_state"], const.CROSS_POST_STATE_PENDING)
        await_upload.assert_not_called()

    def test_an_upload_failure_is_reported(self):
        outcome, _start, _schedule, _await = self._run(
            {"videos": ["/videos/uno.mp4"], "cross_post_state": None},
            upload={
                "success": False,
                "cross_post_state": const.CROSS_POST_STATE_FAILED,
                "error": "instagram rechazó el vídeo",
            },
        )
        self.assertFalse(outcome["success"])
        self.assertEqual(outcome["error"], "instagram rechazó el vídeo")


class TestRunSlot(unittest.TestCase):
    def setUp(self):
        self.settings = _settings()
        self.state = ap.empty_state()
        self.state["date"] = "2026-10-26"
        self.state["slots"] = [ap._new_slot(1, datetime(2026, 10, 26, 11, 0))]
        self.slot = self.state["slots"][0]
        self.saves = []

    def _persist(self):
        self.saves.append(json.dumps(self.state, default=str))

    def _publisher(self, outcome):
        def publisher(settings, state, subject, stop_event=None, task_id=None):
            publisher.calls.append(subject)
            return outcome

        publisher.calls = []
        return publisher

    def test_a_published_slot_is_marked_done(self):
        publisher = self._publisher(
            {
                "success": True,
                "task_id": "t1",
                "videos": ["/videos/uno.mp4"],
                "cross_post_state": const.CROSS_POST_STATE_COMPLETE,
            }
        )
        with patch.object(ap, "resolve_subject", return_value="Fondo de emergencia"):
            published = ap.run_slot(
                self.settings, self.state, self.slot, self._persist, publisher=publisher
            )
        self.assertTrue(published)
        self.assertEqual(self.slot["status"], ap.SLOT_STATE_DONE)
        self.assertEqual(self.slot["subject"], "Fondo de emergencia")
        self.assertEqual(self.slot["task_id"], "t1")
        self.assertEqual(self.state["recent_subjects"], ["Fondo de emergencia"])
        self.assertGreaterEqual(len(self.saves), 3)

    def test_a_failure_schedules_a_retry_inside_the_window(self):
        publisher = self._publisher({"success": False, "error": "pexels sin resultados"})
        with patch.object(ap, "resolve_subject", return_value="Un tema"):
            published = ap.run_slot(
                self.settings,
                self.state,
                self.slot,
                self._persist,
                publisher=publisher,
                now_fn=lambda: datetime(2026, 10, 26, 11, 0),
            )
        self.assertFalse(published)
        self.assertEqual(self.slot["status"], ap.SLOT_STATE_PENDING)
        self.assertEqual(self.slot["retry_at"], "2026-10-26T11:15:00")
        self.assertEqual(self.slot["attempts"], 1)

    def test_the_last_attempt_fails_for_good(self):
        publisher = self._publisher({"success": False, "error": "pexels sin resultados"})
        self.slot["attempts"] = 1
        with patch.object(ap, "resolve_subject", return_value="Un tema"):
            ap.run_slot(
                self.settings,
                self.state,
                self.slot,
                self._persist,
                publisher=publisher,
                now_fn=lambda: datetime(2026, 10, 26, 11, 0),
            )
        self.assertEqual(self.slot["status"], ap.SLOT_STATE_FAILED)
        self.assertEqual(self.slot["attempts"], 2)

    def test_a_retry_outside_the_window_fails_for_good(self):
        publisher = self._publisher({"success": False, "error": "pexels sin resultados"})
        with patch.object(ap, "resolve_subject", return_value="Un tema"):
            ap.run_slot(
                self.settings,
                self.state,
                self.slot,
                self._persist,
                publisher=publisher,
                now_fn=lambda: datetime(2026, 10, 26, 21, 55),
            )
        self.assertEqual(self.slot["status"], ap.SLOT_STATE_FAILED)

    def test_an_unconfirmed_upload_is_never_retried(self):
        """Pararse mientras sube no se puede reintentar: el vídeo puede estar ya fuera."""
        publisher = self._publisher(
            {
                "success": False,
                "task_id": "t1",
                "videos": ["/videos/uno.mp4"],
                "cross_post_state": const.CROSS_POST_STATE_PROCESSING,
                "error": "la publicación multiplataforma no terminó en 3900 segundos",
            }
        )
        with patch.object(ap, "resolve_subject", return_value="Un tema"):
            published = ap.run_slot(
                self.settings,
                self.state,
                self.slot,
                self._persist,
                publisher=publisher,
                now_fn=lambda: datetime(2026, 10, 26, 11, 0),
            )
        self.assertFalse(published)
        self.assertEqual(self.slot["status"], ap.SLOT_STATE_FAILED)
        self.assertEqual(self.slot["retry_at"], "")
        self.assertEqual(self.slot["attempts"], 1)

    def test_a_rejected_upload_is_retried(self):
        """Un rechazo confirmado no publicó nada, así que reintentarlo es seguro."""
        publisher = self._publisher(
            {
                "success": False,
                "task_id": "t1",
                "videos": ["/videos/uno.mp4"],
                "cross_post_state": const.CROSS_POST_STATE_FAILED,
                "error": "tiktok rechazó el vídeo",
            }
        )
        with patch.object(ap, "resolve_subject", return_value="Un tema"):
            ap.run_slot(
                self.settings,
                self.state,
                self.slot,
                self._persist,
                publisher=publisher,
                now_fn=lambda: datetime(2026, 10, 26, 11, 0),
            )
        self.assertEqual(self.slot["status"], ap.SLOT_STATE_PENDING)
        self.assertEqual(self.slot["retry_at"], "2026-10-26T11:15:00")

    def test_a_retry_reuses_the_subject_already_chosen(self):
        publisher = self._publisher({"success": False, "error": "fallo"})
        self.slot["subject"] = "Tema ya elegido"
        with patch.object(ap, "resolve_subject") as resolve:
            ap.run_slot(
                self.settings, self.state, self.slot, self._persist, publisher=publisher
            )
        resolve.assert_not_called()
        self.assertEqual(publisher.calls, ["Tema ya elegido"])

    def test_a_subject_error_fails_the_slot_without_generating(self):
        publisher = self._publisher({"success": True})
        with patch.object(ap, "resolve_subject", side_effect=ValueError("sin LLM")):
            published = ap.run_slot(
                self.settings, self.state, self.slot, self._persist, publisher=publisher
            )
        self.assertFalse(published)
        self.assertEqual(publisher.calls, [])
        self.assertIn("sin LLM", self.slot["error"])

    def test_the_task_id_is_recorded_before_the_video_is_generated(self):
        """Un hueco interrumpido tiene que dejar apuntada la tarea que hay que revisar."""
        seen = {}

        def publisher(settings, state, subject, stop_event=None, task_id=None):
            # En este punto la generación aún no ha empezado, y el estado ya está escrito.
            seen["task_id"] = task_id
            seen["persisted"] = json.loads(self.saves[-1])["slots"][0]["task_id"]
            seen["status"] = json.loads(self.saves[-1])["slots"][0]["status"]
            return {"success": True, "task_id": task_id, "videos": ["/videos/uno.mp4"]}

        with patch.object(ap, "resolve_subject", return_value="Un tema"):
            ap.run_slot(
                self.settings, self.state, self.slot, self._persist, publisher=publisher
            )
        self.assertTrue(seen["task_id"])
        self.assertEqual(seen["persisted"], seen["task_id"])
        self.assertEqual(seen["status"], ap.SLOT_STATE_RUNNING)
        self.assertEqual(self.slot["task_id"], seen["task_id"])

    def test_each_attempt_uses_its_own_task(self):
        used = []

        def publisher(settings, state, subject, stop_event=None, task_id=None):
            used.append(task_id)
            return {"success": False, "task_id": task_id, "error": "fallo"}

        with patch.object(ap, "resolve_subject", return_value="Un tema"):
            for _ in range(2):
                ap.run_slot(
                    self.settings,
                    self.state,
                    self.slot,
                    self._persist,
                    publisher=publisher,
                    now_fn=lambda: datetime(2026, 10, 26, 11, 0),
                )
        self.assertEqual(len(used), 2)
        self.assertEqual(len(set(used)), 2)
        self.assertEqual(self.slot["status"], ap.SLOT_STATE_FAILED)

    def test_an_unexpected_error_does_not_escape(self):
        def publisher(*_args, **_kwargs):
            raise RuntimeError("fallo del disco")

        with patch.object(ap, "resolve_subject", return_value="Un tema"):
            published = ap.run_slot(
                self.settings, self.state, self.slot, self._persist, publisher=publisher
            )
        self.assertFalse(published)
        self.assertIn("fallo del disco", self.slot["error"])


class TestSchedulerLoop(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._directory = tempfile.TemporaryDirectory()
        self.path = str(Path(self._directory.name) / "state.json")
        self.addCleanup(self._directory.cleanup)
        self.published = []

    def _publisher(self, success=True):
        def publisher(settings, state, subject, stop_event=None, task_id=None):
            self.published.append(subject)
            return {
                "success": success,
                "task_id": f"t{len(self.published)}",
                "videos": ["/videos/uno.mp4"] if success else [],
                "cross_post_state": (
                    const.CROSS_POST_STATE_COMPLETE if success else const.CROSS_POST_STATE_FAILED
                ),
                "error": None if success else "fallo de subida",
            }

        return publisher

    def test_one_pass_publishes_only_what_is_due(self):
        settings = _settings()
        with (
            patch.object(ap, "resolve_subject", side_effect=lambda *_: "Un tema"),
            patch.object(ap, "plan_slot_times", return_value=[
                datetime(2026, 10, 26, 11, 0),
                datetime(2026, 10, 26, 16, 0),
                datetime(2026, 10, 26, 20, 0),
            ]),
        ):
            summary = ap.run_due_slots(
                settings,
                path=self.path,
                now_fn=lambda: datetime(2026, 10, 26, 17, 0),
                publisher=self._publisher(),
            )
        self.assertEqual(summary["published"], 2)
        self.assertEqual(summary["counts"][ap.SLOT_STATE_DONE], 2)
        self.assertEqual(summary["counts"][ap.SLOT_STATE_PENDING], 1)
        self.assertEqual(len(self.published), 2)

    def test_a_second_pass_does_not_republish(self):
        settings = _settings()
        times = [
            datetime(2026, 10, 26, 11, 0),
            datetime(2026, 10, 26, 16, 0),
            datetime(2026, 10, 26, 20, 0),
        ]
        for _ in range(2):
            with (
                patch.object(ap, "resolve_subject", side_effect=lambda *_: "Un tema"),
                patch.object(ap, "plan_slot_times", return_value=times),
            ):
                ap.run_due_slots(
                    settings,
                    path=self.path,
                    now_fn=lambda: datetime(2026, 10, 26, 17, 0),
                    publisher=self._publisher(),
                )
        self.assertEqual(len(self.published), 2)

    def test_the_plan_survives_a_restart(self):
        settings = _settings()
        with patch.object(ap, "plan_slot_times", return_value=[
            datetime(2026, 10, 26, 11, 0),
        ]):
            ap.run_due_slots(
                settings,
                path=self.path,
                now_fn=lambda: datetime(2026, 10, 26, 9, 0),
                publisher=self._publisher(),
            )
        planned = [slot["at"] for slot in ap.load_state(self.path)["slots"]]
        with patch.object(ap, "plan_slot_times", return_value=[
            datetime(2026, 10, 26, 19, 0),
        ]):
            ap.run_due_slots(
                settings,
                path=self.path,
                now_fn=lambda: datetime(2026, 10, 26, 10, 0),
                publisher=self._publisher(),
            )
        self.assertEqual([slot["at"] for slot in ap.load_state(self.path)["slots"]], planned)

    def test_three_videos_are_published_every_day(self):
        settings = _settings()
        day = [date(2026, 10, 26)]
        clock = [datetime(2026, 10, 26, 21, 0)]

        def now():
            return clock[0]

        def times(target_day, _settings, rng=None, not_before=None):
            return [
                datetime(target_day.year, target_day.month, target_day.day, hour)
                for hour in (11, 15, 19)
            ]

        per_day = []
        with (
            patch.object(ap, "resolve_subject", side_effect=lambda *_: "Un tema"),
            patch.object(ap, "plan_slot_times", side_effect=times),
        ):
            for offset in range(5):
                current = day[0] + timedelta(days=offset)
                clock[0] = datetime(current.year, current.month, current.day, 21, 0)
                before = len(self.published)
                ap.run_due_slots(
                    settings, path=self.path, now_fn=now, publisher=self._publisher()
                )
                per_day.append(len(self.published) - before)
        self.assertEqual(per_day, [3, 3, 3, 3, 3])

    def test_the_loop_stops_when_asked(self):
        settings = _settings(poll_interval_seconds=5)
        stop = threading.Event()
        calls = []

        def publisher(_settings, _state, subject, stop_event=None, task_id=None):
            calls.append(subject)
            stop.set()
            return {
                "success": True,
                "task_id": "t1",
                "videos": ["/videos/uno.mp4"],
                "cross_post_state": const.CROSS_POST_STATE_COMPLETE,
            }

        with (
            patch.object(ap, "resolve_subject", side_effect=lambda *_: "Un tema"),
            patch.object(ap, "plan_slot_times", return_value=[datetime(2026, 10, 26, 11, 0)]),
        ):
            summary = ap.run_forever(
                settings,
                path=self.path,
                now_fn=lambda: datetime(2026, 10, 26, 12, 0),
                stop_event=stop,
                publisher=publisher,
                max_cycles=5,
            )
        self.assertEqual(summary["published"], 1)
        self.assertEqual(len(calls), 1)

    def test_a_cycle_error_does_not_kill_the_loop(self):
        settings = _settings(poll_interval_seconds=5)
        stop = threading.Event()
        attempts = []

        def exploding(*_args, **_kwargs):
            attempts.append(1)
            if len(attempts) >= 2:
                stop.set()
            raise RuntimeError("Redis no responde")

        with patch.object(ap, "run_due_slots", side_effect=exploding):
            summary = ap.run_forever(
                settings,
                path=self.path,
                now_fn=lambda: datetime(2026, 10, 26, 12, 0),
                stop_event=stop,
                max_cycles=5,
            )
        self.assertEqual(len(attempts), 2)
        self.assertEqual(summary["published"], 0)

    def test_a_configuration_error_stops_the_loop(self):
        settings = _settings()
        with patch.object(
            ap, "run_due_slots", side_effect=ap.AutopilotConfigError("falta el nicho")
        ):
            with self.assertRaises(ap.AutopilotConfigError):
                ap.run_forever(
                    settings,
                    path=self.path,
                    now_fn=lambda: datetime(2026, 10, 26, 12, 0),
                    stop_event=threading.Event(),
                    max_cycles=3,
                )

    def test_status_report_describes_the_plan(self):
        settings = _settings()
        with patch.object(ap, "plan_slot_times", return_value=[datetime(2026, 10, 26, 11, 0)]):
            ap.run_due_slots(
                settings,
                path=self.path,
                now_fn=lambda: datetime(2026, 10, 26, 9, 0),
                publisher=self._publisher(),
            )
        service = MagicMock()
        service.enabled = True
        service.api_key = "key"
        service.username = "usuario"
        service.platforms = ["tiktok"]
        with patch("app.services.upload_post.upload_post_service", service):
            report = ap.status_report(settings, path=self.path)
        self.assertEqual(report["videos_per_day"], 3)
        self.assertEqual(report["window"], "10:00-22:00")
        self.assertEqual(report["problems"], [])
        self.assertEqual(len(report["slots"]), 1)


class TestSingleInstanceLock(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._directory = tempfile.TemporaryDirectory()
        self.path = str(Path(self._directory.name) / "state.json.lock")
        self.addCleanup(self._directory.cleanup)

    def test_the_lock_is_taken_and_released(self):
        with ap.SingleInstanceLock(self.path) as lock:
            self.assertEqual(lock.owner()["pid"], ap.os.getpid())
        self.assertFalse(Path(self.path).exists())

    def test_a_live_owner_blocks_a_second_scheduler(self):
        Path(self.path).write_text(
            json.dumps({"host": ap.socket.gethostname(), "pid": 4242}), encoding="utf-8"
        )
        with patch.object(ap, "_is_process_alive", return_value=True):
            # Un tipo propio: una tarea programada que se encuentra la pasada anterior
            # todavía generando no debe informar de un fallo.
            with self.assertRaises(ap.AutopilotBusyError):
                ap.SingleInstanceLock(self.path).acquire()

    def test_a_dead_owner_is_reclaimed(self):
        Path(self.path).write_text(
            json.dumps({"host": ap.socket.gethostname(), "pid": 4242}), encoding="utf-8"
        )
        with patch.object(ap, "_is_process_alive", return_value=False):
            with ap.SingleInstanceLock(self.path):
                pass
        self.assertFalse(Path(self.path).exists())

    def test_another_host_is_not_reclaimed(self):
        Path(self.path).write_text(
            json.dumps({"host": "otro-equipo", "pid": 1}), encoding="utf-8"
        )
        with self.assertRaises(ap.AutopilotError):
            ap.SingleInstanceLock(self.path).acquire()

    def test_a_corrupt_lock_is_reclaimed(self):
        Path(self.path).write_text("no es json", encoding="utf-8")
        with ap.SingleInstanceLock(self.path):
            pass
        self.assertFalse(Path(self.path).exists())


class TestVideoParams(unittest.TestCase):
    def test_each_slot_publishes_exactly_one_video(self):
        settings = _settings(video_language="es-ES")
        params = ap.build_video_params(settings, "Fondo de emergencia")
        self.assertEqual(params.video_subject, "Fondo de emergencia")
        self.assertEqual(params.video_count, 1)
        self.assertEqual(params.video_language, "es-ES")

    def test_cli_options_can_override_the_defaults(self):
        settings = _settings(cli_options=("--video-source", "pexels", "--video-aspect", "16:9"))
        params = ap.build_video_params(settings, "Un tema")
        self.assertEqual(params.video_source, "pexels")
        self.assertEqual(params.video_aspect.value, "16:9")

    def test_invalid_cli_options_are_reported_as_configuration(self):
        settings = _settings(cli_options=("--no-existe",))
        with self.assertRaises(ap.AutopilotConfigError):
            ap.build_video_params(settings, "Un tema")


class TestCrossPostScheduling(unittest.TestCase):
    """La publicación automática publica aunque upload_post_auto_upload siga en false."""

    def test_the_wrapper_queues_with_the_configured_account(self):
        from app.services import task as tm

        service = tm.upload_post.upload_post_service
        params = MagicMock()
        with (
            patch.object(type(service), "is_configured", return_value=True),
            patch.object(
                type(service), "platforms", new_callable=PropertyMock,
                return_value=["tiktok", "instagram"],
            ),
            patch.object(
                type(service), "youtube_privacy_status", new_callable=PropertyMock,
                return_value="unlisted",
            ),
            patch.object(
                type(service), "youtube_made_for_kids", new_callable=PropertyMock,
                return_value=False,
            ),
            patch.object(tm, "_patch_cross_post_state", return_value=True) as patch_state,
            patch.object(tm, "_schedule_cross_post", return_value=None) as schedule,
        ):
            error = tm.schedule_cross_post_for_task(
                task_id="t1",
                video_paths=["/videos/uno.mp4"],
                params=params,
                video_script="guion",
            )
        self.assertIsNone(error)
        self.assertEqual(
            patch_state.call_args.kwargs["cross_post_state"], const.CROSS_POST_STATE_PENDING
        )
        self.assertEqual(schedule.call_args.kwargs["platforms"], ["tiktok", "instagram"])
        self.assertEqual(schedule.call_args.kwargs["youtube_privacy_status"], "unlisted")

    def test_an_unconfigured_account_is_reported(self):
        from app.services import task as tm

        service = tm.upload_post.upload_post_service
        with patch.object(type(service), "is_configured", return_value=False):
            error = tm.schedule_cross_post_for_task("t1", ["/videos/uno.mp4"], MagicMock())
        self.assertIn("no está configurado", error)

    def test_a_deleted_task_is_reported(self):
        from app.services import task as tm

        service = tm.upload_post.upload_post_service
        with (
            patch.object(type(service), "is_configured", return_value=True),
            patch.object(
                type(service), "platforms", new_callable=PropertyMock, return_value=["tiktok"]
            ),
            patch.object(tm, "_patch_cross_post_state", return_value=False),
            patch.object(tm, "_schedule_cross_post") as schedule,
        ):
            error = tm.schedule_cross_post_for_task("t1", ["/videos/uno.mp4"], MagicMock())
        self.assertIn("ya no existe", error)
        schedule.assert_not_called()

    def test_without_platforms_nothing_is_queued(self):
        from app.services import task as tm

        service = tm.upload_post.upload_post_service
        with (
            patch.object(type(service), "is_configured", return_value=True),
            patch.object(
                type(service), "platforms", new_callable=PropertyMock, return_value=[]
            ),
            patch.object(tm, "_schedule_cross_post") as schedule,
        ):
            error = tm.schedule_cross_post_for_task("t1", ["/videos/uno.mp4"], MagicMock())
        self.assertIn("plataformas", error)
        schedule.assert_not_called()


class TestFullDayIntegration(unittest.TestCase):
    """
    Un día completo con la cola de publicación real.

    Solo se simulan los dos bordes externos: la generación del vídeo y la llamada HTTP a
    Upload-Post. El plan, el estado, el encolado de la publicación, el worker en segundo
    plano y la espera de confirmación son el código de producción, para que el test
    detecte una ruptura del cableado y no solo de las piezas por separado.
    """

    def setUp(self):
        import tempfile

        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)
        self.state_file = str(self.directory / "state.json")
        self.generated = []
        self.posted = []

    def _fake_start(self, task_id, params, stop_at="video", **_kwargs):
        from app.services import task as tm

        self.generated.append((task_id, params.video_subject, stop_at))
        video = self.directory / f"{task_id}.mp4"
        video.write_bytes(b"video simulado")
        # El flujo real deja la tarea completa en el backend de estado antes de encolar
        # la publicación; la espera de confirmación la consulta desde ahí.
        tm.sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            videos=[str(video)],
            script="un guion",
            cross_post_state=None,
        )
        return {"videos": [str(video)], "script": "un guion", "cross_post_state": None}

    def _fake_post(self, *_args, **kwargs):
        fields = kwargs.get("data") or []
        self.posted.append(
            {
                "user": next((value for key, value in fields if key == "user"), None),
                "platforms": [value for key, value in fields if key == "platform[]"],
            }
        )
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "success": True,
            "request_id": f"req-{len(self.posted)}",
            "results": {"tiktok": {"success": True}, "instagram": {"success": True}},
        }
        return response

    def _run_day(self, settings, now):
        from app.models.schema import VideoParams
        from app.services import task as tm
        from app.services import upload_post

        service = upload_post.upload_post_service
        real_await = ap.await_cross_post
        with (
            patch.object(tm, "start", side_effect=self._fake_start),
            patch.object(
                type(service), "enabled", new_callable=PropertyMock, return_value=True
            ),
            patch.object(
                type(service), "api_key", new_callable=PropertyMock, return_value="clave"
            ),
            patch.object(
                type(service),
                "username",
                new_callable=PropertyMock,
                return_value="usuario",
            ),
            patch.object(
                type(service),
                "platforms",
                new_callable=PropertyMock,
                return_value=["tiktok", "instagram"],
            ),
            patch.object(
                ap,
                "build_video_params",
                side_effect=lambda _settings, subject: VideoParams(
                    video_subject=subject, video_count=1
                ),
            ),
            # Las horas se fijan a primera hora del día para que las tres estén vencidas
            # y la recuperación de huecos perdidos las publique en una sola pasada.
            patch.object(
                ap,
                "plan_slot_times",
                return_value=[
                    datetime(2026, 10, 26, 11, 0),
                    datetime(2026, 10, 26, 15, 0),
                    datetime(2026, 10, 26, 19, 0),
                ],
            ),
            # El sondeo real espera 5 segundos entre consultas; el test no necesita
            # medir esa espera, solo que la confirmación se lea del estado de la tarea.
            patch.object(
                ap,
                "await_cross_post",
                side_effect=lambda task_id, **kwargs: real_await(
                    task_id, **{**kwargs, "poll_interval_seconds": 0.05}
                ),
            ),
            patch("app.services.upload_post.requests.post", side_effect=self._fake_post),
            patch(
                "app.services.llm.generate_social_metadata",
                return_value={
                    "title": "un titulo",
                    "caption": "un texto",
                    "hashtags": ["#uno"],
                },
            ),
        ):
            return ap.run_due_slots(settings, path=self.state_file, now_fn=lambda: now)

    def test_a_whole_day_generates_and_publishes_three_videos(self):
        settings = _settings(
            subjects=("Tema uno", "Tema dos", "Tema tres"),
            state_file=self.state_file,
            upload_timeout_seconds=60,
        )
        summary = self._run_day(settings, datetime(2026, 10, 26, 21, 0))

        self.assertEqual(summary["published"], 3)
        self.assertEqual(summary["counts"][ap.SLOT_STATE_DONE], 3)
        self.assertEqual(
            [subject for _task, subject, _stage in self.generated],
            ["Tema uno", "Tema dos", "Tema tres"],
        )
        self.assertEqual({stage for _t, _s, stage in self.generated}, {"video"})
        self.assertEqual(len(self.posted), 3)
        self.assertTrue(
            all(entry["platforms"] == ["tiktok", "instagram"] for entry in self.posted)
        )

        state = ap.load_state(self.state_file)
        self.assertTrue(
            all(
                slot["cross_post_state"] == const.CROSS_POST_STATE_COMPLETE
                and len(slot["videos"]) == 1
                for slot in state["slots"]
            )
        )
        self.assertEqual(
            state["recent_subjects"], ["Tema uno", "Tema dos", "Tema tres"]
        )

    def test_a_second_pass_of_the_same_day_republishes_nothing(self):
        settings = _settings(
            subjects=("Tema uno", "Tema dos", "Tema tres"),
            state_file=self.state_file,
            upload_timeout_seconds=60,
        )
        self._run_day(settings, datetime(2026, 10, 26, 21, 0))
        again = self._run_day(settings, datetime(2026, 10, 26, 21, 30))
        self.assertEqual(again["executed"], 0)
        self.assertEqual(len(self.posted), 3)
        self.assertEqual(len(self.generated), 3)


class TestSubjectGeneration(unittest.TestCase):
    def test_the_prompt_carries_the_niche_and_the_published_subjects(self):
        from app.services import llm

        prompt = llm.build_video_subject_prompt(
            "finanzas personales", "es", ["Interés compuesto", "Interés compuesto"]
        )
        self.assertIn("finanzas personales", prompt)
        self.assertIn("- Interés compuesto", prompt)
        self.assertEqual(prompt.count("- Interés compuesto"), 1)
        self.assertIn("Write the subject in this language: es.", prompt)

    def test_without_published_subjects_the_prompt_says_so(self):
        from app.services import llm

        self.assertIn("(nothing published yet)", llm.build_video_subject_prompt("un nicho"))

    def test_a_list_marker_and_quotes_are_stripped(self):
        from app.services import llm

        self.assertEqual(
            llm._parse_video_subject('2) "El error del presupuesto"'),
            "El error del presupuesto",
        )

    def test_a_label_prefix_is_stripped(self):
        from app.services import llm

        self.assertEqual(
            llm._parse_video_subject("Tema: Interés compuesto\nlínea de sobra"),
            "Interés compuesto",
        )

    def test_an_empty_answer_is_rejected(self):
        from app.services import llm

        with self.assertRaises(ValueError):
            llm._parse_video_subject("   \n  ")

    def test_a_provider_error_is_not_retried(self):
        from app.services import llm

        with patch.object(
            llm, "_generate_response", return_value="Error: sin crédito"
        ) as generate:
            with self.assertRaises(ValueError):
                llm.generate_video_subject("un nicho")
        self.assertEqual(generate.call_count, 1)

    def test_an_empty_niche_is_rejected(self):
        from app.services import llm

        with self.assertRaises(ValueError):
            llm.generate_video_subject("   ")

    def test_a_valid_answer_is_returned(self):
        from app.services import llm

        with patch.object(llm, "_generate_response", return_value="Fondo de emergencia"):
            self.assertEqual(
                llm.generate_video_subject("finanzas", recent_subjects=["otro"]),
                "Fondo de emergencia",
            )


if __name__ == "__main__":
    unittest.main()
