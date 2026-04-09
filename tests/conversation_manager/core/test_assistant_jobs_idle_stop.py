"""Tests for idle-timeout session stop behavior in AssistantJobs helpers."""

import importlib
from unittest.mock import patch


class TestMarkJobDoneIdleStop:
    """`mark_job_done()` should only stop the session for true idle shutdowns."""

    @patch("unity.conversation_manager.assistant_jobs.release_pool_vm")
    @patch("unity.conversation_manager.assistant_jobs.mark_job_label")
    @patch("unity.conversation_manager.assistant_jobs.stop_assistant_session")
    @patch("unity.conversation_manager.assistant_jobs.SETTINGS")
    @patch("unity.conversation_manager.assistant_jobs.SESSION_DETAILS")
    def test_mark_job_done_stops_session_before_marking_job_done(
        self,
        mock_session_details,
        mock_settings,
        mock_stop_session,
        mock_mark_job_label,
        _mock_release_pool_vm,
    ):
        assistant_jobs = importlib.import_module(
            "unity.conversation_manager.assistant_jobs",
        )
        mock_settings.conversation.COMMS_URL = "http://comms:8080"
        mock_settings.ORCHESTRA_ADMIN_KEY.get_secret_value.return_value = "key"
        mock_session_details.assistant.agent_id = 123
        mock_session_details.assistant.desktop_mode = "none"

        call_order: list[str] = []
        mock_stop_session.side_effect = lambda *args, **kwargs: call_order.append(
            "stop",
        )
        mock_mark_job_label.side_effect = lambda *args, **kwargs: call_order.append(
            "mark",
        )

        with patch.object(assistant_jobs, "_session_start_perf", None):
            assistant_jobs.mark_job_done(
                "unity-job-1",
                shutdown_reason="idle_timeout",
            )

        mock_stop_session.assert_called_once_with("http://comms:8080", "key", "123")
        mock_mark_job_label.assert_called_once_with("unity-job-1", "done")
        assert call_order == ["stop", "mark"]

    @patch("unity.conversation_manager.assistant_jobs.release_pool_vm")
    @patch("unity.conversation_manager.assistant_jobs.mark_job_label")
    @patch("unity.conversation_manager.assistant_jobs.stop_assistant_session")
    @patch("unity.conversation_manager.assistant_jobs.SETTINGS")
    @patch("unity.conversation_manager.assistant_jobs.SESSION_DETAILS")
    def test_mark_job_done_keeps_non_idle_shutdowns_on_restart_path(
        self,
        mock_session_details,
        mock_settings,
        mock_stop_session,
        mock_mark_job_label,
        _mock_release_pool_vm,
    ):
        assistant_jobs = importlib.import_module(
            "unity.conversation_manager.assistant_jobs",
        )
        mock_settings.conversation.COMMS_URL = "http://comms:8080"
        mock_settings.ORCHESTRA_ADMIN_KEY.get_secret_value.return_value = "key"
        mock_session_details.assistant.agent_id = 123
        mock_session_details.assistant.desktop_mode = "none"

        with patch.object(assistant_jobs, "_session_start_perf", None):
            assistant_jobs.mark_job_done(
                "unity-job-1",
                shutdown_reason="external_sigterm",
            )

        mock_stop_session.assert_not_called()
        mock_mark_job_label.assert_called_once_with("unity-job-1", "done")
