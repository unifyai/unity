"""Tests for optional startup-ack label helpers."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


# =========================================================================
# patch_job_label: ack_ts parameter
# =========================================================================


class TestPatchJobLabelAck:
    """patch_job_label should include unity-startup-ack when ack_ts is set."""

    @patch("unity.conversation_manager.assistant_jobs_api.requests")
    def test_ack_ts_included_in_label_patch(self, mock_requests):
        from unity.conversation_manager.assistant_jobs_api import patch_job_label

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_requests.patch.return_value = mock_resp

        result = patch_job_label(
            comms_url="http://comms:8080",
            admin_key="key",
            job_name="unity-test-job",
            status="running",
            ack_ts="1711800000",
        )

        assert result is True
        call_kwargs = mock_requests.patch.call_args
        labels_json = call_kwargs.kwargs.get("data", call_kwargs[1].get("data", {}))[
            "labels"
        ]
        labels = json.loads(labels_json)

        assert labels["unity-status"] == "running"
        assert labels["unity-startup-ack"] == "1711800000"

    @patch("unity.conversation_manager.assistant_jobs_api.requests")
    def test_no_ack_ts_when_omitted(self, mock_requests):
        from unity.conversation_manager.assistant_jobs_api import patch_job_label

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_requests.patch.return_value = mock_resp

        patch_job_label(
            comms_url="http://comms:8080",
            admin_key="key",
            job_name="unity-test-job",
            status="idle",
        )

        call_kwargs = mock_requests.patch.call_args
        labels_json = call_kwargs.kwargs.get("data", call_kwargs[1].get("data", {}))[
            "labels"
        ]
        labels = json.loads(labels_json)

        assert "unity-startup-ack" not in labels

    @patch("unity.conversation_manager.assistant_jobs_api.requests")
    def test_ack_ts_coexists_with_assistant_id(self, mock_requests):
        from unity.conversation_manager.assistant_jobs_api import patch_job_label

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_requests.patch.return_value = mock_resp

        patch_job_label(
            comms_url="http://comms:8080",
            admin_key="key",
            job_name="unity-test-job",
            status="running",
            assistant_id="82",
            ack_ts="1711800000",
        )

        call_kwargs = mock_requests.patch.call_args
        labels_json = call_kwargs.kwargs.get("data", call_kwargs[1].get("data", {}))[
            "labels"
        ]
        labels = json.loads(labels_json)

        assert labels["unity-status"] == "running"
        assert labels["assistant-id"] == "82"
        assert labels["unity-startup-ack"] == "1711800000"


# =========================================================================
# mark_job_label: ack_ts passthrough
# =========================================================================


class TestMarkJobLabelAck:
    """mark_job_label should forward ack_ts to the underlying patch_job_label."""

    @patch("unity.conversation_manager.assistant_jobs.patch_job_label")
    @patch("unity.conversation_manager.assistant_jobs.SETTINGS")
    def test_ack_ts_forwarded(self, mock_settings, mock_patch):
        from unity.conversation_manager.assistant_jobs import mark_job_label

        mock_settings.conversation.COMMS_URL = "http://comms:8080"
        mock_settings.ORCHESTRA_ADMIN_KEY.get_secret_value.return_value = "key"
        mock_patch.return_value = True

        mark_job_label("unity-test-job", "running", ack_ts="1711800000")

        mock_patch.assert_called_once_with(
            "http://comms:8080",
            "key",
            "unity-test-job",
            "running",
            None,
            ack_ts="1711800000",
            timeout=30,
            retries=0,
        )
