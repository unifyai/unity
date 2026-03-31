"""
Tests for the startup acknowledgment mechanism.

When an idle container is claimed via K8s label patching, it must write a
``unity-startup-ack`` label back to prove it initialized. The adapter uses
this label to distinguish live containers from zombies (containers that
were claimed but never started processing).

These tests verify:
1. patch_job_label includes the ack label when ack_ts is provided
2. _poll_for_assignment writes the ack after detecting assignment
"""

from __future__ import annotations

import asyncio
import json
import time
import pytest
from unittest.mock import patch, MagicMock

from unity.conversation_manager.in_memory_event_broker import (
    create_in_memory_event_broker,
    reset_in_memory_event_broker,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_in_memory_event_broker()
    yield
    reset_in_memory_event_broker()


@pytest.fixture
def broker():
    return create_in_memory_event_broker()


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


# =========================================================================
# _poll_for_assignment: writes ack after detecting claim
# =========================================================================


class TestPollForAssignmentAck:
    """_poll_for_assignment should write a startup-ack label after assignment."""

    @pytest.mark.asyncio
    async def test_poll_writes_ack_on_assignment(self, broker):
        from unity.conversation_manager.comms_manager import CommsManager

        startup_config = json.dumps(
            {
                "api_key": "test_key",
                "medium": "unify_message",
                "assistant_id": "82",
                "user_id": "user_123",
                "assistant_first_name": "Oliver",
                "assistant_surname": "Peterson",
                "assistant_age": "35",
                "assistant_nationality": "UK",
                "assistant_about": "Test",
                "assistant_timezone": "UTC",
                "assistant_number": "",
                "assistant_email": "",
                "user_first_name": "Peter",
                "user_surname": "Scholes",
                "user_number": "",
                "user_email": "peter@test.com",
                "voice_provider": "elevenlabs",
                "voice_id": "test_voice",
            },
        )

        poll_count = 0

        def fake_read_own_job(comms_url, admin_key, job_name):
            nonlocal poll_count
            poll_count += 1
            if poll_count == 1:
                return {"labels": {"unity-status": "idle"}, "annotations": {}}
            return {
                "labels": {"unity-status": "running"},
                "annotations": {"unity-startup-config": startup_config},
            }

        mark_label_calls = []

        def fake_mark_job_label(job_name, status, **kwargs):
            mark_label_calls.append({"job_name": job_name, "status": status, **kwargs})
            return True

        with (
            patch(
                "unity.conversation_manager.comms_manager.SESSION_DETAILS",
            ) as mock_sd,
            patch("unity.conversation_manager.comms_manager.SETTINGS") as mock_settings,
            patch(
                "unity.conversation_manager.comms_manager.read_own_job",
                side_effect=fake_read_own_job,
            ),
            patch(
                "unity.conversation_manager.comms_manager.mark_job_label",
                side_effect=fake_mark_job_label,
            ),
        ):
            mock_sd.assistant.agent_id = None
            mock_settings.conversation.COMMS_URL = "http://comms:8080"
            mock_settings.ORCHESTRA_ADMIN_KEY.get_secret_value.return_value = "key"
            mock_settings.conversation.JOB_NAME = "unity-test-job"
            mock_settings.conversation.ASSIGNMENT_POLL_INTERVAL = 0.01
            mock_settings.GCP_PROJECT_ID = "test-project"

            cm = CommsManager(broker)

            async with broker.pubsub() as pubsub:
                await pubsub.subscribe("app:comms:startup")

                task = asyncio.create_task(cm._poll_for_assignment())
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except asyncio.TimeoutError:
                    pytest.fail("_poll_for_assignment did not return after assignment")

            # Give the daemon thread a moment to execute
            await asyncio.sleep(0.1)

            ack_calls = [c for c in mark_label_calls if c.get("ack_ts")]
            assert (
                len(ack_calls) == 1
            ), f"Expected exactly 1 ack call, got {len(ack_calls)}: {mark_label_calls}"
            assert ack_calls[0]["status"] == "running"
            assert ack_calls[0]["job_name"] == "unity-test-job"
            ack_ts = int(ack_calls[0]["ack_ts"])
            assert abs(ack_ts - int(time.time())) < 10
