"""Tests for pool VM release semantics in AssistantJobs API."""

import importlib
from unittest.mock import MagicMock, patch


class TestReleasePoolVm:
    """`release_pool_vm()` should respect Comms' async release contract."""

    @patch("unity.conversation_manager.assistant_jobs_api.requests")
    def test_release_pool_vm_treats_releasing_response_as_success(self, mock_requests):
        release_pool_vm = importlib.import_module(
            "unity.conversation_manager.assistant_jobs_api",
        ).release_pool_vm

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "released": False,
            "pool_role": "releasing",
            "message": "Release already in progress",
        }
        mock_requests.post.return_value = mock_resp

        release_pool_vm(
            comms_url="http://comms:8080",
            admin_key="key",
            assistant_id="assistant-123",
            binding_id="binding-123",
            max_attempts=1,
        )

        mock_requests.post.assert_called_once_with(
            "http://comms:8080/infra/vm/pool/release",
            json={"assistant_id": "assistant-123", "binding_id": "binding-123"},
            headers={"Authorization": "Bearer key"},
            timeout=60,
        )

    @patch("unity.conversation_manager.assistant_jobs_api.requests")
    def test_release_pool_vm_does_not_fallback_to_manual_detach(self, mock_requests):
        release_pool_vm = importlib.import_module(
            "unity.conversation_manager.assistant_jobs_api",
        ).release_pool_vm

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "released": False,
            "message": "No VM assigned",
        }
        mock_requests.post.return_value = mock_resp

        release_pool_vm(
            comms_url="http://comms:8080",
            admin_key="key",
            assistant_id="assistant-123",
            binding_id="binding-123",
            max_attempts=1,
        )

        assert mock_requests.post.call_count == 1

    @patch("unity.conversation_manager.assistant_jobs_api.requests")
    def test_release_pool_vm_includes_job_name_target(self, mock_requests):
        release_pool_vm = importlib.import_module(
            "unity.conversation_manager.assistant_jobs_api",
        ).release_pool_vm

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "released": True,
            "pool_role": "releasing",
        }
        mock_requests.post.return_value = mock_resp

        release_pool_vm(
            comms_url="http://comms:8080",
            admin_key="key",
            assistant_id="assistant-123",
            binding_id="binding-123",
            job_name="unity-job-1",
            max_attempts=1,
        )

        mock_requests.post.assert_called_once_with(
            "http://comms:8080/infra/vm/pool/release",
            json={
                "assistant_id": "assistant-123",
                "binding_id": "binding-123",
                "job_name": "unity-job-1",
            },
            headers={"Authorization": "Bearer key"},
            timeout=60,
        )
