"""
Tests for MagnitudeBackend session ID export to environment variable.

When MagnitudeBackend initialises and receives a sessionId from the
agent-service ``/start`` endpoint, it must export that ID via
``os.environ["AGENT_SERVICE_SESSION_ID"]`` so that subprocesses
(e.g. the ConversationManager fast brain) can include it in their own
HTTP requests to the agent-service.
"""

import os
from unittest.mock import patch

import pytest


class TestMagnitudeBackendSessionExport:
    """MagnitudeBackend.__init__ should export sessionId to env."""

    @pytest.fixture(autouse=True)
    def _clean_env(self):
        """Ensure the env var is absent before and cleaned up after each test."""
        old = os.environ.pop("AGENT_SERVICE_SESSION_ID", None)
        yield
        if old is not None:
            os.environ["AGENT_SERVICE_SESSION_ID"] = old
        else:
            os.environ.pop("AGENT_SERVICE_SESSION_ID", None)

    def test_session_id_exported_on_init(self):
        """__init__ sets AGENT_SERVICE_SESSION_ID in os.environ."""
        fake_start_response = {"sessionId": "test-sess-42"}

        with (
            patch(
                "unity.function_manager.computer_backends.MagnitudeBackend._sync_request",
                return_value=fake_start_response,
            ),
            patch(
                "unity.function_manager.computer_backends.MagnitudeBackend._check_service_ready",
            ),
        ):
            from unity.function_manager.computer_backends import MagnitudeBackend

            MagnitudeBackend(
                agent_server_url="http://localhost:3000",
                headless=False,
                agent_mode="desktop",
            )

        assert os.environ.get("AGENT_SERVICE_SESSION_ID") == "test-sess-42"
