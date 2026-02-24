"""
Tests for agent-service URL resolution in the screenshot capture path.

The ConversationManager fast brain captures assistant desktop screenshots
via ``_resolve_agent_service_url``. This must produce the correct URL for
both local dev (localhost:3000) and managed VMs (desktop_url + /api).
"""

from unittest.mock import patch, MagicMock

import pytest

_TEST_CASES = [
    # (desktop_url, expected_base_url)
    (None, "http://localhost:3000"),
    ("https://vm-123.run.app", "https://vm-123.run.app/api"),
    ("https://vm-123.run.app/", "https://vm-123.run.app/api"),
    ("http://10.0.0.5:8080", "http://10.0.0.5:8080/api"),
    ("http://10.0.0.5:8080/", "http://10.0.0.5:8080/api"),
]


class TestScreenshotUrlResolution:
    """_resolve_agent_service_url must return correct URLs."""

    @pytest.mark.parametrize("desktop_url,expected", _TEST_CASES)
    def test_capture_screenshot_resolver(self, desktop_url, expected):
        """_resolve_agent_service_url returns the expected URL."""
        from unity.conversation_manager.medium_scripts.common import (
            _resolve_agent_service_url,
        )

        mock_session = MagicMock()
        mock_session.assistant.desktop_url = desktop_url

        with patch(
            "unity.session_details.SESSION_DETAILS",
            mock_session,
        ):
            result = _resolve_agent_service_url()

        assert result == expected, (
            f"Screenshot resolver: desktop_url={desktop_url!r} → "
            f"{result!r}, expected {expected!r}"
        )
