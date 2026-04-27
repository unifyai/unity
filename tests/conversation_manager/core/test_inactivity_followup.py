"""Unit tests for the inactivity-followup unity-side wiring.

Covers:
    1. Event factories
        - _inactivity_followup_event_from_payload accepts any dict, copies reason
        - _inactivity_followup_event_from_wake_reason gates on type=inactivity_followup
    2. Handler side effects
        - pushes a notification to cm.notifications_bar
        - logs to the session logger
    3. EventHandler registration
    4. activity_sync.touch_assistant_activity
        - returns False on missing config
        - posts to the right URL with admin auth on success
        - swallows exceptions
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from unity.comms.primitives import CommsPrimitives
from unity.conversation_manager.domains.event_handlers import EventHandler
from unity.conversation_manager.domains.inactivity import (
    _DEFAULT_REASON,
    _handle_inactivity_followup_event,
    _inactivity_followup_event_from_payload,
    _inactivity_followup_event_from_wake_reason,
    _inactivity_notification_text,
)
from unity.conversation_manager.domains.notifications import NotificationBar
from unity.conversation_manager.events import InactivityFollowup
from unity.transcript_manager.activity_sync import (
    cancel_assistant_termination_via_orchestra,
    terminate_assistant_via_orchestra,
    touch_assistant_activity,
)

# ===========================================================================
# 1. Event factories
# ===========================================================================


class TestInactivityEventFactories:
    def test_payload_factory_returns_event_with_default_reason(self):
        event = _inactivity_followup_event_from_payload({})
        assert isinstance(event, InactivityFollowup)
        assert event.reason == _DEFAULT_REASON

    def test_payload_factory_uses_explicit_reason_when_given(self):
        event = _inactivity_followup_event_from_payload({}, reason="custom reason")
        assert event is not None
        assert event.reason == "custom reason"

    def test_payload_factory_returns_none_for_non_dict(self):
        assert _inactivity_followup_event_from_payload(None) is None  # type: ignore[arg-type]
        assert _inactivity_followup_event_from_payload("not a dict") is None  # type: ignore[arg-type]

    def test_wake_reason_factory_accepts_inactivity_type(self):
        event = _inactivity_followup_event_from_wake_reason(
            {"type": "inactivity_followup"},
        )
        assert isinstance(event, InactivityFollowup)
        assert event.reason == _DEFAULT_REASON

    def test_wake_reason_factory_rejects_other_types(self):
        assert (
            _inactivity_followup_event_from_wake_reason(
                {"type": "task_due", "task_id": 1},
            )
            is None
        )
        assert _inactivity_followup_event_from_wake_reason("not a dict") is None
        assert _inactivity_followup_event_from_wake_reason(None) is None


# ===========================================================================
# 2. Handler side effects
# ===========================================================================


class TestInactivityHandler:
    @pytest.mark.anyio
    async def test_handler_pushes_notification_and_logs(self):
        cm = MagicMock()
        cm.notifications_bar = NotificationBar()
        cm._session_logger = MagicMock()

        event = InactivityFollowup(reason="Time to follow up.")
        result = await _handle_inactivity_followup_event(event, cm)

        assert result is True
        assert len(cm.notifications_bar.notifications) == 1
        notif = cm.notifications_bar.notifications[0]
        assert notif.type == "Inactivity"
        # Reason text must be in the notification body
        assert "Time to follow up." in notif.content
        cm._session_logger.info.assert_called_once()

    def test_notification_text_includes_variant_guidance(self):
        text = _inactivity_notification_text(InactivityFollowup(reason="hi"))
        assert "hi" in text
        # Brain should know to inspect transcript history to choose variant
        assert "transcript" in text.lower()

    def test_notification_text_includes_lifecycle_primitives(self):
        text = _inactivity_notification_text(InactivityFollowup(reason="hi"))
        # Brain must know it can opt-out / cancel termination via primitives
        assert "terminate_self" in text
        assert "cancel_self_termination" in text


# ===========================================================================
# 3. Registration
# ===========================================================================


class TestInactivityRegistration:
    def test_inactivity_followup_is_registered(self):
        assert (
            InactivityFollowup in EventHandler._registry
        ), "InactivityFollowup handler should be registered"


# ===========================================================================
# 4. activity_sync.touch_assistant_activity
# ===========================================================================


class TestTouchAssistantActivity:
    def test_returns_false_when_assistant_id_missing(self):
        assert touch_assistant_activity(None) is False

    def test_returns_false_when_assistant_id_not_int(self):
        assert touch_assistant_activity("not-an-int") is False

    def test_returns_false_when_orchestra_url_missing(self):
        with (
            patch(
                "unity.transcript_manager.activity_sync._base_url",
                return_value=None,
            ),
            patch(
                "unity.transcript_manager.activity_sync._admin_key",
                return_value="key",
            ),
        ):
            assert touch_assistant_activity(42) is False

    def test_returns_false_when_admin_key_missing(self):
        with (
            patch(
                "unity.transcript_manager.activity_sync._base_url",
                return_value="http://orchestra.test",
            ),
            patch(
                "unity.transcript_manager.activity_sync._admin_key",
                return_value=None,
            ),
        ):
            assert touch_assistant_activity(42) is False

    def test_posts_to_correct_url_with_auth(self):
        captured: dict = {}

        def fake_post(url, headers=None, timeout=None, **kwargs):
            captured["url"] = url
            captured["headers"] = headers
            resp = MagicMock()
            resp.status_code = 200
            return resp

        fake_http = MagicMock()
        fake_http.post = fake_post

        with (
            patch(
                "unity.transcript_manager.activity_sync._base_url",
                return_value="http://orchestra.test",
            ),
            patch(
                "unity.transcript_manager.activity_sync._admin_key",
                return_value="test-admin-key",
            ),
            patch.dict("sys.modules", {"unify.utils": MagicMock(http=fake_http)}),
        ):
            assert touch_assistant_activity(42) is True

        assert (
            captured["url"] == "http://orchestra.test/admin/assistant/42/touch-activity"
        )
        assert captured["headers"] == {"Authorization": "Bearer test-admin-key"}

    def test_returns_false_on_non_2xx(self):
        def fake_post(url, headers=None, timeout=None, **kwargs):
            resp = MagicMock()
            resp.status_code = 500
            resp.text = "boom"
            return resp

        fake_http = MagicMock()
        fake_http.post = fake_post

        with (
            patch(
                "unity.transcript_manager.activity_sync._base_url",
                return_value="http://orchestra.test",
            ),
            patch(
                "unity.transcript_manager.activity_sync._admin_key",
                return_value="key",
            ),
            patch.dict("sys.modules", {"unify.utils": MagicMock(http=fake_http)}),
        ):
            assert touch_assistant_activity(42) is False

    def test_swallows_exceptions(self):
        def fake_post(*_a, **_kw):
            raise RuntimeError("network down")

        fake_http = MagicMock()
        fake_http.post = fake_post

        with (
            patch(
                "unity.transcript_manager.activity_sync._base_url",
                return_value="http://orchestra.test",
            ),
            patch(
                "unity.transcript_manager.activity_sync._admin_key",
                return_value="key",
            ),
            patch.dict("sys.modules", {"unify.utils": MagicMock(http=fake_http)}),
        ):
            # Should not raise
            assert touch_assistant_activity(42) is False


# ===========================================================================
# 5. Lifecycle helpers (terminate / cancel-termination via orchestra)
# ===========================================================================


class TestLifecycleSyncHelpers:
    """terminate_assistant_via_orchestra + cancel_assistant_termination_via_orchestra."""

    def _make_fake_http(self, *, status: int = 200):
        captured: dict = {}

        def fake_post(url, headers=None, timeout=None, **kwargs):
            captured["url"] = url
            captured["headers"] = headers
            resp = MagicMock()
            resp.status_code = status
            resp.text = "" if status < 400 else "boom"
            return resp

        http_module = MagicMock()
        http_module.post = fake_post
        return http_module, captured

    def test_terminate_posts_to_correct_url(self):
        http_module, captured = self._make_fake_http(status=200)
        with (
            patch(
                "unity.transcript_manager.activity_sync._base_url",
                return_value="http://orchestra.test",
            ),
            patch(
                "unity.transcript_manager.activity_sync._admin_key",
                return_value="test-admin-key",
            ),
            patch.dict("sys.modules", {"unify.utils": MagicMock(http=http_module)}),
        ):
            assert terminate_assistant_via_orchestra(42) is True

        assert captured["url"] == "http://orchestra.test/admin/assistant/42/terminate"
        assert captured["headers"] == {"Authorization": "Bearer test-admin-key"}

    def test_cancel_termination_posts_to_correct_url(self):
        http_module, captured = self._make_fake_http(status=200)
        with (
            patch(
                "unity.transcript_manager.activity_sync._base_url",
                return_value="http://orchestra.test",
            ),
            patch(
                "unity.transcript_manager.activity_sync._admin_key",
                return_value="test-admin-key",
            ),
            patch.dict("sys.modules", {"unify.utils": MagicMock(http=http_module)}),
        ):
            assert cancel_assistant_termination_via_orchestra(42) is True

        assert (
            captured["url"]
            == "http://orchestra.test/admin/assistant/42/cancel-termination"
        )
        assert captured["headers"] == {"Authorization": "Bearer test-admin-key"}

    def test_terminate_returns_false_on_missing_config(self):
        with (
            patch(
                "unity.transcript_manager.activity_sync._base_url",
                return_value=None,
            ),
            patch(
                "unity.transcript_manager.activity_sync._admin_key",
                return_value="key",
            ),
        ):
            assert terminate_assistant_via_orchestra(42) is False
        with (
            patch(
                "unity.transcript_manager.activity_sync._base_url",
                return_value="http://orchestra.test",
            ),
            patch(
                "unity.transcript_manager.activity_sync._admin_key",
                return_value=None,
            ),
        ):
            assert terminate_assistant_via_orchestra(42) is False

    def test_terminate_returns_false_on_non_2xx(self):
        http_module, _ = self._make_fake_http(status=500)
        with (
            patch(
                "unity.transcript_manager.activity_sync._base_url",
                return_value="http://orchestra.test",
            ),
            patch(
                "unity.transcript_manager.activity_sync._admin_key",
                return_value="key",
            ),
            patch.dict("sys.modules", {"unify.utils": MagicMock(http=http_module)}),
        ):
            assert terminate_assistant_via_orchestra(42) is False

    def test_cancel_termination_returns_false_on_non_2xx(self):
        http_module, _ = self._make_fake_http(status=500)
        with (
            patch(
                "unity.transcript_manager.activity_sync._base_url",
                return_value="http://orchestra.test",
            ),
            patch(
                "unity.transcript_manager.activity_sync._admin_key",
                return_value="key",
            ),
            patch.dict("sys.modules", {"unify.utils": MagicMock(http=http_module)}),
        ):
            assert cancel_assistant_termination_via_orchestra(42) is False

    def test_swallow_exceptions(self):
        def raising_post(*_a, **_kw):
            raise RuntimeError("network down")

        http_module = MagicMock()
        http_module.post = raising_post

        with (
            patch(
                "unity.transcript_manager.activity_sync._base_url",
                return_value="http://orchestra.test",
            ),
            patch(
                "unity.transcript_manager.activity_sync._admin_key",
                return_value="key",
            ),
            patch.dict("sys.modules", {"unify.utils": MagicMock(http=http_module)}),
        ):
            assert terminate_assistant_via_orchestra(42) is False
            assert cancel_assistant_termination_via_orchestra(42) is False


# ===========================================================================
# 6. CommsPrimitives.terminate_self / cancel_self_termination
# ===========================================================================


class TestCommsLifecyclePrimitives:
    """The brain-callable wrappers on CommsPrimitives."""

    def test_lifecycle_methods_in_primitive_methods(self):
        """They must be discoverable by the function-manager registry."""
        assert "terminate_self" in CommsPrimitives._PRIMITIVE_METHODS
        assert "cancel_self_termination" in CommsPrimitives._PRIMITIVE_METHODS

    @pytest.mark.anyio
    async def test_terminate_self_calls_helper_and_returns_shape(self):
        primitives = CommsPrimitives.__new__(CommsPrimitives)
        primitives._cm = None
        primitives._event_broker = MagicMock()

        fake_session = MagicMock()
        fake_session.assistant.agent_id = 99

        with (
            patch("unity.comms.primitives.SESSION_DETAILS", fake_session),
            patch(
                "unity.transcript_manager.activity_sync.terminate_assistant_via_orchestra",
                return_value=True,
            ) as mock_terminate,
        ):
            result = await primitives.terminate_self()

        mock_terminate.assert_called_once_with(99)
        assert result == {"success": True, "assistant_id": 99}

    @pytest.mark.anyio
    async def test_cancel_self_termination_calls_helper_and_returns_shape(self):
        primitives = CommsPrimitives.__new__(CommsPrimitives)
        primitives._cm = None
        primitives._event_broker = MagicMock()

        fake_session = MagicMock()
        fake_session.assistant.agent_id = 99

        with (
            patch("unity.comms.primitives.SESSION_DETAILS", fake_session),
            patch(
                "unity.transcript_manager.activity_sync.cancel_assistant_termination_via_orchestra",
                return_value=True,
            ) as mock_cancel,
        ):
            result = await primitives.cancel_self_termination()

        mock_cancel.assert_called_once_with(99)
        assert result == {"success": True, "assistant_id": 99}

    @pytest.mark.anyio
    async def test_lifecycle_propagates_failure(self):
        primitives = CommsPrimitives.__new__(CommsPrimitives)
        primitives._cm = None
        primitives._event_broker = MagicMock()

        fake_session = MagicMock()
        fake_session.assistant.agent_id = 99

        with (
            patch("unity.comms.primitives.SESSION_DETAILS", fake_session),
            patch(
                "unity.transcript_manager.activity_sync.terminate_assistant_via_orchestra",
                return_value=False,
            ),
        ):
            result = await primitives.terminate_self()
        assert result == {"success": False, "assistant_id": 99}

    @pytest.mark.anyio
    async def test_lifecycle_handles_missing_agent_id(self):
        primitives = CommsPrimitives.__new__(CommsPrimitives)
        primitives._cm = None
        primitives._event_broker = MagicMock()

        fake_session = MagicMock()
        fake_session.assistant.agent_id = None

        with (
            patch("unity.comms.primitives.SESSION_DETAILS", fake_session),
            patch(
                "unity.transcript_manager.activity_sync.terminate_assistant_via_orchestra",
                return_value=False,
            ),
        ):
            result = await primitives.terminate_self()
        assert result == {"success": False, "assistant_id": None}
