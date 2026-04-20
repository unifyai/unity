"""
tests/conversation_manager/voice/test_slow_brain_failure_surfacing.py
=====================================================================

Regression tests for surfacing slow-brain LLM failures to the fast brain.

Background
----------
When the ConversationManager's slow-brain turn raises a transient provider
error (e.g. Anthropic HTTP 529 ``overloaded_error`` → ``litellm.InternalServerError``),
unillm's ``retry_transient_400_async`` retries with exponential backoff. If
the provider remains unhealthy for the whole retry budget the exception
escapes ``_run_llm``. Before this fix the only visible outcome was a
``Slow-brain task failed: ...`` log line — the user's utterance was
silently dropped and ``ProactiveSpeech`` kept emitting "still looking…"
filler for a request the slow brain had given up on.

The fix (``ConversationManager._run_llm_with_failure_notification``) wraps
``_run_llm`` and, when a transient LLM error escapes during a voice mode
turn:

1. Publishes a ``FastBrainNotification`` with ``should_speak=True`` and
   an explicit ``response_text`` so the fast brain utters the apology via
   TTS directly (bypassing its own LLM, which may be hitting the same
   outage).
2. Cancels any pending proactive-speech cycle so it stops emitting "still
   looking" filler.
3. Re-raises the exception so the existing ``log_task_exc`` failure log
   is preserved.

These tests exercise the wrapper directly using a lightweight stand-in for
``ConversationManager`` so they stay fast and deterministic.
"""

from __future__ import annotations

import json
import types
from unittest.mock import AsyncMock, MagicMock

import litellm
import pytest

from unity.conversation_manager.conversation_manager import ConversationManager
from unity.conversation_manager.cm_types import Mode
from unity.conversation_manager.events import Event, FastBrainNotification

BOSS_CONTACT = {
    "contact_id": 1,
    "first_name": "Boss",
    "surname": "User",
    "email_address": "boss@test.com",
    "phone_number": "+15555551111",
}


def _make_anthropic_overloaded() -> litellm.InternalServerError:
    """Reproduce the Anthropic HTTP 529 ``overloaded_error`` seen on staging."""
    return litellm.InternalServerError(
        message=(
            'AnthropicError - {"type":"error","error":'
            '{"type":"overloaded_error","message":"Overloaded"},'
            '"request_id":"req_011CaDyjzVH6u63qtgmLzafV"}'
        ),
        model="claude-4.6-opus",
        llm_provider="anthropic",
    )


class _CMStub:
    """Minimal stand-in for ConversationManager.

    Carries only the attributes the wrapper and its helpers touch, and has
    the three real methods bound so we exercise the production code paths
    without pulling in ConversationManager's heavyweight initialisation
    (state managers, event broker subscriptions, etc.).
    """


def _make_cm_stub(
    *,
    mode: Mode,
    run_llm_side_effect=None,
    run_llm_return_value=None,
) -> _CMStub:
    cm = _CMStub()
    cm.mode = mode
    cm._session_logger = MagicMock()
    cm.event_broker = MagicMock()
    cm.event_broker.publish = AsyncMock(return_value=0)
    cm.get_active_contact = MagicMock(return_value=BOSS_CONTACT)
    cm.cancel_proactive_speech = AsyncMock()

    if run_llm_side_effect is not None:
        cm._run_llm = AsyncMock(side_effect=run_llm_side_effect)
    else:
        cm._run_llm = AsyncMock(return_value=run_llm_return_value)

    # Bind the real wrapper + helpers so the test exercises the production
    # code. ``_is_transient_llm_error`` is a staticmethod on
    # ConversationManager so we expose the raw function directly.
    cm._run_llm_with_failure_notification = types.MethodType(
        ConversationManager._run_llm_with_failure_notification,
        cm,
    )
    cm._notify_fast_brain_of_slow_brain_failure = types.MethodType(
        ConversationManager._notify_fast_brain_of_slow_brain_failure,
        cm,
    )
    cm._is_transient_llm_error = ConversationManager._is_transient_llm_error

    return cm


def _captured_fast_brain_notifications(
    publish_mock: AsyncMock,
    channel: str = "app:call:notification",
) -> list[FastBrainNotification]:
    """Decode every ``FastBrainNotification`` published on ``channel``."""
    results: list[FastBrainNotification] = []
    for call in publish_mock.call_args_list:
        args, kwargs = call
        chan = kwargs.get("channel") if "channel" in kwargs else args[0]
        payload = kwargs.get("message") if "message" in kwargs else args[1]
        if chan != channel:
            continue
        evt = Event.from_json(payload)
        if isinstance(evt, FastBrainNotification):
            results.append(evt)
    return results


# =============================================================================
# Wrapper behaviour on transient LLM failure (voice mode)
# =============================================================================


class TestSlowBrainFailureSurfacing:
    """Transient LLM failures in voice modes must reach the fast brain."""

    @pytest.mark.asyncio
    async def test_anthropic_overloaded_surfaces_fast_brain_notification(self):
        """Anthropic 529 in CALL mode publishes a speakable failure notification.

        This is the exact scenario from Rachel's staging call: the slow-brain
        turn triggered by an ``InboundUnifyMeetUtterance`` raised
        ``litellm.InternalServerError`` (Anthropic 529 ``overloaded_error``)
        after unillm retries were exhausted. Previously the failure was
        silently logged and the user was left in filler-only silence. Now
        it must produce a ``FastBrainNotification`` that the fast brain can
        utter via TTS to explicitly apologise and invite a retry.
        """
        exc = _make_anthropic_overloaded()
        cm = _make_cm_stub(mode=Mode.CALL, run_llm_side_effect=exc)

        with pytest.raises(litellm.InternalServerError):
            await cm._run_llm_with_failure_notification(
                trace_meta={"origin_event_name": "InboundUnifyMeetUtterance"},
            )

        cm._run_llm.assert_awaited_once_with(
            trace_meta={"origin_event_name": "InboundUnifyMeetUtterance"},
        )

        call_channel_notifs = _captured_fast_brain_notifications(
            cm.event_broker.publish,
            channel="app:call:notification",
        )
        comms_channel_notifs = _captured_fast_brain_notifications(
            cm.event_broker.publish,
            channel="app:comms:assistant_notification",
        )
        assert len(call_channel_notifs) == 1, (
            "Exactly one FastBrainNotification should be published on "
            "'app:call:notification' when the slow-brain turn fails with a "
            "transient provider error."
        )
        assert len(comms_channel_notifs) == 1, (
            "The failure notification must also be published on "
            "'app:comms:assistant_notification' to match the existing "
            "guide_voice_agent pathway, so any comms-subscribed consumers "
            "see the same signal."
        )

        notif = call_channel_notifs[0]
        assert notif.should_speak is True, (
            "Failure notification MUST set should_speak=True so the fast "
            "brain utters the apology via session.say() directly. Leaving "
            "it silent would reproduce the original bug where the user "
            "hears nothing after an LLM outage."
        )
        assert notif.response_text, (
            "response_text must be populated so the fast brain can speak "
            "the apology without a fresh LLM roundtrip — critical when the "
            "provider is the outage cause."
        )
        assert notif.source == "slow_brain_failure", (
            f"Expected source='slow_brain_failure', got {notif.source!r}. "
            "A dedicated source tag lets downstream handlers and analytics "
            "distinguish provider-error fillers from normal slow-brain "
            "guidance."
        )
        assert notif.contact == BOSS_CONTACT, (
            "Notification should carry the active call contact so the "
            "fast brain threads it into the correct conversation."
        )

    @pytest.mark.asyncio
    async def test_failure_notification_cancels_proactive_speech(self):
        """A failed slow-brain turn must stop the 'still looking' filler loop.

        ProactiveSpeech runs on its own cycle and will keep emitting
        "still looking into it" utterances for whichever request the slow
        brain is ostensibly handling. When the slow brain has given up, we
        must cancel the pending proactive cycle so the user stops hearing
        contradictory filler.
        """
        cm = _make_cm_stub(
            mode=Mode.CALL,
            run_llm_side_effect=_make_anthropic_overloaded(),
        )

        with pytest.raises(litellm.InternalServerError):
            await cm._run_llm_with_failure_notification()

        cm.cancel_proactive_speech.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_meet_mode_also_surfaces_failure(self):
        """MEET mode (e.g. Unify Meet / Google Meet) must surface failures too.

        The original bug was observed in a Unify Meet session, so MEET mode
        must take the same path as CALL mode.
        """
        cm = _make_cm_stub(
            mode=Mode.MEET,
            run_llm_side_effect=_make_anthropic_overloaded(),
        )

        with pytest.raises(litellm.InternalServerError):
            await cm._run_llm_with_failure_notification()

        notifs = _captured_fast_brain_notifications(cm.event_broker.publish)
        assert len(notifs) == 1
        assert notifs[0].should_speak is True
        assert notifs[0].source == "slow_brain_failure"

    @pytest.mark.asyncio
    async def test_service_unavailable_is_surfaced(self):
        """HTTP 503 ServiceUnavailable is also treated as transient.

        Anthropic's upstream-connect-reset errors surface as ``litellm.
        ServiceUnavailableError`` and represent the same user-visible
        condition as 529 Overloaded — the request cannot complete.
        """
        exc = litellm.ServiceUnavailableError(
            message=(
                "AnthropicException - upstream connect error or "
                "disconnect/reset before headers. reset reason: overflow"
            ),
            model="claude-4.6-opus",
            llm_provider="anthropic",
        )
        cm = _make_cm_stub(mode=Mode.CALL, run_llm_side_effect=exc)

        with pytest.raises(litellm.ServiceUnavailableError):
            await cm._run_llm_with_failure_notification()

        notifs = _captured_fast_brain_notifications(cm.event_broker.publish)
        assert len(notifs) == 1, (
            "ServiceUnavailableError must surface via FastBrainNotification "
            "using the same path as InternalServerError."
        )

    @pytest.mark.asyncio
    async def test_rate_limit_is_surfaced(self):
        """HTTP 429 RateLimit is also treated as transient."""
        exc = litellm.RateLimitError(
            message="Rate limit exceeded. Please retry after 1 second.",
            model="claude-4.6-opus",
            llm_provider="anthropic",
        )
        cm = _make_cm_stub(mode=Mode.CALL, run_llm_side_effect=exc)

        with pytest.raises(litellm.RateLimitError):
            await cm._run_llm_with_failure_notification()

        notifs = _captured_fast_brain_notifications(cm.event_broker.publish)
        assert len(notifs) == 1


# =============================================================================
# Wrapper behaviour on non-transient failures and success
# =============================================================================


class TestSlowBrainFailureSurfacingNonTriggers:
    """Scenarios that must NOT trigger the failure-notification path."""

    @pytest.mark.asyncio
    async def test_success_does_not_publish_failure_notification(self):
        """A successful slow-brain turn must not publish any failure filler."""
        cm = _make_cm_stub(
            mode=Mode.CALL,
            run_llm_return_value=["guide_voice_agent", "web_act"],
        )

        result = await cm._run_llm_with_failure_notification()

        assert result == ["guide_voice_agent", "web_act"]
        cm.event_broker.publish.assert_not_awaited()
        cm.cancel_proactive_speech.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_text_mode_does_not_publish_fast_brain_notification(self):
        """TEXT mode has no fast brain — publishing voice guidance is wrong.

        In TEXT mode the ``FastBrainNotification`` channel has no subscriber,
        so surfacing a voice apology makes no sense. The exception is still
        re-raised so the existing log path records the failure.
        """
        cm = _make_cm_stub(
            mode=Mode.TEXT,
            run_llm_side_effect=_make_anthropic_overloaded(),
        )

        with pytest.raises(litellm.InternalServerError):
            await cm._run_llm_with_failure_notification()

        cm.event_broker.publish.assert_not_awaited()
        cm.cancel_proactive_speech.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_transient_error_is_not_converted_to_filler(self):
        """Programmer errors (e.g. ``ValueError``) bubble up unchanged.

        We only want to tell the user "I'm having trouble thinking" when the
        cause is actually a provider outage. A bug in our own code should
        propagate so it's visible in crash logs — silencing it behind a
        polite apology would hide real defects.
        """
        cm = _make_cm_stub(
            mode=Mode.CALL,
            run_llm_side_effect=ValueError("bug in brain_spec construction"),
        )

        with pytest.raises(ValueError):
            await cm._run_llm_with_failure_notification()

        cm.event_broker.publish.assert_not_awaited()
        cm.cancel_proactive_speech.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cancellation_propagates_without_filler(self):
        """Task cancellation (e.g. preempting stale turns) is not a failure.

        The speech-urgency evaluator deliberately cancels in-flight slow-brain
        tasks when a fresher user utterance arrives. That is a *feature*, not
        a provider outage, and must not produce an apology notification.
        """
        import asyncio

        cm = _make_cm_stub(
            mode=Mode.CALL,
            run_llm_side_effect=asyncio.CancelledError(),
        )

        with pytest.raises(asyncio.CancelledError):
            await cm._run_llm_with_failure_notification()

        cm.event_broker.publish.assert_not_awaited()
        cm.cancel_proactive_speech.assert_not_awaited()


# =============================================================================
# Defensive behaviour of the notification helper itself
# =============================================================================


class TestFailureNotificationRobustness:
    """The failure path must never hide the underlying exception."""

    @pytest.mark.asyncio
    async def test_publish_failure_does_not_mask_llm_exception(self):
        """If publishing the notification itself fails, re-raise the LLM error.

        Worst case: the event broker is also unhealthy. We still want the
        original ``InternalServerError`` to surface to ``log_task_exc``, not
        the publish-time exception. The wrapper wraps ``_notify_*`` in
        ``contextlib.suppress(Exception)`` for exactly this reason.
        """
        exc = _make_anthropic_overloaded()
        cm = _make_cm_stub(mode=Mode.CALL, run_llm_side_effect=exc)
        cm.event_broker.publish = AsyncMock(side_effect=RuntimeError("broker down"))

        with pytest.raises(litellm.InternalServerError):
            await cm._run_llm_with_failure_notification()

    @pytest.mark.asyncio
    async def test_notification_payload_is_valid_json(self):
        """The serialised event must round-trip through ``Event.from_json``.

        Downstream subscribers deserialise via ``Event.from_json``; a
        malformed payload would cause silent drops on the receiving side.
        """
        cm = _make_cm_stub(
            mode=Mode.CALL,
            run_llm_side_effect=_make_anthropic_overloaded(),
        )

        with pytest.raises(litellm.InternalServerError):
            await cm._run_llm_with_failure_notification()

        publish_calls = [
            call
            for call in cm.event_broker.publish.call_args_list
            if call.args[0] == "app:call:notification"
        ]
        assert publish_calls, "Expected at least one call:notification publish."
        raw = publish_calls[0].args[1]
        payload = json.loads(raw)
        assert "event_name" in payload
        assert payload["event_name"] == "FastBrainNotification"
