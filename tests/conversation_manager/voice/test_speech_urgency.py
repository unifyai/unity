"""
tests/conversation_manager/voice/test_speech_urgency.py
========================================================

Eval tests for the SpeechUrgencyEvaluator — a structured-output sidecar that
classifies incoming user utterances as urgent (preempt the slow brain) or not
(let the queue proceed normally).

Each test constructs a scenario with a user utterance and slow-brain context,
then asserts on the `urgent` boolean returned by the model.
"""

import pytest

from unity.conversation_manager.domains.speech_urgency import SpeechUrgencyEvaluator
from unity.settings import SETTINGS

pytestmark = pytest.mark.eval

MODEL = SETTINGS.conversation.FAST_BRAIN_MODEL


@pytest.fixture
def evaluator():
    return SpeechUrgencyEvaluator(model=MODEL)


# ---- Urgent scenarios ----


@pytest.mark.asyncio
async def test_fresh_directive_vs_noise_event(evaluator):
    """User gives a new actionable directive while slow brain handles a system event."""
    decision = await evaluator.evaluate(
        utterance="Open the browser",
        origin_event="WebCamStarted",
        elapsed_seconds=15.0,
        actions_summary="none",
    )
    assert decision.urgent is True


@pytest.mark.asyncio
async def test_redirect_cancels_current_work(evaluator):
    """User explicitly redirects while slow brain is working on something else."""
    decision = await evaluator.evaluate(
        utterance="Stop, do something else instead",
        origin_event="InboundPhoneUtterance",
        elapsed_seconds=10.0,
        actions_summary="act: 'open the browser'",
    )
    assert decision.urgent is True


@pytest.mark.asyncio
async def test_fresh_directive_vs_stale_small_talk(evaluator):
    """User gives an actionable directive while slow brain processes old small talk."""
    decision = await evaluator.evaluate(
        utterance="Click the submit button",
        origin_event="InboundPhoneUtterance",
        elapsed_seconds=12.0,
        actions_summary="none",
    )
    assert decision.urgent is True


@pytest.mark.asyncio
async def test_directive_while_processing_low_priority_event(evaluator):
    """User wants action while slow brain is stuck on a screen share event."""
    decision = await evaluator.evaluate(
        utterance="Go to costar.com",
        origin_event="ScreenShareStarted",
        elapsed_seconds=20.0,
        actions_summary="none",
    )
    assert decision.urgent is True


# ---- Not urgent scenarios ----


@pytest.mark.asyncio
async def test_small_talk_while_processing_user_action(evaluator):
    """User checks in while slow brain is executing their previous request."""
    decision = await evaluator.evaluate(
        utterance="How's it going?",
        origin_event="InboundPhoneUtterance",
        elapsed_seconds=8.0,
        actions_summary="act: 'send the email to John'",
    )
    assert decision.urgent is False


@pytest.mark.asyncio
async def test_acknowledgment(evaluator):
    """User acknowledges — should not preempt anything."""
    decision = await evaluator.evaluate(
        utterance="Sounds good",
        origin_event="InboundPhoneUtterance",
        elapsed_seconds=5.0,
        actions_summary="act: 'look up the restaurant reservations'",
    )
    assert decision.urgent is False


@pytest.mark.asyncio
async def test_additional_context_for_in_progress_work(evaluator):
    """User adds a detail to work already underway — not a new directive."""
    decision = await evaluator.evaluate(
        utterance="Make sure it's Chrome",
        origin_event="InboundPhoneUtterance",
        elapsed_seconds=3.0,
        actions_summary="act: 'open the browser'",
    )
    assert decision.urgent is False
