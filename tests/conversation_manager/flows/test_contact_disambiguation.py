"""
Flow regression tests: contact disambiguation should not loop.

These tests encode general behavioral invariants:
- When a tool result contains multiple plausible matches and asks the user to choose,
  the CM should surface enough detail for the user to choose.
- After asking the user to choose, the CM should WAIT (not re-delegate repeatedly).

We intentionally avoid overfitting to a specific prompt string by asserting:
- multiple candidate identifiers are present
- a disambiguation question is asked
- the CM does not call action tools repeatedly for the same input

The disambiguation result is injected at both the Actor and ContactManager levels
so the test is routing-agnostic: the LLM may use ``act`` (routed to Actor) or
``ask_about_contacts`` (routed directly to ContactManager.ask).
"""

from __future__ import annotations

import pytest

from tests.conversation_manager.cm_helpers import filter_events_by_type
from tests.conversation_manager.conftest import BOSS
from tests.helpers import _handle_project
from unity.common.async_tool_loop import SteerableToolHandle
from unity.conversation_manager.events import SMSReceived, SMSSent

pytestmark = pytest.mark.eval

_ACTION_TOOLS = {"act", "ask_about_contacts", "update_contacts", "query_past_transcripts"}

_DISAMBIGUATION_RESULT = (
    'Found 3 contacts named "Bob":\n'
    "1. Bob Miller — +15555550001\n"
    "2. Bob Chen — +15555550002\n"
    "3. Bob Williams — +15555550003\n\n"
    "Which Bob would you like to interact with?"
)


class _ImmediateResultHandle(SteerableToolHandle):
    """A minimal handle that is already complete with a deterministic result."""

    def __init__(self, result_text: str):
        self._result_text = result_text

    async def ask(self, question: str, **kwargs):  # type: ignore[override]
        return self

    def interject(self, message: str, **kwargs):  # type: ignore[override]
        return None

    async def pause(self):  # type: ignore[override]
        return None

    async def resume(self):  # type: ignore[override]
        return None

    def stop(self, reason: str | None = None, **kwargs):  # type: ignore[override]
        return None

    async def result(self):  # type: ignore[override]
        return self._result_text

    def done(self) -> bool:  # type: ignore[override]
        return True

    async def next_clarification(self):  # type: ignore[override]
        return {}

    async def next_notification(self):  # type: ignore[override]
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:  # type: ignore[override]
        return None


class _DeterministicActor:
    """Actor stub that returns a deterministic multi-candidate disambiguation result."""

    async def act(self, description: str, **kwargs):  # type: ignore[override]
        return _ImmediateResultHandle(_DISAMBIGUATION_RESULT)


async def _deterministic_contact_ask(text: str, **kwargs) -> _ImmediateResultHandle:
    """ContactManager.ask stub returning the same disambiguation result."""
    return _ImmediateResultHandle(_DISAMBIGUATION_RESULT)


@pytest.mark.asyncio
@_handle_project
async def test_contact_lookup_disambiguation_is_not_lossy_and_does_not_loop(
    initialized_cm,
):
    cm = initialized_cm

    # Inject a deterministic disambiguation result on *both* routing paths so the
    # test is agnostic to which tool the LLM picks:
    #   act              → Actor.act          (monkeypatched below)
    #   ask_about_contacts → ContactManager.ask (monkeypatched below)
    original_actor = cm.cm.actor
    original_contact_ask = cm.cm.contact_manager.ask
    # Ensure Actor watcher tasks publish onto the same broker the step driver patches.
    # (Some CM helper modules cache a broker singleton at import time.)
    from unity.conversation_manager.domains import managers_utils as _managers_utils

    original_broker = getattr(_managers_utils, "event_broker", None)
    cm.cm.actor = _DeterministicActor()
    cm.cm.contact_manager.ask = _deterministic_contact_ask
    try:
        _managers_utils.event_broker = cm.cm.event_broker
        result = await cm.step_until_wait(
            SMSReceived(
                contact=BOSS,
                content="Text Bob asking if he's free today.",
            ),
            max_steps=6,  # guard against infinite loops
        )
    finally:
        cm.cm.actor = original_actor
        cm.cm.contact_manager.ask = original_contact_ask
        if original_broker is not None:
            _managers_utils.event_broker = original_broker

    # The key regression: CM should not re-delegate repeatedly once it has
    # a valid multi-candidate result and should be asking the user to choose.
    action_calls = [t for t in cm.all_tool_calls if t in _ACTION_TOOLS]
    assert len(action_calls) <= 1, (
        "Expected CM to call an action tool at most once for this input, "
        f"then ask user and wait. all_tool_calls={cm.all_tool_calls}"
    )

    # We expect at least one SMS back to the boss/user.
    boss_sms = [
        e
        for e in filter_events_by_type(result.output_events, SMSSent)
        if e.contact.get("contact_id") == BOSS["contact_id"]
    ]
    assert boss_sms, "Expected an SMSSent response to the boss/user"
    combined = "\n".join(
        (e.content or "").strip() for e in boss_sms if (e.content or "").strip()
    )

    # Non-lossy disambiguation: include at least two candidate identifiers.
    # (Avoid exact formatting assertions; focus on the information needed to choose.)
    assert (
        ("Bob Miller" in combined and "Bob Chen" in combined)
        or ("Bob Miller" in combined and "Bob Williams" in combined)
        or ("Bob Chen" in combined and "Bob Williams" in combined)
    ), (
        "Expected response to include multiple candidate identifiers for disambiguation, "
        f"got: {combined!r}"
    )

    # Ask a disambiguation question.
    #
    # Avoid brittle string matching (exact phrasing varies). We only require that
    # the message contains an explicit question prompt after presenting candidates.
    #
    # Using "?" is a robust signal here: if the CM is asking the user to choose,
    # it should include a question mark somewhere in the combined outbound text.
    assert "?" in combined, (
        "Expected CM to ask the user to choose between candidates (a real question), "
        f"got: {combined!r}"
    )

    # CM should settle by calling wait within the step budget.
    assert cm.all_tool_calls and cm.all_tool_calls[-1] == "wait", (
        "Expected CM to end the turn by calling `wait` after asking the user. "
        f"all_tool_calls={cm.all_tool_calls}"
    )
