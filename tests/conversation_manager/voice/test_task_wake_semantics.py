"""
Eval tests for task-wake semantics in fast-brain voice turns.

These tests exercise the fast brain with realistic wake notifications and
verify that it behaves like a natural virtual colleague rather than leaking
internal task-machine phrasing.
"""

from __future__ import annotations
import re

import pytest
from livekit.agents import llm

if not hasattr(llm, "Tool"):
    llm.Tool = object  # type: ignore[attr-defined]

from tests.helpers import _handle_project
from unity.common.llm_client import new_llm_client
from unity.conversation_manager.livekit_unify_adapter import UnifyLLM
from unity.conversation_manager.prompt_builders import (
    build_opening_greeting_messages,
    build_voice_agent_prompt,
)
from unity.settings import SETTINGS

pytestmark = pytest.mark.eval

MODEL = SETTINGS.conversation.FAST_BRAIN_MODEL

SCHEDULED_WAKE_NOTIFICATION = (
    "[notification] Background context: the scheduled task 'Morning briefing' "
    "is due now. Summary: Prepare and send the Monday client update based on "
    "current project status. This is a recurring task. Default is silent "
    "action unless the user is needed. The slow brain is handling the wake "
    "reason."
)

TRIGGERED_WAKE_NOTIFICATION = (
    "[notification] Background context: this phone call from Alice may relate "
    "to the task 'Invoice follow-up'. Summary: Help handle invoice-related "
    "requests from Alice. The slow brain is still deciding whether the trigger "
    "truly applies. Do not mention the task unless it naturally helps the "
    "conversation."
)

INTERNAL_JARGON_MARKERS = [
    "slow brain",
    "task_id",
    "trigger candidate",
    "notification",
    "activation revision",
    "wake reason",
    "mechanically matched",
    "semantic judgement",
    "visibility policy",
    "recurrence hint",
]

DEFERRAL_PATTERNS = [
    r"let me check",
    r"let me look",
    r"i'll check",
    r"i'll look",
    r"one moment",
    r"hold on",
    r"give me a (moment|second|sec)",
]


def _build_boss_prompt() -> str:
    """Return the fast-brain prompt for a boss call."""

    return build_voice_agent_prompt(
        bio="A helpful and efficient assistant.",
        assistant_name="Alex",
        boss_first_name="Alex",
        boss_surname="Demo",
        boss_phone_number="+15550001234",
        boss_email_address="alex@example.com",
        boss_bio="Founder and engineer. Prefers concise, practical updates.",
        is_boss_user=True,
        contact_rolling_summary=None,
    ).flatten()


def _build_alice_prompt() -> str:
    """Return the fast-brain prompt for a call with Alice."""

    return build_voice_agent_prompt(
        bio="A helpful and efficient assistant.",
        assistant_name="Alex",
        boss_first_name="Alex",
        boss_surname="Demo",
        boss_phone_number="+15550001234",
        boss_email_address="alex@example.com",
        boss_bio="Founder and engineer. Prefers concise, practical updates.",
        is_boss_user=False,
        contact_first_name="Alice",
        contact_surname="Example",
        contact_phone_number="+15550009999",
        contact_email="alice@example.com",
        contact_bio="Finance contact who often calls about invoice follow-ups.",
        contact_rolling_summary=None,
    ).flatten()


async def _get_response(system_prompt: str, messages: list[dict[str, str]]) -> str:
    """Run one fast-brain turn with the production voice adapter."""

    llm_instance = UnifyLLM(model=MODEL, reasoning_effort="low")
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content=system_prompt)
    for msg in messages:
        chat_ctx.add_message(role=msg["role"], content=msg["content"])
    chat_ctx.add_message(
        role="user",
        content="Respond as the assistant. Keep it concise and conversational (voice call).",
    )

    stream = llm_instance.chat(chat_ctx=chat_ctx)
    parts: list[str] = []
    try:
        async for chunk in stream:
            if hasattr(chunk, "delta") and chunk.delta:
                content = getattr(chunk.delta, "content", None)
                if content:
                    parts.append(content)
    finally:
        await stream.aclose()
    return "".join(parts)


async def _get_opening_greeting(
    system_prompt: str,
    history_messages: list[dict[str, str]],
) -> str:
    """Run the real startup greeting sidecar path used before first speech."""

    greeting_client = new_llm_client(
        model=MODEL,
        origin="fast_brain_greeting",
        reasoning_effort="low",
    )
    response = await greeting_client.generate(
        messages=build_opening_greeting_messages(
            system_prompt=system_prompt,
            history_messages=history_messages,
        ),
    )
    if isinstance(response, str):
        return response.strip()
    return str(response).strip()


def _contains_any(text: str, markers: list[str]) -> bool:
    """Return True when any marker appears case-insensitively in text."""

    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def _has_deferral(text: str) -> bool:
    """Return True when the response uses lookup-style deferral language."""

    lowered = text.lower().replace("\u2018", "'").replace("\u2019", "'")
    return any(re.search(pattern, lowered) for pattern in DEFERRAL_PATTERNS)


def _assert_no_internal_jargon(response: str) -> None:
    """Ensure the fast brain does not speak internal task-machine phrasing aloud."""

    lowered = response.lower()
    leaked = [marker for marker in INTERNAL_JARGON_MARKERS if marker in lowered]
    assert not leaked, (
        "Fast brain leaked internal wake-contract jargon.\n"
        f"Leaked markers: {leaked}\n"
        f"Response: {response}"
    )


class TestTaskWakeSemanticContexts:
    """Transcript-level evals for scheduled, triggered, and mixed-intent wakes."""

    @pytest.mark.asyncio
    @_handle_project
    async def test_scheduled_wake_answers_what_work_is_due(self):
        """A scheduled wake should let the assistant explain the due work directly."""

        response = await _get_response(
            _build_boss_prompt(),
            [
                {"role": "assistant", "content": SCHEDULED_WAKE_NOTIFICATION},
                {"role": "user", "content": "What were you about to work on?"},
            ],
        )

        assert _contains_any(
            response,
            ["morning briefing", "client update", "monday update"],
        ), (
            "Expected the fast brain to use the scheduled wake context when asked "
            "what work was due.\n"
            f"Response: {response}"
        )
        assert not _has_deferral(response), (
            "This question should be answered from wake context, not deferred.\n"
            f"Response: {response}"
        )
        _assert_no_internal_jargon(response)

    @pytest.mark.asyncio
    @_handle_project
    async def test_triggered_wake_explains_topic_naturally(self):
        """A trigger wake should explain the topic naturally without internal jargon."""

        response = await _get_response(
            _build_alice_prompt(),
            [
                {"role": "assistant", "content": TRIGGERED_WAKE_NOTIFICATION},
                {"role": "user", "content": "Sure, what is this about?"},
            ],
        )

        assert _contains_any(
            response,
            ["invoice", "follow-up", "alice"],
        ), (
            "Expected the fast brain to use the trigger wake context to explain "
            "the likely topic naturally.\n"
            f"Response: {response}"
        )
        _assert_no_internal_jargon(response)

    @pytest.mark.asyncio
    @_handle_project
    async def test_mixed_intent_keeps_live_request_in_focus(self):
        """Background task context should not override a direct live user request."""

        response = await _get_response(
            _build_boss_prompt(),
            [
                {"role": "assistant", "content": SCHEDULED_WAKE_NOTIFICATION},
                {
                    "role": "user",
                    "content": "Before anything else, can you check tomorrow's schedule for me?",
                },
            ],
        )

        assert _contains_any(
            response,
            [
                "schedule",
                "tomorrow",
                "calendar",
                "check",
                "moment",
                "let me",
                "one sec",
            ],
        ), (
            "Expected the fast brain to stay focused on the direct live request.\n"
            f"Response: {response}"
        )
        assert not _contains_any(
            response,
            ["morning briefing", "client update", "monday update"],
        ), (
            "The scheduled wake should remain in the background while the user asks "
            "for something else.\n"
            f"Response: {response}"
        )
        _assert_no_internal_jargon(response)


class TestTaskWakeFirstTurn:
    """Voice first-turn eval for preloaded scheduled wake context."""

    @pytest.mark.asyncio
    @_handle_project
    async def test_first_turn_stays_natural_with_scheduled_wake_context(self):
        """Preloaded scheduled context should not spoil the opening greeting."""

        response = await _get_opening_greeting(
            _build_boss_prompt(),
            [{"role": "system", "content": SCHEDULED_WAKE_NOTIFICATION}],
        )

        response_lower = response.lower().strip()
        assert (
            len(response) < 220
        ), f"First turn is too long for a natural greeting.\nResponse: {response}"
        assert not _contains_any(
            response,
            ["morning briefing", "client update", "scheduled task"],
        ), (
            "The first spoken turn should not proactively blurt the scheduled task.\n"
            f"Response: {response}"
        )
        assert not response_lower.startswith("got it"), (
            "There is nothing to acknowledge before the first spoken turn.\n"
            f"Response: {response}"
        )
        assert not any(
            indicator in response for indicator in ["- ", "* ", "1.", "2."]
        ), (
            "The first spoken turn should be a natural greeting, not a list.\n"
            f"Response: {response}"
        )
        _assert_no_internal_jargon(response)
