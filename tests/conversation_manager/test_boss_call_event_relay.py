"""
tests/conversation_manager/test_boss_call_event_relay.py
=========================================================

Eval tests verifying that the fast brain relays system notifications
concisely when the boss is on the call.

In boss-on-call mode the fast brain receives raw system events (SMS,
emails, action progress, etc.) as [notification] messages.  These raw
events can be quite verbose — multiple sentences of detail.  On a live
phone call the fast brain should distill each notification into a
brief, natural spoken relay (1-2 sentences) rather than parroting the
full content.

Each test feeds a verbose notification into the conversation context
and checks that the response is:
  1. Concise (phone-call brevity)
  2. Accurate (key facts preserved)
  3. Natural (no internal jargon, no mention of notifications)
"""

from __future__ import annotations

import pytest

from unity.common.llm_client import new_llm_client
from unity.conversation_manager.prompt_builders import build_voice_agent_prompt

# Mark every test in this file as eval — these exercise LLM reasoning,
# not deterministic infrastructure.
pytestmark = [pytest.mark.eval, pytest.mark.llm_call]

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

FAST_BRAIN_MODEL = "gpt-5-mini@openai"

INTERNAL_JARGON = [
    "notification",
    "event",
    "system",
    "backend",
    "slow brain",
    "fast brain",
    "internal",
    "ipc",
    "event broker",
    "socket",
    "[notification]",
]


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _build_boss_prompt(**overrides) -> str:
    defaults = {
        "bio": "I am a virtual assistant working for a tech startup.",
        "assistant_name": "Alex",
        "boss_first_name": "Sarah",
        "boss_surname": "Chen",
        "boss_phone_number": "+15551234567",
        "boss_email_address": "sarah@meridianlabs.com",
        "boss_bio": "CEO of Meridian Labs, focused on AI products.",
        "is_boss_user": True,
    }
    defaults.update(overrides)
    return build_voice_agent_prompt(**defaults).flatten()


async def _ask_with_notification(
    notification: str,
    user_message: str | None = None,
    prior_turns: list[dict] | None = None,
) -> str:
    """Simulate a boss call where a notification arrives.

    Builds a message sequence: system prompt, optional prior turns,
    the [notification], then an optional follow-up user utterance.
    """
    system_prompt = _build_boss_prompt()
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    if prior_turns:
        messages.extend(prior_turns)

    messages.append(
        {
            "role": "system",
            "content": f"[notification] {notification}",
        },
    )

    if user_message:
        messages.append({"role": "user", "content": user_message})

    client = new_llm_client(model=FAST_BRAIN_MODEL, reasoning_effort="low")
    return (await client.generate(messages=messages)).strip()


def assert_concise(response: str, max_words: int = 35, context: str = "") -> None:
    word_count = len(response.split())
    assert word_count <= max_words, (
        f"Response too verbose for a phone call ({word_count} words, max {max_words})!\n"
        f"Full response: {response}\n"
        f"{f'Context: {context}' if context else ''}"
    )


def assert_no_jargon(response: str, context: str = "") -> None:
    lower = response.lower()
    for term in INTERNAL_JARGON:
        assert term.lower() not in lower, (
            f"Internal jargon '{term}' leaked into spoken response!\n"
            f"Full response: {response}\n"
            f"{f'Context: {context}' if context else ''}"
        )


def assert_contains_any(
    response: str,
    keywords: list[str],
    context: str = "",
) -> None:
    lower = response.lower()
    found = [kw for kw in keywords if kw.lower() in lower]
    assert found, (
        f"None of the expected keywords {keywords} found in response!\n"
        f"Full response: {response}\n"
        f"{f'Context: {context}' if context else ''}"
    )


# ─────────────────────────────────────────────────────────────────────
# Tests: Verbose SMS relay
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
# Tests: Action progress relay
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestActionProgressRelay:
    """The fast brain should relay action progress updates concisely,
    especially when the user is waiting."""

    async def test_verbose_progress_condensed(self):
        prior_turns = [
            {
                "role": "user",
                "content": "Can you find me a good Italian restaurant nearby?",
            },
            {"role": "assistant", "content": "Let me look into that."},
        ]
        notification = (
            "Action progress: Searching the web for highly rated Italian "
            "restaurants within a 5 mile radius of the user's current "
            "location in downtown San Francisco. Found 12 initial results "
            "from Google Maps and Yelp. Now filtering by rating (4+ stars) "
            "and availability for tonight. 5 candidates remaining after "
            "initial filter."
        )
        response = await _ask_with_notification(
            notification,
            prior_turns=prior_turns,
        )

        assert_concise(response, max_words=45, context="progress update")
        assert_no_jargon(response, context="progress update")

    async def test_completion_with_results_condensed(self):
        prior_turns = [
            {"role": "user", "content": "Can you look up John Davis's contact info?"},
            {"role": "assistant", "content": "Checking now."},
        ]
        notification = (
            "Action completed successfully: Contact lookup for John Davis "
            "returned the following results — Phone: +1 (555) 987-6543, "
            "Email: john.davis@boardmembers.org, Company: Meridian Labs "
            "Board of Directors, Title: Independent Board Member since "
            "2021. Last contacted via email on January 15th regarding "
            "the annual governance review."
        )
        response = await _ask_with_notification(
            notification,
            prior_turns=prior_turns,
        )

        assert_concise(response, max_words=40, context="completion with results")
        assert_no_jargon(response, context="completion with results")
        assert_contains_any(
            response,
            ["john", "555", "987", "davis"],
            context="should mention the contact or their details",
        )


# ─────────────────────────────────────────────────────────────────────
# Tests: No jargon leakage
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestNoJargonLeakage:
    """The fast brain must never expose internal system terminology
    when relaying notifications to the boss."""

    async def test_sms_relay_no_notification_mention(self):
        notification = "SMS from Tom: Running 10 minutes late, stuck in traffic."
        response = await _ask_with_notification(notification)

        assert_no_jargon(response, context="simple SMS relay")
        assert_concise(response, max_words=25, context="simple SMS relay")

    async def test_action_relay_no_system_mention(self):
        prior_turns = [
            {"role": "user", "content": "Send an email to James about the contract."},
            {"role": "assistant", "content": "On it."},
        ]
        notification = "Action completed successfully: Email sent to James Watson."
        response = await _ask_with_notification(
            notification,
            prior_turns=prior_turns,
        )

        assert_no_jargon(response, context="action completion relay")
        assert_concise(response, max_words=25, context="action completion relay")


# ─────────────────────────────────────────────────────────────────────
# Tests: Participant comms with [SMS from Name] tags
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
# Tests: Fast brain says nothing for irrelevant notifications
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestSayNothingForIrrelevantNotifications:
    """Verify the fast brain produces empty or near-empty output when
    a notification is not relevant to the current conversation.

    The "say nothing" behavior is critical for avoiding noise — not
    every system event warrants speech on a phone call."""

    async def test_irrelevant_progress_during_unrelated_conversation(self):
        """An action progress update about a task the user isn't actively
        asking about should produce at most a brief mention, not a detailed
        relay of the progress content."""
        response = await _ask_with_notification(
            "Action progress: Syncing calendar entries from Google Calendar. "
            "Found 47 entries, processing batch 2 of 5.",
            prior_turns=[
                {"role": "user", "content": "So tell me about your background."},
                {
                    "role": "assistant",
                    "content": "I've been working as a virtual assistant for about 3 years.",
                },
            ],
        )
        assert_concise(
            response,
            max_words=35,
            context="irrelevant progress during unrelated chat",
        )

    async def test_redundant_notification_after_already_relayed(self):
        """If the fast brain already mentioned the same information, a
        follow-up notification with the same content should produce at
        most a brief progress line, not a full re-relay."""
        response = await _ask_with_notification(
            "Action progress: Still searching for Italian restaurants. "
            "Currently at 8 results.",
            prior_turns=[
                {
                    "role": "user",
                    "content": "Find me an Italian restaurant nearby.",
                },
                {"role": "assistant", "content": "Looking into that now."},
                {
                    "role": "system",
                    "content": "[notification] Action progress: Searching for Italian restaurants nearby. Found 5 initial results.",
                },
                {
                    "role": "assistant",
                    "content": "I'm searching for Italian places nearby — found a few options so far.",
                },
            ],
        )
        assert_concise(
            response,
            max_words=30,
            context="redundant progress update",
        )

    async def test_trivial_system_event_produces_minimal_speech(self):
        """A purely internal system event (e.g. contacts synced) should
        produce at most a brief acknowledgment, not a verbose relay."""
        response = await _ask_with_notification(
            "Contacts synced: manual sync",
            prior_turns=[
                {"role": "user", "content": "Hey, how's it going?"},
                {"role": "assistant", "content": "Good! What can I help with?"},
            ],
        )
        word_count = len(response.split())
        assert word_count <= 15, (
            f"Fast brain should say little or nothing for a trivial system sync "
            f"event, but produced {word_count} words: {response}"
        )
