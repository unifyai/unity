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
pytestmark = pytest.mark.eval

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


@pytest.mark.asyncio
class TestVerboseSMSRelay:
    """The fast brain should distill verbose SMS notifications into
    brief spoken relays."""

    async def test_long_sms_relayed_concisely(self):
        notification = (
            "SMS from Marcus Rivera: Hey Sarah, just wanted to let you "
            "know that the quarterly board meeting has been rescheduled "
            "from next Tuesday at 2pm to Thursday at 10am. The venue is "
            "also changing from the downtown office to the Hilton "
            "conference center on 5th Avenue. Please update your "
            "calendar accordingly and let me know if Thursday works for you."
        )
        response = await _ask_with_notification(notification)

        assert_concise(response, max_words=55, context="long SMS relay")
        assert_no_jargon(response, context="long SMS relay")
        assert_contains_any(
            response,
            ["marcus", "thursday", "10", "rescheduled", "board meeting"],
            context="should mention key facts",
        )

    async def test_multi_topic_sms_stays_brief(self):
        notification = (
            "SMS from Priya Sharma: Hi Sarah, two things — first, the "
            "client demo is confirmed for Friday at 3pm, I've sent the "
            "calendar invite. Second, the design team finished the new "
            "mockups and uploaded them to the shared drive under "
            "Projects/Q3-Redesign/Final. Let me know when you've had a "
            "chance to review them."
        )
        response = await _ask_with_notification(notification)

        assert_concise(response, max_words=55, context="multi-topic SMS")
        assert_no_jargon(response, context="multi-topic SMS")
        assert_contains_any(
            response,
            ["priya", "demo", "friday", "mockups", "design"],
            context="should mention at least one key topic",
        )


# ─────────────────────────────────────────────────────────────────────
# Tests: Verbose email relay
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestVerboseEmailRelay:
    """The fast brain should concisely summarise verbose email
    notifications."""

    async def test_long_email_relayed_concisely(self):
        notification = (
            "Email from James Watson: Subject: Updated Contract Terms — "
            "Hi Sarah, following our discussion last week I've revised "
            "the contract to reflect the new payment schedule (net-45 "
            "instead of net-30), added the liability cap at $500K as "
            "agreed, and included the additional IP assignment clause "
            "your legal team requested. The redlined version is attached. "
            "Please review sections 4.2 and 7.1 specifically and let me "
            "know if anything needs further adjustment before we send "
            "to their counsel."
        )
        response = await _ask_with_notification(notification)

        assert_concise(response, max_words=70, context="long email relay")
        assert_no_jargon(response, context="long email relay")
        assert_contains_any(
            response,
            ["james", "contract", "email"],
            context="should mention sender or topic",
        )

    async def test_email_with_attachment_mention(self):
        notification = (
            "Email from Lisa Park: Subject: Q3 Revenue Report — "
            "Attached is the final Q3 revenue report. Total revenue came "
            "in at $4.2M, which is 18% above target. The services "
            "division drove most of the upside with a 32% beat. I've "
            "included a breakdown by segment and region on pages 3-5. "
            "Happy to walk through the details whenever you're free."
        )
        response = await _ask_with_notification(
            notification,
            prior_turns=[
                {"role": "user", "content": "I'm waiting on the Q3 numbers from Lisa."},
                {"role": "assistant", "content": "I'll keep an eye out."},
            ],
        )

        assert_concise(response, max_words=60, context="email with attachment")
        assert_no_jargon(response, context="email with attachment")
        assert_contains_any(
            response,
            ["lisa", "revenue", "q3", "4.2"],
            context="should mention sender or key figure",
        )


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


def _build_contact_call_prompt(**overrides) -> str:
    """Build a fast brain prompt for a call with a non-boss contact."""
    defaults = {
        "bio": "I am a virtual assistant working for a tech startup.",
        "assistant_name": "Alex",
        "boss_first_name": "Sarah",
        "boss_surname": "Chen",
        "boss_phone_number": "+15551234567",
        "boss_email_address": "sarah@meridianlabs.com",
        "boss_bio": "CEO of Meridian Labs, focused on AI products.",
        "is_boss_user": False,
        "contact_first_name": "Marcus",
        "contact_surname": "Rivera",
        "contact_phone_number": "+15559876543",
        "contact_email": "marcus@clientcorp.com",
        "contact_bio": "VP of Engineering at ClientCorp.",
    }
    defaults.update(overrides)
    return build_voice_agent_prompt(**defaults).flatten()


async def _ask_with_tagged_comms(
    tagged_msg: str,
    user_message: str | None = None,
    prior_turns: list[dict] | None = None,
    is_boss: bool = False,
) -> str:
    """Simulate a call where a tagged comms message arrives."""
    prompt = _build_boss_prompt() if is_boss else _build_contact_call_prompt()
    messages: list[dict] = [{"role": "system", "content": prompt}]
    if prior_turns:
        messages.extend(prior_turns)
    messages.append({"role": "system", "content": tagged_msg})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    client = new_llm_client(model=FAST_BRAIN_MODEL, reasoning_effort="low")
    return (await client.generate(messages=messages)).strip()


@pytest.mark.asyncio
class TestParticipantCommsTagFormat:
    """Verify the fast brain handles [SMS from Name] / [Email from Name]
    tagged messages concisely — on any call, not just boss calls."""

    async def test_sms_from_caller_during_call(self):
        """When the person on the call sends an SMS, the fast brain should
        mention it naturally and concisely."""
        tagged = "[SMS from Marcus Rivera] Just sent you the contract PDF via email, check when you get a chance."
        response = await _ask_with_tagged_comms(
            tagged,
            prior_turns=[
                {
                    "role": "user",
                    "content": "Hey, I'll send the contract over shortly.",
                },
                {"role": "assistant", "content": "Sounds good, I'll keep an eye out."},
            ],
        )

        assert_concise(response, max_words=35, context="SMS from caller")
        assert_no_jargon(response, context="SMS from caller")
        assert_contains_any(
            response,
            ["contract", "email", "pdf", "sent"],
            context="should acknowledge the SMS content",
        )

    async def test_email_from_caller_during_call(self):
        """When the person on the call sends an email mid-conversation."""
        tagged = "[Email from Marcus Rivera] Q3 Review Deck — Here's the deck we discussed, 15 slides covering revenue, pipeline, and headcount."
        response = await _ask_with_tagged_comms(tagged)

        assert_concise(response, max_words=35, context="email from caller")
        assert_no_jargon(response, context="email from caller")
        assert_contains_any(
            response,
            ["email", "deck", "q3", "slides"],
            context="should mention the email topic",
        )

    async def test_tagged_comms_no_tag_leakage(self):
        """The fast brain must never say 'SMS from' or mention the tag format."""
        tagged = (
            "[SMS from Marcus Rivera] Can you also book the conference room for 3pm?"
        )
        response = await _ask_with_tagged_comms(tagged)

        assert_no_jargon(response, context="tag leakage check")
        lower = response.lower()
        assert "[sms" not in lower, f"Tag format leaked: {response}"
        assert "[email" not in lower, f"Tag format leaked: {response}"
        assert "[message" not in lower, f"Tag format leaked: {response}"

    async def test_boss_call_tagged_comms_also_works(self):
        """Tagged comms should work on boss calls too (superset behavior)."""
        tagged = "[SMS from Sarah Chen] Grabbing coffee, back in 5."
        response = await _ask_with_tagged_comms(
            tagged,
            is_boss=True,
            prior_turns=[
                {"role": "user", "content": "I need to step out for a sec."},
                {"role": "assistant", "content": "No problem, take your time."},
            ],
        )

        assert_concise(response, max_words=25, context="boss SMS on boss call")
        assert_no_jargon(response, context="boss SMS on boss call")


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
