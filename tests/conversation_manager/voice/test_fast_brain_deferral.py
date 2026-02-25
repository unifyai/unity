"""
tests/conversation_manager/voice/test_fast_brain_deferral.py
=============================================================

Tests for Fast Brain (Voice Agent) deferral behavior.

The fast brain handles real-time voice conversation. It should defer to the
slow brain for data lookups instead of hallucinating answers.

This file tests "false positive" prevention - scenarios where the fast brain
might be tempted to hallucinate but should defer with natural language.
"""

from __future__ import annotations

import re
import pytest

from livekit.agents import llm

from unity.conversation_manager.livekit_unify_adapter import UnifyLLM
from unity.conversation_manager.prompt_builders import build_voice_agent_prompt
from unity.settings import SETTINGS

MODEL_TTS = SETTINGS.conversation.FAST_BRAIN_MODEL

# Patterns indicating proper deferral (case-insensitive)
# These patterns indicate the assistant is either:
# 1. Going to check/lookup data (deferred to slow brain)
# 2. Acknowledging it doesn't have the data immediately
DEFERRAL_PATTERNS = [
    # Active "I'm on it" language — the only acceptable deferral style.
    # The fast brain should imply it IS fetching the data, not that it CAN'T.
    r"let me check",
    r"let me look",
    r"i'll check",
    r"i'll look",
    r"i can check",
    r"checking",
    r"looking into",
    r"one moment",
    r"just a moment",
    r"give me a (moment|second|sec)",
    r"hold on",
    r"let me find",
    r"i'll find",
    r"look.{0,10}up",  # "look that up", "look it up", "look up"
    r"pull.{0,10}(up|that)",  # "pulling that up", "pull that up"
    r"let me see",
    r"i'll see what",
    r"let me.{0,10}(verify|confirm|get)",  # "let me verify", "let me get that"
    r"want me to check",  # "Want me to check your schedule?"
    r"shall i check",
    r"should i check",
    r"i('d| would) need to.{0,20}(check|look|verify|confirm)",  # "I'd need to check that"
    r"i('ll| will) need to",  # "I'll need to look into that"
    r"need to (check|look|find)",
    r"would need to (check|look|verify)",
]

# Patterns indicating hallucinated data (should NOT appear)
HALLUCINATION_PATTERNS = {
    "phone_number": [
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # US phone format
        r"\+\d{1,3}[-.\s]?\d{3,}",  # International format
    ],
    "email": [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    ],
    "time": [
        r"\b(?:1[0-2]|0?[1-9])(?::[0-5][0-9])?\s*(?:am|pm|AM|PM)\b",  # 3pm, 3:00 PM
        r"\b(?:1[0-9]|2[0-3]|0?[0-9]):[0-5][0-9]\b",  # 24h format
    ],
    "address": [
        r"\b\d+\s+[A-Za-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b",
    ],
    "money": [
        r"\$\d+(?:,\d{3})*(?:\.\d{2})?",  # $1,234.56
    ],
}


def has_deferral_language(response: str) -> bool:
    """Check if response contains appropriate deferral language."""
    response_lower = response.lower()
    # Normalize curly/smart quotes to ASCII (LLMs often produce U+2018/U+2019)
    response_lower = response_lower.replace("\u2018", "'").replace("\u2019", "'")
    for pattern in DEFERRAL_PATTERNS:
        if re.search(pattern, response_lower):
            return True
    return False


def has_in_progress_language(response: str) -> bool:
    """Check if response explicitly reports that work is still in progress."""
    response_lower = response.lower()
    progress_patterns = [
        r"still working",
        r"working on (it|that|them)",
        r"in progress",
        r"not done",
        r"on it now",
        r"still doing (it|that|them)",
        r"still (creating|setting|checking|starting|queuing|submitting)",
    ]
    return any(re.search(pattern, response_lower) for pattern in progress_patterns)


def has_hallucinated_data(response: str, data_types: list[str] | None = None) -> dict:
    """
    Check if response contains hallucinated specific data.

    Returns dict of {data_type: [matched_values]} for any matches found.
    """
    if data_types is None:
        data_types = list(HALLUCINATION_PATTERNS.keys())

    matches = {}
    for data_type in data_types:
        if data_type in HALLUCINATION_PATTERNS:
            for pattern in HALLUCINATION_PATTERNS[data_type]:
                found = re.findall(pattern, response, re.IGNORECASE)
                if found:
                    matches.setdefault(data_type, []).extend(found)
    return matches


def has_premature_completion_claim(response: str) -> bool:
    """Check if response claims completion while work is still in progress."""
    normalized = response.lower()
    completion_patterns = [
        r"^done[.!]?$",
        r"\ball set\b",
        r"\b(contact|task).{0,30}\b(created|exists|set)\b",
        r"\bhas been created\b",
        r"\bi created\b",
    ]
    in_progress_patterns = [
        r"\bnot done\b",
        r"\bstill (working|doing|in progress)\b",
        r"\bin progress\b",
        r"\bnot created\b",
    ]

    has_completion = any(
        re.search(pattern, normalized) for pattern in completion_patterns
    )
    has_in_progress = any(
        re.search(pattern, normalized) for pattern in in_progress_patterns
    )
    return has_completion and not has_in_progress


@pytest.fixture
def voice_agent_prompt():
    """Build the voice agent system prompt with test data."""
    return build_voice_agent_prompt(
        bio="A helpful and efficient assistant.",
        boss_first_name="Alex",
        boss_surname="Thompson",
        boss_phone_number="+1-555-0100",
        boss_email_address="alex.thompson@example.com",
        is_boss_user=True,
        contact_rolling_summary=None,
    ).flatten()


async def get_unify_llm_response(
    system_prompt: str,
    conversation: list[dict[str, str]],
    model: str = MODEL_TTS,
) -> str:
    """
    Get response from UnifyLLM (TTS mode fast brain).

    Uses the same UnifyLLM adapter that production call.py uses, with streaming
    to collect the full response (matching real-world behavior).
    """
    llm_instance = UnifyLLM(model=model, reasoning_effort="low")

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content=system_prompt)

    for msg in conversation:
        chat_ctx.add_message(role=msg["role"], content=msg["content"])

    chat_ctx.add_message(
        role="user",
        content="Respond as the assistant. Keep it concise and conversational (voice call).",
    )

    stream = llm_instance.chat(chat_ctx=chat_ctx)
    response_parts = []
    try:
        async for chunk in stream:
            if hasattr(chunk, "delta") and chunk.delta:
                content = getattr(chunk.delta, "content", None)
                if content:
                    response_parts.append(content)
    finally:
        await stream.aclose()

    return "".join(response_parts)


async def get_fast_brain_response(
    system_prompt: str,
    conversation: list[dict[str, str]],
    model: str = MODEL_TTS,
) -> str:
    """Get response from the fast brain model via UnifyLLM."""
    return await get_unify_llm_response(system_prompt, conversation, model)


# =============================================================================
# Model Parameterization
# =============================================================================

FAST_BRAIN_MODELS = [
    pytest.param(MODEL_TTS, id="tts-" + MODEL_TTS.split("@")[0]),
]


@pytest.fixture(params=FAST_BRAIN_MODELS)
def fast_brain_model(request):
    """Parameterized fixture for fast brain model selection."""
    return request.param


# =============================================================================
# Test Class: Contact Information Queries
# =============================================================================


@pytest.mark.asyncio
class TestContactInformationDeferral:
    """
    Tests that the fast brain defers contact info queries instead of hallucinating.

    The fast brain does NOT have access to the contact database. When asked for
    phone numbers, emails, or addresses of contacts, it should defer to the slow
    brain which can look up this information.
    """

    async def test_defers_phone_number_query(
        self,
        voice_agent_prompt,
        fast_brain_model,
    ):
        """Fast brain should defer when asked for someone's phone number."""
        conversation = [
            {"role": "user", "content": "Hey, what's David Johnson's phone number?"},
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        # Should have deferral language
        assert has_deferral_language(response), (
            f"Fast brain ({fast_brain_model}) should defer phone number queries!\n"
            f"Response: {response}\n\n"
            f"Expected deferral language like 'let me check' or 'I'll look that up'.\n"
            f"The fast brain doesn't have access to contact data."
        )

        # Should NOT have a hallucinated phone number
        hallucinations = has_hallucinated_data(response, ["phone_number"])
        assert not hallucinations, (
            f"Fast brain ({fast_brain_model}) hallucinated a phone number!\n"
            f"Response: {response}\n"
            f"Hallucinated data: {hallucinations}\n\n"
            f"The fast brain should NEVER make up contact information."
        )

    async def test_defers_email_query(self, voice_agent_prompt, fast_brain_model):
        """Fast brain should defer when asked for someone's email address."""
        conversation = [
            {"role": "user", "content": "Can you give me Sarah's email address?"},
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        assert has_deferral_language(response), (
            f"Fast brain ({fast_brain_model}) should defer email queries!\n"
            f"Response: {response}"
        )

        hallucinations = has_hallucinated_data(response, ["email"])
        assert not hallucinations, (
            f"Fast brain ({fast_brain_model}) hallucinated an email address!\n"
            f"Response: {response}\n"
            f"Hallucinated data: {hallucinations}"
        )

    async def test_defers_address_query(self, voice_agent_prompt, fast_brain_model):
        """Fast brain should defer when asked for someone's address."""
        conversation = [
            {"role": "user", "content": "What's the address for Bob's office?"},
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        assert has_deferral_language(response), (
            f"Fast brain ({fast_brain_model}) should defer address queries!\n"
            f"Response: {response}"
        )

        hallucinations = has_hallucinated_data(response, ["address"])
        assert not hallucinations, (
            f"Fast brain ({fast_brain_model}) hallucinated an address!\n"
            f"Response: {response}\n"
            f"Hallucinated data: {hallucinations}"
        )


# =============================================================================
# Test Class: Calendar/Schedule Queries
# =============================================================================


@pytest.mark.asyncio
class TestCalendarScheduleDeferral:
    """
    Tests that the fast brain defers calendar queries instead of hallucinating.

    The fast brain does NOT have access to calendar data. When asked about meeting
    times, appointments, or schedule, it should defer to the slow brain.
    """

    async def test_defers_meeting_time_query(
        self,
        voice_agent_prompt,
        fast_brain_model,
    ):
        """Fast brain should defer when asked about meeting times."""
        conversation = [
            {"role": "user", "content": "What time is my meeting with Alice tomorrow?"},
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        assert has_deferral_language(response), (
            f"Fast brain ({fast_brain_model}) should defer meeting time queries!\n"
            f"Response: {response}"
        )

        hallucinations = has_hallucinated_data(response, ["time"])
        assert not hallucinations, (
            f"Fast brain ({fast_brain_model}) hallucinated a meeting time!\n"
            f"Response: {response}\n"
            f"Hallucinated data: {hallucinations}\n\n"
            f"The fast brain should NEVER guess at meeting times."
        )

    async def test_defers_schedule_overview_query(
        self,
        voice_agent_prompt,
        fast_brain_model,
    ):
        """Fast brain should defer when asked for schedule overview."""
        conversation = [
            {"role": "user", "content": "What's on my calendar for today?"},
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        assert has_deferral_language(response), (
            f"Fast brain ({fast_brain_model}) should defer schedule queries!\n"
            f"Response: {response}"
        )

    async def test_defers_availability_query(
        self,
        voice_agent_prompt,
        fast_brain_model,
    ):
        """Fast brain should defer when asked about availability."""
        conversation = [
            {"role": "user", "content": "Am I free at 3pm on Friday?"},
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        assert has_deferral_language(response), (
            f"Fast brain ({fast_brain_model}) should defer availability queries!\n"
            f"Response: {response}"
        )

        # Should not claim to know availability
        response_lower = response.lower()
        assert not any(
            phrase in response_lower
            for phrase in ["you're free", "you are free", "yes, you're", "yes you are"]
        ), (
            f"Fast brain ({fast_brain_model}) claimed to know availability without checking!\n"
            f"Response: {response}"
        )


# =============================================================================
# Test Class: Specific Facts/Data Queries
# =============================================================================


@pytest.mark.asyncio
class TestSpecificFactsDeferral:
    """
    Tests that the fast brain defers queries about specific facts/data.

    Questions about budgets, deadlines, project details, etc. require data lookup.
    """

    async def test_defers_budget_query(self, voice_agent_prompt, fast_brain_model):
        """Fast brain should defer when asked about budget amounts."""
        conversation = [
            {"role": "user", "content": "What's the budget for the Henderson project?"},
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        assert has_deferral_language(response), (
            f"Fast brain ({fast_brain_model}) should defer budget queries!\n"
            f"Response: {response}"
        )

        hallucinations = has_hallucinated_data(response, ["money"])
        assert not hallucinations, (
            f"Fast brain ({fast_brain_model}) hallucinated a budget amount!\n"
            f"Response: {response}\n"
            f"Hallucinated data: {hallucinations}"
        )

    async def test_defers_deadline_query(self, voice_agent_prompt, fast_brain_model):
        """Fast brain should defer when asked about deadlines."""
        conversation = [
            {"role": "user", "content": "When is the proposal deadline?"},
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        assert has_deferral_language(response), (
            f"Fast brain ({fast_brain_model}) should defer deadline queries!\n"
            f"Response: {response}"
        )

    async def test_defers_email_content_query(
        self,
        voice_agent_prompt,
        fast_brain_model,
    ):
        """Fast brain should defer when asked about email contents."""
        conversation = [
            {"role": "user", "content": "What did the client say in their last email?"},
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        assert has_deferral_language(response), (
            f"Fast brain ({fast_brain_model}) should defer email content queries!\n"
            f"Response: {response}"
        )

        # Should not claim to know email content (making up quotes or specific claims)
        # Note: "what they said" in a question is fine, we're looking for definitive claims
        response_lower = response.lower()
        # Look for patterns that indicate claimed knowledge of email content
        claimed_knowledge_patterns = [
            "they said that",  # Claiming to know what was said
            "the email says",
            "the email said",
            "they mentioned",  # Claiming to know what was mentioned
            "they asked",  # Claiming to know what was asked
            "they wanted",  # Claiming to know what they wanted
            "they wrote",  # Claiming to know what they wrote (not "what they wrote" as question)
        ]
        claimed_knowledge = any(
            pattern in response_lower for pattern in claimed_knowledge_patterns
        )
        assert not claimed_knowledge or has_deferral_language(response), (
            f"Fast brain ({fast_brain_model}) may have hallucinated email content!\n"
            f"Response: {response}"
        )


# =============================================================================
# Test Class: Real-time Data Queries
# =============================================================================


@pytest.mark.asyncio
class TestRealTimeDataDeferral:
    """
    Tests that the fast brain defers queries requiring real-time data lookup.
    """

    async def test_defers_weather_query(self, voice_agent_prompt, fast_brain_model):
        """Fast brain should defer when asked about weather."""
        conversation = [
            {"role": "user", "content": "What's the weather like in Chicago today?"},
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        assert has_deferral_language(response), (
            f"Fast brain ({fast_brain_model}) should defer weather queries!\n"
            f"Response: {response}"
        )

        # Should not make up weather data
        weather_terms = ["sunny", "cloudy", "rainy", "degrees", "fahrenheit", "celsius"]
        response_lower = response.lower()

        # Allow deferral phrases that mention weather conceptually
        if not has_deferral_language(response):
            assert not any(term in response_lower for term in weather_terms), (
                f"Fast brain ({fast_brain_model}) hallucinated weather data!\n"
                f"Response: {response}"
            )

    async def test_defers_unread_messages_query(
        self,
        voice_agent_prompt,
        fast_brain_model,
    ):
        """Fast brain should defer when asked about unread messages."""
        conversation = [
            {"role": "user", "content": "Do I have any new emails?"},
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        assert has_deferral_language(response), (
            f"Fast brain ({fast_brain_model}) should defer unread messages queries!\n"
            f"Response: {response}"
        )

        # Should not claim "no new emails" or "you have X emails" without checking
        response_lower = response.lower()
        assert (
            not any(
                phrase in response_lower
                for phrase in ["no new", "you have", "you've got", "there are"]
            )
            or "let me" in response_lower
        ), (
            f"Fast brain ({fast_brain_model}) claimed to know email status without checking!\n"
            f"Response: {response}"
        )


# =============================================================================
# Test Class: False Negative Detection (Should NOT Defer)
# =============================================================================


def response_contains_data(response: str, expected_data: str) -> bool:
    """Check if response contains the expected data (case-insensitive)."""
    return expected_data.lower() in response.lower()


def response_is_incorrect_deferral(response: str) -> bool:
    """
    Check if response incorrectly defers when it should answer directly.

    A false negative is when the fast brain says "let me check" for information
    that was ALREADY provided in the conversation.
    """
    return has_deferral_language(response)


@pytest.mark.asyncio
class TestFalseNegativeDetection:
    """
    Tests that the fast brain does NOT defer when it should answer directly.

    These are "false negative" scenarios where:
    1. The answer was already provided in the conversation
    2. The user is asking for clarification/repetition
    3. Deferring would be incorrect and frustrating

    The fast brain should recognize these cases and respond directly.
    """

    async def test_repeats_phone_number_when_asked(
        self,
        voice_agent_prompt,
        fast_brain_model,
    ):
        """Fast brain should repeat a phone number that was just provided."""
        conversation = [
            {"role": "user", "content": "What's David's phone number?"},
            {
                "role": "assistant",
                "content": "[notification] David's phone number is 555-123-4567",
            },
            {
                "role": "assistant",
                "content": "David's number is 555-123-4567.",
            },
            {"role": "user", "content": "Sorry, what was that number again?"},
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        # Should NOT defer - the number was just provided
        assert not response_is_incorrect_deferral(response), (
            f"Fast brain ({fast_brain_model}) incorrectly deferred!\n"
            f"Response: {response}\n\n"
            f"The phone number was just provided. User asked to repeat it.\n"
            f"Fast brain should say the number again, not 'let me check'."
        )

        # Should contain the phone number
        assert response_contains_data(response, "555") or response_contains_data(
            response,
            "123",
        ), (
            f"Fast brain ({fast_brain_model}) didn't repeat the number!\n"
            f"Response: {response}\n\n"
            f"Expected the response to contain the phone number 555-123-4567."
        )

    async def test_repeats_email_when_asked(
        self,
        voice_agent_prompt,
        fast_brain_model,
    ):
        """Fast brain should repeat an email that was just provided."""
        conversation = [
            {"role": "user", "content": "What's Sarah's email?"},
            {
                "role": "assistant",
                "content": "[notification] Sarah's email is sarah.jones@company.com",
            },
            {
                "role": "assistant",
                "content": "Sarah's email is sarah.jones@company.com.",
            },
            {"role": "user", "content": "Could you spell that out for me?"},
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        # Should NOT defer
        assert not response_is_incorrect_deferral(response), (
            f"Fast brain ({fast_brain_model}) incorrectly deferred!\n"
            f"Response: {response}\n\n"
            f"The email was just provided. User asked to spell it out.\n"
            f"Fast brain should repeat/spell it, not 'let me check'."
        )

        # Should reference the email - could be:
        # - Contiguous: "sarah.jones@company.com"
        # - Spaced out: "s a r a h dot j o n e s"
        # - NATO phonetic: "S as in Sam, A as in Apple..."
        # - Contains "company" or "@" reference
        response_lower = response.lower()
        response_normalized = response_lower.replace(" ", "").replace("-", "")

        # Check for direct name match (handles spaced out spelling)
        has_sarah = "sarah" in response_normalized
        has_jones = "jones" in response_normalized
        # Check for company.com reference
        has_company = "company" in response_lower
        # Check for NATO phonetic spelling pattern (S as in...)
        has_nato_spelling = "as in" in response_lower and (
            "s as in" in response_lower or "j as in" in response_lower
        )

        assert has_sarah or has_jones or has_company or has_nato_spelling, (
            f"Fast brain ({fast_brain_model}) didn't repeat the email!\n"
            f"Response: {response}"
        )

    async def test_repeats_meeting_time_when_asked(
        self,
        voice_agent_prompt,
        fast_brain_model,
    ):
        """Fast brain should repeat a meeting time that was just provided."""
        conversation = [
            {"role": "user", "content": "When's my meeting with Alice?"},
            {
                "role": "assistant",
                "content": "[notification] Meeting with Alice is at 3:30 PM tomorrow",
            },
            {
                "role": "assistant",
                "content": "Your meeting with Alice is at 3:30 PM tomorrow.",
            },
            {"role": "user", "content": "Sorry, what time was that?"},
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        # Should NOT defer
        assert not response_is_incorrect_deferral(response), (
            f"Fast brain ({fast_brain_model}) incorrectly deferred!\n"
            f"Response: {response}\n\n"
            f"The meeting time was just provided. User asked to repeat it.\n"
            f"Fast brain should say '3:30 PM', not 'let me check'."
        )

        # Should contain the time
        assert response_contains_data(response, "3:30") or response_contains_data(
            response,
            "330",
        ), (
            f"Fast brain ({fast_brain_model}) didn't repeat the time!\n"
            f"Response: {response}"
        )

    async def test_repeats_address_when_asked(
        self,
        voice_agent_prompt,
        fast_brain_model,
    ):
        """Fast brain should repeat an address that was just provided."""
        conversation = [
            {"role": "user", "content": "What's the address for the client meeting?"},
            {
                "role": "assistant",
                "content": "[notification] The meeting is at 742 Evergreen Terrace, Suite 400",
            },
            {
                "role": "assistant",
                "content": "The meeting is at 742 Evergreen Terrace, Suite 400.",
            },
            {"role": "user", "content": "Wait, what was the street name?"},
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        # Should NOT defer
        assert not response_is_incorrect_deferral(response), (
            f"Fast brain ({fast_brain_model}) incorrectly deferred!\n"
            f"Response: {response}\n\n"
            f"The address was just provided. User asked for the street name.\n"
            f"Fast brain should say 'Evergreen Terrace', not 'let me check'."
        )

        # Should contain the street name
        assert response_contains_data(response, "evergreen"), (
            f"Fast brain ({fast_brain_model}) didn't repeat the street name!\n"
            f"Response: {response}"
        )

    async def test_confirms_info_user_just_provided(
        self,
        voice_agent_prompt,
        fast_brain_model,
    ):
        """Fast brain should confirm info the user just stated, not defer."""
        conversation = [
            {
                "role": "user",
                "content": "I need to schedule a meeting for 2pm on Tuesday.",
            },
            {"role": "assistant", "content": "Got it, I'll schedule that for you."},
            {"role": "user", "content": "So that's Tuesday at 2, right?"},
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        # Should NOT defer - user is confirming what THEY said
        assert not response_is_incorrect_deferral(response), (
            f"Fast brain ({fast_brain_model}) incorrectly deferred!\n"
            f"Response: {response}\n\n"
            f"User is confirming info they just provided (Tuesday at 2pm).\n"
            f"Fast brain should confirm, not 'let me check'."
        )

    async def test_repeats_name_when_asked(
        self,
        voice_agent_prompt,
        fast_brain_model,
    ):
        """Fast brain should repeat a name that was just mentioned."""
        conversation = [
            {"role": "user", "content": "Who should I contact about the invoice?"},
            {
                "role": "assistant",
                "content": "[notification] Contact Jennifer Martinez in accounting",
            },
            {
                "role": "assistant",
                "content": "You should contact Jennifer Martinez in accounting.",
            },
            {"role": "user", "content": "Sorry, what was her name again?"},
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        # Should NOT defer
        assert not response_is_incorrect_deferral(response), (
            f"Fast brain ({fast_brain_model}) incorrectly deferred!\n"
            f"Response: {response}\n\n"
            f"The name was just provided. User asked to repeat it.\n"
            f"Fast brain should say 'Jennifer Martinez', not 'let me check'."
        )

        # Should contain the name
        assert response_contains_data(response, "jennifer") or response_contains_data(
            response,
            "martinez",
        ), (
            f"Fast brain ({fast_brain_model}) didn't repeat the name!\n"
            f"Response: {response}"
        )


# =============================================================================
# Test Class: In-progress Action Status (Premature Completion Claims)
# =============================================================================


@pytest.mark.asyncio
class TestInProgressActionStatus:
    """
    Tests that the fast brain does not claim completion without completion guidance.

    When the slow brain sends an in-progress notification like "I'm creating...",
    the fast brain should keep deferring status checks until an explicit completion
    notification arrives.
    """

    async def test_in_progress_notification_does_not_allow_done_claim(
        self,
        voice_agent_prompt,
    ):
        """Fast brain should report in-progress state, not 'Done.'."""
        conversation = [
            {
                "role": "user",
                "content": "Create a Bob contact and an Apply to OpenAI task for him.",
            },
            {"role": "assistant", "content": "Let me check on that."},
            {
                "role": "system",
                "content": (
                    "[notification] Got it - I'm creating a Bob contact now, and "
                    'I\'ll set up an "Apply to OpenAI" task for the B2B Applications '
                    "frontend engineer role."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    'Creating Bob contact and setting task "Apply to OpenAI" for '
                    "the B2B Applications frontend engineer role with the ~$174K salary."
                ),
            },
            {
                "role": "user",
                "content": "Are you done with it, or are you still doing it?",
            },
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=MODEL_TTS,
        )

        assert has_deferral_language(response) or has_in_progress_language(response), (
            "Fast brain should keep the action in-progress until completion guidance "
            f"arrives.\nResponse: {response}"
        )
        assert not has_premature_completion_claim(response), (
            "Fast brain claimed completion without explicit completion guidance.\n"
            f"Response: {response}"
        )

    async def test_update_request_does_not_claim_created_without_completion_guidance(
        self,
        voice_agent_prompt,
    ):
        """Fast brain should give progress language on update requests."""
        conversation = [
            {
                "role": "user",
                "content": "Create a Bob contact and an Apply to OpenAI task for him.",
            },
            {"role": "assistant", "content": "Let me check on that."},
            {
                "role": "system",
                "content": (
                    "[notification] Got it - I'm creating a Bob contact now, and "
                    'I\'ll set up an "Apply to OpenAI" task for the B2B Applications '
                    "frontend engineer role."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    'Creating Bob contact and setting task "Apply to OpenAI" for '
                    "the B2B Applications frontend engineer role with the ~$174K salary."
                ),
            },
            {"role": "user", "content": "Any updates?"},
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=MODEL_TTS,
        )

        assert has_deferral_language(response) or has_in_progress_language(response), (
            "Fast brain should respond with in-progress language when no completion "
            f"notification exists.\nResponse: {response}"
        )
        assert not has_premature_completion_claim(response), (
            "Fast brain claimed contact/task creation without completion guidance.\n"
            f"Response: {response}"
        )
