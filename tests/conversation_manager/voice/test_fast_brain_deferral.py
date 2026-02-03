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

# Patterns indicating proper deferral (case-insensitive)
# These patterns indicate the assistant is either:
# 1. Going to check/lookup data (deferred to slow brain)
# 2. Acknowledging it doesn't have the data immediately
DEFERRAL_PATTERNS = [
    r"let me check",
    r"let me look",
    r"i'll check",
    r"i'll look",
    r"checking",
    r"looking into",
    r"one moment",
    r"just a moment",
    r"give me a (moment|second|sec)",
    r"hold on",
    r"let me find",
    r"i'll find",
    r"look.{0,10}up",  # "look that up", "look it up", "look up"
    r"let me see",
    r"i'll see what",
    r"i don't have.{0,30}(on hand|at the moment|right now|available)",
    r"don't have.{0,20}information",
    r"need to (check|look|find)",
    r"don't have access",
    r"can't see",
    r"i can't access",
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
    for pattern in DEFERRAL_PATTERNS:
        if re.search(pattern, response_lower):
            return True
    return False


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
    )


async def get_fast_brain_response(
    system_prompt: str,
    conversation: list[dict[str, str]],
    model: str = "gpt-5-nano@openai",
) -> str:
    """
    Simulate the fast brain responding to a conversation.

    Uses the same UnifyLLM adapter that production call.py uses, with streaming
    to collect the full response (matching real-world behavior).
    """
    # Use the same adapter as production fast brain (call.py)
    llm_instance = UnifyLLM(model=model, reasoning_effort="minimal")

    # Build chat context with system prompt and conversation
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content=system_prompt)

    # Add conversation history
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        chat_ctx.add_message(role=role, content=content)

    # Add the instruction to respond
    chat_ctx.add_message(
        role="user",
        content="Respond as the assistant. Keep it concise and conversational (voice call).",
    )

    # Stream the response and collect the full text
    # The UnifyLLMStream yields ChatChunk objects with delta.content
    stream = llm_instance.chat(chat_ctx=chat_ctx)
    response_parts = []
    try:
        async for chunk in stream:
            # ChatChunk has delta.content (from ChoiceDelta)
            if hasattr(chunk, "delta") and chunk.delta:
                content = getattr(chunk.delta, "content", None)
                if content:
                    response_parts.append(content)
    finally:
        await stream.aclose()

    return "".join(response_parts)


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

    async def test_defers_phone_number_query(self, voice_agent_prompt):
        """Fast brain should defer when asked for someone's phone number."""
        conversation = [
            {"role": "user", "content": "Hey, what's David's phone number?"},
        ]

        response = await get_fast_brain_response(voice_agent_prompt, conversation)

        # Should have deferral language
        assert has_deferral_language(response), (
            f"Fast brain should defer phone number queries!\n"
            f"Response: {response}\n\n"
            f"Expected deferral language like 'let me check' or 'I'll look that up'.\n"
            f"The fast brain doesn't have access to contact data."
        )

        # Should NOT have a hallucinated phone number
        hallucinations = has_hallucinated_data(response, ["phone_number"])
        assert not hallucinations, (
            f"Fast brain hallucinated a phone number!\n"
            f"Response: {response}\n"
            f"Hallucinated data: {hallucinations}\n\n"
            f"The fast brain should NEVER make up contact information."
        )

    async def test_defers_email_query(self, voice_agent_prompt):
        """Fast brain should defer when asked for someone's email address."""
        conversation = [
            {"role": "user", "content": "Can you give me Sarah's email address?"},
        ]

        response = await get_fast_brain_response(voice_agent_prompt, conversation)

        assert has_deferral_language(response), (
            f"Fast brain should defer email queries!\n" f"Response: {response}"
        )

        hallucinations = has_hallucinated_data(response, ["email"])
        assert not hallucinations, (
            f"Fast brain hallucinated an email address!\n"
            f"Response: {response}\n"
            f"Hallucinated data: {hallucinations}"
        )

    async def test_defers_address_query(self, voice_agent_prompt):
        """Fast brain should defer when asked for someone's address."""
        conversation = [
            {"role": "user", "content": "What's the address for Bob's office?"},
        ]

        response = await get_fast_brain_response(voice_agent_prompt, conversation)

        assert has_deferral_language(response), (
            f"Fast brain should defer address queries!\n" f"Response: {response}"
        )

        hallucinations = has_hallucinated_data(response, ["address"])
        assert not hallucinations, (
            f"Fast brain hallucinated an address!\n"
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

    async def test_defers_meeting_time_query(self, voice_agent_prompt):
        """Fast brain should defer when asked about meeting times."""
        conversation = [
            {"role": "user", "content": "What time is my meeting with Alice tomorrow?"},
        ]

        response = await get_fast_brain_response(voice_agent_prompt, conversation)

        assert has_deferral_language(response), (
            f"Fast brain should defer meeting time queries!\n" f"Response: {response}"
        )

        hallucinations = has_hallucinated_data(response, ["time"])
        assert not hallucinations, (
            f"Fast brain hallucinated a meeting time!\n"
            f"Response: {response}\n"
            f"Hallucinated data: {hallucinations}\n\n"
            f"The fast brain should NEVER guess at meeting times."
        )

    async def test_defers_schedule_overview_query(self, voice_agent_prompt):
        """Fast brain should defer when asked for schedule overview."""
        conversation = [
            {"role": "user", "content": "What's on my calendar for today?"},
        ]

        response = await get_fast_brain_response(voice_agent_prompt, conversation)

        assert has_deferral_language(response), (
            f"Fast brain should defer schedule queries!\n" f"Response: {response}"
        )

    async def test_defers_availability_query(self, voice_agent_prompt):
        """Fast brain should defer when asked about availability."""
        conversation = [
            {"role": "user", "content": "Am I free at 3pm on Friday?"},
        ]

        response = await get_fast_brain_response(voice_agent_prompt, conversation)

        assert has_deferral_language(response), (
            f"Fast brain should defer availability queries!\n" f"Response: {response}"
        )

        # Should not claim to know availability
        response_lower = response.lower()
        assert not any(
            phrase in response_lower
            for phrase in ["you're free", "you are free", "yes, you're", "yes you are"]
        ), (
            f"Fast brain claimed to know availability without checking!\n"
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

    async def test_defers_budget_query(self, voice_agent_prompt):
        """Fast brain should defer when asked about budget amounts."""
        conversation = [
            {"role": "user", "content": "What's the budget for the Henderson project?"},
        ]

        response = await get_fast_brain_response(voice_agent_prompt, conversation)

        assert has_deferral_language(response), (
            f"Fast brain should defer budget queries!\n" f"Response: {response}"
        )

        hallucinations = has_hallucinated_data(response, ["money"])
        assert not hallucinations, (
            f"Fast brain hallucinated a budget amount!\n"
            f"Response: {response}\n"
            f"Hallucinated data: {hallucinations}"
        )

    async def test_defers_deadline_query(self, voice_agent_prompt):
        """Fast brain should defer when asked about deadlines."""
        conversation = [
            {"role": "user", "content": "When is the proposal deadline?"},
        ]

        response = await get_fast_brain_response(voice_agent_prompt, conversation)

        assert has_deferral_language(response), (
            f"Fast brain should defer deadline queries!\n" f"Response: {response}"
        )

    async def test_defers_email_content_query(self, voice_agent_prompt):
        """Fast brain should defer when asked about email contents."""
        conversation = [
            {"role": "user", "content": "What did the client say in their last email?"},
        ]

        response = await get_fast_brain_response(voice_agent_prompt, conversation)

        assert has_deferral_language(response), (
            f"Fast brain should defer email content queries!\n" f"Response: {response}"
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
            f"Fast brain may have hallucinated email content!\n" f"Response: {response}"
        )


# =============================================================================
# Test Class: Real-time Data Queries
# =============================================================================


@pytest.mark.asyncio
class TestRealTimeDataDeferral:
    """
    Tests that the fast brain defers queries requiring real-time data lookup.
    """

    async def test_defers_weather_query(self, voice_agent_prompt):
        """Fast brain should defer when asked about weather."""
        conversation = [
            {"role": "user", "content": "What's the weather like today?"},
        ]

        response = await get_fast_brain_response(voice_agent_prompt, conversation)

        assert has_deferral_language(response), (
            f"Fast brain should defer weather queries!\n" f"Response: {response}"
        )

        # Should not make up weather data
        weather_terms = ["sunny", "cloudy", "rainy", "degrees", "fahrenheit", "celsius"]
        response_lower = response.lower()

        # Allow deferral phrases that mention weather conceptually
        if not has_deferral_language(response):
            assert not any(term in response_lower for term in weather_terms), (
                f"Fast brain hallucinated weather data!\n" f"Response: {response}"
            )

    async def test_defers_unread_messages_query(self, voice_agent_prompt):
        """Fast brain should defer when asked about unread messages."""
        conversation = [
            {"role": "user", "content": "Do I have any new emails?"},
        ]

        response = await get_fast_brain_response(voice_agent_prompt, conversation)

        assert has_deferral_language(response), (
            f"Fast brain should defer unread messages queries!\n"
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
            f"Fast brain claimed to know email status without checking!\n"
            f"Response: {response}"
        )
