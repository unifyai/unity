"""
tests/conversation_manager/voice/test_fast_brain_deferral.py
=============================================================

Tests for Fast Brain (Voice Agent) deferral behavior.

The fast brain handles real-time voice conversation. It should defer to the
slow brain for data lookups instead of hallucinating answers.

This file tests "false positive" prevention - scenarios where the fast brain
might be tempted to hallucinate but should defer with natural language.

Tests are parameterized to run against both:
- gpt-5-nano@openai (TTS mode fast brain, via UnifyLLM)
- gpt-realtime (STS mode fast brain, via OpenAI Realtime API with text modality)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import pytest
import websockets

import unillm
from livekit.agents import llm

from unity.conversation_manager.livekit_unify_adapter import UnifyLLM
from unity.conversation_manager.prompt_builders import build_voice_agent_prompt

# Model identifiers
MODEL_GPT5_NANO = "gpt-5-nano@openai"
MODEL_GPT_REALTIME = "gpt-realtime"

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
    ).flatten()


async def get_unify_llm_response(
    system_prompt: str,
    conversation: list[dict[str, str]],
    model: str = "gpt-5-nano@openai",
) -> str:
    """
    Get response from UnifyLLM (for gpt-5-nano TTS mode).

    Uses the same UnifyLLM adapter that production call.py uses, with streaming
    to collect the full response (matching real-world behavior).
    """
    llm_instance = UnifyLLM(model=model, reasoning_effort="minimal")

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


async def get_realtime_response(
    system_prompt: str,
    conversation: list[dict[str, str]],
    model: str = "gpt-realtime",
    timeout: float = 30.0,
) -> str:
    """
    Get response from OpenAI Realtime API with text modality.

    Uses WebSocket connection to OpenAI's Realtime API, matching the
    production sts_call.py architecture but with text-only mode for testing.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set - skipping realtime model test")

    url = f"wss://api.openai.com/v1/realtime?model={model}"
    headers = {"Authorization": f"Bearer {api_key}"}

    response_text = ""

    async with websockets.connect(
        url,
        additional_headers=headers,
        close_timeout=5,
    ) as ws:
        # Configure session with instructions (no modalities in session.update)
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "instructions": system_prompt,
                    },
                },
            ),
        )

        # Wait for session.updated confirmation
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
            event = json.loads(msg)
            if event.get("type") == "session.updated":
                break
            if event.get("type") == "error":
                raise RuntimeError(f"Session update error: {event}")

        # Add conversation items
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]

            # User messages use input_text, assistant messages use output_text
            content_type = "input_text" if role == "user" else "output_text"

            await ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": role,
                            "content": [{"type": content_type, "text": content}],
                        },
                    },
                ),
            )

            # Wait for item created confirmation
            while True:
                item_msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
                item_event = json.loads(item_msg)
                if item_event.get("type") in (
                    "conversation.item.created",
                    "conversation.item.added",
                ):
                    break
                if item_event.get("type") == "error":
                    raise RuntimeError(f"Conversation item error: {item_event}")

        # Request response generation with text-only output modality
        await ws.send(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "output_modalities": ["text"],
                    },
                },
            ),
        )

        # Collect response text
        response_parts = []
        usage = None
        while True:
            resp_msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
            resp_event = json.loads(resp_msg)
            event_type = resp_event.get("type", "")

            # Collect text deltas (GA API uses response.output_text.delta)
            if event_type in ("response.text.delta", "response.output_text.delta"):
                delta = resp_event.get("delta", "")
                response_parts.append(delta)
            # Response complete — extract usage from the event payload
            elif event_type == "response.done":
                resp_obj = resp_event.get("response", {})
                usage = resp_obj.get("usage")
                break
            elif event_type == "error":
                raise RuntimeError(f"Response error: {resp_event}")

        response_text = "".join(response_parts)

    # Log via unillm so LLM I/O appears in UNILLM_LOG_DIR
    transcript = [{"role": "system", "content": system_prompt}]
    transcript.extend(conversation)
    if response_text:
        transcript.append({"role": "assistant", "content": response_text})
    unillm.log_usage(
        model,
        usage or {},
        transcript=transcript,
        label=model,
    )

    return response_text


async def get_fast_brain_response(
    system_prompt: str,
    conversation: list[dict[str, str]],
    model: str = MODEL_GPT5_NANO,
) -> str:
    """
    Get response from the fast brain model.

    Dispatches to the appropriate backend based on model:
    - gpt-5-nano@openai: Uses UnifyLLM (TTS mode)
    - gpt-realtime: Uses OpenAI Realtime WebSocket API (STS mode)
    """
    if model == MODEL_GPT_REALTIME:
        return await get_realtime_response(system_prompt, conversation, model)
    else:
        return await get_unify_llm_response(system_prompt, conversation, model)


# =============================================================================
# Model Parameterization
# =============================================================================

# List of models to test
FAST_BRAIN_MODELS = [
    pytest.param(MODEL_GPT5_NANO, id="tts-gpt5nano"),
    pytest.param(MODEL_GPT_REALTIME, id="sts-realtime"),
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
            {"role": "user", "content": "What's the weather like today?"},
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
