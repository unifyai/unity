"""
tests/conversation_manager/test_voice_agent_prompt.py
=====================================================

Tests for the Voice Agent (fast brain) prompt builder, verifying that the LLM
can answer questions directly using context provided in the system prompt,
rather than deferring to the slow brain.

These tests validate the three context enrichments:
1. **Assistant name**: The fast brain knows its own name and can introduce itself.
2. **Contact bio**: The fast brain knows the bio/background of the person on the call.
3. **Meet participants**: The fast brain knows all participant details in multi-party calls.

Each test builds a voice agent prompt with the relevant context, sends a user
question to the LLM, and asserts the response answers directly (no deferral
phrases like "let me check").
"""

from __future__ import annotations

import pytest

from unity.common.llm_client import new_llm_client
from unity.conversation_manager.prompt_builders import build_voice_agent_prompt

# =============================================================================
# Constants
# =============================================================================

# Deferral phrases the fast brain uses when it doesn't have data.
# If any of these appear in the response, the fast brain is deferring
# instead of answering directly.
DEFERRAL_PHRASES = [
    "let me check",
    "let me look",
    "i'm looking into",
    "i'll check",
    "i'll look into",
    "i need to check",
    "one moment",
    "hold on",
    "let me find",
    "checking on that",
    "looking into that",
]

# The model used by the fast brain in production
FAST_BRAIN_MODEL = "gpt-5-nano@openai"


# =============================================================================
# Helpers
# =============================================================================


async def ask_fast_brain(system_prompt: str, user_message: str) -> str:
    """Send a user message to the fast brain LLM and return the response.

    Args:
        system_prompt: The voice agent system prompt.
        user_message: The user's spoken message.

    Returns:
        The assistant's response text.
    """
    client = new_llm_client(
        model=FAST_BRAIN_MODEL,
        reasoning_effort="minimal",
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    response = await client.generate(messages=messages)
    return response.strip()


def assert_no_deferral(response: str, context: str = "") -> None:
    """Assert that the response does not contain deferral phrases.

    Args:
        response: The assistant's response text.
        context: Optional description of what we're testing.
    """
    response_lower = response.lower()
    for phrase in DEFERRAL_PHRASES:
        assert phrase not in response_lower, (
            f"Fast brain deferred instead of answering directly!\n"
            f"Deferral phrase found: '{phrase}'\n"
            f"Full response: {response}\n"
            f"{f'Context: {context}' if context else ''}"
        )


def assert_contains(response: str, expected: str, context: str = "") -> None:
    """Assert that the response contains the expected substring (case-insensitive).

    Args:
        response: The assistant's response text.
        expected: The substring that should appear.
        context: Optional description of what we're testing.
    """
    assert expected.lower() in response.lower(), (
        f"Expected '{expected}' in response but not found!\n"
        f"Full response: {response}\n"
        f"{f'Context: {context}' if context else ''}"
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_prompt_kwargs() -> dict:
    """Base keyword arguments for build_voice_agent_prompt."""
    return {
        "bio": "I am a virtual assistant working for a tech startup.",
        "assistant_name": "Alex",
        "boss_first_name": "Sarah",
        "boss_surname": "Chen",
        "boss_phone_number": "+15551234567",
        "boss_email_address": "sarah@techstartup.com",
        "boss_bio": "CEO of TechStartup Inc., focused on AI products.",
    }


@pytest.fixture
def boss_call_prompt(base_prompt_kwargs: dict) -> str:
    """Voice agent prompt for a call with the boss."""
    return build_voice_agent_prompt(
        **base_prompt_kwargs,
        is_boss_user=True,
    ).flatten()


@pytest.fixture
def contact_call_prompt(base_prompt_kwargs: dict) -> str:
    """Voice agent prompt for a call with an external contact."""
    return build_voice_agent_prompt(
        **base_prompt_kwargs,
        is_boss_user=False,
        contact_first_name="Marcus",
        contact_surname="Rivera",
        contact_phone_number="+15559876543",
        contact_email="marcus@clientcorp.com",
        contact_bio="VP of Engineering at ClientCorp. Leading their cloud migration project. Prefers concise communication.",
    ).flatten()


@pytest.fixture
def meet_prompt(base_prompt_kwargs: dict) -> str:
    """Voice agent prompt for a multi-party Unify Meet."""
    return build_voice_agent_prompt(
        **base_prompt_kwargs,
        is_boss_user=True,
        participants=[
            {
                "first_name": "Sarah",
                "surname": "Chen",
                "bio": "CEO of TechStartup Inc., focused on AI products.",
            },
            {
                "first_name": "Marcus",
                "surname": "Rivera",
                "bio": "VP of Engineering at ClientCorp. Leading their cloud migration project.",
            },
            {
                "first_name": "Priya",
                "surname": "Sharma",
                "bio": "Product Manager at TechStartup. Coordinates between engineering and clients.",
            },
        ],
    ).flatten()


# =============================================================================
# Test Class: Assistant Name
# =============================================================================


@pytest.mark.asyncio
class TestAssistantName:
    """Tests that the fast brain knows and uses its own name."""

    async def test_assistant_states_own_name(self, boss_call_prompt: str):
        """
        When asked "what's your name?", the fast brain should answer directly
        with its name instead of deferring.
        """
        response = await ask_fast_brain(boss_call_prompt, "What's your name?")

        assert_no_deferral(response, "Asked for assistant's own name")
        assert_contains(response, "Alex", "Assistant should state its name")

    async def test_assistant_introduces_itself(self, boss_call_prompt: str):
        """
        When greeting and being asked to introduce themselves, the fast brain
        should include its name naturally.
        """
        response = await ask_fast_brain(
            boss_call_prompt,
            "Hi there! Who am I speaking with?",
        )

        assert_no_deferral(response, "Asked who they're speaking with")
        assert_contains(response, "Alex", "Assistant should mention its name")

    async def test_prompt_without_name_still_works(self, base_prompt_kwargs: dict):
        """
        When assistant_name is not provided, the prompt should still work
        without errors (graceful degradation).
        """
        kwargs = {**base_prompt_kwargs, "assistant_name": None}
        prompt = build_voice_agent_prompt(**kwargs, is_boss_user=True).flatten()

        # Should not contain "My name is" when name is not set
        assert "My name is" not in prompt


# =============================================================================
# Test Class: Contact Bio
# =============================================================================


@pytest.mark.asyncio
class TestContactBio:
    """Tests that the fast brain can use contact bio information."""

    async def test_knows_contact_role(self, contact_call_prompt: str):
        """
        When asked about the contact's role, the fast brain should answer
        directly using the bio context.
        """
        response = await ask_fast_brain(
            contact_call_prompt,
            "Remind me, what does Marcus do at his company?",
        )

        assert_no_deferral(response, "Asked about contact's role from bio")
        assert_contains(
            response,
            "engineer",
            "Should mention engineering role from bio",
        )

    async def test_knows_contact_project(self, contact_call_prompt: str):
        """
        When asked about what the contact is working on, the fast brain
        should use the bio context.
        """
        response = await ask_fast_brain(
            contact_call_prompt,
            "What project is Marcus leading?",
        )

        assert_no_deferral(response, "Asked about contact's project from bio")
        assert_contains(
            response,
            "cloud migration",
            "Should mention cloud migration from bio",
        )

    async def test_boss_bio_available(self, boss_call_prompt: str):
        """
        When on a call with the boss, the fast brain should know the boss's
        background from the bio.
        """
        response = await ask_fast_brain(
            boss_call_prompt,
            "What's the name of my company again?",
        )

        assert_no_deferral(response, "Asked about boss's company from bio")
        assert_contains(
            response,
            "TechStartup",
            "Should mention company name from boss bio",
        )


# =============================================================================
# Test Class: Meet Participants
# =============================================================================


@pytest.mark.asyncio
class TestMeetParticipants:
    """Tests that the fast brain knows about all participants in a meet."""

    async def test_knows_all_participant_names(self, meet_prompt: str):
        """
        When asked who is on the call, the fast brain should list all
        participants without deferring.
        """
        response = await ask_fast_brain(
            meet_prompt,
            "Who's on this call right now?",
        )

        assert_no_deferral(response, "Asked who is on the meet")
        assert_contains(response, "Sarah", "Should mention Sarah")
        assert_contains(response, "Marcus", "Should mention Marcus")
        assert_contains(response, "Priya", "Should mention Priya")

    async def test_knows_participant_role(self, meet_prompt: str):
        """
        When asked about a specific participant's role, the fast brain
        should answer using their bio.
        """
        response = await ask_fast_brain(
            meet_prompt,
            "What's Priya's role?",
        )

        assert_no_deferral(response, "Asked about participant's role")
        assert_contains(
            response,
            "product manager",
            "Should mention Priya's role from bio",
        )

    async def test_knows_participant_company(self, meet_prompt: str):
        """
        When asked about a specific participant's company, the fast brain
        should answer using their bio.
        """
        response = await ask_fast_brain(
            meet_prompt,
            "Which company is Marcus from?",
        )

        assert_no_deferral(response, "Asked about participant's company")
        assert_contains(
            response,
            "ClientCorp",
            "Should mention Marcus's company from bio",
        )


# =============================================================================
# Test Class: Prompt Content Verification (non-LLM)
# =============================================================================


class TestPromptContent:
    """Unit tests verifying the prompt builder includes expected content."""

    def test_assistant_name_in_prompt(self, boss_call_prompt: str):
        """Prompt includes assistant name when provided."""
        assert "My name is Alex." in boss_call_prompt

    def test_boss_bio_in_prompt(self, boss_call_prompt: str):
        """Prompt includes boss bio when provided."""
        assert "CEO of TechStartup" in boss_call_prompt

    def test_contact_bio_in_prompt(self, contact_call_prompt: str):
        """Prompt includes contact bio when provided."""
        assert "VP of Engineering at ClientCorp" in contact_call_prompt

    def test_participants_in_prompt(self, meet_prompt: str):
        """Prompt includes all participant details."""
        assert "Call participants" in meet_prompt
        assert "Sarah Chen" in meet_prompt
        assert "Marcus Rivera" in meet_prompt
        assert "Priya Sharma" in meet_prompt
        assert "CEO of TechStartup" in meet_prompt
        assert "VP of Engineering at ClientCorp" in meet_prompt
        assert "Product Manager at TechStartup" in meet_prompt

    def test_no_name_when_none(self, base_prompt_kwargs: dict):
        """Prompt omits name line when assistant_name is None."""
        kwargs = {**base_prompt_kwargs, "assistant_name": None}
        prompt = build_voice_agent_prompt(**kwargs, is_boss_user=True).flatten()
        assert "My name is" not in prompt

    def test_no_contact_bio_when_none(self, base_prompt_kwargs: dict):
        """Prompt omits contact bio line when contact_bio is None."""
        prompt = build_voice_agent_prompt(
            **base_prompt_kwargs,
            is_boss_user=False,
            contact_first_name="John",
            contact_surname="Doe",
            contact_phone_number="+15550000000",
            contact_email="john@example.com",
            contact_bio=None,
        ).flatten()
        assert "- Bio:" not in prompt.split("Contact details")[1].split("\n\n")[0]

    def test_no_participants_when_none(self, base_prompt_kwargs: dict):
        """Prompt omits participants section when not provided."""
        prompt = build_voice_agent_prompt(
            **base_prompt_kwargs,
            is_boss_user=True,
        ).flatten()
        assert "Call participants" not in prompt

    def test_data_handling_section_present(self, boss_call_prompt: str):
        """Prompt contains the data handling rules section."""
        assert "How I handle data" in boss_call_prompt
        assert "Never fabricate data" in boss_call_prompt

    def test_deferral_examples_present(self, boss_call_prompt: str):
        """Prompt contains active deferral examples (not refusal language)."""
        assert "Let me check on that" in boss_call_prompt
        assert "I can't access" not in boss_call_prompt.split("NEVER say")[0]
