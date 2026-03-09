"""
Naturalness target tests for fast-brain voice behavior.

These tests define desired conversation quality targets that go beyond the current
deferral-safety baseline. Some remain xfail until prompt/policy updates land.
"""

from __future__ import annotations

import pytest

from unity.conversation_manager.prompt_builders import build_voice_agent_prompt

from tests.conversation_manager.voice.test_fast_brain_deferral import (
    MODEL_TTS,
    get_fast_brain_response,
)

pytestmark = pytest.mark.eval


def _build_target_prompt() -> str:
    return build_voice_agent_prompt(
        bio="A helpful and efficient assistant.",
        assistant_name="Alex",
        boss_first_name="Yusha",
        boss_surname="Arif",
        boss_phone_number="+19294608302",
        boss_email_address="yusha@unify.ai",
        boss_bio="Founder and engineer. Prefers concise, practical updates.",
        is_boss_user=True,
        contact_rolling_summary=None,
    ).flatten()


class TestNaturalnessTargets:
    @pytest.mark.asyncio
    async def test_progress_update_mentions_active_work_item(self):
        prompt = _build_target_prompt()
        conversation = [
            {
                "role": "user",
                "content": "Create a Bob contact and set an Apply to OpenAI task.",
            },
            {"role": "assistant", "content": "Let me check on that."},
            {
                "role": "assistant",
                "content": (
                    "[notification] Got it - I'm creating a Bob contact now, and "
                    "setting up an Apply to OpenAI task."
                ),
            },
            {
                "role": "assistant",
                "content": "Creating Bob contact and setting up the Apply to OpenAI task.",
            },
            {"role": "user", "content": "Any updates?"},
        ]

        response = await get_fast_brain_response(prompt, conversation, model=MODEL_TTS)
        response_lower = response.lower()
        mentions_active_work = any(
            marker in response_lower
            for marker in [
                "bob",
                "contact",
                "task",
                "apply to openai",
                "backend engineer",
                "frontend engineer",
            ]
        )
        assert mentions_active_work, (
            "Expected a specific in-progress update tied to the active work item.\n"
            f"Response: {response}"
        )

    @pytest.mark.asyncio
    async def test_action_request_sets_realistic_time_expectation(self):
        """For multi-step action requests, the fast brain should signal that the
        work may take minutes — not imply near-instant completion with "one moment"
        or "just a second". Data lookups are genuinely quick and short deferrals
        are fine for those (covered by test_fast_brain_deferral.py), but action
        requests like sending emails or creating records take several minutes.
        """
        prompt = _build_target_prompt()
        conversation = [
            {
                "role": "user",
                "content": (
                    "Can you draft and send an email to Sarah about the Q3 report?"
                ),
            },
        ]

        response = await get_fast_brain_response(prompt, conversation, model=MODEL_TTS)
        response_lower = response.lower()

        short_wait_only = [
            "one moment",
            "just a second",
            "just a sec",
            "one sec",
            "give me a second",
            "bear with me",
        ]
        long_wait_markers = [
            "minute",
            "a while",
            "a bit",
            "take a little",
            "let you know",
            "when i'm done",
            "when it's done",
            "when it's ready",
            "update you",
            "circle back",
            "get back to you",
        ]

        uses_short_wait = any(p in response_lower for p in short_wait_only)
        sets_expectation = any(m in response_lower for m in long_wait_markers)

        assert not uses_short_wait or sets_expectation, (
            "For multi-step action requests, the fast brain should set realistic "
            "time expectations (minutes, not seconds). Short-wait filler like "
            "'one moment' is misleading for tasks that take several minutes.\n"
            f"Response: {response}"
        )

    @pytest.mark.asyncio
    async def test_redundant_checking_guidance_avoids_same_deferral_phrase(self):
        prompt = _build_target_prompt()
        conversation = [
            {
                "role": "user",
                "content": "Do I have a contact named Bob?",
            },
            {"role": "assistant", "content": "Let me check on that."},
            {
                "role": "assistant",
                "content": (
                    "[notification] Sure, let me check my records for a contact "
                    "named Bob. One moment."
                ),
            },
        ]

        response = await get_fast_brain_response(prompt, conversation, model=MODEL_TTS)
        response_lower = response.lower()

        assert "let me check on that" not in response_lower, (
            "Expected response to avoid repeating the exact prior deferral phrase.\n"
            f"Response: {response}"
        )
