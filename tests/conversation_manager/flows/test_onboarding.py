"""
tests/conversation_manager/flows/test_onboarding.py
====================================================

Tests for onboarding flows — verifying the brain can handle "what can you do?",
"how do I get started?", and similar questions naturally and accurately.

Uses an LLM judge to evaluate whether the assistant's responses are:
- Accurate (consistent with the onboarding reference)
- Natural (not reciting a feature list or reading from a script)
- Concise (not dumping everything at once)

These tests deliberately use tricky, ambiguous, or boundary-pushing questions
to stress-test the onboarding behavior.
"""

import json
import re
import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import (
    filter_events_by_type,
)
from tests.conversation_manager.conftest import BOSS
from unity.common.llm_client import new_llm_client
from unity.conversation_manager.events import (
    SMSReceived,
    SMSSent,
)

pytestmark = pytest.mark.eval


# =============================================================================
# LLM Judge
# =============================================================================


def _llm_judge_onboarding(
    *,
    user_question: str,
    assistant_response: str,
    criteria: str,
    anti_criteria: str | None = None,
) -> None:
    """Use an LLM to judge whether the assistant's onboarding response is adequate.

    Parameters
    ----------
    user_question : str
        The question the user asked.
    assistant_response : str
        The assistant's response text.
    criteria : str
        What the response SHOULD contain or convey (semantic, not exact match).
    anti_criteria : str | None
        What the response should NOT contain (e.g., feature dumps, scripted tone).
    """
    judge = new_llm_client(async_client=False)
    system_prompt = (
        "You are a strict but fair judge evaluating a virtual assistant's onboarding response. "
        "The assistant is meant to behave like a remote colleague — natural, concise, and helpful. "
        "You will be given the user's question, the assistant's response, criteria the response "
        "SHOULD satisfy, and optionally anti-criteria the response should NOT exhibit. "
        "\n\n"
        "Judge the response on:\n"
        "1. ACCURACY: Does it convey the information described in the criteria?\n"
        "2. NATURALNESS: Does it sound like a real colleague, not a chatbot reading a script?\n"
        "3. CONCISENESS: Is it appropriately brief — not a wall of text or feature dump?\n"
        "\n"
        "A response can be correct even if it doesn't use the exact words from the criteria — "
        "semantic equivalence is fine. Minor omissions are acceptable if the core point is conveyed. "
        "Be strict about anti-criteria violations though.\n"
        "\n"
        'Respond ONLY with valid JSON: {"pass": true, "reason": "..."} or {"pass": false, "reason": "..."}'
    )
    judge.set_system_message(system_prompt)

    payload = {
        "user_question": user_question,
        "assistant_response": assistant_response,
        "criteria": criteria,
        "anti_criteria": anti_criteria or "None specified.",
    }
    result_text = judge.generate(json.dumps(payload, indent=2))

    # Extract verdict
    verdict = _extract_pass_bool(result_text)
    reason = _extract_reason(result_text)

    assert verdict is True, (
        f"\nLLM Judge FAILED for onboarding question.\n"
        f"Question: {user_question!r}\n"
        f"Response: {assistant_response!r}\n"
        f"Criteria: {criteria!r}\n"
        f"Anti-criteria: {anti_criteria!r}\n"
        f"Judge reason: {reason}\n"
        f"Raw judge output: {result_text!r}"
    )
    print(f"LLM Judge: PASS — {reason}")


def _extract_pass_bool(text: str) -> bool | None:
    """Extract the 'pass' boolean from judge output, tolerating formatting quirks."""
    s = (text or "").strip()

    # Try direct JSON parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and isinstance(obj.get("pass"), bool):
            return obj["pass"]
    except Exception:
        pass

    # Try code fences
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, dict) and isinstance(obj.get("pass"), bool):
                return obj["pass"]
        except Exception:
            pass

    # Loose regex
    m2 = re.search(r'"?pass"?\s*[:=]\s*(true|false)\b', s, re.IGNORECASE)
    if m2:
        return m2.group(1).lower() == "true"

    return None


def _extract_reason(text: str) -> str:
    """Extract the 'reason' string from judge output."""
    s = (text or "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj.get("reason", "No reason provided.")
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, dict):
                return obj.get("reason", "No reason provided.")
        except Exception:
            pass
    return text[:200] if text else "No output from judge."


# =============================================================================
# Helper: send a message from the boss and get the SMS reply
# =============================================================================


async def _ask_and_get_reply(cm, question: str) -> str:
    """Send an SMS from the boss, step until wait, and return the reply text."""
    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=question,
        ),
    )
    sms_replies = filter_events_by_type(result.output_events, SMSSent)
    assert (
        sms_replies
    ), f"Expected at least one SMS reply, got none for question: {question!r}"
    # Use the first SMS reply — this is the brain's intended response.
    # Subsequent replies (if any) are typically duplicates from extra LLM steps
    # before the brain calls `wait`.
    return sms_replies[0].content


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_what_can_you_do(initialized_cm):
    """Classic onboarding question — should give a natural, brief answer."""
    reply = await _ask_and_get_reply(initialized_cm, "What can you actually do?")

    _llm_judge_onboarding(
        user_question="What can you actually do?",
        assistant_response=reply,
        criteria=(
            "Should convey that the assistant is a remote virtual employee capable of "
            "communication, research, using software, handling tasks, and similar work "
            "a remote colleague could do. Should mention at least some concrete capabilities."
        ),
        anti_criteria=(
            "Should NOT be an exhaustive bullet-point feature list. Should NOT sound "
            "like reading from a product brochure or FAQ. Should be concise — a few "
            "sentences at most, not a wall of text."
        ),
    )


@pytest.mark.asyncio
@_handle_project
async def test_how_do_i_get_started(initialized_cm):
    """User is new and unsure how to begin — should be guided to unify.ai."""
    reply = await _ask_and_get_reply(
        initialized_cm,
        "I'm interested but I don't really know where to start. What do I do?",
    )

    _llm_judge_onboarding(
        user_question="I'm interested but I don't really know where to start. What do I do?",
        assistant_response=reply,
        criteria=(
            "Should mention unify.ai as the platform to visit and/or suggest the user "
            "can just tell the assistant what they need. Should be welcoming and practical."
        ),
        anti_criteria=(
            "Should NOT be overly long or dump the entire onboarding process. "
            "Should NOT sound robotic or scripted."
        ),
    )


@pytest.mark.asyncio
@_handle_project
async def test_are_you_a_bot(initialized_cm):
    """Deliberately provocative — tries to get the assistant to break character."""
    reply = await _ask_and_get_reply(
        initialized_cm,
        "Be honest with me — are you just a chatbot?",
    )

    _llm_judge_onboarding(
        user_question="Be honest with me — are you just a chatbot?",
        assistant_response=reply,
        criteria=(
            "Should frame itself as a remote virtual employee / remote colleague, "
            "not as a chatbot or AI assistant. Should maintain the identity naturally "
            "without being evasive or dishonest."
        ),
        anti_criteria=(
            "Should NOT say 'I am an AI language model' or 'I am a chatbot'. "
            "Should NOT be defensive or give an unnaturally long explanation."
        ),
    )


@pytest.mark.asyncio
@_handle_project
async def test_can_you_use_my_computer(initialized_cm):
    """Specific capability question about remote access."""
    reply = await _ask_and_get_reply(
        initialized_cm,
        "I need help with something on my laptop — can you actually access it?",
    )

    _llm_judge_onboarding(
        user_question="I need help with something on my laptop — can you actually access it?",
        assistant_response=reply,
        criteria=(
            "Should confirm that remote access to the user's machine is possible, "
            "and/or offer to help directly if access is granted."
        ),
        anti_criteria=(
            "Should NOT refuse or say it's impossible. Should NOT give a long "
            "technical explanation of how remote access works."
        ),
    )


@pytest.mark.asyncio
@_handle_project
async def test_vague_existential_question(initialized_cm):
    """Deliberately vague — tests that the brain doesn't over-explain."""
    reply = await _ask_and_get_reply(
        initialized_cm,
        "So... what exactly is this?",
    )

    _llm_judge_onboarding(
        user_question="So... what exactly is this?",
        assistant_response=reply,
        criteria=(
            "Should give a brief explanation of what the assistant is — a remote virtual "
            "employee or remote colleague. Should be conversational and not overly formal."
        ),
        anti_criteria=(
            "Should NOT launch into a comprehensive feature dump. Should NOT be more "
            "than a few sentences. The response should match the casual tone of the question."
        ),
    )
