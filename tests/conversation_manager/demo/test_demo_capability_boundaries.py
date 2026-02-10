"""
tests/conversation_manager/demo/test_demo_capability_boundaries.py
===================================================================

Eval tests verifying that the assistant handles capability boundaries
gracefully in demo mode.

When the prospect asks the assistant to do a specific task beyond demo
capabilities (research, task management, file access, etc.), the
assistant should redirect to sign-up at unify.ai.

When the prospect asks a general "what can you do?" question, the
assistant should paint an impressive picture without forcing a sign-up
pitch — it should feel like talking to a capable new colleague, not
watching an advert.
"""

import json
import re
import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import (
    filter_events_by_type,
)
from unity.common.llm_client import new_llm_client
from unity.conversation_manager.events import (
    SMSReceived,
    SMSSent,
    ActorHandleStarted,
)

pytestmark = pytest.mark.eval


# ─────────────────────────────────────────────────────────────────────────────
# LLM Judge
# ─────────────────────────────────────────────────────────────────────────────


def _llm_judge_demo_response(
    *,
    user_question: str,
    assistant_response: str,
    criteria: str,
    anti_criteria: str | None = None,
) -> None:
    """Use an LLM to judge whether the assistant's demo response is adequate."""
    judge = new_llm_client(async_client=False)
    system_prompt = (
        "You are a strict but fair judge evaluating a virtual assistant's response "
        "during a demo (pre-signup) interaction. The assistant has limited capabilities "
        "in demo mode — it cannot search records, manage tasks, browse the web, etc. "
        "\n\n"
        "For specific capability requests (research, tasks, files), the assistant should "
        "redirect toward sign-up at unify.ai. For general 'what can you do?' questions, "
        "painting a compelling picture is sufficient — a sign-up mention is welcome but "
        "not required. The response should feel like talking to a capable colleague, not "
        "watching an advertisement.\n"
        "\n"
        "Judge the response on:\n"
        "1. WARMTH: Does it sound enthusiastic and positive, not apologetic or broken?\n"
        "2. NATURALNESS: Does it sound like a real colleague, not a canned error or ad?\n"
        "3. CRITERIA FIT: Does it meet the specific criteria provided for this question?\n"
        "\n"
        "A response can use different words — semantic equivalence is fine. "
        "The key is that the user feels excited about what's possible, not frustrated.\n"
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

    verdict = _extract_pass_bool(result_text)
    reason = _extract_reason(result_text)

    assert verdict is True, (
        f"\nLLM Judge FAILED for demo capability boundary.\n"
        f"Question: {user_question!r}\n"
        f"Response: {assistant_response!r}\n"
        f"Criteria: {criteria!r}\n"
        f"Anti-criteria: {anti_criteria!r}\n"
        f"Judge reason: {reason}\n"
        f"Raw judge output: {result_text!r}"
    )
    print(f"LLM Judge: PASS — {reason}")


def _extract_pass_bool(text: str) -> bool | None:
    s = (text or "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and isinstance(obj.get("pass"), bool):
            return obj["pass"]
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, dict) and isinstance(obj.get("pass"), bool):
                return obj["pass"]
        except Exception:
            pass
    m2 = re.search(r'"?pass"?\s*[:=]\s*(true|false)\b', s, re.IGNORECASE)
    if m2:
        return m2.group(1).lower() == "true"
    return None


def _extract_reason(text: str) -> str:
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


async def _ask_and_get_reply(cm, contact: dict, question: str) -> str:
    """Send an SMS and return the first reply content."""
    result = await cm.step_until_wait(
        SMSReceived(contact=contact, content=question),
    )
    sms_replies = filter_events_by_type(result.output_events, SMSSent)
    assert sms_replies, f"Expected at least one SMS reply for: {question!r}"
    return sms_replies[0].content


async def _ask_and_check_no_act(cm, contact: dict, question: str) -> str:
    """Send an SMS, verify no act was triggered, and return the reply."""
    result = await cm.step_until_wait(
        SMSReceived(contact=contact, content=question),
    )

    # Verify act was NOT triggered
    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert len(actor_events) == 0, (
        f"act should not be triggered in demo mode for: {question!r} — "
        f"got {len(actor_events)} ActorHandleStarted events"
    )

    sms_replies = filter_events_by_type(result.output_events, SMSSent)
    assert sms_replies, f"Expected at least one SMS reply for: {question!r}"
    return sms_replies[0].content


def _setup_boss_with_name(cm):
    """Set up boss with a name so they're a realistic contact."""
    cm.cm.contact_manager.update_contact(
        contact_id=1,
        first_name="Richard",
        phone_number="+447700900123",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Capability boundary messaging
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_research_request_handled_gracefully(initialized_cm):
    """When the prospect asks for web research, the assistant should redirect to sign-up."""
    _setup_boss_with_name(initialized_cm)
    boss = initialized_cm.contact_index.get_contact(1)

    reply = await _ask_and_check_no_act(
        initialized_cm,
        boss,
        "Can you research the latest trends in AI for me?",
    )

    _llm_judge_demo_response(
        user_question="Can you research the latest trends in AI for me?",
        assistant_response=reply,
        criteria=(
            "Should acknowledge the request positively and explain that research "
            "capabilities are available after sign-up at unify.ai. Should sound "
            "enthusiastic about what's possible, not apologetic."
        ),
        anti_criteria=(
            "Should NOT say 'I can't do that' flatly. Should NOT sound broken or "
            "confused. Should NOT try to make up research results."
        ),
    )


@pytest.mark.asyncio
@_handle_project
async def test_task_request_handled_gracefully(initialized_cm):
    """When the prospect asks about task management, redirect to sign-up."""
    _setup_boss_with_name(initialized_cm)
    boss = initialized_cm.contact_index.get_contact(1)

    reply = await _ask_and_check_no_act(
        initialized_cm,
        boss,
        "Can you set a reminder for me to follow up with the investor next Tuesday?",
    )

    _llm_judge_demo_response(
        user_question="Can you set a reminder for me to follow up with the investor next Tuesday?",
        assistant_response=reply,
        criteria=(
            "Should acknowledge this is exactly the kind of thing the assistant can "
            "handle, and direct toward sign-up at unify.ai to enable it."
        ),
        anti_criteria=(
            "Should NOT pretend to set the reminder. Should NOT be dismissive. "
            "Should NOT give a long technical explanation."
        ),
    )


@pytest.mark.asyncio
@_handle_project
async def test_what_can_you_do_in_demo(initialized_cm):
    """Classic 'what can you do?' question — should be impressive without a sales pitch."""
    _setup_boss_with_name(initialized_cm)
    boss = initialized_cm.contact_index.get_contact(1)

    reply = await _ask_and_check_no_act(
        initialized_cm,
        boss,
        "So what can you actually do for me?",
    )

    _llm_judge_demo_response(
        user_question="So what can you actually do for me?",
        assistant_response=reply,
        criteria=(
            "Should convey that the assistant is a capable remote virtual employee "
            "who can handle communication, tasks, research, software, and more. "
            "Should be enthusiastic and paint a compelling picture of what working "
            "together looks like. A sign-up mention is fine but NOT required."
        ),
        anti_criteria=(
            "Should NOT sound limited or apologetic. Should NOT list only what it "
            "CAN'T do. Should NOT be a long bullet-point feature list. "
            "Should NOT feel like a sales pitch or advertisement."
        ),
    )


@pytest.mark.asyncio
@_handle_project
async def test_file_request_handled_gracefully(initialized_cm):
    """When asked about files/documents, redirect to sign-up."""
    _setup_boss_with_name(initialized_cm)
    boss = initialized_cm.contact_index.get_contact(1)

    reply = await _ask_and_check_no_act(
        initialized_cm,
        boss,
        "I have a spreadsheet I need you to update — can you handle that?",
    )

    _llm_judge_demo_response(
        user_question="I have a spreadsheet I need you to update — can you handle that?",
        assistant_response=reply,
        criteria=(
            "Should confirm that spreadsheet work is within its capabilities "
            "and direct toward sign-up to enable it. Should be positive."
        ),
        anti_criteria=(
            "Should NOT refuse outright. Should NOT pretend to handle the spreadsheet. "
            "Should NOT be overly technical about why it can't do it right now."
        ),
    )


@pytest.mark.asyncio
@_handle_project
async def test_simple_conversation_works_fine(initialized_cm):
    """Normal conversational exchange should work perfectly in demo mode."""
    _setup_boss_with_name(initialized_cm)
    boss = initialized_cm.contact_index.get_contact(1)

    reply = await _ask_and_get_reply(
        initialized_cm,
        boss,
        "Hi Lucy! Nice to meet you. How are you doing?",
    )

    # Simple conversation should just work — no sign-up nudge needed
    assert reply, "Assistant should respond to casual conversation"
    assert len(reply) > 5, "Reply should be substantive"
