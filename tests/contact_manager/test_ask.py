from __future__ import annotations

import asyncio
import pytest
import json
import re
from typing import List, Dict, Any, Optional

from unity.contact_manager.contact_manager import ContactManager
from unity.contact_manager.types.contact import Contact
from unity.common.llm_client import new_llm_client
from tests.assertion_helpers import assertion_failed, find_tool_calls_and_results
from tests.helpers import _handle_project

# All tests in this file exercise end-to-end LLM reasoning for contact retrieval
pytestmark = pytest.mark.eval


def _llm_judge_contact_retrieval(
    question: str,
    expected_answer_fragment: str,  # A key piece of info expected in the answer
    candidate_answer: str,
    reasoning_steps: List[Dict[str, Any]],
    all_contacts_for_context: Optional[List[Contact]] = None,
) -> None:
    """
    Uses an LLM to judge if the candidate_answer correctly answers the question,
    focusing on the presence of expected_answer_fragment.
    """
    judge = new_llm_client(async_client=False)
    system_prompt = (
        "You are a meticulous but fair unit-test judge for contact information retrieval. "
        "You will be given a question, an expected key piece of information that the answer should contain, "
        "and a candidate answer from the system. "
        "Your task is to decide if the candidate answer accurately and sufficiently answers the question, "
        "specifically checking if it includes the expected key information. "
        "Minor formatting or wording differences are acceptable as long as the core factual information is present. "
        "Note: The contact list always includes two system contacts (a default assistant and a default user) "
        "which are part of the standard data model. These may or may not appear in answers depending on the query, "
        "and their presence or absence should not affect your judgment of correctness. "
        "Focus only on whether the expected_key_information is present in the candidate_answer. "
        'Respond ONLY with valid JSON of the form {"correct": true} or {"correct": false}.'
    )
    judge.set_system_message(system_prompt)

    payload_dict = {
        "question": question,
        "expected_key_information": expected_answer_fragment,
        "candidate_answer": candidate_answer,
    }
    if all_contacts_for_context:
        payload_dict["relevant_contacts_data_for_context"] = [
            c.model_dump_json() for c in all_contacts_for_context
        ]

    payload = json.dumps(payload_dict, indent=2)
    result_text = judge.generate(payload)

    # Be tolerant to non-JSON wrappers (markdown fences, extra prose) while
    # preserving the same semantic contract: a boolean "correct" field decides.
    def _extract_correct_bool(text: str) -> Optional[bool]:
        s = (text or "").strip()

        # 1) Direct JSON
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and isinstance(obj.get("correct"), bool):
                return obj.get("correct")
        except Exception:
            pass

        # 2) Code fences (```json ... ``` or ``` ... ```)
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
        if m:
            inner = m.group(1).strip()
            try:
                obj = json.loads(inner)
                if isinstance(obj, dict) and isinstance(obj.get("correct"), bool):
                    return obj.get("correct")
            except Exception:
                pass

        # 3) First balanced JSON object in the string (best-effort)
        def _balanced_json_candidates(text_: str):
            start = None
            depth = 0
            for i, ch in enumerate(text_):
                if ch == "{":
                    if start is None:
                        start = i
                    depth += 1
                elif ch == "}":
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and start is not None:
                            yield text_[start : i + 1]
                            start = None

        for cand in _balanced_json_candidates(s):
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict) and isinstance(obj.get("correct"), bool):
                    return obj.get("correct")
            except Exception:
                continue

        # 4) Loose regex on key:value pairs (e.g., correct: true)
        m2 = re.search(r"\b\"?correct\"?\s*[:=]\s*(true|false)\b", s, re.IGNORECASE)
        if m2:
            return m2.group(1).lower() == "true"

        # 5) Bare booleans (rare, but cheap to support)
        bare = s.lower()
        if bare == "true":
            return True
        if bare == "false":
            return False
        return None

    verdict_bool = _extract_correct_bool(result_text)
    is_correct = bool(verdict_bool) if verdict_bool is not None else False

    assert is_correct is True, assertion_failed(
        f"Answer containing '{expected_answer_fragment}'",
        candidate_answer,
        reasoning_steps,
        f"LLM Judge validation for: {question}",
        {
            "all_contacts_for_context": [
                c.model_dump() for c in all_contacts_for_context or []
            ],
        },
    )
    print(
        f"LLM Judge: OK for question '{question}' - found '{expected_answer_fragment}' in answer.",
    )


# Test questions and their expected semantic content
QUESTIONS_CONTACT_ASK = [
    ("What is Alice Smith's phone number?", "1112223333"),
    ("Find Bob Johnson's email.", "bobbyj@example.net"),
    ("Who has the email address diana@themyscira.com?", "Diana Prince"),
    ("List all contacts with the surname Smith.", "Alice Smith"),  # Expect Alice Smith
    (
        "Are there any contacts without a phone number?",
        "Charlie Brown",
    ),  # Charlie has no phone
]


@_handle_project
@pytest.mark.asyncio
@pytest.mark.parametrize("question, expected_fragment", QUESTIONS_CONTACT_ASK)
async def test_ask_semantic(
    question: str,
    expected_fragment: str,
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Tests various semantic queries against the ContactManager's ask method."""
    cm, _ = contact_manager_scenario

    handle = await cm.ask(question, _return_reasoning_steps=True)
    candidate_answer, reasoning_steps = await handle.result()

    # For better judgment context, fetch all contacts to pass to the LLM judge
    all_contacts_dict = await asyncio.to_thread(cm.filter_contacts)
    all_contacts = (
        all_contacts_dict["contacts"]
        if isinstance(all_contacts_dict, dict)
        else all_contacts_dict
    )

    _llm_judge_contact_retrieval(
        question,
        expected_fragment,
        candidate_answer,
        reasoning_steps,
        all_contacts,
    )


@_handle_project
@pytest.mark.asyncio
async def test_ask_parent_context(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test ask method with parent_chat_context for disambiguation."""
    cm, _ = contact_manager_scenario
    # Alice Smith and Alice Wonder exist.

    parent_ctx = [
        {"role": "user", "content": "We were discussing Alice Smith earlier."},
        {"role": "assistant", "content": "Okay, Alice Smith. What about her?"},
    ]
    question = "What's her email address?"
    expected_email = "alice.smith@example.com"  # Email of Alice Smith

    handle = await cm.ask(
        question,
        _parent_chat_context=parent_ctx,
        _return_reasoning_steps=True,
    )
    candidate_answer, reasoning_steps = await handle.result()

    all_contacts_dict = await asyncio.to_thread(cm.filter_contacts)
    all_contacts = (
        all_contacts_dict["contacts"]
        if isinstance(all_contacts_dict, dict)
        else all_contacts_dict
    )
    _llm_judge_contact_retrieval(
        question,
        expected_email,
        candidate_answer,
        reasoning_steps,
        all_contacts,
    )
    assert "alice.wonder@example.com" not in candidate_answer.lower()


@_handle_project
@pytest.mark.asyncio
async def test_ask_time_check(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """If it's 17:00 UTC and the contact is at UTC+9, local time is ~02:00 → not sensible."""
    cm, _ = contact_manager_scenario

    # Bob Johnson has timezone="Asia/Tokyo" (UTC+9) in seed data,
    # so 17:00 UTC → 02:00 local time
    #
    # Ask the high-level question; include the UTC time in the user message
    question = (
        "It's 17:00 UTC now. I'd like to send an email to Bob Johnson. "
        "Is now a sensible time? Please consider his timezone and include his local time."
    )
    handle = await cm.ask(question, _return_reasoning_steps=True)
    candidate_answer, reasoning_steps = await handle.result()

    answer_lower = (candidate_answer or "").lower()
    # Expect a negative recommendation and mention of ~2am local time
    neg = any(token in answer_lower for token in ["no", "not a good time", "not ideal"])
    mentions_two_am = any(
        pat in answer_lower
        for pat in [
            "2am",
            "2 am",
            "02:00",
            "2:00 am",
            "02:00 am",
            "2:00am",
            "02:00am",
        ]
    )

    assert neg, f"Expected a negative recommendation, got: {candidate_answer!r}"
    assert (
        mentions_two_am
    ), f"Expected mention of ~02:00 local time, got: {candidate_answer!r}"


@_handle_project
@pytest.mark.asyncio
async def test_ask_clarification(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test ask method with clarification request and response when query is ambiguous."""
    cm, _ = contact_manager_scenario
    # Two "Alice" contacts exist.

    clar_up_q = asyncio.Queue()
    clar_down_q = asyncio.Queue()

    question = "What is Alice's phone number? If there is more than one then request a clarification."
    # We expect clarification, then provide info for Alice Wonder (phone 1110001111)
    expected_phone_after_clarification = "1110001111"

    handle = await cm.ask(
        question,
        _clarification_up_q=clar_up_q,
        _clarification_down_q=clar_down_q,
        _return_reasoning_steps=True,
    )

    # Expect a clarification question
    clarification_question_text = await asyncio.wait_for(clar_up_q.get(), timeout=300)

    # Provide clarification
    await clar_down_q.put("I mean Alice Wonder.")

    candidate_answer, reasoning_steps = await handle.result()
    all_contacts_dict = await asyncio.to_thread(cm.filter_contacts)
    all_contacts = (
        all_contacts_dict["contacts"]
        if isinstance(all_contacts_dict, dict)
        else all_contacts_dict
    )
    _llm_judge_contact_retrieval(
        question + " (after clarifying 'Alice Wonder')",
        expected_phone_after_clarification,
        candidate_answer,
        reasoning_steps,
        all_contacts,
    )
    assert "1112223333" not in candidate_answer  # Phone of Alice Smith


@_handle_project
@pytest.mark.asyncio
async def test_ask_interject(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test interjecting a new query while an ask operation is in progress."""
    cm, _ = contact_manager_scenario

    initial_question = "Find Charlie Brown's contact details."
    interjected_question = "Also, what is Bob Johnson's phone number?"
    expected_fragment_charlie = "goodgrief@example.org"  # Charlie's email
    expected_fragment_bob = "4445556666"  # Bob's phone

    handle = await cm.ask(initial_question, _return_reasoning_steps=True)
    await asyncio.sleep(0.1)  # Allow initial query to start
    await handle.interject(interjected_question)
    candidate_answer, reasoning_steps = await handle.result()

    all_contacts_dict = await asyncio.to_thread(cm.filter_contacts)
    all_contacts = (
        all_contacts_dict["contacts"]
        if isinstance(all_contacts_dict, dict)
        else all_contacts_dict
    )
    _llm_judge_contact_retrieval(
        f"{initial_question} AND {interjected_question}",
        expected_fragment_charlie,
        candidate_answer,
        reasoning_steps,
        all_contacts,
    )
    _llm_judge_contact_retrieval(
        f"{initial_question} AND {interjected_question}",
        expected_fragment_bob,
        candidate_answer,
        reasoning_steps,
        all_contacts,
    )


@_handle_project
@pytest.mark.asyncio
async def test_ask_stop(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test stopping an ask operation."""
    cm, _ = contact_manager_scenario
    handle = await cm.ask(
        "Find all contacts and list their full details, this might take a while.",
    )
    await asyncio.sleep(0.05)  # Let it start
    await handle.stop()
    await handle.result()
    assert handle.done()


@_handle_project
@pytest.mark.asyncio
async def test_ask_uses_reduce_for_numeric_aggregation(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Verify LLM uses reduce tool for numeric aggregation questions."""
    cm, _ = contact_manager_scenario

    handle = await cm.ask(
        "What is the maximum contact_id stored?",
        _return_reasoning_steps=True,
    )
    answer, steps = await handle.result()

    # Assert reduce tool was called
    reduce_calls, _ = find_tool_calls_and_results(steps, "reduce")
    assert reduce_calls, assertion_failed(
        "reduce tool to be called",
        f"steps without reduce: {[s for s in steps if s.get('role') == 'assistant']}",
        steps,
        "LLM should use reduce tool for numeric aggregation",
    )
