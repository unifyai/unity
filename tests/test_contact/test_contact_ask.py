from __future__ import annotations

import asyncio
import pytest
import json
from typing import List, Dict, Any, Optional

import unify
from unity.contact_manager.contact_manager import ContactManager
from unity.contact_manager.types.contact import Contact
from tests.assertion_helpers import assertion_failed
from tests.helpers import _handle_project, SETTINGS


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
    judge = unify.Unify(
        "o4-mini@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    system_prompt = (
        "You are a meticulous but fair unit-test judge for contact information retrieval. "
        "You will be given a question, an expected key piece of information that the answer should contain, "
        "and a candidate answer from the system. "
        "Your task is to decide if the candidate answer accurately and sufficiently answers the question, "
        "specifically checking if it includes the expected key information. "
        "Minor formatting or wording differences are acceptable as long as the core factual information is present. "
        'Respond ONLY with valid JSON of the form {"correct": true} or {"correct": false}.'
    )
    judge.set_system_message(system_prompt)

    payload_dict = {
        "question": question,
        "expected_key_information": expected_answer_fragment,
        "candidate_answer": candidate_answer,
    }
    if all_contacts_for_context:  # Provide more context to the judge if helpful
        payload_dict["relevant_contacts_data_for_context"] = [
            c.model_dump_json() for c in all_contacts_for_context
        ]

    payload = json.dumps(payload_dict, indent=2)
    result_json = judge.generate(payload)

    try:
        verdict = json.loads(result_json)
        is_correct = verdict.get("correct")
    except json.JSONDecodeError:
        is_correct = False  # Failed to parse judge's response

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
@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("question, expected_fragment", QUESTIONS_CONTACT_ASK)
async def test_ask_semantic_queries(
    question: str,
    expected_fragment: str,
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Tests various semantic queries against the ContactManager's ask method."""
    cm, _ = contact_manager_scenario

    handle = await cm.ask(question, _return_reasoning_steps=True)
    candidate_answer, reasoning_steps = await handle.result()

    # For better judgment context, fetch all contacts to pass to the LLM judge
    all_contacts = await asyncio.to_thread(cm._filter_contacts)

    _llm_judge_contact_retrieval(
        question,
        expected_fragment,
        candidate_answer,
        reasoning_steps,
        all_contacts,
    )


@_handle_project
@pytest.mark.eval
@pytest.mark.asyncio
async def test_ask_with_parent_context(
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
        parent_chat_context=parent_ctx,
        _return_reasoning_steps=True,
    )
    candidate_answer, reasoning_steps = await handle.result()

    all_contacts = await asyncio.to_thread(cm._filter_contacts)
    _llm_judge_contact_retrieval(
        question,
        expected_email,
        candidate_answer,
        reasoning_steps,
        all_contacts,
    )
    assert "alice.wonder@example.com" not in candidate_answer.lower()


@_handle_project
@pytest.mark.eval
@pytest.mark.asyncio
async def test_ask_with_clarification(
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
        clarification_up_q=clar_up_q,
        clarification_down_q=clar_down_q,
        _return_reasoning_steps=True,
    )

    # Expect a clarification question
    clarification_question_text = await asyncio.wait_for(clar_up_q.get(), timeout=300)

    # Provide clarification
    await clar_down_q.put("I mean Alice Wonder.")

    candidate_answer, reasoning_steps = await handle.result()
    all_contacts = await asyncio.to_thread(cm._filter_contacts)
    _llm_judge_contact_retrieval(
        question + " (after clarifying 'Alice Wonder')",
        expected_phone_after_clarification,
        candidate_answer,
        reasoning_steps,
        all_contacts,
    )
    assert "1112223333" not in candidate_answer  # Phone of Alice Smith


@_handle_project
@pytest.mark.eval
@pytest.mark.asyncio
async def test_ask_interjection(
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

    all_contacts = await asyncio.to_thread(cm._filter_contacts)
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
@pytest.mark.eval
@pytest.mark.asyncio
async def test_ask_stop_operation(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test stopping an ask operation."""
    cm, _ = contact_manager_scenario
    handle = await cm.ask(
        "Find all contacts and list their full details, this might take a while.",
    )
    await asyncio.sleep(0.05)  # Let it start
    handle.stop()
    await handle.result()
    assert handle.done()
