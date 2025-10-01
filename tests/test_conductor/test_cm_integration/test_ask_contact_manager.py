from __future__ import annotations

import asyncio

import pytest

from unity.conductor.simulated import SimulatedConductor
from tests.helpers import _handle_project


def _tool_names_from_messages(msgs: list[dict]) -> list[str]:
    names: list[str] = []
    for m in msgs:
        if m.get("role") == "tool":
            name = m.get("name") or ""
            if name and not str(name).startswith("check_status_"):
                names.append(str(name))
    return names


def _assistant_requested_tool_names(msgs: list[dict]) -> list[str]:
    names: list[str] = []
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m.get("tool_calls") or []:
                fn = (tc or {}).get("function", {}) or {}
                name = fn.get("name") or ""
                if name and not str(name).startswith("check_status_"):
                    names.append(str(name))
    return names


CONTACT_QUESTIONS: list[str] = [
    "Which of our contacts prefers to be contacted by phone?",
    "Find the email address for the contact named Sarah (use your contacts only).",
    "List any contacts located in Berlin.",
    "Who is the primary point of contact for the Contoso account?",
]


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("question", CONTACT_QUESTIONS)
@_handle_project
async def test_contact_questions_use_only_contact_manager_tool(question: str):
    cond = SimulatedConductor(
        description=(
            "Operations assistant managing contacts, tasks, transcripts, and knowledge."
        ),
    )

    handle = await cond.ask(
        question,
        _return_reasoning_steps=True,
    )

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # Verify that the only executed tool is ContactManager.ask and it ran exactly once
    executed_list = _tool_names_from_messages(messages)
    executed = set(executed_list)
    assert executed, "Expected at least one tool call to occur"
    assert executed == {
        "ContactManager_ask",
    }, f"Only ContactManager_ask should run, saw: {sorted(executed)}"
    assert (
        executed_list.count("ContactManager_ask") == 1
    ), f"Expected exactly one ContactManager_ask call, saw order: {executed_list}"

    # Additionally confirm that any assistant tool selection(s) referenced only that tool
    requested = set(_assistant_requested_tool_names(messages))
    assert requested, "Assistant should have requested at least one tool"
    assert requested <= {
        "ContactManager_ask",
    }, f"Assistant should request only ContactManager_ask, saw: {sorted(requested)}"
