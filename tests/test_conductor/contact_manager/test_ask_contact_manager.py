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

    # Verify that the only executed tool(s) are SimulatedContactManager.ask
    executed = set(_tool_names_from_messages(messages))
    # It is acceptable for the loop to make multiple calls to the same tool;
    # what matters is that no other tools ran.
    assert executed, "Expected at least one tool call to occur"
    assert executed == {
        "SimulatedContactManager_ask",
    }, f"Only SimulatedContactManager_ask should run, saw: {sorted(executed)}"

    # Additionally confirm that any assistant tool selection(s) referenced only that tool
    requested = set(_assistant_requested_tool_names(messages))
    assert requested, "Assistant should have requested at least one tool"
    assert requested <= {
        "SimulatedContactManager_ask",
    }, f"Assistant should request only SimulatedContactManager_ask, saw: {sorted(requested)}"
