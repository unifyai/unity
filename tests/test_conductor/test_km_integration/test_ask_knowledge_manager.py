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


KNOWLEDGE_QUESTIONS: list[str] = [
    "Summarise the employee onboarding policy.",
    "What are our office hours?",
    "List return policies for ACME by effective date.",
    "What warranty information do we hold about Tesla vehicles?",
]


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("question", KNOWLEDGE_QUESTIONS)
@_handle_project
async def test_knowledge_questions_use_only_knowledge_manager_tool(question: str):
    cond = SimulatedConductor(
        description=(
            "Assistant focused on stored knowledge; contacts, tasks, and transcripts exist but are not needed for these queries."
        ),
    )

    handle = await cond.ask(
        question,
        _return_reasoning_steps=True,
    )

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # The only executed tool must be SimulatedKnowledgeManager.ask and it should run exactly once
    executed_list = _tool_names_from_messages(messages)
    executed = set(executed_list)
    assert executed, "Expected at least one tool call to occur"
    assert executed == {
        "SimulatedKnowledgeManager_ask",
    }, f"Only SimulatedKnowledgeManager_ask should run, saw: {sorted(executed)}"
    assert (
        executed_list.count("SimulatedKnowledgeManager_ask") == 1
    ), f"Expected exactly one SimulatedKnowledgeManager_ask call, saw order: {executed_list}"

    # Additionally confirm that any assistant tool selection(s) referenced only that tool
    requested = set(_assistant_requested_tool_names(messages))
    assert requested, "Assistant should have requested at least one tool"
    assert requested <= {
        "SimulatedKnowledgeManager_ask",
    }, f"Assistant should request only SimulatedKnowledgeManager_ask, saw: {sorted(requested)}"
