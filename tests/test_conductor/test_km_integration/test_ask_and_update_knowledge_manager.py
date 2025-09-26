from __future__ import annotations

import asyncio

import pytest

from unity.conductor.simulated import SimulatedConductor
from tests.helpers import _handle_project


def _normalise_tool_name(name: str) -> str:
    if not name:
        return name
    s = str(name)
    if s.startswith("continue_SimulatedKnowledgeManager_ask"):
        return "SimulatedKnowledgeManager_ask"
    if s.startswith("continue_SimulatedKnowledgeManager_update"):
        return "SimulatedKnowledgeManager_update"
    return s


def _tool_names_from_messages(msgs: list[dict]) -> list[str]:
    names: list[str] = []
    for m in msgs:
        if m.get("role") == "tool":
            name = m.get("name") or ""
            if name and not str(name).startswith("check_status_"):
                names.append(_normalise_tool_name(str(name)))
    return names


def _assistant_requested_tool_names(msgs: list[dict]) -> list[str]:
    names: list[str] = []
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m.get("tool_calls") or []:
                fn = (tc or {}).get("function", {}) or {}
                name = fn.get("name") or ""
                if name and not str(name).startswith("check_status_"):
                    names.append(_normalise_tool_name(str(name)))
    return names


# Each query intentionally contains an unrelated read (ask) and write (update)
COMBINED_QUERIES: list[str] = [
    (
        "What are our office hours? Also store that the support inbox is monitored 7 days a week."
    ),
    (
        "Summarise the onboarding policy for engineers. Also add that a security review occurs in week two."
    ),
    (
        "Tell me the refund policy for ACME by date; also record that exchanges are allowed within 45 days."
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("combined_text", COMBINED_QUERIES)
@_handle_project
async def test_combined_queries_call_both_ask_and_update_once_each(combined_text: str):
    cond = SimulatedConductor(
        description=(
            "Assistant that can both read from and update the knowledge-base; combined queries include a separate question and update."
        ),
    )

    handle = await cond.request(
        combined_text,
        _return_reasoning_steps=True,
    )

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    executed_list = _tool_names_from_messages(messages)
    executed = set(executed_list)
    assert executed, "Expected at least one tool call to occur"

    # Must include both at least once (dynamic continue tools permitted)
    assert (
        executed_list.count("SimulatedKnowledgeManager_ask") >= 1
    ), f"Expected at least one SimulatedKnowledgeManager_ask call, saw order: {executed_list}"
    assert (
        executed_list.count("SimulatedKnowledgeManager_update") >= 1
    ), f"Expected at least one SimulatedKnowledgeManager_update call, saw order: {executed_list}"

    # Order is not strictly enforced, but both must appear
    assert {
        "SimulatedKnowledgeManager_ask",
        "SimulatedKnowledgeManager_update",
    }.issubset(
        executed,
    ), f"Both ask and update must be executed, saw: {sorted(executed)}"

    # Assistant tool requests should reference only ask/update (dynamic continues normalised)
    requested = set(_assistant_requested_tool_names(messages))
    assert requested, "Assistant should have requested at least one tool"
    assert requested <= {
        "SimulatedKnowledgeManager_ask",
        "SimulatedKnowledgeManager_update",
    }, f"Assistant should only request ask/update for KnowledgeManager, saw: {sorted(requested)}"
