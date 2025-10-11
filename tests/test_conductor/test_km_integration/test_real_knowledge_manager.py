from __future__ import annotations

import asyncio

import pytest

from unity.conductor.simulated import SimulatedConductor
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.knowledge_manager.types import ColumnType

from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)


# ---------------------------------------------------------------------------
#  Real Conductor → KnowledgeManager.ask
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_real_conductor_knowledge_ask_calls_knowledge_manager():
    # Seed a simple knowledge table with a fact about office hours
    km = KnowledgeManager()
    km._create_table(
        name="Policies",
        description="Company policies and procedures",
        columns={
            "title": ColumnType.str,
            "content": ColumnType.str,
        },
    )
    km._add_rows(
        table="Policies",
        rows=[
            {
                "title": "Office Hours",
                "content": "Office hours are 9–5 PT.",
            },
        ],
    )

    # SimulatedConductor wired to the real KnowledgeManager instance
    cond = SimulatedConductor(knowledge_manager=km)

    handle = await cond.ask(
        "What are our office hours?",
        _return_reasoning_steps=True,
    )
    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)

    # Basic content check – answer should be non-empty
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # Ensure KnowledgeManager.ask was invoked (and nothing else from any manager)
    executed_list = tool_names_from_messages(messages, "KnowledgeManager")
    requested_list = assistant_requested_tool_names(messages, "KnowledgeManager")
    assert executed_list, "Expected at least one tool call"
    assert set(executed_list) == {
        "KnowledgeManager_ask",
    }, f"Only KnowledgeManager_ask should run; saw: {sorted(set(executed_list))}"
    assert (
        executed_list.count("KnowledgeManager_ask") == 1
    ), f"Expected exactly one KnowledgeManager_ask call, saw order: {executed_list}"
    assert set(requested_list) <= {
        "KnowledgeManager_ask",
    }, f"Assistant should request only KnowledgeManager_ask, saw: {sorted(set(requested_list))}"

    # Global exclusivity: verify no other manager tools ran
    all_tool_names = [
        str(m.get("name"))
        for m in messages
        if m.get("role") == "tool"
        and not str(m.get("name") or "").startswith("check_status_")
    ]
    assert all_tool_names, "Expected at least one tool call overall"
    assert all(
        n.startswith("KnowledgeManager_ask")
        or n.startswith("continue_KnowledgeManager_ask")
        for n in all_tool_names
    ), f"Unexpected tools executed: {sorted(set(all_tool_names))}"
    assert (
        len(all_tool_names) == 1
    ), f"Only one tool call expected; saw: {all_tool_names}"


# ---------------------------------------------------------------------------
#  Real Conductor → KnowledgeManager.update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_real_conductor_knowledge_update_calls_knowledge_manager():
    km = KnowledgeManager()

    cond = SimulatedConductor(knowledge_manager=km)

    request_text = "Store: Office hours are 9–5 PT."
    handle = await cond.request(request_text, _return_reasoning_steps=True)
    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)

    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # Ensure only KnowledgeManager.update was invoked (not ask) and nothing else
    executed_list = tool_names_from_messages(messages, "KnowledgeManager")
    requested_list = assistant_requested_tool_names(messages, "KnowledgeManager")
    assert executed_list, "Expected at least one tool call"
    assert (
        executed_list[0] == "KnowledgeManager_update"
    ), f"First call must be KnowledgeManager_update, saw order: {executed_list}"
    assert set(executed_list) <= {
        "KnowledgeManager_update",
    }, f"Only KnowledgeManager_update should run, saw: {sorted(set(executed_list))}"
    assert "KnowledgeManager_ask" not in set(
        executed_list,
    ), f"KnowledgeManager_ask must not run, saw: {sorted(set(executed_list))}"
    assert set(requested_list) <= {
        "KnowledgeManager_update",
    }, f"Assistant should request only KnowledgeManager_update, saw: {sorted(set(requested_list))}"

    # Global exclusivity: verify no other manager tools ran
    all_tool_names = [
        str(m.get("name"))
        for m in messages
        if m.get("role") == "tool"
        and not str(m.get("name") or "").startswith("check_status_")
    ]
    assert all_tool_names, "Expected at least one tool call overall"
    assert all(
        n.startswith("KnowledgeManager_update")
        or n.startswith("continue_KnowledgeManager_update")
        for n in all_tool_names
    ), f"Unexpected tools executed: {sorted(set(all_tool_names))}"
    assert (
        len(all_tool_names) == 1
    ), f"Only one tool call expected; saw: {all_tool_names}"
