from __future__ import annotations

import asyncio
import pytest

pytestmark = pytest.mark.eval

from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)
from unity.conductor.simulated import SimulatedConductor
from unity.skill_manager.skill_manager import SkillManager
from unity.function_manager.function_manager import FunctionManager


@pytest.mark.asyncio
@_handle_project
async def test_ask_calls_manager():
    # Seed real FunctionManager so SkillManager has content to surface
    fm = FunctionManager()
    src1 = (
        "def add(a: int, b: int) -> int:\n"
        '    """Add two numbers"""\n'
        "    return a + b\n"
    )
    src2 = (
        "def price_total(p: float, tax: float) -> float:\n"
        '    """Return total price including tax"""\n'
        "    return p + tax\n"
    )
    fm.add_functions(implementations=[src1, src2])

    sk = SkillManager()
    cond = SimulatedConductor(skill_manager=sk)

    handle = await cond.ask(
        "List your available skills and include the underlying function names.",
        _return_reasoning_steps=True,
    )
    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)

    assert isinstance(answer, str) and answer.strip()

    # Ensure SkillManager.ask was invoked (and nothing else from any manager)
    executed_list = tool_names_from_messages(messages, "SkillManager")
    requested_list = assistant_requested_tool_names(messages, "SkillManager")
    assert executed_list, "Expected at least one tool call"
    assert set(executed_list) == {
        "SkillManager_ask",
    }, f"Only SkillManager_ask should run; saw: {sorted(set(executed_list))}"
    assert (
        executed_list.count("SkillManager_ask") == 1
    ), f"Expected exactly one SkillManager_ask call, saw order: {executed_list}"
    assert set(requested_list) <= {
        "SkillManager_ask",
    }, f"Assistant should request only SkillManager_ask, saw: {sorted(set(requested_list))}"

    # Global exclusivity: verify no other manager tools ran
    all_tool_names = [
        str(m.get("name"))
        for m in messages
        if m.get("role") == "tool"
        and not str(m.get("name") or "").startswith("check_status_")
    ]
    assert all_tool_names, "Expected at least one tool call overall"
    assert all(
        n.startswith("SkillManager_ask") or n.startswith("continue_SkillManager_ask")
        for n in all_tool_names
    ), f"Unexpected tools executed: {sorted(set(all_tool_names))}"
    assert (
        len(all_tool_names) == 1
    ), f"Only one tool call expected; saw: {all_tool_names}"


@pytest.mark.asyncio
@_handle_project
async def test_request_routes_to_ask():
    # Seed functions so a request phrased as a mutation still routes read-only to SkillManager.ask
    fm = FunctionManager()
    fm.add_functions(
        implementations=[
            (
                "def send_email(to: str, subject: str, body: str) -> None:\n"
                '    """Send an email"""\n'
                "    return None\n"
            ),
        ],
    )

    sk = SkillManager()
    cond = SimulatedConductor(skill_manager=sk)

    # Although this uses request(), SkillManager has no update/write surface; Conductor should route to ask
    req = "Tell me which skill can send emails and show its signature."
    handle = await cond.request(req, _return_reasoning_steps=True)
    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)

    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    executed_list = tool_names_from_messages(messages, "SkillManager")
    requested_list = assistant_requested_tool_names(messages, "SkillManager")
    assert executed_list, "Expected at least one tool call"
    # Must not call an update/execute on SkillManager; only ask should appear
    assert set(executed_list) <= {
        "SkillManager_ask",
    }, f"Only SkillManager_ask should run, saw: {sorted(set(executed_list))}"
    assert "SkillManager_update" not in set(
        executed_list,
    ), f"SkillManager_update must not run, saw: {sorted(set(executed_list))}"
    assert set(requested_list) <= {
        "SkillManager_ask",
    }, f"Assistant should request only SkillManager_ask, saw: {sorted(set(requested_list))}"

    # Global exclusivity: verify no other manager tools ran
    all_tool_names = [
        str(m.get("name"))
        for m in messages
        if m.get("role") == "tool"
        and not str(m.get("name") or "").startswith("check_status_")
    ]
    assert all_tool_names, "Expected at least one tool call overall"
    assert all(
        n.startswith("SkillManager_ask") or n.startswith("continue_SkillManager_ask")
        for n in all_tool_names
    ), f"Unexpected tools executed: {sorted(set(all_tool_names))}"
    assert (
        len(all_tool_names) >= 1
    ), f"At least one tool call expected; saw: {all_tool_names}"
