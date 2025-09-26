from __future__ import annotations

import asyncio

import pytest

from unity.conductor.simulated import SimulatedConductor
from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)


MANAGER = "SimulatedSkillManager"


SKILL_QUESTIONS: list[str] = [
    "What skills do you have for web browsing and extracting information?",
    "Are you familiar with using PowerPoint?",
    "Have you done much lead generation work before?",
]


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("question", SKILL_QUESTIONS)
@_handle_project
async def test_skill_questions_use_only_skill_manager_tool(question: str):
    cond = SimulatedConductor(
        description=(
            "Assistant that explains its skills via a skills catalogue; other managers exist but are not needed for these queries."
        ),
    )

    # Nudge routing deterministically to SkillManager by explicitly stating the surface
    handle = await cond.ask(
        question + " Use only your skill manager.",
        _return_reasoning_steps=True,
    )

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # The only executed tool must be SimulatedSkillManager.ask and it should run exactly once
    executed_list = tool_names_from_messages(messages, MANAGER)
    executed = set(executed_list)
    assert executed, "Expected at least one tool call to occur"
    assert executed == {
        "SimulatedSkillManager_ask",
    }, f"Only SimulatedSkillManager_ask should run, saw: {sorted(executed)}"
    assert (
        executed_list.count("SimulatedSkillManager_ask") == 1
    ), f"Expected exactly one SimulatedSkillManager_ask call, saw order: {executed_list}"

    # Additionally confirm that any assistant tool selection(s) referenced only that tool
    requested = set(assistant_requested_tool_names(messages, MANAGER))
    assert requested, "Assistant should have requested at least one tool"
    assert requested <= {
        "SimulatedSkillManager_ask",
    }, f"Assistant should request only SimulatedSkillManager_ask, saw: {sorted(requested)}"
