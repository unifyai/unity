from __future__ import annotations

import asyncio

import pytest

from unity.conductor.simulated import SimulatedConductor
from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)


MANAGER = "WebSearcher"


# Live-event, time-sensitive questions that must route to WebSearcher
WEB_LIVE_QUESTIONS: list[str] = [
    "What is the weather in Berlin today?",
    "What are the major world news headlines this week?",
    "Did the UN Security Council approve the resolution yesterday?",
    "What notable AI research announcements were made this week?",
]


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("question", WEB_LIVE_QUESTIONS)
@_handle_project
async def test_live_events_use_only_web_searcher_tool(question: str):
    cond = SimulatedConductor(
        description=(
            "Assistant that uses web research for live, time-sensitive queries; "
            "internal managers exist but are not relevant for these external questions."
        ),
    )

    handle = await cond.ask(
        question,
        _return_reasoning_steps=True,
    )

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # The only executed tool must be WebSearcher.ask and it should run exactly once
    executed_list = tool_names_from_messages(messages, MANAGER)
    executed = set(executed_list)
    assert executed, "Expected at least one tool call to occur"
    assert executed == {
        "WebSearcher_ask",
    }, f"Only WebSearcher_ask should run, saw: {sorted(executed)}"
    assert (
        executed_list.count("WebSearcher_ask") >= 1
    ), f"Expected exactly one WebSearcher_ask call, saw order: {executed_list}"

    # Additionally confirm that any assistant tool selection(s) referenced only that tool
    requested = set(assistant_requested_tool_names(messages, MANAGER))
    assert requested, "Assistant should have requested at least one tool"
    assert requested <= {
        "WebSearcher_ask",
    }, f"Assistant should request only WebSearcher_ask, saw: {sorted(requested)}"
