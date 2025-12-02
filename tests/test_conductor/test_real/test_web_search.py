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
from unity.web_searcher.web_searcher import WebSearcher


@pytest.mark.asyncio
@_handle_project
async def test_ask_calls_searcher(monkeypatch):
    """
    Ensure Conductor.ask routes an external/general knowledge request to the
    real WebSearcher.ask exactly once, with no other manager tools executed.

    No stubbing of the WebSearcher loop: we use the real manager and ask a
    general-knowledge style question so routing picks WebSearcher once.
    """

    # Wire a SimulatedConductor to the real WebSearcher instance
    ws = WebSearcher()
    cond = SimulatedConductor(web_searcher=ws)

    # Ask a straightforward web-style question that should route to WebSearcher
    handle = await cond.ask(
        "What is the Eisenhower Matrix and when should it be used?",
        _return_reasoning_steps=True,
    )
    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)

    # Basic content check – answer should be non-empty
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # Ensure WebSearcher.ask was invoked (and nothing else from any manager)
    executed_list = tool_names_from_messages(messages, "WebSearcher")
    requested_list = assistant_requested_tool_names(messages, "WebSearcher")
    assert executed_list, "Expected at least one tool call"
    assert set(executed_list) == {
        "WebSearcher_ask",
    }, f"Only WebSearcher_ask should run; saw: {sorted(set(executed_list))}"
    assert (
        executed_list.count("WebSearcher_ask") == 1
    ), f"Expected exactly one WebSearcher_ask call, saw order: {executed_list}"
    assert set(requested_list) <= {
        "WebSearcher_ask",
    }, f"Assistant should request only WebSearcher_ask, saw: {sorted(set(requested_list))}"

    # Global exclusivity: verify no other manager tools ran
    all_tool_names = [
        str(m.get("name"))
        for m in messages
        if m.get("role") == "tool"
        and not str(m.get("name") or "").startswith("check_status_")
    ]
    assert all_tool_names, "Expected at least one tool call overall"
    assert all(
        n.startswith("WebSearcher_ask") or n.startswith("continue_WebSearcher_ask")
        for n in all_tool_names
    ), f"Unexpected tools executed: {sorted(set(all_tool_names))}"
    assert (
        len(all_tool_names) == 1
    ), f"Only one tool call expected; saw: {all_tool_names}"
