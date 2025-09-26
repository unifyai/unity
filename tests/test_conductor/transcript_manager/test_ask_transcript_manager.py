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


TRANSCRIPT_QUESTIONS: list[str] = [
    # High-level, transcript-first questions – should NOT route via ContactManager
    "What did David say last week?",
    "Show me the most recent message that mentions the Q3 budget.",
    "List messages from Alice in the last 24 hours.",
    "Find our last WhatsApp message with Sarah.",
]


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("question", TRANSCRIPT_QUESTIONS)
@_handle_project
async def test_transcript_questions_use_only_transcript_manager_tool(question: str):
    cond = SimulatedConductor(
        description=(
            "Assistant focused on transcripts; tasks, contacts, and knowledge exist but are not needed for these queries."
        ),
    )

    handle = await cond.ask(
        question,
        _return_reasoning_steps=True,
    )

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # The only executed tool must be SimulatedTranscriptManager.ask and it should run exactly once
    executed_list = _tool_names_from_messages(messages)
    executed = set(executed_list)
    assert executed, "Expected at least one tool call to occur"
    assert executed == {
        "SimulatedTranscriptManager_ask",
    }, f"Only SimulatedTranscriptManager_ask should run, saw: {sorted(executed)}"
    assert (
        executed_list.count("SimulatedTranscriptManager_ask") == 1
    ), f"Expected exactly one SimulatedTranscriptManager_ask call, saw order: {executed_list}"

    # Additionally confirm that any assistant tool selection(s) referenced only that tool
    requested = set(_assistant_requested_tool_names(messages))
    assert requested, "Assistant should have requested at least one tool"
    assert requested <= {
        "SimulatedTranscriptManager_ask",
    }, f"Assistant should request only SimulatedTranscriptManager_ask, saw: {sorted(requested)}"
