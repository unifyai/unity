from __future__ import annotations


import pytest
import asyncio
from unity.transcript_manager.transcript_manager import TranscriptManager
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_summarize_uses_parent_context(tm_scenario: TranscriptManager):
    """
    Guidance references an *earlier* agreement only present in parent ctx.
    The summary must therefore obey that agreement without asking.
    """
    tm, _ = tm_scenario
    parent_ctx = [
        {"role": "user", "content": "When we mention Carlos, call him 'Alpha'."},
        {"role": "assistant", "content": "Understood – Carlos → Alpha."},
    ]

    summary = await tm.summarize(
        exchange_ids=[2],  # Carlos ⇆ Dan e-mail seeded by ScenarioBuilder
        guidance="Use the abbreviation we discussed earlier for Carlos.",
        parent_chat_context=parent_ctx,
    )

    assert "Alpha" in summary, "Parent-context instruction not applied."
    assert "Carlos" not in summary, "Original name leaked despite instruction."


@pytest.mark.asyncio
@_handle_project
async def test_summarize_requests_clarification(tm_scenario: TranscriptManager):
    """
    Ambiguous guidance + no parent context → summarize should ask a
    clarification, then incorporate the answer.
    """
    tm, _ = tm_scenario

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    handle = asyncio.create_task(
        tm.summarize(
            exchange_ids=[2],
            guidance="Include Carlos' surname in the summary.",
            clarification_up_q=up_q,
            clarification_down_q=down_q,
        ),
    )

    # 1) Wait for the clarification request
    question = await asyncio.wait_for(up_q.get(), timeout=30)
    assert "surname" in question.lower()

    # 2) Provide the missing information
    await down_q.put("Carlos' surname is Rodriguez.")

    # 3) Assertion on final summary
    summary = await asyncio.wait_for(handle, timeout=30)
    assert "Rodriguez" in summary, "Clarification answer not respected."
