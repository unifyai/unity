"""Tests for TranscriptManager response_format parameter."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field
from typing import List

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.simulated import SimulatedTranscriptManager
from tests.helpers import _handle_project

# ────────────────────────────────────────────────────────────────────────────
# Response format schemas
# ────────────────────────────────────────────────────────────────────────────


class TranscriptSummary(BaseModel):
    """Structured summary of transcript search results."""

    total_messages: int = Field(..., description="Total number of messages found")
    participants: List[str] = Field(
        ...,
        description="Names or IDs of participants in the exchanges",
    )
    key_topics: List[str] = Field(
        ...,
        description="Main topics discussed in the messages",
    )
    summary: str = Field(..., description="Brief natural language summary")


# ────────────────────────────────────────────────────────────────────────────
# Simulated TranscriptManager tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_simulated_ask_response_format():
    """Simulated TranscriptManager.ask should return structured output when response_format is provided."""
    tm = SimulatedTranscriptManager("Demo transcript DB with several conversations.")

    handle = await tm.ask(
        "What topics were discussed in recent emails?",
        response_format=TranscriptSummary,
    )
    result = await handle.result()

    assert isinstance(result, TranscriptSummary)
    assert isinstance(result.total_messages, int)
    assert result.total_messages >= 0
    assert isinstance(result.participants, list)
    assert isinstance(result.key_topics, list)
    assert result.summary.strip(), "Summary should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# Real TranscriptManager tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_real_ask_response_format(
    tm_manager_scenario: tuple[TranscriptManager, dict],
):
    """Real TranscriptManager.ask should return structured output when response_format is provided."""
    tm, _ = tm_manager_scenario

    handle = await tm.ask(
        "Summarize all phone calls and list who was involved",
        response_format=TranscriptSummary,
    )
    result = await handle.result()

    assert isinstance(result, TranscriptSummary)
    # We know from the fixture there are some messages
    assert isinstance(result.total_messages, int)
    assert isinstance(result.participants, list)
    assert isinstance(result.key_topics, list)
    assert result.summary.strip(), "Summary should be non-empty"
