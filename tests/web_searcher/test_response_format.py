"""Tests for WebSearcher response_format parameter."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field
from typing import List

from unity.web_searcher.web_searcher import WebSearcher
from unity.web_searcher.simulated import SimulatedWebSearcher
from tests.helpers import _handle_project

pytestmark = pytest.mark.llm_call

# ────────────────────────────────────────────────────────────────────────────
# Response format schemas
# ────────────────────────────────────────────────────────────────────────────


class SearchResult(BaseModel):
    """Structured result from a web search query."""

    sources_count: int = Field(..., description="Number of sources consulted")
    key_findings: List[str] = Field(
        ...,
        description="List of key findings from the search",
    )
    source_urls: List[str] = Field(
        default_factory=list,
        description="URLs of sources consulted",
    )
    summary: str = Field(..., description="Brief natural language summary")


# ────────────────────────────────────────────────────────────────────────────
# Simulated WebSearcher tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_simulated_ask_response_format():
    """Simulated WebSearcher.ask should return structured output when response_format is provided."""
    ws = SimulatedWebSearcher("Demo web searcher with technology news sources.")

    handle = await ws.ask(
        "Search for recent developments in AI and list the key findings",
        response_format=SearchResult,
    )
    result = await handle.result()

    assert isinstance(result, SearchResult)
    assert isinstance(result.sources_count, int)
    assert result.sources_count >= 0
    assert isinstance(result.key_findings, list)
    assert isinstance(result.source_urls, list)
    assert result.summary.strip(), "Summary should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# Real WebSearcher tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_real_ask_response_format():
    """Real WebSearcher.ask should return structured output when response_format is provided."""
    ws = WebSearcher()

    handle = await ws.ask(
        "Search for the latest Python release and summarize key findings",
        response_format=SearchResult,
    )
    result = await handle.result()

    assert isinstance(result, SearchResult)
    assert isinstance(result.sources_count, int)
    assert isinstance(result.key_findings, list)
    assert result.summary.strip(), "Summary should be non-empty"
