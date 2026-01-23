"""Tests for WebSearcher response_format parameter."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field
from typing import List, Optional

from unity.web_searcher.web_searcher import WebSearcher
from unity.web_searcher.simulated import SimulatedWebSearcher
from tests.helpers import _handle_project

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


class WebsiteUpdateResult(BaseModel):
    """Structured result after a website update operation."""

    success: bool = Field(..., description="Whether the update was successful")
    website_host: Optional[str] = Field(
        None,
        description="Host of the website that was modified",
    )
    action_taken: str = Field(..., description="Description of what was done")


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

    # Should be valid JSON conforming to the schema
    parsed = SearchResult.model_validate_json(result)

    assert isinstance(parsed.sources_count, int)
    assert parsed.sources_count >= 0
    assert isinstance(parsed.key_findings, list)
    assert isinstance(parsed.source_urls, list)
    assert parsed.summary.strip(), "Summary should be non-empty"


@pytest.mark.asyncio
@_handle_project
async def test_simulated_update_response_format():
    """Simulated WebSearcher.update should return structured output when response_format is provided."""
    ws = SimulatedWebSearcher("Demo web searcher for testing updates.")

    handle = await ws.update(
        "Add a new website entry for techcrunch.com with tags ['tech', 'news']",
        response_format=WebsiteUpdateResult,
    )
    result = await handle.result()

    parsed = WebsiteUpdateResult.model_validate_json(result)

    assert isinstance(parsed.success, bool)
    assert parsed.action_taken.strip(), "Action description should be non-empty"


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

    parsed = SearchResult.model_validate_json(result)

    assert isinstance(parsed.sources_count, int)
    assert isinstance(parsed.key_findings, list)
    assert parsed.summary.strip(), "Summary should be non-empty"


@pytest.mark.asyncio
@_handle_project
async def test_real_update_response_format():
    """Real WebSearcher.update should return structured output when response_format is provided."""
    ws = WebSearcher()

    handle = await ws.update(
        "Create a website entry for host=example.org with name='Example Site' and tags=['demo']",
        response_format=WebsiteUpdateResult,
    )
    result = await handle.result()

    parsed = WebsiteUpdateResult.model_validate_json(result)

    assert isinstance(parsed.success, bool)
    assert parsed.action_taken.strip(), "Action description should be non-empty"
