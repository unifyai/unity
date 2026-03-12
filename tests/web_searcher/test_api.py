import pytest

pytestmark = pytest.mark.eval

from tests.helpers import _handle_project
from unity.web_searcher.web_searcher import WebSearcher


@pytest.mark.asyncio
@_handle_project
async def test_ask_invokes_search_tool():
    """
    Use real WebSearcher.ask, Tavily, and async tool loop. Verify that at least
    one `search` tool call was made by inspecting the conversation history.
    """
    ws = WebSearcher()
    handle = await ws.ask(
        "Find the latest headline about the Python programming language and cite a source.",
    )
    final = await handle.result()

    assert isinstance(final, str) and final.strip() != ""

    history = handle.get_history()
    assert history is not None


@pytest.mark.asyncio
@_handle_project
async def test_ask_with_reasoning_steps_wrapper():
    """
    With _return_reasoning_steps=True, the real handle.result() should return
    (answer, messages) where messages is the model transcript.
    """
    ws = WebSearcher()
    handle = await ws.ask(
        "Summarize a recent technology headline with one sentence and cite a source.",
        _return_reasoning_steps=True,
    )
    answer, messages = await handle.result()

    assert isinstance(answer, str) and answer.strip() != ""
    assert isinstance(messages, list) and len(messages) > 0


@pytest.mark.asyncio
@_handle_project
async def test_ask_with_parent_context():
    """
    Provide a parent chat context and ensure the call succeeds and history is present.
    """
    ws = WebSearcher()
    parent_ctx = [
        {"role": "user", "content": "We were reviewing last week's AI headlines."},
        {
            "role": "assistant",
            "content": "Key items included new model releases and safety reports.",
        },
    ]
    handle = await ws.ask(
        "Briefly summarize one recent AI news item and cite a source.",
        _parent_chat_context=parent_ctx,
    )
    res = await handle.result()

    assert isinstance(res, str) and res.strip() != ""
    history = handle.get_history()
    assert history is not None


@pytest.mark.asyncio
@_handle_project
async def test_ask_with_response_format():
    """Verify structured output by providing a Pydantic response_format."""
    from pydantic import BaseModel, Field

    class SimpleSummary(BaseModel):
        summary: str = Field(..., description="One-sentence summary of the finding")

    ws = WebSearcher()
    handle = await ws.ask(
        "Provide a one-sentence summary of a recent technology news item; output JSON matching the provided schema.",
        response_format=SimpleSummary,
    )
    result = await handle.result()

    assert isinstance(result, SimpleSummary)
    assert isinstance(result.summary, str) and result.summary.strip() != ""


def test_clear_initialises_and_resets_caches():
    """
    Ensure WebSearcher.clear flushes internal caches and keeps them provisioned.
    """
    ws = WebSearcher()

    # Sanity: caches exist after construction
    assert hasattr(ws, "_last_results")
    assert hasattr(ws, "_last_extractions")
    assert hasattr(ws, "_last_crawls")
    assert hasattr(ws, "_last_maps")

    # Seed caches with dummy content
    ws._last_results = [{"k": "v"}]
    ws._last_extractions = {"u": "x"}
    ws._last_crawls = {"a": 1}
    ws._last_maps = {"b": 2}

    # Execute clear
    ws.clear()

    # After clear: caches should exist and be reset
    assert hasattr(ws, "_last_results") and ws._last_results == []
    assert hasattr(ws, "_last_extractions") and ws._last_extractions == {}
    assert hasattr(ws, "_last_crawls") and ws._last_crawls == {}
    assert hasattr(ws, "_last_maps") and ws._last_maps == {}

    # Tools should still be provisioned (now just the 4 Tavily tools)
    assert {"search", "extract", "crawl", "map"} == set(ws.get_tools("ask").keys())


# ---------------------------------------------------------------------------
# Regression: same-day date range
# ---------------------------------------------------------------------------


def test_search_same_day_date_range_widened():
    """_search should widen start_date when start_date == end_date.

    Tavily rejects identical dates with BadRequestError. The fix pushes
    start_date back by one day so the request succeeds.
    """
    from unittest.mock import MagicMock

    ws = WebSearcher()
    ws.tavily_client = MagicMock()
    ws.tavily_client.search.return_value = {
        "answer": "ok",
        "results": [],
        "images": [],
    }

    ws._search("test", start_date="2026-03-12", end_date="2026-03-12")

    call_kw = ws.tavily_client.search.call_args.kwargs
    assert (
        call_kw["start_date"] == "2026-03-11"
    ), f"start_date should be widened to the day before, got {call_kw['start_date']}"
    assert call_kw["end_date"] == "2026-03-12"


def test_search_different_dates_unchanged():
    """_search should pass through dates untouched when they differ."""
    from unittest.mock import MagicMock

    ws = WebSearcher()
    ws.tavily_client = MagicMock()
    ws.tavily_client.search.return_value = {
        "answer": "ok",
        "results": [],
        "images": [],
    }

    ws._search("test", start_date="2026-03-10", end_date="2026-03-12")

    call_kw = ws.tavily_client.search.call_args.kwargs
    assert call_kw["start_date"] == "2026-03-10"
    assert call_kw["end_date"] == "2026-03-12"


def test_search_no_dates_unchanged():
    """_search should leave dates as None when not provided."""
    from unittest.mock import MagicMock

    ws = WebSearcher()
    ws.tavily_client = MagicMock()
    ws.tavily_client.search.return_value = {
        "answer": "ok",
        "results": [],
        "images": [],
    }

    ws._search("test")

    call_kw = ws.tavily_client.search.call_args.kwargs
    assert call_kw["start_date"] is None
    assert call_kw["end_date"] is None
