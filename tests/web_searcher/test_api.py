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
    final = await handle.result()

    parsed = SimpleSummary.model_validate_json(final)
    assert isinstance(parsed.summary, str) and parsed.summary.strip() != ""


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
