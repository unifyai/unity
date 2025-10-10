from dotenv import load_dotenv

load_dotenv(override=True)

import asyncio
import pytest
from typing import Any, Dict, List

from tests.helpers import _handle_project
from unity.web_searcher.web_searcher import WebSearcher
from unity.common.llm_helpers import inject_broader_context


@pytest.mark.asyncio
@_handle_project
async def test_ask_tool_selection_real_loop(monkeypatch):
    """
    Use a real LLM client and real tool loop, but replace the WebSearcher tools with
    lightweight dummies and a deterministic system prompt to force tool selection.
    Ensures the LLM uses search, extract, crawl, and map exactly once.
    """

    # Strong instruction to the model to call each tool once, then stop.
    def forced_prompt_builder(*, tools: Dict[str, Any]) -> str:
        return (
            "You are an automated test assistant.\n"
            "1) Call `search` with query='weather information' exactly once.\n"
            "2) Then call `extract` with urls='https://finance.example.com/news' exactly once.\n"
            "3) Then call `crawl` with start_url='https://finance.example.com/news' exactly once.\n"
            "4) Then call `map` with url='https://finance.example.com/news' exactly once.\n"
            "After completing step 4, respond with the single token 'ws_tools_ok'.\n"
            "Do not call any extra tools."
        )

    # Patch the prompt builder to our deterministic instruction.
    monkeypatch.setattr(
        "unity.web_searcher.prompt_builders.build_ask_prompt",
        forced_prompt_builder,
        raising=True,
    )

    # Create the WebSearch instance (Tavily client won't be used because we override tools).
    ws = WebSearcher()

    # Track calls for each tool.
    calls = {"search": 0, "extract": 0, "crawl": 0, "map": 0}

    # Provide lightweight dummy tool implementations with helpful signatures.
    async def dummy_search(
        query: str,
        *,
        max_results: int = 5,
        start_date: str | None = None,
        end_date: str | None = None,
        include_images: bool = False,
    ) -> str:
        calls["search"] += 1
        return f"searched:{query}"

    async def dummy_extract(
        urls: str | List[str],
        *,
        include_images: bool = False,
    ) -> Dict[str, Any]:
        calls["extract"] += 1
        return {"results": [{"url": urls, "content": "ok"}], "failed_results": []}

    async def dummy_crawl(
        start_url: str,
        *,
        instructions: str | None = None,
        max_depth: int | None = None,
        max_breadth: int | None = None,
        limit: int | None = None,
        include_images: bool | None = None,
    ) -> Dict[str, Any]:
        calls["crawl"] += 1
        return {"base_url": start_url, "results": []}

    async def dummy_map(
        url: str,
        *,
        instructions: str | None = None,
        max_depth: int | None = None,
        max_breadth: int | None = None,
        limit: int | None = None,
        include_images: bool | None = None,
    ) -> Dict[str, Any]:
        calls["map"] += 1
        return {"base_url": None, "results": []}

    # Override the tools exposed to the tool loop to our dummies.
    ws._ask_tools = {
        "search": dummy_search,
        "extract": dummy_extract,
        "crawl": dummy_crawl,
        "map": dummy_map,
    }

    handle = await ws.ask("begin")
    final = await asyncio.wait_for(handle.result(), timeout=180)

    assert "ws_tools_ok" in str(final).strip().lower()
    assert calls["search"] == 1
    assert calls["extract"] == 1
    assert calls["crawl"] == 1
    assert calls["map"] == 1


@pytest.mark.asyncio
async def test_ask_with_reasoning_steps_wrapper(monkeypatch):
    """
    Verify that when _return_reasoning_steps=True, the handle.result() returns
    (answer, messages) where messages come from the client.
    """
    from unittest.mock import AsyncMock, MagicMock

    # Mock the async unify client
    mock_client = MagicMock()
    mock_client.set_system_message = MagicMock()
    mock_client.messages = [{"role": "assistant", "content": "Test reasoning"}]

    monkeypatch.setattr(
        "unity.web_searcher.web_searcher.unify.AsyncUnify",
        lambda *a, **kw: mock_client,
        raising=True,
    )

    # Mock the tool loop to return a handle with a deterministic result
    mock_handle = MagicMock()
    mock_handle.result = AsyncMock(return_value="Answer")

    def fake_loop(client, message, tools, **kwargs):
        return mock_handle

    monkeypatch.setattr(
        "unity.web_searcher.web_searcher.start_async_tool_loop",
        fake_loop,
        raising=True,
    )

    ws = WebSearcher()
    handle = await ws.ask("What is this?", _return_reasoning_steps=True)
    answer, messages = await handle.result()

    mock_client.set_system_message.assert_called_once()
    assert answer == "Answer"
    assert messages == mock_client.messages


@pytest.mark.asyncio
async def test_ask_forwards_parent_context_and_preprocess(monkeypatch):
    """
    Ensure parent_chat_context is forwarded to the loop and preprocess is inject_broader_context.
    Also validate provided tool names.
    """
    from unittest.mock import MagicMock

    # Minimal client stub
    mock_client = MagicMock()
    mock_client.set_system_message = MagicMock()
    monkeypatch.setattr(
        "unity.web_searcher.web_searcher.unify.AsyncUnify",
        lambda *a, **kw: mock_client,
        raising=True,
    )

    captured: Dict[str, Any] = {}

    class DummyHandle:
        async def result(self):
            return "ok"

    def capture_loop(client, message, tools, **kwargs):
        captured.update(kwargs)
        # Keep a snapshot of tool names passed in
        captured["tool_names"] = set(tools.keys())
        return DummyHandle()

    monkeypatch.setattr(
        "unity.web_searcher.web_searcher.start_async_tool_loop",
        capture_loop,
        raising=True,
    )

    ws = WebSearcher()
    parent_ctx = [
        {"role": "user", "content": "Context A."},
        {"role": "assistant", "content": "Context B."},
    ]
    handle = await ws.ask("hello", parent_chat_context=parent_ctx)
    res = await handle.result()

    assert res == "ok"
    assert captured.get("parent_chat_context") is parent_ctx
    assert captured.get("preprocess_msgs") is inject_broader_context
    assert {"search", "extract", "crawl", "map"}.issubset(
        captured.get("tool_names", set()),
    )


@pytest.mark.unit
def test_web_searcher_clear_initialises_and_resets_caches():
    """
    Ensure WebSearcher.clear flushes internal caches and keeps them provisioned.
    Mirrors the ContactManager clear test style by verifying state before/after.
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

    # Tools should still be provisioned
    assert {"search", "extract", "crawl", "map"}.issubset(set(ws._ask_tools.keys()))


@pytest.mark.unit
def test_simulated_web_searcher_clear_reinitialises():
    """
    Ensure SimulatedWebSearcher.clear re-runs the constructor (fresh stateful LLM
    and tools mapping stays provisioned).
    """
    from unity.web_searcher.simulated import SimulatedWebSearcher

    sim = SimulatedWebSearcher()
    old_llm = getattr(sim, "_llm", None)
    assert old_llm is not None
    assert isinstance(sim._ask_tools, dict) and sim._ask_tools

    sim.clear()

    # After clear, llm handle should be replaced and tools still present
    assert getattr(sim, "_llm", None) is not None and sim._llm is not old_llm
    assert isinstance(sim._ask_tools, dict) and sim._ask_tools
