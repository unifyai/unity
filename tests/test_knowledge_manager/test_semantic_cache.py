from __future__ import annotations

import pytest
import unify

from typing import List, Optional
from dataclasses import dataclass, field
from unittest.mock import patch

from tests.helpers import _handle_project
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.settings import SETTINGS
from unity.common._async_tool.semantic_cache import (
    _Config,
    SemanticCacheResult,
    search_semantic_cache,
)
from unity.common._async_tool import semantic_cache as sc

# Tests semantic cache behavior which depends on LLM responses
pytestmark = pytest.mark.eval


@dataclass
class CacheCall:
    """Record of a single call to search_semantic_cache."""

    user_message: str
    namespace: str
    result: Optional[SemanticCacheResult]

    @property
    def was_hit(self) -> bool:
        return self.result is not None


@dataclass
class CacheSpy:
    """
    Spy that wraps search_semantic_cache to capture all calls and results.
    This enables direct verification of cache behavior rather than inferring
    from indirect side effects like tool call counts.
    """

    calls: List[CacheCall] = field(default_factory=list)
    _original_fn: callable = None

    def wrap(self, user_message: str, namespace: str) -> Optional[SemanticCacheResult]:
        result = self._original_fn(user_message, namespace)
        self.calls.append(CacheCall(user_message, namespace, result))
        return result

    @property
    def hit_count(self) -> int:
        return sum(1 for c in self.calls if c.was_hit)

    @property
    def miss_count(self) -> int:
        return sum(1 for c in self.calls if not c.was_hit)

    def get_last_call(self) -> Optional[CacheCall]:
        return self.calls[-1] if self.calls else None

    def clear(self):
        self.calls.clear()


@pytest.fixture
def cache_spy(monkeypatch):
    """
    Fixture that provides a spy around search_semantic_cache.
    Use this to directly verify cache hits/misses.
    """
    spy = CacheSpy()
    spy._original_fn = search_semantic_cache

    monkeypatch.setattr(
        "unity.common._async_tool.semantic_cache.search_semantic_cache",
        spy.wrap,
    )
    return spy


@pytest.fixture(autouse=True)
def _patch_semantic_cache_config(monkeypatch):
    """
    Configure semantic cache for testing.
    Uses a moderate threshold (0.5) that should match semantically similar
    queries while rejecting truly different ones.
    """

    class _DynamicConfig(_Config):
        threshold = 0.5

        @property
        def context(self):
            return f"{unify.get_active_context()['write']}/SemanticCache"

    monkeypatch.setattr(
        "unity.common._async_tool.semantic_cache._CONFIG",
        _DynamicConfig(),
    )


def _count_tool_calls_in_reasoning(reasoning_steps) -> tuple[int, set[str]]:
    """Count the number of tool calls in the reasoning steps, excluding semantic_search."""
    tool_call_count = 0
    tool_names = set()
    for step in reasoning_steps:
        if step.get("role") == "tool":
            if step.get("name") == "semantic_search":
                continue
            tool_call_count += 1
            tool_names.add(step.get("name"))
    return tool_call_count, tool_names


def _setup_single_city_scenario():
    """Set up a KnowledgeManager with one city."""
    km = KnowledgeManager()
    km._create_table(name="MyTable", columns={"city": "str", "customers": "int"})
    km._add_rows(
        table="MyTable",
        rows=[
            {"city": "New York", "customers": 10},
        ],
    )
    return km


def _setup_multi_city_scenario():
    """Set up a KnowledgeManager with multiple cities."""
    km = KnowledgeManager()
    km._create_table(name="MyTable", columns={"city": "str", "customers": "int"})
    km._add_rows(
        table="MyTable",
        rows=[
            {"city": "New York", "customers": 10},
            {"city": "Los Angeles", "customers": 512},
        ],
    )
    return km


# =============================================================================
# Test 1: Exact Match - Same query twice should hit cache
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_exact_match_cache_hit(cache_spy: CacheSpy):
    """
    Test that an identical query on the second run hits the semantic cache.

    Scenario:
    1. First query: "What is the number of customers in New York?" → cache MISS
    2. Second query: identical → cache HIT, cached trajectory reused

    Verification:
    - Direct: cache_spy confirms cache hit on second call
    - Indirect: second run has zero/minimal tool calls
    """
    km = _setup_single_city_scenario()
    query = (
        "What is the number of customers in New York? "
        "Answer with the number of customers only."
    )

    with patch.object(SETTINGS, "UNITY_SEMANTIC_CACHE", True):
        # First run - should miss cache and call tools
        handle_1 = await km.ask(query, _return_reasoning_steps=True)
        answer_1, reasoning_1 = await handle_1.result()
        tool_calls_1, _ = _count_tool_calls_in_reasoning(reasoning_1)

        # Verify first run behavior
        assert tool_calls_1 > 0, "First run should make tool calls to fetch data"
        assert "10" in answer_1, f"Expected '10' in answer, got: {answer_1}"

        # Check cache spy: first call should be a miss
        first_cache_call = cache_spy.get_last_call()
        assert first_cache_call is not None, "Cache should have been searched"
        assert not first_cache_call.was_hit, "First query should be a cache MISS"

        sc._SEMANTIC_CACHE_SAVER.wait()
        cache_spy.clear()

        # Second run - should hit cache
        handle_2 = await km.ask(query, _return_reasoning_steps=True)
        answer_2, reasoning_2 = await handle_2.result()
        tool_calls_2, tools_called = _count_tool_calls_in_reasoning(reasoning_2)

        # Verify answer is correct
        assert "10" in answer_2, f"Expected '10' in answer, got: {answer_2}"

        # Direct verification: cache was hit
        second_cache_call = cache_spy.get_last_call()
        assert second_cache_call is not None, "Cache should have been searched"
        assert second_cache_call.was_hit, (
            f"Second (identical) query should be a cache HIT. " f"Query: '{query}'"
        )

        # Indirect verification: minimal tool calls
        assert tool_calls_2 <= 1, (
            f"Cache hit should result in minimal tool calls. "
            f"First: {tool_calls_1}, Second: {tool_calls_2}, Tools: {tools_called}"
        )


# =============================================================================
# Test 2: Semantically Equivalent - Same data, different phrasing
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_semantically_equivalent_queries(cache_spy: CacheSpy):
    """
    Test that semantically equivalent queries about the SAME data hit the cache.

    Scenario:
    1. "What is the number of customers in New York?" → cache MISS
    2. "How many customers are there in New York?" → cache HIT (same data, different phrasing)

    This tests true semantic similarity: the queries are structurally different
    but ask for the same information.
    """
    km = _setup_single_city_scenario()

    with patch.object(SETTINGS, "UNITY_SEMANTIC_CACHE", True):
        # First query
        question_1 = (
            "What is the number of customers in New York? "
            "Answer with the number of customers only."
        )
        handle_1 = await km.ask(question_1, _return_reasoning_steps=True)
        answer_1, reasoning_1 = await handle_1.result()

        assert "10" in answer_1
        first_call = cache_spy.get_last_call()
        assert not first_call.was_hit, "First query should be a cache MISS"

        sc._SEMANTIC_CACHE_SAVER.wait()
        cache_spy.clear()

        # Second query - different phrasing, same data
        question_2 = (
            "How many customers are there in New York? "
            "Answer with the number of customers only."
        )
        handle_2 = await km.ask(question_2, _return_reasoning_steps=True)
        answer_2, reasoning_2 = await handle_2.result()

        # Direct verification: should hit cache
        second_call = cache_spy.get_last_call()
        assert second_call.was_hit, (
            f"Semantically equivalent query should hit cache. "
            f"Q1: '{question_1}', Q2: '{question_2}'"
        )

        # Answer should still be correct
        assert "10" in answer_2

        # Minimal tool calls due to cache
        tool_calls_2, tools_called = _count_tool_calls_in_reasoning(reasoning_2)
        assert tool_calls_2 <= 1, (
            f"Cache hit should result in minimal tool calls. "
            f"Got {tool_calls_2} calls. Tools: {tools_called}"
        )


# =============================================================================
# Test 3: Superset Query - First query contains data for second
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_superset_query_reuse(cache_spy: CacheSpy):
    """
    Test that a broader query's cached data can answer a narrower follow-up.

    Scenario:
    1. "How many customers in all cities?" → fetches all city data
    2. "How many customers in Los Angeles?" → can use cached data

    The first query fetches a superset of data that includes the answer
    to the second query.
    """
    km = _setup_multi_city_scenario()

    with patch.object(SETTINGS, "UNITY_SEMANTIC_CACHE", True):
        # First query - fetch all cities
        question_1 = (
            "How many customers are there in all cities? "
            "Answer with the number of customers only."
        )
        handle_1 = await km.ask(question_1)
        await handle_1.result()

        sc._SEMANTIC_CACHE_SAVER.wait()
        cache_spy.clear()

        # Second query - specific city (data already fetched)
        question_2 = (
            "How many customers are there in Los Angeles? "
            "Answer with the number of customers only."
        )
        handle_2 = await km.ask(question_2, _return_reasoning_steps=True)
        answer_2, reasoning_2 = await handle_2.result()

        # Answer should be correct
        assert "512" in answer_2

        # If cache was hit, minimal tool calls expected
        second_call = cache_spy.get_last_call()
        if second_call.was_hit:
            tool_calls_2, tools_called = _count_tool_calls_in_reasoning(reasoning_2)
            assert tool_calls_2 <= 1, (
                f"Cache hit should result in minimal tool calls. "
                f"Got {tool_calls_2} calls. Tools: {tools_called}"
            )


# =============================================================================
# Test 4: Aligned Queries - Same structure, same data, minimal tool calls
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_aligned_queries_minimal_tool_calls(cache_spy: CacheSpy):
    """
    Test that nearly identical queries result in minimal additional tool calls.

    This test uses aligned queries that ask for the same information in
    slightly different ways. The semantic cache should hit, and since the
    cached trajectory contains FRESH data (read-only tools are automatically
    re-executed), the LLM should need at most minimal additional tool calls.

    Scenario:
    1. "What is the customer count in New York?" → cache MISS, tools called
    2. "Tell me the customer count in New York" → cache HIT, minimal tool calls
    """
    km = _setup_single_city_scenario()

    with patch.object(SETTINGS, "UNITY_SEMANTIC_CACHE", True):
        # First query
        question_1 = (
            "What is the customer count in New York? " "Answer with the number only."
        )
        handle_1 = await km.ask(question_1, _return_reasoning_steps=True)
        answer_1, reasoning_1 = await handle_1.result()

        assert "10" in answer_1
        first_call = cache_spy.get_last_call()
        assert not first_call.was_hit, "First query should be a cache MISS"

        sc._SEMANTIC_CACHE_SAVER.wait()
        cache_spy.clear()

        # Second query - same intent, slightly different phrasing
        question_2 = (
            "Tell me the customer count in New York. " "Answer with the number only."
        )
        handle_2 = await km.ask(question_2, _return_reasoning_steps=True)
        answer_2, reasoning_2 = await handle_2.result()

        # Verify correct answer
        assert "10" in answer_2

        # Cache should be hit
        second_call = cache_spy.get_last_call()
        assert second_call.was_hit, (
            f"Semantically equivalent query should hit cache. "
            f"Q1: '{question_1}', Q2: '{question_2}'"
        )

        # The cached trajectory contains FRESH data (read-only tools are
        # automatically re-executed), so the LLM should mostly trust this data.
        # We allow at most 1 additional call for occasional verification behavior.
        tool_calls, tools_found = _count_tool_calls_in_reasoning(reasoning_2)

        assert tool_calls <= 1, (
            f"Cache hit with fresh data should result in minimal tool calls. "
            f"Got {tool_calls} tool calls. Tools: {tools_found}."
        )
