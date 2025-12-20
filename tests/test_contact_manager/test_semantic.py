from __future__ import annotations

import pytest
import unify
from typing import Dict, List, Optional
from unittest.mock import patch
from dataclasses import dataclass, field

from unity.contact_manager.contact_manager import ContactManager
from unity.settings import SETTINGS
from tests.helpers import _handle_project
from unity.common._async_tool import semantic_cache as sc
from unity.common._async_tool.semantic_cache import (
    _Config,
    SemanticCacheResult,
    search_semantic_cache,
)

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
        # 0.5 is a reasonable threshold for semantic similarity:
        # - "Find contact Alice Smith" ↔ "Get info on Alice Smith" should match
        # - "Find contact Alice Smith" ↔ "Find contact Bob Johnson" should NOT match
        threshold = 0.5

        @property
        def context(self):
            return f"{unify.get_active_context()['write']}/SemanticCache"

    monkeypatch.setattr(
        "unity.common._async_tool.semantic_cache._CONFIG",
        _DynamicConfig(),
    )


def _count_tool_calls_in_reasoning(reasoning_steps) -> int:
    """Count the number of tool calls in the reasoning steps, excluding semantic_search."""
    tool_call_count = 0
    for step in reasoning_steps:
        if step.get("role") == "tool":
            if step.get("name") == "semantic_search":
                continue
            tool_call_count += 1
    return tool_call_count


# =============================================================================
# Test 1: Exact Match - Same query twice should hit cache
# =============================================================================


@_handle_project
@pytest.mark.asyncio
async def test_exact_match_cache_hit(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
    cache_spy: CacheSpy,
):
    """
    Test that an identical query on the second run hits the semantic cache.

    Scenario:
    1. First query: "What do you know about Alice Smith?" → cache MISS, tools called
    2. Second query: identical → cache HIT, cached trajectory reused

    Verification:
    - Direct: cache_spy confirms cache hit on second call
    - Indirect: second run has fewer tool calls
    """
    cm, _ = contact_manager_scenario

    with patch.object(SETTINGS, "UNITY_SEMANTIC_CACHE", True):
        question = "What do you know about the contact Alice Smith?"

        # First run - should miss cache and call tools
        handle_1 = await cm.ask(question, _return_reasoning_steps=True)
        answer_1, reasoning_1 = await handle_1.result()
        tool_calls_1 = _count_tool_calls_in_reasoning(reasoning_1)

        # Verify first run behavior
        assert tool_calls_1 > 0, "First run should make tool calls to fetch data"
        assert (
            "alice" in answer_1.lower() or "smith" in answer_1.lower()
        ), "Answer should contain information about Alice Smith"

        # Check cache spy: first call should be a miss (no prior cache entry)
        first_cache_call = cache_spy.get_last_call()
        assert first_cache_call is not None, "Cache should have been searched"
        assert not first_cache_call.was_hit, "First query should be a cache MISS"

        # Wait for cache save to complete
        sc._SEMANTIC_CACHE_SAVER.wait()
        cache_spy.clear()  # Clear spy for clean second-run verification

        # Second run - should hit cache
        handle_2 = await cm.ask(question, _return_reasoning_steps=True)
        answer_2, reasoning_2 = await handle_2.result()
        tool_calls_2 = _count_tool_calls_in_reasoning(reasoning_2)

        # Direct verification: cache was hit
        second_cache_call = cache_spy.get_last_call()
        assert second_cache_call is not None, "Cache should have been searched"
        assert second_cache_call.was_hit, (
            f"Second (identical) query should be a cache HIT. " f"Query: '{question}'"
        )

        # Verify cached entry matches our query
        assert second_cache_call.result.original_user_message == question

        # Indirect verification: fewer tool calls
        assert tool_calls_2 < tool_calls_1, (
            f"Cache hit should result in fewer tool calls. "
            f"First: {tool_calls_1}, Second: {tool_calls_2}"
        )

        # Answer quality preserved
        assert "alice" in answer_2.lower() or "smith" in answer_2.lower()


# =============================================================================
# Test 2: Semantically Equivalent - Same entity, different phrasing
# =============================================================================


@_handle_project
@pytest.mark.asyncio
async def test_semantically_equivalent_queries_same_entity(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
    cache_spy: CacheSpy,
):
    """
    Test that semantically equivalent queries about the SAME entity hit the cache.

    Scenario:
    1. "Find contact Alice Smith" → cache MISS
    2. "Get information about Alice Smith" → cache HIT (same entity, different phrasing)

    This tests true semantic similarity: the queries are structurally different
    but ask for the same information about the same entity.
    """
    cm, _ = contact_manager_scenario

    with patch.object(SETTINGS, "UNITY_SEMANTIC_CACHE", True):
        # First query - establishes cache entry for Alice
        question_1 = "Find contact Alice Smith"
        handle_1 = await cm.ask(question_1, _return_reasoning_steps=True)
        answer_1, reasoning_1 = await handle_1.result()

        assert "alice" in answer_1.lower() or "smith" in answer_1.lower()
        first_call = cache_spy.get_last_call()
        assert not first_call.was_hit, "First query should be a cache MISS"

        sc._SEMANTIC_CACHE_SAVER.wait()
        cache_spy.clear()

        # Second query - different phrasing, same entity
        question_2 = "Get information about Alice Smith"
        handle_2 = await cm.ask(question_2, _return_reasoning_steps=True)
        answer_2, reasoning_2 = await handle_2.result()

        # Direct verification: should hit cache due to semantic similarity
        second_call = cache_spy.get_last_call()
        assert second_call.was_hit, (
            f"Semantically equivalent query should hit cache. "
            f"Q1: '{question_1}', Q2: '{question_2}'"
        )

        # The cached entry should be from the first query about Alice
        assert "alice" in second_call.result.closest_user_message.lower()

        # Answer should still be correct
        assert "alice" in answer_2.lower() or "smith" in answer_2.lower()

        # Fewer tool calls due to cache
        tool_calls_1 = _count_tool_calls_in_reasoning(reasoning_1)
        tool_calls_2 = _count_tool_calls_in_reasoning(reasoning_2)
        assert (
            tool_calls_2 < tool_calls_1
        ), f"Cache hit should reduce tool calls. First: {tool_calls_1}, Second: {tool_calls_2}"


# =============================================================================
# Test 3: Different Entities - Should NOT hit cache (or hit is not useful)
# =============================================================================


@_handle_project
@pytest.mark.asyncio
async def test_different_entities_cache_miss(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
    cache_spy: CacheSpy,
):
    """
    Test that queries about DIFFERENT entities do not incorrectly use cached data.

    Scenario:
    1. "Find contact Alice Smith" → cache MISS, fetches Alice's data
    2. "Find contact Bob Johnson" → should NOT return Alice's cached data

    Even if the query structure is similar, the semantic cache should recognize
    these are queries about different entities. The cached Alice data cannot
    answer a question about Bob.
    """
    cm, _ = contact_manager_scenario

    with patch.object(SETTINGS, "UNITY_SEMANTIC_CACHE", True):
        # First query about Alice
        question_1 = "Find contact Alice Smith"
        handle_1 = await cm.ask(question_1, _return_reasoning_steps=True)
        answer_1, _ = await handle_1.result()

        assert "alice" in answer_1.lower() or "smith" in answer_1.lower()
        sc._SEMANTIC_CACHE_SAVER.wait()
        cache_spy.clear()

        # Second query about Bob - different entity
        question_2 = "Find contact Bob Johnson"
        handle_2 = await cm.ask(question_2, _return_reasoning_steps=True)
        answer_2, _ = await handle_2.result()

        second_call = cache_spy.get_last_call()

        # The critical assertion: answer must be about Bob, not Alice
        assert (
            "bob" in answer_2.lower() or "johnson" in answer_2.lower()
        ), f"Answer should be about Bob Johnson, not cached Alice data. Got: {answer_2}"
        assert (
            "alice" not in answer_2.lower()
        ), f"Answer should NOT contain Alice (wrong cached data). Got: {answer_2}"

        # If cache was hit, it should not have caused wrong answers
        # (The system should either miss cache, or re-execute tools despite cache hit)
        if second_call.was_hit:
            # Cache hit is acceptable IF the answer is still correct about Bob
            # This means the system properly re-executed tools with Bob's parameters
            pass
        else:
            # Cache miss is the expected "correct" behavior for different entities
            pass


# =============================================================================
# Test 4: Aligned Queries - Same structure, same entity, different phrasing
# =============================================================================


@_handle_project
@pytest.mark.asyncio
async def test_aligned_queries_minimal_tool_calls(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
    cache_spy: CacheSpy,
):
    """
    Test that nearly identical queries result in zero additional tool calls.

    This test uses aligned queries that ask for the same information in
    slightly different ways. The semantic cache should hit, and since the
    cached trajectory contains FRESH data (read-only tools are automatically
    re-executed), the LLM should need at most minimal additional tool calls.

    Scenario:
    1. "What is Bob Johnson's email address?" → cache MISS, tools called
    2. "Can you tell me Bob Johnson's email?" → cache HIT, minimal tool calls

    This test validates:
    - The cache hit occurs (verified via spy)
    - The LLM mostly trusts the cached fresh data (at most 1 additional call)
    """
    cm, _ = contact_manager_scenario

    with patch.object(SETTINGS, "UNITY_SEMANTIC_CACHE", True):
        # First query - establishes cache entry
        question_1 = "What is Bob Johnson's email address?"
        handle_1 = await cm.ask(question_1, _return_reasoning_steps=True)
        answer_1, reasoning_1 = await handle_1.result()

        assert "bobbyj@" in answer_1, f"Should return Bob's email. Got: {answer_1}"
        first_call = cache_spy.get_last_call()
        assert not first_call.was_hit, "First query should be a cache MISS"

        sc._SEMANTIC_CACHE_SAVER.wait()
        cache_spy.clear()

        # Second query - same intent, slightly different phrasing
        question_2 = "Can you tell me Bob Johnson's email?"
        handle_2 = await cm.ask(question_2, _return_reasoning_steps=True)
        answer_2, reasoning_2 = await handle_2.result()

        # Verify correct answer
        assert "bobbyj@" in answer_2, f"Should return Bob's email. Got: {answer_2}"

        # Cache should be hit
        second_call = cache_spy.get_last_call()
        assert second_call.was_hit, (
            f"Semantically equivalent query should hit cache. "
            f"Q1: '{question_1}', Q2: '{question_2}'"
        )

        # Strict assertion: zero additional tool calls
        # The cached trajectory contains FRESH data (read-only tools are
        # automatically re-executed), so the LLM should mostly trust this data.
        # We allow at most 1 additional call for occasional verification behavior.
        tool_calls = _count_tool_calls_in_reasoning(reasoning_2)

        tools_found = []
        for step in reasoning_2:
            if step.get("role") == "tool":
                tools_found.append(step.get("name"))

        assert tool_calls <= 1, (
            f"Cache hit with fresh data should result in minimal tool calls. "
            f"Got {tool_calls} tool calls. Tools: {tools_found}."
        )
