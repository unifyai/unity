from __future__ import annotations

import pytest
import unify
import json

from typing import List, Optional
from dataclasses import dataclass, field
from unittest.mock import patch
from datetime import datetime

from tests.helpers import _handle_project
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.contact_manager.contact_manager import ContactManager
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


@pytest.fixture
def transcript_scenario():
    """
    Set up a TranscriptManager with two contacts and messages between them.
    Returns (tm, john_id, bob_id) for use in tests.
    """
    cm = ContactManager()
    john = cm._create_contact(first_name="John", surname="Doe")
    bob = cm._create_contact(first_name="Bob", surname="Alice")
    john_id = john["details"]["contact_id"]
    bob_id = bob["details"]["contact_id"]

    tm = TranscriptManager()
    tm.log_first_message_in_new_exchange(
        {
            "medium": "email",
            "sender_id": john_id,
            "receiver_ids": [bob_id],
            "timestamp": datetime.now(),
            "content": "Hey there!",
        },
    )
    tm.log_first_message_in_new_exchange(
        {
            "medium": "email",
            "sender_id": bob_id,
            "receiver_ids": [john_id],
            "timestamp": datetime.now(),
            "content": "This is Bob!",
        },
    )

    return tm, john_id, bob_id


# =============================================================================
# Test 1: Exact Match - Same query twice should hit cache
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_exact_match_cache_hit(transcript_scenario, cache_spy: CacheSpy):
    """
    Test that an identical query on the second run hits the semantic cache.

    Scenario:
    1. First query: ask for John's latest message → cache MISS, tools called
    2. Second query: identical → cache HIT, cached trajectory reused

    Verification:
    - Direct: cache_spy confirms cache hit on second call
    - Indirect: second run has fewer tool calls
    """
    tm, _, _ = transcript_scenario

    question = (
        "Is there any contact with name John? if so, please provide the latest message by John. "
        "Provide only the message content if any in the JSON format of {message: <message_content>}"
    )

    with patch.object(SETTINGS, "UNITY_SEMANTIC_CACHE", True):
        # First run - should miss cache and call tools
        handle_1 = await tm.ask(question, _return_reasoning_steps=True)
        answer_1, reasoning_1 = await handle_1.result()
        tool_calls_1, _ = _count_tool_calls_in_reasoning(reasoning_1)

        # Verify first run behavior
        assert tool_calls_1 > 0, "First run should make tool calls to fetch data"
        assert json.loads(answer_1)["message"] == "Hey there!"

        # Check cache spy: first call should be a miss
        first_cache_call = cache_spy.get_last_call()
        assert first_cache_call is not None, "Cache should have been searched"
        assert not first_cache_call.was_hit, "First query should be a cache MISS"

        sc._SEMANTIC_CACHE_SAVER.wait()
        cache_spy.clear()

        # Second run - should hit cache
        handle_2 = await tm.ask(question, _return_reasoning_steps=True)
        answer_2, reasoning_2 = await handle_2.result()
        tool_calls_2, tools_called = _count_tool_calls_in_reasoning(reasoning_2)

        # Verify answer is correct
        assert json.loads(answer_2)["message"] == "Hey there!"

        # Direct verification: cache was hit
        second_cache_call = cache_spy.get_last_call()
        assert second_cache_call is not None, "Cache should have been searched"
        assert second_cache_call.was_hit, (
            f"Second (identical) query should be a cache HIT. " f"Query: '{question}'"
        )

        # Indirect verification: fewer tool calls
        assert tool_calls_2 < tool_calls_1, (
            f"Cache hit should result in fewer tool calls. "
            f"First: {tool_calls_1}, Second: {tool_calls_2}"
        )

        # No search/filter tools should be called on cache hit
        assert "search_messages" not in tools_called
        assert "filter_messages" not in tools_called


# =============================================================================
# Test 2: Semantically Equivalent - Same entity, different phrasing
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_semantically_equivalent_queries_same_entity(
    transcript_scenario,
    cache_spy: CacheSpy,
):
    """
    Test that semantically equivalent queries about the SAME entity hit the cache.

    Scenario:
    1. "What is the latest message from John?" → cache MISS
    2. "Can you get John's most recent message?" → cache HIT (same entity, different phrasing)

    This tests true semantic similarity: the queries are structurally different
    but ask for the same information about the same entity.
    """
    tm, _, _ = transcript_scenario

    with patch.object(SETTINGS, "UNITY_SEMANTIC_CACHE", True):
        # First query about John's message
        question_1 = (
            "What is the latest message from John? "
            "Provide only the message content in JSON format: {message: <content>}"
        )
        handle_1 = await tm.ask(question_1, _return_reasoning_steps=True)
        answer_1, reasoning_1 = await handle_1.result()

        assert json.loads(answer_1)["message"] == "Hey there!"
        first_call = cache_spy.get_last_call()
        assert not first_call.was_hit, "First query should be a cache MISS"

        sc._SEMANTIC_CACHE_SAVER.wait()
        cache_spy.clear()

        # Second query - different phrasing, same entity
        question_2 = (
            "Can you get John's most recent message? "
            "Provide only the message content in JSON format: {message: <content>}"
        )
        handle_2 = await tm.ask(question_2, _return_reasoning_steps=True)
        answer_2, reasoning_2 = await handle_2.result()

        # Direct verification: should hit cache
        second_call = cache_spy.get_last_call()
        assert second_call.was_hit, (
            f"Semantically equivalent query should hit cache. "
            f"Q1: '{question_1}', Q2: '{question_2}'"
        )

        # Answer should still be correct
        assert json.loads(answer_2)["message"] == "Hey there!"

        # Fewer tool calls due to cache
        tool_calls_1, _ = _count_tool_calls_in_reasoning(reasoning_1)
        tool_calls_2, _ = _count_tool_calls_in_reasoning(reasoning_2)
        assert tool_calls_2 < tool_calls_1, (
            f"Cache hit should reduce tool calls. "
            f"First: {tool_calls_1}, Second: {tool_calls_2}"
        )


# =============================================================================
# Test 3: Different Entities - Should NOT hit cache (or hit is not useful)
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_different_entities_correct_answers(
    transcript_scenario,
    cache_spy: CacheSpy,
):
    """
    Test that queries about DIFFERENT entities do not incorrectly use cached data.

    Scenario:
    1. "Get John's latest message" → cache MISS, returns "Hey there!"
    2. "Get Bob's latest message" → should return "This is Bob!", NOT cached John data

    Even if the query structure is similar, the cached John data cannot
    answer a question about Bob.
    """
    tm, _, _ = transcript_scenario

    with patch.object(SETTINGS, "UNITY_SEMANTIC_CACHE", True):
        # First query about John
        question_1 = (
            "Get John's latest message. "
            "Provide only the message content in JSON format: {message: <content>}"
        )
        handle_1 = await tm.ask(question_1, _return_reasoning_steps=True)
        answer_1, _ = await handle_1.result()

        assert json.loads(answer_1)["message"] == "Hey there!"
        sc._SEMANTIC_CACHE_SAVER.wait()
        cache_spy.clear()

        # Second query about Bob - different entity
        question_2 = (
            "Get Bob's latest message. "
            "Provide only the message content in JSON format: {message: <content>}"
        )
        handle_2 = await tm.ask(question_2, _return_reasoning_steps=True)
        answer_2, _ = await handle_2.result()

        # The critical assertion: answer must be about Bob, not John
        assert (
            json.loads(answer_2)["message"] == "This is Bob!"
        ), f"Answer should be Bob's message, not cached John data. Got: {answer_2}"


# =============================================================================
# Test 4: Aligned Queries - Same structure, same entity, minimal tool calls
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_aligned_queries_minimal_tool_calls(
    transcript_scenario,
    cache_spy: CacheSpy,
):
    """
    Test that nearly identical queries result in zero additional tool calls.

    This test uses aligned queries that ask for the same information in
    slightly different ways. The semantic cache should hit, and since the
    cached trajectory contains FRESH data (read-only tools are automatically
    re-executed), the LLM should need at most minimal additional tool calls.

    Scenario:
    1. "What is John's latest message?" → cache MISS, tools called
    2. "Can you tell me John's latest message?" → cache HIT, minimal tool calls
    """
    tm, _, _ = transcript_scenario

    with patch.object(SETTINGS, "UNITY_SEMANTIC_CACHE", True):
        # First query
        question_1 = (
            "What is John's latest message? "
            "Provide only the message content in JSON format: {message: <content>}"
        )
        handle_1 = await tm.ask(question_1, _return_reasoning_steps=True)
        answer_1, reasoning_1 = await handle_1.result()

        assert json.loads(answer_1)["message"] == "Hey there!"
        first_call = cache_spy.get_last_call()
        assert not first_call.was_hit, "First query should be a cache MISS"

        sc._SEMANTIC_CACHE_SAVER.wait()
        cache_spy.clear()

        # Second query - same intent, slightly different phrasing
        question_2 = (
            "Can you tell me John's latest message? "
            "Provide only the message content in JSON format: {message: <content>}"
        )
        handle_2 = await tm.ask(question_2, _return_reasoning_steps=True)
        answer_2, reasoning_2 = await handle_2.result()

        # Verify correct answer
        assert json.loads(answer_2)["message"] == "Hey there!"

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
