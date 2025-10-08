from __future__ import annotations

import pytest
import time
import unify
from typing import Dict
from unittest.mock import patch

from unity.contact_manager.contact_manager import ContactManager
from tests.helpers import _handle_project
from unity.common._async_tool import semantic_cache as sc
from unity.common._async_tool.semantic_cache import _Config


@pytest.fixture(autouse=True)
def _patch_semantic_cache_config(monkeypatch):
    class _DynamicConfig(_Config):
        @property
        def context(self):
            return f"{unify.get_active_context()['write']}/SemanticCache"

    monkeypatch.setattr(
        "unity.common._async_tool.semantic_cache._CONFIG",
        _DynamicConfig(),
    )


def _count_tool_calls_in_reasoning(reasoning_steps) -> int:
    """Count the number of tool calls in the reasoning steps."""
    tool_call_count = 0
    for step in reasoning_steps:
        if step.get("role") == "assistant" and step.get("tool_calls"):
            tool_call_count += len(step["tool_calls"])
    return tool_call_count


@_handle_project
@pytest.mark.asyncio
async def test_semantic_cache_exact_match_no_extra_tool_calls(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """
    Test that exact match queries leverage semantic cache.

    On the first run, the LLM should call tools to get contact information.
    On the second run with the same query, the semantic cache should be hit,
    and the LLM should return the cached response without additional tool calls.
    """
    cm, _ = contact_manager_scenario

    # Ensure semantic cache is enabled for this test
    with patch(
        "unity.contact_manager.contact_manager.is_semantic_cache_enabled",
        return_value=True,
    ):
        question = "What do you know about the contact Alice Smith?"

        # First run - should make tool calls
        handle_1 = await cm.ask(question, _return_reasoning_steps=True)
        answer_1, reasoning_1 = await handle_1.result()
        tool_calls_1 = _count_tool_calls_in_reasoning(reasoning_1)

        # Verify first run made tool calls
        assert (
            tool_calls_1 > 0
        ), f"First run should make tool calls, but got {tool_calls_1} tool calls"

        # Verify answer contains expected information
        assert (
            "alice" in answer_1.lower() or "smith" in answer_1.lower()
        ), f"First answer should contain contact information about Alice Smith"

        sc._SEMANTIC_CACHE_SAVER.wait()

        # Second run - should leverage cache with fewer or no tool calls
        handle_2 = await cm.ask(question, _return_reasoning_steps=True)
        answer_2, reasoning_2 = await handle_2.result()
        tool_calls_2 = _count_tool_calls_in_reasoning(reasoning_2)

        # Verify second run made fewer tool calls (ideally zero due to cache)
        assert tool_calls_2 < tool_calls_1, (
            f"Second run should make fewer or equal tool calls than first run. "
            f"First: {tool_calls_1}, Second: {tool_calls_2}"
        )

        # Verify answer still contains expected information
        assert (
            "alice" in answer_2.lower() or "smith" in answer_2.lower()
        ), f"Second answer should contain contact information about Alice Smith"


@_handle_project
@pytest.mark.asyncio
async def test_semantic_cache_performance_improvement(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """
    Test that semantic cache mode improves performance.

    Time without semantic cache (OFF) should be >= time with semantic cache (ON)
    when running the same query twice with cache enabled.
    """
    cm, _ = contact_manager_scenario

    question = "What is Bob Johnson's email and phone number?"

    # === Run WITHOUT semantic cache (baseline) ===
    with patch(
        "unity.contact_manager.contact_manager.is_semantic_cache_enabled",
        return_value=False,
    ):
        start_no_cache = time.time()

        # First query without cache
        handle_1 = await cm.ask(question)
        answer_1 = await handle_1.result()

        # Second query without cache
        handle_2 = await cm.ask(question)
        answer_2 = await handle_2.result()

        time_no_cache = time.time() - start_no_cache

    # === Run WITH semantic cache ===
    with patch(
        "unity.contact_manager.contact_manager.is_semantic_cache_enabled",
        return_value=True,
    ):
        start_with_cache = time.time()

        # First query with cache (cache miss, will populate cache)
        handle_3 = await cm.ask(question)
        answer_3 = await handle_3.result()

        sc._SEMANTIC_CACHE_SAVER.wait()

        # Second query with cache (cache hit, should be faster)
        handle_4 = await cm.ask(question)
        answer_4 = await handle_4.result()

        time_with_cache = time.time() - start_with_cache

    # Verify both answers contain expected information
    for answer in [answer_1, answer_2, answer_3, answer_4]:
        assert (
            "bob" in answer.lower() or "johnson" in answer.lower()
        ), "Answers should contain information about Bob Johnson"

    # We expect cache to be faster, but allow for some variance
    # Cache should provide at least some benefit or be comparable
    improvement_ratio = (
        time_no_cache / time_with_cache if time_with_cache > 0 else float("inf")
    )

    assert improvement_ratio >= 0.8, (
        f"Semantic cache should improve or maintain performance. "
        f"Without cache: {time_no_cache:.2f}s, With cache: {time_with_cache:.2f}s, "
        f"Ratio: {improvement_ratio:.2f}"
    )


@_handle_project
@pytest.mark.asyncio
async def test_semantic_cache_similar_queries_benefit(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """
    Test that similar queries with different parameters benefit from semantic cache.

    The second query should take a similar or shorter path (fewer reasoning steps)
    because the LLM can leverage the cached pattern from the first query.
    """
    cm, _ = contact_manager_scenario
    first_contact = "Alice Smith"
    second_contact = "Bob Johnson"

    with patch(
        "unity.contact_manager.contact_manager.is_semantic_cache_enabled",
        return_value=True,
    ):
        # First query - establish pattern
        question_1 = f"Find contact {first_contact}"
        handle_1 = await cm.ask(question_1, _return_reasoning_steps=True)
        answer_1, reasoning_1 = await handle_1.result()

        sc._SEMANTIC_CACHE_SAVER.wait()

        # Verify first answer
        name_parts_1 = first_contact.lower().split()
        assert any(
            part in answer_1.lower() for part in name_parts_1
        ), f"First answer should contain information about {first_contact}"

        # Second query - should follow similar path
        question_2 = f"Find contact {second_contact}"
        start_time_2 = time.time()
        handle_2 = await cm.ask(question_2, _return_reasoning_steps=True)
        answer_2, reasoning_2 = await handle_2.result()
        time_2 = time.time() - start_time_2

        # Verify second answer
        name_parts_2 = second_contact.lower().split()
        assert any(
            part in answer_2.lower() for part in name_parts_2
        ), f"Second answer should contain information about {second_contact}"

        # Count reasoning steps
        steps_1 = len(reasoning_1)
        steps_2 = len(reasoning_2)

        # Count tool calls
        tool_calls_1 = _count_tool_calls_in_reasoning(reasoning_1)
        tool_calls_2 = _count_tool_calls_in_reasoning(reasoning_2)

        # The second query should follow a similar or more efficient path
        # We allow some flexibility as the queries might differ in complexity
        assert steps_2 <= steps_1 * 1.5, (
            f"Second query should follow a similar path. "
            f"First: {steps_1} steps, Second: {steps_2} steps"
        )

        assert tool_calls_2 <= tool_calls_1 * 1.5, (
            f"Second query should not make significantly more tool calls. "
            f"First: {tool_calls_1} calls, Second: {tool_calls_2} calls"
        )
