import pytest
import unify

from unittest.mock import patch
from tests.helpers import _handle_project

from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.common._async_tool.semantic_cache import _Config
from unity.common._async_tool import semantic_cache as sc


@pytest.fixture(autouse=True)
def _patch_semantic_cache_config(monkeypatch):
    class _DynamicConfig(_Config):
        # Raise threshold to ensure cache is always hit during the test
        threshold = 0.5

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
        if step.get("role") == "tool":
            if step.get("name") == "semantic_search":
                continue
            tool_call_count += 1
    return tool_call_count


@pytest.mark.asyncio
@_handle_project
async def test_semantic_cache_exact_match():
    km = KnowledgeManager()
    km._create_table(name="MyTable", columns={"city": "str", "customers": "int"})
    km._add_rows(
        table="MyTable",
        rows=[
            {"city": "New York", "customers": 10},
        ],
    )

    query = "What is the number of customers in New York? Answer with the number of customers only."

    with patch(
        "unity.knowledge_manager.knowledge_manager.is_semantic_cache_enabled",
        return_value=True,
    ):
        handle = await km.ask(query, _return_reasoning_steps=True)
        answer, reasoning_steps_first = await handle.result()
        assert answer == "10"

        sc._SEMANTIC_CACHE_SAVER.wait()

        handle = await km.ask(query, _return_reasoning_steps=True)
        answer, reasoning_steps_second = await handle.result()
        assert answer == "10"

        # Reasoning should take less steps as no tools should be called
        total_calls_second = _count_tool_calls_in_reasoning(reasoning_steps_second)

        # All information should be available from the first query
        assert total_calls_second == 0


@pytest.mark.asyncio
@_handle_project
async def test_semantic_cache_no_exact_match():
    km = KnowledgeManager()
    km._create_table(name="MyTable", columns={"city": "str", "customers": "int"})
    km._add_rows(
        table="MyTable",
        rows=[
            {"city": "New York", "customers": 10},
        ],
    )

    first_query = "What is the number of customers in New York? Answer with the number of customers only."
    second_query = "How many customers are there in New York? Answer with the number of customers only."

    with patch(
        "unity.knowledge_manager.knowledge_manager.is_semantic_cache_enabled",
        return_value=True,
    ):
        handle = await km.ask(first_query)
        answer = await handle.result()
        assert "10" in answer

        sc._SEMANTIC_CACHE_SAVER.wait()

        handle = await km.ask(second_query, _return_reasoning_steps=True)
        answer, reasoning_steps_second = await handle.result()
        assert "10" in answer

        # Reasoning should take less steps as no tools should be called
        total_calls_second = _count_tool_calls_in_reasoning(reasoning_steps_second)

        # All information should be available from the first query
        assert total_calls_second == 0


@pytest.mark.asyncio
@_handle_project
async def test_semantic_cache_similar_query_benefit():
    km = KnowledgeManager()
    km._create_table(name="MyTable", columns={"city": "str", "customers": "int"})
    km._add_rows(
        table="MyTable",
        rows=[
            {"city": "New York", "customers": 10},
            {"city": "Los Angeles", "customers": 512},
        ],
    )

    first_query = "How many customers are there in all cities? Answer with the number of customers only."
    second_query = "How many customers are there in Los Angeles? Answer with the number of customers only."

    with patch(
        "unity.knowledge_manager.knowledge_manager.is_semantic_cache_enabled",
        return_value=True,
    ):
        handle = await km.ask(first_query)
        await handle.result()

        sc._SEMANTIC_CACHE_SAVER.wait()

        handle = await km.ask(second_query, _return_reasoning_steps=True)
        answer, reasoning_steps_second = await handle.result()
        assert "512" in answer

        total_calls_second = _count_tool_calls_in_reasoning(reasoning_steps_second)

        # Previous query should already contain the result required for the second query
        assert total_calls_second == 0
