import pytest
import unify
from tests.helpers import _handle_project
from unity.contact_manager.contact_manager import ContactManager
from unity.common._async_tool.semantic_cache import _Config
from unity.common._async_tool import semantic_cache as sc
from unity.conductor.conductor import Conductor


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
async def test_ask_semantic_one_manager(monkeypatch):
    monkeypatch.setenv("UNITY_SEMANTIC_CACHE", "true")

    cm = ContactManager()
    first_contact = cm._create_contact(first_name="John", surname="Doe")
    second_contact = cm._create_contact(first_name="Bob", surname="Alice")
    fc_id = first_contact["details"]["contact_id"]
    sc_id = second_contact["details"]["contact_id"]

    query = "Is there any contacts with name John? Answer with either Yes or No."

    manager = Conductor()
    handle = await manager.ask(query)
    res = await handle.result()
    assert "Yes" in res

    sc._SEMANTIC_CACHE_SAVER.wait()

    second_handle = await manager.ask(query, _return_reasoning_steps=True)
    second_res, reasoning_steps = await second_handle.result()

    assert "Yes" in second_res
    assert (
        _count_tool_calls_in_reasoning(reasoning_steps) == 0
    ), "No tool calls should be made"
