from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.task import Task


_EXPECTED_KEYS_ORDER = [
    "task_keys_to_shorthand",
    "tasks",
    "shorthand_to_task_keys",
]


def _find_tool(tools: dict, needle: str) -> str | None:
    lowered = needle.lower()
    for name in tools.keys():
        if lowered in name.lower():
            return name
    return None


@pytest.mark.unit
@_handle_project
def test_filter_tasks_tool_return_shape():
    ts = TaskScheduler()
    # Ensure at least one task exists
    ts._create_task(name="RT shape", description="ensure one row")

    tools = ts.get_tools("ask")
    filt_name = _find_tool(tools, "filter_tasks")
    assert filt_name is not None

    out = tools[filt_name](limit=1)

    # Key order
    assert list(out.keys()) == _EXPECTED_KEYS_ORDER

    # Legend mappings
    assert out["task_keys_to_shorthand"] == Task.shorthand_map()
    assert out["shorthand_to_task_keys"] == Task.shorthand_inverse_map()

    # Types
    assert isinstance(out["tasks"], list)


@pytest.mark.unit
@_handle_project
def test_search_tasks_tool_return_shape():
    ts = TaskScheduler()
    # Seed a couple tasks
    ts._create_task(name="A", description="alpha")
    ts._create_task(name="B", description="beta")

    tools = ts.get_tools("ask")
    search_name = _find_tool(tools, "search_tasks")
    assert search_name is not None

    out = tools[search_name](references=None, k=1)

    # Key order
    assert list(out.keys()) == _EXPECTED_KEYS_ORDER

    # Legend mappings
    assert out["task_keys_to_shorthand"] == Task.shorthand_map()
    assert out["shorthand_to_task_keys"] == Task.shorthand_inverse_map()

    # Types
    assert isinstance(out["tasks"], list)
