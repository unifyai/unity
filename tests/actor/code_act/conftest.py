import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from unity.function_manager.computer_backends import ActResult
from unity.function_manager.function_manager import FunctionManager
from unity.function_manager.primitives import ComputerPrimitives

_FM_METHOD_NAMES = (
    "search_functions",
    "filter_functions",
    "list_functions",
    "add_functions",
    "delete_function",
    "add_venv",
    "list_venvs",
    "get_venv",
    "update_venv",
    "delete_venv",
    "set_function_venv",
    "get_function_venv",
)


def make_fm_mock() -> MagicMock:
    """Create a FunctionManager MagicMock compatible with ``methods_to_tool_dict``.

    ``methods_to_tool_dict`` derives tool keys from ``fn.__self__.__class__``
    and ``fn.__name__``, which plain ``MagicMock`` methods lack. Using
    ``spec=FunctionManager`` gives the correct MRO so the canonical class name
    resolves to ``FunctionManager``, and we set ``__name__`` / ``__self__`` on
    each method so they look like real bound methods.
    """
    fm = MagicMock(spec=FunctionManager)
    for name in _FM_METHOD_NAMES:
        method = getattr(fm, name)
        method.__name__ = name
        method.__self__ = fm
    return fm


@pytest.fixture
def mock_computer_primitives():
    """Fixture to create a mock ComputerPrimitives for testing.

    Returns a mock with ``desktop`` sub-namespace matching the real API.
    """
    desktop_ns = MagicMock()
    desktop_ns.navigate = AsyncMock(return_value="navigated")
    desktop_ns.act = AsyncMock(
        return_value=ActResult(summary="acted", screenshot=""),
    )
    desktop_ns.observe = AsyncMock(return_value={"data": "observed_data"})
    desktop_ns.get_screenshot = AsyncMock(return_value=MagicMock())

    mock_provider = MagicMock(spec=ComputerPrimitives)
    mock_provider.desktop = desktop_ns
    return mock_provider


async def wait_for_turn_completion(task, initial_history_len, timeout=30):
    """
    Wait for the agent to process an interjection and enter an idle state.

    An idle state is detected when the last message is from the assistant and contains
    no tool calls, indicating it's waiting for the next command.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout

    while loop.time() < deadline:
        if len(task.get_history()) > initial_history_len:
            last_message = task.get_history()[-1]
            if last_message.get("role") == "assistant" and not last_message.get(
                "tool_calls",
            ):
                return
        await asyncio.sleep(0.5)

    raise AssertionError("Timed out waiting for turn completion")
