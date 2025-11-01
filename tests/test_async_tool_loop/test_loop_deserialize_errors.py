from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.common.async_tool_loop import AsyncToolLoopHandle


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_unknown_manager_class_raises():
    snap = {
        "version": 1,
        "entrypoint": {
            "type": "manager_method",
            "class_name": "NoSuchManager",
            "method_name": "ask",
        },
        "loop_id": "NoSuchManager.ask(xxx)",
        "system_message": "You are helpful.",
        "initial_user_message": "Hello",
        "assistant_steps": [],
        "tool_results": [],
    }

    with pytest.raises(ValueError, match="Manager class not found"):
        AsyncToolLoopHandle.deserialize(snap)


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_unknown_manager_method_raises():
    snap = {
        "version": 1,
        "entrypoint": {
            "type": "manager_method",
            "class_name": "ContactManager",
            "method_name": "nope",
        },
        "loop_id": "ContactManager.nope(xxx)",
        "system_message": "You are helpful.",
        "initial_user_message": "Hello",
        "assistant_steps": [],
        "tool_results": [],
    }

    with pytest.raises(ValueError, match="No tools registered for ContactManager.nope"):
        AsyncToolLoopHandle.deserialize(snap)
