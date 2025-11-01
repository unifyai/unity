from __future__ import annotations

import pytest

from unity.common.loop_snapshot import (
    LoopSnapshot,
    EntryPointManagerMethod,
    validate_snapshot,
)


def _sample_snapshot_dict():
    snap = LoopSnapshot(
        entrypoint=EntryPointManagerMethod(
            class_name="ContactManager",
            method_name="ask",
        ),
        loop_id="ContactManager.ask(abcdef)",
        system_message="You are helpful.",
        initial_user_message="Find contact Alice",
        assistant_steps=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_contacts",
                            "arguments": "{}",
                        },
                    },
                ],
            },
        ],
        tool_results=[
            {
                "id": "call_1",
                "name": "search_contacts",
                "content": "[]",
            },
        ],
    ).model_dump()
    return snap


def test_loop_snapshot_roundtrip():
    data = _sample_snapshot_dict()
    validated = validate_snapshot(data)
    assert validated.version == 1
    assert validated.entrypoint.class_name == "ContactManager"
    assert validated.entrypoint.method_name == "ask"
    assert isinstance(validated.assistant_steps, list)
    assert isinstance(validated.tool_results, list)


def test_loop_snapshot_unsupported_version():
    data = _sample_snapshot_dict()
    data["version"] = 999
    with pytest.raises(ValueError):
        validate_snapshot(data)
