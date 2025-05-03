import json
import difflib
import pytest

from planner.update_handler import _build_diff_payload, context


def test_build_diff_payload(monkeypatch):
    # Arrange: sample old and new source code
    old = "def foo():\n    pass\n"
    new = "def foo():\n    click_on('bar')\n"

    # Stub context to provide dummy call stack and browser state
    monkeypatch.setattr(context, "get_call_stack", lambda: ["foo"])
    monkeypatch.setattr(context, "last_state_snapshot", lambda: {"active_tab": "tab1"})

    # Act: build the diff payload
    payload = _build_diff_payload("foo", old, new)
    data = json.loads(payload)

    # Assert: payload contains expected fields
    assert "browser_state" in data
    assert "diff" in data and "--- a/foo" in data["diff"]
    assert data["function_name"] == "foo"
