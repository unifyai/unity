import asyncio
import pytest

from unity.contact_manager.contact_manager import ContactManager
from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_condition,
)


@pytest.mark.asyncio
@_handle_project
async def test_flat_ask():
    """
    Verify a flat, in‑flight ContactManager.ask loop reports a minimal structure.
    """
    cm = ContactManager()

    h = await cm.ask("Who is the contact living in Berlin working as a designer?")

    try:
        structure = await h.nested_structure()  # type: ignore[attr-defined]
        expected = {
            "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
            "tool": "ContactManager.ask",
            "children": [],
        }
        assert structure == expected
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=60)  # type: ignore[attr-defined]
        except Exception:
            pass


@pytest.mark.asyncio
@_handle_project
async def test_flat_update_before_nested(monkeypatch):
    """
    Verify a flat, in‑flight ContactManager.update loop reports a minimal structure
    when the first‑turn nested ask has been requested but not yet adopted.
    """
    gate = asyncio.Event()

    # Gate the manager's ask so it does not return a handle yet (keeps structure flat)
    original_ask = ContactManager.ask

    async def _gated_ask(self, *args, **kwargs):
        await gate.wait()
        # Return a simple string result to avoid creating a nested handle in this test
        return "ok"

    # Ensure the dynamic tool name exposed to the LLM remains exactly "ask"
    _gated_ask.__name__ = "ask"  # type: ignore[attr-defined]
    _gated_ask.__qualname__ = "ask"  # type: ignore[attr-defined]

    monkeypatch.setattr(ContactManager, "ask", _gated_ask, raising=True)

    cm = ContactManager()
    h = await cm.update(
        "Please update the contact's policy.",
    )  # should require ask first

    try:
        # Wait until the assistant has requested the first-turn 'ask' tool
        client = getattr(h, "_client", None)  # internal test-only access
        assert (
            client is not None
        ), "Expected AsyncToolLoopHandle to expose its client for tests"
        await _wait_for_tool_request(client, "ask")

        # Ask has been requested but is still blocked by the gate → no nested handle yet
        structure = await h.nested_structure()  # type: ignore[attr-defined]
        expected = {
            "handle": "AsyncToolLoopHandle",
            "tool": "ContactManager.update",
            "children": [],
        }
        assert structure == expected
    finally:
        # Release the gate so the loop can finish cleanly
        try:
            gate.set()
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=120)  # type: ignore[attr-defined]
        except Exception:
            try:
                h.stop("cleanup")  # type: ignore[attr-defined]
            except Exception:
                pass


@pytest.mark.asyncio
@_handle_project
async def test_update_then_ask_nested():
    """
    Verify a nested structure for ContactManager.update → ContactManager.ask
    (hard-coded policy requires 'ask' on the first turn).
    """
    cm = ContactManager()
    h = await cm.update(
        "Mark respond_to=True for the footballer who wrapped up a kickoff call.",
    )

    try:
        # Wait deterministically until the nested ask handle has been adopted
        async def _ask_child_adopted():
            try:
                task_info = getattr(getattr(h, "_task", None), "task_info", {})  # type: ignore[attr-defined]
                if isinstance(task_info, dict):
                    return any(
                        getattr(meta, "name", None) == "ask"
                        and getattr(meta, "handle", None) is not None
                        for meta in task_info.values()
                    )
            except Exception:
                return False
            return False

        await _wait_for_condition(_ask_child_adopted, poll=0.01, timeout=60.0)

        structure = await h.nested_structure()  # type: ignore[attr-defined]
        expected = {
            "handle": "AsyncToolLoopHandle",
            "tool": "ContactManager.update",
            "children": [
                {
                    "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
                    "tool": "ContactManager.ask",
                    "children": [],
                },
            ],
        }
        assert structure == expected
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=120)  # type: ignore[attr-defined]
        except Exception:
            pass
