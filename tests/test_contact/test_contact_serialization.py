import asyncio
import pytest

from unity.contact_manager.contact_manager import ContactManager
from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_condition,
)


def _assert_subset(expected: dict, actual: dict, path: list[str] | None = None) -> None:
    """
    Assert that `expected` is a subset of `actual` with exact value matches
    for all specified keys, recursively for dicts.
    """
    path = path or []
    assert isinstance(actual, dict), f"Expected dict at {'.'.join(path) or '<root>'}"
    for k, v in expected.items():
        assert k in actual, f"Missing key at {'.'.join(path + [k])}"
        av = actual[k]
        if isinstance(v, dict):
            assert isinstance(
                av,
                dict,
            ), f"Expected dict at {'.'.join(path + [k])}, got {type(av)}"
            _assert_subset(v, av, path + [k])
        else:
            assert av == v, f"Value mismatch at {'.'.join(path + [k])}: {av!r} != {v!r}"


@pytest.mark.asyncio
@_handle_project
async def test_serialize_flat_contactmanager_ask():
    """
    Verify a flat ContactManager.ask loop serializes to the expected shape.
    """
    question = "Who is the contact living in Berlin working as a designer?"

    cm = ContactManager()
    h = await cm.ask(question)

    try:
        snap = h.serialize()  # capture without recursion for a flat loop

        # Human-readable expected subset shape
        expected = {
            "version": 1,
            "entrypoint": {"class_name": "ContactManager", "method_name": "ask"},
            "root": {
                "handle": "AsyncToolLoopHandle",
                "tool": "ContactManager.ask",
            },
            "initial_user_message": question,
        }

        # Stable prefix assertion for loop_id
        assert snap.get("loop_id", "").startswith("ContactManager.ask")

        # Meta subset checks (dynamic fields like timestamps are not asserted exactly)
        meta = snap.get("meta") or {}
        assert isinstance(meta.get("run_id", ""), str) and len(meta["run_id"]) > 0
        # semantic_cache_namespace present for manager entrypoints (ask)
        assert meta.get("semantic_cache_namespace") == "ContactManager.ask"

        _assert_subset(expected, snap)
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
async def test_serialize_flat_contactmanager_update_before_nested(monkeypatch):
    """
    Serialize a ContactManager.update loop while the first-turn nested ask
    has been requested but is not yet adopted. The snapshot should reflect a
    flat parent with the correct entrypoint/root and initial message.
    """
    gate = asyncio.Event()

    # Gate the manager's ask so it does not return a nested handle yet (keeps structure flat)
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
    request_text = "Please update the contact's policy."
    h = await cm.update(request_text)

    try:
        # Wait until the assistant has requested the first-turn 'ask' tool
        client = getattr(h, "_client", None)  # internal test-only access
        assert (
            client is not None
        ), "Expected AsyncToolLoopHandle to expose its client for tests"
        await _wait_for_tool_request(client, "ask")

        # Snapshot while nested ask has not been adopted
        snap = h.serialize()

        expected = {
            "version": 1,
            "root": {
                "handle": "AsyncToolLoopHandle",
                "tool": "InlineTools",
            },
            "initial_user_message": request_text,
        }

        assert snap.get("loop_id", "").startswith("ContactManager.update")
        _assert_subset(expected, snap)

        # EntryPoint: update uses inline tools (no manager semantic namespace set)
        ep = snap.get("entrypoint") or {}
        assert ep.get("type") == "inline_tools"
        tools = ep.get("tools") or []
        assert isinstance(tools, list) and len(tools) > 0
        # Ensure the inline tools include 'ask' (first-turn policy requires ask)
        assert any(isinstance(t, dict) and t.get("name") == "ask" for t in tools)
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
        finally:
            # Restore original ask (best-effort)
            try:
                monkeypatch.setattr(ContactManager, "ask", original_ask, raising=True)
            except Exception:
                pass


@pytest.mark.asyncio
@_handle_project
async def test_serialize_contactmanager_update_then_ask_nested():
    """
    Verify the recursive snapshot contains a child entry for ContactManager.ask
    when ContactManager.update triggers an ask on the first turn.
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

        snap = h.serialize(recursive=True)

        # Top-level subset assertions (root & version)
        expected_top = {
            "version": 1,
            "root": {
                "handle": "AsyncToolLoopHandle",
                "tool": "InlineTools",
            },
        }
        _assert_subset(expected_top, snap)

        # EntryPoint: update snapshots encode inline tools
        ep = snap.get("entrypoint") or {}
        assert ep.get("type") == "inline_tools"
        tlist = ep.get("tools") or []
        assert isinstance(tlist, list) and len(tlist) > 0

        # Children manifest must include a ContactManager.ask child
        meta = snap.get("meta") or {}
        children = meta.get("children") or []
        assert isinstance(children, list) and len(children) >= 1

        # Find the ask child
        ask_child = None
        for ch in children:
            if isinstance(ch, dict) and ch.get("tool") == "ContactManager.ask":
                ask_child = ch
                break
        assert ask_child is not None, "Expected ContactManager.ask child in snapshot"

        # Human-readable shape for the child (subset – state/call_id/snapshot may vary)
        expected_child_subset = {
            "tool": "ContactManager.ask",
            "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
        }
        _assert_subset(expected_child_subset, ask_child)

        # Optional: state is either 'in_flight' or 'done'
        st = ask_child.get("state")
        assert st in ("in_flight", "done"), f"Unexpected child state: {st}"
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=120)  # type: ignore[attr-defined]
        except Exception:
            pass
