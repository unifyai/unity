import asyncio
import pytest

from unity.contact_manager.contact_manager import ContactManager
from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_condition,
    _wait_for_tool_request,
)


def _assert_dict_subset(expected: dict, actual: dict):
    """
    Recursively assert that `expected` is a subset of `actual`.
    Values in `expected` must match exactly for corresponding keys in `actual`.
    """
    assert isinstance(actual, dict), "Actual value is not a dict"
    for k, v in expected.items():
        assert k in actual, f"Missing key in actual snapshot: {k!r}"
        if isinstance(v, dict):
            assert isinstance(
                actual[k],
                dict,
            ), f"Expected dict at key {k!r}, got {type(actual[k]).__name__}"
            _assert_dict_subset(v, actual[k])
        elif isinstance(v, list):
            assert isinstance(
                actual[k],
                list,
            ), f"Expected list at key {k!r}, got {type(actual[k]).__name__}"
            # For lists, ensure each expected item is present (subset semantics).
            # - If dict: require at least one actual item to be a superset of this dict.
            # - Else: require exact membership.
            for exp_item in v:
                if isinstance(exp_item, dict):
                    found = False
                    for act_item in actual[k]:
                        if isinstance(act_item, dict):
                            try:
                                _assert_dict_subset(exp_item, act_item)
                                found = True
                                break
                            except AssertionError:
                                continue
                    assert (
                        found
                    ), f"List at key {k!r} missing an item matching subset {exp_item!r}"
                else:
                    assert (
                        exp_item in actual[k]
                    ), f"List at key {k!r} missing item {exp_item!r}"
        else:
            assert (
                actual[k] == v
            ), f"Value mismatch at key {k!r}: {actual[k]!r} != {v!r}"


@pytest.mark.asyncio
@_handle_project
async def test_serialize_flat_contactmanager_ask():
    """
    Verify a flat ContactManager.ask snapshot contains the expected minimal shape.

    We validate a subset of the snapshot that is stable and human‑readable:
      - version
      - loop_id prefix
      - root.tool and root.handle
      - assistant and tools types
    """
    cm = ContactManager()
    h = await cm.ask("Who is the contact living in Berlin working as a designer?")

    try:
        snap = h.serialize()  # type: ignore[attr-defined]

        # Human-readable expected subset
        expected = {
            "version": 1,
            "root": {
                "tool": "ContactManager.ask",
                "handle": "AsyncToolLoopHandle",
            },
        }
        _assert_dict_subset(expected, snap)
        assert snap.get("loop_id", "").startswith(
            "ContactManager.ask",
        ), "loop_id must start with ContactManager.ask"
        assert isinstance(snap.get("assistant"), list)
        assert isinstance(snap.get("tools"), list)
        # For non-recursive serialize, 'children' should be absent or an empty list
        if "children" in snap:
            assert isinstance(snap["children"], list) and len(snap["children"]) == 0
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
    Verify a flat ContactManager.update snapshot when the first-turn nested ask
    has been requested but not adopted (keeps snapshot flat and non-recursive).
    """
    gate = asyncio.Event()

    # Gate the manager's ask so it does not return a handle yet (keeps snapshot flat)
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
    )

    try:
        # Wait until the assistant has requested the first-turn 'ask' tool
        client = getattr(h, "_client", None)  # internal test-only access
        assert (
            client is not None
        ), "Expected AsyncToolLoopHandle to expose its client for tests"
        await _wait_for_tool_request(client, "ask")

        # Ask has been requested but is still blocked by the gate → no nested handle yet
        snap = h.serialize()  # type: ignore[attr-defined]
        expected = {
            "version": 1,
            "root": {
                "tool": "ContactManager.update",
                "handle": "AsyncToolLoopHandle",
            },
        }
        _assert_dict_subset(expected, snap)
        assert snap.get("loop_id", "").startswith(
            "ContactManager.update",
        ), "loop_id must start with ContactManager.update"
        assert isinstance(snap.get("assistant"), list)
        assert isinstance(snap.get("tools"), list)
        # For non-recursive serialize, 'children' should be absent or an empty list
        if "children" in snap:
            assert isinstance(snap["children"], list) and len(snap["children"]) == 0
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
async def test_serialize_contactmanager_update_then_ask_nested():
    """
    Verify a recursive snapshot for ContactManager.update → ContactManager.ask
    (policy requires 'ask' on the first turn).

    We assert a human-readable subset that presents the nested structure in one dict:
      - root.tool/handle
      - "children" displayed at the base level containing an entry for ContactManager.ask
      - child's handle shape
      - child's state is either 'in_flight' or 'done'
      - if 'in_flight', child includes an inline snapshot
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

        snap = h.serialize(recursive=True)  # type: ignore[attr-defined]

        # Human-readable overview of expected nested shape (subset semantics).
        expected_overview = {
            "version": 1,
            "root": {
                "tool": "ContactManager.update",
                "handle": "AsyncToolLoopHandle",
            },
            "children": [
                {
                    "tool": "ContactManager.ask",
                    "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
                },
            ],
        }
        _assert_dict_subset(expected_overview, snap)
        assert snap.get("loop_id", "").startswith("ContactManager.update")
        assert isinstance(snap.get("assistant"), list)
        assert isinstance(snap.get("tools"), list)

        # Children schema: expect at least one child and one of them must be ContactManager.ask
        children = snap.get("children") or []
        assert isinstance(children, list) and len(children) >= 1
        child = None
        for ch in children:
            if (ch or {}).get("tool") == "ContactManager.ask":
                child = ch
                break
        assert (
            child is not None
        ), "Expected a child for ContactManager.ask in recursive snapshot"

        # Assert child subset (human-readable, stable shape)
        expected_child_subset = {
            "tool": "ContactManager.ask",
            "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
        }
        _assert_dict_subset(expected_child_subset, child)

        # State must be 'in_flight' or 'done'
        assert child.get("state") in ("in_flight", "done")
        if child.get("state") == "in_flight":
            assert isinstance(
                child.get("snapshot"),
                dict,
            ), "In-flight child must include an inline snapshot"
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=120)  # type: ignore[attr-defined]
        except Exception:
            pass
