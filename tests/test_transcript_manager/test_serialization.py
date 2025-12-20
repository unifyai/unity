import asyncio
import pytest

from unity.transcript_manager.transcript_manager import TranscriptManager
from tests.helpers import _handle_project
from unity.common.async_tool_loop import AsyncToolLoopHandle


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
async def test_serialize_flat_ask():
    """
    Verify a flat TranscriptManager.ask snapshot contains the expected minimal, human-readable shape.
    """
    tm = TranscriptManager()
    h = await tm.ask("Show me the most recent message mentioning budgeting or banking.")

    try:
        snap = h.serialize()  # type: ignore[attr-defined]

        expected = {
            "version": 1,
            "root": {
                "tool": "TranscriptManager.ask",
                "handle": "AsyncToolLoopHandle",
            },
        }
        _assert_dict_subset(expected, snap)
        assert snap.get("loop_id", "").startswith(
            "TranscriptManager.ask",
        ), "loop_id must start with TranscriptManager.ask"
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
async def test_deserialize_and_continue_ask():
    """
    Start from a flat TranscriptManager.ask snapshot, resume, add an interjection, and verify completion and transcript.
    """
    snap = {
        "version": 1,
        "loop_id": "TranscriptManager.ask(static)",
        "root": {"tool": "TranscriptManager.ask", "handle": "AsyncToolLoopHandle"},
        "system_message": "You are helpful.",
        "initial_user_message": "Show me the most recent message mentioning budgeting or banking.",
        "assistant": [],
        "tools": [],
    }

    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)  # type: ignore[assignment]
    interjection_text = "Prefer concise output"
    await resumed.interject(interjection_text)  # type: ignore[attr-defined]
    out = await asyncio.wait_for(resumed.result(), timeout=240)  # type: ignore[attr-defined]
    assert isinstance(out, str) and len(out) > 0

    hist = resumed.get_history()  # type: ignore[attr-defined]
    assert isinstance(hist, list)
    seen = [
        m
        for m in hist
        if isinstance(m, dict)
        and m.get("role") == "user"
        and interjection_text in str(m.get("content", ""))
    ]
    assert len(seen) == 1
