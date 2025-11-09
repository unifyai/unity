import asyncio
import pytest

from unity.conductor.conductor import Conductor
from unity.common.async_tool_loop import AsyncToolLoopHandle
from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_condition,
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


# ─────────────────────────────────────────────────────────────────────────────
# Serialization tests (mirror nested structure tests; use recursive snapshots)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_conductor_serialize_contactmanager_ask():
    cond = Conductor()
    h = await cond.ask("Who is the contact living in Berlin working as a designer?")

    try:
        client = getattr(h, "_client", None)  # internal test-only access
        assert client is not None, "Expected AsyncToolLoopHandle to expose its client"
        await _wait_for_tool_request(client, "ContactManager_ask")

        async def _child_adopted():
            try:
                task_info = getattr(getattr(h, "_task", None), "task_info", {})  # type: ignore[attr-defined]
                if isinstance(task_info, dict):
                    return any(
                        getattr(meta, "name", None) == "ContactManager_ask"
                        and getattr(meta, "handle", None) is not None
                        for meta in task_info.values()
                    )
            except Exception:
                return False
            return False

        await _wait_for_condition(_child_adopted, poll=0.01, timeout=60.0)

        snap = h.serialize(recursive=True)  # type: ignore[attr-defined]

        expected = {
            "version": 1,
            "root": {
                "tool": "Conductor.ask",
                "handle": "AsyncToolLoopHandle",
            },
            "children": [
                {
                    "tool": "ContactManager.ask",
                },
            ],
        }
        _assert_dict_subset(expected, snap)
        assert snap.get("loop_id", "").startswith("Conductor.ask")
        assert isinstance(snap.get("assistant"), list)
        assert isinstance(snap.get("tools"), list)
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=120)  # type: ignore[attr-defined]
        except Exception:
            pass


@pytest.mark.asyncio
@_handle_project
async def test_conductor_serialize_transcriptmanager_ask():
    cond = Conductor()
    h = await cond.ask(
        "Show me the most recent message mentioning budgeting or banking.",
    )

    try:
        client = getattr(h, "_client", None)
        assert client is not None
        await _wait_for_tool_request(client, "TranscriptManager_ask")

        async def _child_adopted():
            try:
                task_info = getattr(getattr(h, "_task", None), "task_info", {})  # type: ignore[attr-defined]
                if isinstance(task_info, dict):
                    return any(
                        getattr(meta, "name", None) == "TranscriptManager_ask"
                        and getattr(meta, "handle", None) is not None
                        for meta in task_info.values()
                    )
            except Exception:
                return False
            return False

        await _wait_for_condition(_child_adopted, poll=0.01, timeout=60.0)

        snap = h.serialize(recursive=True)  # type: ignore[attr-defined]
        expected = {
            "version": 1,
            "root": {
                "tool": "Conductor.ask",
                "handle": "AsyncToolLoopHandle",
            },
            "children": [
                {
                    "tool": "TranscriptManager.ask",
                },
            ],
        }
        _assert_dict_subset(expected, snap)
        assert snap.get("loop_id", "").startswith("Conductor.ask")
        assert isinstance(snap.get("assistant"), list)
        assert isinstance(snap.get("tools"), list)
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=120)  # type: ignore[attr-defined]
        except Exception:
            pass


@pytest.mark.asyncio
@_handle_project
async def test_conductor_serialize_guidancemanager_ask():
    cond = Conductor()
    h = await cond.ask("Find the onboarding demo guidance.")

    try:
        client = getattr(h, "_client", None)
        assert client is not None
        await _wait_for_tool_request(client, "GuidanceManager_ask")

        async def _child_adopted():
            try:
                task_info = getattr(getattr(h, "_task", None), "task_info", {})  # type: ignore[attr-defined]
                if isinstance(task_info, dict):
                    return any(
                        getattr(meta, "name", None) == "GuidanceManager_ask"
                        and getattr(meta, "handle", None) is not None
                        for meta in task_info.values()
                    )
            except Exception:
                return False
            return False

        await _wait_for_condition(_child_adopted, poll=0.01, timeout=60.0)

        snap = h.serialize(recursive=True)  # type: ignore[attr-defined]
        expected = {
            "version": 1,
            "root": {
                "tool": "Conductor.ask",
                "handle": "AsyncToolLoopHandle",
            },
            "children": [
                {
                    "tool": "GuidanceManager.ask",
                },
            ],
        }
        _assert_dict_subset(expected, snap)
        assert snap.get("loop_id", "").startswith("Conductor.ask")
        assert isinstance(snap.get("assistant"), list)
        assert isinstance(snap.get("tools"), list)
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=120)  # type: ignore[attr-defined]
        except Exception:
            pass


@pytest.mark.asyncio
@_handle_project
async def test_conductor_serialize_secretmanager_ask():
    cond = Conductor()
    h = await cond.ask("Which secrets are currently stored?")

    try:
        client = getattr(h, "_client", None)
        assert client is not None
        await _wait_for_tool_request(client, "SecretManager_ask")

        async def _child_adopted():
            try:
                task_info = getattr(getattr(h, "_task", None), "task_info", {})  # type: ignore[attr-defined]
                if isinstance(task_info, dict):
                    return any(
                        getattr(meta, "name", None) == "SecretManager_ask"
                        and getattr(meta, "handle", None) is not None
                        for meta in task_info.values()
                    )
            except Exception:
                return False
            return False

        await _wait_for_condition(_child_adopted, poll=0.01, timeout=60.0)

        snap = h.serialize(recursive=True)  # type: ignore[attr-defined]
        expected = {
            "version": 1,
            "root": {
                "tool": "Conductor.ask",
                "handle": "AsyncToolLoopHandle",
            },
            "children": [
                {
                    "tool": "SecretManager.ask",
                },
            ],
        }
        _assert_dict_subset(expected, snap)
        assert snap.get("loop_id", "").startswith("Conductor.ask")
        assert isinstance(snap.get("assistant"), list)
        assert isinstance(snap.get("tools"), list)
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=120)  # type: ignore[attr-defined]
        except Exception:
            pass


@pytest.mark.asyncio
@_handle_project
async def test_conductor_serialize_skillmanager_ask():
    cond = Conductor()
    h = await cond.ask("What high-level skills do you have for spreadsheets and CSVs?")

    try:
        client = getattr(h, "_client", None)
        assert client is not None
        await _wait_for_tool_request(client, "SkillManager_ask")

        async def _child_adopted():
            try:
                task_info = getattr(getattr(h, "_task", None), "task_info", {})  # type: ignore[attr-defined]
                if isinstance(task_info, dict):
                    return any(
                        getattr(meta, "name", None) == "SkillManager_ask"
                        and getattr(meta, "handle", None) is not None
                        for meta in task_info.values()
                    )
            except Exception:
                return False
            return False

        await _wait_for_condition(_child_adopted, poll=0.01, timeout=60.0)

        snap = h.serialize(recursive=True)  # type: ignore[attr-defined]
        expected = {
            "version": 1,
            "root": {
                "tool": "Conductor.ask",
                "handle": "AsyncToolLoopHandle",
            },
            "children": [
                {
                    "tool": "SkillManager.ask",
                },
            ],
        }
        _assert_dict_subset(expected, snap)
        assert snap.get("loop_id", "").startswith("Conductor.ask")
        assert isinstance(snap.get("assistant"), list)
        assert isinstance(snap.get("tools"), list)
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=120)  # type: ignore[attr-defined]
        except Exception:
            pass


@pytest.mark.asyncio
@_handle_project
async def test_conductor_serialize_taskscheduler_ask():
    cond = Conductor()
    h = await cond.ask("What tasks are scheduled for today?")

    try:
        client = getattr(h, "_client", None)
        assert client is not None
        await _wait_for_tool_request(client, "TaskScheduler_ask")

        async def _child_adopted():
            try:
                task_info = getattr(getattr(h, "_task", None), "task_info", {})  # type: ignore[attr-defined]
                if isinstance(task_info, dict):
                    return any(
                        getattr(meta, "name", None) == "TaskScheduler_ask"
                        and getattr(meta, "handle", None) is not None
                        for meta in task_info.values()
                    )
            except Exception:
                return False
            return False

        await _wait_for_condition(_child_adopted, poll=0.01, timeout=60.0)

        snap = h.serialize(recursive=True)  # type: ignore[attr-defined]
        expected = {
            "version": 1,
            "root": {
                "tool": "Conductor.ask",
                "handle": "AsyncToolLoopHandle",
            },
            "children": [
                {
                    "tool": "TaskScheduler.ask",
                },
            ],
        }
        _assert_dict_subset(expected, snap)
        assert snap.get("loop_id", "").startswith("Conductor.ask")
        assert isinstance(snap.get("assistant"), list)
        assert isinstance(snap.get("tools"), list)
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=120)  # type: ignore[attr-defined]
        except Exception:
            pass


@pytest.mark.asyncio
@_handle_project
async def test_conductor_serialize_websearcher_ask():
    cond = Conductor()
    h = await cond.ask("What are the latest developments in retrieval for LLMs?")

    try:
        client = getattr(h, "_client", None)
        assert client is not None
        await _wait_for_tool_request(client, "WebSearcher_ask")

        async def _child_adopted():
            try:
                task_info = getattr(getattr(h, "_task", None), "task_info", {})  # type: ignore[attr-defined]
                if isinstance(task_info, dict):
                    return any(
                        getattr(meta, "name", None) == "WebSearcher_ask"
                        and getattr(meta, "handle", None) is not None
                        for meta in task_info.values()
                    )
            except Exception:
                return False
            return False

        await _wait_for_condition(_child_adopted, poll=0.01, timeout=60.0)

        snap = h.serialize(recursive=True)  # type: ignore[attr-defined]
        expected = {
            "version": 1,
            "root": {
                "tool": "Conductor.ask",
                "handle": "AsyncToolLoopHandle",
            },
            "children": [
                {
                    "tool": "WebSearcher.ask",
                },
            ],
        }
        _assert_dict_subset(expected, snap)
        assert snap.get("loop_id", "").startswith("Conductor.ask")
        assert isinstance(snap.get("assistant"), list)
        assert isinstance(snap.get("tools"), list)
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=120)  # type: ignore[attr-defined]
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Deserialization tests (hard-coded snapshots; verify continuation via interject)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_and_continue_conductor_ask_contactmanager():
    snap = {
        "version": 1,
        "loop_id": "Conductor.ask(static-contact)",
        "root": {"tool": "Conductor.ask", "handle": "AsyncToolLoopHandle"},
        "system_message": "You are helpful.",
        "initial_user_message": "Who is the contact living in Berlin working as a designer?",
        "assistant": [],
        "tools": [],
        "children": [
            {
                "call_id": None,
                "tool": "ContactManager.ask",
                "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
                "passthrough": False,
                "state": "done",
            },
        ],
    }

    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
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
        and m.get("role") == "system"
        and interjection_text in str(m.get("content", ""))
    ]
    assert len(seen) == 1


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_and_continue_conductor_ask_transcriptmanager():
    snap = {
        "version": 1,
        "loop_id": "Conductor.ask(static-transcript)",
        "root": {"tool": "Conductor.ask", "handle": "AsyncToolLoopHandle"},
        "system_message": "You are helpful.",
        "initial_user_message": "Show me the most recent message mentioning budgeting or banking.",
        "assistant": [],
        "tools": [],
        "children": [
            {
                "call_id": None,
                "tool": "TranscriptManager.ask",
                "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
                "passthrough": False,
                "state": "done",
            },
        ],
    }

    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    interjection_text = "Prefer compact layout"
    await resumed.interject(interjection_text)  # type: ignore[attr-defined]
    out = await asyncio.wait_for(resumed.result(), timeout=240)  # type: ignore[attr-defined]
    assert isinstance(out, str) and len(out) > 0
    hist = resumed.get_history()  # type: ignore[attr-defined]
    assert isinstance(hist, list)
    seen = [
        m
        for m in hist
        if isinstance(m, dict)
        and m.get("role") == "system"
        and interjection_text in str(m.get("content", ""))
    ]
    assert len(seen) == 1


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_and_continue_conductor_ask_guidancemanager():
    snap = {
        "version": 1,
        "loop_id": "Conductor.ask(static-guidance)",
        "root": {"tool": "Conductor.ask", "handle": "AsyncToolLoopHandle"},
        "system_message": "You are helpful.",
        "initial_user_message": "Find the onboarding demo guidance.",
        "assistant": [],
        "tools": [],
        "children": [
            {
                "call_id": None,
                "tool": "GuidanceManager.ask",
                "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
                "passthrough": False,
                "state": "done",
            },
        ],
    }

    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    interjection_text = "Please proceed"
    await resumed.interject(interjection_text)  # type: ignore[attr-defined]
    out = await asyncio.wait_for(resumed.result(), timeout=240)  # type: ignore[attr-defined]
    assert isinstance(out, str) and len(out) > 0
    hist = resumed.get_history()  # type: ignore[attr-defined]
    assert isinstance(hist, list)
    seen = [
        m
        for m in hist
        if isinstance(m, dict)
        and m.get("role") == "system"
        and interjection_text in str(m.get("content", ""))
    ]
    assert len(seen) == 1


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_and_continue_conductor_ask_secretmanager():
    snap = {
        "version": 1,
        "loop_id": "Conductor.ask(static-secret)",
        "root": {"tool": "Conductor.ask", "handle": "AsyncToolLoopHandle"},
        "system_message": "You are helpful.",
        "initial_user_message": "Which secrets are currently stored?",
        "assistant": [],
        "tools": [],
        "children": [
            {
                "call_id": None,
                "tool": "SecretManager.ask",
                "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
                "passthrough": False,
                "state": "done",
            },
        ],
    }

    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    interjection_text = "Prefer short answer"
    await resumed.interject(interjection_text)  # type: ignore[attr-defined]
    out = await asyncio.wait_for(resumed.result(), timeout=240)  # type: ignore[attr-defined]
    assert isinstance(out, str) and len(out) > 0
    hist = resumed.get_history()  # type: ignore[attr-defined]
    assert isinstance(hist, list)
    seen = [
        m
        for m in hist
        if isinstance(m, dict)
        and m.get("role") == "system"
        and interjection_text in str(m.get("content", ""))
    ]
    assert len(seen) == 1


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_and_continue_conductor_ask_skillmanager():
    snap = {
        "version": 1,
        "loop_id": "Conductor.ask(static-skill)",
        "root": {"tool": "Conductor.ask", "handle": "AsyncToolLoopHandle"},
        "system_message": "You are helpful.",
        "initial_user_message": "What high-level skills do you have for spreadsheets and CSVs?",
        "assistant": [],
        "tools": [],
        "children": [
            {
                "call_id": None,
                "tool": "SkillManager.ask",
                "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
                "passthrough": False,
                "state": "done",
            },
        ],
    }

    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    interjection_text = "Prefer bullet list"
    await resumed.interject(interjection_text)  # type: ignore[attr-defined]
    out = await asyncio.wait_for(resumed.result(), timeout=240)  # type: ignore[attr-defined]
    assert isinstance(out, str) and len(out) > 0
    hist = resumed.get_history()  # type: ignore[attr-defined]
    assert isinstance(hist, list)
    seen = [
        m
        for m in hist
        if isinstance(m, dict)
        and m.get("role") == "system"
        and interjection_text in str(m.get("content", ""))
    ]
    assert len(seen) == 1


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_and_continue_conductor_ask_taskscheduler():
    snap = {
        "version": 1,
        "loop_id": "Conductor.ask(static-task)",
        "root": {"tool": "Conductor.ask", "handle": "AsyncToolLoopHandle"},
        "system_message": "You are helpful.",
        "initial_user_message": "What tasks are scheduled for today?",
        "assistant": [],
        "tools": [],
        "children": [
            {
                "call_id": None,
                "tool": "TaskScheduler.ask",
                "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
                "passthrough": False,
                "state": "done",
            },
        ],
    }

    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    interjection_text = "Prefer overview"
    await resumed.interject(interjection_text)  # type: ignore[attr-defined]
    out = await asyncio.wait_for(resumed.result(), timeout=240)  # type: ignore[attr-defined]
    assert isinstance(out, str) and len(out) > 0
    hist = resumed.get_history()  # type: ignore[attr-defined]
    assert isinstance(hist, list)
    seen = [
        m
        for m in hist
        if isinstance(m, dict)
        and m.get("role") == "system"
        and interjection_text in str(m.get("content", ""))
    ]
    assert len(seen) == 1


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_and_continue_conductor_ask_websearcher():
    snap = {
        "version": 1,
        "loop_id": "Conductor.ask(static-web)",
        "root": {"tool": "Conductor.ask", "handle": "AsyncToolLoopHandle"},
        "system_message": "You are helpful.",
        "initial_user_message": "What are the latest developments in retrieval for LLMs?",
        "assistant": [],
        "tools": [],
        "children": [
            {
                "call_id": None,
                "tool": "WebSearcher.ask",
                "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
                "passthrough": False,
                "state": "done",
            },
        ],
    }

    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    interjection_text = "Prefer brief summary"
    await resumed.interject(interjection_text)  # type: ignore[attr-defined]
    out = await asyncio.wait_for(resumed.result(), timeout=240)  # type: ignore[attr-defined]
    assert isinstance(out, str) and len(out) > 0
    hist = resumed.get_history()  # type: ignore[attr-defined]
    assert isinstance(hist, list)
    seen = [
        m
        for m in hist
        if isinstance(m, dict)
        and m.get("role") == "system"
        and interjection_text in str(m.get("content", ""))
    ]
    assert len(seen) == 1
