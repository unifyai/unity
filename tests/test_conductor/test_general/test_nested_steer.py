import asyncio
import pytest

from unity.conductor.conductor import Conductor
from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_condition,
)


def _msg_in_user_visible_history(hist: list, msg: str) -> bool:
    for m in hist or []:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str) and c == msg:
                return True
            if isinstance(c, dict) and c.get("message") == msg:
                return True
    return False


async def _adopted_child_handle(h, tool_name: str):
    ti = getattr(h, "_task", None)  # type: ignore[attr-defined]
    task_info = getattr(ti, "task_info", {}) if ti is not None else {}
    if isinstance(task_info, dict):
        for meta in task_info.values():
            nm = getattr(meta, "name", None)
            hd = getattr(meta, "handle", None)
            if nm == tool_name and hd is not None:
                return hd
    return None


@pytest.mark.asyncio
@_handle_project
async def test_conductor_nested_steer_interject_reaches_contactmanager_ask():
    cond = Conductor()
    h = await cond.request("Who is the contact living in Berlin working as a designer?")

    try:
        client = getattr(h, "_client", None)  # internal test-only access
        assert (
            client is not None
        ), "Expected AsyncToolLoopHandle to expose its client for tests"
        await _wait_for_tool_request(client, "ContactManager_ask")

        async def _child_adopted():
            return (await _adopted_child_handle(h, "ContactManager_ask")) is not None

        await _wait_for_condition(_child_adopted, poll=0.01, timeout=60.0)

        msg = "conductor→contact nested-steer interjection"
        spec = {
            "children": [
                {
                    "tool": "ContactManager.ask",
                    "steps": [{"method": "interject", "args": msg}],
                },
            ],
        }
        await h.nested_steer(spec)  # type: ignore[attr-defined]

        child = await _adopted_child_handle(h, "ContactManager_ask")
        assert child is not None, "Expected ContactManager.ask handle to be adopted"

        async def _interjection_visible_on_inner():
            try:
                hist = getattr(child, "_user_visible_history", [])  # type: ignore[attr-defined]
                return _msg_in_user_visible_history(hist, msg)
            except Exception:
                return False

        await _wait_for_condition(
            _interjection_visible_on_inner,
            poll=0.01,
            timeout=60.0,
        )

        outer_hist = getattr(h, "_user_visible_history", [])  # type: ignore[attr-defined]
        assert not _msg_in_user_visible_history(
            outer_hist,
            msg,
        ), "Interjection should bypass the outer Conductor.ask loop"
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
async def test_conductor_nested_steer_interject_reaches_transcriptmanager_ask():
    cond = Conductor()
    h = await cond.request(
        "Show me the most recent message mentioning budgeting or banking.",
    )

    try:
        client = getattr(h, "_client", None)
        assert (
            client is not None
        ), "Expected AsyncToolLoopHandle to expose its client for tests"
        await _wait_for_tool_request(client, "TranscriptManager_ask")

        async def _child_adopted():
            return (await _adopted_child_handle(h, "TranscriptManager_ask")) is not None

        await _wait_for_condition(_child_adopted, poll=0.01, timeout=60.0)

        msg = "conductor→transcripts nested-steer interjection"
        spec = {
            "children": [
                {
                    "tool": "TranscriptManager.ask",
                    "steps": [{"method": "interject", "args": msg}],
                },
            ],
        }
        await h.nested_steer(spec)  # type: ignore[attr-defined]

        child = await _adopted_child_handle(h, "TranscriptManager_ask")
        assert child is not None, "Expected TranscriptManager.ask handle to be adopted"

        async def _interjection_visible_on_inner():
            try:
                hist = getattr(child, "_user_visible_history", [])  # type: ignore[attr-defined]
                return _msg_in_user_visible_history(hist, msg)
            except Exception:
                return False

        await _wait_for_condition(
            _interjection_visible_on_inner,
            poll=0.01,
            timeout=60.0,
        )

        outer_hist = getattr(h, "_user_visible_history", [])  # type: ignore[attr-defined]
        assert not _msg_in_user_visible_history(outer_hist, msg)
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
async def test_conductor_nested_steer_interject_reaches_guidancemanager_ask():
    cond = Conductor()
    h = await cond.request("Find the onboarding demo guidance.")

    try:
        client = getattr(h, "_client", None)
        assert (
            client is not None
        ), "Expected AsyncToolLoopHandle to expose its client for tests"
        await _wait_for_tool_request(client, "GuidanceManager_ask")

        async def _child_adopted():
            return (await _adopted_child_handle(h, "GuidanceManager_ask")) is not None

        await _wait_for_condition(_child_adopted, poll=0.01, timeout=60.0)

        msg = "conductor→guidance nested-steer interjection"
        spec = {
            "children": [
                {
                    "tool": "GuidanceManager.ask",
                    "steps": [{"method": "interject", "args": msg}],
                },
            ],
        }
        await h.nested_steer(spec)  # type: ignore[attr-defined]

        child = await _adopted_child_handle(h, "GuidanceManager_ask")
        assert child is not None, "Expected GuidanceManager.ask handle to be adopted"

        async def _interjection_visible_on_inner():
            try:
                hist = getattr(child, "_user_visible_history", [])  # type: ignore[attr-defined]
                return _msg_in_user_visible_history(hist, msg)
            except Exception:
                return False

        await _wait_for_condition(
            _interjection_visible_on_inner,
            poll=0.01,
            timeout=60.0,
        )

        outer_hist = getattr(h, "_user_visible_history", [])  # type: ignore[attr-defined]
        assert not _msg_in_user_visible_history(outer_hist, msg)
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
async def test_conductor_nested_steer_interject_reaches_secretmanager_ask():
    cond = Conductor()
    h = await cond.request("Which secrets are currently stored?")

    try:
        client = getattr(h, "_client", None)
        assert (
            client is not None
        ), "Expected AsyncToolLoopHandle to expose its client for tests"
        await _wait_for_tool_request(client, "SecretManager_ask")

        async def _child_adopted():
            return (await _adopted_child_handle(h, "SecretManager_ask")) is not None

        await _wait_for_condition(_child_adopted, poll=0.01, timeout=60.0)

        msg = "conductor→secret nested-steer interjection"
        spec = {
            "children": [
                {
                    "tool": "SecretManager.ask",
                    "steps": [{"method": "interject", "args": msg}],
                },
            ],
        }
        await h.nested_steer(spec)  # type: ignore[attr-defined]

        child = await _adopted_child_handle(h, "SecretManager_ask")
        assert child is not None, "Expected SecretManager.ask handle to be adopted"

        async def _interjection_visible_on_inner():
            try:
                hist = getattr(child, "_user_visible_history", [])  # type: ignore[attr-defined]
                return _msg_in_user_visible_history(hist, msg)
            except Exception:
                return False

        await _wait_for_condition(
            _interjection_visible_on_inner,
            poll=0.01,
            timeout=60.0,
        )

        outer_hist = getattr(h, "_user_visible_history", [])  # type: ignore[attr-defined]
        assert not _msg_in_user_visible_history(outer_hist, msg)
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
async def test_conductor_nested_steer_interject_reaches_taskscheduler_ask():
    cond = Conductor()
    h = await cond.request("What tasks are scheduled for today?")

    try:
        client = getattr(h, "_client", None)
        assert (
            client is not None
        ), "Expected AsyncToolLoopHandle to expose its client for tests"
        await _wait_for_tool_request(client, "TaskScheduler_ask")

        async def _child_adopted():
            return (await _adopted_child_handle(h, "TaskScheduler_ask")) is not None

        await _wait_for_condition(_child_adopted, poll=0.01, timeout=60.0)

        msg = "conductor→tasks nested-steer interjection"
        spec = {
            "children": [
                {
                    "tool": "TaskScheduler.ask",
                    "steps": [{"method": "interject", "args": msg}],
                },
            ],
        }
        await h.nested_steer(spec)  # type: ignore[attr-defined]

        child = await _adopted_child_handle(h, "TaskScheduler_ask")
        assert child is not None, "Expected TaskScheduler.ask handle to be adopted"

        async def _interjection_visible_on_inner():
            try:
                hist = getattr(child, "_user_visible_history", [])  # type: ignore[attr-defined]
                return _msg_in_user_visible_history(hist, msg)
            except Exception:
                return False

        await _wait_for_condition(
            _interjection_visible_on_inner,
            poll=0.01,
            timeout=60.0,
        )

        outer_hist = getattr(h, "_user_visible_history", [])  # type: ignore[attr-defined]
        assert not _msg_in_user_visible_history(outer_hist, msg)
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
async def test_conductor_nested_steer_interject_reaches_websearcher_ask():
    cond = Conductor()
    h = await cond.request("What are the latest developments in retrieval for LLMs?")

    try:
        client = getattr(h, "_client", None)
        assert (
            client is not None
        ), "Expected AsyncToolLoopHandle to expose its client for tests"
        await _wait_for_tool_request(client, "WebSearcher_ask")

        async def _child_adopted():
            return (await _adopted_child_handle(h, "WebSearcher_ask")) is not None

        await _wait_for_condition(_child_adopted, poll=0.01, timeout=60.0)

        msg = "conductor→web nested-steer interjection"
        spec = {
            "children": [
                {
                    "tool": "WebSearcher.ask",
                    "steps": [{"method": "interject", "args": msg}],
                },
            ],
        }
        await h.nested_steer(spec)  # type: ignore[attr-defined]

        child = await _adopted_child_handle(h, "WebSearcher_ask")
        assert child is not None, "Expected WebSearcher.ask handle to be adopted"

        async def _interjection_visible_on_inner():
            try:
                hist = getattr(child, "_user_visible_history", [])  # type: ignore[attr-defined]
                return _msg_in_user_visible_history(hist, msg)
            except Exception:
                return False

        await _wait_for_condition(
            _interjection_visible_on_inner,
            poll=0.01,
            timeout=60.0,
        )

        outer_hist = getattr(h, "_user_visible_history", [])  # type: ignore[attr-defined]
        assert not _msg_in_user_visible_history(outer_hist, msg)
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=120)  # type: ignore[attr-defined]
        except Exception:
            pass
