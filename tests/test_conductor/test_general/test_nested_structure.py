import asyncio
import pytest

from unity.conductor.conductor import Conductor
from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_condition,
)


@pytest.mark.asyncio
@_handle_project
async def test_conductor_nested_structure_contactmanager_ask():
    """
    Verify Conductor.request → ContactManager.ask nested structure.
    """
    cond = Conductor()
    h = await cond.request("Who is the contact living in Berlin working as a designer?")

    try:
        # Wait until the Conductor loop has requested the ContactManager_ask tool
        client = getattr(h, "_client", None)  # internal test-only access
        assert (
            client is not None
        ), "Expected AsyncToolLoopHandle to expose its client for tests"
        await _wait_for_tool_request(client, "ContactManager_ask")

        # Wait for the nested ContactManager.ask handle to be adopted
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

        structure = await h.nested_structure()  # type: ignore[attr-defined]
        expected = {
            "handle": "ConductorRequestHandle(AsyncToolLoopHandle)",
            "tool": "Conductor.request",
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


@pytest.mark.asyncio
@_handle_project
async def test_conductor_nested_structure_transcriptmanager_ask():
    """
    Verify Conductor.request → TranscriptManager.ask nested structure.
    """
    cond = Conductor()
    h = await cond.request(
        "Show me the most recent message mentioning budgeting or banking.",
    )

    try:
        client = getattr(h, "_client", None)  # internal test-only access
        assert (
            client is not None
        ), "Expected AsyncToolLoopHandle to expose its client for tests"
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

        structure = await h.nested_structure()  # type: ignore[attr-defined]
        expected = {
            "handle": "ConductorRequestHandle(AsyncToolLoopHandle)",
            "tool": "Conductor.request",
            "children": [
                {
                    "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
                    "tool": "TranscriptManager.ask",
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


@pytest.mark.asyncio
@_handle_project
async def test_conductor_nested_structure_guidancemanager_ask():
    """
    Verify Conductor.request → GuidanceManager.ask nested structure.
    """
    cond = Conductor()
    h = await cond.request("Find the onboarding demo guidance.")

    try:
        client = getattr(h, "_client", None)  # internal test-only access
        assert (
            client is not None
        ), "Expected AsyncToolLoopHandle to expose its client for tests"
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

        structure = await h.nested_structure()  # type: ignore[attr-defined]
        expected = {
            "handle": "ConductorRequestHandle(AsyncToolLoopHandle)",
            "tool": "Conductor.request",
            "children": [
                {
                    "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
                    "tool": "GuidanceManager.ask",
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


@pytest.mark.asyncio
@_handle_project
async def test_conductor_nested_structure_secretmanager_ask():
    """
    Verify Conductor.request → SecretManager.ask nested structure.
    """
    cond = Conductor()
    h = await cond.request("Which secrets are currently stored?")

    try:
        client = getattr(h, "_client", None)  # internal test-only access
        assert (
            client is not None
        ), "Expected AsyncToolLoopHandle to expose its client for tests"
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

        structure = await h.nested_structure()  # type: ignore[attr-defined]
        expected = {
            "handle": "ConductorRequestHandle(AsyncToolLoopHandle)",
            "tool": "Conductor.request",
            "children": [
                {
                    "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
                    "tool": "SecretManager.ask",
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


@pytest.mark.asyncio
@_handle_project
async def test_conductor_nested_structure_taskscheduler_ask():
    """
    Verify Conductor.request → TaskScheduler.ask nested structure.
    """
    cond = Conductor()
    h = await cond.request("What tasks are scheduled for today?")

    try:
        client = getattr(h, "_client", None)  # internal test-only access
        assert (
            client is not None
        ), "Expected AsyncToolLoopHandle to expose its client for tests"
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

        structure = await h.nested_structure()  # type: ignore[attr-defined]
        expected = {
            "handle": "ConductorRequestHandle(AsyncToolLoopHandle)",
            "tool": "Conductor.request",
            "children": [
                {
                    "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
                    "tool": "TaskScheduler.ask",
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


@pytest.mark.asyncio
@_handle_project
async def test_conductor_nested_structure_websearcher_ask():
    """
    Verify Conductor.request → WebSearcher.ask nested structure.
    """
    cond = Conductor()
    h = await cond.request("What are the latest developments in retrieval for LLMs?")

    try:
        client = getattr(h, "_client", None)  # internal test-only access
        assert (
            client is not None
        ), "Expected AsyncToolLoopHandle to expose its client for tests"
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

        structure = await h.nested_structure()  # type: ignore[attr-defined]
        expected = {
            "handle": "ConductorRequestHandle(AsyncToolLoopHandle)",
            "tool": "Conductor.request",
            "children": [
                {
                    "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
                    "tool": "WebSearcher.ask",
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
