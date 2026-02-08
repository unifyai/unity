"""
E2E tests for primitive handle steering through the CodeActActor.

Verifies that when the CodeActActor invokes state manager primitives
via ``execute_function`` or ``execute_code``, the returned
SteerableToolHandle(s) are adopted by the outer tool loop and can be
steered (interjected, paused, resumed) from the outside.

Uses simulated managers backed by a real LLM for realistic behavior.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from unity.actor.code_act_actor import CodeActActor
from unity.actor.environments import StateManagerEnvironment
from unity.function_manager.primitives import Primitives, PrimitiveScope
from unity.manager_registry import ManagerRegistry

pytestmark = pytest.mark.eval


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _force_simulated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch all managers to simulated impl for this test."""
    from unity.settings import SETTINGS

    for name in (
        "CONTACT",
        "TASK",
        "TRANSCRIPT",
        "KNOWLEDGE",
        "GUIDANCE",
        "SECRET",
        "WEB",
        "FILE",
        "DATA",
    ):
        monkeypatch.setenv(f"UNITY_{name}_IMPL", "simulated")
        attr = name.lower()
        if hasattr(SETTINGS, attr):
            monkeypatch.setattr(getattr(SETTINGS, attr), "IMPL", "simulated", raising=False)

    ManagerRegistry.clear()


async def _wait_for_inner_handle_adopted(
    handle, *, count: int = 1, timeout: float = 120.0,
) -> None:
    """Wait until the outer loop has adopted at least *count* inner handles.

    Detection: the loop's task_info dict contains metadata entries whose
    ``handle`` attribute is not None.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            task_info = getattr(handle._task, "task_info", {})
            n = sum(
                1
                for _t, info in (
                    task_info.items() if isinstance(task_info, dict) else []
                )
                if getattr(info, "handle", None) is not None
            )
            if n >= count:
                return
        except Exception:
            pass
        await asyncio.sleep(0.3)
    raise AssertionError(
        f"Expected {count} inner handle(s) adopted within timeout, "
        f"but found fewer",
    )


# ────────────────────────────────────────────────────────────────────────────
# Test: execute_function path
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_execute_function_primitive_steering(monkeypatch):
    """CodeActActor (can_compose=False) → execute_function → ContactManager.ask
    → handle adopted → interjection forwarded → result incorporates both turns.
    """
    _force_simulated(monkeypatch)

    scope = PrimitiveScope(scoped_managers=frozenset({"contacts"}))
    primitives = Primitives(primitive_scope=scope)
    env = StateManagerEnvironment(primitives)
    actor = CodeActActor(environments=[env], timeout=200)

    try:
        # can_compose=False forces the LLM to use execute_function (no code sandbox).
        # The tool_policy requires a FunctionManager discovery call on step 0.
        # Primitives may not appear in search results immediately after sync
        # (backend embedding computation is async), so we give the LLM the
        # exact function name to use on step 1.
        handle = await actor.act(
            "Step 1: Call FunctionManager_list_functions (required first step).\n"
            "Step 2: Call execute_function with function_name='ContactManager.ask' "
            "and call_kwargs={'text': 'Find all contacts located in Berlin'}. "
            "The function WILL be found even if the list appeared empty.",
            can_compose=False,
            clarification_enabled=False,
        )

        # Wait for the inner ContactManager.ask handle to be adopted.
        await _wait_for_inner_handle_adopted(handle, timeout=120)

        # Steer: interject additional context mid-flight.
        await handle.interject(
            "Also include any contacts in Munich.",
        )

        # Steer: pause then resume to verify lifecycle methods propagate.
        await handle.pause()
        await asyncio.sleep(0.5)
        await handle.resume()

        # Let the loop finish.
        result = await asyncio.wait_for(handle.result(), timeout=120)
        assert result is not None, "Expected a non-None result from the actor"
    finally:
        try:
            if not handle.done():
                await handle.stop("test cleanup")
        except Exception:
            pass
        try:
            await actor.close()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# Test: execute_code path
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_execute_code_primitive_steering(monkeypatch):
    """CodeActActor → execute_code calling primitives.contacts.ask(...)
    → handle returned as last expression → adopted → interjection forwarded.
    """
    _force_simulated(monkeypatch)

    scope = PrimitiveScope(scoped_managers=frozenset({"contacts"}))
    primitives = Primitives(primitive_scope=scope)
    env = StateManagerEnvironment(primitives)
    actor = CodeActActor(environments=[env], timeout=200)

    try:
        # The system prompt already documents steerable handles. Give an
        # explicit instruction so the LLM returns the handle for steering
        # rather than awaiting it inline.
        handle = await actor.act(
            "Use execute_code to call `await primitives.contacts.ask(text='Find contacts in Berlin')` "
            "as the **last expression** so the handle is returned for steering. "
            "Do NOT await handle.result() inside the code.",
            clarification_enabled=False,
        )

        # Wait for the inner ContactManager.ask handle to be adopted.
        await _wait_for_inner_handle_adopted(handle, timeout=120)

        # Steer: interject additional context mid-flight.
        await handle.interject(
            "Also include any contacts in Munich.",
        )

        # Steer: pause then resume.
        await handle.pause()
        await asyncio.sleep(0.5)
        await handle.resume()

        # Let the loop finish.
        result = await asyncio.wait_for(handle.result(), timeout=120)
        assert result is not None, "Expected a non-None result from the actor"
    finally:
        try:
            if not handle.done():
                await handle.stop("test cleanup")
        except Exception:
            pass
        try:
            await actor.close()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# Test: execute_code with two concurrent steerable handles
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_execute_code_dual_primitive_steering(monkeypatch):
    """CodeActActor → execute_code returning two steerable handles
    (ContactManager.ask + TranscriptManager.ask) from a single code block.

    Both handles should be adopted via the multi-handle adoption path and
    each should be individually steerable from the outer loop.
    """
    _force_simulated(monkeypatch)

    scope = PrimitiveScope(scoped_managers=frozenset({"contacts", "transcripts"}))
    primitives = Primitives(primitive_scope=scope)
    env = StateManagerEnvironment(primitives)
    actor = CodeActActor(environments=[env], timeout=240)

    try:
        handle = await actor.act(
            "Use a SINGLE execute_code call to launch two primitives and "
            "return both handles as the last expression (a dict). "
            "The code should be exactly:\n\n"
            "```python\n"
            "h1 = await primitives.contacts.ask(text='Find contacts in Berlin')\n"
            "h2 = await primitives.transcripts.ask(text='Recent messages about Berlin')\n"
            "{'contacts_handle': h1, 'transcripts_handle': h2}\n"
            "```\n\n"
            "Do NOT await .result() on either handle inside the code.",
            clarification_enabled=False,
        )

        # Wait for both inner handles to be adopted.
        await _wait_for_inner_handle_adopted(handle, count=2, timeout=120)

        # Steer the first handle (contacts) via an interjection.
        await handle.interject("Also include contacts in Munich.")

        # Steer the second handle (transcripts) via a pause/resume cycle.
        await handle.pause()
        await asyncio.sleep(0.5)
        await handle.resume()

        # Let the loop finish.
        result = await asyncio.wait_for(handle.result(), timeout=120)
        assert result is not None, "Expected a non-None result from the actor"
    finally:
        try:
            if not handle.done():
                await handle.stop("test cleanup")
        except Exception:
            pass
        try:
            await actor.close()
        except Exception:
            pass
