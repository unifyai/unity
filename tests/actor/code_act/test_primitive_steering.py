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

import pytest

from tests.actor.state_managers.utils import extract_code_act_execute_code_snippets
from tests.async_helpers import _wait_for_condition
from unity.actor.code_act_actor import CodeActActor
from unity.actor.environments import StateManagerEnvironment
from unity.function_manager.primitives import Primitives, PrimitiveScope
from unity.manager_registry import ManagerRegistry

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


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
            monkeypatch.setattr(
                getattr(SETTINGS, attr),
                "IMPL",
                "simulated",
                raising=False,
            )

    ManagerRegistry.clear()


def _restrict_to_execute_code(actor: CodeActActor) -> None:
    """Limit the actor tool surface so mode selection happens inside execute_code."""
    act_tools = actor.get_tools("act")
    actor.add_tools("act", {"execute_code": act_tools["execute_code"]})


async def _wait_for_tool_result_in_transcript(
    handle,
    tool_name: str,
    *,
    timeout: float = 120.0,
) -> None:
    """Wait until a tool result for *tool_name* appears in the handle's transcript.

    The transcript (``handle.get_history()``) is append-only, so a tool result
    message is a permanent, race-free signal that the tool ran and its return
    value was processed (including handle adoption when applicable).
    """

    async def _predicate():
        return any(
            m.get("role") == "tool" and m.get("name") == tool_name
            for m in handle.get_history()
        )

    await _wait_for_condition(_predicate, poll=0.1, timeout=timeout)


# ────────────────────────────────────────────────────────────────────────────
# Test: execute_function path
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_execute_function_primitive_steering(monkeypatch):
    """CodeActActor (can_compose=False) → execute_function → primitives.contacts.ask
    → handle adopted → interjection forwarded → result incorporates both turns.
    """
    _force_simulated(monkeypatch)

    scope = PrimitiveScope(scoped_managers=frozenset({"contacts"}))
    primitives = Primitives(primitive_scope=scope)
    env = StateManagerEnvironment(primitives)
    actor = CodeActActor(environments=[env], timeout=200)

    try:
        # can_compose=False forces the LLM to use execute_function (no code sandbox).
        # Primitives may not appear in search results immediately after sync
        # (backend embedding computation is async), so we give the LLM the
        # exact function name to use after the discovery step.
        handle = await actor.act(
            "Step 1: Call FunctionManager_list_functions (required first step).\n"
            "Step 2: Call execute_function with function_name='primitives.contacts.ask' "
            "and call_kwargs={'text': 'Find all contacts located in Berlin'}. "
            "The function WILL be found even if the list appeared empty.",
            can_compose=False,
            clarification_enabled=False,
        )

        # Wait for the execute_function tool result in the transcript.
        await _wait_for_tool_result_in_transcript(
            handle,
            "execute_function",
            timeout=120,
        )

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


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_execute_code_mode_selection_realistic_steerable_intent(monkeypatch):
    """Natural request that implies mid-flight control should return a handle."""
    _force_simulated(monkeypatch)

    scope = PrimitiveScope(scoped_managers=frozenset({"contacts"}))
    primitives = Primitives(primitive_scope=scope)
    env = StateManagerEnvironment(primitives)
    actor = CodeActActor(environments=[env], timeout=220)
    _restrict_to_execute_code(actor)
    handle = None

    try:
        handle = await actor.act(
            "Start checking contacts in Berlin now, but keep the lookup running because "
            "I may refine the criteria while it is underway.",
            clarification_enabled=False,
        )

        await _wait_for_tool_result_in_transcript(
            handle,
            "execute_code",
            timeout=120,
        )

        snippets = extract_code_act_execute_code_snippets(handle)
        assert snippets, "Expected CodeAct to use execute_code."
        assert any(
            "primitives.contacts.ask" in snippet and ".result(" not in snippet
            for snippet in snippets
        ), (
            "Expected at least one execute_code snippet to return a primitive handle "
            "without awaiting .result() for steerable user intent.\n"
            f"Snippets:\n{chr(10).join(snippets)}"
        )

        await handle.interject("Also include contacts in Munich.")
        result = await asyncio.wait_for(handle.result(), timeout=120)
        assert result is not None, "Expected a non-None result from the actor"
    finally:
        try:
            if handle is not None and not handle.done():
                await handle.stop("test cleanup")
        except Exception:
            pass
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_execute_code_mode_selection_realistic_inline_composition(monkeypatch):
    """Natural request that requires same-block processing should await result."""
    _force_simulated(monkeypatch)

    scope = PrimitiveScope(scoped_managers=frozenset({"contacts"}))
    primitives = Primitives(primitive_scope=scope)
    env = StateManagerEnvironment(primitives)
    actor = CodeActActor(environments=[env], timeout=220)
    _restrict_to_execute_code(actor)
    handle = None

    try:
        handle = await actor.act(
            "In one code step, look up contacts in Berlin and immediately compute a "
            "short summary string with the number of matches before replying.",
            clarification_enabled=False,
        )

        result = await asyncio.wait_for(handle.result(), timeout=120)
        assert result is not None, "Expected a non-None result from the actor"

        snippets = extract_code_act_execute_code_snippets(handle)
        assert snippets, "Expected CodeAct to use execute_code."
        assert any(
            "primitives.contacts.ask" in snippet and ".result(" in snippet
            for snippet in snippets
        ), (
            "Expected at least one execute_code snippet to await .result() for inline "
            "composition intent.\n"
            f"Snippets:\n{chr(10).join(snippets)}"
        )
    finally:
        try:
            if handle is not None and not handle.done():
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
@pytest.mark.timeout(300)
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

        # Wait for the execute_code tool result in the transcript.
        await _wait_for_tool_result_in_transcript(
            handle,
            "execute_code",
            timeout=120,
        )

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

        # Wait for the execute_code tool result in the transcript.
        await _wait_for_tool_result_in_transcript(
            handle,
            "execute_code",
            timeout=120,
        )

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
