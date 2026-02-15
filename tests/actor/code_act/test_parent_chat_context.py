import asyncio

import pytest
from pydantic import BaseModel, Field
from unittest.mock import AsyncMock, MagicMock

from unity.actor.code_act_actor import CodeActActor
from unity.actor.execution.session import PythonExecutionSession, _PARENT_CHAT_CONTEXT
from unity.actor.environments.state_managers import StateManagerEnvironment
from unity.function_manager.primitives import Primitives
from unity.function_manager.primitives.scope import PrimitiveScope


class SecretModel(BaseModel):
    secret: int = Field(description="The secret number from context.")


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_code_act_initial_parent_chat_context_is_used():
    """CodeActActor should append _parent_chat_context before the first LLM turn."""
    SecretModel.model_rebuild()

    actor = CodeActActor(headless=True, computer_mode="mock", timeout=60)
    actor._computer_primitives.navigate = AsyncMock(return_value=None)
    actor._computer_primitives.act = AsyncMock(return_value="Action completed")
    actor._computer_primitives.observe = AsyncMock(return_value="Page content observed")

    parent_ctx = [
        {"role": "user", "content": "The secret number is 456."},
        {"role": "assistant", "content": "Acknowledged."},
    ]

    handle = await actor.act(
        "What is the secret number? Return {secret: <int>} and do not guess.",
        clarification_enabled=False,
        response_format=SecretModel,
        _parent_chat_context=parent_ctx,
        persist=False,
    )
    try:
        res = await asyncio.wait_for(handle.result(), timeout=90)
        assert isinstance(res, SecretModel)
        assert res.secret == 456
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_execute_function_forwards_parent_chat_context():
    """Parent chat context should flow from the outer act() loop through the
    execute_function tool into the sandbox via the _PARENT_CHAT_CONTEXT
    ContextVar, just like execute_code.

    Scenario: two contacts named Lucy exist. The parent conversation
    mentions "Baker" as the surname, but the act() description just says
    "Find Lucy's phone number."  We inject a spy ContactManager into a
    real Primitives instance and verify that _parent_chat_context arrives.

    execute_function now synthesises code and routes through the same
    SessionExecutor path as execute_code, so context forwarding is handled
    by PythonExecutionSession's ContextForwardingProxy wrapping.
    """
    spy = _SpyContactManager()

    prims = Primitives(
        primitive_scope=PrimitiveScope(scoped_managers=frozenset({"contacts"})),
    )
    prims._managers["contacts"] = spy

    env = StateManagerEnvironment(prims)

    actor = CodeActActor(
        environments=[env],
        headless=True,
        computer_mode="mock",
        timeout=60,
    )

    parent_ctx = [
        {
            "role": "user",
            "content": ("Can you find Lucy's number? I think her surname is Baker."),
        },
        {"role": "assistant", "content": "Sure, let me look that up for you."},
    ]

    try:
        handle = await actor.act(
            "Find Lucy's phone number from contacts.",
            can_compose=False,
            persist=False,
            clarification_enabled=False,
            _parent_chat_context=parent_ctx,
        )
        await asyncio.wait_for(handle.result(), timeout=90)

        assert len(spy.ask_calls) > 0, "primitives.contacts.ask was never called"
        assert spy.ask_calls[0]["_parent_chat_context"] is not None, (
            "primitives.contacts.ask was called without _parent_chat_context — "
            "execute_function needs to set _PARENT_CHAT_CONTEXT for the sandbox"
        )
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# Helpers for execute_code test
# ────────────────────────────────────────────────────────────────────────────


class _FakeHandle:
    """Minimal SteerableToolHandle stand-in so LLM code like
    ``handle = await primitives.contacts.ask(...); await handle.result()``
    doesn't crash."""

    def __init__(self, value: str) -> None:
        self._value = value

    async def result(self) -> str:
        return self._value

    def done(self) -> bool:
        return True


class _SpyContactManager:
    """Records calls to ask() so the test can inspect received kwargs."""

    def __init__(self) -> None:
        self.ask_calls: list[dict] = []

    async def ask(
        self,
        text: str,
        _parent_chat_context: list[dict] | None = None,
        **kwargs,
    ):
        self.ask_calls.append(
            {"text": text, "_parent_chat_context": _parent_chat_context},
        )
        return _FakeHandle("Lucy Baker: 555-0199")

    async def update(self, *args, **kwargs):
        return _FakeHandle("updated")


# ────────────────────────────────────────────────────────────────────────────
# execute_code context forwarding
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_execute_code_forwards_parent_chat_context():
    """Parent chat context should be forwarded to primitives called from
    within execute_code via ContextForwardingProxy wrapping.

    Same Lucy Baker scenario as the execute_function test, but here the LLM
    generates code that calls ``primitives.contacts.ask(...)`` directly in
    the sandbox.  We inject a spy ContactManager and assert that its ask()
    method received _parent_chat_context.
    """
    spy = _SpyContactManager()

    # Build a real Primitives instance scoped to contacts, but pre-populate
    # the manager cache with our spy so ManagerRegistry is never hit.
    prims = Primitives(
        primitive_scope=PrimitiveScope(scoped_managers=frozenset({"contacts"})),
    )
    prims._managers["contacts"] = spy

    env = StateManagerEnvironment(prims)

    actor = CodeActActor(
        environments=[env],
        headless=True,
        computer_mode="mock",
        timeout=60,
    )

    parent_ctx = [
        {
            "role": "user",
            "content": ("Can you find Lucy's number? I think her surname is Baker."),
        },
        {"role": "assistant", "content": "Sure, let me look that up for you."},
    ]

    try:
        handle = await actor.act(
            "Find Lucy's phone number from contacts.",
            persist=False,
            clarification_enabled=False,
            _parent_chat_context=parent_ctx,
        )
        await asyncio.wait_for(handle.result(), timeout=90)

        assert len(spy.ask_calls) > 0, "primitives.contacts.ask was never called"
        assert spy.ask_calls[0]["_parent_chat_context"] is not None, (
            "primitives.contacts.ask was called without _parent_chat_context — "
            "execute_code needs to wrap primitives with ContextForwardingProxy"
        )
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# Symbolic: PythonExecutionSession wraps primitives via _PARENT_CHAT_CONTEXT
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sandbox_execute_wraps_primitives_via_contextvar():
    """PythonExecutionSession.execute() should wrap global_state['primitives']
    with ContextForwardingProxy when the _PARENT_CHAT_CONTEXT ContextVar is
    set.  This is the single wrapping site that covers ALL execution paths
    (stateless, stateful session 0, persistent sessions, read-only)."""
    spy = _SpyContactManager()

    sb = PythonExecutionSession()
    sb.global_state["primitives"] = MagicMock(contacts=spy)

    ctx = [{"role": "user", "content": "Her surname is Baker"}]
    token = _PARENT_CHAT_CONTEXT.set(ctx)
    try:
        res = await sb.execute(
            'await primitives.contacts.ask(text="Lucy number?")',
        )
    finally:
        _PARENT_CHAT_CONTEXT.reset(token)

    assert res["error"] is None, f"sandbox execution failed: {res['error']}"
    assert len(spy.ask_calls) == 1
    assert spy.ask_calls[0]["_parent_chat_context"] is ctx


@pytest.mark.asyncio
async def test_sandbox_execute_no_wrap_when_contextvar_unset():
    """When _PARENT_CHAT_CONTEXT is not set (default None), the sandbox
    should NOT wrap primitives — ask() receives None."""
    spy = _SpyContactManager()

    sb = PythonExecutionSession()
    sb.global_state["primitives"] = MagicMock(contacts=spy)

    res = await sb.execute(
        'await primitives.contacts.ask(text="Lucy number?")',
    )

    assert res["error"] is None, f"sandbox execution failed: {res['error']}"
    assert len(spy.ask_calls) == 1
    assert spy.ask_calls[0]["_parent_chat_context"] is None


@pytest.mark.asyncio
async def test_sandbox_execute_restores_original_primitives():
    """After execution the original primitives object must be restored in
    global_state to avoid stacking proxies across calls."""
    spy = _SpyContactManager()
    original_prims = MagicMock(contacts=spy)

    sb = PythonExecutionSession()
    sb.global_state["primitives"] = original_prims

    ctx = [{"role": "user", "content": "context"}]
    token = _PARENT_CHAT_CONTEXT.set(ctx)
    try:
        await sb.execute('await primitives.contacts.ask(text="test")')
    finally:
        _PARENT_CHAT_CONTEXT.reset(token)

    assert (
        sb.global_state["primitives"] is original_prims
    ), "original primitives not restored after execution"
