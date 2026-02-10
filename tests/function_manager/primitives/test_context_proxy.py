"""
Tests for ContextForwardingProxy — a transparent wrapper around Primitives
that auto-injects _parent_chat_context into method calls whose signature
accepts it.

Coverage
========
✓ Two-level proxy forwards _parent_chat_context through manager.method() access
✓ Injection is selective: skips non-accepting methods, respects explicit kwargs,
  transparent when context is None
"""

from __future__ import annotations

import pytest

from unity.function_manager.primitives.context_proxy import ContextForwardingProxy


# ────────────────────────────────────────────────────────────────────────────
# Helpers: mock manager + primitives-like object
# ────────────────────────────────────────────────────────────────────────────


class _FakeManager:
    """Simulates a state manager with mixed method signatures."""

    async def ask(
        self,
        text: str,
        _parent_chat_context: list[dict] | None = None,
    ) -> dict:
        return {"text": text, "ctx": _parent_chat_context}

    async def update(self, payload: str) -> str:
        """Method that does NOT accept _parent_chat_context."""
        return f"updated:{payload}"


class _FakePrimitives:
    """Simulates the Primitives namespace with two managers."""

    def __init__(self):
        self.contacts = _FakeManager()
        self.knowledge = _FakeManager()


# ────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_proxy_forwards_context_to_accepting_method():
    """primitives.contacts.ask(...) receives _parent_chat_context when the
    method signature accepts it."""
    ctx = [{"role": "user", "content": "Hello"}]
    proxy = ContextForwardingProxy(_FakePrimitives(), _parent_chat_context=ctx)

    result = await proxy.contacts.ask(text="Who is Alice?")

    assert result["text"] == "Who is Alice?"
    assert result["ctx"] is ctx


@pytest.mark.asyncio
async def test_proxy_injection_is_selective():
    """Exercises the edges around context injection in a single flow:
    forwarding works, non-accepting methods are unaffected, explicit
    kwargs are preserved, and a None context is fully transparent."""
    ctx = [{"role": "user", "content": "Hello"}]
    prims = _FakePrimitives()
    proxy = ContextForwardingProxy(prims, _parent_chat_context=ctx)

    # Accepting method on a different manager → still injected.
    result = await proxy.knowledge.ask(text="facts")
    assert result["ctx"] is ctx, "context not forwarded to knowledge.ask"

    # Non-accepting method → no TypeError, no injection.
    assert await proxy.contacts.update(payload="x") == "updated:x"

    # Explicit _parent_chat_context → not overwritten by the proxy.
    explicit = [{"role": "user", "content": "explicit"}]
    result2 = await proxy.contacts.ask(text="t", _parent_chat_context=explicit)
    assert result2["ctx"] is explicit, "proxy overwrote explicit _parent_chat_context"

    # None context → behaves like the unwrapped object.
    proxy_none = ContextForwardingProxy(prims, _parent_chat_context=None)
    result3 = await proxy_none.contacts.ask(text="t2")
    assert result3["ctx"] is None, "None context should not inject anything"
