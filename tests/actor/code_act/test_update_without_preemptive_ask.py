"""
Verify the CodeActActor routes "check-then-save" requests directly to
``primitives.contacts.update(...)`` without a preemptive
``primitives.contacts.ask(...)`` call.

Mutation methods already inspect existing records before writing, so a
preceding read is duplicative. The prompt instructs the actor to bundle
the full intent into a single mutation call.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.actor.state_managers.utils import (
    instrument_basic_primitives_calls,
)
from unity.actor.code_act_actor import CodeActActor
from unity.actor.environments import StateManagerEnvironment
from unity.function_manager.primitives import Primitives, PrimitiveScope
from unity.manager_registry import ManagerRegistry

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


def _force_simulated_contacts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch contact manager to simulated impl (minimal scope)."""
    from unity.settings import SETTINGS

    monkeypatch.setenv("UNITY_CONTACT_IMPL", "simulated")
    monkeypatch.setattr(SETTINGS.contact, "IMPL", "simulated", raising=False)
    ManagerRegistry.clear()


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_check_then_save_goes_straight_to_update(monkeypatch):
    """A 'check if exists, then save' request should produce a single
    ``primitives.contacts.update(...)`` call — no preemptive
    ``primitives.contacts.ask(...)``."""
    _force_simulated_contacts(monkeypatch)

    scope = PrimitiveScope(scoped_managers=frozenset({"contacts"}))
    primitives = Primitives(primitive_scope=scope)
    calls = instrument_basic_primitives_calls(primitives)
    env = StateManagerEnvironment(primitives)
    actor = CodeActActor(environments=[env], timeout=200)

    handle = None
    try:
        handle = await actor.act(
            "Check if we already have a contact for Jane Doe, and if not "
            "save her email jane@example.com. Do not ask clarifying questions.",
            clarification_enabled=False,
        )

        result = await asyncio.wait_for(handle.result(), timeout=120)
        assert result is not None

        assert "primitives.contacts.update" in set(
            calls,
        ), f"Expected primitives.contacts.update to be called; saw: {calls}"

        ask_before_update = []
        update_seen = False
        for c in calls:
            if c == "primitives.contacts.update":
                update_seen = True
            elif c == "primitives.contacts.ask" and not update_seen:
                ask_before_update.append(c)

        assert not ask_before_update, (
            f"Expected NO preemptive primitives.contacts.ask before update; "
            f"full call sequence: {calls}"
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
