"""
tests/conversation_manager/core/test_boot_hydration_gate.py
===========================================================

Symbolic tests for the boot hydration render gate: a slow-brain turn
requested while boot hydration is still in flight must hold until the
global thread is hydrated, so the first reply after a wake never renders
an empty view of a conversation whose history is seconds from landing.

The production shape being reproduced: converse (the session persists its
Comms events), the pod retires, a new pod boots over the same durable
world, and an inbound lands immediately — before ``hydrate_global_thread``
has restored the prior conversation. These tests rebuild that window at
the component level: the gate closed exactly as ``init_conv_manager``
closes it, the inbound stepped in, and hydration run through the real
``run_boot_hydration`` wrapper — asserting the ordering contract without
spinning full manager init a second time.
"""

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.conftest import BOSS
from unify.conversation_manager.domains.contact_index import ContactIndex
from unify.conversation_manager.domains.managers_utils import run_boot_hydration
from unify.conversation_manager.events import (
    UnifyMessageReceived,
    UnifyMessageSent,
)

PRIOR_USER_ASK = "Please file the Week 2 expense report from the spreadsheet I shared."
PRIOR_ASSISTANT_REPLY = (
    "On it — I'll file the Week 2 expenses and confirm once it's done."
)
INBOUND_AFTER_WAKE = "Did you file the Week 2 expenses yet?"


def _prior_session_bus_events():
    """The durable world a rebooted CM hydrates: last week's exchange.

    Returned newest-first, matching the real ``EventBus.search`` ordering
    ``hydrate_global_thread`` expects.
    """
    from unify.common.prompt_helpers import now as prompt_now

    base = prompt_now(as_string=False) - timedelta(days=7)
    events = [
        UnifyMessageReceived(
            contact=BOSS,
            content=PRIOR_USER_ASK,
            timestamp=base,
        ),
        UnifyMessageSent(
            contact=BOSS,
            content=PRIOR_ASSISTANT_REPLY,
            timestamp=base + timedelta(minutes=1),
        ),
    ]
    bus_events = [ev.to_bus_event() for ev in events]
    bus_events.reverse()
    return bus_events


# =============================================================================
# run_boot_hydration: the gate reopens on every hydration outcome
# =============================================================================


class TestRunBootHydration:

    def _mock_cm(self):
        """Minimal CM mid-boot: real ContactIndex, gate closed."""
        cm = MagicMock()
        cm.contact_index = ContactIndex()
        cm._hydration_gate = asyncio.Event()
        return cm

    @pytest.mark.asyncio
    async def test_gate_reopens_after_restore(self):
        cm = self._mock_cm()
        with patch(
            "unify.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=_prior_session_bus_events())
            restored = await run_boot_hydration(cm)

        assert restored == 2
        assert cm._hydration_gate.is_set()

    @pytest.mark.asyncio
    async def test_gate_reopens_when_store_is_empty(self):
        cm = self._mock_cm()
        with patch(
            "unify.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=[])
            restored = await run_boot_hydration(cm)

        assert restored == 0
        assert cm._hydration_gate.is_set()

    @pytest.mark.asyncio
    async def test_gate_reopens_when_hydration_fails(self):
        cm = self._mock_cm()
        with patch(
            "unify.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(side_effect=RuntimeError("search down"))
            with pytest.raises(RuntimeError):
                await run_boot_hydration(cm)

        assert cm._hydration_gate.is_set()


# =============================================================================
# The slow brain holds at the gate and renders the hydrated view
# =============================================================================


class _ObservedGate(asyncio.Event):
    """A hydration gate that reports when a turn arrives at the hold."""

    def __init__(self):
        super().__init__()
        self.turn_arrived = asyncio.Event()

    async def wait(self):
        self.turn_arrived.set()
        return await super().wait()


@pytest.mark.asyncio
@_handle_project
async def test_first_turn_after_wake_renders_hydrated_history(initialized_cm):
    """An inbound that lands mid-boot renders only after hydration.

    The turn is requested while the gate is closed, holds without touching
    the renderer, and — once hydration prepends the prior session and the
    real ``run_boot_hydration`` wrapper reopens the gate — renders a
    snapshot containing both the hydrated history and the new inbound.
    This is the ordering contract: the brain's first look at a rebooted
    conversation is never the empty pre-hydration view.
    """
    cm = initialized_cm
    # The booted fixture leaves the gate open — the steady-state contract
    # that every ordinary turn passes through without waiting.
    assert cm.cm._hydration_gate.is_set()

    renders = []
    real_render = cm.cm.prompt_renderer.render_state

    def recording_render(*args, **kwargs):
        snapshot = real_render(*args, **kwargs)
        renders.append(snapshot)
        return snapshot

    gate = _ObservedGate()
    cm.cm._hydration_gate = gate
    cm.cm.initialized = False
    cm.cm.prompt_renderer.render_state = recording_render
    try:
        step_task = asyncio.create_task(
            cm.step(
                UnifyMessageReceived(contact=BOSS, content=INBOUND_AFTER_WAKE),
            ),
        )

        # The turn must reach the hold without having rendered anything.
        await asyncio.wait_for(gate.turn_arrived.wait(), timeout=60)
        assert not step_task.done()
        assert renders == []

        # Hydration lands: the prior session prepends into the global
        # thread and the boot wrapper reopens the gate.
        with patch(
            "unify.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=_prior_session_bus_events())
            restored = await run_boot_hydration(cm.cm)
        assert restored == 2

        result = await asyncio.wait_for(step_task, timeout=300)
        assert result.llm_ran

        first_render = renders[0].full_render
        assert PRIOR_USER_ASK in first_render
        assert PRIOR_ASSISTANT_REPLY in first_render
        assert INBOUND_AFTER_WAKE in first_render
    finally:
        cm.cm.prompt_renderer.render_state = real_render
        cm.cm._hydration_gate = gate
        gate.set()
        cm.cm.initialized = True


@pytest.mark.asyncio
@_handle_project
async def test_held_turn_renders_eagerly_when_hydration_is_stuck(
    initialized_cm,
    monkeypatch,
):
    """A gate that never reopens degrades to the eager pre-hydration render.

    The bounded wait exists so a hung hydration costs latency, not silence:
    after ``BOOT_HYDRATION_MAX_WAIT_SECONDS`` the turn renders whatever view
    it has rather than never answering at all.
    """
    from unify.conversation_manager import conversation_manager as cm_module

    cm = initialized_cm
    monkeypatch.setattr(cm_module, "BOOT_HYDRATION_MAX_WAIT_SECONDS", 0.05)

    renders = []
    real_render = cm.cm.prompt_renderer.render_state

    def recording_render(*args, **kwargs):
        snapshot = real_render(*args, **kwargs)
        renders.append(snapshot)
        return snapshot

    stuck_gate = asyncio.Event()
    cm.cm._hydration_gate = stuck_gate
    cm.cm.initialized = False
    cm.cm.prompt_renderer.render_state = recording_render
    try:
        result = await asyncio.wait_for(
            cm.step(
                UnifyMessageReceived(contact=BOSS, content=INBOUND_AFTER_WAKE),
            ),
            timeout=300,
        )

        assert result.llm_ran
        assert len(renders) >= 1
        assert INBOUND_AFTER_WAKE in renders[0].full_render
    finally:
        cm.cm.prompt_renderer.render_state = real_render
        stuck_gate.set()
        cm.cm.initialized = True
