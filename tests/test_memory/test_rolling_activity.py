"""tests/test_memory/test_rolling_activity.py

Simple unit test for `MemoryManager.get_rolling_activity`.

Ensures that when **no** activity has been recorded yet, the helper
returns an *empty* string – callers can then safely omit the Historic
Activity block from prompts.
"""

from __future__ import annotations


import unify
import asyncio

from tests.helpers import _handle_project

import pytest


# ---------------------------------------------------------------------------
#  Test – empty rolling activity                                             |
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_get_rolling_activity_empty(monkeypatch):
    """`get_rolling_activity` should return an empty string with no logs."""

    from unity.memory_manager.memory_manager import MemoryManager

    # ------------------------------------------------------------------ #
    # 1.  Stub heavy helpers so instantiation is lightweight             #
    # ------------------------------------------------------------------ #
    async def _noop(self, *_, **__):
        """Async no-op used to replace `_setup_rolling_callbacks`."""

    # Avoid costly callback registration & context/field creation
    monkeypatch.setattr(MemoryManager, "_setup_rolling_callbacks", _noop, raising=True)
    monkeypatch.setattr(
        MemoryManager,
        "_ensure_rolling_context",
        lambda self: "ctx",
        raising=True,
    )

    # Ensure *no* rows are returned so the method must fall back to "empty"
    monkeypatch.setattr(unify, "get_logs", lambda *a, **kw: [], raising=True)

    # ------------------------------------------------------------------ #
    # 2.  Exercise & verify                                              #
    # ------------------------------------------------------------------ #
    mm = MemoryManager()

    assert (
        mm.get_rolling_activity() == ""
    ), "Expected empty string when no activity logged"


# ---------------------------------------------------------------------------
#  Test – single manager call populates rolling activity                     |
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_single_manager_call_populates_rolling_activity():
    """
    After *one* outgoing manager call, the MemoryManager should persist a new
    Rolling-Activity snapshot and expose it via `get_rolling_activity` (in
    *interaction* mode).
    """

    from unity.contact_manager.simulated import SimulatedContactManager
    from unity.memory_manager.memory_manager import MemoryManager
    from unity.events.event_bus import EVENT_BUS

    # 1. Set up a SimulatedContactManager that emits ManagerMethod events
    cm = SimulatedContactManager(log_events=True)

    # 2. Wire the ContactManager into a fresh MemoryManager (registers callbacks)
    mm = MemoryManager(contact_manager=cm)

    # Wait a moment so the asynchronous callback registration completes
    await asyncio.sleep(0.05)

    # Baseline – rows *before* any interaction
    initial_rows = len(unify.get_logs(context=mm._rolling_ctx, limit=100))
    assert initial_rows == 0

    # 3. Trigger ONE outgoing manager call
    handle = await cm.ask("Just say hello.")  # noqa: S106 – plain test prompt
    await handle.result()

    # Ensure the ManagerMethod event has been fully processed
    EVENT_BUS.join_published()

    # Ensure the associated rolling summary callback has been processed
    EVENT_BUS.join_callbacks()

    # 4. Verify that at least one additional RollingActivity row was created
    updated_rows = len(unify.get_logs(context=mm._rolling_ctx, limit=100))
    assert updated_rows == 1

    # 5. Interaction-based summary should now be non-empty
    summary = mm.get_rolling_activity(mode="interaction")
    assert isinstance(summary, str) and summary.strip(), "Summary should not be empty"
