"""Tests for the SimulatedHandleMixin completion gate.

The completion gate allows tests to hold a simulated manager's handle
in an "alive" state (done() == False) until trigger_completion() is called
explicitly, eliminating timing races.
"""

from __future__ import annotations

import asyncio

import pytest

from unity.settings import SETTINGS

pytestmark = pytest.mark.llm_call

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _force_simulated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch contact and transcript managers to simulated impl."""
    for name in ("CONTACT", "TRANSCRIPT"):
        monkeypatch.setenv(f"UNITY_{name}_IMPL", "simulated")
        attr = name.lower()
        if hasattr(SETTINGS, attr):
            monkeypatch.setattr(
                getattr(SETTINGS, attr),
                "IMPL",
                "simulated",
                raising=False,
            )


# ---------------------------------------------------------------------------
# Tests: SimulatedContactManager completion gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hold_completion_false_completes_normally(monkeypatch):
    """Default hold_completion=False: handle completes as soon as result() is awaited."""
    _force_simulated(monkeypatch)
    from unity.contact_manager.simulated import SimulatedContactManager

    cm = SimulatedContactManager(hold_completion=False)
    handle = await cm.ask("Who is contact #1?")

    result = await asyncio.wait_for(handle.result(), timeout=120)
    assert result is not None
    assert handle.done()


@pytest.mark.asyncio
async def test_hold_completion_blocks_done_until_triggered(monkeypatch):
    """With hold_completion=True, done() stays False and result() blocks
    until trigger_completion() is called."""
    _force_simulated(monkeypatch)
    from unity.contact_manager.simulated import SimulatedContactManager

    cm = SimulatedContactManager(hold_completion=True)
    handle = await cm.ask("Who is contact #1?")

    # Start result() in the background — it should block on the gate.
    result_task = asyncio.create_task(handle.result())

    # Wait long enough for the LLM call to finish (uncached calls take
    # 10-30s).  The gate should keep the handle alive regardless.
    for _ in range(120):
        await asyncio.sleep(0.5)
        # Once the task is no longer "pending LLM work" but still not done,
        # the gate is holding.  We just need to verify done() stays False.
        if not handle.done():
            continue
        break

    assert not handle.done(), "Handle should NOT be done while gate is closed"
    assert not result_task.done(), "result() should be blocked by the gate"

    # Release the gate.
    handle.trigger_completion()

    # result() should now return quickly.
    result = await asyncio.wait_for(result_task, timeout=10)
    assert result is not None
    assert handle.done()


@pytest.mark.asyncio
async def test_stop_bypasses_gate(monkeypatch):
    """stop() should open the gate and allow result() to return immediately."""
    _force_simulated(monkeypatch)
    from unity.contact_manager.simulated import SimulatedContactManager

    cm = SimulatedContactManager(hold_completion=True)
    handle = await cm.ask("Who is contact #1?")

    result_task = asyncio.create_task(handle.result())

    # stop() immediately opens the gate and cancels the work.
    await handle.stop("test cleanup")

    result = await asyncio.wait_for(result_task, timeout=120)
    assert handle.done()
    # After stop, the result is a cancellation message.
    assert "stopped" in str(result).lower() or "no result" in str(result).lower()


# ---------------------------------------------------------------------------
# Tests: SimulatedActor completion gate (refactored to use shared mechanism)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_actor_hold_completion_blocks_done(monkeypatch):
    """SimulatedActor with hold_completion=True: the handle stays alive
    after the background action finishes, until trigger_completion()."""
    _force_simulated(monkeypatch)
    from unity.actor.simulated import SimulatedActor

    actor = SimulatedActor(duration=0.5, hold_completion=True)
    handle = await actor.act("Simulate a quick task")

    # The background action completes after ~0.5s duration, but the gate holds.
    await asyncio.sleep(2)
    assert not handle.done(), (
        "Handle should NOT be done while gate is closed, "
        "even though the background action finished"
    )

    # Release the gate.
    handle.trigger_completion()

    result = await asyncio.wait_for(handle.result(), timeout=10)
    assert result is not None
    assert handle.done()


@pytest.mark.asyncio
async def test_actor_trigger_completion_also_finishes_action():
    """trigger_completion() on the actor both completes the action and
    opens the gate in one call."""
    from unity.actor.simulated import SimulatedActor

    # No steps or duration → the action runs indefinitely until triggered.
    actor = SimulatedActor(hold_completion=True)
    handle = await actor.act("Run forever until told to stop")

    await asyncio.sleep(0.5)
    assert not handle.done()

    handle.trigger_completion("Custom result from trigger")

    result = await asyncio.wait_for(handle.result(), timeout=10)
    assert "Custom result from trigger" in result
    assert handle.done()


@pytest.mark.asyncio
async def test_actor_stop_bypasses_gate():
    """stop() on the actor should open the gate and complete immediately."""
    from unity.actor.simulated import SimulatedActor

    actor = SimulatedActor(hold_completion=True)
    handle = await actor.act("Run forever")

    await asyncio.sleep(0.5)
    assert not handle.done()

    await handle.stop("test cleanup")

    result = await asyncio.wait_for(handle.result(), timeout=10)
    assert handle.done()
    assert "stop" in result.lower()


# ---------------------------------------------------------------------------
# Tests: Multiple handles with hold_completion (the motivating use case)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_handles_simultaneously_alive(monkeypatch):
    """Two simulated managers with hold_completion=True: both handles stay
    alive simultaneously, enabling deterministic observation."""
    _force_simulated(monkeypatch)
    from unity.contact_manager.simulated import SimulatedContactManager
    from unity.transcript_manager.simulated import SimulatedTranscriptManager

    cm = SimulatedContactManager(hold_completion=True)
    tm = SimulatedTranscriptManager(hold_completion=True)

    h1 = await cm.ask("Find contacts in Berlin")
    h2 = await tm.ask("Recent messages about Berlin")

    # Start both result() calls in background.
    t1 = asyncio.create_task(h1.result())
    t2 = asyncio.create_task(h2.result())

    # Wait for both LLM calls to finish (uncached: 10-30s each, running
    # concurrently).  The gate should keep both handles alive regardless.
    for _ in range(120):
        await asyncio.sleep(0.5)

    # BOTH handles should be alive simultaneously (gate is holding).
    assert not h1.done(), "Contact handle should still be alive"
    assert not h2.done(), "Transcript handle should still be alive"

    # Release them one at a time.
    h1.trigger_completion()
    await asyncio.wait_for(t1, timeout=10)
    assert h1.done()
    assert not h2.done(), "Transcript handle should still be alive"

    h2.trigger_completion()
    await asyncio.wait_for(t2, timeout=10)
    assert h2.done()
