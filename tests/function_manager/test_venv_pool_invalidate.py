"""VenvPool retirement contract under credential rotation.

Credential mutations (a body rebinding a secret, an OAuth token refresh,
an admin update pushed from Orchestra) update ``os.environ``. Already
spawned venv subprocesses captured the previous environment at spawn
time, so they must be retired before the next function execution to
avoid leaking a stale credential into the function's runtime.

These tests exercise the retirement machinery as a unit: a fake
connection doubles for a real venv subprocess so the logic under test
is exactly the pool's bookkeeping (generation counter, connection
eviction, idle-vs-in-flight classification, process-wide sweep) rather
than a roundtrip through ``asyncio.create_subprocess_exec``. The
subprocess startup and shutdown paths have their own coverage in
``test_venv_persistent_connections.py``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List

import pytest

from unity.function_manager.function_manager import (
    SessionMetadata,
    VenvPool,
    _VenvConnection,
)


@dataclass
class _FakeConn:
    """A minimal stand-in for :class:`_VenvConnection` used in pool tests.

    Mirrors exactly the three surfaces :meth:`VenvPool.invalidate_all`
    reads from a pooled connection: ``_generation``, ``is_in_flight()``,
    and ``shutdown()``. The real connection owns a live subprocess; the
    fake owns a counter so a test can assert whether ``shutdown`` ran
    without paying the cost of actually spawning Python.
    """

    generation: int = 0
    in_flight: bool = False
    shutdown_calls: List[float] = field(default_factory=list)
    raise_on_shutdown: bool = False

    @property
    def _generation(self) -> int:
        return self.generation

    @_generation.setter
    def _generation(self, value: int) -> None:
        self.generation = value

    def is_in_flight(self) -> bool:
        return self.in_flight

    def is_alive(self) -> bool:
        return True

    async def shutdown(self, timeout: float = 5.0) -> None:
        self.shutdown_calls.append(timeout)
        if self.raise_on_shutdown:
            raise RuntimeError("simulated shutdown failure")


def _inject(pool: VenvPool, key: tuple[int, int], conn: _FakeConn) -> None:
    """Place ``conn`` into ``pool`` under ``key`` as if it had been spawned.

    Bypasses ``get_or_create_connection`` so the test isolates the
    retirement path. ``SessionMetadata`` is required because
    ``invalidate_all`` clears both maps and any skew between them would
    mask a regression where one map is cleared but the other isn't.
    """
    from datetime import datetime, timezone

    pool._connections[key] = conn  # type: ignore[assignment]
    now = datetime.now(timezone.utc)
    pool._metadata[key] = SessionMetadata(
        venv_id=key[0],
        session_id=key[1],
        created_at=now,
        last_used=now,
    )


# ────────────────────────── Generation counter ──────────────────────────


@pytest.mark.asyncio
async def test_generation_starts_at_zero_and_increments_monotonically():
    """A fresh pool starts at generation 0; each invalidate bumps by one."""
    pool = VenvPool()
    assert pool._generation == 0

    await pool.invalidate_all()
    assert pool._generation == 1

    await pool.invalidate_all()
    await pool.invalidate_all()
    assert pool._generation == 3


@pytest.mark.asyncio
async def test_invalidate_all_on_closed_pool_is_noop():
    """A closed pool doesn't bump its generation or touch its maps."""
    pool = VenvPool()
    pool._closed = True

    await pool.invalidate_all()

    assert pool._generation == 0


# ────────────────────────── Connection eviction ─────────────────────────


@pytest.mark.asyncio
async def test_invalidate_all_clears_connections_and_metadata():
    """After invalidation, both pool maps are empty."""
    pool = VenvPool()
    _inject(pool, (1, 0), _FakeConn())
    _inject(pool, (2, 0), _FakeConn())
    _inject(pool, (2, 1), _FakeConn())

    await pool.invalidate_all()

    assert pool._connections == {}
    assert pool._metadata == {}


@pytest.mark.asyncio
async def test_invalidate_all_shuts_down_idle_connections():
    """Every idle connection receives exactly one ``shutdown`` call."""
    pool = VenvPool()
    idle_a = _FakeConn()
    idle_b = _FakeConn()
    _inject(pool, (1, 0), idle_a)
    _inject(pool, (2, 0), idle_b)

    await pool.invalidate_all()

    assert len(idle_a.shutdown_calls) == 1
    assert len(idle_b.shutdown_calls) == 1


@pytest.mark.asyncio
async def test_invalidate_all_leaves_in_flight_subprocess_alone():
    """An in-flight connection is dropped from the pool but not shut down.

    Killing a subprocess that is currently executing a user function
    would turn a graceful credential rotation into a visible tool-call
    failure. The pool orphans the connection from its map so it cannot
    be reused, and lets the current call complete on its existing
    subprocess under the pre-rotation environment.
    """
    pool = VenvPool()
    in_flight = _FakeConn(in_flight=True)
    idle = _FakeConn()
    _inject(pool, (1, 0), in_flight)
    _inject(pool, (2, 0), idle)

    await pool.invalidate_all()

    assert in_flight.shutdown_calls == []
    assert len(idle.shutdown_calls) == 1
    assert pool._connections == {}


@pytest.mark.asyncio
async def test_invalidate_all_swallows_shutdown_failures():
    """One failing ``shutdown`` must not break the retirement of the rest.

    ``_shutdown_silently`` is what gives the pool that property; the
    test pins the contract so a future refactor cannot quietly switch
    to a raising shutdown path and regress the batch behaviour.
    """
    pool = VenvPool()
    failing = _FakeConn(raise_on_shutdown=True)
    healthy = _FakeConn()
    _inject(pool, (1, 0), failing)
    _inject(pool, (2, 0), healthy)

    await pool.invalidate_all()

    assert len(failing.shutdown_calls) == 1
    assert len(healthy.shutdown_calls) == 1
    assert pool._connections == {}


# ────────────────────────── Process-wide sweep ──────────────────────────


@pytest.mark.asyncio
async def test_new_pool_registers_in_all_pools_weakset():
    """Every constructed pool joins the process-wide registry."""
    pool = VenvPool()

    assert pool in VenvPool._all_pools


@pytest.mark.asyncio
async def test_invalidate_all_pools_sweeps_every_live_pool():
    """One classmethod call must invalidate every live pool in the process."""
    pool_a = VenvPool()
    pool_b = VenvPool()
    conn_a = _FakeConn()
    conn_b = _FakeConn()
    _inject(pool_a, (1, 0), conn_a)
    _inject(pool_b, (1, 0), conn_b)

    starting_gen_a = pool_a._generation
    starting_gen_b = pool_b._generation

    await VenvPool.invalidate_all_pools()

    assert pool_a._generation == starting_gen_a + 1
    assert pool_b._generation == starting_gen_b + 1
    assert pool_a._connections == {}
    assert pool_b._connections == {}
    assert len(conn_a.shutdown_calls) == 1
    assert len(conn_b.shutdown_calls) == 1


@pytest.mark.asyncio
async def test_invalidate_all_pools_continues_on_per_pool_failure(monkeypatch):
    """A failing pool doesn't stop the sweep from invalidating the rest.

    Uses ``return_exceptions=True`` under the hood so one pool that
    raises does not cancel the ``gather`` of its peers; this test pins
    that behaviour so a future refactor to plain ``asyncio.gather`` is
    caught immediately.
    """
    good_pool = VenvPool()
    bad_pool = VenvPool()
    good_conn = _FakeConn()
    _inject(good_pool, (1, 0), good_conn)

    async def _raising_invalidate(self: VenvPool) -> None:
        raise RuntimeError("simulated pool failure")

    monkeypatch.setattr(VenvPool, "invalidate_all", _raising_invalidate)

    starting_gen_good = good_pool._generation

    await VenvPool.invalidate_all_pools()

    assert good_pool._generation == starting_gen_good


@pytest.mark.asyncio
async def test_invalidate_all_pools_with_no_live_pools_is_noop():
    """Empty registry → no error, no work, clean return."""
    from weakref import WeakSet

    original = VenvPool._all_pools
    VenvPool._all_pools = WeakSet()
    try:
        await VenvPool.invalidate_all_pools()
    finally:
        VenvPool._all_pools = original


# ────────────────── Generation mismatch on reuse attempt ─────────────────


@pytest.mark.asyncio
async def test_stale_connection_is_evicted_on_next_get_or_create(monkeypatch):
    """A pooled connection spawned under an old generation isn't reused.

    Simulates the race a credential rotation creates: the pool has a
    live connection spawned under generation N; ``invalidate_all`` (or
    a direct field bump, here) ticks the generation to N+1; the next
    ``get_or_create_connection`` for that key must treat the pooled
    connection as stale and spawn a replacement rather than silently
    handing back a subprocess with the old ``os.environ``.
    """
    pool = VenvPool()

    stale_conn = _FakeConn(generation=0)
    _inject(pool, (1, 0), stale_conn)

    pool._generation = 1

    fresh_conn = _FakeConn(generation=1)
    create_calls: List[int] = []

    async def _fake_create(
        cls,
        *,
        venv_id: int,
        function_manager,
        timeout: float = 30.0,
    ) -> _FakeConn:
        create_calls.append(venv_id)
        return fresh_conn

    monkeypatch.setattr(_VenvConnection, "create", classmethod(_fake_create))

    result = await pool.get_or_create_connection(
        venv_id=1,
        function_manager=object(),  # type: ignore[arg-type]
        session_id=0,
    )

    # Yield so the pool's ``create_task(_shutdown_silently(stale_conn))``
    # fires before assertions run.
    await asyncio.sleep(0)

    assert result is fresh_conn
    assert create_calls == [1]
    assert pool._connections[(1, 0)] is fresh_conn
    assert fresh_conn._generation == 1
    assert len(stale_conn.shutdown_calls) == 1
