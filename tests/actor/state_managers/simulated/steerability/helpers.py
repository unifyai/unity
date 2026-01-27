from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from tests.actor.state_managers.utils import (
    get_pane_steering_events,
    wait_for_pane_event,
)

from .timeouts import PLAN_COMPLETION_TIMEOUT


def start_canned_plan(h: Any, *, actor: Any, source: str) -> str:
    """Inject a sanitized canned plan into a handle and start execution.

    Returns the sanitized plan source so tests can assert routing-only behavior
    (i.e., `h.plan_source_code` doesn't change after steering).
    """
    h.plan_source_code = actor._sanitize_code(source, h)
    original_plan_source = h.plan_source_code
    h._execution_task = asyncio.create_task(h._initialize_and_run())
    return str(original_plan_source)


def get_ok_steering_handle_ids(
    obj: Any,
    *,
    method: str,
    n: int = 800,
) -> set[str]:
    """Return the set of handle_ids with `steering_applied(status="ok")` for `method`."""
    ok_events = [
        e
        for e in get_pane_steering_events(obj, n=n, method=method)
        if (e.get("payload") or {}).get("status") == "ok"
    ]
    return {str(e.get("handle_id")) for e in ok_events}


async def wait_for_handle_registered(
    obj: Any,
    *,
    origin_tool_prefix: str,
    timeout: float,
) -> str:
    """Wait for a `handle_registered` pane event whose origin_tool starts with `origin_tool_prefix`."""

    ev = await wait_for_pane_event(
        obj,
        predicate=lambda e: e.get("type") == "handle_registered"
        and str((e.get("origin") or {}).get("origin_tool") or "").startswith(
            origin_tool_prefix,
        ),
        timeout=timeout,
    )
    return str(ev.get("handle_id"))


async def release_gate(obj: Any, gate_name: str = "TEST_GATE") -> None:
    """Release a plan-local asyncio.Event gate and resume execution if paused."""
    h = obj  # handle-like
    gate = (getattr(h, "execution_namespace", {}) or {}).get(gate_name)
    if not isinstance(gate, asyncio.Event):
        raise AssertionError(
            f"Expected plan gate {gate_name!r} to be an asyncio.Event. Got {gate!r}. "
            "If the plan restarted unexpectedly, check routing-only assertions.",
        )
    gate.set()

    # Best-effort: ensure the handle/runtime isn't left paused by previous steering.
    with contextlib.suppress(Exception):
        await h.resume()
    with contextlib.suppress(Exception):
        h.runtime.resume()


async def finish(obj: Any, *, timeout: float = PLAN_COMPLETION_TIMEOUT) -> str:
    """Await final result with a hard timeout and basic sanity check."""
    h = obj  # handle-like
    result = await asyncio.wait_for(h.result(), timeout=timeout)
    assert isinstance(result, str) and result.strip()
    return result


async def release_gate_and_finish(
    obj: Any,
    *,
    gate_name: str = "TEST_GATE",
    timeout: float = PLAN_COMPLETION_TIMEOUT,
) -> str:
    await release_gate(obj, gate_name)
    return await finish(obj, timeout=timeout)
