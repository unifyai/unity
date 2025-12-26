"""SteerableToolPane — deterministic runtime aggregation for steerable handles.

`SteerableToolPane` is a runtime component (no LLM loop) that aggregates:
- a registry of in-flight `SteerableToolHandle`s (with stable ids + metadata)
- a durable, append-only events log (canonical record)
- a bounded wakeup queue (delivery hint: may drop tokens when full)
- watcher tasks that stream bottom-up handle events (clarifications/notifications)
- steering APIs that fan-in top-down control (pause/resume/stop/interject/answer_clarification)

The pane is concurrency-safe (guarded by an `asyncio.Lock`) and is designed to be
used by higher-level orchestration (e.g., an Actor handle) without introducing any
additional async tool loops.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, TypedDict

from ..common.async_tool_loop import SteerableToolHandle
from ..common._async_tool.utils import maybe_await

logger = logging.getLogger(__name__)


HandleStatus = Literal[
    "running",
    "paused",
    "waiting_for_clarification",
    "completed",
    "failed",
    "stopped",
]


@dataclass(slots=True)
class HandleMetadata:
    """Metadata captured when a `SteerableToolHandle` is registered with the pane."""

    handle_id: str
    handle: SteerableToolHandle
    parent_handle_id: str | None
    origin_tool: str
    origin_step: int
    environment_namespace: str
    created_at: float
    status: HandleStatus
    capabilities: list[str]
    call_stack: str | None = None


class PaneEventOrigin(TypedDict):
    origin_tool: str
    origin_step: int
    environment_namespace: str


PaneEventType = Literal[
    "handle_registered",
    "clarification",
    "notification",
    "completed",
    "failed",
    "stopped",
    "steering_applied",
    "pane_overflow",
]


class PaneEvent(TypedDict):
    type: PaneEventType
    run_id: str
    handle_id: str
    origin: PaneEventOrigin
    ts: float
    emitted_at_step: int | None
    payload: dict[str, Any]


@dataclass(slots=True)
class BroadcastFilter:
    """Filter configuration for broadcast steering operations.

    All filters are inclusive-only (whitelist). If a field is None, it matches all
    handles. If specified, only handles matching ALL criteria are targeted.
    """

    statuses: list[HandleStatus] = field(
        default_factory=lambda: ["running", "paused", "waiting_for_clarification"],
    )
    origin_tool_prefixes: list[str] | None = None
    capabilities: list[str] | None = None
    created_after_step: int | None = None
    created_before_step: int | None = None


class SteerableToolPane:
    """Aggregates events and control for all in-flight `SteerableToolHandle`s in an Actor run.

    Responsibilities:
    - Maintain a registry (`handle_id -> HandleMetadata`) for discovered handles.
    - Maintain an append-only durable events log (canonical record).
    - Provide a bounded wakeup queue (delivery tokens) to notify a supervisor.
    - Maintain an index of pending clarifications for fast "what's blocking?" queries.
    - Run watcher tasks that surface `next_clarification()` / `next_notification()` events.
    - Provide steering methods (`interject`, `pause`, `resume`, `stop`, `answer_clarification`).

    Completion detection is best-effort:
    - Watchers are always cancelled on `cleanup()`.
    - Watchers may also be cancelled by `_cleanup_handle(...)` when the surrounding
      runtime observes handle completion (typically when `.result()` is awaited).
    - If a plan never awaits `.result()`, watchers may live until `cleanup()`.
    """

    MAX_EVENTS: int = 50_000
    WAKEUP_QUEUE_SIZE: int = 1_000

    _CRITICAL_OVERFLOW_TYPES: frozenset[PaneEventType] = frozenset(
        {
            "clarification",
            "failed",
            "stopped",
            "handle_registered",
            "steering_applied",
        },
    )

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id

        self._registry: dict[str, HandleMetadata] = {}
        self._events_log: list[PaneEvent] = []
        # Wakeup queue carries event indices; it may drop tokens (QueueFull).
        self._events_q: asyncio.Queue[int] = asyncio.Queue(
            maxsize=self.WAKEUP_QUEUE_SIZE,
        )
        self._watcher_tasks: dict[str, asyncio.Task[None]] = {}
        # Index: (handle_id, call_id) -> clarification PaneEvent
        self._pending_clarifications: dict[tuple[str, str], PaneEvent] = {}

        self._lock = asyncio.Lock()
        self._overflow_occurred = False
        self._cleanup_started = False

        logger.debug("SteerableToolPane created for run_id=%s", run_id)

    def _unknown_origin(self) -> PaneEventOrigin:
        return {
            "origin_tool": "unknown",
            "origin_step": -1,
            "environment_namespace": "unknown",
        }

    def _origin_from_meta(self, meta: HandleMetadata) -> PaneEventOrigin:
        return {
            "origin_tool": meta.origin_tool,
            "origin_step": meta.origin_step,
            "environment_namespace": meta.environment_namespace,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Event log
    # ──────────────────────────────────────────────────────────────────────────

    def _get_current_actor_step(self) -> int | None:
        """Best-effort retrieval of current Actor runtime step.

        The pane is actor-agnostic; by default this returns None. A surrounding
        runtime may extend/wrap the pane to supply a step provider.
        """

        return None

    