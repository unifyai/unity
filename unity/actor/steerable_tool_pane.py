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
