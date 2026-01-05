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

    async def _emit_event(self, event: dict[str, Any]) -> Optional[int]:
        """Append an event to the durable log and best-effort signal the wakeup queue.

        Overflow semantics:
        - The durable log has a hard cap (`MAX_EVENTS`).
        - On overflow:
          - Critical events fail-fast with `RuntimeError`.
          - `notification` may be dropped, but only after emitting a one-time
            `pane_overflow` marker (best-effort).
        - The wakeup queue may drop *tokens* but the durable log remains canonical.
        """

        async with self._lock:
            if self._cleanup_started:
                # Best-effort: once cleanup begins, ignore new events.
                return None

            # Fill in shared metadata.
            event["run_id"] = self.run_id
            event["ts"] = time.monotonic()
            event["emitted_at_step"] = self._get_current_actor_step()
            # Ensure durable log entries always conform to PaneEvent schema.
            if "payload" not in event:
                event["payload"] = {}

            # One-time overflow marker for notifications: if we're about to hit the cap,
            # reserve the last slot for `pane_overflow` (and drop this notification).
            if (
                event["type"] == "notification"
                and not self._overflow_occurred
                and len(self._events_log) == self.MAX_EVENTS - 1
            ):
                self._overflow_occurred = True
                logger.warning(
                    "Pane event log at cap-1 (cap=%s); reserving final slot for pane_overflow and dropping notifications thereafter",
                    self.MAX_EVENTS,
                )
                marker: PaneEvent = {
                    "type": "pane_overflow",
                    "run_id": self.run_id,
                    "handle_id": event["handle_id"],
                    "origin": event["origin"],
                    "ts": time.monotonic(),
                    "emitted_at_step": self._get_current_actor_step(),
                    "payload": {
                        "message": "Pane durable log reached cap; dropping notification events.",
                        "cap": self.MAX_EVENTS,
                    },
                }
                idx = len(self._events_log)
                self._events_log.append(marker)
                try:
                    self._events_q.put_nowait(idx)
                except asyncio.QueueFull:
                    logger.debug(
                        "Wakeup queue full; overflow marker idx=%s recorded in durable log",
                        idx,
                    )
                return idx

            if len(self._events_log) >= self.MAX_EVENTS:
                if event["type"] in self._CRITICAL_OVERFLOW_TYPES:
                    raise RuntimeError(
                        f"SteerableToolPane events_log overflow (cap={self.MAX_EVENTS}). "
                        f"Refusing to drop critical event type={event['type']}",
                    )

                if event["type"] == "notification":
                    if not self._overflow_occurred:
                        self._overflow_occurred = True
                        logger.warning(
                            "Pane event log overflow (cap=%s); dropping notifications thereafter",
                            self.MAX_EVENTS,
                        )
                    return None

                # Non-notification, non-critical: drop silently (should be rare).
                return None

            event_idx = len(self._events_log)
            self._events_log.append(event)  # type: ignore[arg-type]

            try:
                self._events_q.put_nowait(event_idx)
            except asyncio.QueueFull:
                logger.debug(
                    "Wakeup queue full; event idx=%s recorded in durable log",
                    event_idx,
                )

            return event_idx

    # ──────────────────────────────────────────────────────────────────────────
    # Introspection
    # ──────────────────────────────────────────────────────────────────────────

    async def list_handles(
        self,
        status: HandleStatus | None = None,
    ) -> list[dict[str, Any]]:
        """List registered handles (metadata only; never returns live handle objects)."""

        async with self._lock:
            metas = list(self._registry.values())

        if status is not None:
            metas = [m for m in metas if m.status == status]

        return [
            {
                "handle_id": m.handle_id,
                "parent_handle_id": m.parent_handle_id,
                "origin_tool": m.origin_tool,
                "origin_step": m.origin_step,
                "environment_namespace": m.environment_namespace,
                "created_at": m.created_at,
                "status": m.status,
                "capabilities": list(m.capabilities),
                "call_stack": m.call_stack,
            }
            for m in metas
        ]

    def get_recent_events(self, n: int = 50) -> list[PaneEvent]:
        """Return the last `n` events from the durable log."""

        if n <= 0:
            return []
        return self._events_log[-n:]

    async def get_pending_clarifications(self) -> list[PaneEvent]:
        """Return all pending clarification events currently indexed by the pane."""

        async with self._lock:
            return list(self._pending_clarifications.values())

    # ──────────────────────────────────────────────────────────────────────────
    # Watcher infrastructure
    # ──────────────────────────────────────────────────────────────────────────

    async def register_handle(
        self,
        *,
        handle: SteerableToolHandle,
        handle_id: str,
        parent_handle_id: str | None,
        origin_tool: str,
        origin_step: int,
        environment_namespace: str,
        capabilities: list[str],
        call_stack: str | None = None,
    ) -> None:
        """Register a handle and start its watcher task.

        This method:
        - Stores handle metadata in the registry.
        - Emits a `handle_registered` event.
        - Spawns a background watcher task that listens for clarifications and
          notifications via `next_clarification()` / `next_notification()`.
        """

        async with self._lock:
            if self._cleanup_started:
                logger.warning(
                    "Pane cleanup started; ignoring registration for handle_id=%s",
                    handle_id,
                )
                return

            if handle_id in self._registry:
                # Treat duplicates as an idempotent no-op (callers should provide stable IDs).
                logger.debug("Handle already registered: handle_id=%s", handle_id)
                return

            meta = HandleMetadata(
                handle_id=handle_id,
                handle=handle,
                parent_handle_id=parent_handle_id,
                origin_tool=origin_tool,
                origin_step=origin_step,
                environment_namespace=environment_namespace,
                created_at=time.monotonic(),
                status="running",
                capabilities=list(capabilities),
                call_stack=call_stack,
            )
            self._registry[handle_id] = meta

            # Spawn watcher task (event-driven; no polling).
            self._watcher_tasks[handle_id] = asyncio.create_task(
                self._watch_handle(handle_id, handle),
                name=f"pane_watcher_{handle_id[:8]}",
            )

        await self._emit_event(
            {
                "type": "handle_registered",
                "handle_id": handle_id,
                "origin": {
                    "origin_tool": origin_tool,
                    "origin_step": origin_step,
                    "environment_namespace": environment_namespace,
                },
                "payload": {
                    "capabilities": list(capabilities),
                    "parent_handle_id": parent_handle_id,
                },
            },
        )

        logger.debug(
            "Registered handle_id=%s origin_tool=%s (watcher started)",
            handle_id,
            origin_tool,
        )

    async def _watch_handle(self, handle_id: str, handle: SteerableToolHandle) -> None:
        """Watch a handle for bottom-up events (clarifications/notifications).

        This watcher is purely event-driven: it races `next_clarification()` vs
        `next_notification()` and processes whichever completes first.
        """

        clar_task: asyncio.Task | None = None
        notif_task: asyncio.Task | None = None

        try:
            while True:
                clar_task = asyncio.create_task(
                    maybe_await(handle.next_clarification()),
                    name=f"pane_clar_{handle_id[:8]}",
                )
                notif_task = asyncio.create_task(
                    maybe_await(handle.next_notification()),
                    name=f"pane_notif_{handle_id[:8]}",
                )

                done, pending = await asyncio.wait(
                    {clar_task, notif_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Process the first completed event. IMPORTANT:
                # Some handle implementations (eg: simulated ones) incorrectly return `{}` immediately rather than
                # blocking until an event exists. Treat empty dicts as "no event" and then
                # fall back to awaiting the *other* channel instead of cancelling it.
                processed = False
                t_first = next(iter(done))

                if t_first is clar_task:
                    clar = await t_first
                    if isinstance(clar, dict):
                        if clar:
                            await self._handle_clarification(handle_id, clar)
                            processed = True
                    if not processed:
                        # Await notification as the blocking source of truth.
                        notif = await notif_task
                        if isinstance(notif, dict):
                            if notif:
                                await self._handle_notification(handle_id, notif)
                                processed = True
                else:
                    notif = await t_first
                    if isinstance(notif, dict):
                        if notif:
                            await self._handle_notification(handle_id, notif)
                            processed = True
                    if not processed:
                        # Await clarification as the blocking source of truth.
                        clar = await clar_task
                        if isinstance(clar, dict):
                            if clar:
                                await self._handle_clarification(handle_id, clar)
                                processed = True

                # If neither channel produced a real event, this handle likely doesn't support
                # bottom-up streaming correctly (returns `{}` immediately). Exit watcher to
                # avoid a tight-loop flood; control-plane steering still works via pane methods.
                if not processed:
                    return

                # Cancel whichever task (if any) is still pending, now that we've consumed the needed one.
                for t in (clar_task, notif_task):
                    if t.done():
                        continue
                    t.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await t

        except asyncio.CancelledError:
            logger.debug("Watcher cancelled for handle_id=%s", handle_id)
            raise
        except Exception as e:
            logger.error(
                "Watcher failed for handle_id=%s (%s): %s",
                handle_id,
                type(e).__name__,
                str(e),
            )
            origin: PaneEventOrigin = {
                "origin_tool": "unknown",
                "origin_step": -1,
                "environment_namespace": "unknown",
            }
            async with self._lock:
                meta = self._registry.get(handle_id)
                if meta is not None:
                    meta.status = "failed"
                    origin = {
                        "origin_tool": meta.origin_tool,
                        "origin_step": meta.origin_step,
                        "environment_namespace": meta.environment_namespace,
                    }

            await self._emit_event(
                {
                    "type": "failed",
                    "handle_id": handle_id,
                    "origin": origin,
                    "payload": {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                },
            )
        finally:
            # Best-effort cleanup for any leftover inner tasks.
            for t in (clar_task, notif_task):
                if t is None:
                    continue
                if not t.done():
                    t.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t

    async def _handle_clarification(self, handle_id: str, clar: dict[str, Any]) -> None:
        call_id = str(clar.get("call_id", "unknown"))
        tool_name = str(clar.get("tool_name", "unknown"))
        question = str(clar.get("question", ""))

        async with self._lock:
            meta = self._registry.get(handle_id)
            if meta is None:
                logger.warning("Clarification for unknown handle_id=%s", handle_id)
                origin: PaneEventOrigin = {
                    "origin_tool": "unknown",
                    "origin_step": -1,
                    "environment_namespace": "unknown",
                }
            else:
                meta.status = "waiting_for_clarification"
                origin = {
                    "origin_tool": meta.origin_tool,
                    "origin_step": meta.origin_step,
                    "environment_namespace": meta.environment_namespace,
                }

        event: dict[str, Any] = {
            "type": "clarification",
            "handle_id": handle_id,
            "origin": origin,
            "payload": {
                "call_id": call_id,
                "tool_name": tool_name,
                "question": question,
            },
        }
        idx = await self._emit_event(event)

        if idx is None:
            return

        async with self._lock:
            # Index for fast lookup / "what's blocking?" queries.
            # Store the *durable* log entry (with run_id/ts/emitted_at_step populated).
            self._pending_clarifications[(handle_id, call_id)] = self._events_log[idx]

    async def _handle_notification(self, handle_id: str, notif: dict[str, Any]) -> None:
        async with self._lock:
            meta = self._registry.get(handle_id)
            if meta is None:
                logger.warning("Notification for unknown handle_id=%s", handle_id)
                origin: PaneEventOrigin = {
                    "origin_tool": "unknown",
                    "origin_step": -1,
                    "environment_namespace": "unknown",
                }
            else:
                origin = {
                    "origin_tool": meta.origin_tool,
                    "origin_step": meta.origin_step,
                    "environment_namespace": meta.environment_namespace,
                }

        await self._emit_event(
            {
                "type": "notification",
                "handle_id": handle_id,
                "origin": origin,
                "payload": notif,
            },
        )

    async def _cleanup_handle(
        self,
        handle_id: str,
        *,
        emit_completed: bool = True,
    ) -> None:
        """Cancel a specific watcher task and best-effort mark handle completed."""

        task: asyncio.Task[None] | None = None
        async with self._lock:
            task = self._watcher_tasks.pop(handle_id, None)

        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(
                    "Exception while cancelling watcher for handle_id=%s: %s",
                    handle_id,
                    e,
                )

        origin: PaneEventOrigin | None = None
        async with self._lock:
            meta = self._registry.get(handle_id)
            if meta is not None:
                if meta.status not in ("completed", "failed", "stopped"):
                    meta.status = "completed"
                origin = {
                    "origin_tool": meta.origin_tool,
                    "origin_step": meta.origin_step,
                    "environment_namespace": meta.environment_namespace,
                }

            # Remove any pending clarifications for this handle.
            for key in [
                k for k in self._pending_clarifications.keys() if k[0] == handle_id
            ]:
                self._pending_clarifications.pop(key, None)

        if emit_completed and origin is not None:
            await self._emit_event(
                {
                    "type": "completed",
                    "handle_id": handle_id,
                    "origin": origin,
                    "payload": {},
                },
            )

    async def interject(
        self,
        handle_id: str,
        message: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
        images: list | dict | None = None,
    ) -> Optional[str]:
        """Interject into a specific handle.

        Safe no-op for terminal handles (`completed`, `failed`, `stopped`), but still
        records a `steering_applied` event with status `no-op`.
        """

        truncated = message[:200]
        no_op_event: dict[str, Any] | None = None
        async with self._lock:
            meta = self._registry.get(handle_id)
            if meta is None:
                no_op_event = {
                    "type": "steering_applied",
                    "handle_id": handle_id,
                    "origin": self._unknown_origin(),
                    "payload": {
                        "method": "interject",
                        "args": {"message": truncated},
                        "status": "no-op",
                        "reason": "handle not found",
                    },
                }
                handle = None
                origin = self._unknown_origin()
            else:
                origin = self._origin_from_meta(meta)
                if meta.status in ("completed", "failed", "stopped"):
                    no_op_event = {
                        "type": "steering_applied",
                        "handle_id": handle_id,
                        "origin": origin,
                        "payload": {
                            "method": "interject",
                            "args": {"message": truncated},
                            "status": "no-op",
                            "reason": f"handle already {meta.status}",
                        },
                    }
                    handle = None
                else:
                    handle = meta.handle

        if no_op_event is not None:
            await self._emit_event(no_op_event)
            return None

        try:
            result = await maybe_await(
                handle.interject(
                    message,
                    parent_chat_context_cont=parent_chat_context_cont,
                    images=images,
                ),
            )
            await self._emit_event(
                {
                    "type": "steering_applied",
                    "handle_id": handle_id,
                    "origin": origin,
                    "payload": {
                        "method": "interject",
                        "args": {"message": truncated},
                        "status": "ok",
                    },
                },
            )
            return result
        except Exception as e:
            logger.error(
                "interject failed for handle_id=%s: %s",
                handle_id,
                e,
                exc_info=True,
            )
            await self._emit_event(
                {
                    "type": "steering_applied",
                    "handle_id": handle_id,
                    "origin": origin,
                    "payload": {
                        "method": "interject",
                        "args": {"message": truncated},
                        "status": "error",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                },
            )
            return None

    async def pause(self, handle_id: str) -> Optional[str]:
        """Pause a specific handle (safe no-op for terminal handles)."""

        no_op_event: dict[str, Any] | None = None
        async with self._lock:
            meta = self._registry.get(handle_id)
            if meta is None:
                origin = self._unknown_origin()
                no_op_event = {
                    "type": "steering_applied",
                    "handle_id": handle_id,
                    "origin": origin,
                    "payload": {
                        "method": "pause",
                        "status": "no-op",
                        "reason": "handle not found",
                    },
                }
                handle = None
            else:
                origin = self._origin_from_meta(meta)
                if meta.status in ("completed", "failed", "stopped"):
                    no_op_event = {
                        "type": "steering_applied",
                        "handle_id": handle_id,
                        "origin": origin,
                        "payload": {
                            "method": "pause",
                            "status": "no-op",
                            "reason": f"handle already {meta.status}",
                        },
                    }
                    handle = None
                else:
                    handle = meta.handle

        if no_op_event is not None:
            await self._emit_event(no_op_event)
            return None

        try:
            result = await maybe_await(handle.pause())
            async with self._lock:
                meta2 = self._registry.get(handle_id)
                if meta2 is not None and meta2.status not in (
                    "completed",
                    "failed",
                    "stopped",
                ):
                    meta2.status = "paused"
            await self._emit_event(
                {
                    "type": "steering_applied",
                    "handle_id": handle_id,
                    "origin": origin,
                    "payload": {"method": "pause", "status": "ok"},
                },
            )
            return result
        except Exception as e:
            logger.error(
                "pause failed for handle_id=%s: %s",
                handle_id,
                e,
                exc_info=True,
            )
            await self._emit_event(
                {
                    "type": "steering_applied",
                    "handle_id": handle_id,
                    "origin": origin,
                    "payload": {
                        "method": "pause",
                        "status": "error",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                },
            )
            return None

    async def resume(self, handle_id: str) -> Optional[str]:
        """Resume a specific handle (safe no-op for terminal handles)."""

        no_op_event: dict[str, Any] | None = None
        async with self._lock:
            meta = self._registry.get(handle_id)
            if meta is None:
                origin = self._unknown_origin()
                no_op_event = {
                    "type": "steering_applied",
                    "handle_id": handle_id,
                    "origin": origin,
                    "payload": {
                        "method": "resume",
                        "status": "no-op",
                        "reason": "handle not found",
                    },
                }
                handle = None
            else:
                origin = self._origin_from_meta(meta)
                if meta.status in ("completed", "failed", "stopped"):
                    no_op_event = {
                        "type": "steering_applied",
                        "handle_id": handle_id,
                        "origin": origin,
                        "payload": {
                            "method": "resume",
                            "status": "no-op",
                            "reason": f"handle already {meta.status}",
                        },
                    }
                    handle = None
                else:
                    handle = meta.handle

        if no_op_event is not None:
            await self._emit_event(no_op_event)
            return None

        try:
            result = await maybe_await(handle.resume())
            async with self._lock:
                meta2 = self._registry.get(handle_id)
                if meta2 is not None and meta2.status not in (
                    "completed",
                    "failed",
                    "stopped",
                ):
                    meta2.status = "running"
            await self._emit_event(
                {
                    "type": "steering_applied",
                    "handle_id": handle_id,
                    "origin": origin,
                    "payload": {"method": "resume", "status": "ok"},
                },
            )
            return result
        except Exception as e:
            logger.error(
                "resume failed for handle_id=%s: %s",
                handle_id,
                e,
                exc_info=True,
            )
            await self._emit_event(
                {
                    "type": "steering_applied",
                    "handle_id": handle_id,
                    "origin": origin,
                    "payload": {
                        "method": "resume",
                        "status": "error",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                },
            )
            return None

    async def stop(
        self,
        handle_id: str,
        reason: str | None = None,
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ) -> Optional[str]:
        """Stop a specific handle (safe no-op for terminal handles)."""

        truncated_reason = (reason or "")[:200]
        no_op_event: dict[str, Any] | None = None
        async with self._lock:
            meta = self._registry.get(handle_id)
            if meta is None:
                origin = self._unknown_origin()
                no_op_event = {
                    "type": "steering_applied",
                    "handle_id": handle_id,
                    "origin": origin,
                    "payload": {
                        "method": "stop",
                        "args": {"reason": truncated_reason},
                        "status": "no-op",
                        "reason": "handle not found",
                    },
                }
                handle = None
            else:
                origin = self._origin_from_meta(meta)
                if meta.status in ("completed", "failed", "stopped"):
                    no_op_event = {
                        "type": "steering_applied",
                        "handle_id": handle_id,
                        "origin": origin,
                        "payload": {
                            "method": "stop",
                            "args": {"reason": truncated_reason},
                            "status": "no-op",
                            "reason": f"handle already {meta.status}",
                        },
                    }
                    handle = None
                else:
                    handle = meta.handle

        if no_op_event is not None:
            await self._emit_event(no_op_event)
            return None

        try:
            result = await maybe_await(
                handle.stop(reason, parent_chat_context_cont=parent_chat_context_cont),
            )
            async with self._lock:
                meta2 = self._registry.get(handle_id)
                if meta2 is not None:
                    meta2.status = "stopped"
            await self._emit_event(
                {
                    "type": "steering_applied",
                    "handle_id": handle_id,
                    "origin": origin,
                    "payload": {
                        "method": "stop",
                        "args": {"reason": truncated_reason},
                        "status": "ok",
                    },
                },
            )
            # After stopping, cancel watcher and clean indices (do not emit completed).
            await self._cleanup_handle(handle_id, emit_completed=False)
            return result
        except Exception as e:
            logger.error(
                "stop failed for handle_id=%s: %s",
                handle_id,
                e,
                exc_info=True,
            )
            await self._emit_event(
                {
                    "type": "steering_applied",
                    "handle_id": handle_id,
                    "origin": origin,
                    "payload": {
                        "method": "stop",
                        "args": {"reason": truncated_reason},
                        "status": "error",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                },
            )
            return None

    async def broadcast_interject(
        self,
        message: str,
        *,
        filter: BroadcastFilter | None = None,
        origin_tool_prefixes: list[str] | None = None,
        parent_chat_context_cont: list[dict] | None = None,
        images: list | dict | None = None,
    ) -> dict[str, Any]:
        """Broadcast an interjection to a filtered set of in-flight handles."""

        # Back-compat: allow callers to pass just origin_tool_prefixes.
        if filter is None:
            filter = BroadcastFilter(origin_tool_prefixes=origin_tool_prefixes)
        elif origin_tool_prefixes is not None and filter.origin_tool_prefixes is None:
            filter.origin_tool_prefixes = origin_tool_prefixes

        async with self._lock:
            metas = list(self._registry.values())

        targets: list[str] = []
        for m in metas:
            if m.status not in (filter.statuses or []):
                continue
            if filter.origin_tool_prefixes is not None and not any(
                m.origin_tool.startswith(p) for p in filter.origin_tool_prefixes
            ):
                continue
            if filter.capabilities is not None and not all(
                cap in m.capabilities for cap in filter.capabilities
            ):
                continue
            if (
                filter.created_after_step is not None
                and m.origin_step <= filter.created_after_step
            ):
                continue
            if (
                filter.created_before_step is not None
                and m.origin_step >= filter.created_before_step
            ):
                continue
            targets.append(m.handle_id)

        results: dict[str, Any] = {}
        for hid in targets:
            results[hid] = await self.interject(
                hid,
                message,
                parent_chat_context_cont=parent_chat_context_cont,
                images=images,
            )

        return {"targets": targets, "count": len(targets), "results": results}

    async def answer_clarification(
        self,
        handle_id: str,
        call_id: str,
        answer: str,
    ) -> None:
        """Answer a clarification for a specific handle (safe no-op if not pending)."""

        key = (handle_id, call_id)
        truncated = answer[:200]

        no_op_event: dict[str, Any] | None = None
        async with self._lock:
            meta = self._registry.get(handle_id)
            origin = (
                self._origin_from_meta(meta)
                if meta is not None
                else self._unknown_origin()
            )
            pending = key in self._pending_clarifications
            if meta is None:
                no_op_event = {
                    "type": "steering_applied",
                    "handle_id": handle_id,
                    "origin": origin,
                    "payload": {
                        "method": "answer_clarification",
                        "args": {"call_id": call_id, "answer": truncated},
                        "status": "no-op",
                        "reason": "handle not found",
                    },
                }
                handle = None
            elif not pending:
                no_op_event = {
                    "type": "steering_applied",
                    "handle_id": handle_id,
                    "origin": origin,
                    "payload": {
                        "method": "answer_clarification",
                        "args": {"call_id": call_id, "answer": truncated},
                        "status": "no-op",
                        "reason": "no pending clarification",
                    },
                }
                handle = None
            else:
                handle = meta.handle

        if no_op_event is not None:
            await self._emit_event(no_op_event)
            return

        try:
            await maybe_await(handle.answer_clarification(call_id, answer))
            async with self._lock:
                self._pending_clarifications.pop(key, None)
                meta2 = self._registry.get(handle_id)
                if meta2 is not None and meta2.status == "waiting_for_clarification":
                    meta2.status = "running"
            await self._emit_event(
                {
                    "type": "steering_applied",
                    "handle_id": handle_id,
                    "origin": origin,
                    "payload": {
                        "method": "answer_clarification",
                        "args": {"call_id": call_id, "answer": truncated},
                        "status": "ok",
                    },
                },
            )
        except Exception as e:
            logger.error(
                "answer_clarification failed for handle_id=%s call_id=%s: %s",
                handle_id,
                call_id,
                e,
                exc_info=True,
            )
            await self._emit_event(
                {
                    "type": "steering_applied",
                    "handle_id": handle_id,
                    "origin": origin,
                    "payload": {
                        "method": "answer_clarification",
                        "args": {"call_id": call_id, "answer": truncated},
                        "status": "error",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                },
            )

    async def cleanup(self) -> None:
        """Cancel watcher tasks and prevent further registrations."""

        async with self._lock:
            if self._cleanup_started:
                return
            self._cleanup_started = True
            tasks = list(self._watcher_tasks.values())
            self._watcher_tasks.clear()

        for t in tasks:
            t.cancel()

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception) and not isinstance(
                    r,
                    asyncio.CancelledError,
                ):
                    logger.warning("Exception during pane watcher cleanup: %s", r)
