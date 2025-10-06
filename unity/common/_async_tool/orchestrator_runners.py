from __future__ import annotations

import asyncio
import inspect
from typing import Callable

from .messages import generate_with_preprocess as _gwp
from .utils import maybe_await
from .orchestrator_events import (
    LLMCompletedEvent,
    ToolCompletedEvent,
)


class LLMRunner:
    """Child runner for LLM turns that posts events instead of bubbling cancels.

    Skeleton only – wired but not invoked yet. Intended usage:
        runner = LLMRunner(orch)
        runner.schedule_generate({"tools": [...], ...})
    """

    def __init__(self, orchestrator: "Orchestrator") -> None:
        self._orch = orchestrator

    def schedule_generate(self, gen_kwargs: dict) -> None:
        if self._orch._tg is None:
            raise RuntimeError("TaskGroup not initialized")

        async def _task():
            try:
                msg = await _gwp(
                    self._orch.client,
                    self._orch.preprocess_msgs,
                    **gen_kwargs,
                )
                evt: LLMCompletedEvent = {"type": "llm_completed", "message": msg}
                await self._orch.events.put(evt)
            except asyncio.CancelledError:
                # Treat as preemption; do NOT re-raise (prevent orchestrator cancel)
                await self._orch.events.put({"type": "llm_preempted"})
            except Exception as e:
                await self._orch.events.put({"type": "llm_failed", "error": str(e)})

        child = self._orch._tg.create_task(_task())
        self._orch._register_child(child)


class ToolRunner:
    """Child runner for base tool executions that posts completion/failure events.

    Skeleton only – not yet integrated with ToolsData scheduling or placeholders.
    """

    def __init__(self, orchestrator: "Orchestrator") -> None:
        self._orch = orchestrator

    def schedule_tool(self, name: str, call_id: str, fn: Callable, *a, **kw) -> None:
        if self._orch._tg is None:
            raise RuntimeError("TaskGroup not initialized")

        # Inject supported, safe kwargs (do not provide clarification queues yet)
        try:
            sig = inspect.signature(fn)
            params = sig.parameters
        except Exception:
            params = {}

        # Optional pause_event support
        if "pause_event" in params and "pause_event" not in kw:
            kw["pause_event"] = self._orch.pause_event

        # Optional notification queue: attach and surface events upward
        notification_up_q = None
        if "notification_up_q" in params and "notification_up_q" not in kw:
            notification_up_q = asyncio.Queue()
            kw["notification_up_q"] = notification_up_q

        # Clarification queues: when supported, create and bridge to orchestrator events
        clar_up_q = None
        clar_down_q = None
        if ("clarification_up_q" in params and "clarification_down_q" in params) and (
            "clarification_up_q" not in kw and "clarification_down_q" not in kw
        ):
            clar_up_q = asyncio.Queue()
            clar_down_q = asyncio.Queue()
            kw["clarification_up_q"] = clar_up_q
            kw["clarification_down_q"] = clar_down_q
            # Register down queue so clarify_* answers can be routed
            self._orch._clar_down[call_id] = clar_down_q

        async def _task():
            try:
                res = await maybe_await(fn(*a, **kw))
                evt: ToolCompletedEvent = {
                    "type": "tool_completed",
                    "call_id": call_id,
                    "name": name,
                    "result": res,
                }
                await self._orch.events.put(evt)
            except asyncio.CancelledError:
                # Treat tool cancellation as a normal stop; do not bubble
                await self._orch.events.put(
                    {
                        "type": "tool_failed",
                        "call_id": call_id,
                        "name": name,
                        "error": "cancelled",
                    },
                )
            except Exception as e:
                await self._orch.events.put(
                    {
                        "type": "tool_failed",
                        "call_id": call_id,
                        "name": name,
                        "error": str(e),
                    },
                )

        self._orch._register_child(self._orch._tg.create_task(_task()))

        # If we attached a notification queue, schedule a watcher
        if notification_up_q is not None:

            async def _watch_notifications():
                try:
                    while True:
                        payload = await notification_up_q.get()
                        msg = None
                        try:
                            if isinstance(payload, dict):
                                msg = payload.get("message")
                            else:
                                msg = str(payload)
                        except Exception:
                            msg = None
                        await self._orch.events.put(
                            {
                                "type": "notification_received",
                                "call_id": call_id,
                                "tool_name": name,
                                "message": msg or "",
                            },
                        )
                except asyncio.CancelledError:
                    return

            self._orch._register_child(
                self._orch._tg.create_task(_watch_notifications()),
            )

        # Clarification watcher: bubble requests
        if clar_up_q is not None:

            async def _watch_clarifications():
                try:
                    while True:
                        q = await clar_up_q.get()
                        # Requests may be raw strings or dicts with images/question
                        question = q.get("question") if isinstance(q, dict) else str(q)
                        await self._orch.events.put(
                            {
                                "type": "clarification_requested",
                                "call_id": call_id,
                                "tool_name": name,
                                "question": question or "",
                            },
                        )
                except asyncio.CancelledError:
                    return

            self._orch._register_child(
                self._orch._tg.create_task(_watch_clarifications()),
            )

    @staticmethod
    def is_safe_to_schedule_without_clar(fn: Callable) -> bool:
        """Return True if the tool does not require clarification/interject queues."""
        try:
            sig = inspect.signature(fn)
        except Exception:
            return True
        req = set()
        for name, p in sig.parameters.items():
            if p.default is inspect._empty and p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                req.add(name)
        blocked = {
            "clarification_up_q",
            "clarification_down_q",
            "interject_queue",
        }
        return not (req & blocked)
