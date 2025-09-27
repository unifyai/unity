"""
ActiveTask provides a handle for a single running task backed by an actor.

It wraps a SteerableToolHandle returned by a BaseActor and, when a scheduler
is provided, mirrors lifecycle status to the task row and clears the scheduler's
active pointer on completion or stop. It supports read-only ask, steering via
interject (cancel/defer), pausing/resuming, stopping, and result retrieval. On
defer, it attempts to reinstate the task to its previous queue position using
the scheduler.
"""

import functools
import asyncio
from typing import Optional, Dict, TYPE_CHECKING, List, Any

from .base import BaseActiveTask
from ..actor.base import BaseActor
from unity.common.async_tool_loop import SteerableToolHandle
from .llm import new_llm_client


async def classify_steering_intent(
    message: str,
    parent_chat_context: Optional[List[Dict[str, Any]]] = None,  # type: ignore[name-defined]
) -> tuple[str, str]:
    """Classify steering into: cancel | defer | pause | resume | continue | none."""
    try:
        client = new_llm_client("gpt-5@openai")
        system = (
            "You are a router that classifies an in-flight steering message.\n"
            "Labels: cancel | defer | pause | resume | continue | none.\n"
            "Definitions:\n"
            "- cancel: abandon/kill/drop the task (terminal).\n"
            "- defer: stop for now but resume later / return to prior queue/schedule.\n"
            "- pause: temporarily pause, expecting explicit resume soon.\n"
            "- resume: continue after a pause.\n"
            "- continue: keep going (no change).\n"
            "- none: message is not a steering instruction.\n"
            'Output ONLY JSON with rationale first: {"rationale": <short string>, "action": <label>, "reason": <short substring or null>}'
        )
        client.set_system_message(system)

        def _format_ctx(ctx: Optional[List[Dict[str, Any]]], limit_chars: int = 2000) -> str:  # type: ignore[name-defined]
            try:
                if not ctx:
                    return "(no prior context)"
                lines: List[str] = []
                total = 0
                for msg in reversed(ctx[-20:]):
                    role = str(msg.get("role", "")).strip() or "user"
                    content = str(msg.get("content", "")).strip()
                    line = f"{role}: {content}"
                    if total + len(line) > limit_chars:
                        break
                    lines.append(line)
                    total += len(line)
                return "\n".join(reversed(lines)) if lines else "(no prior context)"
            except Exception:
                return "(no prior context)"

        ctx_block = _format_ctx(parent_chat_context)
        user = (
            "Recent conversation (most recent last):\n"
            f"{ctx_block}\n\n"
            "Steering message:\n"
            f"{(message or '').strip()}"
        )
        raw = await client.generate(user)
        import json as _json  # local import

        try:
            data = _json.loads(raw)
            action = str(data.get("action", "none")).strip().lower()
            reason = data.get("reason")
            if action not in {"cancel", "defer", "pause", "resume", "continue", "none"}:
                action = "none"
            if reason is not None:
                reason = str(reason)
            else:
                reason = message
            return action, str(reason)
        except Exception:
            low = (raw or "").lower()
            for tok in ["cancel", "defer", "pause", "resume", "continue"]:
                if tok in low:
                    return tok, message
            return "none", message
    except Exception:
        return "none", message


if TYPE_CHECKING:
    from .task_scheduler import TaskScheduler


class ActiveTask(BaseActiveTask):
    def __init__(
        self,
        actor_handle: SteerableToolHandle,
        *,
        task_id: Optional[int] = None,
        instance_id: Optional[int] = None,
        scheduler: Optional["TaskScheduler"] = None,
    ):
        """
        Thin wrapper around an actor-backed active plan handle, keeping the
        corresponding Tasks row in sync when a scheduler is provided.

        Use ``ActiveTask.create(...)`` to construct an instance from a
        ``BaseActor`` and a task description.
        """
        self._actor_handle = actor_handle
        self._scheduler: Optional["TaskScheduler"] = scheduler
        self._task_id: Optional[int] = task_id
        self._instance_id: Optional[int] = instance_id
        self._was_stopped: bool = False
        self._last_intent: Optional[str] = None
        self._last_intent_reason: Optional[str] = None

    @classmethod
    async def create(
        cls,
        actor: BaseActor,
        *,
        task_description: str,
        parent_chat_context: Optional[list[dict]] = None,
        clarification_up_q: Optional["asyncio.Queue[str]"] = None,
        clarification_down_q: Optional["asyncio.Queue[str]"] = None,
        task_id: Optional[int] = None,
        instance_id: Optional[int] = None,
        scheduler: Optional["TaskScheduler"] = None,
    ) -> "ActiveTask":
        """
        Create an ActiveTask by starting work on the provided ``actor``.

        This is the preferred constructor: it ensures the underlying active
        handle is running before returning an instance.
        """
        actor_steerable_handle = await actor.act(
            task_description,
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )
        return cls(
            actor_steerable_handle,  # type: ignore[arg-type]
            task_id=task_id,
            instance_id=instance_id,
            scheduler=scheduler,
        )

    @functools.wraps(BaseActiveTask.ask, updated=())
    async def ask(
        self,
        message: str,
        *,
        _return_reasoning_steps: bool = False,
    ) -> SteerableToolHandle:
        """Answer a read-only question about the live activity and return a handle."""
        answer: str = await self._actor_handle.ask(message)

        # Lightweight static handle that simply returns the captured answer
        class _AnswerHandle(SteerableToolHandle):  # type: ignore[abstract-method]
            def __init__(self) -> None:
                pass

            async def interject(self, message: str): ...

            def stop(self, reason: Optional[str] = None): ...

            def pause(self): ...

            def resume(self): ...

            def done(self) -> bool:
                return True

            async def result(self) -> str:
                # Ignoring _return_reasoning_steps for ActiveTask.ask; only answer string is returned.
                return answer

            async def ask(self, question: str) -> "SteerableToolHandle":  # type: ignore[override]
                return self

        return _AnswerHandle()

    @functools.wraps(BaseActiveTask.interject, updated=())
    async def interject(self, message: str) -> None:
        # Classify steering intent and enforce lifecycle synchronization for stop/defer/cancel.
        intent: Optional[str] = None
        reason: Optional[str] = None

        try:
            if self._scheduler is not None and hasattr(
                self._scheduler,
                "_classify_steering_intent",
            ):
                intent, reason = await self._scheduler._classify_steering_intent(  # type: ignore[attr-defined]
                    message,
                    parent_chat_context=None,
                )
            else:
                intent, reason = await classify_steering_intent(
                    message,
                    parent_chat_context=None,
                )
        except Exception:
            intent, reason = None, None

        self._last_intent = intent
        self._last_intent_reason = reason or message

        # If the interjection semantically requests stopping, enforce correct lifecycle handling.
        if intent in ("cancel", "defer"):
            try:
                if hasattr(self._actor_handle, "stop"):
                    self._actor_handle.stop(reason)  # type: ignore[call-arg]
            except Exception:
                pass

            self._was_stopped = True  # prevents result() from marking 'completed'

            try:
                if self._scheduler and self._task_id is not None:
                    if intent == "cancel":
                        # Explicit cancellation: mark cancelled.
                        self._mirror_status("cancelled")
                    else:
                        # Defer: restore prior queue/schedule position via public API when available.
                        try:
                            self._call_reinstate_public(task_id=self._task_id)
                        except Exception:
                            # Fallback: downgrade status to prior state from plan or 'queued'.
                            try:
                                plan = None
                                if self._instance_id is not None:
                                    plan = (self._scheduler._reintegration_plans or {}).get(  # type: ignore[attr-defined]
                                        (self._task_id, self._instance_id),
                                    )
                                prior_status = (
                                    str(getattr(plan, "original_status", ""))
                                    if plan is not None
                                    else ""
                                )
                                target_status = (
                                    prior_status if prior_status else "queued"
                                )
                                if self._instance_id is not None:
                                    self._scheduler._update_task_status_instance(  # type: ignore[attr-defined]
                                        task_id=self._task_id,
                                        instance_id=self._instance_id,
                                        new_status=target_status,
                                    )
                            except Exception:
                                pass
            except Exception:
                # Best-effort: failure to reinstate or fallback must not break stop semantics
                pass

            self._clear_active_pointer()
            return

        # No stop/defer/cancel intent ⇒ forward interjection to the actor.
        await self._actor_handle.interject(message)

    @functools.wraps(BaseActiveTask.stop, updated=())
    def stop(self, *, cancel: bool, reason: Optional[str] = None) -> Optional[str]:
        """Stop the running activity with explicit intent.

        When ``cancel`` is True the task instance is marked cancelled. When False, the
        task is deferred and we attempt to reinstate it to its previous queue/schedule
        position using the stored reintegration plan (when available).
        """
        # Be tolerant if the underlying actor has already finished; treat stop as a no-op.
        try:
            ret = self._actor_handle.stop(reason)  # type: ignore[call-arg]
        except Exception:
            ret = "Stopped."
        self._was_stopped = True

        # Cancel → mark cancelled; Defer → try reinstatement
        if cancel:
            self._mirror_status("cancelled")
        else:
            try:
                if self._scheduler and self._task_id is not None:
                    # Prefer strict reinstatement using the stored plan when present.
                    self._call_reinstate_public(task_id=self._task_id)
            except Exception:
                # Best-effort – failure to reinstate must not break stop semantics
                pass

        self._clear_active_pointer()
        return ret

    @functools.wraps(BaseActiveTask.pause, updated=())
    def pause(self) -> Optional[str]:
        ret = self._actor_handle.pause()
        self._mirror_status("paused")
        return ret

    @functools.wraps(BaseActiveTask.resume, updated=())
    def resume(self) -> Optional[str]:
        ret = self._actor_handle.resume()
        self._mirror_status("active")
        return ret

    @functools.wraps(BaseActiveTask.done, updated=())
    def done(self) -> bool:
        return self._actor_handle.done()

    @functools.wraps(BaseActiveTask.result, updated=())
    async def result(self) -> str:
        ret = await self._actor_handle.result()
        # If the task wasn't explicitly cancelled/failed, mark as completed.
        if self._scheduler and self._task_id is not None and not self._was_stopped:
            rows = self._scheduler._filter_tasks(  # type: ignore[attr-defined]
                filter=f"task_id == {self._task_id} and instance_id == {self._instance_id}",
                limit=1,
            )
            cur_status = None
            try:
                if rows:
                    cur_status = rows[0].get("status")
            except Exception:
                cur_status = None
            if rows and cur_status not in ("cancelled", "failed"):
                self._mirror_status("completed")
        self._clear_active_pointer()
        return ret

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _mirror_status(self, new_status: str) -> None:
        """Update the task-row status if we were instantiated by a scheduler."""
        if (
            self._scheduler
            and self._task_id is not None
            and self._instance_id is not None
        ):
            self._scheduler._update_task_status_instance(  # type: ignore[attr-defined]
                task_id=self._task_id,
                instance_id=self._instance_id,
                new_status=new_status,
            )

    def _clear_active_pointer(self) -> None:
        """Free the scheduler's active-task slot, if any."""
        if self._scheduler and getattr(self._scheduler, "_active_task", None):
            active = self._scheduler._active_task  # type: ignore[attr-defined]
            if (
                getattr(active, "task_id", None) == self._task_id
                and getattr(active, "instance_id", None) == self._instance_id
            ):
                self._scheduler._active_task = None  # type: ignore[attr-defined]

    # Centralized reinstate caller to avoid duplication and select an available scheduler API
    def _call_reinstate_public(self, *, task_id: int) -> None:
        sched = self._scheduler
        if sched is None:
            return
        try:
            # Prefer public API when available
            if hasattr(sched, "_reinstate_to_previous_queue"):
                try:
                    # Try with allow_active=True
                    sched._reinstate_to_previous_queue(task_id=task_id, allow_active=True)  # type: ignore[attr-defined]
                except TypeError:
                    # Fallback without allow_active (defensive)
                    sched._reinstate_to_previous_queue(task_id=task_id)  # type: ignore[attr-defined]
                return
        except Exception:
            pass

        # Fallback to private method if the public API is unavailable; handle optional arguments
        try:
            sched._reinstate_task_to_previous_queue(task_id=task_id, _allow_active=True)  # type: ignore[attr-defined]
        except TypeError:
            sched._reinstate_task_to_previous_queue(task_id=task_id)  # type: ignore[attr-defined]
