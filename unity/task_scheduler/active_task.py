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
import textwrap
from typing import Optional, Dict, TYPE_CHECKING, List, Any

from .base import BaseActiveTask
from ..actor.base import BaseActor
from unity.common.async_tool_loop import SteerableToolHandle
from unity.common.task_execution_context import current_task_execution_delegate
from .types.status import Status
from ..common.llm_client import new_llm_client
import logging
from ..common.handle_wrappers import HandleWrapperMixin

logger = logging.getLogger(__name__)


async def classify_steering_intent(
    message: str,
    parent_chat_context: Optional[List[Dict[str, Any]]] = None,  # type: ignore[name-defined]
) -> tuple[str, str]:
    """Classify steering into: cancel | defer | pause | resume | continue | none."""
    try:
        client = new_llm_client()
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


class ActiveTask(BaseActiveTask, HandleWrapperMixin):
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

        # Register the underlying actor handle for standardized wrapper discovery
        self._wrap_handle(actor_handle)

    @classmethod
    async def create(
        cls,
        actor: BaseActor,
        *,
        task_description: str,
        _parent_chat_context: Optional[list[dict]] = None,
        _clarification_up_q: Optional["asyncio.Queue[str]"] = None,
        _clarification_down_q: Optional["asyncio.Queue[str]"] = None,
        task_id: Optional[int] = None,
        instance_id: Optional[int] = None,
        scheduler: Optional["TaskScheduler"] = None,
        entrypoint: Optional[int] = None,
    ) -> "ActiveTask":
        """
        Create an ActiveTask by starting work on the provided ``actor``.

        This is the preferred constructor: it ensures the underlying active
        handle is running before returning an instance.
        """
        delegate = current_task_execution_delegate.get()
        if delegate is not None:
            actor_steerable_handle = await delegate.start_task_run(
                task_description=task_description,
                entrypoint=entrypoint,
                parent_chat_context=_parent_chat_context,
                clarification_up_q=_clarification_up_q,
                clarification_down_q=_clarification_down_q,
                images=None,
            )
        else:
            actor_steerable_handle = await actor.act(
                task_description,
                _parent_chat_context=_parent_chat_context,
                _clarification_up_q=_clarification_up_q,
                _clarification_down_q=_clarification_down_q,
                # Always pass entrypoint to the actor so it can immediately run the function
                entrypoint=entrypoint,
                persist=False,  # Scheduler-run plans should complete instead of pausing for interjection
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
        return await self._actor_handle.ask(message)

    @functools.wraps(BaseActiveTask.interject, updated=())
    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
        images: object | None = None,
    ) -> None:
        # Classify steering intent and enforce lifecycle synchronization for stop/defer/cancel.
        intent: Optional[str] = None
        reason: Optional[str] = None

        try:
            # Attempt scheduler-provided classifier first (broad signature compatibility)
            if self._scheduler is not None and hasattr(
                self._scheduler,
                "_classify_steering_intent",
            ):
                try:
                    # Prefer calling with only the message to avoid kwarg name mismatches
                    intent, reason = await self._scheduler._classify_steering_intent(  # type: ignore[attr-defined]
                        message,
                    )
                except TypeError:
                    # Retry with underscore-style kwarg for compatibility with some implementations
                    intent, reason = await self._scheduler._classify_steering_intent(  # type: ignore[attr-defined]
                        message,
                        _parent_chat_context=None,
                    )
            else:
                intent, reason = await classify_steering_intent(
                    message,
                    parent_chat_context=_parent_chat_context_cont,
                )
        except Exception as e:
            # Robust fallback: use built-in classifier to avoid losing the steering signal entirely
            try:
                intent, reason = await classify_steering_intent(
                    message,
                    parent_chat_context=_parent_chat_context_cont,
                )
            except Exception as _e:
                intent, reason = None, None

        self._last_intent = intent
        self._last_intent_reason = reason or message

        # If the interjection semantically requests stopping, enforce correct lifecycle handling.
        if intent in ("cancel", "defer"):
            # Stop the underlying actor handle. Some handle implementations expose stop() as async;
            # we must not drop the coroutine. Also, some stop() variants accept (reason=..., cancel=...).
            stop_reason = reason or message
            try:
                if hasattr(self._actor_handle, "stop"):
                    try:
                        ret = self._actor_handle.stop(  # type: ignore[call-arg]
                            reason=stop_reason,
                            cancel=(intent == "cancel"),
                        )
                    except TypeError:
                        # Legacy signature: stop(reason: str | None = None)
                        ret = self._actor_handle.stop(stop_reason)  # type: ignore[call-arg]

                    if asyncio.iscoroutine(ret):
                        asyncio.create_task(ret)
            except Exception:
                pass

            self._was_stopped = True  # prevents result() from marking 'completed'

            try:
                if self._scheduler and self._task_id is not None:
                    if intent == "cancel":
                        # Explicit cancellation: mark cancelled.
                        self._mirror_status(Status.cancelled)
                    else:
                        # Defer: restore prior queue/schedule position via public API when available.
                        try:
                            self._call_reinstate_public(task_id=self._task_id)
                        except Exception:
                            # Fallback: downgrade status to prior state from plan or 'queued'.
                            try:
                                plan = None
                                if self._instance_id is not None:
                                    plan = self._scheduler._reintegration_plans.get(  # type: ignore[attr-defined]
                                        (self._task_id, self._instance_id),
                                    )
                                prior_status = (
                                    str(getattr(plan, "original_status", ""))
                                    if plan is not None
                                    else ""
                                )
                                target_status = (
                                    prior_status if prior_status else Status.queued
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
        # Avoid passing images kwarg when None to preserve compatibility with wrappers
        # that don't declare the images kwarg (e.g., some test monkeypatches).
        if images is None:
            await self._actor_handle.interject(message)  # type: ignore[arg-type]
        else:
            await self._actor_handle.interject(message, images=images)  # type: ignore[arg-type]

    @functools.wraps(BaseActiveTask.stop, updated=())
    async def stop(
        self,
        *,
        cancel: bool = False,
        reason: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        """Stop the running activity with explicit intent.

        When ``cancel`` is True the task instance is marked cancelled. When False, the
        task is deferred and we attempt to reinstate it to its previous queue/schedule
        position using the stored reintegration plan (when available).
        """
        # Be tolerant if the underlying actor has already finished; treat stop as a no-op.
        try:
            # Prefer passing cancel/reason when supported, but fall back for compatibility.
            try:
                ret = await self._actor_handle.stop(reason=reason, cancel=cancel)  # type: ignore[call-arg]
            except TypeError:
                # Legacy signature: stop(reason: str | None = None)
                ret = await self._actor_handle.stop(reason)  # type: ignore[call-arg]
        except Exception:
            ret = "Stopped."
        self._was_stopped = True

        final_status = "cancelled" if cancel else "stopped"

        # Cancel → mark cancelled; Defer → try reinstatement
        if cancel:
            self._mirror_status(Status.cancelled)
        else:
            try:
                if self._scheduler and self._task_id is not None:
                    # Prefer strict reinstatement using the stored plan when present.
                    self._call_reinstate_public(task_id=self._task_id)
            except Exception:
                # Best-effort – failure to reinstate must not break stop semantics
                pass

        asyncio.create_task(self._save_final_summary(final_status))

        self._clear_active_pointer()
        return ret

    @functools.wraps(BaseActiveTask.pause, updated=())
    async def pause(self) -> Optional[str]:
        ret = await self._actor_handle.pause()
        self._mirror_status(Status.paused)
        return ret

    @functools.wraps(BaseActiveTask.resume, updated=())
    async def resume(self) -> Optional[str]:
        ret = await self._actor_handle.resume()
        self._mirror_status(Status.active)
        return ret

    @functools.wraps(BaseActiveTask.done, updated=())
    def done(self) -> bool:
        return self._actor_handle.done()

    async def _generate_summary_from_log(self, action_log: List[str]) -> str:
        """
        Generates a concise, human-readable summary of the execution from the Actor's action_log which captures a trace of the task's execution.
        """
        client = new_llm_client()
        prompt = textwrap.dedent(
            f"""
            You are an assistant summarizing a complex task's execution log.
            Your summary will be stored in a database `info` column
            to provide a quick overview of "what actually happened".

            - Focus on the final outcome (e.g., completed, stopped, error).
            - Mention any user interjections or clarifications.
            - Mention any major verification failures and recoveries.
            - Be concise (1-3 sentences).

            EXECUTION LOG:
            ---
            {chr(10).join(action_log)}
            ---

            Concisely summarize what happened:
        """,
        )
        try:
            summary = await client.generate(prompt)
            return summary.strip()
        except Exception as e:
            logger.error("Error during summary generation: %s", e)
            return "Summary generation failed. Final Status: <UNKNOWN>"  # Status added in _save_final_summary

    async def _save_final_summary(self, final_status: str):
        """
        Generates the final summary and updates the task row in the database.
        """
        if (
            self._scheduler
            and self._task_id is not None
            and self._instance_id is not None
        ):
            summary = "No execution log was available to generate a summary."
            try:
                # The _actor_handle is the HierarchicalActorHandle, which has the action_log
                if (
                    hasattr(self._actor_handle, "action_log")
                    and self._actor_handle.action_log
                ):
                    summary = await self._generate_summary_from_log(
                        self._actor_handle.action_log,
                    )
                else:
                    summary = f"Task finished with status '{final_status}'. No detailed log found."

                # Replace <UNKNOWN> status in fallback summaries
                summary = summary.replace("<UNKNOWN>", final_status)

                # Update the task instance with the generated summary
                # Write the human-readable summary to the 'info' field (status is finalized in result())
                self._scheduler._update_task_instance(  # type: ignore[attr-defined]
                    task_id=self._task_id,
                    instance_id=self._instance_id,
                    info=summary,
                )

            except Exception as e:
                logger.error(
                    "Error saving final task summary: %s",
                    e,
                )

    @functools.wraps(BaseActiveTask.result, updated=())
    async def result(self) -> str:
        final_status: Optional[str] = None
        ret: Optional[str] = None
        error: Optional[Exception] = None

        try:
            # Await the underlying actor's result
            ret = await self._actor_handle.result()
            # If we get here without error and it wasn't stopped, mark as completed
            if not self._was_stopped:
                final_status = "completed"

        except Exception as e:
            # Capture the error if the actor's result raised one
            error = e
            # Only mark as failed if it wasn't explicitly stopped/cancelled beforehand
            if not self._was_stopped:
                final_status = "failed"
                ret = f"Task failed with error: {type(e).__name__}({e})"
                logger.error(
                    "--- Task %s.%s failed: %s ---",
                    self._task_id,
                    self._instance_id,
                    e,
                )

        finally:
            # Save summary if a terminal status (completed/failed) was determined
            # during this call AND the task wasn't already marked as stopped externally.
            if (
                final_status
                and not self._was_stopped
                and self._scheduler
                and self._task_id is not None
                and self._instance_id is not None
            ):
                # Finalize terminal status synchronously so callers observe a non-active row
                # and the reintegration plan is cleared immediately after result() returns.
                self._scheduler._update_task_status_instance(  # type: ignore[attr-defined]
                    task_id=self._task_id,
                    instance_id=self._instance_id,
                    new_status=final_status,
                )

                # Idempotently schedule generation of the human-readable summary
                if not getattr(self, "_summary_scheduled", False):
                    try:
                        logger.info(
                            "--- Scheduling save_final_summary for %s.%s with status: %s ---",
                            self._task_id,
                            self._instance_id,
                            final_status,
                        )
                        asyncio.create_task(self._save_final_summary(final_status))
                        self._summary_scheduled = True  # type: ignore[attr-defined]
                    except Exception as summary_e:
                        logger.error("Error creating summary task: %s", summary_e)

            # Clear the scheduler's active pointer if the task reached a terminal state
            # (completed/failed) OR if it was stopped externally (_was_stopped).
            if final_status or self._was_stopped:
                self._clear_active_pointer()

        if error and final_status == "failed":
            # If an error occurred and we marked it as failed, re-raise the original error
            raise error
        elif self._was_stopped and not ret:
            # If stopped but no specific result string was set (e.g. via stop reason)
            return "Task stopped."  # Provide a default message
        elif ret is not None:
            # Return the result from the actor or the formatted error message for failures
            return ret
        else:
            return "Task finished."

    # ------------------------------------------------------------------ #
    # Bottom-up event APIs (delegate to underlying actor handle)         #
    # ------------------------------------------------------------------ #
    @functools.wraps(SteerableToolHandle.next_clarification, updated=())
    async def next_clarification(self) -> dict:
        try:
            return await self._actor_handle.next_clarification()  # type: ignore[attr-defined]
        except Exception:
            return {}

    @functools.wraps(SteerableToolHandle.next_notification, updated=())
    async def next_notification(self) -> dict:
        try:
            return await self._actor_handle.next_notification()  # type: ignore[attr-defined]
        except Exception:
            return {}

    @functools.wraps(SteerableToolHandle.answer_clarification, updated=())
    async def answer_clarification(self, call_id: str, answer: str) -> None:
        try:
            await self._actor_handle.answer_clarification(call_id, answer)  # type: ignore[attr-defined]
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _mirror_status(self, new_status: Status) -> None:
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
