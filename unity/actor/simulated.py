import asyncio
import threading
import time

import unillm
from .base import BaseActor, BaseActorHandle
import functools
from typing import Any, Optional, Type
from pydantic import BaseModel
from unity.manager_registry import ManagerRegistry
from unity.constants import LOGGER
from unity.common.simulated import (
    SimulatedLineage,
    SimulatedLog,
    simulated_llm_roundtrip,
    SimulatedHandleMixin,
)
from unity.common.llm_client import new_llm_client
from unity.common.async_tool_loop import SteerableToolHandle


class _StaticAnswerHandle(SteerableToolHandle):
    """Trivial handle that wraps a static answer string for ask() returns."""

    def __init__(self, answer: str) -> None:
        self._answer = answer

    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
        images: object | None = None,
    ) -> "SteerableToolHandle":
        return self

    async def interject(
        self,
        message: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
        images: object | None = None,
    ) -> Optional[str]:
        return None

    def stop(
        self,
        reason: Optional[str] = None,
    ) -> Optional[str]:
        return None

    async def pause(self) -> str:
        return "Already completed."

    async def resume(self) -> str:
        return "Already completed."

    def done(self) -> bool:
        return True

    def trigger_completion(self, result: str | None = None) -> None:
        """No-op for static answer handles (already complete)."""

    async def result(self) -> str:
        return self._answer

    async def next_clarification(self) -> dict:
        return {}

    async def next_notification(self) -> dict:
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        pass


class SimulatedActorHandle(BaseActorHandle, SimulatedHandleMixin):
    """
    A lightweight, actor-scoped handle for simulating execution of a series of actions.

    This mirrors the public surface expected by higher layers/tests:
    - ask(question) -> str
    - interject(instruction) -> None
    - pause() / resume() -> str
    - stop(reason) -> str
    - result() -> str (async)
    - done() -> bool
    """

    # Per-run file sink for simulated LLM I/O logs (request/response)
    _SIM_ACT_LLM_IO_DIR: "str | None" = None

    def __init__(
        self,
        llm: unillm.AsyncUnify,
        description: str,
        *,
        steps: int | None,
        duration: float | None = None,
        parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        log_mode: "str | None" = "log",
        # Optional: function entrypoint context and prebaked result
        entrypoint_info: dict | None = None,
        planned_result: str | None = None,
        # New: optional session suffix to reuse across simulated logs
        session_suffix: "str | None" = None,
        # Optional response format for structured output
        response_format: Optional[Type[BaseModel]] = None,
        # Whether to emit notifications via next_notification()
        emit_notifications: bool = True,
    ) -> None:
        self._llm = llm
        self._description = description
        self._steps = steps
        self._duration = duration
        self._parent_chat_context = parent_chat_context
        self._clarification_up_q = clarification_up_q
        self._clarification_down_q = clarification_down_q
        self._requests_clarification = _requests_clarification
        self._log_mode: str | None = (
            log_mode if log_mode in ("print", "log", None) else "log"
        )
        self._response_format = response_format
        self._emit_notifications = emit_notifications

        # Store optional entrypoint metadata and a planned completion result
        self._entrypoint_info: dict | None = entrypoint_info
        self._planned_result: str | None = planned_result or None

        self._steps_taken = 0
        self._step_lock = threading.Lock()
        # Track remaining time (freezes while paused)
        self._remaining_duration: float | None = duration
        self._last_started_at: float | None = None

        self._done_event = threading.Event()
        self._result_str: str | None = None
        self._paused = None
        self._action_thread: threading.Thread | None = None
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None

        # Human-friendly log label derived from current lineage. Reuse a provided session suffix when present.
        # "<outer...>->SimulatedActor.act(abcd)"
        try:
            if session_suffix:
                self._log_label = SimulatedLineage.make_label_with_suffix(
                    "SimulatedActor.act",
                    session_suffix,
                )
            else:
                self._log_label = SimulatedLineage.make_label("SimulatedActor.act")
        except Exception:
            self._log_label = "SimulatedActor.act"

        self._start()

    @property
    def clarification_up_q(self) -> Optional[asyncio.Queue[str]]:
        return self._clarification_up_q

    @property
    def clarification_down_q(self) -> Optional[asyncio.Queue[str]]:
        return self._clarification_down_q

    def _run_actions(self, description: str) -> None:
        try:
            while True:
                if self._requests_clarification:
                    try:
                        q_text = (
                            "Can you please clarify what exactly you'd like me to do?"
                        )
                        self._clarification_up_q.put_nowait(q_text)
                        try:
                            SimulatedLog.log_clarification_request(
                                self._log_label,
                                q_text,
                            )
                        except Exception:
                            pass
                    except asyncio.QueueFull:
                        pass
                    while True:
                        # Allow immediate termination while waiting for a clarification
                        if self._stop_event.is_set():
                            return
                        try:
                            answer: str = self._clarification_down_q.get_nowait()
                            break
                        except asyncio.QueueEmpty:
                            time.sleep(0.05)
                    try:
                        SimulatedLog.log_clarification_answer(self._log_label, answer)
                    except Exception:
                        pass
                    self._complete(f"Clarification received: {answer}")
                    return
                if self._stop_event.is_set():
                    return
                if (
                    self._remaining_duration is not None
                    and self._last_started_at is not None
                    and (time.monotonic() - self._last_started_at)
                    >= self._remaining_duration
                ):
                    # Prefer prebaked function-aware result if available
                    msg = (
                        self._planned_result
                        or f"Completed '{description}' after {self._duration}\u2009s duration."
                    )
                    self._complete(msg)
                    return
                if self._steps is not None and self._steps_taken >= (self._steps or 0):
                    msg = (
                        self._planned_result
                        or f"Completed '{description}' in {self._steps} steps."
                    )
                    self._complete(msg)
                    return
                self._pause_event.wait()
                time.sleep(0.1)
        finally:
            self._description = None
            self._paused = None
            self._action_thread = None
            self._pause_event.set()
            self._stop_event.clear()

    def _start(self):
        self._paused = False
        self._pause_event.set()
        self._stop_event.clear()
        self._last_started_at = time.monotonic()
        self._action_thread = threading.Thread(
            target=self._run_actions,
            args=(self._description,),
            daemon=True,
        )
        self._action_thread.start()
        # Start a periodic monitor that emits remaining duration every 20 seconds
        if self._duration is not None:

            def _monitor():
                try:
                    while not self._done_event.is_set():
                        rem = self.get_remaining_duration_seconds()
                        if rem is not None:
                            self._emit_status(
                                f"⏳ Duration remaining: {max(0.0, rem):.1f}s",
                            )
                        # Sleep in small chunks to be responsive to done-event (~20s total)
                        for _ in range(200):
                            if self._done_event.is_set():
                                break
                            time.sleep(0.1)
                finally:
                    return

            self._monitor_thread = threading.Thread(target=_monitor, daemon=True)
            self._monitor_thread.start()

    def _complete(self, message: str) -> None:
        if not self._done_event.is_set():
            # Ensure any waiting threads (e.g., paused waits) are released
            try:
                self._pause_event.set()
            except Exception:
                pass
            self._stop_event.set()
            self._result_str = message
            self._done_event.set()
            import threading as _th

            if (
                self._action_thread
                and self._action_thread.is_alive()
                and _th.current_thread() is not self._action_thread
            ):
                self._action_thread.join(timeout=1)
            # Best-effort join of the monitor thread
            try:
                if self._monitor_thread and self._monitor_thread.is_alive():
                    self._monitor_thread.join(timeout=1)
            except Exception:
                pass

    def simulate_step(self):
        if not self._done_event.is_set():
            with self._step_lock:
                self._steps_taken += 1
            if self._steps is not None and self._steps_taken >= self._steps:
                self._complete(
                    f"Completed '{self._description}' in {self._steps} steps.",
                )
            # Emit steps remaining after each user-visible interaction that consumes a step
            try:
                if self._steps is not None:
                    remaining = max(0, int(self._steps) - int(self._steps_taken))
                    self._emit_status(f"🪜 Steps remaining: {remaining}")
            except Exception:
                pass

    async def result(self) -> str:
        # Wait for action to complete using polling instead of asyncio.to_thread().
        # Using asyncio.to_thread(_done_event.wait) creates executor threads that block
        # indefinitely, which prevents pytest-asyncio from cleaning up the event loop
        # after tests complete. Polling with asyncio.sleep() allows the coroutine to be
        # cancelled cleanly during event loop shutdown.
        while not self._done_event.is_set():
            await asyncio.sleep(0.1)
        raw_result = self._result_str  # type: ignore

        # If response_format is specified, generate structured output
        if self._response_format is not None:
            try:
                import json

                schema = self._response_format.model_json_schema()
                prompt = (
                    f"The following action was completed:\n{raw_result}\n\n"
                    "Return valid JSON matching the following schema. "
                    "Do NOT include any extra keys or commentary.\n"
                    f"{json.dumps(schema, indent=2)}"
                )
                self._llm.set_response_format(self._response_format)
                try:
                    structured_result = await self._llm.generate(prompt)
                finally:
                    try:
                        self._llm.reset_response_format()
                    except Exception:
                        pass
                return structured_result
            except Exception:
                pass

        return raw_result

    def stop(
        self,
        reason: Optional[str] = None,
    ) -> str:
        """Stop the in-flight handle.

        Args:
            reason: Optional reason for stopping.
        """
        if self._done_event.is_set():
            return (
                self._result_str or "Already stopped."
            )  # Return existing result if done
        if not self._description:
            raise Exception("No actions are currently being performed.")
        msg = f"Stopped '{self._description}' for reason: {reason}"
        try:
            suffix = f" – reason: {reason}" if reason else ""
            LOGGER.info(f"🛑 [{self._log_label}] Stop requested{suffix}")
        except Exception:
            pass
        # Unpause immediately so the action loop can observe the stop signal
        try:
            self._pause_event.set()
        except Exception:
            pass
        self._complete(msg)
        return msg

    async def interject(
        self,
        message: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
        images: object | None = None,
    ) -> None:
        if not self._description:
            raise Exception("No actions are currently being performed.")
        self.simulate_step()

        # Human-facing interject log (lineage-aligned)
        self._log_interject(message)

        # Build a content list for a user message that includes the instruction and any attached images.
        # Images are resolved to handles and added as image_url blocks (data URLs or signed URLs where available).
        content_blocks: list[dict] = [{"type": "text", "text": str(message)}]

        # Best-effort image resolution and block construction
        if images is not None:
            try:
                # Support ImageRefs containers (duck-typed via .root) and plain lists
                items = list(getattr(images, "root", images) or [])
            except Exception:
                items = []

            # Collect candidate handles by id or direct handle objects
            handles: list[object] = []
            ids_to_fetch: list[int] = []

            # Local imports to avoid module import cycles
            try:
                from unity.image_manager.types import (
                    RawImageRef as _RawImageRef,
                    AnnotatedImageRef as _AnnotatedImageRef,
                )
            except Exception:  # pragma: no cover - robustness
                _RawImageRef = object  # type: ignore
                _AnnotatedImageRef = object  # type: ignore

            for ref in items:
                try:
                    # Direct handle case (has raw() method and image_id attr)
                    if hasattr(ref, "raw") and hasattr(ref, "image_id"):
                        handles.append(ref)
                        continue
                    # Typed refs
                    if isinstance(ref, _AnnotatedImageRef):
                        ids_to_fetch.append(int(ref.raw_image_ref.image_id))
                    elif isinstance(ref, _RawImageRef):
                        ids_to_fetch.append(int(ref.image_id))
                    # Primitive id
                    elif isinstance(ref, int):
                        ids_to_fetch.append(int(ref))
                except Exception:
                    continue

            # Resolve any remaining ids to handles via ManagerRegistry
            if ids_to_fetch:
                try:
                    mgr = ManagerRegistry.get_image_manager()
                    fetched = mgr.get_images(ids_to_fetch)
                    for h in fetched:
                        if h is not None:
                            handles.append(h)
                except Exception:
                    pass

            # Convert handles to content blocks and append
            for ih in handles:
                try:
                    # Prefer direct URLs when present on the underlying image record
                    data_str = getattr(getattr(ih, "_image", object), "data", None)
                    content_block: dict
                    if isinstance(data_str, str) and (
                        data_str.startswith("http://")
                        or data_str.startswith("https://")
                        or data_str.startswith("data:image/")
                        or data_str.startswith("gs://")
                    ):
                        # Best-effort: sign GCS URLs if a storage client is available, else fall back to raw bytes
                        if data_str.startswith("gs://") or data_str.startswith(
                            "https://storage.googleapis.com/",
                        ):
                            try:
                                from datetime import timedelta as _timedelta

                                storage_client = getattr(
                                    getattr(ih, "_manager", object),
                                    "storage_client",
                                    None,
                                )
                                if storage_client is not None:
                                    from urllib.parse import urlparse as _urlparse

                                    parsed_url = _urlparse(data_str)
                                    bucket_name = ""
                                    object_path = ""
                                    if parsed_url.scheme == "gs":
                                        bucket_name = parsed_url.netloc
                                        object_path = parsed_url.path.lstrip("/")
                                    elif (
                                        parsed_url.hostname == "storage.googleapis.com"
                                    ):
                                        parts = parsed_url.path.lstrip("/").split(
                                            "/",
                                            1,
                                        )
                                        if len(parts) == 2:
                                            bucket_name, object_path = parts
                                    bucket = storage_client.bucket(bucket_name)
                                    blob = bucket.blob(object_path)
                                    signed_url = blob.generate_signed_url(
                                        version="v4",
                                        expiration=_timedelta(hours=1),
                                        method="GET",
                                    )
                                    content_block = {
                                        "type": "image_url",
                                        "image_url": {"url": signed_url},
                                    }
                                else:
                                    raise RuntimeError("no storage client")
                            except Exception:
                                raw = ih.raw()  # type: ignore[attr-defined]
                                import base64 as _b64  # local import

                                head = (
                                    bytes(raw[:10])
                                    if isinstance(raw, (bytes, bytearray))
                                    else b""
                                )
                                if head.startswith(b"\xff\xd8"):
                                    mime = "image/jpeg"
                                elif head.startswith(b"\x89PNG\r\n\x1a\n"):
                                    mime = "image/png"
                                else:
                                    mime = "image/png"
                                b64 = _b64.b64encode(raw).decode("ascii")
                                content_block = {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                                }
                        else:
                            content_block = {
                                "type": "image_url",
                                "image_url": {"url": data_str},
                            }
                    else:
                        # Fallback to raw bytes from handle
                        raw = ih.raw()  # type: ignore[attr-defined]
                        import base64 as _b64  # local import

                        head = (
                            bytes(raw[:10])
                            if isinstance(raw, (bytes, bytearray))
                            else b""
                        )
                        if head.startswith(b"\xff\xd8"):
                            mime = "image/jpeg"
                        elif head.startswith(b"\x89PNG\r\n\x1a\n"):
                            mime = "image/png"
                        else:
                            mime = "image/png"
                        b64 = _b64.b64encode(raw).decode("ascii")
                        content_block = {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        }

                    content_blocks.append(content_block)
                except Exception:
                    # Skip malformed handle entries; continue attaching the rest
                    continue

        # Append a single composite user message so the vision model can "see" the images with the instruction
        try:
            self._llm.messages.append({"role": "user", "content": content_blocks})
        except Exception:
            pass

        # Compose prompt (kept consistent with previous behaviour)
        if self._entrypoint_info:
            fn = self._entrypoint_info
            prompt = (
                "You are mid-execution of a function-driven simulated task.\n"
                f"Task: {self._description}\n"
                f"Entrypoint: {fn.get('name')} {fn.get('argspec','')} (id={fn.get('function_id')})\n"
                f"Docstring:\n{fn.get('docstring','')}\n\n"
                "Use any images attached in the most recent user message as ground truth for visual details."
            )
        else:
            prompt = (
                f"Current simulated actions:\n{self._description}\n\n"
                "Use any images attached in the most recent user message as ground truth for visual details."
            )
        # Unified LLM roundtrip (includes timing, gated body, and optional dumps)
        try:
            _sys = getattr(self._llm, "system_message", None)
        except Exception:
            _sys = None
        answer = await simulated_llm_roundtrip(
            self._llm,
            label=self._log_label,
            prompt=prompt,
        )

    async def pause(self) -> str:
        if not self._description:
            raise Exception("The actor is not running, so nothing to pause.")
        if self._paused:
            return "Actor is already paused."
        self._paused = True
        self._pause_event.clear()
        # Freeze clock by reducing remaining duration and clearing start marker
        if self._remaining_duration is not None and self._last_started_at is not None:
            elapsed = time.monotonic() - self._last_started_at
            self._remaining_duration = max(0.0, self._remaining_duration - elapsed)
            self._last_started_at = None
        self.simulate_step()
        self._log_pause()
        return f"Paused '{self._description}'."

    async def resume(self) -> str:
        if not self._description:
            raise Exception("No actor is running, so nothing to resume.")
        if not self._paused:
            return "Actor is already running."
        self._paused = False
        self._pause_event.set()
        # Restart the clock from now (remaining duration preserved)
        if self._remaining_duration is not None:
            self._last_started_at = time.monotonic()
        self.simulate_step()
        self._log_resume()
        return f"Resumed '{self._description}'."

    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
        images: object | None = None,
    ) -> SteerableToolHandle:
        """Ask a question about the current state.

        Args:
            question: The question to ask.
            _parent_chat_context: Optional parent chat context for the inspection loop.
                Accepted for API parity with real handles but not currently used.
            images: Optional image references. Accepted for API parity with real handles
                but not currently used.

        Returns:
            A SteerableToolHandle whose result() returns the answer string.
        """
        if not self._description:
            raise Exception("No actions are currently being performed.")
        self.simulate_step()
        # Build a concise child label consistent with simulated scheduler
        try:
            q_label = SimulatedLineage.question_label(self._log_label)
        except Exception:
            q_label = f"Question({getattr(self, '_log_label', 'SimulatedActor.act')})"
        try:
            SimulatedLog.log_request("ask", q_label, question)
        except Exception:
            pass

        if self._entrypoint_info:
            fn = self._entrypoint_info
            prompt = (
                "You are executing a simulated function as part of a task. Answer briefly.\n"
                f"Task: {self._description}\n"
                f"Entrypoint: {fn.get('name')} {fn.get('argspec','')} (id={fn.get('function_id')})\n"
                f"Docstring:\n{fn.get('docstring','')}\n\n"
                f"User asks: {question}"
            )
        else:
            prompt = (
                f"You are working on simulating these actions:\n{self._description}\n\n"
                f"User asks: {question}"
            )
        try:
            _sys = getattr(self._llm, "system_message", None)
        except Exception:
            _sys = None
        answer = await simulated_llm_roundtrip(
            self._llm,
            label=q_label,
            prompt=prompt,
        )
        return _StaticAnswerHandle(answer)

    def done(self) -> bool:
        return self._done_event.is_set()

    def trigger_completion(self, result: str | None = None) -> None:
        """Trigger immediate completion of the simulated actor.

        This is a test-only method that forces the actor to complete immediately,
        unblocking any awaiting result() calls. Useful for deterministic testing
        without relying on step counts or durations.

        Args:
            result: Optional result string. If not provided, uses a default
                    completion message.

        Note: Idempotent - calling on an already-completed actor has no effect.
        """
        if self._done_event.is_set():
            return  # Already done, no-op

        msg = (
            result
            if result is not None
            else f"Completed '{self._description}' (triggered)."
        )
        self._complete(msg)

    # ------------------------
    # Status query helpers
    # ------------------------
    def _emit_status(self, message: str) -> None:
        """Emit a status line to the central logger so it reaches the broadcast port."""
        try:
            LOGGER.info(
                f"[{getattr(self, '_log_label', 'SimulatedActor.act')}] {message}",
            )
        except Exception:
            pass

    def get_remaining_duration_seconds(self) -> float | None:
        """Return the current wall-clock seconds remaining until auto-completion, or None.

        When paused, this returns the frozen remaining amount. When running, it
        subtracts the elapsed time since the last start/resume.
        """
        if self._remaining_duration is None:
            return None
        if self._last_started_at is None:
            return max(0.0, float(self._remaining_duration))
        elapsed = time.monotonic() - self._last_started_at
        return max(0.0, float(self._remaining_duration) - float(elapsed))

    def get_remaining_steps(self) -> int | None:
        """Return remaining steps until auto-completion, or None if unlimited."""
        if self._steps is None:
            return None
        try:
            return max(0, int(self._steps) - int(self._steps_taken))
        except Exception:
            return None

    # ------------------------
    # Event APIs required by SteerableToolHandle
    # ------------------------
    async def next_clarification(self) -> dict:
        # If no clarification queue, block until the action completes
        # (similar to next_notification when emit_notifications=False)
        # Use polling instead of asyncio.to_thread() to allow clean cancellation
        if self._clarification_up_q is None:
            while not self._done_event.is_set():
                await asyncio.sleep(0.1)
            return {}

        try:
            msg = await self._clarification_up_q.get()
            return {
                "type": "clarification",
                "call_id": "unknown",
                "tool_name": "simulated_actor",
                "question": msg,
            }
        except Exception:
            pass
        return {}

    async def next_notification(self) -> dict:
        # If notifications are disabled, block until the action completes
        # Use polling instead of asyncio.to_thread() to allow clean cancellation
        if not self._emit_notifications:
            while not self._done_event.is_set():
                await asyncio.sleep(0.1)
            return {}

        # Report progress without consuming steps (observing isn't work)
        # Compose a small progress message consistent with the configured mode
        try:
            desc = str(self._description) if self._description else "activity"
        except Exception:
            desc = "activity"

        message = f"Progress update for '{desc}'."
        try:
            rem_steps = self.get_remaining_steps()
            rem_secs = self.get_remaining_duration_seconds()
            if rem_steps is not None:
                message = f"Progress update: working on '{desc}'. Steps remaining: {max(0, rem_steps)}"
            elif rem_secs is not None:
                # Round to one decimal place for readability
                try:
                    rem_str = f"{max(0.0, float(rem_secs)):.1f}s"
                except Exception:
                    rem_str = str(rem_secs)
                message = (
                    f"Progress update: working on '{desc}'. Time remaining: {rem_str}"
                )
        except Exception:
            # Fall back to the generic message
            pass

        try:
            SimulatedLog.log_notification(self._log_label, message)
        except Exception:
            pass
        return {
            "type": "notification",
            "tool_name": "simulated_actor",
            "message": message,
        }

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        try:
            if self._clarification_down_q is not None:
                await self._clarification_down_q.put(answer)
        except Exception:
            pass


class SimulatedActor(BaseActor):
    def __init__(
        self,
        *,
        steps: int | None = None,
        duration: float | None = None,
        _requests_clarification: bool = False,
        log_mode: "str | None" = "log",
        # New: simulation-only guidance (does not alter TaskScheduler flow)
        simulation_guidance: Optional[str] = None,
        # Whether handles emit notifications via next_notification()
        emit_notifications: bool = True,
        # Accept but ignore parameters that real Actor may use
        description: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Initialize a simulated actor.

        Args:
            steps:      *(Optional)* Maximum tool steps each activity should run
                        before auto-completion.
            duration:   *(Optional)* Maximum wall-clock seconds before an activity
                        auto-completes. Pauses do not count toward this limit.
            emit_notifications: Whether handles should emit notifications via
                        next_notification(). When False, next_notification() blocks
                        until the action completes. Defaults to True.
        """
        self._steps = steps
        self._duration = duration
        self._requests_clarification = _requests_clarification
        self._log_mode: str | None = (
            log_mode if log_mode in ("print", "log", None) else "log"
        )
        self._emit_notifications = emit_notifications
        # Store simulation-only guidance
        self._sim_guidance: Optional[str] = simulation_guidance

        # One shared, memory-retaining LLM for all activities
        self._llm = new_llm_client(stateful=True)
        # Compose a system message that preserves default behaviour while
        # allowing optional simulation guidance to influence simulated responses.
        _base_sys = (
            "You are a simulated actor and executor. "
            "Invent plausible progress and remain internally consistent "
            "across multiple calls."
        )
        if self._sim_guidance:
            _base_sys += (
                "\n\nSimulation guidance (influences the simulation only; do not reinterpret the task description):\n"
                f"- {self._sim_guidance.strip()}"
            )
        self._llm.set_system_message(_base_sys)

    @functools.wraps(BaseActor.act, updated=())
    async def act(
        self,
        description: str,
        *,
        clarification_enabled: bool = True,
        response_format: Optional[Type[BaseModel]] = None,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        # optional function entrypoint id
        entrypoint: Optional[int] = None,
        # New: optional session suffix to reuse across simulated logs
        session_suffix: Optional[str] = None,
        **kwargs,
    ) -> SimulatedActorHandle:
        if not clarification_enabled:
            _clarification_up_q = None
            _clarification_down_q = None

        # Emit a scheduler-like nested log for starting an action
        try:
            parts = SimulatedLineage.parent_lineage()
        except Exception:
            parts = []
        try:
            if session_suffix:
                _act_label = SimulatedLineage.make_label_with_suffix(
                    "SimulatedActor.act",
                    session_suffix,
                )
            else:
                _act_label = SimulatedLineage.make_label("SimulatedActor.act")
        except Exception:
            _act_label = "SimulatedActor.act"
        # Tool-style scheduled log (only when no parent lineage)
        try:
            from unity.common.simulated import (  # noqa: WPS433
                maybe_tool_log_scheduled_with_label as _log_sched,
            )

            _log_sched(
                _act_label,
                "act",
                {"description": description},
            )
        except Exception:
            pass

        entrypoint_info: dict | None = None
        planned_result: str | None = None

        # If an entrypoint is provided, fetch real function metadata/code and prebake a result
        if entrypoint is not None:
            try:
                fm = ManagerRegistry.get_function_manager()
                log = fm._get_log_by_function_id(function_id=int(entrypoint), raise_if_missing=True)  # type: ignore[attr-defined]
                ent = log.entries if hasattr(log, "entries") else {}
                entrypoint_info = {
                    "function_id": ent.get("function_id", entrypoint),
                    "name": ent.get("name"),
                    "argspec": ent.get("argspec"),
                    "docstring": ent.get("docstring") or "",
                    "implementation": ent.get("implementation") or "",
                }

                # Compose a concise final completion sentence consistent with the function
                impl = entrypoint_info.get("implementation", "")
                name = entrypoint_info.get("name") or f"function_{entrypoint}"
                sig = entrypoint_info.get("argspec", "")
                doc = entrypoint_info.get("docstring", "")
                prompt = (
                    "You are simulating the execution of a Python function inside a task.\n"
                    "Return ONE short past-tense sentence that STARTS with 'Completed',\n"
                    "summarising the concrete outcome of running the function in the context below.\n"
                    "Do not include code or steps. Keep it under two sentences.\n\n"
                    f"Task description: {description}\n"
                    f"Function: {name} {sig} (id={entrypoint})\n"
                    f"Docstring:\n{doc}\n\n"
                    f"Implementation:\n{impl}"
                )
                planned_result = await self._llm.generate(prompt)
                if not isinstance(planned_result, str) or not planned_result.strip():
                    planned_result = None
            except Exception:
                entrypoint_info = None
                planned_result = None

        # Construct the simulated handle with optional entrypoint context
        return SimulatedActorHandle(
            self._llm,
            description,
            steps=self._steps,
            duration=self._duration,
            parent_chat_context=_parent_chat_context,
            _requests_clarification=self._requests_clarification,
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
            log_mode=self._log_mode,
            entrypoint_info=entrypoint_info,
            planned_result=planned_result,
            session_suffix=session_suffix,
            response_format=response_format,
            emit_notifications=self._emit_notifications,
        )
