import asyncio
import os
import json
import threading
import time

import unify
from ..constants import LOGGER
from .base import BaseActor
from typing import Optional


class SimulatedActorHandle:
    """
    A lightweight, actor-scoped handle for simulating execution of a series of actions.

    This mirrors the public surface expected by higher layers/tests:
    - ask(question) -> str
    - interject(instruction) -> None
    - pause() / resume() -> str
    - stop(reason) -> str
    - result() -> str (async)
    - done() -> bool
    - valid_tools (property)
    """

    def __init__(
        self,
        llm: unify.AsyncUnify,
        description: str,
        *,
        steps: int | None,
        duration: float | None = None,
        parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        log_mode: "str | None" = "log",
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
                        self._clarification_up_q.put_nowait(
                            "Can you please clarify what exactly you'd like me to do?",
                        )
                    except asyncio.QueueFull:
                        pass
                    while True:
                        try:
                            answer: str = self._clarification_down_q.get_nowait()
                            break
                        except asyncio.QueueEmpty:
                            time.sleep(0.05)
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
                    self._complete(
                        f"Completed '{description}' after {self._duration}\u2009s duration.",
                    )
                    return
                if self._steps is not None and self._steps_taken >= (self._steps or 0):
                    self._complete(
                        f"Completed '{description}' in {self._steps} steps.",
                    )
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
                                f"⏳ SimulatedActor Duration remaining: {max(0.0, rem):.1f}s",
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
            # Emit steps remaining after each user-visible interaction that consumes a step
            try:
                if self._steps is not None:
                    remaining = max(0, int(self._steps) - int(self._steps_taken))
                    self._emit_status(f"🪜 SimulatedActor Steps remaining: {remaining}")
            except Exception:
                pass

    async def result(self) -> str:
        await asyncio.to_thread(self._done_event.wait)
        return self._result_str  # type: ignore

    def stop(self, reason: Optional[str] = None) -> str:
        if not self._description:
            raise Exception("No actions are currently being performed.")
        msg = f"Stopped '{self._description}' for reason: {reason}"
        self._complete(msg)
        return msg

    async def interject(self, instruction: str) -> None:
        if not self._description:
            raise Exception("No actions are currently being performed.")
        self.simulate_step()
        prompt = (
            f"Current simulated actions:\n{self._description}\n\n"
            f"User instruction to adjust the plan:\n{instruction}"
        )
        await self._llm.generate(prompt)

    def pause(self) -> str:
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
        return f"Paused '{self._description}'."

    def resume(self) -> str:
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
        return f"Resumed '{self._description}'."

    async def ask(self, question: str) -> str:
        if not self._description:
            raise Exception("No actions are currently being performed.")
        self.simulate_step()
        prompt = (
            f"You are working on simulating these actions:\n{self._description}\n\n"
            f"User asks: {question}"
        )
        return await self._llm.generate(prompt)

    def done(self) -> bool:
        return self._done_event.is_set()

    @property
    def valid_tools(self):
        if self._description is None:
            return {}
        available = {
            self.stop.__name__: self.stop,
            self.interject.__name__: self.interject,
            self.ask.__name__: self.ask,
        }
        if self._paused:
            available[self.resume.__name__] = self.resume
        else:
            available[self.pause.__name__] = self.pause
        return available

    # ------------------------
    # Status query helpers
    # ------------------------
    def _emit_status(self, message: str) -> None:
        """Emit a status line according to configured log mode: print | log | None."""
        try:
            if self._log_mode == "print":
                print(message)
            elif self._log_mode == "log":
                try:
                    LOGGER.info(message)
                except Exception:
                    pass
            else:
                # None ⇒ suppressed
                pass
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
    ) -> None:
        """
        Initialize a simulated actor.

        Args:
            steps:      *(Optional)* Maximum tool steps each activity should run
                        before auto-completion.
            duration:   *(Optional)* Maximum wall-clock seconds before an activity
                        auto-completes. Pauses do not count toward this limit.
        """
        self._steps = steps
        self._duration = duration
        self._requests_clarification = _requests_clarification
        self._log_mode: str | None = (
            log_mode if log_mode in ("print", "log", None) else "log"
        )
        # Store simulation-only guidance
        self._sim_guidance: Optional[str] = simulation_guidance

        # One shared, memory-retaining LLM for all activities
        self._llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
            stateful=True,
        )
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

    async def act(
        self,
        description: str,
        *,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SimulatedActorHandle:
        # Pass the original TaskScheduler-provided description unchanged.
        return SimulatedActorHandle(
            self._llm,
            description,
            steps=self._steps,
            duration=self._duration,
            parent_chat_context=parent_chat_context,
            _requests_clarification=self._requests_clarification,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
            log_mode=self._log_mode,
        )
