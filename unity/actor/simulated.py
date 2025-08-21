import asyncio
import os
import json
import time
import threading

import unify
from .base import BaseActor
from typing import Optional, List, Dict, Callable


class _SimulatedActorHandle:
    """
    Minimal steerable handle for the simulated actor's activity.
    Implements the :class:`unity.common.llm_helpers.SteerableToolHandle` surface
    without referencing tasks.
    """

    def __init__(
        self,
        llm: "unify.AsyncUnify",
        description: str,
        *,
        steps: int | None,
        timeout: float | None,
        parent_chat_context: list[dict] | None,
        requests_clarification: bool,
        clarification_up_q: asyncio.Queue[str] | None,
        clarification_down_q: asyncio.Queue[str] | None,
    ) -> None:
        self._llm = llm
        self._description = description
        self._steps_budget = steps
        self._timeout = timeout
        self._parent_chat_context = parent_chat_context
        self._requests_clarification = requests_clarification
        self._clar_up_q = clarification_up_q
        self._clar_down_q = clarification_down_q

        # runtime state
        self._interjections: List[str] = []
        self._paused = False
        self._done_event = threading.Event()
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._start_time: float | None = None
        self._steps_taken = 0
        self._step_lock = threading.Lock()
        self._result: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

        self._start()

    # lifecycle
    def _start(self) -> None:
        self._paused = False
        self._pause_event.set()
        self._stop_event.clear()
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _complete(self, message: str) -> None:
        if self._done_event.is_set():
            return
        self._stop_event.set()
        self._result = message
        self._done_event.set()
        if (
            self._thread is not None
            and self._thread.is_alive()
            and threading.current_thread() is not self._thread
        ):
            self._thread.join(timeout=1)

    def _run(self) -> None:
        try:
            # optional clarification path
            if self._requests_clarification and self._clar_up_q and self._clar_down_q:
                try:
                    self._clar_up_q.put_nowait(
                        "Could you clarify the instruction before I proceed?",
                    )
                except asyncio.QueueFull:
                    pass
                # wait for answer non-blocking
                while True:
                    if self._stop_event.is_set():
                        return
                    try:
                        answer: str = self._clar_down_q.get_nowait()  # type: ignore[attr-defined]
                        break
                    except Exception:
                        time.sleep(0.05)
                self._complete(f"Clarification received: {answer}")
                return

            # normal simulated progress
            while not self._stop_event.is_set():
                # timeout check
                if (
                    self._timeout is not None
                    and self._start_time is not None
                    and (time.monotonic() - self._start_time) >= self._timeout
                ):
                    self._complete(
                        f"Completed activity after {self._timeout}\u2009s timeout.",
                    )
                    return

                # steps budget
                if (
                    self._steps_budget is not None
                    and self._steps_taken >= self._steps_budget
                ):
                    self._complete(
                        f"Completed activity in {self._steps_budget} steps.",
                    )
                    return

                # honor pause
                self._pause_event.wait()
                time.sleep(0.1)
        finally:
            self._pause_event.set()
            self._stop_event.clear()
            self._thread = None

    # SteerableToolHandle API
    async def result(self) -> str:
        await asyncio.to_thread(self._done_event.wait)
        return self._result or ""

    def interject(self, message: str) -> str:
        if self._done_event.is_set():
            return "Interaction already finished."
        self._interjections.append(message)
        return "Noted."

    def stop(self, reason: Optional[str] = None) -> str:
        if self._done_event.is_set():
            return "Already stopped."
        msg = "Stopped." if reason is None else f"Stopped: {reason}"
        self._complete(msg)
        return msg

    def pause(self) -> str:
        if self._done_event.is_set():
            return "Already finished."
        if self._paused:
            return "Already paused."
        self._paused = True
        self._pause_event.clear()
        with self._step_lock:
            self._steps_taken += 1
        return "Paused."

    def resume(self) -> str:
        if self._done_event.is_set():
            return "Already finished."
        if not self._paused:
            return "Already running."
        self._paused = False
        self._pause_event.set()
        with self._step_lock:
            self._steps_taken += 1
        return "Resumed."

    def done(self) -> bool:
        return self._done_event.is_set()

    @property
    def valid_tools(self) -> Dict[str, Callable]:
        tools: Dict[str, Callable] = {
            self.interject.__name__: self.interject,
            self.stop.__name__: self.stop,
        }
        if self._paused:
            tools[self.resume.__name__] = self.resume
        else:
            tools[self.pause.__name__] = self.pause
        return tools

    async def ask(self, question: str) -> str:
        prompt = (
            f"You are carrying out the following activity:\n{self._description}\n\n"
            f"User asks: {question}"
        )
        with self._step_lock:
            self._steps_taken += 1
        return await self._llm.generate(prompt)


class SimulatedActor(BaseActor):
    def __init__(
        self,
        *,
        steps: int | None = None,
        timeout: float | None = None,
        _requests_clarification: bool = False,
    ) -> None:
        """
        Initialize a simulated actor.

        Args:
            steps:      *(Optional)* Maximum tool steps each activity should run
                        before auto-completion.
            timeout:    *(Optional)* Maximum wall-clock seconds before an activity
                        auto-completes.
        """
        self._steps = steps
        self._timeout = timeout
        self._requests_clarification = _requests_clarification

        # One shared, memory-retaining LLM for all activities
        self._llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
            stateful=True,
        )
        self._llm.set_system_message(
            "You are a simulated actor and executor. "
            "Invent plausible progress and remain internally consistent "
            "across multiple calls.",
        )

    async def act(
        self,
        description: str,
        *,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> _SimulatedActorHandle:
        return _SimulatedActorHandle(
            self._llm,
            description,
            steps=self._steps,
            timeout=self._timeout,
            parent_chat_context=parent_chat_context,
            requests_clarification=self._requests_clarification,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )
