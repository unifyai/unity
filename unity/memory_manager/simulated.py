from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional

import unify

from .base import BaseMemoryManager
from ..common.llm_helpers import SteerableToolHandle


# ––– tiny helper handle ––––––––––––––––––––––––––––––––––––––––––––––––––––
class _SimulatedMemoryHandle(SteerableToolHandle):
    def __init__(
        self,
        llm: unify.Unify,
        initial_prompt: str,
        *,
        _return_reasoning_steps: bool,
    ) -> None:
        self._llm = llm
        self._prompt = initial_prompt
        self._want_steps = _return_reasoning_steps
        self._done = threading.Event()
        self._answer: Optional[str] = None
        self._messages: List[Dict[str, Any]] = []

    async def result(self):
        if not self._done.is_set():
            self._answer = await self._llm.generate(self._prompt)
            self._messages = [
                {"role": "user", "content": self._prompt},
                {"role": "assistant", "content": self._answer},
            ]
            self._done.set()
        return (self._answer, self._messages) if self._want_steps else self._answer

    # trivial stubs – no real interactivity required for tests / demos
    def pause(self):
        return "Paused (simulated)."

    def resume(self):
        return "Resumed (simulated)."

    def stop(self):
        self._done.set()
        return "Stopped (simulated)."

    def interject(self, message: str):
        return "Cannot interject – simulated handle."  # no‑op

    def done(self):
        return self._done.is_set()

    @property
    def valid_tools(self):
        return {}


# ––– simulated manager ––––––––––––––––––––––––––––––––––––––––––––––––––––
class SimulatedMemoryManager(BaseMemoryManager):
    """Lightweight stand‑in that *imagines* all operations."""

    def __init__(self) -> None:
        self._llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )
        # no tools; we just answer directly

    async def _fake(
        self,
        prompt: str,
        *,
        _return_reasoning_steps: bool,
    ) -> SteerableToolHandle:
        return _SimulatedMemoryHandle(
            self._llm,
            prompt,
            _return_reasoning_steps=_return_reasoning_steps,
        )

    async def update_contacts(
        self,
        transcript: str,
        *,
        _return_reasoning_steps: bool = False,
        **_,
    ) -> SteerableToolHandle:  # noqa: D401,E501 – signature inherited
        prompt = "Simulate update_contacts. Transcript:\n" + transcript
        return await self._fake(prompt, _return_reasoning_steps=_return_reasoning_steps)

    async def update_contact_bio(
        self,
        transcript: str,
        latest_bio: Optional[str] = None,
        *,
        _return_reasoning_steps: bool = False,
        **_,
    ) -> SteerableToolHandle:  # noqa: D401,E501
        prompt = (
            "Simulate update_contact_bio. Transcript:\n"
            + transcript
            + "\nExisting bio:"
            + str(latest_bio)
        )
        return await self._fake(prompt, _return_reasoning_steps=_return_reasoning_steps)

    async def update_contact_rolling_summary(
        self,
        transcript: str,
        latest_rolling_summary: Optional[str] = None,
        *,
        _return_reasoning_steps: bool = False,
        **_,
    ) -> SteerableToolHandle:  # noqa: D401,E501
        prompt = (
            "Simulate update_contact_rolling_summary. Transcript:\n"
            + transcript
            + "\nExisting summary:"
            + str(latest_rolling_summary)
        )
        return await self._fake(prompt, _return_reasoning_steps=_return_reasoning_steps)
