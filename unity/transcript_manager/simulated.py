# unity/transcript_manager/simulated_transcript_manager.py
from __future__ import annotations

import asyncio
import json
import os
import threading
import functools
import inspect
import ast
from typing import List, Optional, Dict, Any

import unify

from ..common.llm_helpers import SteerableToolHandle
from .base import BaseTranscriptManager
from .transcript_manager import TranscriptManager  # real implementation
from ..contact_manager.contact_manager import ContactManager as _RealContactManager
from .prompt_builders import (
    build_ask_prompt,
    build_simulated_method_prompt,
)
from ..events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
    wrap_handle_with_logging,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to mirror the real TranscriptManager's tool list programmatically
# ─────────────────────────────────────────────────────────────────────────────


def _extract_tm_tool_attrs_from_real() -> list[tuple[str, str]]:
    """
    Parse TranscriptManager.__init__ to recover the methods passed into
    methods_to_tool_dict for self._tools.

    Returns a list of (owner, attr_name) where owner ∈ {"TranscriptManager", "ContactManager"}.
    """
    try:
        src = inspect.getsource(TranscriptManager.__init__)
    except Exception:
        return []

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    results: list[tuple[str, str]] = []

    class _InitVisitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:  # type: ignore[override]
            # Look for: self._tools = methods_to_tool_dict(...)
            # Ensure target is self._tools
            if not node.targets:
                return
            target = node.targets[0]
            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "_tools"
            ):
                return
            value = node.value
            if not isinstance(value, ast.Call):
                return
            func = value.func
            if not (
                (isinstance(func, ast.Name) and func.id == "methods_to_tool_dict")
                or (
                    isinstance(func, ast.Attribute)
                    and func.attr == "methods_to_tool_dict"
                )
            ):
                return
            # Extract arguments – they are attribute nodes like self._contact_manager._filter_contacts
            for arg in value.args:
                owner = None
                name = None
                cur = arg
                # Walk down attribute chain to find the right-most attribute name
                # and identify whether it came from self._contact_manager or self
                if isinstance(cur, ast.Attribute):
                    # Capture the tail name
                    tail = cur.attr
                    # Ascend to see the root
                    root = cur.value
                    if isinstance(root, ast.Attribute) and isinstance(
                        root.value,
                        ast.Name,
                    ):
                        if root.value.id == "self" and root.attr == "_contact_manager":
                            owner = "ContactManager"
                            name = tail
                    elif isinstance(root, ast.Name) and root.id == "self":
                        owner = "TranscriptManager"
                        name = tail
                if owner and name:
                    results.append((owner, name))

    _InitVisitor().visit(tree)
    return results


def _build_tools_from_real_tm() -> Dict[str, Any]:
    """
    Build a tools-dict mirroring the real TranscriptManager's tools for prompt construction.
    Uses a static fallback mirroring current real tools if reflection fails.
    """
    mapping = _extract_tm_tool_attrs_from_real()
    methods: list[Any] = []
    if mapping:
        for owner, name in mapping:
            try:
                if owner == "ContactManager":
                    methods.append(getattr(_RealContactManager, name))
                elif owner == "TranscriptManager":
                    methods.append(getattr(TranscriptManager, name))
            except Exception:
                # Ignore missing attrs and fall back below
                methods = []
                break
    if methods:
        from ..common.llm_helpers import methods_to_tool_dict as _m2t

        return _m2t(*methods, include_class_name=False)

    # Fallback – keep aligned with current implementation
    from ..common.llm_helpers import methods_to_tool_dict as _m2t

    return _m2t(
        _RealContactManager._filter_contacts,
        TranscriptManager._filter_messages,
        TranscriptManager._search_messages,
        include_class_name=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────
class _SimulatedTranscriptHandle(SteerableToolHandle):
    """
    A very small, LLM-backed handle used by SimulatedTranscriptManager.ask.
    """

    def __init__(
        self,
        llm: unify.Unify,
        initial_text: str,
        *,
        _return_reasoning_steps: bool,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None,
        clarification_down_q: asyncio.Queue[str] | None,
    ):
        self._llm = llm
        self._initial = initial_text
        self._want_steps = _return_reasoning_steps
        self._clar_up_q = clarification_up_q
        self._clar_down_q = clarification_down_q
        if _requests_clarification and (
            not clarification_up_q or not clarification_down_q
        ):
            raise ValueError(
                "Clarification queues must be provided when _requests_clarification is True",
            )
        self._needs_clar = _requests_clarification

        # fire clarification immediately if queues supplied
        if self._needs_clar:
            try:
                self._clar_up_q.put_nowait(
                    "Could you clarify your information-need around the transcripts?",
                )
            except asyncio.QueueFull:
                pass

        self._extra_user_msgs: List[str] = []

        # completion primitives
        self._done = threading.Event()
        self._cancelled = False
        self._answer: Optional[str] = None
        self._msgs: List[Dict[str, Any]] = []
        self._paused = False

    # ──  API expected by SteerableToolHandle  ──────────────────────────────
    async def result(self):
        if self._cancelled:
            raise asyncio.CancelledError()

        while self._paused and not self._cancelled:
            await asyncio.sleep(0.05)

        if not self._done.is_set():
            # wait for clarification reply if requested
            if self._needs_clar:
                clar_reply = await self._clar_down_q.get()
                self._extra_user_msgs.append(f"Clarification: {clar_reply}")

            prompt = "\n\n---\n\n".join([self._initial] + self._extra_user_msgs)
            answer = await self._llm.generate(prompt)
            self._answer = answer
            self._msgs = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            self._done.set()

        if self._want_steps:
            return self._answer, self._msgs
        return self._answer

    def interject(self, message: str) -> str:
        if self._cancelled:
            return "Interaction has been stopped."
        self._extra_user_msgs.append(message)
        return "Acknowledged."

    def stop(self) -> str:
        self._cancelled = True
        self._done.set()
        return "Stopped."

    def pause(self) -> str:
        if self._paused:
            return "Already paused."
        self._paused = True
        return "Paused."

    def resume(self) -> str:
        if not self._paused:
            return "Already running."
        self._paused = False
        return "Resumed."

    def done(self) -> bool:
        return self._done.is_set()

    @property
    def valid_tools(self):
        tools = {
            self.interject.__name__: self.interject,
            self.stop.__name__: self.stop,
        }
        if self._paused:
            tools[self.resume.__name__] = self.resume
        else:
            tools[self.pause.__name__] = self.pause
        return tools

    async def ask(self, question: str) -> "SteerableToolHandle":
        q_msg = (
            f"Your only task is to simulate an answer to the following question: {question}\n\n"
            "However, there is a also ongoing simulated process which had the instructions given below. "
            "Please make your answer realastic and conceivable given the provided context of the simulated taks."
        )
        follow_up_prompt = "\n\n---\n\n".join(
            [q_msg]
            + [self._initial]
            + self._extra_msgs
            + [f"Question to answer (as a reminder!): {question}"],
        )

        return _SimulatedTranscriptHandle(
            self._llm,
            follow_up_prompt,
            _return_reasoning_steps=self._want_steps,
            _requests_clarification=False,
            clarification_up_q=self._clar_up_q,
            clarification_down_q=self._clar_down_q,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public Simulated Manager
# ─────────────────────────────────────────────────────────────────────────────
class SimulatedTranscriptManager(BaseTranscriptManager):
    """
    Lightweight, fake implementation of TranscriptManager that only uses an
    LLM to invent plausible answers.  Suitable for offline demos and tests
    where the real storage layer is unnecessary.
    """

    def __init__(
        self,
        description: str = "nothing fixed, make up some imaginary scenario",
        *,
        log_events: bool = False,
        rolling_summary_in_prompts: bool = True,
    ) -> None:
        self._description = description
        self._log_events = log_events
        self._rolling_summary_in_prompts = rolling_summary_in_prompts

        # Shared, *stateful* **asynchronous** LLM
        self._llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )
        tools_for_prompt = _build_tools_from_real_tm()
        ask_sys = build_ask_prompt(
            tools_for_prompt,
            include_activity=self._rolling_summary_in_prompts,
        )

        self._llm.set_system_message(
            "You are a *simulated* transcript assistant. "
            "There is **no** backing datastore – create plausible yet "
            "self-consistent answers.\n\n"
            "For reference, here are the *real* system messages used by the "
            "production implementation:\n"
            f"\n\n'ask' system message:\n{ask_sys}\n\n"
            f"Back-story: {self._description}",
        )

    # --------------------------------------------------------------------- #
    # ask                                                                   #
    # --------------------------------------------------------------------- #
    @functools.wraps(BaseTranscriptManager.ask, updated=())
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        should_log = self._log_events or log_events
        call_id = None

        if should_log:
            call_id = new_call_id()
            await publish_manager_method_event(
                call_id,
                "TranscriptManager",
                "ask",
                phase="incoming",
                question=text,
            )

        instruction = build_simulated_method_prompt(
            "ask",
            text,
            parent_chat_context=parent_chat_context,
        )
        handle = _SimulatedTranscriptHandle(
            self._llm,
            instruction,
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "TranscriptManager",
                "ask",
            )

        return handle
