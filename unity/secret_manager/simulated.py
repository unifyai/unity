from __future__ import annotations

import asyncio
import functools
import json
from typing import Any, Dict, Optional, Type, TYPE_CHECKING
from pydantic import BaseModel

import unillm
from .base import BaseSecretManager
from .types import Secret
from ..common.llm_client import new_llm_client
from ..common.context_dump import make_messages_safe_for_context_dump
from .prompt_builders import build_ask_prompt, build_update_prompt
from ..common.async_tool_loop import SteerableToolHandle
from ..common.simulated import (
    mirror_secret_manager_tools,
    SimulatedLineage,
    SimulatedLog,
    simulated_llm_roundtrip,
    SimulatedHandleMixin,
    build_followup_prompt,
    maybe_tool_log_scheduled,
    maybe_tool_log_completed,
)
from ..logger import LOGGER


class _SimulatedSecretHandle(SimulatedHandleMixin, SteerableToolHandle):
    """Minimal LLM-backed handle used by SimulatedSecretManager.ask / update."""

    def __init__(
        self,
        llm: unillm.Unify,
        initial_text: str,
        *,
        mode: str,
        _return_reasoning_steps: bool,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None,
        clarification_down_q: asyncio.Queue[str] | None,
        response_format: Optional[Type[BaseModel]] = None,
        hold_completion: bool = False,
    ) -> None:
        self._llm = llm
        self._initial = initial_text
        self._mode = str(mode or "ask")
        self._want_steps = _return_reasoning_steps
        self._clar_up_q = clarification_up_q
        self._clar_down_q = clarification_down_q
        self._response_format = response_format
        if _requests_clarification and (
            not clarification_up_q or not clarification_down_q
        ):
            raise ValueError(
                "Clarification queues must be provided when _requests_clarification is True",
            )
        self._needs_clar = _requests_clarification

        # Human-friendly lineage-aware label:
        # "<outer...>->SimulatedSecretManager.<mode>(abcd)"
        self._log_label = SimulatedLineage.make_label(
            f"SimulatedSecretManager.{self._mode}",
        )

        if self._needs_clar:
            try:
                q_text = "Could you clarify your request about secrets?"
                self._clar_up_q.put_nowait(q_text)
                try:
                    SimulatedLog.log_clarification_request(self._log_label, q_text)
                except Exception:
                    pass
                try:
                    LOGGER.info(f"❓ [{self._log_label}] Clarification requested")
                except Exception:
                    pass
            except asyncio.QueueFull:
                pass

        self._extra_msgs: list[str] = []
        self._done = asyncio.Event()
        self._cancelled = False
        self._paused = False
        self._answer: str | None = None
        self._messages: list[dict[str, Any]] = []
        # Async cancellation signal to break clarification waits
        self._cancel_event: asyncio.Event = asyncio.Event()

        self._init_completion_gate(hold_completion)

    async def result(self):
        if self._cancelled:
            raise asyncio.CancelledError()

        while self._paused and not self._cancelled:
            await asyncio.sleep(0.05)

        if not self._done.is_set():
            extra: list[str] = []
            if self._needs_clar and self._clar_down_q is not None:
                try:
                    LOGGER.info(
                        f"⏳ [{self._log_label}] Waiting for clarification answer…",
                    )
                except Exception:
                    pass
                # Race clarification against cancellation
                get_task = asyncio.create_task(self._clar_down_q.get())
                cancel_task = asyncio.create_task(self._cancel_event.wait())
                done, pending = await asyncio.wait(
                    {get_task, cancel_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                if cancel_task in done:
                    # Preserve semantic: result raises when cancelled
                    raise asyncio.CancelledError()
                try:
                    clar = get_task.result()
                except Exception:
                    clar = None
                if clar is not None:
                    extra.append(f"Clarification: {clar}")
                    try:
                        SimulatedLog.log_clarification_answer(self._log_label, clar)
                    except Exception:
                        pass
                    try:
                        LOGGER.info(
                            f"💬 [{self._log_label}] Clarification answer received",
                        )
                    except Exception:
                        pass

            prompt = "\n\n---\n\n".join([self._initial] + self._extra_msgs + extra)
            # Unified simulated roundtrip with lineage-aware logging and gated response preview
            try:
                sys_msg = getattr(self._llm, "system_message", None)
            except Exception:
                sys_msg = None
            answer = await simulated_llm_roundtrip(
                self._llm,
                label=self._log_label,
                prompt=prompt,
                response_format=self._response_format,
            )
            self._answer = answer
            self._messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            self._done.set()

        # If cancellation happened after the coroutine started, raise consistently.
        if self._cancelled:
            raise asyncio.CancelledError()
        if self._want_steps:
            return self._answer, self._messages
        return self._answer

    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
    ) -> None:
        """Interject a message into the in-flight handle.

        Args:
            message: The interjection message to inject.
            _parent_chat_context_cont: Optional continuation of parent chat context.
                Accepted for API parity with real handles but not currently used.
        """
        if self._cancelled:
            return
        self._log_interject(message)
        self._extra_msgs.append(message)

    async def stop(
        self,
        reason: str | None = None,
        **kwargs,
    ) -> None:
        """Stop the in-flight handle.

        Args:
            reason: Optional reason for stopping.
        """
        self._log_stop(reason)
        self._open_completion_gate()
        self._cancelled = True
        try:
            self._cancel_event.set()
        except Exception:
            pass
        try:
            self._done.set()
        except Exception:
            pass

    async def pause(self) -> str:
        if self._paused:
            return "Already paused."
        self._log_pause()
        self._paused = True
        return "Paused."

    async def resume(self) -> str:
        if not self._paused:
            return "Already running."
        self._log_resume()
        self._paused = False
        return "Resumed."

    def done(self) -> bool:  # type: ignore[override]
        return self._done.is_set()

    # --- event APIs required by SteerableToolHandle ---------------------
    async def next_clarification(self) -> dict:
        """Block until a clarification arrives, or forever if not requested."""
        if not getattr(self, "_needs_clar", False):
            return await super().next_clarification()
        try:
            if self._clar_up_q is not None:
                msg = await self._clar_up_q.get()
                return {
                    "type": "clarification",
                    "call_id": "unknown",
                    "tool_name": "unknown",
                    "question": msg,
                }
        except Exception:
            pass
        return await super().next_clarification()

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        try:
            if self._clar_down_q is not None:
                await self._clar_down_q.put(answer)
        except Exception:
            pass

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

    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
    ) -> "SteerableToolHandle":
        """Ask a follow-up question about the current operation.

        Args:
            question: The question to ask.
            parent_chat_context: Optional parent chat context for the inspection loop.
                Accepted for API parity with real handles but not currently used.
        """
        follow_up_prompt = build_followup_prompt(
            question=question,
            initial_instruction=self._initial,
            extra_messages=list(self._extra_msgs),
        )
        handle = _SimulatedSecretHandle(
            self._llm,
            follow_up_prompt,
            mode=self._mode,
            _return_reasoning_steps=self._want_steps,
            _requests_clarification=False,
            clarification_up_q=self._clar_up_q,
            clarification_down_q=self._clar_down_q,
        )
        # Align with other simulated components: concise "Question(<parent>)" label
        try:
            handle._log_label = SimulatedLineage.question_label(self._log_label)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            SimulatedLog.log_request("ask", getattr(handle, "_log_label", ""), question)  # type: ignore[arg-type]
        except Exception:
            pass
        return handle


class SimulatedSecretManager(BaseSecretManager):
    """Drop-in simulated SecretManager with imaginary data and stateful LLM memory."""

    def __init__(
        self,
        description: str = "simulate secret storage and actions (no real database)",
        *,
        simulation_guidance: Optional[str] = None,
        hold_completion: bool = False,
        # Accept but ignore extra parameters for compatibility
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._description = description
        self._simulation_guidance = simulation_guidance
        self._hold_completion = hold_completion

        # Shared, stateful async LLM
        self._llm = new_llm_client(stateful=True)

        # Mirror the real manager's tool exposure using reflection helper
        ask_tools = mirror_secret_manager_tools("ask")
        upd_tools = mirror_secret_manager_tools("update")

        ask_msg = build_ask_prompt(tools=ask_tools).flatten()
        upd_msg = build_update_prompt(tools=upd_tools).flatten()

        # Seed the LLM with a combined system message describing behaviour
        self._llm.set_system_message(
            "You are a simulated secret-manager assistant. "
            "There is no real database; invent plausible secret records and keep your story consistent across turns.\n\n"
            "As a reference, the system messages for the real secret-manager 'ask' and 'update' methods are as follows.\n"
            "You do not have access to real tools here – produce a final answer to the question/request. "
            f"\n\n'ask' system message:\n{ask_msg}\n\n"
            f"\n\n'update' system message:\n{upd_msg}\n\n"
            f"Back-story: {self._description}"
            + (
                f"\n\nGuidance: {self._simulation_guidance}"
                if self._simulation_guidance
                else ""
            ),
        )

    @functools.wraps(BaseSecretManager.clear, updated=())
    def clear(self) -> None:
        sched = maybe_tool_log_scheduled(
            "SimulatedSecretManager.clear",
            "clear",
            {},
        )
        type(self).__init__(
            self,
            description=getattr(
                self,
                "_description",
                "simulate secret storage and actions (no real database)",
            ),
            simulation_guidance=getattr(self, "_simulation_guidance", None),
            hold_completion=getattr(self, "_hold_completion", False),
        )
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(label, cid, "clear", {"outcome": "reset"}, t0)

    @functools.wraps(BaseSecretManager.ask, updated=())
    async def ask(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        # Tool-style scheduled log (only when no parent lineage)
        maybe_tool_log_scheduled(
            "SimulatedSecretManager.ask",
            "ask",
            {"text": text, "requests_clarification": _requests_clarification},
        )

        # Build the simulated instruction (no real tools will run)
        instruction = (
            "Simulate the behaviour of SecretManager.ask for the following user message. "
            "Never reveal any raw secret values; always refer to placeholders like ${name}.\n\n"
            f"User message: {text}\n\n"
            + (
                f"Parent context: {json.dumps(make_messages_safe_for_context_dump(_parent_chat_context))}\n\n"
                if _parent_chat_context
                else ""
            )
        )
        handle = _SimulatedSecretHandle(
            self._llm,
            instruction,
            mode="ask",
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
            response_format=response_format,
        )

        return handle

    async def from_placeholder(self, text: str) -> str:
        """Simulate resolving ${name} placeholders to opaque values (no LLM)."""
        import re

        def _repl(m: "re.Match[str]") -> str:
            name = m.group(1)
            return f"<value:{name}>"

        return re.sub(r"\$\{([^}]+)\}", _repl, text)

    async def to_placeholder(self, text: str) -> str:
        """Simulate redacting known raw values back to ${name} placeholders (no LLM)."""
        result = text
        for name in self._list_secret_keys():
            token = f"<value:{name}>"
            placeholder = f"${{{name}}}"
            if token in result:
                result = result.replace(token, placeholder)

        return result

    @functools.wraps(BaseSecretManager.update, updated=())
    async def update(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        # Tool-style scheduled log (only when no parent lineage)
        maybe_tool_log_scheduled(
            "SimulatedSecretManager.update",
            "update",
            {"text": text, "requests_clarification": _requests_clarification},
        )

        instruction = (
            "Simulate the behaviour of SecretManager.update for the following request. "
            "Never reveal raw secret values; reference secrets via ${name}.\n\n"
            f"User request: {text}\n\n"
            + (
                f"Parent context: {json.dumps(make_messages_safe_for_context_dump(_parent_chat_context))}\n\n"
                if _parent_chat_context
                else ""
            )
        )
        handle = _SimulatedSecretHandle(
            self._llm,
            instruction,
            mode="update",
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
            response_format=response_format,
            hold_completion=self._hold_completion,
        )

        return handle

    # ------------------------------------------------------------------ #
    #  Simulated private helpers (satisfy abstract base)                 #
    # ------------------------------------------------------------------ #
    def _list_secret_keys(self) -> list[str]:
        """Return a deterministic, simulated set of secret names (no I/O)."""
        # Keep this lightweight – callers use this only for display/flow control
        return [
            "api_key",
            "db_password",
            "assistant_email",
        ]

    def _search_secrets(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> list[Secret]:
        """Simulate a semantic search over secrets (returns redacted models)."""
        # Provide a small, stable set without exposing values
        base = [
            Secret(
                secret_id=1,
                name="api_key",
                value="",
                description="API access token",
            ),
            Secret(
                secret_id=2,
                name="db_password",
                value="",
                description="database password",
            ),
        ]
        return base[: max(0, int(k))]

    def _filter_secrets(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list[Secret]:
        """Simulate a filter over secrets (returns redacted models)."""
        rows = [
            Secret(
                secret_id=1,
                name="api_key",
                value="",
                description="API access token",
            ),
            Secret(
                secret_id=2,
                name="db_password",
                value="",
                description="database password",
            ),
        ]
        return rows[offset : offset + limit] if limit is not None else rows[offset:]

    def _create_secret(
        self,
        *,
        name: str,
        value: str,
        description: Optional[str] = None,
    ) -> "ToolOutcome":
        """Simulate secret creation and acknowledge (no persistence)."""
        if not name or not value:
            raise AssertionError("Both name and value are required.")
        return {
            "outcome": "secret created (simulated)",
            "details": {"name": name, "description": description or ""},
        }

    def _update_secret(
        self,
        *,
        name: str,
        value: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "ToolOutcome":
        """Simulate secret update and acknowledge (no persistence)."""
        if value is None and description is None:
            raise ValueError("No updates provided.")
        changes: Dict[str, Any] = {}
        if value is not None:
            changes["value"] = "<redacted>"
        if description is not None:
            changes["description"] = description
        return {
            "outcome": "secret updated (simulated)",
            "details": {"name": name, **changes},
        }

    def _delete_secret(self, *, name: str) -> "ToolOutcome":
        """Simulate secret deletion and acknowledge (no persistence)."""
        return {"outcome": "secret deleted (simulated)", "details": {"name": name}}


if TYPE_CHECKING:  # typing support only
    from ..common.tool_outcome import ToolOutcome
