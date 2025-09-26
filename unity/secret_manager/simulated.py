from __future__ import annotations

import asyncio
import functools
import json
from typing import Any, Dict, Optional, TYPE_CHECKING

import unify
from .base import BaseSecretManager
from .types import Secret
from .prompt_builders import build_ask_prompt, build_update_prompt
from ..common.async_tool_loop import SteerableToolHandle
from ..events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
    wrap_handle_with_logging,
)
from ..common.simulated import mirror_secret_manager_tools


class _SimulatedSecretHandle(SteerableToolHandle):
    """Minimal LLM-backed handle used by SimulatedSecretManager.ask / update."""

    def __init__(
        self,
        llm: unify.Unify,
        initial_text: str,
        *,
        _return_reasoning_steps: bool,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None,
        clarification_down_q: asyncio.Queue[str] | None,
    ) -> None:
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

        if self._needs_clar:
            try:
                self._clar_up_q.put_nowait(
                    "Could you clarify your request about secrets?",
                )
            except asyncio.QueueFull:
                pass

        self._extra_msgs: list[str] = []
        self._done = asyncio.Event()
        self._cancelled = False
        self._paused = False
        self._answer: str | None = None
        self._messages: list[dict[str, Any]] = []

    async def result(self):
        if self._cancelled:
            raise asyncio.CancelledError()

        while self._paused and not self._cancelled:
            await asyncio.sleep(0.05)

        if not self._done.is_set():
            extra = []
            if self._needs_clar and self._clar_down_q is not None:
                clar = await self._clar_down_q.get()
                extra.append(f"Clarification: {clar}")

            prompt = "\n\n---\n\n".join([self._initial] + self._extra_msgs + extra)
            answer = await self._llm.generate(prompt)
            self._answer = answer
            self._messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            self._done.set()

        if self._want_steps:
            return self._answer, self._messages
        return self._answer

    def interject(self, message: str) -> str:
        if self._cancelled:
            return "Interaction stopped."
        self._extra_msgs.append(message)
        return "Acknowledged."

    def stop(self, reason: str | None = None) -> str:
        self._cancelled = True
        try:
            self._done.set()
        except Exception:
            pass
        return "Stopped." if reason is None else f"Stopped: {reason}"

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

    def done(self) -> bool:  # type: ignore[override]
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
            "However, there is also an ongoing simulated process with the instructions given below. "
            "Please make your answer realistic and consistent with the provided context."
        )
        follow_up_prompt = "\n\n---\n\n".join(
            [q_msg]
            + [self._initial]
            + self._extra_msgs
            + [f"Question (as a reminder): {question}"],
        )
        return _SimulatedSecretHandle(
            self._llm,
            follow_up_prompt,
            _return_reasoning_steps=self._want_steps,
            _requests_clarification=False,
            clarification_up_q=self._clar_up_q,
            clarification_down_q=self._clar_down_q,
        )


class SimulatedSecretManager(BaseSecretManager):
    """Drop-in simulated SecretManager with imaginary data and stateful LLM memory."""

    def __init__(
        self,
        description: str = "simulate secret storage and actions (no real database)",
        *,
        log_events: bool = False,
        simulation_guidance: Optional[str] = None,
    ) -> None:
        self._description = description
        self._log_events = log_events
        self._simulation_guidance = simulation_guidance

        # Shared, stateful async LLM
        self._llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=True,
            traced=True,
            stateful=True,
        )

        # Mirror the real manager's tool exposure using reflection helper
        ask_tools = mirror_secret_manager_tools("ask")
        upd_tools = mirror_secret_manager_tools("update")

        ask_msg = build_ask_prompt(tools=ask_tools)
        upd_msg = build_update_prompt(tools=upd_tools)

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

    @functools.wraps(BaseSecretManager.ask, updated=())
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
                "SecretManager",
                "ask",
                phase="incoming",
                question=text,
            )

        # Build the simulated instruction (no real tools will run)
        instruction = (
            "Simulate the behaviour of SecretManager.ask for the following user message. "
            "Never reveal any raw secret values; always refer to placeholders like ${name}.\n\n"
            f"User message: {text}\n\n"
            + (
                f"Parent context: {json.dumps(parent_chat_context)}\n\n"
                if parent_chat_context
                else ""
            )
        )
        handle = _SimulatedSecretHandle(
            self._llm,
            instruction,
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(handle, call_id, "SecretManager", "ask")

        return handle

    @functools.wraps(BaseSecretManager.update, updated=())
    async def update(
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
                "SecretManager",
                "update",
                phase="incoming",
                request=text,
            )

        instruction = (
            "Simulate the behaviour of SecretManager.update for the following request. "
            "Never reveal raw secret values; reference secrets via ${name}.\n\n"
            f"User request: {text}\n\n"
            + (
                f"Parent context: {json.dumps(parent_chat_context)}\n\n"
                if parent_chat_context
                else ""
            )
        )
        handle = _SimulatedSecretHandle(
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
                "SecretManager",
                "update",
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
            Secret(name="api_key", value="", description="API access token"),
            Secret(name="db_password", value="", description="database password"),
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
            Secret(name="api_key", value="", description="API access token"),
            Secret(name="db_password", value="", description="database password"),
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
