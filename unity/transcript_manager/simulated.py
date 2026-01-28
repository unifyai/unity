# unity/transcript_manager/simulated_transcript_manager.py
from __future__ import annotations

import asyncio
import threading
import functools
from typing import List, Optional, Dict, Any, Type, Union

import unillm
from pydantic import BaseModel

from ..common.async_tool_loop import SteerableToolHandle
from .base import BaseTranscriptManager
from .prompt_builders import (
    build_ask_prompt,
    build_simulated_method_prompt,
)
from ..common.simulated import (
    mirror_transcript_manager_tools,
    SimulatedLineage,
    SimulatedLog,
    simulated_llm_roundtrip,
    SimulatedHandleMixin,
    build_followup_prompt,
    maybe_tool_log_scheduled,
    maybe_tool_log_completed,
)
from .types.message import Message
from .types.exchange import Exchange
from ..common.llm_client import new_llm_client
from ..constants import LOGGER


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────
class _SimulatedTranscriptHandle(SteerableToolHandle, SimulatedHandleMixin):
    """
    A very small, LLM-backed handle used by SimulatedTranscriptManager.ask.
    """

    def __init__(
        self,
        llm: unillm.Unify,
        initial_text: str,
        *,
        _return_reasoning_steps: bool,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None,
        clarification_down_q: asyncio.Queue[str] | None,
        response_format: Optional[Type[BaseModel]] = None,
    ):
        self._llm = llm
        self._initial = initial_text
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
        # Human-friendly log label derived from current lineage, mirroring other simulated managers:
        # "<outer...>->SimulatedTranscriptManager.ask(abcd)"
        self._log_label = SimulatedLineage.make_label("SimulatedTranscriptManager.ask")

        # fire clarification immediately if queues supplied
        if self._needs_clar:
            try:
                q_text = (
                    "Could you clarify your information-need around the transcripts?"
                )
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

        self._extra_user_msgs: List[str] = []

        # completion primitives
        self._done = threading.Event()
        self._cancelled = False
        self._answer: Optional[str] = None
        self._msgs: List[Dict[str, Any]] = []
        self._paused = False
        # label already set above
        # Async cancellation signal to break clarification waits
        self._cancel_event: asyncio.Event = asyncio.Event()

    # ──  API expected by SteerableToolHandle  ──────────────────────────────
    async def result(self):
        if self._cancelled:
            return "processed stopped early, no result"

        while self._paused and not self._cancelled:
            await asyncio.sleep(0.05)

        if not self._done.is_set():
            # wait for clarification reply if requested
            if self._needs_clar:
                try:
                    LOGGER.info(
                        f"⏳ [{self._log_label}] Waiting for clarification answer…",
                    )
                except Exception:
                    pass
                clar_reply: str | None = None
                get_task = asyncio.create_task(self._clar_down_q.get())
                cancel_task = asyncio.create_task(self._cancel_event.wait())
                done, pending = await asyncio.wait(
                    {get_task, cancel_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                if cancel_task in done:
                    self._done.set()
                    return "processed stopped early, no result"
                try:
                    clar_reply = get_task.result()
                except Exception:
                    clar_reply = None
                if clar_reply is None:
                    self._done.set()
                    return "processed stopped early, no result"
                self._extra_user_msgs.append(f"Clarification: {clar_reply}")
                try:
                    SimulatedLog.log_clarification_answer(self._log_label, clar_reply)
                except Exception:
                    pass
                try:
                    LOGGER.info(f"💬 [{self._log_label}] Clarification answer received")
                except Exception:
                    pass

            prompt = "\n\n---\n\n".join([self._initial] + self._extra_user_msgs)
            # Unified simulated LLM roundtrip (with conditional response logging and optional dumps)
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
            self._msgs = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            self._done.set()

        # If cancellation happened after the coroutine started, return a stable post-cancel value.
        if self._cancelled:
            return "processed stopped early, no result"
        if self._want_steps:
            return self._answer, self._msgs
        return self._answer

    def interject(
        self,
        message: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
        images: list | dict | None = None,
    ) -> str:
        """Interject a message into the in-flight handle.

        Args:
            message: The interjection message to inject.
            parent_chat_context_cont: Optional continuation of parent chat context.
                Accepted for API parity with real handles but not currently used.
            images: Optional image references. Accepted for API parity with real handles
                but not currently used.
        """
        if self._cancelled:
            return "Interaction has been stopped."
        self._log_interject(message)
        self._extra_user_msgs.append(message)
        return "Acknowledged."

    def stop(
        self,
        reason: Optional[str] = None,
    ) -> str:
        """Stop the in-flight handle.

        Args:
            reason: Optional reason for stopping.
        """
        self._log_stop(reason)
        self._cancelled = True
        try:
            self._cancel_event.set()
        except Exception:
            pass
        self._done.set()
        return "Stopped." if reason is None else f"Stopped: {reason}"

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

    def done(self) -> bool:
        return self._done.is_set()

    async def ask(
        self,
        question: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
        images: list | dict | None = None,
    ) -> "SteerableToolHandle":
        """Ask a follow-up question about the current operation.

        Args:
            question: The question to ask.
            parent_chat_context_cont: Optional continuation of parent chat context.
                Accepted for API parity with real handles but not currently used.
            images: Optional image references. Accepted for API parity with real handles
                but not currently used.
        """
        follow_up_prompt = build_followup_prompt(
            question=question,
            initial_instruction=self._initial,
            extra_messages=list(self._extra_user_msgs),
        )

        handle = _SimulatedTranscriptHandle(
            self._llm,
            follow_up_prompt,
            _return_reasoning_steps=self._want_steps,
            _requests_clarification=False,
            clarification_up_q=self._clar_up_q,
            clarification_down_q=self._clar_down_q,
        )
        # Align with other simulated components: concise "Question(<parent_label>)" label
        try:
            handle._log_label = SimulatedLineage.question_label(self._log_label)  # type: ignore[attr-defined]
        except Exception:
            pass
        # Emit a human-facing log for the nested ask
        try:
            SimulatedLog.log_request("ask", getattr(handle, "_log_label", ""), question)  # type: ignore[arg-type]
        except Exception:
            pass
        return handle

    # --- event APIs required by SteerableToolHandle ---------------------
    async def next_clarification(self) -> dict:
        """Retrieve the next clarification request, if any.

        Only surfaces clarification events when this handle explicitly requested
        clarification. This prevents cross-handle consumption of shared clarification
        queues that may be injected by external processes.
        """
        if not getattr(self, "_needs_clar", False):
            return {}
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
        return {}

    async def next_notification(self) -> dict:
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        try:
            if self._clar_down_q is not None:
                await self._clar_down_q.put(answer)
        except Exception:
            pass


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
        simulation_guidance: Optional[str] = None,
        # Accept but ignore parameters that real TranscriptManager uses
        contact_manager: Any = None,
        **kwargs: Any,
    ) -> None:
        self._description = description
        self._log_events = log_events
        self._rolling_summary_in_prompts = rolling_summary_in_prompts
        self._simulation_guidance = simulation_guidance

        # Shared, *stateful* **asynchronous** LLM (reusing common client)
        self._llm = new_llm_client(stateful=True)
        # Minimal in-memory simulation store for programmatic helpers
        self._sim_next_message_id: int = 1
        self._sim_next_exchange_id: int = 1
        self._sim_messages: List[Message] = []
        self._sim_exchanges: Dict[int, Exchange] = {}
        # Use shared helper to mirror the real TranscriptManager's tools
        tools_for_prompt = mirror_transcript_manager_tools()
        # Provide placeholder counts/columns for the simulated environment
        fake_columns = [{k: str(v.annotation)} for k, v in Message.model_fields.items()]
        # Include sender contact columns for clarity
        from ..contact_manager.types.contact import Contact as _Contact

        fake_contact_columns = [
            {k: str(v.annotation)} for k, v in _Contact.model_fields.items()
        ]
        ask_sys = build_ask_prompt(
            tools_for_prompt,
            num_messages=10,
            transcript_columns=fake_columns,
            contact_columns=fake_contact_columns,
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

    def reduce(
        self,
        *,
        metric: str,
        keys: str | list[str],
        filter: Optional[str | dict[str, str]] = None,
        group_by: Optional[str | list[str]] = None,
    ) -> Any:
        """
        Simulated counterpart of the TranscriptManager.reduce tool.

        This implementation does not query a real transcripts store; it returns
        deterministic values with the same shapes as the concrete tool:

        * Single key, no grouping  → scalar.
        * Multiple keys, no grouping → ``dict[key -> scalar]``.
        * With grouping             → nested ``dict[group -> value or dict]``.
        """

        def _scalar(k: str) -> float:
            return float(len(str(k)) or 1)

        key_list: list[str] = [keys] if isinstance(keys, str) else list(keys)

        if group_by is None:
            if isinstance(keys, str):
                return _scalar(keys)
            return {k: _scalar(k) for k in key_list}

        groups: list[str] = (
            [group_by] if isinstance(group_by, str) else [str(g) for g in group_by]
        )
        if isinstance(keys, str):
            return {g: _scalar(keys) for g in groups}
        return {g: {k: _scalar(k) for k in key_list} for g in groups}

    # --------------------------------------------------------------------- #
    # ask                                                                   #
    # --------------------------------------------------------------------- #
    @functools.wraps(BaseTranscriptManager.ask, updated=())
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
        images: object | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        should_log = self._log_events or log_events
        call_id = None

        # No EventBus publishing for simulated managers
        call_id = None

        # Tool-style scheduled log (no parent-lineage)
        maybe_tool_log_scheduled(
            "SimulatedTranscriptManager.ask",
            "ask",
            {
                "text": text if isinstance(text, str) else repr(text),
                "requests_clarification": _requests_clarification,
            },
        )

        instruction = build_simulated_method_prompt(
            "ask",
            text,
            parent_chat_context=_parent_chat_context,
        )
        handle = _SimulatedTranscriptHandle(
            self._llm,
            instruction,
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
            response_format=response_format,
        )

        # Do not emit ❓ Ask requested here; tool-style scheduled log above already captures inputs

        # No EventBus publishing for simulated managers
        return handle

    @functools.wraps(BaseTranscriptManager.clear, updated=())
    def clear(self) -> None:
        sched = maybe_tool_log_scheduled(
            "SimulatedTranscriptManager.clear",
            "clear",
            {},
        )
        type(self).__init__(
            self,
            description=getattr(
                self,
                "_description",
                "nothing fixed, make up some imaginary scenario",
            ),
            log_events=getattr(self, "_log_events", False),
            rolling_summary_in_prompts=getattr(
                self,
                "_rolling_summary_in_prompts",
                True,
            ),
            simulation_guidance=getattr(self, "_simulation_guidance", None),
        )
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(label, cid, "clear", {"outcome": "reset"}, t0)

    # ------------------------------------------------------------------ #
    # Programmatic helpers (simulated, non-LLM or lightly structured)    #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseTranscriptManager.log_messages, updated=())
    def log_messages(
        self,
        messages: Union[
            Union[Dict[str, Any], Message],
            List[Union[Dict[str, Any], Message]],
        ],
        synchronous: bool = False,
    ) -> List[Message]:
        """
        Simulated message logging: validates inputs, assigns ids, and stores in-memory.
        """
        sched = maybe_tool_log_scheduled(
            "SimulatedTranscriptManager.log_messages",
            "log_messages",
            {
                "count": (len(messages) if isinstance(messages, list) else 1),
                "synchronous": synchronous,
            },
        )

        if not messages:
            if sched:
                label, cid, t0 = sched
                maybe_tool_log_completed(label, cid, "log_messages", {"created": 0}, t0)
            return []

        batch = messages if isinstance(messages, list) else [messages]
        created: List[Message] = []
        for raw in batch:
            payload = (
                raw.model_dump(mode="python") if isinstance(raw, Message) else dict(raw)
            )
            # Require receiver_ids
            if "receiver_ids" not in payload:
                raise ValueError("Each message must include 'receiver_ids'.")
            # Require an explicit exchange_id (use log_first_message_in_new_exchange for new)
            exid = payload.get("exchange_id", None)
            if exid is None or int(exid) < 0:
                raise ValueError(
                    "exchange_id is required in simulated log_messages; use log_first_message_in_new_exchange to start a new thread.",
                )
            # Assign message_id when missing/unassigned
            try:
                mid = int(payload.get("message_id", -1))
            except Exception:
                mid = -1
            if mid < 0:
                mid = self._sim_next_message_id
                self._sim_next_message_id += 1
            payload["message_id"] = mid
            # Normalise model and persist to in-memory store
            msg = Message(**payload)
            created.append(msg)
            self._sim_messages.append(msg)
            # Ensure an exchange row exists
            exid_int = int(getattr(msg, "exchange_id"))
            if exid_int not in self._sim_exchanges:
                try:
                    medium = str(getattr(msg, "medium", ""))  # best-effort
                except Exception:
                    medium = ""
                self._sim_exchanges[exid_int] = Exchange(
                    exchange_id=exid_int,
                    metadata={},
                    medium=medium,
                )

        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(
                label,
                cid,
                "log_messages",
                {"created": len(created)},
                t0,
            )
        return created

    def join_published(self) -> None:
        """
        No-op in simulation; included for API compatibility.
        """
        sched = maybe_tool_log_scheduled(
            "SimulatedTranscriptManager.join_published",
            "join_published",
            {},
        )
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(
                label,
                cid,
                "join_published",
                {"outcome": "no-op"},
                t0,
            )

    @staticmethod
    @functools.wraps(BaseTranscriptManager.build_plain_transcript, updated=())
    def build_plain_transcript(
        messages: list[dict],
        *,
        contact_manager: Optional[Any] = None,
    ) -> str:
        """
        Return a plain-text transcript ("Full Name: content") for provided messages.
        """
        # Local import avoids widening module deps at import time
        from unity.contact_manager.contact_manager import (
            ContactManager as _CM,
        )  # noqa: WPS433

        cm = contact_manager or _CM()
        name_cache: dict[int, str] = {}

        def _name_for_cid(cid: int) -> str:
            if cid in name_cache:
                return name_cache[cid]
            try:
                recs = cm.filter_contacts(filter=f"contact_id == {cid}", limit=1)
                if recs:
                    rec = recs[0]
                    full = " ".join(
                        p for p in [rec.first_name, rec.surname] if p
                    ).strip()
                    if not full:
                        full = (rec.first_name or "").strip()
                    if full:
                        name_cache[cid] = full
                        return full
            except Exception:
                pass
            name_cache[cid] = str(cid)
            return name_cache[cid]

        lines: list[str] = []
        for itm in messages:
            if "kind" in itm:
                if itm.get("kind") != "message":
                    continue
                data = itm.get("data", {})
                sender_val = data.get("sender_id")
                content_val = data.get("content", "")
                if sender_val is None:
                    continue
                sender_name = _name_for_cid(int(sender_val))
            else:
                # Sandbox-style dicts: {"sender": "...", "content": "..."} or {"sender_id": 3, ...}
                if "sender" in itm:
                    sender_name = str(itm.get("sender"))
                else:
                    sid = itm.get("sender_id")
                    sender_name = (
                        _name_for_cid(int(sid)) if sid is not None else "Unknown"
                    )
                content_val = str(itm.get("content", ""))
            lines.append(f"{sender_name}: {content_val}")
        return "\n".join(lines)

    def update_contact_id(
        self,
        *,
        original_contact_id: int,
        new_contact_id: int,
    ) -> Dict[str, Any]:
        """
        Simulated in-place contact id substitution across the in-memory messages.
        """
        sched = maybe_tool_log_scheduled(
            "SimulatedTranscriptManager.update_contact_id",
            "update_contact_id",
            {"from": original_contact_id, "to": new_contact_id},
        )
        if original_contact_id == new_contact_id:
            raise ValueError("original_contact_id and new_contact_id must differ.")
        updated_count = 0
        for i, msg in enumerate(list(self._sim_messages)):
            changed = False
            if msg.sender_id is not None and int(msg.sender_id) == original_contact_id:
                msg = msg.model_copy(update={"sender_id": new_contact_id})
                changed = True
            if any(
                rid is not None and rid == original_contact_id
                for rid in msg.receiver_ids
            ):
                new_rids = [
                    new_contact_id if rid == original_contact_id else rid
                    for rid in msg.receiver_ids
                ]
                # Preserve order; best-effort de-dup
                seen: set[int] = set()
                deduped: list[int] = []
                for rid in new_rids:
                    if rid not in seen:
                        seen.add(rid)
                        deduped.append(rid)
                msg = msg.model_copy(update={"receiver_ids": deduped})
                changed = True
            if changed:
                self._sim_messages[i] = msg
                updated_count += 1
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(
                label,
                cid,
                "update_contact_id",
                {
                    "outcome": "contact ids updated",
                    "details": {
                        "old_contact_id": original_contact_id,
                        "new_contact_id": new_contact_id,
                        "updated_messages": updated_count,
                    },
                },
                t0,
            )
        return {
            "outcome": "contact ids updated",
            "details": {
                "old_contact_id": original_contact_id,
                "new_contact_id": new_contact_id,
                "updated_messages": updated_count,
            },
        }

    def get_exchange_metadata(self, exchange_id: int) -> Exchange:
        """
        Simulated fetch of exchange metadata from in-memory exchanges.
        """
        sched = maybe_tool_log_scheduled(
            "SimulatedTranscriptManager.get_exchange_metadata",
            "get_exchange_metadata",
            {"exchange_id": exchange_id},
        )
        ex = self._sim_exchanges.get(int(exchange_id))
        if ex is None:
            # Create a plausible default record
            ex = Exchange(exchange_id=int(exchange_id), metadata={}, medium="")
            self._sim_exchanges[int(exchange_id)] = ex
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(
                label,
                cid,
                "get_exchange_metadata",
                {
                    "exchange_id": ex.exchange_id,
                    "medium": ex.medium,
                    "metadata_keys": list(ex.metadata.keys()),
                },
                t0,
            )
        return ex

    def update_exchange_metadata(
        self,
        exchange_id: int,
        metadata: Dict[str, Any],
    ) -> Exchange:
        """
        Simulated upsert of exchange metadata in the in-memory store.
        """
        sched = maybe_tool_log_scheduled(
            "SimulatedTranscriptManager.update_exchange_metadata",
            "update_exchange_metadata",
            {
                "exchange_id": exchange_id,
                "metadata_keys": list((metadata or {}).keys()),
            },
        )
        cur = self._sim_exchanges.get(int(exchange_id))
        if cur is None:
            cur = Exchange(
                exchange_id=int(exchange_id),
                metadata=dict(metadata or {}),
                medium="",
            )
        else:
            cur = Exchange(
                exchange_id=cur.exchange_id,
                metadata=dict(metadata or {}),
                medium=cur.medium,
            )
        self._sim_exchanges[int(exchange_id)] = cur
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(
                label,
                cid,
                "update_exchange_metadata",
                {
                    "exchange_id": cur.exchange_id,
                    "metadata_keys": list(cur.metadata.keys()),
                },
                t0,
            )
        return cur

    @functools.wraps(
        BaseTranscriptManager.log_first_message_in_new_exchange,
        updated=(),
    )
    def log_first_message_in_new_exchange(
        self,
        message: Union[Dict[str, Any], Message],
        *,
        exchange_initial_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Start a new exchange, assign a fresh id, log the first message, and upsert metadata.
        """
        sched = maybe_tool_log_scheduled(
            "SimulatedTranscriptManager.log_first_message_in_new_exchange",
            "log_first_message_in_new_exchange",
            {"has_metadata": exchange_initial_metadata is not None},
        )
        payload = (
            message.model_dump(mode="python")
            if isinstance(message, Message)
            else dict(message)
        )
        if payload.get("exchange_id") is not None:
            raise ValueError(
                "exchange_id must NOT be provided when starting a new exchange; use log_messages for existing threads.",
            )
        new_exid = self._sim_next_exchange_id
        self._sim_next_exchange_id += 1
        # Seed exchange record
        self._sim_exchanges[new_exid] = Exchange(
            exchange_id=new_exid,
            metadata=dict(exchange_initial_metadata or {}),
            medium=str(payload.get("medium") or ""),
        )
        # Log first message
        payload["exchange_id"] = new_exid
        # Let log_messages validate and store
        self.log_messages(payload, synchronous=True)
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(
                label,
                cid,
                "log_first_message_in_new_exchange",
                {"exchange_id": new_exid},
                t0,
            )
        return new_exid

    @functools.wraps(BaseTranscriptManager.filter_exchanges, updated=())
    def filter_exchanges(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int | None = 100,
    ) -> Dict[str, Any]:
        """
        Simulated filter over the in-memory exchanges using a restricted eval.
        """
        sched = maybe_tool_log_scheduled(
            "SimulatedTranscriptManager.filter_exchanges",
            "filter_exchanges",
            {"filter": filter, "offset": offset, "limit": limit},
        )
        exchanges = list(self._sim_exchanges.values())
        if filter:
            filtered: list[Exchange] = []
            for ex in exchanges:
                env = {
                    "exchange_id": ex.exchange_id,
                    "medium": ex.medium,
                    "metadata": ex.metadata,
                }
                try:
                    ok = bool(
                        eval(str(filter), {"__builtins__": {}}, env),
                    )  # noqa: S307
                except Exception:
                    ok = True  # best-effort fallback
                if ok:
                    filtered.append(ex)
            exchanges = filtered
        # Apply slicing
        if offset:
            exchanges = exchanges[offset:]
        if limit is not None:
            exchanges = exchanges[: int(limit)]
        out = {"exchanges": exchanges}
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(
                label,
                cid,
                "filter_exchanges",
                {"count": len(exchanges), "offset": offset, "limit": limit},
                t0,
            )
        return out
