import asyncio
import unify
import os
import functools
import json
from datetime import datetime, timezone
from contextlib import suppress
from typing import (
    Optional,
    Awaitable,
    Dict,
    Callable,
    Tuple,
    Any,
    Union,
    Type,
    TYPE_CHECKING,
    Literal,
)
from ..constants import LOGGER, SESSION_ID
from .llm_helpers import short_id
from ._async_tool.loop_config import TOOL_LOOP_LINEAGE
from ._async_tool.messages import forward_handle_call
from ._async_tool.loop import async_tool_loop_inner
from .loop_snapshot import (
    LoopSnapshot as _LoopSnapshot,
    EntryPointManagerMethod as _EntryPointManagerMethod,
    EntryPointInlineTools as _EntryPointInlineTools,
    ToolRef as _ToolRef,
    validate_snapshot as _validate_snapshot,
    migrate_snapshot as _migrate_snapshot,
)

if TYPE_CHECKING:
    from ..image_manager.types.image_refs import ImageRefs

# Tiny handle objects exposed to callers
# ─────────────────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod


class SteerableHandle(ABC):
    """Abstract base class for steerable handles.

    Notes on parent_chat_context_cont
    ---------------------------------
    Some steering methods accept an optional parameter ``parent_chat_context_cont``.
    This represents the parent chat context continued since the start of this loop –
    i.e., a continuation of the parent "conversation" (which may itself be another
    tool loop). Implementations should ensure that, when provided, this context is
    surfaced to the LLM in an appropriate way.
    """

    @abstractmethod
    async def ask(
        self,
        question: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ) -> "SteerableHandle":
        """
        Ask a question to the running process.

        Parameters
        ----------
        question : str
            The follow-up user question.
        parent_chat_context_cont : list[dict] | None, optional
            The parent chat context continued since the start of this loop.
            This is the continuation of the parent conversation to date. When
            provided, implementations should thread this into the LLM input. The
            user message should be packaged as a dict content containing keys
            "parent_chat_context_continuted" and "message".
        """

    @abstractmethod
    def interject(
        self,
        message: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ) -> Awaitable[Optional[str]] | Optional[str]:
        """Inject an additional *user* turn into the running conversation.

        Parameters
        ----------
        message : str
            The user interjection to inject into the loop.
        parent_chat_context_cont : list[dict] | None, optional
            The parent chat context continued since the start of this loop.
            When provided, implementations should ensure the LLM sees this
            continuation alongside the interjection.
        """


class SteerableToolHandle(SteerableHandle):
    """Abstract base class for steerable tool handles."""

    @abstractmethod
    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def stop(
        self,
        reason: Optional[str] = None,
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ) -> Awaitable[Optional[str]] | Optional[str]:
        """Shutdown the loop, killing any pending work in the process.

        Parameters
        ----------
        reason : str | None
            Optional human-readable reason for stopping.
        parent_chat_context_cont : list[dict] | None, optional
            The parent chat context continued since the start of this loop.
            Included for signature parity; no LLM call is made here, but this
            value is forwarded to any delegated handle if present.
        """

    @abstractmethod
    def pause(self) -> Awaitable[Optional[str]] | Optional[str]:
        """Temporarily freeze the outer loop (tools keep running)."""

    @abstractmethod
    def resume(self) -> Awaitable[Optional[str]] | Optional[str]:
        """Un-freeze a loop that was paused with :pyfunc:`pause`."""

    @abstractmethod
    def done(self) -> Awaitable[bool] | bool:
        """Flag for whether or not this task is done."""

    @abstractmethod
    def result(self) -> Awaitable[str] | str:
        """Wait for the assistant's *final* reply."""

    # ── bottom-up event APIs (abstract surface) -------------------------------
    @abstractmethod
    async def next_clarification(self) -> dict:
        """Await the next clarification event pushed by a running tool."""

    @abstractmethod
    async def next_notification(self) -> dict:
        """Await the next notification pushed by a running tool."""

    @abstractmethod
    async def answer_clarification(self, call_id: str, answer: str) -> None:
        """Programmatically answer a clarification for a pending tool call.

        This looks up the down-queue for the given call and pushes the answer.
        Falls through silently if the mapping is missing (tool may have finished).
        """

    # --- snapshotting (skeleton; non-abstract stubs in v1) -----------------
    def serialize(self) -> dict:
        """Return a serializable snapshot of this handle's state (stub).

        This is a non-abstract stub in v1 to avoid breaking existing subclasses.
        A concrete implementation will be provided for the main async tool loop
        handle in a subsequent step.
        """
        raise NotImplementedError("serialize() is not implemented yet")

    @classmethod
    def deserialize(cls, snapshot: dict) -> "SteerableToolHandle":
        """Recreate a handle from a serialized snapshot (stub).

        This classmethod is a stub in v1 and will be implemented incrementally.
        """
        raise NotImplementedError("deserialize() is not implemented yet")

    def get_history(self) -> list[dict]:
        """Returns the conversational history of the loop.

        Default implementation returns empty list. Subclasses with
        LLM clients should override to return the full conversation
        history including tool calls and reasoning.

        Returns
        -------
        list[dict]
            List of message dicts in the format used by the LLM client.
            For handles without an LLM client, returns an empty list.
        """
        return []


class AsyncToolLoopHandle(SteerableToolHandle):
    """
    Returned by `start_async_tool_loop`.  Lets you
      • queue extra user messages while the loop runs and
      • stop the loop at any time.
    """

    def __init__(
        self,
        *,
        task: asyncio.Task,
        interject_queue: asyncio.Queue[dict | str],
        cancel_event: asyncio.Event,
        stop_event: asyncio.Event,
        pause_event: Optional[asyncio.Event] = None,
        client: "unify.AsyncUnify | None" = None,
        loop_id: str = "",
        initial_user_message: Optional[Any] = None,
    ):
        self._task = task
        self._queue = interject_queue
        self._cancel_event = cancel_event
        self._stop_event = stop_event
        # "running" ⇢ Event **set**,  "paused" ⇢ Event **cleared**
        self._pause_event = pause_event or asyncio.Event()
        self._client = client
        # No delegate in the new passthrough design – outer loop stays active.
        self._pause_event.set()
        self._loop_id: str = loop_id
        # Human-friendly label for logs (includes 4-hex suffix when available).
        # This is populated by the inner loop as soon as it constructs LoopConfig.
        # Until then, fall back to the bare loop_id.
        self._log_label: str = loop_id
        # Only the top-level handle should emit the public stop log.
        # Nested/adopted handles will inherit False to avoid duplicate logging.
        self._is_root_handle: bool = False

        # Buffer interjections that may arrive **before** a downstream handle
        # (e.g. an `ActiveTask`) has been adopted.  Once a delegate is ready we
        # forward all queued messages so that no early user guidance is lost.
        self._early_interjects: list[dict | str] = []

        # Maintain a user-visible history (what the end-user would see):
        # Records: original prompt (user), interjections (user), ask Q/A (user/assistant).
        self._user_visible_history: list[dict] = []
        if initial_user_message:
            self._user_visible_history.append(
                {"role": "user", "content": initial_user_message},
            )

        # Event streams for bottom-up signals
        self._clar_q: asyncio.Queue[dict] = asyncio.Queue()
        self._notification_q: asyncio.Queue[dict] = asyncio.Queue()
        # Buffer passthrough operations (method_name, kwargs, fallback_keys) while
        # a passthrough handle is not yet ready but tools are scheduled
        self._pending_passthrough_ops: list[tuple[str, dict, tuple[str, ...]]] = []

    # small local helpers to keep user-visible history consistent
    def _append_user_visible_user(
        self,
        message: str,
        parent_chat_context_cont: list[dict] | None,
    ) -> None:
        # NOTE: key name 'parent_chat_context_continuted' is legacy and intentional
        with suppress(Exception):
            if parent_chat_context_cont is not None:
                self._user_visible_history.append(
                    {
                        "role": "user",
                        "content": {
                            "message": message,
                            "parent_chat_context_continuted": parent_chat_context_cont,
                        },
                    },
                )
            else:
                self._user_visible_history.append(
                    {"role": "user", "content": message},
                )

    def _append_user_visible_assistant(self, message: str) -> None:
        with suppress(Exception):
            self._user_visible_history.append(
                {"role": "assistant", "content": message},
            )

    # ── internal: passthrough forwarding/buffering helpers ──────────────────
    def _iter_passthrough_handles(self):
        task_info = {}
        with suppress(Exception):
            task_info = getattr(self._task, "task_info", {})
        if not isinstance(task_info, dict):
            return []
        out = []
        for _t, _inf in task_info.items():
            h = getattr(_inf, "handle", None)
            is_pt = getattr(_inf, "is_passthrough", False)
            if h is not None and is_pt:
                out.append(h)
        return out

    def _has_scheduled_tools(self) -> bool:
        with suppress(Exception):
            ti = getattr(self._task, "task_info", {})
            return isinstance(ti, dict) and len(ti) > 0
        return False

    async def _forward_call_to_handle(
        self,
        handle,
        method_name: str,
        kwargs: dict,
        fallback: tuple[str, ...],
    ):
        with suppress(Exception):
            return await forward_handle_call(
                handle,
                method_name,
                kwargs,
                fallback_positional_keys=fallback,
            )
        return None

    async def _replay_pending_passthrough_ops(self) -> None:
        if not self._pending_passthrough_ops:
            return
        handles = self._iter_passthrough_handles()
        if not handles:
            return
        remaining: list[tuple[str, dict, tuple[str, ...]]] = []
        for name, kw, fb in list(self._pending_passthrough_ops):
            forwarded = False
            for h in handles:
                await self._forward_call_to_handle(h, name, kw, fb)
                forwarded = True
            if not forwarded:
                remaining.append((name, kw, fb))
        self._pending_passthrough_ops = remaining

    async def _try_forward_or_buffer(
        self,
        method_name: str,
        kwargs: dict,
        fallback: tuple[str, ...] = (),
    ) -> None:
        handles = self._iter_passthrough_handles()
        if handles:
            for h in handles:
                await self._forward_call_to_handle(h, method_name, kwargs, fallback)
            return
        if self._has_scheduled_tools():
            try:
                self._pending_passthrough_ops.append(
                    (method_name, dict(kwargs or {}), tuple(fallback or ())),
                )
            except Exception:
                pass

    async def ask(
        self,
        question: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
        _return_reasoning_steps: bool = False,
    ) -> "SteerableToolHandle":
        """
        Answers *question* about this *pending* tool, associated with this handle.
        The question is read-only (the tool state is not modified whatsoever).
        The calling parent loop is left completely untouched.
        When ``parent_chat_context_cont`` is provided, the user message will be
        packaged as a dict with keys {"parent_chat_context_continuted", "message"}
        to clearly signal the continuation of the parent conversation since the
        start of this loop.
        """
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.info(f"❓ [{_label}] Ask requested: {question}")

        # Record the user-visible question immediately (even if delegated)
        self._append_user_visible_user(question, parent_chat_context_cont)

        # No delegate forwarding – outer loop remains in control.

        # 0.  Defensive guard: if the outer loop has already finished we can
        #     just answer from the final transcript without starting another
        #     loop.
        if self.done():
            LOGGER.warning(
                "AsyncToolLoopHandle.ask() called on an already-finished "
                "loop – returning a synthetic handle with a static answer.",
            )

            async def _static() -> str:  # type: ignore[return-type]
                return (
                    "Parent loop is already complete; no additional "
                    "information available."
                )

            class _StaticHandle(SteerableToolHandle):
                def __init__(self): ...

                async def interject(self, message: str): ...

                def stop(self, reason: Optional[str] = None): ...

                def pause(self): ...

                def resume(self): ...

                def done(self):
                    return True

                async def result(self):
                    return await _static()

                async def ask(self, question: str) -> "SteerableToolHandle":
                    return self

            return _StaticHandle()  # pragma: no cover

        # 1.  Gather a *read-only* snapshot of the parent chat.
        parent_ctx = []
        with suppress(Exception):
            msgs = getattr(self._client, "messages", []) if self._client else []
            if msgs is None:
                msgs = []
            parent_ctx = list(msgs)

        # 2.  Prepare an *in-memory* Unify client for the **inspection** loop
        #     (LLM sees only the system header + follow-up user question).
        inspection_client = unify.AsyncUnify(
            "gpt-5@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "false")),
            reasoning_effort="high",
            service_tier="priority",
        )
        inspection_client.set_system_message(
            "You are inspecting a running tool-use conversation. The entire "
            "transcript so far is attached below (read-only):\n"
            f"{json.dumps(parent_ctx, indent=2)}\n\n"
            "Answer the user's follow-up question using ONLY this context.\n"
            "Do not attempt to run new tools unless they are exposed to you.\n"
            "Do not ask the user questions or request clarification. If information is missing,\n"
            "state what is known and, if helpful, briefly note assumptions. Respond in a single, concise paragraph.",
        )

        # 3. Recursive visibility ––––––––––––––––––––––––––––––––––––––––
        # Any *currently pending* SteerableToolHandle (deep-nested) should
        # be made available as a tool so the inspection loop can itself ask
        # follow-up questions.  We approximate this by scanning the parent
        # task_info dict that the outer loop stored on our asyncio.Task in
        # its "._task_info" attribute (injected by the inner loop runner).
        #
        # The attribute is deliberately *weakly* referenced to avoid tight
        # coupling; if it is absent we just skip recursion.
        #
        # NOTE: this is best-effort – individual callers can override ask()
        # for richer behaviour if desired.
        task_info = {}
        with suppress(Exception):
            task_info = getattr(self._task, "task_info", {})

        recursive_tools: dict[str, Callable] = {}

        for _t, _inf in task_info.items():
            h = _inf.handle
            if h is None or not isinstance(h, SteerableToolHandle):
                continue

            async def _proxy(
                _q: str,
                _h=h,  # capture now
            ) -> str:
                nested = await _h.ask(_q)
                return await nested.result()

            # tool name encodes the call-id so collisions are impossible
            _cid = None
            with suppress(Exception):
                _cid = getattr(_inf, "call_id", None)
            _proxy.__name__ = f"ask_{_cid or 'unknown'}"
            recursive_tools[_proxy.__name__] = _proxy
        # ----------------------------------------------------------------

        # Generalized passthrough forwarding/buffering
        await self._replay_pending_passthrough_ops()
        await self._try_forward_or_buffer(
            "ask",
            {
                "question": question,
                "parent_chat_context_cont": parent_chat_context_cont,
            },
        )

        # 4.  Fire off a *stand-alone* read-only loop.
        # Compose a clear loop identifier so logs show exactly which loop the
        # question refers to, e.g. "Question(TaskScheduler.execute)" or
        # "Question(TaskScheduler.execute->TaskScheduler.ask)" when a single
        # nested handle is present.
        parent_label: str = "unknown"
        with suppress(Exception):
            parent_label = (
                getattr(self, "_log_label", None)
                or getattr(self, "_loop_id", "unknown")
                or "unknown"
            )

        loop_id_label = f"Question({parent_label})"

        # Build the message for the inspection loop – either a plain string or
        # a single string that embeds the continued parent context.
        if parent_chat_context_cont is not None:
            _ctx_text = str(parent_chat_context_cont)
            with suppress(Exception):
                _ctx_text = json.dumps(parent_chat_context_cont, indent=2)
            _ask_message = {
                "role": "user",
                "content": f"{question}\n\nparent_chat_context_continuted:\n{_ctx_text}",
            }
        else:
            _ask_message = question

        helper_handle = start_async_tool_loop(
            inspection_client,
            _ask_message,
            recursive_tools,  # may be empty
            loop_id=loop_id_label,
            parent_lineage=[],  # keep label concise (do not prepend outer lineage)
            parent_chat_context=parent_ctx,  # ← nested context
            propagate_chat_context=False,
            prune_tool_duplicates=False,
            interrupt_llm_with_interjections=False,
            max_consecutive_failures=1,
            timeout=60,
        )

        # Monkey-patch result() to record the assistant answer when available
        if not _return_reasoning_steps:
            _orig_result = helper_handle.result

            async def _rec_result():  # type: ignore[return-type]
                ans = await _orig_result()
                self._append_user_visible_assistant(ans)
                return ans

            helper_handle.result = _rec_result  # type: ignore[attr-defined]
            return helper_handle

        async def _wrap():
            answer = await helper_handle.result()
            self._append_user_visible_assistant(answer)
            return answer, inspection_client.messages

        helper_handle.result = _wrap  # type: ignore[attr-defined]
        return helper_handle

    # -- public API -----------------------------------------------------------
    @functools.wraps(SteerableToolHandle.interject, updated=())
    async def interject(
        self,
        message: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
        image_refs: list | None = None,
    ) -> None:
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.debug(f"💬 [{_label}] Interject requested: {message}")
        # No delegate forwarding – outer loop remains in control.
        # Record user-visible immediately
        self._append_user_visible_user(message, parent_chat_context_cont)

        # Generalized passthrough forwarding/buffering
        await self._replay_pending_passthrough_ops()
        await self._try_forward_or_buffer(
            "interject",
            {
                "message": message,
                "parent_chat_context_cont": parent_chat_context_cont,
                "image_refs": image_refs,
            },
            ("content", "message"),
        )

        # Buffer then forward to resolver loop. Support dict payloads when continued context provided.
        payload = (
            {
                "message": message,
                "parent_chat_context_continuted": parent_chat_context_cont,
                "image_refs": image_refs,
            }
            if parent_chat_context_cont is not None or image_refs is not None
            else message
        )
        self._early_interjects.append(payload)
        await self._queue.put(payload)

    @functools.wraps(SteerableToolHandle.stop, updated=())
    def stop(
        self,
        reason: Optional[str] = None,
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ) -> None:
        # Replay any pending buffered passthrough ops
        with suppress(Exception):
            asyncio.get_running_loop().create_task(
                self._replay_pending_passthrough_ops(),
            )
        # Idempotent guard: if already stopping, do nothing and DO NOT log again
        if self._cancel_event.is_set():
            return

        # Flip the cancel event first so concurrent callers see we are stopping
        self._cancel_event.set()

        # Only the root/top-level handle logs the stop request
        if getattr(self, "_is_root_handle", False):
            _label = getattr(self, "_log_label", None) or self._loop_id
            LOGGER.info(
                f"🛑 [{_label}] Stop requested"
                + (f" – reason: {reason}" if reason else ""),
            )
        # Do not directly forward stop to nested handles here; the inner loop
        # will propagate stop exactly once during cancellation to avoid duplicates.
        # No delegate forwarding – outer loop remains in control.

        # Pre-adoption nested handles are stopped by the inner loop via propagate_stop_once.

        # Expedite shutdown of the outer task and signal stop_event for any waiters
        with suppress(Exception):
            self._task.cancel()
        with suppress(Exception):
            self._stop_event.set()

    @functools.wraps(SteerableToolHandle.pause, updated=())
    def pause(self) -> None:
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.info(f"⏸️ [{_label}] Pause requested")
        # Propagate pause to any nested handles first (always)
        with suppress(Exception):
            task_info = getattr(self._task, "task_info", {})
            items = task_info.items() if isinstance(task_info, dict) else []
            for _t, _inf in items:
                h = getattr(_inf, "handle", None)
                if h is not None and hasattr(h, "pause"):
                    with suppress(Exception):
                        maybe = h.pause()  # may be sync or async
                        if asyncio.iscoroutine(maybe):
                            asyncio.create_task(maybe)

        self._pause_event.clear()

    @functools.wraps(SteerableToolHandle.resume, updated=())
    def resume(self) -> None:
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.info(f"▶️ [{_label}] Resume requested")
        # Propagate resume to any nested handles first (always)
        with suppress(Exception):
            task_info = getattr(self._task, "task_info", {})
            items = task_info.items() if isinstance(task_info, dict) else []
            for _t, _inf in items:
                h = getattr(_inf, "handle", None)
                if h is not None and hasattr(h, "resume"):
                    with suppress(Exception):
                        maybe = h.resume()  # may be sync or async
                        if asyncio.iscoroutine(maybe):
                            asyncio.create_task(maybe)

        self._pause_event.set()

    @functools.wraps(SteerableToolHandle.done, updated=())
    def done(self) -> bool:
        return self._task.done()

    @functools.wraps(SteerableToolHandle.result, updated=())
    async def result(self) -> str:
        """Return the final answer once the conversation loop (or delegate) completes."""
        _stopped_notice = "processed stopped early, no result"
        try:
            return await self._task
        except asyncio.CancelledError:
            # When callers cancel the OUTER loop without a delegate, return a stable notice.
            return _stopped_notice

    def get_history(self) -> list[dict]:
        """Returns the full LLM conversation history including tool calls and reasoning.

        This provides access to the rich internal trace of the async tool loop,
        including assistant reasoning, tool calls, and tool outputs. This is
        particularly valuable for understanding the
        decision-making process within the loop.

        Returns
        -------
        list[dict]
            The complete message history from the LLM client, or empty list
            if no client is available.
        """
        if self._client is not None:
            return self._client.messages
        return []

    # ── bottom-up event APIs ---------------------------------------------------
    @functools.wraps(SteerableToolHandle.next_clarification, updated=())
    async def next_clarification(self) -> dict:
        """Await the next clarification event pushed by a running tool."""
        return await self._clar_q.get()

    @functools.wraps(SteerableToolHandle.next_notification, updated=())
    async def next_notification(self) -> dict:
        """Await the next notification pushed by a running tool."""
        return await self._notification_q.get()

    @functools.wraps(SteerableToolHandle.answer_clarification, updated=())
    async def answer_clarification(self, call_id: str, answer: str) -> None:
        """Programmatically answer a clarification for a pending tool call.

        This looks up the down-queue for the given call and pushes the answer.
        Falls through silently if the mapping is missing (tool may have finished).
        """
        task_info, clar_map = {}, {}
        with suppress(Exception):
            task_info = getattr(self._task, "task_info", {})
            clar_map = getattr(self._task, "clarification_channels", {})

        # Direct lookup by full ID; if not present, try suffix matching
        down_q = None
        try:
            if call_id in clar_map:
                down_q = clar_map[call_id][1]
            else:
                for k, (_up, _down) in list(clar_map.items()):
                    if str(k).endswith(str(call_id)):
                        down_q = _down
                        break
        except Exception:
            down_q = None

        with suppress(Exception):
            if down_q is not None:
                await down_q.put(answer)

    # --- snapshotting v1: read-only capture (flat only) ---------------------
    def serialize(self) -> dict:  # type: ignore[override]
        """Return a v1 snapshot of this handle's current state (flat only).

        Behaviour (v1):
        - Proactively cancels the running loop to quiesce any in‑flight LLM/tool work.
        - Snapshot is built from the current transcript; any previously pending
          tool calls will need to be re‑scheduled by a future deserialization.
        - Nested tool loops are not supported and will raise ValueError.
        """
        # Guard: nested tool loops are out of scope for v1.
        # Check BEFORE attempting to stop/cancel, to avoid racing with teardown
        # that might clear task bookkeeping.
        try:
            task_info = getattr(self._task, "task_info", {})
        except Exception:
            task_info = {}

        if isinstance(task_info, dict):
            for _t, _inf in task_info.items():
                if getattr(_inf, "handle", None) is not None:
                    raise ValueError(
                        "Nested tool loops are not supported by v1 snapshot",
                    )

        # Best-effort quiesce: cancel the outer loop if still running. We do not
        # await completion here (serialize is synchronous); inner loop will abort
        # promptly and no further tool results should be appended.
        try:
            if not self.done():
                # Leverage standard stop semantics (sets cancel_event and cancels task)
                self.stop(reason="serialize snapshot")
        except Exception:
            pass

        # Resolve entrypoint candidate from loop_id label (e.g., "ContactManager.ask" or
        # "ContactManager.ask(x2ab)")
        raw_label = str(getattr(self, "_log_label", None) or self._loop_id or "")
        base = raw_label.split("(", 1)[0]
        if "." not in base or not base:
            # Fallback placeholder for generic tool loops; may be replaced by inline_tools entrypoint below
            cls_name, meth_name = "ToolLoop", "run"
        else:
            cls_name, meth_name = base.split(".", 1)

        # Gather transcript fragments
        msgs = []
        try:
            msgs = list(getattr(self._client, "messages", []) or [])
        except Exception:
            msgs = []

        # Build pruning context (allowed base tools for this manager method).
        # Best-effort: resolve manager class and its tools; if not found, fall back
        # to simple structural pruning (drop synthetic check_status_* calls).
        allowed_tool_names: set[str] | None = None
        mgr_cls = None
        try:
            from .state_managers import (
                BaseStateManager as _BaseStateManager,
            )  # noqa: WPS433
            from importlib import import_module as _import_module  # noqa: WPS433

            # Ensure common managers are imported so __subclasses__ is populated
            for _m in (
                "unity.contact_manager.contact_manager",
                "unity.transcript_manager.transcript_manager",
                "unity.knowledge_manager.knowledge_manager",
                "unity.guidance_manager.guidance_manager",
                "unity.secret_manager.secret_manager",
                "unity.skill_manager.skill_manager",
                "unity.task_scheduler.task_scheduler",
                "unity.file_manager.file_manager",
                "unity.image_manager.image_manager",
                "unity.web_searcher.web_searcher",
                "unity.conductor.conductor",
            ):
                try:
                    _import_module(_m)
                except Exception:
                    pass

            def _all_subclasses(cls_):
                out = set()
                for sub in cls_.__subclasses__():
                    out.add(sub)
                    out.update(_all_subclasses(sub))
                return out

            for c in _all_subclasses(_BaseStateManager):
                if getattr(c, "__name__", "") == cls_name:
                    mgr_cls = c
                    break
            if mgr_cls is not None:
                try:
                    manager = mgr_cls()
                    _tools_dict = manager.get_tools(meth_name, include_sub_tools=True)
                    if isinstance(_tools_dict, dict) and _tools_dict:
                        allowed_tool_names = set(_tools_dict.keys())
                except Exception:
                    allowed_tool_names = None
        except Exception:
            allowed_tool_names = None

        # Determine if this looks like a non-manager loop with inline tools
        inline_registry = []
        with suppress(Exception):
            inline_registry = list(getattr(self, "_inline_tools_registry", []) or [])
        use_inline_entrypoint = mgr_cls is None and bool(inline_registry)
        if use_inline_entrypoint and allowed_tool_names is None:
            try:
                allowed_tool_names = set(
                    [
                        t.get("name")
                        for t in inline_registry
                        if isinstance(t, dict) and t.get("name")
                    ],
                )
            except Exception:
                allowed_tool_names = None

        # Helper: filter assistant tool_calls against allowed names, dropping synthetic status calls
        def _prune_assistant_msg(msg: dict) -> dict | None:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                return None
            tool_calls = msg.get("tool_calls") or []
            if not isinstance(tool_calls, list) or not tool_calls:
                return None
            kept: list = []
            for tc in tool_calls:
                try:
                    fn = tc.get("function", {})
                    name = fn.get("name")
                except Exception:
                    name = None
                if not isinstance(name, str) or not name:
                    continue
                # Drop synthetic completion/status helpers
                if name.startswith("check_status_"):
                    continue
                # If we know the allowed tool set, enforce it
                if allowed_tool_names is not None and name not in allowed_tool_names:
                    # Exception: persist clarification helpers even if not part of base registry
                    if name != "request_clarification":
                        continue
                kept.append(tc)
            if not kept:
                return None
            pruned = dict(msg)
            pruned["tool_calls"] = kept
            return pruned

        assistant_steps_raw: list[dict] = []
        assistant_indices_raw: list[int] = []
        tool_results_raw: list[dict] = []
        tool_results_raw_indices: list[int] = []
        interjections: list[dict] = []
        interjections_indices: list[int] = []
        for i, m in enumerate(msgs):
            try:
                role = m.get("role")
            except Exception:
                continue
            if role == "assistant":
                pruned = _prune_assistant_msg(m)
                if pruned is not None:
                    assistant_steps_raw.append(pruned)
                    assistant_indices_raw.append(i)
            elif role == "tool":
                tool_results_raw.append(m)
                tool_results_raw_indices.append(i)
            elif role == "system" and i > 0:
                # Treat any non-leading system message as an interjection
                interjections.append(m)
                interjections_indices.append(i)

        # Build the set of referenced call_ids from pruned assistant steps
        referenced_call_ids: set[str] = set()
        for am in assistant_steps_raw:
            for tc in am.get("tool_calls", []) or []:
                try:
                    _cid = tc.get("id")
                    if isinstance(_cid, str) and _cid:
                        referenced_call_ids.add(_cid)
                except Exception:
                    continue

        # Prune tool_result messages:
        #  - keep only those whose name is allowed (when known)
        #  - keep only those whose tool_call_id is in referenced_call_ids
        #  - deduplicate by tool_call_id keeping the last occurrence, but preserve
        #    original chronological order for the survivors
        last_index_by_call_id: dict[str, int] = {}
        for idx, tm in enumerate(tool_results_raw):
            try:
                name = tm.get("name")
                call_id = tm.get("tool_call_id")
            except Exception:
                continue
            if not isinstance(call_id, str) or not call_id:
                continue
            if call_id not in referenced_call_ids:
                continue
            if allowed_tool_names is not None:
                if not isinstance(name, str) or (
                    name not in allowed_tool_names
                    and not str(name).startswith("clarification_request_")
                    and name != "request_clarification"
                ):
                    continue
            # mark last index
            last_index_by_call_id[call_id] = idx

        tool_results: list[dict] = []
        tool_results_indices: list[int] = []
        for idx, tm in enumerate(tool_results_raw):
            try:
                call_id = tm.get("tool_call_id")
            except Exception:
                call_id = None
            if (
                isinstance(call_id, str)
                and call_id in last_index_by_call_id
                and last_index_by_call_id[call_id] == idx
            ):
                tool_results.append(tm)
                try:
                    tool_results_indices.append(tool_results_raw_indices[idx])
                except Exception:
                    tool_results_indices.append(-1)

        assistant_steps = assistant_steps_raw

        # Build a clarifications summary from clarification_request_* tool messages
        # Map call_id -> base tool name from assistant_steps for readability
        callid_to_base_name: dict[str, str] = {}
        for am in assistant_steps_raw:
            for tc in am.get("tool_calls", []) or []:
                with suppress(Exception):
                    _cid = tc.get("id")
                    _nm = tc.get("function", {}).get("name")
                    if isinstance(_cid, str) and _cid and isinstance(_nm, str) and _nm:
                        callid_to_base_name[_cid] = _nm

        clarifications: list[dict] = []
        for tm in tool_results:
            try:
                _nm = str(tm.get("name"))
                _cid = str(tm.get("tool_call_id"))
                _content = tm.get("content")
            except Exception:
                continue
            if isinstance(_nm, str) and _nm.startswith("clarification_request_"):
                clarifications.append(
                    {
                        "call_id": _cid,
                        "tool_name": callid_to_base_name.get(_cid, ""),
                        "question": _content,
                    },
                )

        # Extract initial user message as recorded for the handle
        initial_user_message = None
        with suppress(Exception):
            if self._user_visible_history:
                first = self._user_visible_history[0]
                if first.get("role") == "user":
                    initial_user_message = first.get("content")

        system_message = None
        with suppress(Exception):
            system_message = getattr(self._client, "system_message", None)

        # Drain any pending notification events so they can be replayed post-deserialize
        notifications: list[dict] = []
        try:
            while True:
                evt = self._notification_q.get_nowait()
                if isinstance(evt, dict):
                    notifications.append(evt)
        except asyncio.QueueEmpty:
            pass
        except Exception:
            # Non-fatal – notifications are best-effort
            pass

        # Capture current live images (ids and annotations) for resume
        images_list: list[dict] = []
        try:
            from ._async_tool.images import (
                get_image_log_entries as _get_img_entries,
            )  # local import

            for iid, ann in _get_img_entries() or []:
                try:
                    images_list.append({"image_id": int(iid), "annotation": ann})
                except Exception:
                    continue
        except Exception:
            images_list = []
        # Fallback to any seed images captured at start time if the inner loop
        # hasn't yet initialised the image context
        if not images_list:
            try:
                for rec in getattr(self, "_seed_images_snapshot", []) or []:
                    if (
                        isinstance(rec, dict)
                        and "image_id" in rec
                        and isinstance(rec["image_id"], int)
                    ):
                        images_list.append(
                            {
                                "image_id": int(rec["image_id"]),
                                "annotation": rec.get("annotation"),
                            },
                        )
            except Exception:
                pass

        # Finalise entrypoint selection
        if use_inline_entrypoint:
            entry_field = _EntryPointInlineTools(
                tools=[_ToolRef(**t) for t in inline_registry if isinstance(t, dict)],
            )
        else:
            entry_field = _EntryPointManagerMethod(
                class_name=cls_name,
                method_name=meth_name,
            )

        snap = _LoopSnapshot(
            entrypoint=entry_field,
            loop_id=str(self._loop_id or ""),
            system_message=system_message,
            initial_user_message=initial_user_message,
            assistant_steps=assistant_steps,
            tool_results=tool_results,
            assistant_indices=assistant_indices_raw,
            tool_results_indices=tool_results_indices,
            interjections=interjections,
            interjections_indices=interjections_indices,
            clarifications=clarifications,
            notifications=notifications,
            images=images_list,
            full_messages=msgs,
            meta={
                "run_id": SESSION_ID,
                "loop_created_at": str(getattr(self, "_created_at_iso", "") or ""),
                "snapshot_at": datetime.now(timezone.utc).isoformat(),
                "assistant_context": (
                    lambda ctx: {"read": ctx.get("read"), "write": ctx.get("write")}
                )(
                    unify.get_active_context() or {},
                ),
                "semantic_cache_namespace": getattr(
                    self,
                    "_semantic_cache_namespace",
                    None,
                ),
            },
        ).model_dump()

        # Enforce v1 shape
        _validate_snapshot(snap)
        return snap

    # --- snapshotting v1: deserialization (manager entrypoints only) ---------
    @classmethod
    def deserialize(cls, snapshot: dict) -> "SteerableToolHandle":  # type: ignore[override]
        """Recreate a running handle from a v1 snapshot (flat only).

        Behaviour (v1):
        - Supports only manager entrypoints (ClassName.method).
        - Rebuilds the manager instance and its tool registry for the method.
        - Restores the system message and a curated transcript containing the
          original user turn (when available), assistant tool-calls, and any
          matching tool results placed immediately after their requesting
          assistant message. Missing results are scheduled via preflight backfill.
        - Returns a fresh handle whose loop resumes execution.
        """
        snap = _validate_snapshot(_migrate_snapshot(snapshot))

        from importlib import import_module as _import_module  # noqa: WPS433
        from .llm_client import (
            new_llm_client as _new_llm_client,
        )  # noqa: WPS433

        # Build tools mapping depending on entrypoint type
        tools: Dict[str, Callable] = {}
        loop_label: str = snap.loop_id or ""

        if snap.entrypoint.type == "manager_method":
            # Resolve manager class by name and collect tools
            from .state_managers import (
                BaseStateManager as _BaseStateManager,
            )  # noqa: WPS433

            _maybe_modules = (
                "unity.contact_manager.contact_manager",
                "unity.transcript_manager.transcript_manager",
                "unity.knowledge_manager.knowledge_manager",
                "unity.guidance_manager.guidance_manager",
                "unity.secret_manager.secret_manager",
                "unity.skill_manager.skill_manager",
                "unity.task_scheduler.task_scheduler",
                "unity.file_manager.file_manager",
                "unity.image_manager.image_manager",
                "unity.web_searcher.web_searcher",
                "unity.conductor.conductor",
            )
            for _m in _maybe_modules:
                try:
                    _import_module(_m)
                except Exception:
                    pass

            def _all_subclasses(cls_):
                out = set()
                for sub in cls_.__subclasses__():
                    out.add(sub)
                    out.update(_all_subclasses(sub))
                return out

            mgr_cls = None
            for c in _all_subclasses(_BaseStateManager):
                if getattr(c, "__name__", "") == snap.entrypoint.class_name:
                    mgr_cls = c
                    break
            if mgr_cls is None:
                # Fallback: flat non-manager loop created via start_async_tool_loop
                # with no manager context (our serializer encodes this as
                # ToolLoop.run). In this case, resume with empty inline tools.
                if str(snap.entrypoint.class_name) == "ToolLoop":
                    tools = {}
                    if not loop_label:
                        loop_label = "InlineTools"
                else:
                    raise ValueError(
                        f"Manager class not found: {snap.entrypoint.class_name}",
                    )
            else:
                manager = mgr_cls()
                method_name = snap.entrypoint.method_name
                tools = dict(manager.get_tools(method_name, include_sub_tools=True))
                if not tools:
                    raise ValueError(
                        f"No tools registered for {snap.entrypoint.class_name}.{method_name}",
                    )
                if not loop_label:
                    loop_label = f"{snap.entrypoint.class_name}.{method_name}"

        else:  # inline tools
            # Resolve each tool by import path and apply flags
            for t in snap.entrypoint.tools:
                mod = _import_module(t.module)
                obj = mod
                try:
                    for part in str(t.qualname).split("."):
                        obj = getattr(obj, part)
                except Exception as exc:
                    raise ValueError(
                        f"Failed to resolve tool {t.name} at {t.module}.{t.qualname}",
                    ) from exc
                # Apply flags expected by normalise_tools()
                try:
                    if t.read_only is True:
                        setattr(obj, "_tool_spec_read_only", True)
                    if t.manager_tool is True:
                        setattr(obj, "_tool_spec_manager_tool", True)
                except Exception:
                    pass
                tools[t.name] = obj
            if not tools:
                raise ValueError("Inline tools entrypoint contains no resolvable tools")
            if not loop_label:
                loop_label = "InlineTools"

        # Build a fresh LLM client and restore system header.
        client = _new_llm_client()
        if snap.system_message:
            client.set_system_message(snap.system_message)

        # Reconstruct minimal transcript:
        # - optional initial user message
        # - assistant tool-calls interleaved with any known results for those call_ids
        msgs: list[dict] = []

        init = snap.initial_user_message
        if init is not None:
            if isinstance(init, dict):
                msgs.append({"role": "user", "content": init})
            else:
                msgs.append({"role": "user", "content": init})

        # If indices are present, reconstruct exact ordering across assistants, tools, and interjections.
        if (
            snap.assistant_indices
            or snap.tool_results_indices
            or snap.interjections_indices
        ):
            combined: list[tuple[int, dict]] = []
            # Assistant messages with indices
            try:
                for idx_val, amsg in zip(
                    snap.assistant_indices or [],
                    snap.assistant_steps or [],
                ):
                    if isinstance(amsg, dict) and amsg.get("role") == "assistant":
                        combined.append((int(idx_val), amsg))
            except Exception:
                pass
            # Tool results with indices (skip clarification wrappers)
            try:
                for idx_val, tmsg in zip(
                    snap.tool_results_indices or [],
                    snap.tool_results or [],
                ):
                    try:
                        nm = tmsg.get("name")
                    except Exception:
                        nm = None
                    if isinstance(nm, str) and nm.startswith("clarification_request_"):
                        continue
                    combined.append((int(idx_val), tmsg))
            except Exception:
                pass
            # Interjections with indices
            try:
                for idx_val, imsg in zip(
                    snap.interjections_indices or [],
                    snap.interjections or [],
                ):
                    if isinstance(imsg, dict) and imsg.get("role") == "system":
                        combined.append((int(idx_val), imsg))
            except Exception:
                pass

            # Sort by original index and append in order
            for _, m in sorted(combined, key=lambda x: x[0]):
                msgs.append(m)
        else:
            # Backward-compat path: pair tool results by call_id after each assistant
            by_call_id: dict[str, dict] = {}
            for tm in snap.tool_results or []:
                try:
                    name_val = tm.get("name")
                    tcid = tm.get("tool_call_id")
                except Exception:
                    name_val, tcid = None, None
                # Skip clarification-request wrappers
                if isinstance(name_val, str) and name_val.startswith(
                    "clarification_request_",
                ):
                    continue
                if isinstance(tcid, str) and tcid:
                    by_call_id[tcid] = tm

            for amsg in snap.assistant_steps or []:
                if not isinstance(amsg, dict) or amsg.get("role") != "assistant":
                    continue
                msgs.append(amsg)
                for tc in amsg.get("tool_calls") or []:
                    try:
                        tcid = tc.get("id")
                    except Exception:
                        tcid = None
                    if isinstance(tcid, str) and tcid in by_call_id:
                        msgs.append(by_call_id.pop(tcid))

            # Any remaining tool results (rare): append at the end; backfill ignores them.
            msgs.extend(by_call_id.values())

        # If snapshot carried images, rebuild ImageRefs for resume
        images_param = None
        try:
            imgs = list(snap.images or [])
            if imgs:
                from unity.image_manager.types import (  # local import to avoid cycles
                    RawImageRef as _RawImageRef,
                    AnnotatedImageRef as _AnnotatedImageRef,
                    ImageRefs as _ImageRefs,
                )

                refs_list = []
                for rec in imgs:
                    try:
                        iid = int(rec.get("image_id"))
                    except Exception:
                        continue
                    ann = rec.get("annotation")
                    if ann is None or ann == "":
                        refs_list.append(_RawImageRef(image_id=iid))
                    else:
                        refs_list.append(
                            _AnnotatedImageRef(
                                raw_image_ref=_RawImageRef(image_id=iid),
                                annotation=str(ann),
                            ),
                        )
                if refs_list:
                    try:
                        images_param = _ImageRefs.model_validate(refs_list)
                    except Exception:
                        images_param = refs_list
        except Exception:
            images_param = None

        # Launch a new loop seeded with the reconstructed transcript.
        handle = start_async_tool_loop(
            client,
            msgs if msgs else (init or ""),
            tools,
            loop_id=loop_label,
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            images=images_param,
        )

        # If the last interjection occurred *after* the last assistant message,
        # request an immediate LLM turn without duplicating the interjection message.
        try:
            ai = snap.assistant_indices or []
            ii = snap.interjections_indices or []
            if ai or ii:
                last_asst = max(ai) if ai else -1
                last_intr = max(ii) if ii else -1
                if last_intr > last_asst:
                    # Queue a sentinel for the inner loop to set llm_turn_required
                    with suppress(Exception):
                        handle._queue.put_nowait({"_replay": True})  # type: ignore[attr-defined]
        except Exception:
            pass

        # Re-inject any pending notifications captured at snapshot time so
        # callers can consume them immediately after resume.
        try:
            for evt in snap.notifications or []:
                try:
                    handle._notification_q.put_nowait(evt)  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass

        return handle


# ─────────────────────────────────────────────────────────────────────────────
# 3.  A convenience wrapper that *starts* the loop and returns the handle
# ─────────────────────────────────────────────────────────────────────────────
def start_async_tool_loop(
    client: unify.AsyncUnify,
    message: str | dict | list[str | dict],
    tools: Dict[str, Callable],
    *,
    loop_id: Optional[str] = None,
    parent_lineage: Optional[list[str]] = None,
    max_consecutive_failures: int = 3,
    prune_tool_duplicates=True,
    interrupt_llm_with_interjections: bool = True,
    propagate_chat_context: bool = True,
    parent_chat_context: Optional[list[dict]] = None,
    log_steps: Union[bool, str] = True,
    max_steps: Optional[int] = 100,
    timeout: Optional[int] = 300,
    raise_on_limit: bool = False,
    include_class_in_dynamic_tool_names: bool = False,
    tool_policy: Optional[
        Callable[[int, Dict[str, Callable]], Tuple[str, Dict[str, Callable]]]
    ] = None,
    preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]] = None,
    response_format: Optional[Any] = None,
    max_parallel_tool_calls: Optional[int] = None,
    handle_cls: Optional[Type[AsyncToolLoopHandle]] = None,
    semantic_cache: Optional[Literal["read", "write", "both"]] = None,
    semantic_cache_namespace: Optional[str] = None,
    images: Optional["ImageRefs"] = None,
    evented: Optional[bool] = None,
) -> AsyncToolLoopHandle:
    """
    Kick off `_async_tool_use_loop_inner` in its own task and give the caller
    a handle for live interaction.

    Parameters
    ----------
    log_steps : bool | str, default True
        Controls verbosity of step logging to `LOGGER`:
          - False: no logging
          - True: log everything except system messages
          - "full": log everything including system messages
    """
    # Ensure a stable loop_id for consistent logging across handle and inner loop
    loop_id = loop_id if loop_id is not None else short_id()
    interject_queue: asyncio.Queue[dict | str] = asyncio.Queue()
    cancel_event = asyncio.Event()
    stop_event = asyncio.Event()
    pause_event = asyncio.Event()
    pause_event.set()  # start un-paused

    # --- enable handle passthrough -----------------------------------------
    # A single-element list is a mutable container that the inner loop can use
    # to call ``_adopt`` on the *real* outer handle once it exists.
    outer_handle_container: list = [None]

    # Determine lineage for this loop start (inherit from context when not provided)
    _parent = (
        parent_lineage if parent_lineage is not None else TOOL_LOOP_LINEAGE.get([])
    )
    _lineage = [*_parent, loop_id]

    # Run the async tool loop

    async def _loop_wrapper():
        try:
            return await async_tool_loop_inner(
                client,
                message,
                tools,
                loop_id=loop_id,
                lineage=_lineage,
                interject_queue=interject_queue,
                cancel_event=cancel_event,
                stop_event=stop_event,
                pause_event=pause_event,
                max_consecutive_failures=max_consecutive_failures,
                prune_tool_duplicates=prune_tool_duplicates,
                interrupt_llm_with_interjections=interrupt_llm_with_interjections,
                propagate_chat_context=propagate_chat_context,
                parent_chat_context=parent_chat_context,
                log_steps=log_steps,
                max_steps=max_steps,
                timeout=timeout,
                raise_on_limit=raise_on_limit,
                include_class_in_dynamic_tool_names=include_class_in_dynamic_tool_names,
                tool_policy=tool_policy,
                preprocess_msgs=preprocess_msgs,
                outer_handle_container=outer_handle_container,
                response_format=response_format,
                max_parallel_tool_calls=max_parallel_tool_calls,
                semantic_cache=semantic_cache,
                semantic_cache_namespace=semantic_cache_namespace,
                images=images,
            )
        except asyncio.CancelledError:
            raise

    task = asyncio.create_task(_loop_wrapper(), name="ToolUseLoop")

    # Determine initial_user_message for the handle from diverse input forms
    init_content = None
    if isinstance(message, dict):
        init_content = message.get("content")
    elif isinstance(message, list):
        for m in message:
            if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
                init_content = m["content"]
                break
            if isinstance(m, str):
                init_content = m
                break
    else:
        init_content = message

    HandleType = handle_cls or AsyncToolLoopHandle
    handle = HandleType(
        task=task,
        interject_queue=interject_queue,
        cancel_event=cancel_event,
        stop_event=stop_event,
        pause_event=pause_event,
        client=client,
        loop_id=loop_id,
        initial_user_message=init_content,
    )

    # Record loop creation timestamp and semantic cache namespace for diagnostics
    try:
        setattr(handle, "_created_at_iso", datetime.now(timezone.utc).isoformat())
    except Exception:
        pass
    try:
        setattr(handle, "_semantic_cache_namespace", semantic_cache_namespace)
    except Exception:
        pass

    # Capture an inline tools registry snapshot for potential serialization
    # of non-manager loops. We record import paths and flags for resolvable
    # top-level functions only (ignore closures and lambdas).
    def _build_inline_tools_registry(raw_tools: Dict[str, Callable]) -> list[dict]:
        out: list[dict] = []
        for _name, _fn in (raw_tools or {}).items():
            try:
                mod = getattr(_fn, "__module__", None)
                qn = getattr(_fn, "__qualname__", None)
                if not isinstance(mod, str) or not isinstance(qn, str):
                    continue
                # Skip closures/local defs – not importable by qualname
                if "<locals>" in qn:
                    continue
                ro = getattr(_fn, "_tool_spec_read_only", None)
                mt = getattr(_fn, "_tool_spec_manager_tool", None)
                out.append(
                    {
                        "name": _name,
                        "module": mod,
                        "qualname": qn,
                        "read_only": bool(ro) if ro is not None else None,
                        "manager_tool": bool(mt) if mt is not None else None,
                    },
                )
            except Exception:
                continue
        return out

    try:
        setattr(handle, "_inline_tools_registry", _build_inline_tools_registry(tools))
    except Exception:
        pass

    # Capture any seed images provided so serialize() can include them even if
    # the inner loop hasn't yet initialised the live image context.
    try:
        seed_images_list: list[dict] = []
        if images:
            # Support ImageRefs and plain lists via duck typing on `root`
            try:
                from ..image_manager.types import (
                    RawImageRef as _RawImageRef,
                    AnnotatedImageRef as _AnnotatedImageRef,
                )  # local import to avoid cycles

                _items = list(getattr(images, "root", images) or [])
                for ref in _items:
                    try:
                        if isinstance(ref, _AnnotatedImageRef):
                            iid = int(ref.raw_image_ref.image_id)
                            ann = str(ref.annotation)
                        elif isinstance(ref, _RawImageRef):
                            iid = int(ref.image_id)
                            ann = None
                        else:
                            continue
                        seed_images_list.append({"image_id": iid, "annotation": ann})
                    except Exception:
                        continue
            except Exception:
                pass
        setattr(handle, "_seed_images_snapshot", seed_images_list)
    except Exception:
        pass

    # Attach lineage to handle for optional external inspection
    with suppress(Exception):
        handle._lineage = list(_lineage)  # type: ignore[attr-defined]

    # Mark this handle as the root/top-level for single-stop logging semantics
    with suppress(Exception):
        handle._is_root_handle = True  # type: ignore[attr-defined]

    # Let the inner coroutine discover the outer handle so it can switch
    # steering when a nested handle requests pass-through behaviour.
    outer_handle_container[0] = handle

    return handle
