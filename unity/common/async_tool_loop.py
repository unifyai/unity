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
from ._async_tool.messages import is_non_final_tool_reply as _is_non_final_tool_reply
from ._async_tool.loop import async_tool_loop_inner
from ._async_tool.inline_tools import (
    capture_inline_tools_registry as _capture_inline_tools_registry,
    resolve_inline_tools as _resolve_inline_tools,
)
from .loop_snapshot import (
    LoopSnapshot as _LoopSnapshot,
    EntryPointManagerMethod as _EntryPointManagerMethod,
    EntryPointInlineTools as _EntryPointInlineTools,
    ToolRef as _ToolRef,
    validate_snapshot as _validate_snapshot,
    migrate_snapshot as _migrate_snapshot,
)
from ._async_tool.transcript_ops import (
    extract_assistant_and_tool_steps as _extract_assistant_and_tool_steps,
    extract_interjections as _extract_interjections,
    extract_clarifications as _extract_clarifications,
    initial_user_from_user_visible_history as _initial_user_from_user_visible_history,
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
        images: list | dict | None = None,
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
        images : list | dict | None, optional
            Live image references to make available during this ask flow.
            Implementations should forward these to any nested asks so inner
            loops can attach/ask about images (optionally with new annotations).
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
    def deserialize(
        cls,
        snapshot: dict,
        *,
        loader: Optional[Callable[[str], dict]] = None,
    ) -> "SteerableToolHandle":
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
        images: list | dict | None = None,
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

        If ``images`` are provided, the spawned inspection loop receives live
        images (helpers exposed, synthetic overview injected) and any nested asks
        can receive images (with optional new annotations).
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

                # Inert stubs for required abstract event APIs
                async def next_clarification(self) -> dict:
                    return {}

                async def next_notification(self) -> dict:
                    return {}

                async def answer_clarification(self, call_id: str, answer: str) -> None:
                    return None

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
                images: dict | list | None = None,
                _h=h,  # capture now
            ) -> str:
                # Robust forward with kwargs normalisation; tolerate older signatures
                nested = await forward_handle_call(
                    _h,
                    "ask",
                    {"question": _q, "images": images},
                    fallback_positional_keys=("question", "content"),
                )
                try:
                    return await nested.result()  # type: ignore[union-attr]
                except Exception:
                    return ""

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
                "images": images,
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
            images=images,
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
        images: list | None = None,
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
                "images": images,
            },
            ("content", "message"),
        )

        # Buffer then forward to resolver loop. Support dict payloads when continued context provided.
        payload = (
            {
                "message": message,
                "parent_chat_context_continuted": parent_chat_context_cont,
                "images": images,
            }
            if parent_chat_context_cont is not None or images is not None
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

    # --- targeted nested steerability (programmatic, no LLM) -----------------
    async def nested_steer(self, spec: dict) -> dict:
        """Apply a nested steering spec to this loop and any matched child handles.

        Programmatic (no LLM) and best‑effort: unknown or missing children are
        silently ignored and traversal stops naturally when no child is found.

        Spec schema (keys optional; omitted treated as None):
        - steps: list[dict] – ordered actions to apply on the current handle.
          Each step is a dict with keys:
            - method: str – method name to invoke (e.g., "pause", "resume",
              "interject", "stop", "ask").
            - args: any – convenience single argument; mapped to a common
              content key if not otherwise provided in ``kwargs``.
            - kwargs: dict – keyword arguments to pass to the method.
        - children: dict[str, dict] – mapping of selector → node to apply on
          matched in‑flight child handles.

        Selector matching (case‑insensitive):
        - Matches against in‑flight tool names at this level (e.g., "TaskScheduler_execute").
        - Dotted vs underscore tolerated ("TaskScheduler.execute").
        - Method‑only suffix accepted ("execute").

        Returns a summary dict: {"applied": [...], "skipped": [...]}, useful for tests.
        See module‑level ``nested_steer_on`` for full behaviour details.
        """

        return await nested_steer_on(self, spec)

    async def nested_structure(self) -> dict:
        """Return a minimal nested, read-only structure of live child loops.

        Shape (per node):
        - handle: class name of the handle (string)
        - tool: canonical entrypoint label "Class.method" when available, else class name
        - children: list of the same node shape for live (in-flight) nested handles only

        Notes
        -----
        - Non-steerable or pending base tool calls without an adopted handle are omitted.
        - Completed child handles are omitted.
        - Canonicalization strips leading "Simulated"/"Base" from class names.
        """
        return await nested_structure_on(self)

    # --- snapshotting v1: read-only capture (flat only) ---------------------
    def serialize(
        self,
        recursive: bool = False,
        *,
        store: Optional[Callable[[dict], str]] = None,
    ) -> dict:  # type: ignore[override]
        """Return a v1 snapshot of this handle's current state.

        Behaviour (v1):
        - Proactively cancels the running loop to quiesce any in‑flight LLM/tool work.
        - Builds the snapshot from the current transcript; any previously pending
          tool calls will be re‑scheduled by deserialization via preflight backfill.
        - When ``recursive=False`` (default), nested tool loops are not supported
          and a ``ValueError`` is raised if any are detected.
        - When ``recursive=True``, in‑flight nested child handles are captured into
          ``meta.children``. For each child, an inline ``snapshot`` is embedded or a
          ``ref.path`` is written using the optional ``store`` callback.
        """
        # Guard / discovery for nested tool loops. When recursive=False (default),
        # nested handles are not supported and will raise a ValueError. When
        # recursive=True, we will attempt to serialize in-flight children as well.
        try:
            task_info = getattr(self._task, "task_info", {})
        except Exception:
            task_info = {}

        if not recursive and isinstance(task_info, dict):
            for _t, _inf in task_info.items():
                if getattr(_inf, "handle", None) is not None:
                    raise ValueError(
                        "Nested tool loops are not supported by v1 snapshot",
                    )

        # NOTE: We intentionally defer quiescing (stop) until AFTER we've captured
        # the current transcript and, when recursive=True, gathered child snapshots.
        # Stopping too early can cancel nested handles before they are recorded.

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

        # Use shared transcript helpers to extract assistant/tool steps, interjections,
        # clarifications and the initial user-visible message.
        extracted = _extract_assistant_and_tool_steps(
            msgs,
            allowed_tools=allowed_tool_names,
        )
        assistant_steps = extracted["assistant_steps"]
        assistant_indices_raw = extracted["assistant_indices"]
        tool_results = extracted["tool_results"]
        tool_results_indices = extracted["tool_results_indices"]
        interjections, interjections_indices = _extract_interjections(msgs)
        clarifications = _extract_clarifications(
            assistant_steps,
            tool_results,
            callid_to_tool_name=extracted.get("callid_to_tool_name", {}),
        )
        initial_user_message = _initial_user_from_user_visible_history(
            getattr(self, "_user_visible_history", []) or [],
        )

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
        # If recursive capture was requested, attach a children manifest under meta.
        if recursive and isinstance(task_info, dict):
            children: list[dict] = []
            for _t, _inf in task_info.items():
                child = getattr(_inf, "handle", None)
                if child is None:
                    continue
                state = "done"
                try:
                    state = "in_flight" if not bool(child.done()) else "done"
                except Exception:
                    pass
                child_snapshot = None
                if state == "in_flight":
                    try:
                        child_snapshot = child.serialize(recursive=True, store=store)
                    except Exception:
                        child_snapshot = None
                entry = {
                    "call_id": getattr(_inf, "call_id", None),
                    "tool_name": getattr(_inf, "name", None),
                    "is_passthrough": bool(getattr(_inf, "is_passthrough", False)),
                    "state": state,
                }
                if isinstance(child_snapshot, dict):
                    ref_path = None
                    try:
                        if store is not None:
                            ref_path = store(child_snapshot)
                    except Exception:
                        ref_path = None
                    if ref_path:
                        entry["ref"] = {"path": ref_path}
                    else:
                        entry["snapshot"] = child_snapshot
                children.append(entry)
            try:
                if children:
                    # Ensure meta exists before augmenting
                    if snap.get("meta") is None:
                        snap["meta"] = {}
                    snap["meta"]["children"] = children
                    try:
                        LOGGER.info(
                            f"🧩 Snapshot captured {len(children)} in-flight child loop(s)",
                        )
                    except Exception:
                        pass
            except Exception:
                pass

        # Best-effort quiesce: cancel the loop after snapshotting to avoid races on resume.
        try:
            if not self.done():
                self.stop(reason="serialize snapshot")
        except Exception:
            pass

        _validate_snapshot(snap)
        return snap

    # --- snapshotting v1: deserialization (manager entrypoints only) ---------
    @classmethod
    def deserialize(
        cls,
        snapshot: dict,
        *,
        loader: Optional[Callable[[str], dict]] = None,
    ) -> "SteerableToolHandle":  # type: ignore[override]
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
            tools = _resolve_inline_tools(snap.entrypoint.tools)
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

        # Detect pending base-tool call_ids from snapshot so we can strip non-final placeholders
        # and let preflight backfill re-schedule them after resume.

        assistant_call_ids: set[str] = set()
        try:
            for am in snap.assistant_steps or []:
                for tc in am.get("tool_calls", []) or []:
                    try:
                        _cid = tc.get("id")
                        if isinstance(_cid, str) and _cid:
                            assistant_call_ids.add(_cid)
                    except Exception:
                        continue
        except Exception:
            assistant_call_ids = set()

        final_call_ids: set[str] = set()
        try:
            for tm in snap.tool_results or []:
                try:
                    _cid = tm.get("tool_call_id")
                except Exception:
                    _cid = None
                if not isinstance(_cid, str) or not _cid:
                    continue
                if _cid not in assistant_call_ids:
                    continue
                # Only treat as final when not a placeholder/progress/clarification wrapper
                if not _is_non_final_tool_reply(tm):
                    final_call_ids.add(_cid)
        except Exception:
            final_call_ids = set()

        pending_call_ids: set[str] = set()
        try:
            pending_call_ids = assistant_call_ids - final_call_ids
        except Exception:
            pending_call_ids = set()

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
            # Tool results with indices (skip clarification wrappers and any pending placeholders)
            try:
                for idx_val, tmsg in zip(
                    snap.tool_results_indices or [],
                    snap.tool_results or [],
                ):
                    try:
                        nm = tmsg.get("name")
                        tcid = tmsg.get("tool_call_id")
                    except Exception:
                        nm, tcid = None, None
                    # Skip if this tool result corresponds to a pending base call
                    if isinstance(tcid, str) and tcid in pending_call_ids:
                        continue
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
                # Skip non-final placeholders for pending calls
                if isinstance(tcid, str) and tcid in pending_call_ids:
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
        # Prepare resume_children payloads (if any) to adopt nested handles in-flight.
        _resume_children_payload: list[dict] = []
        try:
            meta = getattr(snap, "meta", None) or {}
            ch_list = meta.get("children") if isinstance(meta, dict) else None
            if isinstance(ch_list, list):
                for rec in ch_list:
                    try:
                        if not isinstance(rec, dict):
                            continue
                        if rec.get("state") != "in_flight":
                            continue
                        child_snap = rec.get("snapshot")
                        if not isinstance(child_snap, dict):
                            # Try resolve by reference using loader callback
                            try:
                                ref = rec.get("ref") or {}
                                path = (
                                    ref.get("path") if isinstance(ref, dict) else None
                                )
                                if (
                                    loader is not None
                                    and isinstance(path, str)
                                    and path
                                ):
                                    child_snap = loader(path)
                            except Exception:
                                child_snap = None
                        if not isinstance(child_snap, dict):
                            continue
                        child_handle = cls.deserialize(child_snap, loader=loader)
                        _resume_children_payload.append(
                            {
                                "call_id": rec.get("call_id"),
                                "tool_name": rec.get("tool_name"),
                                "is_passthrough": bool(
                                    rec.get("is_passthrough", False),
                                ),
                                "handle": child_handle,
                            },
                        )
                    except Exception:
                        continue
        except Exception:
            _resume_children_payload = []

        handle = start_async_tool_loop(
            client,
            msgs if msgs else (init or ""),
            tools,
            loop_id=loop_label,
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            images=images_param,
            resume_children=_resume_children_payload or None,
        )

        # Diagnostics: log adopted children count
        try:
            if _resume_children_payload:
                LOGGER.info(
                    f"🔁 Resuming with {len(_resume_children_payload)} in-flight child loop(s)",
                )
        except Exception:
            pass

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


# --- module-level generic helper ----------------------------------------------
async def nested_steer_on(handle: Any, spec: dict) -> dict:
    """Apply a nested steering spec to any compatible handle without LLM calls.

    Node schema (keys optional; omitted treated as None):
      - steps: list[dict] – ordered actions to apply on the current handle. Each
        step supports:
          - method: str – method name to invoke (e.g., "pause", "resume",
            "interject", "stop", "ask").
          - args: any – convenience single argument; mapped to a common content
            key if not otherwise provided in kwargs.
          - kwargs: dict – keyword arguments to pass to the method.
      - children: dict[str, dict] – mapping of selector → child node, where selector
        is matched against in-flight tool names at this level (from task_info metadata).

    Behaviour:
      - Apply the local steps in order (when present), using robust forwarding.
      - Discover loop-children via _task.task_info and recurse into matched child handles.
      - If no loop-children matched, fall back to common wrapper attributes ("_actor_handle",
        "_current_handle") when exactly one child node is provided.
      - Unknown selectors are ignored; traversal stops naturally when no child is found.

    Selector matching rules (case-insensitive):
      - Accept exact tool name matches (e.g., "TaskScheduler_execute").
      - Accept dotted form (e.g., "TaskScheduler.execute") matching underscore names.
      - Accept method-only suffix (e.g., "execute" matches "TaskScheduler_execute").
    """

    # Best-effort label for diagnostics
    try:
        label = (
            getattr(handle, "_log_label", None)
            or getattr(handle, "_loop_id", None)
            or getattr(getattr(handle, "__class__", object), "__name__", "handle")
        )
    except Exception:
        label = "handle"

    try:
        steps_count = 0
        children_count = 0
        try:
            if isinstance(spec, dict):
                steps = spec.get("steps") or []
                if isinstance(steps, list):
                    steps_count = len(steps)
                children = spec.get("children") or {}
                if isinstance(children, dict):
                    children_count = len(children)
        except Exception:
            steps_count, children_count = 0, 0
        LOGGER.info(
            f"🎯 [{label}] Nested steer requested – steps={steps_count} children={children_count}",
        )
    except Exception:
        pass

    results: dict = {"applied": [], "skipped": []}

    def _norm(s: str) -> str:
        try:
            return str(s).replace(".", "_").strip().lower()
        except Exception:
            return str(s).lower()

    def _selector_matches_name(selector: str, candidate_name: str) -> bool:
        """Name-only matching with relaxed rules (case-insensitive, suffix)."""
        sel = _norm(selector)
        cand = _norm(candidate_name)
        if not sel or not cand:
            return False
        if sel == cand:
            return True
        # Suffix match: allow targeting by bare method name
        try:
            suffix = cand.split("_", 1)[1]
        except Exception:
            suffix = cand
        return sel == suffix

    def _selector_hits(
        selector: str,
        candidate_name: str,
        candidate_call_id: str | None,
    ) -> bool:
        """Extended selector matcher supporting wildcard and call_id forms.

        Accepted forms:
        - "*" → match all children at the current level
        - exact call_id → matches the tool with that call_id
        - "#<id_suffix>" → match when call_id endswith(<id_suffix>)
        - "<name>#<call_id|suffix>" → match by name AND call_id (or its suffix)
        - name-only → uses relaxed name rules (dotted vs underscore; suffix method)
        """
        try:
            s = str(selector or "").strip()
        except Exception:
            s = str(selector)

        # Wildcard for broadcast at this level
        if s == "*":
            return True

        # Direct call_id match
        if candidate_call_id and s == str(candidate_call_id):
            return True

        # call_id suffix match via leading '#'
        if s.startswith("#"):
            suf = s[1:]
            if candidate_call_id and (
                candidate_call_id.endswith(suf) or candidate_call_id == suf
            ):
                return True
            return False

        # Combined name#id form
        if "#" in s:
            try:
                left, right = s.split("#", 1)
            except Exception:
                left, right = s, ""
            if _selector_matches_name(left, candidate_name):
                if not candidate_call_id:
                    return False
                return candidate_call_id == right or candidate_call_id.endswith(right)
            return False

        # Default: name-only
        return _selector_matches_name(s, candidate_name)

    # ───── status helpers ──────────────────────────────────────────────────
    def _merge_status(a: str, b: str) -> str:
        try:
            if a == b:
                return a
            if a == "full" or b == "full":
                return "full"
            if a == "partial" or b == "partial":
                return "partial"
            return "none"
        except Exception:
            return a or b or "none"

    def _empty_status_node() -> dict:
        return {"self": "none", "children": {}}

    def _aggregate_nodes(nodes: list[dict]) -> dict:
        if not nodes:
            return _empty_status_node()
        agg = _empty_status_node()
        # self
        for n in nodes:
            try:
                agg["self"] = _merge_status(agg["self"], str(n.get("self", "none")))
            except Exception:
                pass
        # children
        child_keys: set[str] = set()
        for n in nodes:
            try:
                child_keys.update(n.get("children", {}).keys())
            except Exception:
                pass
        for k in child_keys:
            lst = []
            for n in nodes:
                try:
                    ch = n.get("children", {}).get(k)
                    if isinstance(ch, dict):
                        lst.append(ch)
                except Exception:
                    continue
            agg["children"][k] = _aggregate_nodes(lst)
        return agg

    def _record_status(flat_key: str, node_status: dict) -> None:
        try:
            children_view = {
                k: (v.get("self") if isinstance(v, dict) else "none")
                for k, v in (node_status.get("children", {}) or {}).items()
            }
            results.setdefault("status", {})[flat_key] = {
                "self": node_status.get("self", "none"),
                "children": children_view,
            }
        except Exception:
            pass

    def _lookup_path(node_status: dict, dotted: str) -> str:
        try:
            segs = [s for s in str(dotted).split(".") if s]
        except Exception:
            segs = []
        cur = node_status
        for s in segs:
            try:
                ch = cur.get("children", {}).get(s)
            except Exception:
                ch = None
            if not isinstance(ch, dict):
                return "none"
            cur = ch
        try:
            return str(cur.get("self", "none"))
        except Exception:
            return "none"

    async def _apply(
        h,
        node: dict | None,
        path: list[str],
        sel_path: list[str],
    ) -> dict:
        node = node or {}
        steps = node.get("steps") or []
        attempted_local = False

        # 1) Apply local steps in order
        if isinstance(steps, list) and steps:
            for step in steps:
                try:
                    method = None
                    args = None
                    kwargs = {}
                    try:
                        method = step.get("method")
                        args = step.get("args")
                        kwargs = step.get("kwargs") or {}
                    except Exception:
                        method, args, kwargs = None, None, {}
                    if not isinstance(method, str) or not method:
                        continue
                    call_kwargs = dict(kwargs)
                    # Build positional args (explicit); no aliasing heuristics
                    call_args: list = []
                    if args is not None:
                        if isinstance(args, (list, tuple)):
                            call_args = list(args)
                        else:
                            call_args = [args]
                    attempted_local = True
                    try:
                        await forward_handle_call(
                            h,
                            method,
                            call_kwargs,
                            call_args=call_args,
                            fallback_positional_keys=(),
                        )
                        try:
                            results["applied"].append(
                                {"path": list(path), "method": method},
                            )
                        except Exception:
                            pass
                        try:
                            _p = "/".join(str(p) for p in path)
                            LOGGER.debug(
                                f"✅ [{label}] Applied method '{method}' at path {_p}",
                            )
                        except Exception:
                            pass
                    except Exception:
                        try:
                            _p = "/".join(str(p) for p in path)
                            LOGGER.debug(
                                f"⚠️  [{label}] Failed to apply '{method}' at path {_p}",
                            )
                        except Exception:
                            pass
                except Exception:
                    # Continue to next step
                    pass

        # 2) Recurse into matched children
        children = node.get("children") or {}

        # Discover loop children via task_info when available
        task_info = {}
        with suppress(Exception):
            task_info = getattr(getattr(h, "_task", None), "task_info", {}) or {}

        matched_any = False
        matched_selectors: set[str] = set()
        per_selector_nodes: dict[str, list[dict]] = {}

        if isinstance(children, dict) and children:
            if isinstance(task_info, dict) and task_info:
                for sel, child_node in children.items():
                    for _t, _inf in list(task_info.items()):
                        try:
                            _name = getattr(_inf, "name", None)
                            _cid = getattr(_inf, "call_id", None)
                            _child = getattr(_inf, "handle", None)
                        except Exception:
                            _name, _cid, _child = None, None, None
                        if (
                            _name
                            and _child is not None
                            and _selector_hits(sel, _name, _cid)
                        ):
                            matched_any = True
                            try:
                                matched_selectors.add(str(sel))
                            except Exception:
                                pass
                            try:
                                _p = "/".join(str(p) for p in path)
                                LOGGER.debug(
                                    f"↘️ [{label}] Descend: selector {sel!r} matched child {_name!r} at {_p}",
                                )
                            except Exception:
                                pass
                            child_status = await _apply(
                                _child,
                                child_node,
                                path + [str(_name)],
                                sel_path + [str(sel)],
                            )
                            per_selector_nodes.setdefault(str(sel), []).append(
                                child_status,
                            )

            # Record any unmatched selectors at this level as skipped
            try:
                for _sel in children.keys():
                    if str(_sel) not in matched_selectors:
                        try:
                            results["skipped"].append(
                                {"path": list(path), "selector": str(_sel)},
                            )
                        except Exception:
                            pass
            except Exception:
                pass

        # Aggregate child statuses per selector
        aggregated_children: dict[str, dict] = {}
        for sel, lst in per_selector_nodes.items():
            aggregated_children[sel] = _aggregate_nodes(lst)
        # Ensure explicit selectors exist (even if no match)
        for sel in (children.keys() if isinstance(children, dict) else []):
            aggregated_children.setdefault(str(sel), _empty_status_node())

        # Compute self status
        if children:
            # Based on direct child selectors' self statuses
            direct = [
                aggregated_children[s]["self"] for s in aggregated_children.keys()
            ]
            if direct and all(s == "full" for s in direct):
                self_status = "full"
            elif direct and (
                "partial" in direct or ("full" in direct and "none" in direct)
            ):
                self_status = "partial"
            elif direct:
                # all none
                self_status = "none"
            else:
                self_status = "none"
        else:
            # No children in spec: local steps attempted → full; else none
            self_status = "full" if attempted_local else "none"

        node_status = {"self": self_status, "children": aggregated_children}

        # 3) Evaluate optional conditions at this node
        conditions = node.get("conditions") or []

        def _eval_when(expr: dict | None) -> bool:
            if not isinstance(expr, dict):
                return False
            if "any" in expr:
                arr = expr.get("any") or expr.get("or") or []
                vals = [
                    bool(_eval_when(e)) for e in (arr if isinstance(arr, list) else [])
                ]
                return any(vals)
            if "all" in expr:
                arr = expr.get("all") or expr.get("and") or []
                vals = [
                    bool(_eval_when(e)) for e in (arr if isinstance(arr, list) else [])
                ]
                return all(vals)
            if "not" in expr:
                return not _eval_when(expr.get("not"))
            # leaf forms
            try:
                if "self" in expr:
                    return str(node_status.get("self", "none")) == str(expr.get("self"))
                if "selector" in expr:
                    sel = str(expr.get("selector"))
                    want = str(expr.get("status", "none"))
                    got = str(
                        (node_status.get("children", {}).get(sel) or {}).get(
                            "self",
                            "none",
                        ),
                    )
                    return got == want
                if "path" in expr:
                    p = str(expr.get("path"))
                    want = str(expr.get("status", "none"))
                    got = _lookup_path(node_status, p)
                    return got == want
            except Exception:
                return False
            return False

        if isinstance(conditions, list) and conditions:
            for cond in conditions:
                try:
                    when = cond.get("when") if isinstance(cond, dict) else None
                    decision = bool(_eval_when(when))
                    applied_here: list[dict] = []
                    alt_applied: list[dict] = []
                    steps_to_apply = (
                        cond.get("then") if decision else cond.get("else_then")
                    )
                    if isinstance(steps_to_apply, list):
                        for step in steps_to_apply:
                            try:
                                method = step.get("method")
                                args = step.get("args")
                                kwargs = step.get("kwargs") or {}
                            except Exception:
                                method, args, kwargs = None, None, {}
                            if not isinstance(method, str) or not method:
                                continue
                            call_kwargs = dict(kwargs)
                            call_args: list = []
                            if args is not None:
                                if isinstance(args, (list, tuple)):
                                    call_args = list(args)
                                else:
                                    call_args = [args]
                            try:
                                await forward_handle_call(
                                    h,
                                    method,
                                    call_kwargs,
                                    call_args=call_args,
                                    fallback_positional_keys=(),
                                )
                                entry = {"path": list(path), "method": method}
                                results["applied"].append(entry)
                                applied_here.append(entry)
                            except Exception:
                                pass
                    try:
                        results.setdefault("conditions_fired", []).append(
                            {
                                "path": list(path),
                                "when": when,
                                "result": decision,
                                "then_applied": applied_here if decision else [],
                                "else_applied": [] if decision else alt_applied,
                            },
                        )
                    except Exception:
                        pass
                except Exception:
                    # ignore malformed condition
                    pass

        # Record flattened status entry for this selector path
        try:
            flat_key = ".".join([str(p) for p in path])
            _record_status(flat_key, node_status)
        except Exception:
            pass

        return node_status

    await _apply(handle, spec or {}, [str(label)], [])
    return results

    await _apply(handle, spec, [str(label)])
    return results


# --- module-level nested structure introspection (read-only) -------------------
async def nested_structure_on(
    handle: Any,
    *,
    max_depth: Optional[int] = None,
) -> dict:
    """Return a minimal nested structure for any compatible handle.

    Each node contains:
      - handle: class name of the handle
      - tool: canonical "Class.method" when available, else class name
      - children: only live, steerable nested handles (pending/done omitted)
    """

    def _canon_cls(name: str) -> str:
        try:
            s = str(name or "")
        except Exception:
            s = str(name)
        if s.startswith("Simulated") and len(s) > 9:
            return s[9:]
        if s.startswith("Base") and len(s) > 4:
            return s[4:]
        return s

    def _tool_of(h) -> str | None:
        # Prefer stable loop_id set by starters (e.g., "ContactManager.ask")
        try:
            raw = getattr(h, "_loop_id", None) or ""
        except Exception:
            raw = ""
        base = str(raw).split("(", 1)[0]
        if "." in base:
            cls, meth = base.split(".", 1)
            return f"{_canon_cls(cls)}.{meth}"
        # Fallback to canonicalized class name
        try:
            cls_name = _canon_cls(
                getattr(getattr(h, "__class__", object), "__name__", ""),
            )
            return cls_name or None
        except Exception:
            return None

    def _is_live(child) -> bool:
        try:
            if hasattr(child, "done"):
                d = child.done()
                if isinstance(d, bool):
                    return not d
        except Exception:
            pass
        # If we cannot determine, treat as live to allow traversal
        return True

    async def _walk(h, depth: int, visited: set[int]) -> dict:
        try:
            hid = id(h)
        except Exception:
            hid = None
        if hid is not None and hid in visited:
            return {
                "handle": getattr(h, "__class__", object).__name__,
                "tool": _tool_of(h),
                "children": [],
            }
        if hid is not None:
            visited.add(hid)

        node: dict = {
            "handle": getattr(h, "__class__", object).__name__,
            "tool": _tool_of(h),
            "children": [],
        }

        if max_depth is not None and depth >= max_depth:
            return node

        # Discover children via task_info
        task_info = {}
        with suppress(Exception):
            task_info = getattr(getattr(h, "_task", None), "task_info", {}) or {}

        seen_child_ids: set[int] = set()
        if isinstance(task_info, dict) and task_info:
            for meta in list(task_info.values()):
                try:
                    child = getattr(meta, "handle", None)
                except Exception:
                    child = None
                if child is None or not _is_live(child):
                    continue
                with suppress(Exception):
                    seen_child_ids.add(id(child))
                nested = await _walk(child, depth + 1, visited)
                node["children"].append(nested)

        # Wrapper discovery via standardized helper
        try:
            from .handle_wrappers import (
                discover_wrapped_handles as _discover_wrapped_handles,
            )
        except Exception:
            _discover_wrapped_handles = None  # type: ignore

        if _discover_wrapped_handles is not None:
            try:
                pairs = list(_discover_wrapped_handles(h) or [])
            except Exception:
                pairs = []

            for _src, child in pairs:
                if child is None or not _is_live(child):
                    continue
                try:
                    cid = id(child)
                    if cid in seen_child_ids:
                        continue
                    seen_child_ids.add(cid)
                except Exception:
                    pass
                nested = await _walk(child, depth + 1, visited)
                node["children"].append(nested)

        return node

    return await _walk(handle, 0, set())


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
    resume_children: Optional[list[dict]] = None,
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
                resume_children=resume_children,
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

    try:
        setattr(handle, "_inline_tools_registry", _capture_inline_tools_registry(tools))
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
