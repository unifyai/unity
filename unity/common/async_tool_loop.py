import asyncio
import unify
import unillm
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
from .llm_helpers import canonicalize_handle_class_name as _canon_handle_name
from ._async_tool.loop_config import TOOL_LOOP_LINEAGE
from ._async_tool.messages import forward_handle_call
from ._async_tool.messages import is_non_final_tool_reply as _is_non_final_tool_reply
from ._async_tool.loop import async_tool_loop_inner
from typing import Iterable

# inline-tools support removed in simplified manager-only snapshots
from .loop_snapshot import (
    LoopSnapshot as _LoopSnapshot,
    validate_snapshot as _validate_snapshot,
    migrate_snapshot as _migrate_snapshot,
)
from ._async_tool.transcript_ops import (
    extract_assistant_and_tool_steps as _extract_assistant_and_tool_steps,
    extract_interjections as _extract_interjections,
    extract_clarifications as _extract_clarifications,
    initial_user_from_user_visible_history as _initial_user_from_user_visible_history,
)
from ._async_tool.multi_handle import (
    MultiHandleCoordinator,
    MultiRequestHandle,
)
from ._async_tool.tagging import tag_message_with_request

if TYPE_CHECKING:
    from ..image_manager.types.image_refs import ImageRefs


# ─────────────────────────────────────────────────────────────────────────────
# Helper: derive entrypoint ("Class.method") from loop_id lineage label
# Accepts labels like:
#   "ContactManager.ask"
#   "ContactManager.ask(x2ab)"
#   "ContactManager.update->ContactManager.ask(x2ab)"
#   "TaskScheduler.execute->TaskScheduler.ask"
# Returns (class_name, method_name) or raises ValueError when not parseable.
# ─────────────────────────────────────────────────────────────────────────────
def _parse_entrypoint_from_loop_id_label(label: str) -> tuple[str, str]:
    s = str(label or "")
    # When lineage is present, keep only the last segment
    try:
        if "->" in s:
            s = s.split("->")[-1]
    except Exception:
        pass
    # Strip any trailing unique-id in parentheses
    try:
        s = s.split("(", 1)[0]
    except Exception:
        s = s
    s = s.strip()
    if "." not in s or not s:
        raise ValueError("Manager entrypoint required (Class.method) in loop_id label")
    cls_name, meth_name = s.split(".", 1)
    if not cls_name or not meth_name:
        raise ValueError("Manager entrypoint required (Class.method) in loop_id label")
    return cls_name, meth_name


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
        Ask about the current status or progress of this running task.

        Use this to check on updates, get a summary of what has happened so far,
        or ask clarifying questions about the task's state without modifying it.

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
        images: list | dict | None = None,
    ) -> Awaitable[Optional[str]] | Optional[str]:
        """Provide additional information or instructions to the running task.

        Use this to give the task new context, correct its approach, or add
        requirements mid-flight without stopping or restarting it.

        Parameters
        ----------
        message : str
            The user interjection to inject into the loop.
        parent_chat_context_cont : list[dict] | None, optional
            The parent chat context continued since the start of this loop.
            When provided, implementations should ensure the LLM sees this
            continuation alongside the interjection.
        images : list | dict | None, optional
            Live image references to make available during this interjection.
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
        """Stop this task immediately, cancelling any pending work.

        Use this when the task should be terminated. This is a destructive
        action that cannot be undone.

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
    async def pause(self) -> Optional[str]:
        """Pause this task temporarily without cancelling it.

        Use this when the user needs to step away or wants to hold the task.
        In-flight operations continue, but no new actions are taken until resumed.

        Behaviour
        ---------
        - Freezes the assistant's next LLM turn until :pyfunc:`resume` is called.
        - Any in‑flight tool calls continue executing; the pause only affects the
          assistant's ability to speak/advance turns.
        - Nested handles (if any) should receive a corresponding pause signal
          before the outer loop transitions into the paused state.
        """

    @abstractmethod
    async def resume(self) -> Optional[str]:
        """Resume a task that was previously paused.

        Use this to continue a paused task. Any work that completed while
        paused will be processed before the task continues.

        Behaviour
        ---------
        - Allows the assistant to proceed with the next LLM turn.
        - If tools completed while paused, their results are processed first and
          then the assistant replies.
        - Nested handles (if any) should receive a corresponding resume signal
          before unfreezing the outer loop.
        """

    @abstractmethod
    def done(self) -> Awaitable[bool] | bool:
        """Check if this task has completed."""

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
        """Answer a clarification question that the task is waiting on.

        Use this when the task has asked for more information and is blocked
        waiting for a response. Provide the call_id from the clarification
        request and the answer text.

        Parameters
        ----------
        call_id : str
            Identifier of the original assistant tool call that requested the
            clarification.
        answer : str
            The clarification answer text to provide to the waiting tool.

        Behaviour
        ---------
        - Looks up the queued clarification channel for ``call_id`` and delivers
          the provided ``answer`` to the waiting tool.
        - If the mapping is missing (e.g., the tool already finished or the loop
          resumed on its own), the call is a no‑op.
        - Implementations should not raise in the absence of a matching channel;
          best‑effort delivery is sufficient.
        """

    async def nested_steer(self, spec: dict) -> dict:
        """Apply a nested steering spec.

        See module-level ``_nested_steer_on`` for details.
        """
        return await _nested_steer_on(self, spec)

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
        client: "unillm.AsyncUnify | None" = None,
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
        # Unified steering event log: records all steering calls made on this handle.
        # Each entry: {"t": perf_counter(), "method": str, "args": tuple, "kwargs": dict, "fallback": tuple}
        self._steer_log: list[dict] = []

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
        # No pending passthrough ops buffer; unified steer_log replaces ad-hoc buffering.

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
        return []

    def _has_scheduled_tools(self) -> bool:
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

    async def _record_and_forward(
        self,
        method_name: str,
        *,
        args: tuple | list | None = None,
        kwargs: dict | None = None,
        fallback: tuple[str, ...] = (),
        had_passthrough: bool | None = None,
        forwarded_to: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        # Record the steering event with a timestamp for later replay at adoption time.
        try:
            from time import perf_counter as _pc  # local import to avoid top pollution
        except Exception:  # pragma: no cover
            _pc = lambda: 0.0  # type: ignore
        # Snapshot which call_ids are already scheduled at the moment of recording.
        # This provides a robust lower bound for adoption-time replay without relying
        # on fragile cross-task timing windows.
        try:
            _ti = getattr(self._task, "task_info", {}) or {}
            _scheduled_ids = []
            for _t, _inf in _ti.items():
                try:
                    _scheduled_ids.append(str(getattr(_inf, "call_id", "")))
                except Exception:
                    continue
        except Exception:
            _scheduled_ids = []
        rec = {
            "t": _pc(),
            "method": str(method_name or ""),
            "args": tuple(args or ()),
            "kwargs": dict(kwargs or {}),
            "fallback": tuple(fallback or ()),
            "had_passthrough": bool(had_passthrough),
            "forwarded_to": list(forwarded_to or []),
            "scheduled_call_ids": _scheduled_ids,
        }
        try:
            self._steer_log.append(rec)
        except Exception:
            pass

    async def ask(
        self,
        question: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
        images: list | dict | None = None,
        _return_reasoning_steps: bool = False,
        **kwargs,
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

        # Centralized steering: record steer event; functional forwarding happens via mirror path
        await self._record_and_forward(
            "ask",
            kwargs={
                "question": question,
                "parent_chat_context_cont": parent_chat_context_cont,
                "images": images,
                **(kwargs or {}),
            },
            had_passthrough=False,
            forwarded_to=[],
        )

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

                async def pause(self): ...

                async def resume(self): ...

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
        from .llm_client import new_llm_client

        inspection_client = new_llm_client()
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
            # Only consider live passthrough child handles for recursive ask tools
            is_passthrough = False
            try:
                is_passthrough = bool(getattr(_inf, "is_passthrough", False))
            except Exception:
                is_passthrough = False
            if (
                h is None
                or not isinstance(h, SteerableToolHandle)
                or not is_passthrough
            ):
                continue

            async def _proxy(
                question: str | None = None,
                images: dict | list | None = None,
                _h=h,  # capture now
                _seed_images=images,  # capture outer ask() images to use by default
            ):
                # Robust forward; return the downstream ask handle so the inspection loop can adopt it
                try:
                    if images is None:
                        images = _seed_images
                except Exception:
                    pass
                return await forward_handle_call(
                    _h,
                    "ask",
                    {"question": question, "images": images},
                    fallback_positional_keys=("question", "content"),
                )

            # tool name encodes the call-id so collisions are impossible
            _cid = None
            with suppress(Exception):
                _cid = getattr(_inf, "call_id", None)
            _proxy.__name__ = f"ask_{_cid or 'unknown'}"
            recursive_tools[_proxy.__name__] = _proxy
        # ----------------------------------------------------------------

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

        # If passthrough children are present, seed assistant tool_calls to invoke ask_* immediately.
        seeded_batch = None
        try:
            if isinstance(recursive_tools, dict) and recursive_tools:
                tool_calls = []
                import json as _json  # local alias to avoid top-level pollution

                for _name in list(recursive_tools.keys()):
                    try:
                        tool_calls.append(
                            {
                                "id": f"seed_{_name}",
                                "type": "function",
                                "function": {
                                    "name": _name,
                                    "arguments": _json.dumps({"question": question}),
                                },
                            },
                        )
                    except Exception:
                        continue
                if tool_calls:
                    # Build a normalized user message dict
                    if isinstance(_ask_message, dict):
                        _user_msg = _ask_message
                    else:
                        _user_msg = {"role": "user", "content": _ask_message}
                    seeded_batch = [
                        _user_msg,
                        {"role": "assistant", "content": "", "tool_calls": tool_calls},
                    ]
        except Exception:
            seeded_batch = None

        helper_handle = start_async_tool_loop(
            inspection_client,
            seeded_batch if seeded_batch is not None else _ask_message,
            recursive_tools,  # may be empty
            loop_id=loop_id_label,
            parent_lineage=[],  # keep label concise (do not prepend outer lineage)
            parent_chat_context=parent_ctx,  # ← nested context
            propagate_chat_context=False,
            prune_tool_duplicates=False,
            interrupt_llm_with_interjections=False,
            max_consecutive_failures=1,
            timeout=300,
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
            # Mirror as synthetic helper tool_call (no LLM step)
            try:
                await self._queue.put(
                    {
                        "_mirror": {
                            "method": "ask",
                            "kwargs": {
                                "question": question,
                                "images": images,
                                **(kwargs or {}),
                            },
                        },
                    },
                )
            except Exception:
                pass
            return helper_handle

        async def _wrap():
            answer = await helper_handle.result()
            self._append_user_visible_assistant(answer)
            return answer, inspection_client.messages

        helper_handle.result = _wrap  # type: ignore[attr-defined]
        # Mirror as synthetic helper tool_call (no LLM step)
        try:
            await self._queue.put(
                {
                    "_mirror": {
                        "method": "ask",
                        "kwargs": {
                            "question": question,
                            "images": images,
                            **(kwargs or {}),
                        },
                    },
                },
            )
        except Exception:
            pass
        return helper_handle

    # -- public API -----------------------------------------------------------
    @functools.wraps(SteerableToolHandle.interject, updated=())
    async def interject(
        self,
        message: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
        images: list | None = None,
        trigger_immediate_llm_turn: bool = True,
        **kwargs,
    ) -> None:
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.debug(f"💬 [{_label}] Interject requested: {message}")
        # Record user-visible immediately
        self._append_user_visible_user(message, parent_chat_context_cont)

        # Buffer then forward to resolver loop. Support dict payloads when continued context provided.
        payload = {
            "message": message,
            "parent_chat_context_continuted": parent_chat_context_cont,
            "images": images,
            "trigger_immediate_llm_turn": trigger_immediate_llm_turn,
        }
        # Use put_nowait to ensure the interjection is registered *synchronously* before
        # we yield control (e.g. in _record_and_forward). This prevents a race where
        # a fast-running loop completes its turn and exits before seeing the queued item.
        self._queue.put_nowait(payload)

        # Centralized steering: record steer event; functional forwarding happens via mirror path
        await self._record_and_forward(
            "interject",
            kwargs={
                "message": message,
                "parent_chat_context_cont": parent_chat_context_cont,
                "images": images,
                "trigger_immediate_llm_turn": trigger_immediate_llm_turn,
                **(kwargs or {}),
            },
            fallback=("content", "message"),
            had_passthrough=False,
            forwarded_to=[],
        )
        # Also mirror as synthetic helper tool_calls immediately (no LLM step)
        try:
            await self._queue.put(
                {
                    "_mirror": {
                        "method": "interject",
                        "kwargs": {
                            "message": message,
                            "images": images,
                            **(kwargs or {}),
                        },
                    },
                },
            )
        except Exception:
            pass

    @functools.wraps(SteerableToolHandle.stop, updated=())
    def stop(
        self,
        reason: Optional[str] = None,
        *,
        parent_chat_context_cont: list[dict] | None = None,
        **kwargs,
    ) -> None:
        # Idempotent guard: if already stopping, do nothing and DO NOT log again
        if self._cancel_event.is_set():
            return

        # Stop request is logged centrally in the loop via mirror path

        # Record steer event (best-effort). Functional forwarding happens via mirror path.
        try:
            asyncio.create_task(
                self._record_and_forward(
                    "stop",
                    kwargs={"reason": reason, **(kwargs or {})},
                    had_passthrough=False,
                    forwarded_to=[],
                ),
            )
        except Exception:
            pass
        # Ensure the loop is not paused so the inner loop can observe and process the stop immediately
        with suppress(Exception):
            self._pause_event.set()
        # Mirror as synthetic helper tool_call (no LLM step) before signalling cancel/stop
        try:
            self._queue.put_nowait(
                {
                    "_mirror": {
                        "method": "stop",
                        "kwargs": {"reason": reason, **(kwargs or {})},
                    },
                },
            )
        except Exception:
            pass
        # Now signal cancellation and stop for any waiters; inner loop will exit after processing mirror
        with suppress(Exception):
            self._cancel_event.set()
        with suppress(Exception):
            self._stop_event.set()

    @functools.wraps(SteerableToolHandle.pause, updated=())
    async def pause(self, **kwargs) -> None:
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.info(f"⏸️ [{_label}] Pause requested")

        # Auto-pause base tools that are currently running
        with suppress(Exception):
            task_info = getattr(self._task, "task_info", {})
            items = task_info.items() if isinstance(task_info, dict) else []
            for _t, _inf in items:
                h = getattr(_inf, "handle", None)
                if h is None:
                    ev = getattr(_inf, "pause_event", None)
                    if ev is not None and hasattr(ev, "clear"):
                        with suppress(Exception):
                            ev.clear()

        self._pause_event.clear()
        # Record steer event (best-effort, async). Functional forwarding happens via mirror path.
        try:
            await self._record_and_forward(
                "pause",
                kwargs=dict(kwargs or {}),
                had_passthrough=False,
                forwarded_to=[],
            )
        except Exception:
            pass
        # Mirror as synthetic helper tool_call (no LLM step)
        try:
            await self._queue.put(
                {
                    "_mirror": {
                        "method": "pause",
                        "kwargs": dict(kwargs or {}),
                    },
                },
            )
        except Exception:
            pass

    @functools.wraps(SteerableToolHandle.resume, updated=())
    async def resume(self, **kwargs) -> None:
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.info(f"▶️ [{_label}] Resume requested")
        # Auto-resume base tools that were started in paused state while the outer loop was paused
        with suppress(Exception):
            task_info = getattr(self._task, "task_info", {})
            items = task_info.items() if isinstance(task_info, dict) else []
            for _t, _inf in items:
                h = getattr(_inf, "handle", None)
                if h is None:
                    ev = getattr(_inf, "pause_event", None)
                    if ev is not None and hasattr(ev, "set"):
                        with suppress(Exception):
                            ev.set()

        self._pause_event.set()
        # Record steer event (best-effort, async). Functional forwarding happens via mirror path.
        try:
            await self._record_and_forward(
                "resume",
                kwargs=dict(kwargs or {}),
                had_passthrough=False,
                forwarded_to=[],
            )
        except Exception:
            pass
        # Mirror as synthetic helper tool_call (no LLM step)
        try:
            await self._queue.put(
                {
                    "_mirror": {
                        "method": "resume",
                        "kwargs": dict(kwargs or {}),
                    },
                },
            )
        except Exception:
            pass

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
        # Centralized steering: record steer event; functional forwarding happens via mirror path.
        try:
            await self._record_and_forward(
                "clarify",
                kwargs={"call_id": call_id, "answer": answer},
                had_passthrough=False,
            )
        except Exception:
            pass
        # Mirror as synthetic helper tool_call (no LLM step)
        try:
            await self._queue.put(
                {
                    "_mirror": {
                        "method": "clarify",
                        "kwargs": {"call_id": call_id, "answer": answer},
                    },
                },
            )
        except Exception:
            pass

    # --- targeted nested steerability (programmatic, no LLM) -----------------
    async def nested_steer(self, spec: dict) -> dict:
        """Apply a nested steering spec using nested_structure’s vocabulary.

        Programmatic (no LLM) and best‑effort: unknown or missing children are
        ignored and traversal stops naturally when no child is found.

        Spec schema (structure‑aligned; keys optional):
        - steps: list[dict]
            Each step is a dict with:
              - method: str  (e.g., "pause", "resume", "interject", "stop", "ask")
              - args: any    (positional argument or list of args; no aliasing)
              - kwargs: dict (keyword arguments)
        - children: list[dict]
            A list of child node specs mirroring nested_structure nodes:
              - tool: str     Canonicalized "Class.method" (preferred identifier)
              - handle: str   Canonicalized handle chain "Leaf(Parent(...))"
              - steps: list[dict]       Local steps to apply on that child
              - children: list[dict]    Further descendants with the same shape

        Matching is done against live children discovered via loop task_info
        and standardized wrapper discovery, using:
          1) tool equality; else
          2) base(handle) equality, where base(x) = x.split("(", 1)[0].

        Returns a summary dict:
          {
            "applied": [{"path": [...], "method": "..."}],
            "skipped": [{"path": [...], "child": {"tool": "...", "handle": "..."}}],
            "status":  {"<path>": {"self": "none|partial|full", "children": {...}}},
            "conditions_fired": [...]
          }
        See module‑level ``_nested_steer_on`` for full behaviour details.
        """

        return await _nested_steer_on(self, spec)

    async def nested_structure(self) -> dict:
        """Return a minimal nested, read-only structure of live child loops.

        Shape (per node):
        - handle: inheritance chain up to AsyncToolLoopHandle, formatted as
          "Leaf(Parent(...(AsyncToolLoopHandle)))" when applicable; otherwise
          the concrete class name.
        - tool: canonical entrypoint label "Class.method" when available, else class name
        - children: list of the same node shape for live (in-flight) nested handles only

        Notes
        -----
        - Non-steerable or pending base tool calls without an adopted handle are omitted.
        - Completed child handles are omitted.
        - Canonicalization strips leading "Simulated"/"Base" from class names for the "tool" field.
        - The inheritance chain stops at AsyncToolLoopHandle and does not include
          "SteerableToolHandle" or any ancestors above it.
        """
        return await _nested_structure_on(self)

    # --- snapshotting v1: read-only capture (flat only) ---------------------
    def serialize(
        self,
        recursive: bool = False,
    ) -> dict:  # type: ignore[override]
        """Return a v1 snapshot of this handle's current state.

        Behaviour (v1):
        - Proactively cancels the running loop to quiesce any in‑flight LLM/tool work.
        - Builds the snapshot from the current transcript; any previously pending
          tool calls will be re‑scheduled by deserialization via preflight backfill.
        - When ``recursive=False`` (default), nested tool loops are not supported
          and a ``ValueError`` is raised if any are detected.
        - When ``recursive=True``, in‑flight nested child handles are captured into
          top-level ``children`` with inline ``snapshot`` for each child.
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
        # "ContactManager.update->ContactManager.ask(x2ab)")
        raw_label = str(getattr(self, "_log_label", None) or self._loop_id or "")
        cls_name, meth_name = _parse_entrypoint_from_loop_id_label(raw_label)

        # Gather transcript fragments
        msgs = []
        try:
            msgs = list(getattr(self._client, "messages", []) or [])
        except Exception:
            msgs = []

        # Minimal pruning policy: no inline-tool binding; keep all assistant tool_calls and final tool results.
        allowed_tool_names: set[str] | None = None

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
        # Mirror pending interjection steering entries from steer_log which may not be in msgs yet.
        # Represent them as user messages appended to the end with monotonically increasing indices.
        # (Changed from system messages for Claude/Gemini compatibility.)
        try:
            steer_log = list(getattr(self, "_steer_log", []) or [])
        except Exception:
            steer_log = []
        # Build a set of existing interjection contents for simple de-duplication
        existing_contents = set()
        try:
            for im in interjections:
                c = im.get("content")
                if isinstance(c, str):
                    existing_contents.add(c)
        except Exception:
            existing_contents = set()
        base_idx = len(msgs)
        appended = 0
        for rec in steer_log:
            try:
                if rec.get("method") != "interject":
                    continue
                kw = rec.get("kwargs") or {}
                content = kw.get("message")
                if not isinstance(content, str) or not content:
                    # positional fallback: args[0] when present
                    args = rec.get("args") or ()
                    if isinstance(args, (list, tuple)) and args:
                        content = args[0]
                if not isinstance(content, str) or not content:
                    continue
                if content in existing_contents:
                    continue
                interjections.append({"role": "user", "content": content})
                interjections_indices.append(base_idx + appended)
                existing_contents.add(content)
                appended += 1
            except Exception:
                continue
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

        # Human-readable root summary from entrypoint (no additional parsing)
        root_summary = {}
        try:
            root_summary["tool"] = f"{cls_name}.{meth_name}"
            root_summary["handle"] = "AsyncToolLoopHandle"
        except Exception:
            root_summary = {}

        snap = _LoopSnapshot(
            loop_id=str(self._loop_id or ""),
            system_message=system_message,
            root=root_summary or None,
            initial_user_message=initial_user_message,
            assistant=assistant_steps,
            tools=tool_results,
            assistant_positions=assistant_indices_raw,
            tool_positions=tool_results_indices,
            system_interjections=interjections,
            interjection_positions=interjections_indices,
            clarifications=clarifications,
            notifications=notifications,
            images=images_list,
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
        # If recursive capture was requested, attach a children manifest at the top-level.
        if recursive and isinstance(task_info, dict):
            children: list[dict] = []
            # Children via task_info (adopted handles)
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
                        child_snapshot = child.serialize(recursive=True)
                    except Exception as e:
                        # Fallback: if we cannot build an inline snapshot for a live child,
                        # record it as completed (state='done') to satisfy schema requirements.
                        state = "done"
                        child_snapshot = None

                # Identify child tool and handle for readability
                def _canon_class(name: str) -> str:
                    """Canonicalize a class-name string by stripping known prefixes.

                    Rules (applied in order):
                    - Strip leading 'Simulated' (e.g., SimulatedFoo → Foo)
                    - Strip leading version prefix 'V<digits>' (e.g., V3Foo → Foo)
                    - Strip leading 'Base' (e.g., BaseFoo → Foo)
                    """
                    s = str(name or "")
                    try:
                        if s.startswith("Simulated") and len(s) > 9:
                            s = s[9:]
                    except Exception:
                        pass
                    try:
                        import re as _re  # noqa: WPS433

                        s = _re.sub(r"^V\\d+", "", s)
                    except Exception:
                        pass
                    try:
                        if s.startswith("Base") and len(s) > 4:
                            s = s[4:]
                    except Exception:
                        pass
                    return s

                def _child_tool(h) -> str | None:
                    try:
                        raw = getattr(h, "_loop_id", None) or ""
                    except Exception:
                        raw = ""
                    base = str(raw).split("(", 1)[0]
                    if "." in base:
                        cls, meth = base.split(".", 1)
                        return f"{_canon_class(cls)}.{meth}"
                    try:
                        cls_name = _canon_class(
                            getattr(getattr(h, "__class__", object), "__name__", ""),
                        )
                        return cls_name or None
                    except Exception:
                        return None

                def _child_handle_chain(h) -> str | None:
                    try:
                        cls = getattr(h, "__class__", object)
                    except Exception:
                        cls = object
                    try:
                        leaf = _canon_handle_name(cls) or "handle"
                    except Exception:
                        leaf = "handle"
                    _SENTINELS = (
                        (AsyncToolLoopHandle, "AsyncToolLoopHandle"),
                        (SteerableToolHandle, "SteerableToolHandle"),
                        (SteerableHandle, "SteerableHandle"),
                    )
                    for typ, label in _SENTINELS:
                        if cls is typ:
                            return label
                    try:
                        mro = list(getattr(cls, "__mro__", ()))
                    except Exception:
                        mro = []
                    parts = [leaf]
                    for base in (mro[1:] if len(mro) > 1 else []):
                        try:
                            bname = getattr(base, "__name__", "")
                        except Exception:
                            bname = ""
                        included = False
                        for typ, label in _SENTINELS:
                            if base is typ:
                                parts.append(label)
                                included = True
                                break
                        if included:
                            break
                        from abc import ABC as _ABC  # noqa: WPS433

                        if base is object or base is _ABC:
                            break
                        if bname and bname.startswith("Base"):
                            continue
                        try:
                            canon = _canon_handle_name(base)
                        except Exception:
                            canon = bname
                        if canon:
                            parts.append(canon)
                    try:
                        s = parts[-1]
                        for p in reversed(parts[:-1]):
                            s = f"{p}({s})"
                        return s
                    except Exception:
                        return leaf

                entry = {
                    "call_id": getattr(_inf, "call_id", None),
                    "tool": _child_tool(child) or getattr(_inf, "name", None),
                    "handle": _child_handle_chain(child),
                    "passthrough": bool(getattr(_inf, "is_passthrough", False)),
                    "state": state,
                }
                if isinstance(child_snapshot, dict):
                    entry["snapshot"] = child_snapshot
                children.append(entry)
            # Children via standardized wrapper discovery
            try:
                from .handle_wrappers import (  # noqa: WPS433
                    discover_wrapped_handles as _discover_wrapped_handles,
                )
            except Exception:
                _discover_wrapped_handles = None  # type: ignore
            if _discover_wrapped_handles is not None:
                try:
                    pairs = list(_discover_wrapped_handles(self) or [])
                except Exception:
                    pairs = []
                for _src, child in pairs:
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
                            child_snapshot = child.serialize(recursive=True)
                        except Exception as e:
                            state = "done"
                            child_snapshot = None
                    entry = {
                        "call_id": None,
                        "tool": _child_tool(child),
                        "handle": _child_handle_chain(child),
                        "passthrough": False,
                        "state": state,
                    }
                    if isinstance(child_snapshot, dict):
                        entry["snapshot"] = child_snapshot
                    children.append(entry)
            try:
                if children:
                    snap["children"] = children
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
                # Request stop and attach a generic after-first-LLM banner for readability
                self.stop(
                    reason="serialize snapshot",
                    _after_first_llm_banner={
                        "text": "Serialization complete",
                        "prefix": "📦",
                    },
                )
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
        parent_lineage: list[str] | None = None,
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

        from .llm_client import (
            new_llm_client as _new_llm_client,
        )  # noqa: WPS433

        # Build tools mapping depending on resolved entrypoint
        tools: Dict[str, Callable] = {}
        loop_label: str = snap.loop_id or ""

        # Resolve manager class by name from the central registry
        from .state_managers import (  # noqa: WPS433
            discover_manager_modules as _discover_manager_modules,
            get_manager_registry as _get_manager_registry,
        )

        _discover_manager_modules()
        _registry = _get_manager_registry()

        # Resolve entrypoint: prefer loop_id lineage label; else fall back to root.tool
        try:
            if loop_label:
                ep_class_name, ep_method_name = _parse_entrypoint_from_loop_id_label(
                    loop_label,
                )
            else:
                root = snap.root or {}
                tool_val = root.get("tool") if isinstance(root, dict) else None
                if not isinstance(tool_val, str) or "." not in tool_val:
                    raise ValueError(
                        "Manager class not found: missing loop_id and root.tool",
                    )
                ep_class_name, ep_method_name = tool_val.split(".", 1)
        except Exception as _exc:
            # Ensure consistent error message for missing manager
            raise ValueError(
                "Manager class not found: unable to derive entrypoint",
            ) from _exc

        mgr_cls = _registry.get(ep_class_name)
        if mgr_cls is None:
            raise ValueError(
                f"Manager class not found: {ep_class_name}",
            )
        manager = mgr_cls()
        method_name = ep_method_name
        tools = dict(manager.get_tools(method_name, include_sub_tools=True))
        if not tools:
            raise ValueError(
                f"No tools registered for {ep_class_name}.{method_name}",
            )
        if not loop_label:
            loop_label = f"{ep_class_name}.{method_name}"

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
            for am in snap.assistant or []:
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
            for tm in snap.tools or []:
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
            snap.assistant_positions
            or snap.tool_positions
            or snap.interjection_positions
        ):
            combined: list[tuple[int, dict]] = []
            # Assistant messages with indices
            try:
                for idx_val, amsg in zip(
                    snap.assistant_positions or [],
                    snap.assistant or [],
                ):
                    if isinstance(amsg, dict) and amsg.get("role") == "assistant":
                        combined.append((int(idx_val), amsg))
            except Exception:
                pass
            # Tool results with indices (skip clarification wrappers and any pending placeholders)
            try:
                for idx_val, tmsg in zip(
                    snap.tool_positions or [],
                    snap.tools or [],
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
            # Interjections with indices (supports both user and system roles for
            # backwards compatibility - new format uses user, old format used system)
            try:
                for idx_val, imsg in zip(
                    snap.interjection_positions or [],
                    snap.system_interjections or [],
                ):
                    if isinstance(imsg, dict) and imsg.get("role") in (
                        "user",
                        "system",
                    ):
                        combined.append((int(idx_val), imsg))
            except Exception:
                pass

            # Sort by original index and append in order
            for _, m in sorted(combined, key=lambda x: x[0]):
                msgs.append(m)
        else:
            # Backward-compat path: pair tool results by call_id after each assistant
            by_call_id: dict[str, dict] = {}
            for tm in snap.tools or []:
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

            for amsg in snap.assistant or []:
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
            ch_list = getattr(snap, "children", None)
            if isinstance(ch_list, list):
                for rec in ch_list:
                    try:
                        if not isinstance(rec, dict):
                            continue
                        if rec.get("state") != "in_flight":
                            continue
                        child_snap = rec.get("snapshot")
                        if not isinstance(child_snap, dict):
                            continue  # by-ref children not supported in simplified v1
                        # Start child under the parent's entrypoint lineage so its logs reflect nested replay
                        parent_entrypoint = f"{ep_class_name}.{ep_method_name}"
                        child_handle = cls.deserialize(
                            child_snap,
                            parent_lineage=[parent_entrypoint],
                        )
                        _resume_children_payload.append(
                            {
                                "call_id": rec.get("call_id"),
                                "tool_name": rec.get("tool"),
                                "is_passthrough": bool(
                                    rec.get("passthrough", False),
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
            parent_lineage=(
                parent_lineage
                if parent_lineage is not None
                else TOOL_LOOP_LINEAGE.get([])
            ),
            images=images_param,
            resume_children=_resume_children_payload or None,
            replay_origin="deserialize",
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
            ai = snap.assistant_positions or []
            ii = snap.interjection_positions or []
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
async def _nested_steer_on(handle: Any, spec: dict) -> dict:
    """Apply a nested steering spec using the same vocabulary as nested_structure.

    Spec schema (structure-aligned; keys optional):
      - steps: list[dict]
          Ordered actions to apply on the current handle. Each step supports:
            - method: str  (e.g., "pause", "resume", "interject", "stop", "ask")
            - args: any    (positional argument or list of args; no aliasing)
            - kwargs: dict (keyword arguments)
      - children: list[dict]
          A list of child node specs. Each child node mirrors a node returned by
          nested_structure and may include:
            - tool: str     Canonicalized "Class.method" (preferred identifier)
            - handle: str   Canonicalized handle chain "Leaf(Parent(...))"
            - steps: list[dict]   Local steps to apply on that child
            - children: list[dict]  Further descendants with the same shape
          Matching is done against live children discovered at this node using:
            1) tool equality (preferred), else
            2) base(handle) equality, where base(x) = x.split(\"(\", 1)[0]

    Behaviour:
      - Apply local steps first on the current handle.
      - Discover live children via both loop task_info and standardized wrappers
        (discover_wrapped_handles), deduplicating by object id and skipping
        completed handles.
      - For each spec child, match one or more live children and recurse.
      - Record a results object with:
          results = {
            "applied": [ {"path": [...], "method": "..."} ],
            "skipped": [ {"path": [...], "child": {"tool": "...", "handle": "..."}} ],
            "status":  {
              "<path>": {"self": "none|partial|full", "children": {<child_id>: "..." }}
            },
            "conditions_fired": [ ... ]  # when conditions are provided
          }

    Conditions (optional, structure-aligned):
      - Node may include "conditions": list[dict]. Each condition has:
          {
            "when": { ... boolean expression ... },
            "then":  [steps...],
            "else_then": [steps...]
          }
        Boolean expression supports:
          - {"self": "none|partial|full"}
          - {"child": {"tool": "..."} , "status": "none|partial|full"}
          - {"child": {"handle": "..."} , "status": "none|partial|full"}
          - {"path": "A.B.C", "status": "none|partial|full"}    # dotted tool-ids
          - {"any": [expr, ...]}  / {"all": [expr, ...]}  / {"not": expr}

    Notes:
      - This refactor unifies naming and discovery with nested_structure:
        canonicalized names, wrapper traversal, and identical child vocabulary.
      - Selector/call_id matching from the previous design is removed in favour
        of structure-based matching by "tool"/"handle".
    """  # noqa: E501

    # Best-effort label for diagnostics
    try:
        label = (
            getattr(handle, "_log_label", None)
            or getattr(handle, "_loop_id", None)
            or getattr(getattr(handle, "__class__", object), "__name__", "handle")
        )
    except Exception:
        label = "handle"

    # Helper: derive a loop-style label for any target handle (child or current)
    def _label_of(h) -> str:
        try:
            return (
                getattr(h, "_log_label", None)
                or getattr(h, "_loop_id", None)
                or getattr(getattr(h, "__class__", object), "__name__", "handle")
            )
        except Exception:
            return "handle"

    # Helper: emit a pre-call log that mirrors native handle logs but marks nested_steer origin
    def _log_pre_steer(h, method: str, args, kwargs: dict) -> None:
        try:
            target_label = _label_of(h)
            m = (method or "").strip().lower()
            # Extract a short text payload for well-known methods (best-effort)
            txt = None
            try:
                if isinstance(args, (list, tuple)) and args:
                    txt = args[0]
            except Exception:
                txt = None
            if not isinstance(txt, str) or not txt:
                # Try common kwargs for message/question/reason/content
                for k in ("message", "content", "question", "reason", "text"):
                    try:
                        v = kwargs.get(k)
                        if isinstance(v, str) and v:
                            txt = v
                            break
                    except Exception:
                        continue
            # Choose icon and level to mirror direct logs
            if m == "pause":
                LOGGER.info(f"⏸️ [{target_label}] Pause requested – via nested_steer")
            elif m == "resume":
                LOGGER.info(f"▶️ [{target_label}] Resume requested – via nested_steer")
            elif m == "stop":
                suffix = f" – reason: {txt}" if isinstance(txt, str) and txt else ""
                LOGGER.info(
                    f"🛑 [{target_label}] Stop requested – via nested_steer{suffix}",
                )
            elif m == "interject":
                suffix = f": {txt}" if isinstance(txt, str) and txt else ""
                LOGGER.debug(
                    f"💬 [{target_label}] Interject requested{suffix} – via nested_steer",
                )
            elif m == "ask":
                suffix = f": {txt}" if isinstance(txt, str) and txt else ""
                LOGGER.info(
                    f"❓ [{target_label}] Ask requested{suffix} – via nested_steer",
                )
            else:
                # Generic steering step
                suffix = f": {txt}" if isinstance(txt, str) and txt else ""
                LOGGER.info(
                    f"🎯 [{target_label}] {method} requested{suffix} – via nested_steer",
                )
        except Exception:
            # Never let logging failures affect control flow
            pass

    try:
        steps_count = 0
        children_count = 0
        try:
            if isinstance(spec, dict):
                steps = spec.get("steps") or []
                if isinstance(steps, list):
                    steps_count = len(steps)
                children = spec.get("children")
                if isinstance(children, list):
                    children_count = len(children)
        except Exception:
            steps_count, children_count = 0, 0
        LOGGER.info(
            f"🎯 [{label}] Nested steer requested – steps={steps_count} children={children_count}",
        )
    except Exception:
        pass

    results: dict = {"applied": [], "skipped": []}
    outer_handle = handle

    # ───── shared canonicalization helpers (mirror nested_structure) ─────────
    def _canon_name(name: str) -> str:
        s = str(name or "")
        if s.startswith("Simulated") and len(s) > 9:
            s = s[9:]
        # Strip leading version prefixes like V3/V12
        try:
            import re as _re  # noqa: WPS433

            s = _re.sub(r"^V\\d+", "", s)
        except Exception:
            pass
        if s.startswith("Base") and len(s) > 4:
            s = s[4:]
        return s

    def _handle_chain_of(h) -> str:
        try:
            cls = getattr(h, "__class__", object)
        except Exception:
            cls = object
        try:
            leaf_name = _canon_handle_name(cls) or "handle"
        except Exception:
            leaf_name = "handle"

        _SENTINELS = (
            (AsyncToolLoopHandle, "AsyncToolLoopHandle"),
            (SteerableToolHandle, "SteerableToolHandle"),
            (SteerableHandle, "SteerableHandle"),
        )
        for typ, label in _SENTINELS:
            if cls is typ:
                return label

        try:
            mro = list(getattr(cls, "__mro__", ()))
        except Exception:
            mro = []

        parts: list[str] = [leaf_name]
        for base in (mro[1:] if len(mro) > 1 else []):
            try:
                bname = getattr(base, "__name__", "")
            except Exception:
                bname = ""

            included_sentinel = False
            for typ, label in _SENTINELS:
                if base is typ:
                    parts.append(label)
                    included_sentinel = True
                    break
            if included_sentinel:
                break

            from abc import ABC as _ABC  # noqa: WPS433

            if base is object or base is _ABC:
                break

            if bname and bname.startswith("Base"):
                continue
            try:
                canon = _canon_handle_name(base)
            except Exception:
                canon = bname
            if canon:
                parts.append(canon)

        try:
            s = parts[-1]
            for p in reversed(parts[:-1]):
                s = f"{p}({s})"
            return s
        except Exception:
            return leaf_name

    def _tool_of(h) -> str | None:
        try:
            raw = getattr(h, "_loop_id", None) or ""
        except Exception:
            raw = ""
        base = str(raw).split("(", 1)[0]
        if "." in base:
            cls, meth = base.split(".", 1)
            return f"{_canon_name(cls)}.{meth}"
        try:
            cls_name = _canon_name(
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
        return True

    def _base_handle_name(s: str | None) -> str:
        try:
            return str(s or "").split("(", 1)[0]
        except Exception:
            return str(s or "")

    def _identity_for_path(h) -> str:
        t = _tool_of(h)
        if isinstance(t, str) and t:
            return t
        return _base_handle_name(_handle_chain_of(h))

    def _discover_children(h) -> list:
        # task_info discovery
        task_info = {}
        from contextlib import suppress as _s

        with _s(Exception):
            task_info = getattr(getattr(h, "_task", None), "task_info", {}) or {}

        children: list = []
        seen: set[int] = set()
        if isinstance(task_info, dict) and task_info:
            for meta in list(task_info.values()):
                try:
                    child = getattr(meta, "handle", None)
                except Exception:
                    child = None
                if child is None or not _is_live(child):
                    continue
                try:
                    cid = id(child)
                    if cid in seen:
                        continue
                    seen.add(cid)
                except Exception:
                    pass
                children.append(child)

        # wrapper discovery (same helper as nested_structure)
        try:
            from .handle_wrappers import (  # noqa: WPS433
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
                    if cid in seen:
                        continue
                    seen.add(cid)
                except Exception:
                    pass
                children.append(child)

        return children

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
            children_view = {}
            try:
                for k, v in (node_status.get("children", {}) or {}).items():
                    if isinstance(v, dict):
                        children_view[k] = v.get("self", "none")
            except Exception:
                children_view = {}
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
        # Fast path: allow a direct child key match for full literal (handles tool ids like "Class.method")
        try:
            direct = cur.get("children", {}).get(dotted)
            if isinstance(direct, dict):
                return str(direct.get("self", "none"))
        except Exception:
            pass
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
    ) -> dict:
        outer_self = outer_handle
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
                        # Emit a pre-call log with the target's own loop label so hierarchy is preserved
                        _log_pre_steer(h, method, call_args, call_kwargs)
                        await forward_handle_call(
                            h,
                            method,
                            call_kwargs,
                            call_args=call_args,
                            fallback_positional_keys=(),
                        )
                        # NEW: Synthesize helper tool_calls in the OUTER transcript for child steps,
                        #      without triggering an immediate LLM turn and without re-executing the child.
                        try:
                            if h is not outer_self:
                                # Resolve child identity from outer task_info
                                target_call_id = None
                                try:
                                    ti = getattr(outer_self._task, "task_info", {})  # type: ignore[attr-defined]
                                    if isinstance(ti, dict):
                                        for _meta in ti.values():
                                            if getattr(_meta, "handle", None) is h:
                                                target_call_id = getattr(
                                                    _meta,
                                                    "call_id",
                                                    None,
                                                )
                                                try:
                                                    target_tool_name = getattr(
                                                        _meta,
                                                        "name",
                                                        None,
                                                    )
                                                except Exception:
                                                    target_tool_name = None
                                                break
                                except Exception:
                                    target_call_id = None
                                # Build kwargs for helper readability (e.g., interject content)
                                mirror_kwargs = dict(call_kwargs or {})
                                if (
                                    isinstance(method, str)
                                    and method.lower().strip() == "interject"
                                    and "content" not in mirror_kwargs
                                    and "message" not in mirror_kwargs
                                ):
                                    if (
                                        isinstance(call_args, (list, tuple))
                                        and call_args
                                    ):
                                        mirror_kwargs["content"] = call_args[0]
                                # Provide a stable helper label for transcript when the target is absent
                                helper_label = None
                                try:
                                    if target_tool_name:
                                        helper_label = str(target_tool_name)
                                except Exception:
                                    helper_label = None
                                if not helper_label:
                                    try:
                                        # Fall back to canonicalized 'tool' or handle chain
                                        lbl = _tool_of(h)
                                        if not lbl:
                                            lbl = _handle_chain_of(h)
                                        helper_label = lbl
                                    except Exception:
                                        helper_label = None
                                if helper_label:
                                    mirror_kwargs["helper_label"] = helper_label
                                # Inject-only (no second execution)
                                mirror_kwargs["_inject_only"] = True
                                # Enqueue mirror sentinel with explicit policy: no LLM turn on outer
                                await outer_self._queue.put(
                                    {
                                        "_mirror": {
                                            "method": str(method or ""),
                                            "kwargs": mirror_kwargs,
                                        },
                                        "_llm_turn": "none",
                                    },
                                )
                        except Exception:
                            pass
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

        # 2) Recurse into matched children (structure-based)
        # Children specs – strict list[dict] only
        children_raw = node.get("children") or []
        children_specs = children_raw if isinstance(children_raw, list) else []

        live_children = _discover_children(h)
        per_child_nodes: dict[str, list[dict]] = {}

        for child_spec in children_specs:
            if not isinstance(child_spec, dict):
                continue
            target_tool = child_spec.get("tool")
            target_handle = _base_handle_name(child_spec.get("handle"))

            matched = False
            matched_nodes: list[dict] = []
            for child in live_children:
                live_tool = _tool_of(child)
                live_handle = _handle_chain_of(child)
                live_handle_base = _base_handle_name(live_handle)

                is_match = False
                if isinstance(target_tool, str) and target_tool:
                    # Exact tool match only
                    is_match = live_tool == target_tool
                elif isinstance(target_handle, str) and target_handle:
                    is_match = live_handle_base == target_handle

                if is_match:
                    matched = True
                    child_id = target_tool or target_handle or _identity_for_path(child)
                    try:
                        _p = "/".join(str(p) for p in path)
                        LOGGER.debug(
                            f"↘️ [{label}] Descend: matched child at {_p} → "
                            f"{child_id!r}",
                        )
                    except Exception:
                        pass
                    node_status = await _apply(
                        child,
                        child_spec,
                        path + [child_id],
                    )
                    per_child_nodes.setdefault(child_id, []).append(node_status)
                    matched_nodes.append(node_status)

            if not matched:
                # No live child matched this spec entry
                ident = {}
                if isinstance(target_tool, str) and target_tool:
                    ident["tool"] = target_tool
                if isinstance(target_handle, str) and target_handle:
                    ident["handle"] = target_handle
                try:
                    results["skipped"].append(
                        {"path": list(path), "child": ident or {"unknown": True}},
                    )
                except Exception:
                    pass

        # Aggregate child statuses by identity key
        aggregated_children: dict[str, dict] = {}
        for k, lst in per_child_nodes.items():
            aggregated_children[k] = _aggregate_nodes(lst)
        # Ensure explicitly specified children exist (even if no live match)
        for child_spec in children_specs:
            try:
                ident = None
                if isinstance(child_spec, dict):
                    t = child_spec.get("tool")
                    h = child_spec.get("handle")
                    if isinstance(t, str) and t:
                        ident = t
                    elif isinstance(h, str) and h:
                        ident = _base_handle_name(h)
                if (
                    isinstance(ident, str)
                    and ident
                    and ident not in aggregated_children
                ):
                    aggregated_children[ident] = _empty_status_node()
            except Exception:
                continue

        # Compute self status
        if children_specs:
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
                if "child" in expr:
                    child_expr = expr.get("child") or {}
                    ident = None
                    if isinstance(child_expr, dict):
                        t = child_expr.get("tool")
                        h = child_expr.get("handle")
                        if isinstance(t, str) and t:
                            ident = t
                        elif isinstance(h, str) and h:
                            ident = _base_handle_name(h)
                    if not ident:
                        return False
                    want = str(expr.get("status", "none"))
                    got = str(
                        (node_status.get("children", {}).get(ident) or {}).get(
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

    await _apply(handle, spec or {}, [str(label)])
    return results


# --- module-level nested structure introspection (read-only) -------------------
async def _nested_structure_on(
    handle: Any,
    *,
    max_depth: Optional[int] = None,
) -> dict:
    """Return a minimal nested structure for any compatible handle.

    Each node contains:
      - handle: canonicalized inheritance chain up to the first sentinel
        (AsyncToolLoopHandle / SteerableToolHandle / SteerableHandle), rendered
        as A(B(C)). Class segments are canonicalized by stripping Simulated,
        V<digits>, and Base prefixes; Base* parents are elided entirely.
      - tool: canonicalized "Class.method" when available (same prefix rules),
        else the canonicalized class name.
      - children: only live, steerable nested handles (pending/done omitted)
    """

    def _canon_name(name: str) -> str:
        s = str(name or "")
        if s.startswith("Simulated") and len(s) > 9:
            s = s[9:]
        # Strip leading version prefixes like V3/V12
        try:
            import re as _re  # local import

            s = _re.sub(r"^V\\d+", "", s)
        except Exception:
            pass
        if s.startswith("Base") and len(s) > 4:
            s = s[4:]
        return s

    def _handle_chain_of(h) -> str:
        """Return the handle name with parent chain nested like A(B(C)).

        Traverses base classes upward and nests the names until reaching one of
        the sentinels: AsyncToolLoopHandle, SteerableToolHandle, or
        SteerableHandle. The sentinel encountered is included once and then
        traversal stops. If the leaf class itself is a sentinel, the result is
        just that class name.
        """
        try:
            cls = getattr(h, "__class__", object)
        except Exception:
            cls = object
        try:
            leaf_name = _canon_handle_name(cls) or "handle"
        except Exception:
            leaf_name = "handle"

        # If the leaf itself is a sentinel, return it directly
        _SENTINELS = (
            (AsyncToolLoopHandle, "AsyncToolLoopHandle"),
            (SteerableToolHandle, "SteerableToolHandle"),
            (SteerableHandle, "SteerableHandle"),
        )
        for typ, label in _SENTINELS:
            if cls is typ:
                return label

        try:
            mro = list(getattr(cls, "__mro__", ()))
        except Exception:
            mro = []

        parts: list[str] = [leaf_name]
        # Walk parents (skip the leaf itself)
        for base in (mro[1:] if len(mro) > 1 else []):
            try:
                bname = getattr(base, "__name__", "")
            except Exception:
                bname = ""

            # Include the first sentinel encountered and then stop
            included_sentinel = False
            for typ, label in _SENTINELS:
                if base is typ:
                    parts.append(label)
                    included_sentinel = True
                    break
            if included_sentinel:
                break

            # Skip Python/ABC/object sentinels
            from abc import ABC as _ABC  # local to avoid top import confusion

            if base is object or base is _ABC:
                break

            # Skip Base* classes entirely; include other intermediates canonicalized
            if bname and bname.startswith("Base"):
                continue
            try:
                canon = _canon_handle_name(base)
            except Exception:
                canon = bname
            if canon:
                parts.append(canon)

        # Compose nested parentheses A(B(C)) from the top down
        try:
            s = parts[-1]
            for p in reversed(parts[:-1]):
                s = f"{p}({s})"
            return s
        except Exception:
            return leaf_name

    def _tool_of(h) -> str | None:
        # Prefer stable loop_id set by starters (e.g., "ContactManager.ask")
        try:
            raw = getattr(h, "_loop_id", None) or ""
        except Exception:
            raw = ""
        base = str(raw).split("(", 1)[0]
        if "." in base:
            cls, meth = base.split(".", 1)
            return f"{_canon_name(cls)}.{meth}"
        # Fallback to canonicalized class name
        try:
            cls_name = _canon_name(
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
            _t = _tool_of(h)
            node_cycle: dict = {
                "handle": getattr(h, "__class__", object).__name__,
                "children": [],
            }
            if isinstance(_t, str) and "." in _t:
                node_cycle["tool"] = _t
            return node_cycle
        if hid is not None:
            visited.add(hid)

        node: dict = {
            "handle": _handle_chain_of(h),
            "children": [],
        }
        _t = _tool_of(h)
        if isinstance(_t, str) and "." in _t:
            node["tool"] = _t

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
    client: unillm.AsyncUnify,
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
    caller_description: Optional[str] = None,
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
    replay_origin: Optional[str] = None,
    persist: bool = False,
    multi_handle: bool = False,
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

    timeout : int | None, default 300
        Activity-based timeout in seconds. The timer resets after each
        observable event (LLM response, tool completion, interjection).
        This timeout guards against hung user-defined tools, NOT slow LLM
        inference. LLM providers have their own timeout mechanisms; if an
        LLM call is in-flight, the loop will wait for it to complete before
        checking the timeout. When ``None``, no timeout is enforced.

    raise_on_limit : bool, default False
        If ``True``, raises ``asyncio.TimeoutError`` or ``RuntimeError``
        when the timeout or max_steps limit is exceeded. If ``False``,
        the loop terminates gracefully with a summary message.

    persist : bool, default False
        If ``True``, the loop does not terminate when the LLM produces content
        without tool calls. Instead, it blocks waiting for the next interjection
        via ``handle.interject()``. When an interjection arrives, the LLM is
        granted another turn. This enables a single persistent loop that can
        process multiple events over time. The loop only terminates when
        explicitly stopped via ``handle.stop()`` or cancelled.

    multi_handle : bool, default False
        If ``True``, enables multi-handle mode where the loop can serve multiple
        concurrent requests. Each request is identified by a request_id, and the
        LLM uses ``final_answer(request_id, answer)`` to complete specific requests.
        The returned handle supports ``add_request(message)`` to add new requests
        to the running loop. Interjections are tagged with request IDs so the LLM
        knows which request they belong to. The loop terminates when all requests
        are completed/cancelled (unless persist=True).
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

    # --- multi-handle mode setup -------------------------------------------
    # Create the coordinator if multi_handle is enabled
    multi_handle_coordinator: MultiHandleCoordinator | None = None
    if multi_handle:
        # We need to reference clarification_channels which is set on the task later
        # Use a placeholder dict that will be updated when the task starts
        _clarification_channels_ref: dict = {}
        multi_handle_coordinator = MultiHandleCoordinator(
            interject_queue=interject_queue,
            clarification_channels=_clarification_channels_ref,
            persist=persist,
        )
        # Register the first request (request_id=0)
        multi_handle_coordinator.register_request()

    # Run the async tool loop

    async def _loop_wrapper():
        try:
            return await async_tool_loop_inner(
                client,
                (
                    message
                    if not multi_handle
                    else tag_message_with_request(
                        message if isinstance(message, str) else str(message),
                        0,
                    )
                ),
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
                caller_description=caller_description,
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
                replay_origin=replay_origin,
                persist=persist,
                multi_handle_coordinator=multi_handle_coordinator,
            )
        except asyncio.CancelledError:
            raise

    task = asyncio.create_task(_loop_wrapper(), name="ToolUseLoop")

    # Make introspection surfaces available immediately on the wrapper task.
    # The inner loop rebinding will point these to the live dicts once running.
    try:  # pragma: no cover
        setattr(task, "task_info", {})  # asyncio.Task -> ToolCallMetadata
        setattr(task, "clarification_channels", {})  # call_id -> (up_q, down_q)
    except Exception:
        pass

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

    # Prefer explicit entrypoint metadata over label parsing during serialize().
    # If a semantic cache namespace like "Class.method" is provided by callers
    # (manager methods do this), record a structured manager entrypoint.
    try:
        _ep: dict | None = None
        if (
            isinstance(semantic_cache_namespace, str)
            and "." in semantic_cache_namespace
        ):
            cls, meth = semantic_cache_namespace.split(".", 1)
            if cls and meth:
                _ep = {"class_name": cls, "method_name": meth}
        if _ep is not None:
            setattr(handle, "_entrypoint_info", _ep)
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

    # --- multi-handle mode: return a MultiRequestHandle for request 0 ---
    if multi_handle and multi_handle_coordinator is not None:
        # Store the underlying handle reference for potential introspection
        multi_handle_coordinator._underlying_handle = handle  # type: ignore[attr-defined]

        # Update the clarification channels reference once the task has it
        # This is a bit of a hack but necessary since the task attr is set after creation
        try:
            multi_handle_coordinator._clarification_channels = getattr(
                task,
                "clarification_channels",
                {},
            )
        except Exception:
            pass

        # Create and return the request handle for request_id=0
        request_handle = MultiRequestHandle(
            request_id=0,
            coordinator=multi_handle_coordinator,
            loop_id=loop_id,
        )

        # Store handle reference in registry
        state = multi_handle_coordinator.registry.get(0)
        if state:
            state.handle_ref = request_handle

        return request_handle  # type: ignore[return-value]

    return handle


# ─────────────────────────────────────────────────────────────────────────────
# Custom steering decorator
# ─────────────────────────────────────────────────────────────────────────────
def custom_steering_method(
    *,
    aliases: Iterable[str] | None = None,
    fallback: Iterable[str] | None = None,
):
    """
    Decorator for custom public steering methods defined on classes derived from
    AsyncToolLoopHandle.

    Behaviour
    ---------
    - Executes the original method.
    - Records the steering event into the handle's steer log so adoption can replay.
    - Enqueues a mirror sentinel that the inner loop consumes to:
        • synthesize helper tool_calls/acks, and
        • forward the call to all in‑flight passthrough children that implement it.
    - The mirror payload carries control keys ("_custom", "_aliases", "_fallback")
      used only by the inner loop; these are NOT forwarded to child handles.

    Parameters
    ----------
    aliases : Iterable[str] | None
        Optional alternative method names to try on children during forwarding.
    fallback : Iterable[str] | None
        Optional ordered list of argument keys to use as positional fallbacks
        when a child's method signature does not accept kwargs (passed to
        forward_handle_call as fallback_positional_keys).
    """

    def _decorator(fn):
        is_async = asyncio.iscoroutinefunction(fn)
        alias_list = list(aliases or [])
        fb_list = list(fallback or [])

        async def _post_call(self: "AsyncToolLoopHandle", args, kwargs, result):
            # Record without control keys
            await self._record_and_forward(
                fn.__name__,
                args=list(args or ()),
                kwargs=dict(kwargs or {}),
                fallback=tuple(fb_list),
                had_passthrough=False,
                forwarded_to=[],
            )
            # Mirror to the inner loop with control keys for routing/dispatch
            try:
                await self._queue.put(
                    {
                        "_mirror": {
                            "method": fn.__name__,
                            "kwargs": dict(kwargs or {}),
                            "_custom": True,
                            "_aliases": list(alias_list),
                            "_fallback": list(fb_list),
                        },
                    },
                )
            except Exception:
                pass
            return result

        if is_async:

            @functools.wraps(fn, updated=())
            async def _async_wrapped(self: "AsyncToolLoopHandle", *a, **kw):
                res = await fn(self, *a, **kw)
                return await _post_call(self, a, kw, res)

            return _async_wrapped

        @functools.wraps(fn, updated=())
        def _sync_wrapped(self: "AsyncToolLoopHandle", *a, **kw):
            res = fn(self, *a, **kw)
            # Fire-and-forget record (do not block caller)
            try:
                asyncio.create_task(
                    self._record_and_forward(
                        fn.__name__,
                        args=list(a or ()),
                        kwargs=dict(kw or {}),
                        fallback=tuple(fb_list),
                        had_passthrough=False,
                        forwarded_to=[],
                    ),
                )
            except Exception:
                pass
            # Mirror sentinel nowait
            try:
                self._queue.put_nowait(
                    {
                        "_mirror": {
                            "method": fn.__name__,
                            "kwargs": dict(kw or {}),
                            "_custom": True,
                            "_aliases": list(alias_list),
                            "_fallback": list(fb_list),
                        },
                    },
                )
            except Exception:
                pass
            return res

        return _sync_wrapped

    return _decorator
