import asyncio
import copy
import unillm
import functools
import json
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
)
from ..logger import LOGGER
from unity.common.hierarchical_logger import ICONS
from .llm_helpers import short_id
from ._async_tool.loop_config import TOOL_LOOP_LINEAGE, _PENDING_LOOP_SUFFIX
from ._async_tool.messages import forward_handle_call
from ._async_tool.loop import async_tool_loop_inner
from ._async_tool.propagation_mode import ChatContextPropagation
from ._async_tool.context_compression import (
    _COMPRESSION_SIGNAL,
    compress_messages,
    render_compressed_context,
)
from .context_dump import make_messages_safe_for_context_dump
from typing import Iterable


from ._async_tool.multi_handle import (
    MultiHandleCoordinator,
    MultiRequestHandle,
)
from ._async_tool.tagging import tag_message_with_request

if TYPE_CHECKING:
    from unillm.types import PromptCacheParam


def _transform_inner_roles(messages: list[dict]) -> list[dict]:
    """Transform 'user'/'assistant' roles to 'inner_user'/'inner_assistant'.

    Disambiguates the inspected loop's transcript from the inspection loop's
    own conversation and from the outer parent context (which uses
    'outer_user'/'outer_assistant').
    """
    transformed = []
    for msg in messages:
        new_msg = dict(msg)
        role = new_msg.get("role", "")
        if role == "user":
            new_msg["role"] = "inner_user"
        elif role == "assistant":
            new_msg["role"] = "inner_assistant"
        transformed.append(new_msg)
    return transformed


_PARENT_CTX_POINTER = (
    "## Parent Chat Context\n"
    "[The parent chat context that was available to this loop has been omitted "
    "from this transcript to avoid duplication. Refer to the Parent Chat Context "
    "section in your system context for the full, up-to-date version.]"
)


def _replace_runtime_parent_context(messages: list[dict]) -> list[dict]:
    """Replace runtime parent-context headers with a short pointer.

    When the inspection loop receives fresh parent context via the standard
    machinery, the stale copy embedded in the inspected transcript is redundant.
    This finds any message tagged ``_parent_chat_context=True`` and replaces
    the Parent Chat Context portion of its content with a pointer, preserving
    other sections (e.g. Caller Context) in the same message.
    """
    result = []
    for msg in messages:
        if msg.get("_parent_chat_context"):
            new_msg = dict(msg)
            # The runtime context message may contain multiple sections
            # (e.g. Caller Context + Parent Chat Context).  Replace only the
            # Parent Chat Context portion.
            content = new_msg.get("content") or ""
            pcc_idx = content.find("## Parent Chat Context")
            if pcc_idx >= 0:
                new_msg["content"] = content[:pcc_idx] + _PARENT_CTX_POINTER
            result.append(new_msg)
        else:
            result.append(msg)
    return result


# Tiny handle objects exposed to callers
# ─────────────────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod


class SteerableToolHandle(ABC):
    """Abstract base class for steerable tool handles.

    Defines the full steering surface: query (``ask``, ``interject``),
    lifecycle (``stop``, ``pause``, ``resume``), completion (``done``,
    ``result``), and event APIs (``next_clarification``,
    ``next_notification``, ``answer_clarification``).

    Notes on context parameters
    ---------------------------
    Steering methods accept context parameters that are plumbing parameters
    automatically hidden from LLM tool schemas (injected by orchestrating code):

    - ``_parent_chat_context_cont`` (for ``interject``): Continuation of the parent
      conversation since this loop started. Used to inject incremental updates into
      an ongoing conversation.

    - ``_parent_chat_context`` (for ``ask``): Full context snapshot for initializing
      a fresh inspection loop. Since ``ask`` spawns a new loop, it needs initial
      context, not a continuation. Hidden from LLM schemas via underscore prefix;
      orchestrating code injects this based on the LLM's ``include_parent_chat_context`` choice.

    Signature extension contract
    ----------------------------
    Derived classes **may** extend any steering method signature with additional
    keyword arguments that are specific to their domain.  For example,
    ``BaseActiveTask.stop`` adds a ``cancel`` kwarg that does not exist on the
    base ``stop(reason)`` signature, and ``ConversationManagerHandle.interject``
    replaces ``_parent_chat_context_cont`` with ``pinned`` / ``interjection_id``.

    The signatures defined here represent the **minimum universal contract** —
    the set of parameters that every handle is guaranteed to accept.  Callers
    that hold a reference typed as ``SteerableToolHandle`` may safely pass only
    these base parameters.

    When dispatching a steering call to a handle whose concrete type is
    unknown, use ``forward_handle_call`` (from
    ``unity.common._async_tool.messages``).  It introspects the target
    method's actual signature, filters out kwargs the target does not accept,
    and applies positional fallbacks — removing the need for hand-written
    ``try/except TypeError`` cascades at every delegation boundary.
    """

    @abstractmethod
    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
    ) -> "SteerableToolHandle":
        """Ask about status/progress if the task is still running, or the retrospective process/method if it has completed.

        Read-only — does not modify the task. This operation is asynchronous:
        it returns immediately and the answer appears in the task's history on
        the next turn.
        """

    @abstractmethod
    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
    ) -> None:
        """Provide additional information or instructions to the running task.

        Use this to give the task new context, correct its approach, or add
        requirements mid-flight without stopping or restarting it.
        """

    @abstractmethod
    async def stop(
        self,
        reason: Optional[str] = None,
    ) -> None:
        """Stop this tool, cancelling any pending work.

        While any tools are still running you cannot end the conversation;
        stop or wait for all in-flight tools to complete, then respond.
        """

    @abstractmethod
    async def pause(self) -> Optional[str]:
        """Pause this task temporarily without cancelling it.

        In-flight operations continue executing, but no new actions are taken
        until resumed.
        """

    @abstractmethod
    async def resume(self) -> Optional[str]:
        """Resume a task that was previously paused.

        Any work that completed while paused will be processed before the
        task continues.
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

        Provide the call_id from the clarification request and the answer text.
        No-op if the tool already finished.
        """

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
        response_format: Optional[Any] = None,
    ):
        self._task = task
        self._queue = interject_queue
        self._cancel_event = cancel_event
        self._stop_event = stop_event
        # "running" ⇢ Event **set**,  "paused" ⇢ Event **cleared**
        self._pause_event = pause_event or asyncio.Event()
        self._client = client
        self._pause_event.set()
        self._loop_id: str = loop_id
        # Human-friendly label for logs (includes 4-hex suffix when available).
        # This is populated by the inner loop as soon as it constructs LoopConfig.
        # Until then, fall back to the bare loop_id.
        self._log_label: str = loop_id
        # Only the top-level handle should emit the public stop log.
        # Nested/adopted handles will inherit False to avoid duplicate logging.
        self._is_root_handle: bool = False
        # When set, result() parses the raw JSON string from the loop into a
        # Pydantic model instance so callers don't need bespoke parsing.
        self._response_format: Optional[Any] = response_format

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

        # Context compression state
        self._raw_message_archives: list[list[dict]] = []
        self._compressed_renders: list[str] = []
        self._compression_count: int = 0
        self._loop_config: Optional[dict] = None

    # small local helpers to keep user-visible history consistent
    def _append_user_visible_user(
        self,
        message: str,
        _parent_chat_context_cont: list[dict] | None,
    ) -> None:
        with suppress(Exception):
            if _parent_chat_context_cont is not None:
                self._user_visible_history.append(
                    {
                        "role": "user",
                        "content": {
                            "message": message,
                            "_parent_chat_context_continued": _parent_chat_context_cont,
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

    # ── internal: steering helpers ──────────────────────────────────────────
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

    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
        _propagate_chat_context: ChatContextPropagation = ChatContextPropagation.LLM_DECIDES,
        _return_reasoning_steps: bool = False,
        **kwargs,
    ) -> "SteerableToolHandle":
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.info(f"{ICONS['clarification']} [{_label}] Ask requested: {question}")

        # Record the user-visible question immediately (even if delegated)
        self._append_user_visible_user(question, _parent_chat_context)

        # 1.  Gather a *read-only* snapshot of the loop being asked about.
        loop_chat_context = []
        with suppress(Exception):
            msgs = getattr(self._client, "messages", []) if self._client else []
            if msgs is None:
                msgs = []
            loop_chat_context = list(msgs)
        loop_chat_context_safe = make_messages_safe_for_context_dump(loop_chat_context)
        parent_chat_context_safe = make_messages_safe_for_context_dump(
            _parent_chat_context,
        )

        # When fresh parent context is provided, replace the stale runtime
        # parent-context header in the transcript with a pointer.  This avoids
        # duplicating the (potentially large) parent context while preserving
        # the structural marker so the inspection LLM knows the loop received
        # parent context and where it appeared in the conversation.
        if _parent_chat_context:
            loop_chat_context_safe = _replace_runtime_parent_context(
                loop_chat_context_safe,
            )

        # 1b. Snapshot ask_* tools available at invocation time so the
        #     inspection loop can propagate questions to inner handles.
        ask_tools: dict = {}
        with suppress(Exception):
            _get_ask_tools = getattr(self._task, "get_ask_tools", lambda: {})
            ask_tools = _get_ask_tools()

        # 2.  Prepare an *in-memory* Unify client for the **inspection** loop
        #     (LLM sees only the system header + follow-up user question).
        from .llm_client import new_llm_client

        parent_model: str | None = None
        with suppress(Exception):
            if self._client is not None:
                parent_model = self._client.endpoint

        inspection_client = new_llm_client(parent_model)

        # Build system message with the inspected loop's transcript.
        # Transform roles to inner_user/inner_assistant so the inspection LLM
        # can distinguish the inspected conversation from its own messages and
        # from the outer parent context (which uses outer_user/outer_assistant).
        loop_chat_context_transformed = _transform_inner_roles(
            loop_chat_context_safe,
        )

        transcript_description = (
            "This is the transcript of the tool/loop you are being asked about. "
            "Messages use 'inner_user' and 'inner_assistant' roles to clearly "
            "distinguish them from your current conversation. "
            "Use this to answer the user's question about the current state or progress."
        )
        if _parent_chat_context:
            transcript_description += (
                " Note: this is separate from the Parent Chat Context that may "
                "appear below — that context shows the broader conversation that "
                "led to this request, while this transcript is what you are "
                "answering questions about."
            )

        sys_msg_parts = [
            "You are inspecting a running tool-use conversation to answer a question about it.",
            "",
            "## Inspected Loop Transcript",
            transcript_description,
            "",
            json.dumps(loop_chat_context_transformed, indent=2),
        ]

        # If inner-handle ask_* tools are available, hint the LLM about them
        if ask_tools:
            sys_msg_parts.extend(
                [
                    "",
                    "## Inner Loop Tools",
                    (
                        "You have access to `ask_*` tools that query inner tool loops for detailed information. "
                        "Each inner tool loop has its own transcript that may contain details NOT visible in the "
                        "Inspected Loop Transcript above. If the transcript does not contain enough information "
                        "to answer the question — for example if a tool's result only shows a placeholder or "
                        "summary — you MUST call the corresponding `ask_*` tool to get details from that "
                        "tool's own internal context. Only answer directly from the transcript when it clearly "
                        "contains the specific information being asked about."
                    ),
                ],
            )

        sys_msg_parts.extend(
            [
                "",
                "Answer the user's follow-up question using the context above and any tools exposed to you.",
                "Do not ask the user questions or request clarification. If information is missing,",
                "state what is known and, if helpful, briefly note assumptions. Respond in a single, concise paragraph.",
            ],
        )

        inspection_client.set_system_message("\n".join(sys_msg_parts))

        # 3.  Fire off a *stand-alone* read-only loop.
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

        # ── Sibling lineage for the ask sub-loop ──────────────────────
        # The ask loop is a *sibling* of the parent loop (not a child).
        # It shares the parent's parent lineage so the frontend can
        # place it at the correct nesting level in the action tree.
        _parent_hierarchy = list(getattr(self, "_log_hierarchy", None) or [])
        _sibling_lineage = _parent_hierarchy[:-1] if len(_parent_hierarchy) > 1 else []

        # ── Boundary ManagerMethod events ─────────────────────────────
        # Publish incoming/outgoing ManagerMethod events around the sub-loop
        # so the frontend can create a distinct node for each ask() call.
        # This mirrors the boundary wrapper pattern used by execute_code
        # and execute_function in CodeActActor.
        from secrets import token_hex as _token_hex
        from ..events.manager_event_logging import (
            new_call_id as _new_call_id,
            publish_manager_method_event as _pub_mm,
        )

        _ask_call_id = _new_call_id()
        _ask_suffix = _token_hex(2)
        _ask_manager = (self._loop_id or "").split(".")[0] or "unknown"
        _ask_hierarchy = [*_sibling_lineage, f"{loop_id_label}({_ask_suffix})"]

        await _pub_mm(
            _ask_call_id,
            _ask_manager,
            "ask",
            phase="incoming",
            display_label="Answering Question",
            question=question,
            hierarchy=_ask_hierarchy,
        )

        # The question is sent as a plain user message (context is in system message)
        _ask_message = question

        # Set _PENDING_LOOP_SUFFIX so the inner LoopConfig picks up
        # the same suffix as our boundary event.
        _suffix_token = _PENDING_LOOP_SUFFIX.set(_ask_suffix)
        try:
            helper_handle = start_async_tool_loop(
                inspection_client,
                _ask_message,
                ask_tools,  # ask_* tools for inner handle propagation
                loop_id=loop_id_label,
                parent_lineage=_sibling_lineage,
                parent_chat_context=(
                    parent_chat_context_safe if _parent_chat_context else None
                ),
                propagate_chat_context=_propagate_chat_context,
                prune_tool_duplicates=False,
                interrupt_llm_with_interjections=False,
                max_consecutive_failures=1,
            )
        finally:
            _PENDING_LOOP_SUFFIX.reset(_suffix_token)

        # Monkey-patch result() to record the assistant answer when available
        # AND publish the outgoing ManagerMethod boundary event.
        if not _return_reasoning_steps:
            _orig_result = helper_handle.result

            async def _rec_result():  # type: ignore[return-type]
                ans = await _orig_result()
                self._append_user_visible_assistant(ans)
                await _pub_mm(
                    _ask_call_id,
                    _ask_manager,
                    "ask",
                    phase="outgoing",
                    display_label="Answering Question",
                    answer=ans if isinstance(ans, str) else str(ans),
                    hierarchy=_ask_hierarchy,
                )
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
            await _pub_mm(
                _ask_call_id,
                _ask_manager,
                "ask",
                phase="outgoing",
                display_label="Answering Question",
                answer=answer if isinstance(answer, str) else str(answer),
                hierarchy=_ask_hierarchy,
            )
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
        _parent_chat_context_cont: list[dict] | None = None,
        trigger_immediate_llm_turn: bool = True,
        **kwargs,
    ) -> None:
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.debug(
            f"{ICONS['interjection']} [{_label}] Interject requested: {message}",
        )
        # Record user-visible immediately
        self._append_user_visible_user(message, _parent_chat_context_cont)

        # Buffer then forward to resolver loop. Support dict payloads when continued context provided.
        payload = {
            "message": message,
            "_parent_chat_context_continued": _parent_chat_context_cont,
            "trigger_immediate_llm_turn": trigger_immediate_llm_turn,
            "suppress_response_notification": kwargs.get(
                "suppress_response_notification",
                False,
            ),
        }
        # Use put_nowait to ensure the interjection is registered *synchronously* before
        # we yield control. This prevents a race where a fast-running loop completes
        # its turn and exits before seeing the queued item.
        self._queue.put_nowait(payload)

        # Also mirror as synthetic helper tool_calls immediately (no LLM step)
        try:
            await self._queue.put(
                {
                    "_mirror": {
                        "method": "interject",
                        "kwargs": {
                            "message": message,
                            **(kwargs or {}),
                        },
                    },
                },
            )
        except Exception:
            pass

    @functools.wraps(SteerableToolHandle.stop, updated=())
    async def stop(
        self,
        reason: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Idempotent guard: if already stopping, do nothing and DO NOT log again
        if self._cancel_event.is_set():
            return

        # Stop request is logged centrally in the loop via mirror path

        # Ensure the loop is not paused so the inner loop can observe and process the stop immediately
        with suppress(Exception):
            self._pause_event.set()
        # Mirror as synthetic helper tool_call (no LLM step) before signalling cancel/stop
        try:
            self._queue.put_nowait(
                {
                    "_mirror": {
                        "method": "stop",
                        "kwargs": {
                            "reason": reason,
                            **(kwargs or {}),
                        },
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
        LOGGER.info(f"{ICONS['pause']} [{_label}] Pause requested")

        # Immediately toggle pause_event for base (non-steerable) tools.
        # Steerable handles (h is not None) are intentionally skipped here;
        # they are paused via the mirror path below, which synthesizes
        # helper tool_calls in the transcript so the outer LLM has full
        # visibility that the inner tool was paused. Base tools have no
        # handle — only a raw pause_event — so the mirror's
        # _dispatch_steering_to_child would reach them too, but toggling
        # the event directly here eliminates any latency window between
        # this call and the next loop iteration that drains the mirror.
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
        # Mirror as synthetic helper tool_call (no LLM step).
        # The inner loop processes this via _synthesize_mirrored_helper_calls,
        # which dispatches pause to ALL children (steerable and base alike).
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
        LOGGER.info(f"{ICONS['resume']} [{_label}] Resume requested")
        # Immediately toggle pause_event for base (non-steerable) tools.
        # Steerable handles are resumed via the mirror path below (see the
        # symmetric comment in pause() for the full rationale). Direct
        # toggling here gives base tools instant resume without waiting
        # for the next loop iteration to drain the mirror.
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
        # Mirror as synthetic helper tool_call (no LLM step).
        # The inner loop processes this via _synthesize_mirrored_helper_calls,
        # which dispatches resume to ALL children (steerable and base alike).
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
    async def result(self):
        """Return the final answer once the conversation loop (or delegate) completes.

        When *response_format* was supplied to ``start_async_tool_loop``, the
        raw JSON string produced by the inner loop is automatically parsed into
        a Pydantic model instance.  Callers receive the typed object directly
        and do not need to call ``model_validate_json`` themselves.

        If the inner loop returns ``_COMPRESSION_SIGNAL``, the handle
        orchestrates context compression and starts a new loop transparently.
        This may repeat multiple times; callers always receive the final
        real result.
        """
        _stopped_notice = "processed stopped early, no result"
        while True:
            try:
                raw = await self._task
            except asyncio.CancelledError:
                return _stopped_notice

            if raw is _COMPRESSION_SIGNAL:
                try:
                    await self._restart_with_compressed_context()
                except Exception as exc:
                    LOGGER.error(
                        f"Context compression failed: {type(exc).__name__}: {exc}",
                    )
                    return _stopped_notice
                continue

            if self._response_format is not None and isinstance(raw, str):
                try:
                    return self._response_format.model_validate_json(raw)
                except Exception:
                    pass
            return raw

    async def _restart_with_compressed_context(self) -> None:
        """Archive messages, compress them, and start a new loop.

        Swaps ``_task`` in-place so that all steering methods automatically target the new loop.
        ``_queue`` (interject queue) and ``_pause_event`` are reused so pending interjections carry over and pause state persists.
        """
        cfg = self._loop_config
        if cfg is None:
            raise RuntimeError(
                "Cannot compress: loop config was not stored on the handle.",
            )

        # 1. Separate system messages (preserved verbatim) from conversation
        #    messages (compressed). System messages carry special attributes
        #    like _runtime_context that must not be touched by the compressor.
        all_messages = copy.deepcopy(self._client.messages)
        system_msgs = [m for m in all_messages if m.get("role") == "system"]
        conversation_messages = [m for m in all_messages if m.get("role") != "system"]

        self._raw_message_archives.append(conversation_messages)

        # 2. Compress with index offset so labels continue from prior passes
        index_offset = sum(len(a) for a in self._raw_message_archives[:-1])
        compressed = await compress_messages(
            conversation_messages,
            self._client.endpoint,
        )
        rendered = render_compressed_context(compressed, index_offset=index_offset)
        self._compressed_renders.append(rendered)

        combined = "## Compressed Prior Context\n" + "\n".join(self._compressed_renders)

        # 3. Build unpack_messages closure over all archives (cumulative indexing)
        archive_ref = self._raw_message_archives

        def unpack_messages(index: int, n: int = 1) -> str:
            """Retrieve one or more uncompressed messages by index.

            ``index`` corresponds to the ``[N]`` label in the compressed
            context summary.  Returns up to ``n`` consecutive original
            messages as a JSON array.
            """
            flat = [msg for archive in archive_ref for msg in archive]
            if index < 0 or index >= len(flat):
                return json.dumps(
                    {"error": f"Index {index} out of range (0-{len(flat) - 1})"},
                )
            end = min(index + n, len(flat))
            return json.dumps(flat[index:end], default=str)

        # 4. Reuse existing client -- carry over all system message blocks.
        #    Replace existing compressed-context message if present,
        #    otherwise append a new one.
        compressed_sys_msg = {
            "role": "system",
            "_compressed_message": True,
            "content": (
                f"{combined}\n\n"
                "When you need details from a compressed message, call "
                "`unpack_messages(index)` with its `[N]` index to "
                "retrieve the full original content. Pass `n` to "
                "retrieve a range of consecutive messages."
            ),
        }

        existing_idx = next(
            (i for i, m in enumerate(system_msgs) if m.get("_compressed_message")),
            None,
        )
        if existing_idx is not None:
            system_msgs[existing_idx] = compressed_sys_msg
        else:
            system_msgs.append(compressed_sys_msg)

        self._client._messages = system_msgs
        self._client._system_message = None

        # 5. Prepare tools: original tools + unpack_messages
        tools = dict(cfg["tools"])
        tools["unpack_messages"] = unpack_messages

        # 6. Reuse all steering events so that any stop/cancel/pause issued
        #    during the compression window carries over to the new loop.
        outer_handle_container: list = [None]

        _parent = cfg["parent_lineage"] or TOOL_LOOP_LINEAGE.get([])
        _lineage = [*_parent, cfg.get("loop_id", "compressed")]

        # All keys in _loop_config map directly to async_tool_loop_inner
        # kwargs except "parent_lineage" (used to compute lineage above)
        # and "tools" (overridden with unpack_messages added).
        inner_kwargs = {
            k: v for k, v in cfg.items() if k not in ("parent_lineage", "tools")
        }

        async def _loop_wrapper():
            return await async_tool_loop_inner(
                self._client,
                "Context was compressed. Continue from where you left off.",
                tools,
                lineage=_lineage,
                interject_queue=self._queue,
                cancel_event=self._cancel_event,
                stop_event=self._stop_event,
                pause_event=self._pause_event,
                outer_handle_container=outer_handle_container,
                **inner_kwargs,
            )

        new_task = asyncio.create_task(_loop_wrapper(), name="ToolUseLoop")

        with suppress(Exception):
            setattr(new_task, "task_info", {})
            setattr(new_task, "clarification_channels", {})
            setattr(new_task, "get_ask_tools", lambda: {})
            setattr(new_task, "get_completed_tool_metadata", lambda: {})

        # 7. Swap task reference (client and events are reused in-place)
        self._task = new_task
        self._compression_count += 1

        outer_handle_container[0] = self

        LOGGER.info(
            f"{ICONS.get('completed', '✓')} [{self._log_label}] "
            f"Context compressed (pass #{self._compression_count}), "
            f"archived {len(conversation_messages)} messages, new loop started.",
        )

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
    propagate_chat_context: ChatContextPropagation = ChatContextPropagation.LLM_DECIDES,
    parent_chat_context: Optional[list[dict]] = None,
    caller_description: Optional[str] = None,
    log_steps: Union[bool, str] = True,
    max_steps: Optional[int] = None,
    timeout: Optional[int] = None,
    raise_on_limit: bool = False,
    include_class_in_dynamic_tool_names: bool = False,
    tool_policy: Optional[
        Union[
            Callable[[int, Dict[str, Callable]], Tuple[str, Dict[str, Callable]]],
            Callable[
                [int, Dict[str, Callable], list[str]],
                Tuple[str, Dict[str, Callable]],
            ],
        ]
    ] = None,
    preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]] = None,
    response_format: Optional[Any] = None,
    max_parallel_tool_calls: Optional[int] = None,
    handle_cls: Optional[Type[AsyncToolLoopHandle]] = None,
    evented: Optional[bool] = None,
    persist: bool = False,
    multi_handle: bool = False,
    prompt_caching: Optional["PromptCacheParam"] = None,
    time_awareness: bool = False,
    extra_ask_tools: Optional[Dict[str, Callable]] = None,
    enable_compression: bool = True,
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

    timeout : int | None, default None
        Activity-based timeout in seconds. When ``None`` (default), no
        timeout is enforced.

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

    time_awareness : bool, default True
        If ``True``, a time-context system message is injected into the
        conversation and updated after each tool completion, giving the LLM
        awareness of wall-clock time and tool execution durations.  If
        ``False``, the time-context table is omitted entirely.
    """
    # Ensure a stable loop_id for consistent logging across handle and inner loop
    if loop_id is not None:
        client.set_origin(loop_id)
    loop_id = loop_id if loop_id is not None else short_id()
    interject_queue: asyncio.Queue[dict | str] = asyncio.Queue()
    cancel_event = asyncio.Event()
    stop_event = asyncio.Event()
    pause_event = asyncio.Event()
    pause_event.set()  # start un-paused

    # A single-element list is a mutable container that the inner loop can use
    # to access the outer handle once it exists.
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
                persist=persist,
                multi_handle_coordinator=multi_handle_coordinator,
                prompt_caching=prompt_caching,
                time_awareness=time_awareness,
                extra_ask_tools=extra_ask_tools,
                enable_compression=enable_compression,
            )
        except asyncio.CancelledError:
            raise

    task = asyncio.create_task(_loop_wrapper(), name="ToolUseLoop")

    # Make introspection surfaces available immediately on the wrapper task.
    # The inner loop rebinding will point these to the live dicts once running.
    try:  # pragma: no cover
        setattr(task, "task_info", {})  # asyncio.Task -> ToolCallMetadata
        setattr(task, "clarification_channels", {})  # call_id -> (up_q, down_q)
        setattr(task, "get_ask_tools", lambda: {})  # snapshot of ask_* dynamic tools
        setattr(
            task,
            "get_completed_tool_metadata",
            lambda: {},
        )  # completed tool metadata with handle refs
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
        response_format=response_format,
    )

    # Store loop config so _restart_with_compressed_context can re-create
    # the loop with identical settings after compression.
    handle._loop_config = {
        "loop_id": loop_id,
        "parent_lineage": list(_parent),
        "tools": dict(tools),
        "max_consecutive_failures": max_consecutive_failures,
        "prune_tool_duplicates": prune_tool_duplicates,
        "interrupt_llm_with_interjections": interrupt_llm_with_interjections,
        "propagate_chat_context": propagate_chat_context,
        "parent_chat_context": parent_chat_context,
        "caller_description": caller_description,
        "log_steps": log_steps,
        "max_steps": max_steps,
        "timeout": timeout,
        "raise_on_limit": raise_on_limit,
        "include_class_in_dynamic_tool_names": include_class_in_dynamic_tool_names,
        "tool_policy": tool_policy,
        "preprocess_msgs": preprocess_msgs,
        "response_format": response_format,
        "max_parallel_tool_calls": max_parallel_tool_calls,
        "persist": persist,
        "multi_handle_coordinator": multi_handle_coordinator,
        "prompt_caching": prompt_caching,
        "time_awareness": time_awareness,
        "extra_ask_tools": extra_ask_tools,
        "enable_compression": enable_compression,
    }

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
    - Enqueues a mirror sentinel that the inner loop consumes to synthesize
      helper tool_calls/acks.
    - The mirror payload carries control keys ("_custom", "_aliases", "_fallback")
      used only by the inner loop.

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

        def _mirror(self: "AsyncToolLoopHandle", kwargs, result):
            # Mirror to the inner loop with control keys for routing/dispatch
            try:
                self._queue.put_nowait(
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
                return _mirror(self, kw, res)

            return _async_wrapped

        @functools.wraps(fn, updated=())
        def _sync_wrapped(self: "AsyncToolLoopHandle", *a, **kw):
            res = fn(self, *a, **kw)
            return _mirror(self, kw, res)

        return _sync_wrapped

    return _decorator
