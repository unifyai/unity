import asyncio
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
from ..constants import LOGGER
from .llm_helpers import short_id
from ._async_tool.loop_config import TOOL_LOOP_LINEAGE
from ._async_tool.messages import forward_handle_call
from ._async_tool.loop import async_tool_loop_inner
from ._async_tool.propagation_mode import ChatContextPropagation
from typing import Iterable


from ._async_tool.multi_handle import (
    MultiHandleCoordinator,
    MultiRequestHandle,
)
from ._async_tool.tagging import tag_message_with_request

if TYPE_CHECKING:
    from ..image_manager.types.image_refs import ImageRefs


# Tiny handle objects exposed to callers
# ─────────────────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod


class SteerableHandle(ABC):
    """Abstract base class for steerable handles.

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
    """

    @abstractmethod
    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
        images: "Optional[ImageRefs]" = None,
    ) -> "SteerableHandle":
        """
        Query the status or progress of this running task (async - result arrives on next turn).

        Use this to check on updates, get a summary of what has happened so far,
        or ask clarifying questions about the task's state without modifying it.

        This operation is asynchronous: it returns immediately with "Query submitted",
        and the actual response appears in the task's history when ready (status
        changes from 'pending' to 'completed'). You will automatically receive
        another turn to see and act on the result.

        Parameters
        ----------
        question : str
            The follow-up user question.
        images : ImageRefs | None, optional
            Live image references to make available during this ask flow.
            Implementations should forward these to any nested asks so inner
            loops can attach/ask about images (optionally with new annotations).
        """

    @abstractmethod
    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
        images: "Optional[ImageRefs]" = None,
    ) -> Optional[str]:
        """Provide additional information or instructions to the running task.

        Use this to give the task new context, correct its approach, or add
        requirements mid-flight without stopping or restarting it.

        Parameters
        ----------
        message : str
            The user interjection to inject into the loop.
        images : ImageRefs | None, optional
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
    async def stop(
        self,
        reason: Optional[str] = None,
        *,
        images: "Optional[ImageRefs]" = None,
    ) -> Optional[str]:
        """Stop this task immediately, cancelling any pending work.

        Use this when the task should be terminated. This is a destructive
        action that cannot be undone.

        Parameters
        ----------
        reason : str | None
            Optional human-readable reason for stopping.
        images : ImageRefs | None, optional
            Live image references to attach at the time of this stop command.
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
        self._pause_event.set()
        self._loop_id: str = loop_id
        # Human-friendly label for logs (includes 4-hex suffix when available).
        # This is populated by the inner loop as soon as it constructs LoopConfig.
        # Until then, fall back to the bare loop_id.
        self._log_label: str = loop_id
        # Only the top-level handle should emit the public stop log.
        # Nested/adopted handles will inherit False to avoid duplicate logging.
        self._is_root_handle: bool = False

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
        images: "Optional[ImageRefs]" = None,
        _return_reasoning_steps: bool = False,
        **kwargs,
    ) -> "SteerableToolHandle":
        """
        Answers *question* about this *pending* tool, associated with this handle.
        The question is read-only (the tool state is not modified whatsoever).
        The calling parent loop is left completely untouched.
        When ``_parent_chat_context`` is provided, the context is included in the
        inspection loop's system message to provide additional context about the
        broader conversation that led to this question.

        If ``images`` are provided, the spawned inspection loop receives live
        images (helpers exposed, synthetic overview injected) and any nested asks
        can receive images (with optional new annotations).
        """
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.info(f"❓ [{_label}] Ask requested: {question}")

        # Record the user-visible question immediately (even if delegated)
        self._append_user_visible_user(question, _parent_chat_context)

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

                async def interject(self, message: str, **kwargs): ...

                async def stop(self, reason: Optional[str] = None, **kwargs): ...

                async def pause(self): ...

                async def resume(self): ...

                def done(self):
                    return True

                async def result(self):
                    return await _static()

                async def ask(self, question: str, **kwargs) -> "SteerableToolHandle":
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

        # Build system message with transcript and optional parent context
        sys_msg_parts = [
            "You are inspecting a running tool-use conversation to answer a question about it.",
            "",
            "## Inspected Loop Transcript",
            (
                "This is the transcript of the tool/loop you are being asked about. "
                "Use this to answer the user's question about the current state or progress."
            ),
            "",
            json.dumps(parent_ctx, indent=2),
        ]

        # If parent context is provided, add it as a separate section
        if _parent_chat_context:
            sys_msg_parts.extend(
                [
                    "",
                    "## Parent Chat Context",
                    (
                        "This is the broader conversation context from which this question originated. "
                        "It may help explain why this question is being asked. Note: this is separate "
                        "from the Inspected Loop Transcript above - that transcript is what you are "
                        "answering questions about."
                    ),
                    "",
                    json.dumps(_parent_chat_context, indent=2),
                ],
            )

        sys_msg_parts.extend(
            [
                "",
                "Answer the user's follow-up question using ONLY this context.",
                "Do not attempt to run new tools unless they are exposed to you.",
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

        # The question is sent as a plain user message (context is in system message)
        _ask_message = question

        helper_handle = start_async_tool_loop(
            inspection_client,
            _ask_message,
            {},  # no recursive tools
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
        _parent_chat_context_cont: list[dict] | None = None,
        images: "Optional[ImageRefs]" = None,
        trigger_immediate_llm_turn: bool = True,
        **kwargs,
    ) -> None:
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.debug(f"💬 [{_label}] Interject requested: {message}")
        # Record user-visible immediately
        self._append_user_visible_user(message, _parent_chat_context_cont)

        # Buffer then forward to resolver loop. Support dict payloads when continued context provided.
        payload = {
            "message": message,
            "_parent_chat_context_continued": _parent_chat_context_cont,
            "images": images,
            "trigger_immediate_llm_turn": trigger_immediate_llm_turn,
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
                            "images": images,
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
        *,
        images: "Optional[ImageRefs]" = None,
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
                            "images": images,
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
    images: Optional["ImageRefs"] = None,
    evented: Optional[bool] = None,
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
                images=images,
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
