import asyncio
import unify
import os
import functools
import json
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

if TYPE_CHECKING:
    from ..image_manager.image_manager import ImageHandle

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
        try:
            # NOTE: key name 'parent_chat_context_continuted' is legacy and intentional
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
        except Exception:
            pass

    def _append_user_visible_assistant(self, message: str) -> None:
        try:
            self._user_visible_history.append(
                {"role": "assistant", "content": message},
            )
        except Exception:
            pass

    # ── internal: passthrough forwarding/buffering helpers ──────────────────
    def _iter_passthrough_handles(self):
        try:
            task_info = getattr(self._task, "task_info", {})
        except Exception:
            task_info = {}
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
        try:
            ti = getattr(self._task, "task_info", {})
            return isinstance(ti, dict) and len(ti) > 0
        except Exception:
            return False

    async def _forward_call_to_handle(
        self,
        handle,
        method_name: str,
        kwargs: dict,
        fallback: tuple[str, ...],
    ):
        try:
            return await forward_handle_call(
                handle,
                method_name,
                kwargs,
                fallback_positional_keys=fallback,
            )
        except Exception:
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
        try:
            msgs = getattr(self._client, "messages", []) if self._client else []
            if msgs is None:
                msgs = []
            parent_ctx = list(msgs)
        except Exception:
            parent_ctx = []

        # 2.  Prepare an *in-memory* Unify client for the **inspection** loop
        #     (LLM sees only the system header + follow-up user question).
        inspection_client = unify.AsyncUnify(
            "gpt-5@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
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
        try:
            task_info = getattr(self._task, "task_info", {})
        except Exception:
            task_info = {}

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
            try:
                _cid = getattr(_inf, "call_id", None)
            except Exception:
                _cid = None
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
        try:
            parent_label: str = (
                getattr(self, "_log_label", None)
                or getattr(self, "_loop_id", "unknown")
                or "unknown"
            )
        except Exception:
            parent_label = "unknown"

        loop_id_label = f"Question({parent_label})"

        # Build the message for the inspection loop – either a plain string or
        # a single string that embeds the continued parent context.
        if parent_chat_context_cont is not None:
            try:
                _ctx_text = json.dumps(parent_chat_context_cont, indent=2)
            except Exception:
                _ctx_text = str(parent_chat_context_cont)
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
        images: dict | None = None,
    ) -> None:
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.info(f"💬 [{_label}] Interject requested: {message}")
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
        try:
            asyncio.get_running_loop().create_task(
                self._replay_pending_passthrough_ops(),
            )
        except Exception:
            pass
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
        try:
            self._task.cancel()
        except Exception:
            pass
        try:
            self._stop_event.set()
        except Exception:
            pass

    @functools.wraps(SteerableToolHandle.pause, updated=())
    def pause(self) -> None:
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.info(f"⏸️ [{_label}] Pause requested")
        # Propagate pause to any nested handles first (always)
        try:
            task_info = getattr(self._task, "task_info", {})
        except Exception:
            task_info = {}
        try:
            items = task_info.items() if isinstance(task_info, dict) else []
            for _t, _inf in items:
                try:
                    h = _inf.handle
                except Exception:
                    h = None
                if h is not None and hasattr(h, "pause"):
                    try:
                        maybe = h.pause()  # may be sync or async
                        if asyncio.iscoroutine(maybe):
                            asyncio.create_task(maybe)
                    except Exception:
                        pass
        except Exception:
            pass

        self._pause_event.clear()

    @functools.wraps(SteerableToolHandle.resume, updated=())
    def resume(self) -> None:
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.info(f"▶️ [{_label}] Resume requested")
        # Propagate resume to any nested handles first (always)
        try:
            task_info = getattr(self._task, "task_info", {})
        except Exception:
            task_info = {}
        try:
            items = task_info.items() if isinstance(task_info, dict) else []
            for _t, _inf in items:
                try:
                    h = _inf.handle
                except Exception:
                    h = None
                if h is not None and hasattr(h, "resume"):
                    try:
                        maybe = h.resume()  # may be sync or async
                        if asyncio.iscoroutine(maybe):
                            asyncio.create_task(maybe)
                    except Exception:
                        pass
        except Exception:
            pass

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
        try:
            task_info = getattr(self._task, "task_info", {})
            clar_map = getattr(self._task, "clarification_channels", {})
        except Exception:
            task_info, clar_map = {}, {}

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

        try:
            if down_q is not None:
                await down_q.put(answer)
        except Exception:
            pass

    # No _adopt: passthrough no longer adopts delegates; outer loop remains active.


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
    semantic_cache: Optional[bool] = False,
    images: Optional[dict[str, "ImageHandle"]] = None,
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

    # Attach lineage to handle for optional external inspection
    try:
        handle._lineage = list(_lineage)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Mark this handle as the root/top-level for single-stop logging semantics
    try:
        handle._is_root_handle = True  # type: ignore[attr-defined]
    except Exception:
        pass

    # Let the inner coroutine discover the outer handle so it can switch
    # steering when a nested handle requests pass-through behaviour.
    outer_handle_container[0] = handle

    return handle
