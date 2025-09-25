import asyncio
import unify
import os
import functools
import json
from typing import Optional, Awaitable, Dict, Callable, Tuple, Any, Union
from ..constants import LOGGER
from .llm_helpers import short_id
from ._async_tool.loop_config import TOOL_LOOP_LINEAGE
from ._async_tool.loop import async_tool_use_loop_inner

# Tiny handle objects exposed to callers
# ─────────────────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod


class SteerableHandle(ABC):
    """Abstract base class for steerable handles."""

    @abstractmethod
    async def ask(self, question: str) -> "SteerableHandle":
        """
        Ask a question to the running process.
        """

    @abstractmethod
    def interject(self, message: str) -> Awaitable[Optional[str]] | Optional[str]:
        """Inject an additional *user* turn into the running conversation."""


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
    ) -> Awaitable[Optional[str]] | Optional[str]:
        """Shutdown the loop, killing any pending work in the process."""

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


class AsyncToolUseLoopHandle(SteerableToolHandle):
    """
    Returned by `start_async_tool_use_loop`.  Lets you
      • queue extra user messages while the loop runs and
      • stop the loop at any time.
    """

    def __init__(
        self,
        *,
        task: asyncio.Task,
        interject_queue: asyncio.Queue[str],
        cancel_event: asyncio.Event,
        stop_event: asyncio.Event,
        pause_event: Optional[asyncio.Event] = None,
        client: "unify.AsyncUnify | None" = None,
        loop_id: str = "",
        initial_user_message: Optional[str] = None,
    ):
        self._task = task
        self._queue = interject_queue
        self._cancel_event = cancel_event
        self._stop_event = stop_event
        # "running" ⇢ Event **set**,  "paused" ⇢ Event **cleared**
        self._pause_event = pause_event or asyncio.Event()
        self._client = client
        # Optional live delegate – set via ``_adopt`` when this handle should
        # forward every steering call to another *SteerableToolHandle*.
        self._delegate: Optional["SteerableToolHandle"] = None
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
        self._early_interjects: list[str] = []

        # Maintain a user-visible history (what the end-user would see):
        # Records: original prompt (user), interjections (user), ask Q/A (user/assistant).
        self._user_visible_history: list[dict] = []
        if initial_user_message:
            self._user_visible_history.append(
                {"role": "user", "content": initial_user_message},
            )

    async def ask(
        self,
        question: str,
        *,
        _return_reasoning_steps: bool = False,
    ) -> "SteerableToolHandle":
        """
        Answers *question* about this *pending* tool, associated with this handle.
        The question is read-only (the tool state is not modified whatsoever).
        The calling parent loop is left completely untouched.
        """
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.info(f"🕹️ [{_label}] Ask requested: {question}")
        # Fast-path: delegated handles answer directly.
        if self._delegate is not None:
            return await self._delegate.ask(
                question,
                _return_reasoning_steps=_return_reasoning_steps,
            )

        # Record the user-visible question immediately (even if delegated)
        try:
            self._user_visible_history.append({"role": "user", "content": question})
        except Exception:
            pass

        # 0.  Defensive guard: if the outer loop has already finished we can
        #     just answer from the final transcript without starting another
        #     loop.
        if self.done():
            LOGGER.warning(
                "AsyncToolUseLoopHandle.ask() called on an already-finished "
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
        parent_ctx = list(self._client.messages) if self._client else []

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
            _proxy.__name__ = f"ask_{_inf['call_id']}"
            recursive_tools[_proxy.__name__] = _proxy
        # ----------------------------------------------------------------

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

        # Best-effort detection of a single nested handle to enrich the label
        child_label: str | None = None
        try:
            _ti = getattr(self._task, "task_info", {})  # type: ignore[attr-defined]
            nested_ids: set[str] = set()
            for _t, _inf in _ti.items() if isinstance(_ti, dict) else []:
                _h = _inf.get("handle")
                _lid = getattr(_h, "_log_label", None) or getattr(_h, "_loop_id", None)
                if isinstance(_lid, str) and _lid:
                    nested_ids.add(_lid)
            if len(nested_ids) == 1:
                child_label = next(iter(nested_ids))
        except Exception:
            child_label = None

        if child_label:
            loop_id_label = f"Question({parent_label}->{child_label})"
        else:
            loop_id_label = f"Question({parent_label})"

        helper_handle = start_async_tool_use_loop(
            inspection_client,
            question,
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
                try:
                    self._user_visible_history.append(
                        {"role": "assistant", "content": ans},
                    )
                except Exception:
                    pass
                return ans

            helper_handle.result = _rec_result  # type: ignore[attr-defined]
            return helper_handle

        async def _wrap():
            answer = await helper_handle.result()
            try:
                self._user_visible_history.append(
                    {"role": "assistant", "content": answer},
                )
            except Exception:
                pass
            return answer, inspection_client.messages

        helper_handle.result = _wrap  # type: ignore[attr-defined]
        return helper_handle

    # -- public API -----------------------------------------------------------
    @functools.wraps(SteerableToolHandle.interject, updated=())
    async def interject(self, message: str) -> None:
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.info(f"️ [{_label}] Interject requested: {message}")
        if self._delegate is not None:
            await self._delegate.interject(message)
            return
        # Buffer then forward to resolver loop.
        self._early_interjects.append(message)
        await self._queue.put(message)

    @functools.wraps(SteerableToolHandle.stop, updated=())
    def stop(
        self,
        reason: Optional[str] = None,
    ) -> None:
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

        # Best-effort forwarding to a delegate (no logging, no early return)
        if self._delegate is not None:
            try:
                self._delegate.stop(reason=reason)  # type: ignore[misc]
            except TypeError:
                try:
                    self._delegate.stop(reason)  # type: ignore[misc]
                except Exception:
                    pass
            except Exception:
                pass

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
        if self._delegate is not None:
            self._delegate.pause()
            return
        self._pause_event.clear()
        # Propagate pause to any nested steerable handles that expose `.pause`
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
                        # Best-effort propagation – never break outer pause
                        pass
        except Exception:
            # Defensive: do not let propagation errors bubble up
            pass

    @functools.wraps(SteerableToolHandle.resume, updated=())
    def resume(self) -> None:
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.info(f"▶️ [{_label}] Resume requested")
        if self._delegate is not None:
            self._delegate.resume()
            return
        self._pause_event.set()
        # Propagate resume to any nested steerable handles that expose `.resume`
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
                        # Best-effort propagation – never break outer resume
                        pass
        except Exception:
            # Defensive: do not let propagation errors bubble up
            pass

    @functools.wraps(SteerableToolHandle.done, updated=())
    def done(self) -> bool:
        if self._delegate is not None:
            return self._delegate.done()
        return self._task.done()

    @functools.wraps(SteerableToolHandle.result, updated=())
    async def result(self) -> str:
        """Return the final answer once the conversation loop (or delegate) completes."""
        if self._delegate is not None:
            # 1) Wait for the delegated (inner) handle to finish and capture its answer.
            ans = await self._delegate.result()
            # 2) Best-effort: also wait for the OUTER loop task to finish so no background work remains.
            #    Swallow any exceptions here to preserve prior semantics (caller receives the inner result).
            try:
                await self._task
            except Exception:
                pass
            return ans
        return await self._task

    # ── internal helper ──────────────────────────────────────────────────────
    def _adopt(self, new_handle: "SteerableToolHandle") -> None:
        """Switch all steering methods to *new_handle* (in-process only).

        Move any *already queued* interjections over to the freshly adopted
        delegate so that early user guidance (issued *before* the delegate was
        ready) is not lost – a common source of hangs during tests that fire
        `interject()` immediately after `execute()` returns.
        """
        # Flush queued interjections collected before the delegate became
        # available.  We dispatch them *asynchronously* so that we keep the
        # adopt operation non-blocking and avoid re-entrancy problems if the
        # delegate itself relies on the outer event-loop.
        import asyncio  # local import to dodge unconditional dependency at top-level

        while not self._queue.empty():
            try:
                msg = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            # Forward the message to the delegate.  We purposefully schedule the
            # coroutine in the background – it is semantically equivalent to the
            # original `interject()` call which also runs fire-and-forget.
            try:
                maybe_coro = new_handle.interject(msg)  # type: ignore[attr-defined]
                if asyncio.iscoroutine(maybe_coro):
                    asyncio.create_task(maybe_coro)
            except Exception:
                # Silently swallow to preserve backwards-compat – early
                # interjections are *best-effort* hints rather than critical.
                pass

        # Keep pause / cancel signals in sync – they might have been toggled
        # before we adopted the delegate.
        try:
            if not self._pause_event.is_set() and hasattr(new_handle, "pause"):
                new_handle.pause()  # type: ignore[attr-defined]
            if self._cancel_event.is_set() and hasattr(new_handle, "stop"):
                new_handle.stop()  # type: ignore[attr-defined]
        except Exception:
            # These are advisory only – failing to propagate them should never
            # break the overall execution.
            pass

        # Ensure only the original top-level handle is considered root for logging
        try:
            setattr(new_handle, "_is_root_handle", False)
        except Exception:
            pass

        self._delegate = new_handle

        # ── Flush any interjections that were consumed by the resolver loop ──
        #     before the delegate became available.
        if self._early_interjects:
            import asyncio as _aio

            for _msg in self._early_interjects:
                try:
                    maybe_coro = new_handle.interject(_msg)  # type: ignore[attr-defined]
                    if _aio.iscoroutine(maybe_coro):
                        _aio.create_task(maybe_coro)
                except Exception:
                    # Advisory only – failure to replay should not break the flow.
                    pass

            self._early_interjects.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  A convenience wrapper that *starts* the loop and returns the handle
# ─────────────────────────────────────────────────────────────────────────────
def start_async_tool_use_loop(
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
) -> AsyncToolUseLoopHandle:
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
    interject_queue: asyncio.Queue[str] = asyncio.Queue()
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

    task = asyncio.create_task(
        async_tool_use_loop_inner(
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
        ),
        name="ToolUseLoop",
    )

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

    handle = AsyncToolUseLoopHandle(
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
