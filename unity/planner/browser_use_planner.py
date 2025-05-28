"""
# BrowserUsePlanner

An asyncio, single–task orchestrator that glues Browser-Use to the core async tool-use loop.

## Public API
| State     | Methods                                                                         |
|-----------|---------------------------------------------------------------------------------|
| **Idle**  | `await execute(task_description)`                                               |
| **Running** | `await stop()` · `await pause()` · `await interject(msg)` · `await query(q)` |
| **Paused** | `await resume()` · `await stop()`                                              |

All methods are async.

## How Pause/Resume works
1. `pause()` → `AsyncToolLoopHandle.stop()` sets the loop’s `cancel_event`.
2. We copy `client.chat_history` before the underlying task exits.
3. `resume()` spins up a brand-new loop with that history passed as
   `parent_chat_context`.

"""

from __future__ import annotations

import asyncio
import enum
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional
import os
import json

from browser_use.controller.service import Controller
from browser_use import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext
from unity.common.llm_helpers import (
    start_async_tool_use_loop,
    AsyncToolLoopHandle,
)
from unify import AsyncUnify
import unify
import uuid
from unity.events.event_bus import EventBus, Event
from unity.communication.transcript_manager.transcript_manager import TranscriptManager

__all__ = ["BrowserUsePlanner"]

logger = logging.getLogger(__name__)


class _PlannerState(enum.Enum):
    IDLE = enum.auto()
    RUNNING = enum.auto()
    PAUSED = enum.auto()


class BrowserUsePlanner:
    """
    A thin, coroutine-based orchestrator around Browser-Use that exposes the
    four UX controls(start/stop/pause/resume + interject + query) while
    delegating *all* reasoning to `async_tool_use_loop_inner`.

    ────────────────────────────────────────────────────────────────────────
    Implementation notes
    --------------------
    • One BrowserSession + one Controller are created up-front and reused.
      The Playwright process therefore survives across tasks, which keeps
      tests quick and results in fewer CAPTCHAs in the wild.

    • **No** extra thread pools or custom controllers – everything runs in
      the caller’s event-loop.

    • Pause/Resume is implemented by:
        – calling `AsyncToolLoopHandle.stop()` which sets the cancel_event
          consumed by the loop and lets it exit cleanly;
        – pulling the entire chat_history from the AsyncUnify client
          (exposed via a tiny helper we tuck into the client at instantiation);
        – feeding that history back in as `parent_chat_context` on resume.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        *,
        base_system_prompt: str | None = "You are a helpful web-browsing assistant.",
        headless: bool = True,
        disable_browser_security: bool = False,
    ) -> None:
        self._state: _PlannerState = _PlannerState.IDLE
        self._loop_handle: Optional[AsyncToolLoopHandle] = None
        self._paused_context: Optional[List[dict]] = None

        # ---- Browser layer -------------------------------------------------
        self._browser = Browser(
            config=BrowserConfig(
                disable_security=disable_browser_security,
                headless=headless,
            ),
        )
        self._browser_context = BrowserContext(browser=self._browser)
        self._controller = Controller()
        # ---- LLM layer -----------------------------------------------------
        self._base_system_prompt = base_system_prompt
        self._client = None

        # ---- Managers / event bus -----------------------------------------
        self._bus: EventBus = EventBus()
        self._tm: TranscriptManager = TranscriptManager(event_bus=self._bus)
        # Shared clarification channels (bubble-up / bubble-down)
        self._clar_up_q: asyncio.Queue[str] = asyncio.Queue()
        self._clar_down_q: asyncio.Queue[str] = asyncio.Queue()

        # Build the tool mapping exactly once – we reuse the same callables
        self._tools: Dict[str, Callable[..., Awaitable[str]]] = self._build_tools()

    # ------------------------------------------------------------ public API

    @property
    def status(self) -> str:
        """‘idle’ | ‘running’ | ‘paused’ – handy for UI bindings."""
        return self._state.name.lower()

    @property
    def clarification_questions(self) -> asyncio.Queue[str]:
        """
        Queue of pending bottom-up clarification questions.
        UI should:  `question = await planner.clarification_questions.get()`
        """
        return self._clar_up_q

    async def answer_clarification(self, answer: str) -> None:
        """Send the user’s answer back to the blocked tool-loop."""
        if self._state is not _PlannerState.RUNNING:
            raise RuntimeError("No running task awaiting clarification.")
        await self._clar_down_q.put(answer)

    async def execute(self, task_description: str) -> str:
        """Start a brand-new task (or replace a paused one)."""
        if self._state is _PlannerState.RUNNING:
            raise RuntimeError("A task is already running – stop or pause it first.")

        # If a task was paused, we implicitly abandon it when starting a new one
        self._paused_context = None
        self._task_id = str(uuid.uuid4())

        client = self._fresh_llm_client()
        handle = start_async_tool_use_loop(
            client=client,
            message=task_description,
            event_bus=self._bus,
            event_type="BROWSER",
            tools=self._tools,
            interrupt_llm_with_interjections=True,
            clarification_capable_tools=set(self._tools),
            clarification_up_q=self._clar_up_q,
            clarification_down_q=self._clar_down_q,
        )

        self._attach_completion_callback(handle)
        self._loop_handle = handle
        self._client = client
        self._set_state(_PlannerState.RUNNING)

        try:
            result: str = await handle.result()
        finally:
            # loop finished by itself
            self._loop_handle = None
            self._client = None
            self._set_state(_PlannerState.IDLE)
        return result

    async def stop(self) -> None:
        """Terminate the running loop (or discard a paused one) and reset state."""
        if self._state is _PlannerState.IDLE:
            return

        if self._loop_handle:
            self._loop_handle.stop()
            # Let the cancellation propagate
            await asyncio.sleep(0)

        # Reset everything
        self._loop_handle = None
        self._paused_context = None
        self._set_state(_PlannerState.IDLE)

    async def pause(self) -> None:
        """Gracefully cancel the loop but keep its full chat history for resume()."""
        if self._state is not _PlannerState.RUNNING or not self._loop_handle:
            raise RuntimeError("No running task to pause.")

        # 1. cancel
        self._loop_handle.stop()
        await self._loop_handle._task

        # 2. capture chat history  – deep-copy so we’re immune to later edits
        self._paused_context = list(self._client.messages)  # type: ignore[arg-type]
        self._loop_handle = None
        self._client = None
        self._set_state(_PlannerState.PAUSED)

    async def resume(self) -> None:
        """Restart a previously-paused task in-place."""
        if self._state is not _PlannerState.PAUSED or not self._paused_context:
            raise RuntimeError("No paused task to resume.")

        client = self._fresh_llm_client(parent_context=self._paused_context)
        handle = start_async_tool_use_loop(
            client=client,
            message="Continue.",
            tools=self._tools,
            parent_chat_context=self._paused_context,
            propagate_chat_context=True,
            interrupt_llm_with_interjections=True,
            event_bus=self._bus,
            event_type="BROWSER",
            clarification_capable_tools=set(self._tools),
            clarification_up_q=self._clar_up_q,
            clarification_down_q=self._clar_down_q,
        )

        self._attach_completion_callback(handle)
        self._loop_handle = handle
        self._client = client
        self._set_state(_PlannerState.RUNNING)
        self._paused_context = None

    async def interject(self, message: str) -> None:
        """Inject a user turn while the agent is thinking/acting."""
        if self._state is not _PlannerState.RUNNING or not self._loop_handle:
            raise RuntimeError("Can only interject while a task is running.")
        await self._loop_handle.interject(message)

    async def query(self, question: str) -> str:
        """
        Ask a question *during* execution.
        • if task **running** → inject + capture next assistant reply
        • if task **paused**  → answer via TranscriptManager.ask
        """
        if self._state is _PlannerState.PAUSED and self._tm:
            h = self._tm.ask(
                question,
                parent_chat_context=self._paused_context,
                clarification_up_q=self._clar_up_q,
                clarification_down_q=self._clar_down_q,
            )
            return await h.result()

        if self._state is not _PlannerState.RUNNING or not self._loop_handle:
            raise RuntimeError("No running task to query.")

        await self._loop_handle.interject(question)

        fut: asyncio.Future[str] = asyncio.get_event_loop().create_future()

        def _listener(msg: dict) -> None:
            if (
                msg.get("role") == "assistant"
                and not msg.get("function_call")
                and not fut.done()
            ):
                fut.set_result(msg["content"])

        unsubscribe = self._client.on("message", _listener)  # type: ignore[attr-defined]
        try:
            return await asyncio.wait_for(fut, timeout=30)
        finally:
            unsubscribe()

    # ---------------------------------------------------------------- tools
    def _build_tools(self) -> Dict[str, Callable[..., Awaitable[str]]]:
        """
        Dynamically wrap every callable in `Controller.registry` so it shows
        up as an LLM-callable function.  Each wrapper:

            • marshals **kwargs into the action's Pydantic model (if any)
            • awaits `controller.execute_action(...)`
            • returns the action's .extracted_content or a fallback string
        """

        tools: Dict[str, Callable[..., Awaitable[str]]] = {}

        from pydantic import BaseModel
        import inspect

        for action_name, action in self._controller.registry.registry.actions.items():

            param_model = getattr(action, "param_model", None)
            description = action.description or f"{action_name} browser action."

            # --- dynamic wrapper ------------------------------------------------
            async def _make_tool(
                *_,
                _action_name=action_name,
                _param_model=param_model,
                **kwargs: Any,
            ) -> str:
                try:
                    params: Any
                    if _param_model and issubclass(_param_model, BaseModel):
                        params = _param_model(**kwargs).model_dump()
                    else:
                        params = kwargs

                    # Build isolated “page extraction” LLM to avoid chat-pollution
                    if (
                        "page_extraction_llm"
                        in inspect.signature(
                            self._controller.registry.execute_action,
                        ).parameters
                    ):
                        extraction_llm = unify.AsyncUnify(
                            "o4-mini@openai",
                            cache=True,
                            traced=False,
                        )
                    else:
                        extraction_llm = None

                    result = await self._controller.registry.execute_action(
                        _action_name,
                        params,
                        browser=self._browser_context,
                        page_extraction_llm=extraction_llm,
                    )
                    return (
                        getattr(result, "extracted_content", None)
                        or getattr(result, "message", "")
                        or "DONE"
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Tool %s failed", _action_name)
                    return f"ERROR: {exc!s}"

            # ---- Make the wrapper look *exactly* like the action ---------------
            _make_tool.__name__ = action_name
            _make_tool.__doc__ = description

            # ---------- build inspect.Signature with required-first ------
            raw_fields = (
                param_model.model_fields  # pydantic-v2
                if hasattr(param_model, "model_fields")
                else param_model.__fields__  # pydantic-v1
            )

            required_params: list[inspect.Parameter] = []
            optional_params: list[inspect.Parameter] = []

            def _is_required(field) -> bool:
                # v1: .required bool ; v2: .is_required()
                return (
                    getattr(field, "required", None)
                    or getattr(field, "is_required", lambda: False)()
                )

            for fname, field in raw_fields.items():
                annotation = getattr(field, "annotation", Any)
                if _is_required(field):
                    param = inspect.Parameter(
                        fname,
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect._empty,
                        annotation=annotation,
                    )
                    required_params.append(param)
                else:
                    default_val = getattr(field, "default", None)
                    param = inspect.Parameter(
                        fname,
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=default_val,
                        annotation=annotation,
                    )
                    optional_params.append(param)

            parameters = required_params + optional_params
            try:
                _make_tool.__signature__ = inspect.Signature(
                    parameters,
                    return_annotation=str,
                )
            except Exception as e:
                logger.exception("Error building tool signature for %s", action_name)
                raise e

            tools[action_name] = _make_tool

        return tools

    # --------------------------------------------------------- helper utils
    def _fresh_llm_client(
        self,
        *,
        parent_context: Optional[List[dict]] = None,
    ) -> AsyncUnify:
        """
        Construct a brand-new AsyncUnify client and (optionally) preload it
        with previous messages so that pause/resume works.
        """
        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        if self._base_system_prompt:
            client.set_system_message(self._base_system_prompt)

        if parent_context:
            client.append_messages(parent_context)  # type: ignore[attr-defined]

        return client

    def _attach_completion_callback(self, handle: AsyncToolLoopHandle) -> None:
        """When the tool loop exits by itself, reset planner state."""

        async def _finalizer() -> None:
            try:
                await handle.result()
            except asyncio.CancelledError:
                # stop()/pause() will land here
                pass
            finally:
                if self._state is _PlannerState.RUNNING:
                    self._set_state(_PlannerState.IDLE)
                self._loop_handle = None
                self._client = None

        asyncio.create_task(_finalizer())

    # ----------------------------------------------------- async context mgr
    async def __aenter__(self) -> "BrowserUsePlanner":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        await self.stop()
        await self._browser_context.close()
        await self._browser.close()

    # --------------------------------------------------- private helpers

    def _set_state(self, new_state: "_PlannerState") -> None:
        self._state = new_state
        # broadcast for TaskListManager listeners
        asyncio.create_task(
            self._bus.publish(
                Event(
                    type="TASK_STATUS",
                    payload={
                        "task_id": getattr(self, "_task_id", "n/a"),
                        "status": self.status,
                    },
                ),
            ),
        )
