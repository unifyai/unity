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
import copy
from browser_use.controller.service import Controller
from browser_use import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext
from unity.common.llm_helpers import (
    start_async_tool_use_loop,
    SteerableToolHandle,
)
from unify import AsyncUnify
import unify
import uuid

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
        – pulling the entire chat history from the AsyncUnify client
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
        self._loop_handle: Optional[SteerableToolHandle] = None
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
        self._extraction_llm = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )

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
            tools=self._tools,
            interrupt_llm_with_interjections=True,
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
        self._paused_context = copy.deepcopy(self._client.messages)  # type: ignore[arg-type]
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
        """

        if self._state is not _PlannerState.RUNNING or not self._loop_handle:
            raise RuntimeError("No running task to query.")

        # Snapshot the current length so we only examine *new* messages
        start_idx = len(self._client.messages)
        await self._loop_handle.interject(question)

        async def _wait_for_reply() -> str:
            while True:
                # Scan any messages added since the interjection
                for msg in self._client.messages[start_idx:]:
                    if msg.get("role") == "assistant" and not msg.get("function_call"):
                        return msg["content"]
                await asyncio.sleep(0.1)  # yield control; keep loop responsive

        return await asyncio.wait_for(_wait_for_reply(), timeout=30)

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
                        self._extraction_llm.reset_messages()  # clear the history

                    result = await self._controller.registry.execute_action(
                        _action_name,
                        params,
                        browser=self._browser_context,
                        page_extraction_llm=self._extraction_llm,
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

        async def request_clarification_tool(question: str) -> str:
            await self._clar_up_q.put(question)
            answer = await self._clar_down_q.get()
            return answer

        request_clarification_tool.__name__ = "request_clarification_tool"
        request_clarification_tool.__doc__ = (
            "Use this tool to ask the end-user a clarifying question if you need more "
            "information to proceed with the task. The user's response will be returned."
        )
        param = inspect.Parameter(
            "question",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=str,
            default=inspect.Parameter.empty,
        )
        request_clarification_tool.__signature__ = inspect.Signature(
            parameters=[param],
            return_annotation=str,
        )

        tools["request_clarification_tool"] = request_clarification_tool
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

    def _attach_completion_callback(self, handle: SteerableToolHandle) -> None:
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

    def _set_state(self, new_state: "_PlannerState") -> None:
        self._state = new_state
