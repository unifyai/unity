import asyncio
import base64
import copy
import contextvars
import enum
import functools
import json
import logging
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type

from pydantic import BaseModel

from unity.common.async_tool_loop import (
    ChatContextPropagation,
    SteerableToolHandle,
    start_async_tool_loop,
)
from unity.common.clarification_tools import add_clarification_tool_with_events
from unity.common.llm_client import new_llm_client
from unity.common.llm_helpers import _strip_image_keys
from unity.image_manager.types.annotated_image_ref import AnnotatedImageRef
from unity.image_manager.types.image_refs import ImageRefs
from unity.image_manager.types.raw_image_ref import RawImageRef

from ..task_scheduler.base import BaseActiveTask
from .base import BaseActorHandle

logger = logging.getLogger(__name__)


class _HandleState(enum.Enum):
    IDLE = enum.auto()
    RUNNING = enum.auto()
    PAUSED = enum.auto()
    COMPLETED = enum.auto()
    STOPPED = enum.auto()
    ERROR = enum.auto()


class ActorHandle(BaseActiveTask, BaseActorHandle):
    """
    A steerable handle for an actor's execution, running an LLM-driven tool loop.

    This class manages the lifecycle of an async tool loop, providing
    pause/resume/stop/interject/ask capabilities for dynamic control.
    """

    MAX_STEPS = 100

    def __init__(
        self,
        task_description: str | dict | list[str | dict],
        tools: Dict[str, Callable[..., Awaitable[Any]]],
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        call_id: Optional[str] = None,
        on_finally: Optional[Callable[[], Awaitable[None]]] = None,
        main_event_loop: Optional[asyncio.AbstractEventLoop] = None,
        timeout: Optional[float] = 1000,
        persist: bool = False,
        custom_system_prompt: str | None = None,
        tool_policy: Optional[Callable] = None,
        computer_primitives: Optional[Any] = None,
        images: Optional[ImageRefs | list[RawImageRef | AnnotatedImageRef]] = None,
        response_format: Optional[Type[BaseModel]] = None,
    ):
        self._initial_task_description = task_description
        self._tools = tools
        # Preserve caller-provided context for the *initial* loop start.
        self._initial_parent_chat_context: Optional[List[dict]] = (
            list(parent_chat_context) if parent_chat_context else None
        )
        # Separate storage for pause/resume context snapshots.
        self._parent_chat_context_on_pause: Optional[List[dict]] = None
        self._chat_history: List[Dict[str, Any]] = []
        self._custom_system_prompt = custom_system_prompt
        self._images = images
        self._response_format = response_format
        self._on_finally = on_finally

        # Clarification queues are optional. When missing, the internal tool loop must not
        # attempt to request clarification (the `request_clarification` tool will not be added).
        self._clar_up_q_internal: Optional[asyncio.Queue[str]] = clarification_up_q
        self._clar_down_q_internal: Optional[asyncio.Queue[str]] = clarification_down_q
        self._call_id: Optional[str] = call_id

        self._state: _HandleState = _HandleState.IDLE
        self._loop_handle: Optional[SteerableToolHandle] = None
        self._result_str: Any = None
        self._error_str: Optional[str] = None

        self._completion_event = asyncio.Event()
        self._resume_requested_event = asyncio.Event()
        # Event signaling that _loop_handle is ready for delegation
        self._loop_handle_ready = asyncio.Event()

        self._task_id = str(uuid.uuid4())
        self._main_event_loop = main_event_loop
        self._timeout = timeout
        self._persist = persist
        self._tool_policy = tool_policy
        self._computer_primitives = computer_primitives

        self._client = new_llm_client(
            "claude-4.5-opus@anthropic",
            reasoning_effort=None,
            service_tier=None,
        )
        self._ask_client = new_llm_client(
            "claude-4.5-opus@anthropic",
            reasoning_effort=None,
            service_tier=None,
        )

        if not self._main_event_loop:
            try:
                self._main_event_loop = asyncio.get_running_loop()
            except RuntimeError as e:
                logger.error(
                    f"Handle {self._task_id}: Could not get running event loop: {e}",
                    exc_info=True,
                )
                self._state = _HandleState.ERROR
                self._error_str = f"Initialization failed: no event loop. {e}"
                self._completion_event.set()
                return

        logger.info(
            f"Handle {self._task_id}: Scheduling execution on loop {self._main_event_loop}.",
        )
        # Preserve caller contextvars (e.g. per-request sandbox binding for CodeActActor).
        ctx = contextvars.copy_context()
        asyncio.run_coroutine_threadsafe(
            ctx.run(self._manage_execution),
            self._main_event_loop,
        )

    @property
    def chat_history(self) -> List[Dict[str, Any]]:
        """Returns a copy of the internal chat history of the tool loop."""
        if self._loop_handle and self._loop_handle._client:
            return list(self._loop_handle._client.messages)
        return list(self._chat_history)

    @property
    def _pause_event(self):
        """Proxy for pause state compatibility with get_handle_paused_state().

        Returns an object with is_set() that follows the async tool loop convention:
        - is_set() == True → running (not paused)
        - is_set() == False → paused

        This allows ActorHandle to work with the shared get_handle_paused_state()
        helper without special-casing, ensuring consistent pause/resume tool
        availability across all handle types.
        """

        class _PauseStateProxy:
            """Minimal proxy exposing is_set() based on ActorHandle._state."""

            def __init__(proxy_self, handle: "ActorHandle"):  # noqa: N805
                proxy_self._handle = handle

            def is_set(proxy_self) -> bool:  # noqa: N805
                # Event set = running; Event cleared = paused
                return proxy_self._handle._state != _HandleState.PAUSED

        return _PauseStateProxy(self)

    def _get_internal_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        current_tools = self._tools.copy()

        # Only add the `request_clarification` tool when queues are supplied.
        if (
            self._clar_up_q_internal is not None
            and self._clar_down_q_internal is not None
        ):
            add_clarification_tool_with_events(
                current_tools,
                self._clar_up_q_internal,
                self._clar_down_q_internal,
                manager="CodeActActor",
                method="act",
                call_id=self._call_id,
            )
        return current_tools

    async def _manage_execution(self):
        current_task_description = self._initial_task_description
        current_parent_chat_context = self._initial_parent_chat_context
        self._initial_parent_chat_context = None
        self._state = _HandleState.IDLE

        try:
            while True:
                if (
                    self._state == _HandleState.STOPPED
                    or self._state == _HandleState.ERROR
                ):
                    logger.info(
                        f"Handle {self._task_id}: Exiting due to state {self._state.name}",
                    )
                    break

                self._state = _HandleState.RUNNING
                logger.info(
                    f"Handle {self._task_id}: Starting/Resuming with: {current_task_description!r}",
                )

                self._client.reset_messages()
                self._client.reset_system_message()

                if self._custom_system_prompt:
                    self._client.set_system_message(self._custom_system_prompt)

                if current_parent_chat_context:
                    self._client.append_messages(current_parent_chat_context)

                current_parent_chat_context = None

                internal_tools = self._get_internal_tools()
                self._loop_handle = start_async_tool_loop(
                    client=self._client,
                    message=current_task_description,
                    tools=internal_tools,
                    loop_id=f"{self.__class__.__name__}.{self._manage_execution.__name__}",
                    propagate_chat_context=ChatContextPropagation.ALWAYS,
                    interrupt_llm_with_interjections=True,
                    log_steps=True,
                    max_steps=self.MAX_STEPS,
                    timeout=self._timeout,
                    tool_policy=self._tool_policy,
                    images=self._images,
                    response_format=self._response_format,
                    persist=bool(self._persist),
                )
                # Signal that _loop_handle is ready for delegation (e.g., next_notification)
                self._loop_handle_ready.set()

                try:
                    loop_result_str = await self._loop_handle.result()
                    # If a response_format was requested, try to coerce the loop output into it.
                    # The underlying tool loop often returns a JSON string (e.g. '{"answer": 123}').
                    if (
                        self._response_format is not None
                        and loop_result_str is not None
                    ):
                        try:
                            if isinstance(loop_result_str, self._response_format):
                                loop_result_str = loop_result_str
                            elif isinstance(loop_result_str, str):
                                loop_result_str = (
                                    self._response_format.model_validate_json(
                                        loop_result_str,
                                    )
                                )
                            elif isinstance(loop_result_str, dict):
                                loop_result_str = self._response_format.model_validate(
                                    loop_result_str,
                                )
                        except Exception:
                            # Best-effort only: if parsing fails, fall back to raw output.
                            pass
                    if self._state == _HandleState.RUNNING:
                        self._state = _HandleState.COMPLETED
                        self._result_str = loop_result_str
                        logger.info(
                            f"Handle {self._task_id}: COMPLETED. Result: {self._result_str}",
                        )
                    elif self._state == _HandleState.PAUSED:
                        logger.info(
                            f"Handle {self._task_id}: Stopped for PAUSE.",
                        )
                    elif self._state == _HandleState.STOPPED:
                        logger.info(
                            f"Handle {self._task_id}: Stopped for STOP.",
                        )
                        if self._result_str is None:
                            self._result_str = f"Handle {self._task_id} was stopped."
                except asyncio.CancelledError:
                    logger.info(
                        f"Handle {self._task_id}: Cancelled. State: {self._state.name}",
                    )
                    if self._state == _HandleState.RUNNING:
                        self._state = _HandleState.STOPPED
                    if self._result_str is None:
                        self._result_str = f"Handle {self._task_id} was {self._state.name.lower()} (cancelled)."
                except Exception as e:
                    logger.error(
                        f"Handle {self._task_id}: Failed: {e}",
                        exc_info=True,
                    )
                    self._state = _HandleState.ERROR
                    self._error_str = str(e)
                    self._result_str = f"Task failed with error: {self._error_str}"

                if self._loop_handle and self._loop_handle._client:
                    self._chat_history = list(self._loop_handle._client.messages)

                self._loop_handle = None

                if self._state == _HandleState.PAUSED:
                    logger.info(
                        f"Handle {self._task_id}: PAUSED, awaiting resume.",
                    )
                    await self._resume_requested_event.wait()
                    self._resume_requested_event.clear()
                    if self._state == _HandleState.STOPPED:
                        logger.info(
                            f"Handle {self._task_id}: Stop called while paused. Terminating.",
                        )
                        break
                    logger.info(f"Handle {self._task_id}: RESUMING.")
                    current_task_description = "The task was paused and is now resumed. Please review the history and continue."
                    current_parent_chat_context = self._parent_chat_context_on_pause
                    self._parent_chat_context_on_pause = None
                    continue
                else:
                    logger.info(
                        f"Handle {self._task_id}: Ended with state {self._state.name}.",
                    )
                    break
        except Exception as e:
            logger.error(
                f"Handle {self._task_id}: Unexpected error: {e}",
                exc_info=True,
            )
            if self._state not in [
                _HandleState.ERROR,
                _HandleState.COMPLETED,
                _HandleState.STOPPED,
            ]:
                self._state = _HandleState.ERROR
            if self._error_str is None:
                self._error_str = str(e)
            if self._result_str is None:
                self._result_str = f"Failed with unexpected error: {self._error_str}"
        finally:
            if self._on_finally is not None:
                try:
                    await self._on_finally()
                except Exception as e:
                    logger.warning(
                        f"Handle {self._task_id}: on_finally callback failed: {e}",
                        exc_info=True,
                    )
            logger.info(
                f"Handle {self._task_id}: Completion event set. Final state: {self._state.name}",
            )
            self._completion_event.set()

    @functools.wraps(BaseActiveTask.result, updated=())
    async def result(self) -> Any:
        await self._completion_event.wait()
        if self._error_str:
            return f"Error: {self._error_str}"
        return (
            self._result_str
            if self._result_str is not None
            else f"Handle {self._task_id} concluded without a result (State: {self._state.name})."
        )

    @functools.wraps(BaseActiveTask.done, updated=())
    def done(self) -> bool:
        return self._completion_event.is_set()

    async def next_clarification(self) -> dict:
        """Await the next clarification question from the running internal loop."""
        if self._clar_up_q_internal is None:
            raise RuntimeError("Clarification is disabled for this handle.")
        question = await self._clar_up_q_internal.get()
        return {"question": question}

    async def next_notification(self) -> dict:
        """Await the next notification from the inner tool loop.

        Notifications bubble up from tools (like execute_code) through the async
        tool loop's internal notification queue. This method delegates to the
        inner loop handle once it's ready.
        """
        # Wait for the inner loop handle to be ready
        await self._loop_handle_ready.wait()
        if self._loop_handle is None:
            raise RuntimeError("Inner loop handle not available.")
        return await self._loop_handle.next_notification()

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        """Provide an answer to the pending clarification (call_id is ignored)."""
        if self._clar_down_q_internal is None:
            raise RuntimeError("Clarification is disabled for this handle.")
        await self._clar_down_q_internal.put(answer)

    def get_history(self) -> list[dict]:
        """Return the user-visible conversation history of the inner loop."""
        return list(self.chat_history)

    @property
    def clarification_up_q(self) -> Optional[asyncio.Queue[str]]:
        """Queue for sending clarification questions upwards."""
        return self._clar_up_q_internal

    @property
    def clarification_down_q(self) -> Optional[asyncio.Queue[str]]:
        return self._clar_down_q_internal

    @property
    def notification_up_q(self) -> Optional[asyncio.Queue[dict]]:
        """Queue for sending notifications upwards (delegated to inner loop)."""
        if self._loop_handle is not None:
            return getattr(self._loop_handle, "_notification_q", None)
        return None

    def _is_valid_method(self, name: str) -> bool:
        if name == "stop":
            return self._state in (
                _HandleState.RUNNING,
                _HandleState.PAUSED,
                _HandleState.IDLE,
            )
        if name == "pause":
            return self._state == _HandleState.RUNNING
        if name == "resume":
            return self._state == _HandleState.PAUSED
        if name == "interject":
            return self._state in (
                _HandleState.RUNNING,
                _HandleState.PAUSED,
                _HandleState.IDLE,
            )
        if name == "ask":
            return self._state in (_HandleState.RUNNING, _HandleState.PAUSED)
        return False

    @functools.wraps(BaseActiveTask.stop, updated=())
    async def stop(
        self,
        reason: Optional[str] = None,
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ) -> str:
        if not self._is_valid_method("stop"):
            if self.done():
                return await self.result()
            raise RuntimeError(
                f"Handle {self._task_id} cannot be stopped in state {self._state.name}.",
            )

        logger.info(
            f"Handle {self._task_id}: Stopping. State: {self._state.name}",
        )
        previous_state = self._state
        self._state = _HandleState.STOPPED
        self._result_str = (
            f"Handle {self._task_id} was stopped."
            if not reason
            else f"Handle {self._task_id} was stopped: {reason}"
        )

        if previous_state == _HandleState.PAUSED:
            self._resume_requested_event.set()

        if self._loop_handle and not self._loop_handle.done():
            try:
                self._loop_handle.stop(
                    reason,
                    parent_chat_context_cont=parent_chat_context_cont,
                )
            except Exception:
                self._loop_handle.stop(reason)
        elif (
            previous_state == _HandleState.IDLE and not self._completion_event.is_set()
        ):
            logger.warning(
                f"Handle {self._task_id}: Stop called in IDLE state. Forcing completion.",
            )
            self._completion_event.set()

        await self._completion_event.wait()
        return self._result_str

    @functools.wraps(BaseActiveTask.pause, updated=())
    async def pause(self) -> str:
        if not self._is_valid_method("pause"):
            raise RuntimeError(
                f"Handle {self._task_id} cannot be paused in state {self._state.name}.",
            )
        logger.info(
            f"Handle {self._task_id}: Pausing. State: {self._state.name}",
        )
        self._state = _HandleState.PAUSED

        if self._client and self._client.messages:
            self._parent_chat_context_on_pause = copy.deepcopy(
                self._client.messages,
            )
            logger.info(
                f"Handle {self._task_id}: Context saved: {len(self._parent_chat_context_on_pause)} messages.",
            )
        else:
            self._parent_chat_context_on_pause = []
            logger.info(
                f"Handle {self._task_id}: No context to save on pause.",
            )

        if self._loop_handle and not self._loop_handle.done():
            logger.info(
                f"Handle {self._task_id}: Stopping internal loop for pause.",
            )
            self._loop_handle.stop()
        else:
            logger.warning(
                f"Handle {self._task_id}: Pause called but no active loop to stop.",
            )

        return f"Handle {self._task_id} paused successfully."

    @functools.wraps(BaseActiveTask.resume, updated=())
    async def resume(self) -> str:
        if not self._is_valid_method("resume"):
            raise RuntimeError(
                f"Handle {self._task_id} cannot be resumed in state {self._state.name}.",
            )
        logger.info(
            f"Handle {self._task_id}: Resuming. State: {self._state.name}",
        )
        self._resume_requested_event.set()
        return f"Handle {self._task_id} is resuming."

    @functools.wraps(BaseActiveTask.interject, updated=())
    async def interject(
        self,
        message: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
        images: list | None = None,
    ) -> str:
        if not self._is_valid_method("interject"):
            if self.done():
                return f"Error: Handle {self._task_id} is done, cannot interject."
            return f"Error: Handle {self._task_id} is in state {self._state.name}, cannot interject."

        if not self._loop_handle:
            logger.info(
                f"Handle {self._task_id}: Interject called before loop ready. Waiting...",
            )
            for _ in range(5):
                if self._loop_handle:
                    break
                await asyncio.sleep(1)

            if not self._loop_handle:
                return f"Error: Handle {self._task_id} did not initialize in time."

        logger.info(
            f"Handle {self._task_id}: Interjecting: '{message}'",
        )
        try:
            await self._loop_handle.interject(
                message=message,
                parent_chat_context_cont=parent_chat_context_cont,
                images=images,
            )
        except TypeError:
            await self._loop_handle.interject(message)
        return f"Interjection sent to handle {self._task_id}."

    @functools.wraps(BaseActiveTask.ask, updated=())
    async def ask(self, question: str) -> SteerableToolHandle:
        """
        Asks a question about the current state by creating an isolated tool loop.
        """
        if not self._is_valid_method("ask"):
            raise RuntimeError(
                f"Cannot ask question for handle {self._task_id} in state {self._state.name}.",
            )

        logger.info(f"Handle {self._task_id}: Answering query: '{question}'")
        current_context_to_share = _strip_image_keys(copy.deepcopy(self.chat_history))
        self._ask_client.reset_messages()
        self._ask_client.reset_system_message()

        system_message = f"""
        You are an AI assistant in the middle of performing a task. The user has just asked a question.
        Based on the provided context (the task history and a screenshot of your browser), give a brief, natural, first-person response.
        Speak as if you are the one doing the work (e.g., "I'm currently looking for...").
        **Task History:**
        The task's history up to this point has been shared with you.
        **User Question:** "{question}"
        **Answer:**
        """

        messages_to_send = [
            {
                "role": "system",
                "content": f"--- Main Task History ---\n{json.dumps(current_context_to_share, indent=2)}",
            },
        ]
        if self._computer_primitives and self._computer_primitives.computer:
            try:
                screenshot = await self._computer_primitives.computer.get_screenshot()
                if isinstance(screenshot, str):
                    screenshot_b64 = screenshot
                else:
                    screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")

                system_message += (
                    "\n**Current Browser View (Screenshot):**\n"
                    "An image of the current browser page has also been provided."
                )
                messages_to_send.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{screenshot_b64}",
                                },
                            },
                        ],
                    },
                )
            except Exception as e:
                logger.warning(f"Could not get screenshot for ask(): {e}")

        self._ask_client.set_system_message(system_message)
        self._ask_client.append_messages(messages_to_send)

        return start_async_tool_loop(
            client=self._ask_client,
            message=question,
            tools={},
            loop_id=f"Question({self._task_id})",
            max_consecutive_failures=1,
            timeout=60,
        )


# Backwards compatibility aliases
Plan = ActorHandle
ToolLoopPlan = ActorHandle
