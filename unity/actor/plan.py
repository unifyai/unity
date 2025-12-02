import asyncio
import base64
import copy
import enum
import functools
import json
import logging
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional

import unify
from unify import AsyncUnify

from unity.common.async_tool_loop import (
    SteerableToolHandle,
    start_async_tool_loop,
)
from unity.common.llm_client import get_cache_setting
from unity.common.llm_helpers import _strip_image_keys
from unity.image_manager.types.annotated_image_ref import AnnotatedImageRef
from unity.image_manager.types.image_refs import ImageRefs
from unity.image_manager.types.raw_image_ref import RawImageRef

from ..task_scheduler.base import BaseActiveTask
from .base import BaseActorHandle

logger = logging.getLogger(__name__)


class _PlanState(enum.Enum):
    IDLE = enum.auto()
    RUNNING = enum.auto()
    PAUSED = enum.auto()
    COMPLETED = enum.auto()
    STOPPED = enum.auto()
    ERROR = enum.auto()


class Plan(BaseActiveTask, BaseActorHandle):
    """
    A steerable execution plan that runs an LLM-driven tool loop.

    This class manages the lifecycle of an async tool loop, providing
    pause/resume/stop/interject/ask capabilities for dynamic control.
    """

    MAX_STEPS = 100

    def __init__(
        self,
        task_description: str,
        tools: Dict[str, Callable[..., Awaitable[Any]]],
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        main_event_loop: Optional[asyncio.AbstractEventLoop] = None,
        timeout: Optional[float] = 1000,
        persist: bool = False,
        custom_system_prompt: str | None = None,
        tool_policy: Optional[Callable] = None,
        action_provider: Optional[Any] = None,
        images: Optional[ImageRefs | list[RawImageRef | AnnotatedImageRef]] = None,
    ):
        self._initial_task_description = task_description
        self._tools = tools
        self._parent_chat_context_on_pause: Optional[List[dict]] = parent_chat_context
        self._chat_history: List[Dict[str, Any]] = []
        self._custom_system_prompt = custom_system_prompt
        self._images = images

        self._clar_up_q_internal: asyncio.Queue[str] = (
            clarification_up_q or asyncio.Queue()
        )
        self._clar_down_q_internal: asyncio.Queue[str] = (
            clarification_down_q or asyncio.Queue()
        )

        self._state: _PlanState = _PlanState.IDLE
        self._loop_handle: Optional[SteerableToolHandle] = None
        self._result_str: Optional[str] = None
        self._error_str: Optional[str] = None

        self._overall_plan_completion_event = asyncio.Event()
        self._resume_requested_event = asyncio.Event()

        self._task_id = str(uuid.uuid4())
        self._main_event_loop = main_event_loop
        self._timeout = timeout
        self._persist = persist
        self._tool_policy = tool_policy
        self._action_provider = action_provider

        self._plan_client = AsyncUnify(
            "claude-4.5-sonnet@anthropic",
            cache=get_cache_setting(),
        )

        self._ask_client = unify.AsyncUnify(
            "claude-4.5-sonnet@anthropic",
            cache=get_cache_setting(),
        )

        if not self._main_event_loop:
            try:
                self._main_event_loop = asyncio.get_running_loop()
            except RuntimeError as e:
                logger.error(
                    f"Plan {self._task_id}: Could not get running event loop and none was provided: {e}",
                    exc_info=True,
                )
                self._state = _PlanState.ERROR
                self._error_str = f"Initialization failed: no event loop. {e}"
                self._overall_plan_completion_event.set()
                return

        logger.info(
            f"Plan {self._task_id}: Scheduling main execution manager on loop {self._main_event_loop}.",
        )
        asyncio.run_coroutine_threadsafe(
            self._manage_plan_execution(),
            self._main_event_loop,
        )

    @property
    def chat_history(self) -> List[Dict[str, Any]]:
        """Returns a copy of the internal chat history of the tool loop."""
        if self._loop_handle and self._loop_handle._client:
            return list(self._loop_handle._client.messages)
        return list(self._chat_history)

    def _get_internal_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        current_tools = self._tools.copy()

        async def request_clarification(question: str) -> str:
            """
            This tool is used to request clarification from the caller.
            """
            logger.info(
                f"Plan {self._task_id}: LLM (internal loop) requesting clarification: '{question}'",
            )
            await self._clar_up_q_internal.put(question)
            answer = await self._clar_down_q_internal.get()
            logger.info(
                f"Plan {self._task_id}: User (via plan) provided clarification: '{answer}'",
            )
            return answer

        request_clarification.__name__ = "request_clarification"
        request_clarification.__qualname__ = "request_clarification"
        current_tools["request_clarification"] = request_clarification
        return current_tools

    async def _manage_plan_execution(self):
        current_task_description = self._initial_task_description
        current_parent_chat_context = None
        self._state = _PlanState.IDLE

        try:
            while True:
                if self._state == _PlanState.STOPPED or self._state == _PlanState.ERROR:
                    logger.info(
                        f"Plan {self._task_id}: Execution manager exiting due to state {self._state.name}",
                    )
                    break

                self._state = _PlanState.RUNNING
                logger.info(
                    f"Plan {self._task_id}: Starting/Resuming internal loop with description: '{current_task_description}'",
                )

                self._plan_client.reset_messages()
                self._plan_client.reset_system_message()

                if self._custom_system_prompt:
                    self._plan_client.set_system_message(self._custom_system_prompt)

                if current_parent_chat_context:
                    self._plan_client.append_messages(current_parent_chat_context)

                current_parent_chat_context = None

                internal_tools = self._get_internal_tools()
                self._loop_handle = start_async_tool_loop(
                    client=self._plan_client,
                    message=current_task_description,
                    tools=internal_tools,
                    loop_id=f"{self.__class__.__name__}.{self._manage_plan_execution.__name__}",
                    propagate_chat_context=True,
                    interrupt_llm_with_interjections=True,
                    log_steps=True,
                    max_steps=self.MAX_STEPS,
                    timeout=self._timeout,
                    tool_policy=self._tool_policy,
                    images=self._images,
                )

                try:
                    loop_result_str = await self._loop_handle.result()
                    if self._state == _PlanState.RUNNING:
                        self._state = _PlanState.COMPLETED
                        self._result_str = loop_result_str
                        logger.info(
                            f"Plan {self._task_id}: Internal loop COMPLETED. Result: {self._result_str}",
                        )
                    elif self._state == _PlanState.PAUSED:
                        logger.info(
                            f"Plan {self._task_id}: Internal loop stopped for PAUSE.",
                        )
                    elif self._state == _PlanState.STOPPED:
                        logger.info(
                            f"Plan {self._task_id}: Internal loop stopped for STOP.",
                        )
                        if self._result_str is None:
                            self._result_str = f"Plan {self._task_id} was stopped."
                except asyncio.CancelledError:
                    logger.info(
                        f"Plan {self._task_id}: Internal loop task was cancelled. Current state: {self._state.name}",
                    )
                    if self._state == _PlanState.RUNNING:
                        self._state = _PlanState.STOPPED
                    if self._result_str is None:
                        self._result_str = f"Plan {self._task_id} was {self._state.name.lower()} (cancelled)."
                except Exception as e:
                    logger.error(
                        f"Plan {self._task_id}: Internal loop failed: {e}",
                        exc_info=True,
                    )
                    self._state = _PlanState.ERROR
                    self._error_str = str(e)
                    self._result_str = f"Task failed with error: {self._error_str}"

                if self._loop_handle and self._loop_handle._client:
                    self._chat_history = list(self._loop_handle._client.messages)

                self._loop_handle = None

                if self._state == _PlanState.PAUSED:
                    logger.info(
                        f"Plan {self._task_id}: Execution PAUSED, awaiting resume signal.",
                    )
                    await self._resume_requested_event.wait()
                    self._resume_requested_event.clear()
                    if self._state == _PlanState.STOPPED:
                        logger.info(
                            f"Plan {self._task_id}: Stop called while paused. Terminating.",
                        )
                        break
                    logger.info(f"Plan {self._task_id}: RESUMING execution.")
                    current_task_description = "The task was paused and is now resumed. Please review the history and continue."
                    current_parent_chat_context = self._parent_chat_context_on_pause
                    self._parent_chat_context_on_pause = None
                    continue
                else:
                    logger.info(
                        f"Plan {self._task_id}: Execution ended with state {self._state.name}. Finalizing.",
                    )
                    break
        except Exception as e:
            logger.error(
                f"Plan {self._task_id}: Unexpected error in _manage_plan_execution: {e}",
                exc_info=True,
            )
            if self._state not in [
                _PlanState.ERROR,
                _PlanState.COMPLETED,
                _PlanState.STOPPED,
            ]:
                self._state = _PlanState.ERROR
            if self._error_str is None:
                self._error_str = str(e)
            if self._result_str is None:
                self._result_str = (
                    f"Plan failed with unexpected error: {self._error_str}"
                )
        finally:
            logger.info(
                f"Plan {self._task_id}: Setting overall completion event. Final state: {self._state.name}",
            )
            self._overall_plan_completion_event.set()

    @functools.wraps(BaseActiveTask.result, updated=())
    async def result(self) -> str:
        await self._overall_plan_completion_event.wait()
        if self._error_str:
            return f"Error: {self._error_str}"
        return (
            self._result_str
            if self._result_str is not None
            else f"Plan {self._task_id} concluded without a specific result (State: {self._state.name})."
        )

    @functools.wraps(BaseActiveTask.done, updated=())
    def done(self) -> bool:
        return self._overall_plan_completion_event.is_set()

    async def next_clarification(self) -> dict:
        """Await the next clarification question from the running internal loop."""
        question = await self._clar_up_q_internal.get()
        return {"question": question}

    async def next_notification(self) -> dict:
        """Await the next notification (not supported for Plan; waits indefinitely)."""
        await asyncio.Event().wait()
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        """Provide an answer to the pending clarification (call_id is ignored)."""
        await self._clar_down_q_internal.put(answer)

    def get_history(self) -> list[dict]:
        """Return the user-visible conversation history of the inner loop."""
        return list(self.chat_history)

    @property
    def clarification_up_q(self) -> asyncio.Queue[str]:
        """Queue for this plan to send clarification questions upwards."""
        return self._clar_up_q_internal

    @property
    def clarification_down_q(self) -> asyncio.Queue[str]:
        return self._clar_down_q_internal

    def _is_valid_method(self, name: str) -> bool:
        if name == "stop":
            return self._state in (
                _PlanState.RUNNING,
                _PlanState.PAUSED,
                _PlanState.IDLE,
            )
        if name == "pause":
            return self._state == _PlanState.RUNNING
        if name == "resume":
            return self._state == _PlanState.PAUSED
        if name == "interject":
            return self._state in (
                _PlanState.RUNNING,
                _PlanState.PAUSED,
                _PlanState.IDLE,
            )
        if name == "ask":
            return self._state in (_PlanState.RUNNING, _PlanState.PAUSED)
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
                f"Plan {self._task_id} cannot be stopped in state {self._state.name}.",
            )

        logger.info(
            f"Plan {self._task_id}: Stopping. Current state: {self._state.name}",
        )
        previous_state = self._state
        self._state = _PlanState.STOPPED
        self._result_str = (
            f"Plan {self._task_id} was stopped."
            if not reason
            else f"Plan {self._task_id} was stopped: {reason}"
        )

        if previous_state == _PlanState.PAUSED:
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
            previous_state == _PlanState.IDLE
            and not self._overall_plan_completion_event.is_set()
        ):
            logger.warning(
                f"Plan {self._task_id}: Stop called in IDLE state. Forcing overall completion.",
            )
            self._overall_plan_completion_event.set()

        await self._overall_plan_completion_event.wait()
        return self._result_str

    @functools.wraps(BaseActiveTask.pause, updated=())
    async def pause(self) -> str:
        if not self._is_valid_method("pause"):
            raise RuntimeError(
                f"Plan {self._task_id} cannot be paused in state {self._state.name}.",
            )
        logger.info(
            f"Plan {self._task_id}: Pausing. Current state: {self._state.name}",
        )
        self._state = _PlanState.PAUSED

        if self._plan_client and self._plan_client.messages:
            self._parent_chat_context_on_pause = copy.deepcopy(
                self._plan_client.messages,
            )
            logger.info(
                f"Plan {self._task_id}: Context saved on pause: {len(self._parent_chat_context_on_pause)} messages.",
            )
        else:
            self._parent_chat_context_on_pause = []
            logger.info(
                f"Plan {self._task_id}: No active LLM context to save on pause.",
            )

        if self._loop_handle and not self._loop_handle.done():
            logger.info(
                f"Plan {self._task_id}: Requesting stop of current internal loop for pause.",
            )
            self._loop_handle.stop()
        else:
            logger.warning(
                f"Plan {self._task_id}: Pause called but no active internal loop_handle to stop.",
            )

        return f"Plan {self._task_id} paused successfully. Awaiting resume."

    @functools.wraps(BaseActiveTask.resume, updated=())
    async def resume(self) -> str:
        if not self._is_valid_method("resume"):
            raise RuntimeError(
                f"Plan {self._task_id} cannot be resumed in state {self._state.name}.",
            )
        logger.info(
            f"Plan {self._task_id}: Requesting resume. Current state: {self._state.name}",
        )
        self._resume_requested_event.set()
        return f"Plan {self._task_id} is resuming."

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
                return f"Error: Plan {self._task_id} is already done, cannot interject."
            return f"Error: Plan {self._task_id} is in state {self._state.name}, cannot interject."

        if not self._loop_handle:
            logger.info(
                f"Plan {self._task_id}: Interject called before loop handle was created. Waiting briefly.",
            )
            for _ in range(5):
                if self._loop_handle:
                    break
                await asyncio.sleep(1)

            if not self._loop_handle:
                return f"Error: Plan {self._task_id} did not initialize in time for interjection."

        logger.info(
            f"Plan {self._task_id}: Interjecting message: '{message}' into active internal loop.",
        )
        try:
            await self._loop_handle.interject(
                message=message,
                parent_chat_context_cont=parent_chat_context_cont,
                images=images,
            )
        except TypeError:
            await self._loop_handle.interject(message)
        return f"Interjection '{message}' sent to plan {self._task_id}."

    @functools.wraps(BaseActiveTask.ask, updated=())
    async def ask(self, question: str) -> SteerableToolHandle:
        """
        Asks a question about the current state of the plan by creating a new,
        isolated tool loop that returns a handle to its result.
        """
        if not self._is_valid_method("ask"):
            raise RuntimeError(
                f"Cannot ask question for plan {self._task_id} in state {self._state.name}.",
            )

        logger.info(f"Plan {self._task_id}: Answering query: '{question}'")
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
        if self._action_provider and self._action_provider.browser:
            try:
                screenshot = await self._action_provider.browser.get_screenshot()
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


# Backwards compatibility alias
ToolLoopPlan = Plan
