import asyncio
import enum
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type
import functools
import os
import json
import copy
import uuid

from unity.common.llm_helpers import (
    start_async_tool_use_loop,
    SteerableToolHandle,
)
from .base import BasePlan, BasePlanner
from unity.controller.controller import Controller
from unify import AsyncUnify
import unify

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


# Dummy ComsManager (will be replaced with the real ComsManager)
class ComsManager:
    async def communicate(self, description: str) -> str:
        logger.info(f"Dummy ComsManager.communicate called with: {description}")
        await asyncio.sleep(0.1)  # Simulate async work
        return f"Communication task initiated for: {description}"

    async def stop(self):
        logger.info("Dummy ComsManager: Stopping")


class _PlanState(enum.Enum):
    IDLE = enum.auto()
    RUNNING = enum.auto()
    PAUSED = enum.auto()
    COMPLETED = enum.auto()
    STOPPED = enum.auto()
    ERROR = enum.auto()


class ToolLoopPlan(BasePlan):
    """
    Represents an active plan being executed by the ToolLoopPlanner.
    Inherits from SteerableToolHandle to provide a consistent interface for interaction.
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
    ):
        self._initial_task_description = task_description

        self._tools = tools
        self._parent_chat_context_on_pause: Optional[List[dict]] = parent_chat_context

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

        self._overall_plan_completion_event = asyncio.Event()  # For the final result()
        self._resume_requested_event = asyncio.Event()  # To signal resume

        self._task_id = str(uuid.uuid4())
        self._main_event_loop = main_event_loop

        self._plan_client = AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )

        self._ask_client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )

        if not self._main_event_loop:
            try:
                self._main_event_loop = asyncio.get_running_loop()
            except RuntimeError as e:
                logger.error(
                    f"ToolLoopPlan {self._task_id}: Could not get running event loop and none was provided: {e}",
                    exc_info=True,
                )
                self._state = _PlanState.ERROR
                self._error_str = f"Initialization failed: no event loop. {e}"
                self._overall_plan_completion_event.set()
                return

        logger.info(
            f"ToolLoopPlan {self._task_id}: Scheduling main execution manager on loop {self._main_event_loop}.",
        )
        asyncio.run_coroutine_threadsafe(
            self._manage_plan_execution(),
            self._main_event_loop,
        )

    def _get_internal_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        current_tools = self._tools.copy()

        async def request_clarification_tool_for_llm(question: str) -> str:
            logger.info(
                f"ToolLoopPlan {self._task_id}: LLM (internal loop) requesting clarification: '{question}'",
            )
            await self._clar_up_q_internal.put(question)
            answer = await self._clar_down_q_internal.get()
            logger.info(
                f"ToolLoopPlan {self._task_id}: User (via plan) provided clarification: '{answer}'",
            )
            return answer

        request_clarification_tool_for_llm.__name__ = (
            "request_clarification_from_plan_caller"
        )
        request_clarification_tool_for_llm.__qualname__ = (
            "request_clarification_from_plan_caller"
        )
        current_tools["request_clarification_from_plan_caller"] = (
            request_clarification_tool_for_llm
        )
        return current_tools

    async def _manage_plan_execution(self):
        current_task_description = self._initial_task_description
        current_parent_chat_context = None
        self._state = _PlanState.IDLE

        try:
            while True:
                if self._state == _PlanState.STOPPED or self._state == _PlanState.ERROR:
                    logger.info(
                        f"ToolLoopPlan {self._task_id}: Execution manager exiting due to state {self._state.name}",
                    )
                    break

                self._state = _PlanState.RUNNING
                logger.info(
                    f"ToolLoopPlan {self._task_id}: Starting/Resuming internal loop with description: '{current_task_description}'",
                )

                self._plan_client.reset_messages()
                self._plan_client.reset_system_message()
                self._plan_client.set_system_message(
                    "You are a helpful web browser assistant. Use the available tools to complete the user's request.",
                )

                if current_parent_chat_context:
                    self._plan_client.append_messages(current_parent_chat_context)

                current_parent_chat_context = None

                internal_tools = self._get_internal_tools()
                self._loop_handle = start_async_tool_use_loop(
                    client=self._plan_client,
                    message=current_task_description,
                    tools=internal_tools,
                    loop_id=f"{self.__class__.__name__}.{self._manage_plan_execution.__name__}",
                    propagate_chat_context=True,
                    interrupt_llm_with_interjections=True,
                    log_steps=False,
                    max_steps=self.MAX_STEPS,
                )

                try:
                    loop_result_str = await self._loop_handle.result()
                    if self._state == _PlanState.RUNNING:
                        self._state = _PlanState.COMPLETED
                        self._result_str = loop_result_str
                        logger.info(
                            f"ToolLoopPlan {self._task_id}: Internal loop COMPLETED. Result: {self._result_str}",
                        )
                    elif self._state == _PlanState.PAUSED:
                        logger.info(
                            f"ToolLoopPlan {self._task_id}: Internal loop stopped for PAUSE.",
                        )
                    elif self._state == _PlanState.STOPPED:
                        logger.info(
                            f"ToolLoopPlan {self._task_id}: Internal loop stopped for STOP.",
                        )
                        if self._result_str is None:
                            self._result_str = f"Plan {self._task_id} was stopped."
                except asyncio.CancelledError:
                    logger.info(
                        f"ToolLoopPlan {self._task_id}: Internal loop task was cancelled. Current state: {self._state.name}",
                    )
                    if self._state == _PlanState.RUNNING:
                        self._state = _PlanState.STOPPED
                    if self._result_str is None:
                        self._result_str = f"Plan {self._task_id} was {self._state.name.lower()} (cancelled)."
                except Exception as e:
                    logger.error(
                        f"ToolLoopPlan {self._task_id}: Internal loop failed: {e}",
                        exc_info=True,
                    )
                    self._state = _PlanState.ERROR
                    self._error_str = str(e)
                    self._result_str = f"Task failed with error: {self._error_str}"

                self._loop_handle = None

                if self._state == _PlanState.PAUSED:
                    logger.info(
                        f"ToolLoopPlan {self._task_id}: Execution PAUSED, awaiting resume signal.",
                    )
                    await self._resume_requested_event.wait()
                    self._resume_requested_event.clear()
                    if self._state == _PlanState.STOPPED:
                        logger.info(
                            f"ToolLoopPlan {self._task_id}: Stop called while paused. Terminating.",
                        )
                        break
                    logger.info(f"ToolLoopPlan {self._task_id}: RESUMING execution.")
                    current_task_description = "The task was paused and is now resumed. Please review the history and continue."
                    current_parent_chat_context = self._parent_chat_context_on_pause
                    self._parent_chat_context_on_pause = None
                    continue
                else:
                    logger.info(
                        f"ToolLoopPlan {self._task_id}: Execution ended with state {self._state.name}. Finalizing.",
                    )
                    break
        except Exception as e:
            logger.error(
                f"ToolLoopPlan {self._task_id}: Unexpected error in _manage_plan_execution: {e}",
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
                f"ToolLoopPlan {self._task_id}: Setting overall completion event. Final state: {self._state.name}",
            )
            self._overall_plan_completion_event.set()

    @functools.wraps(BasePlan.result, updated=())
    async def result(self) -> str:
        await self._overall_plan_completion_event.wait()
        if self._error_str:
            return f"Error: {self._error_str}"
        return (
            self._result_str
            if self._result_str is not None
            else f"Plan {self._task_id} concluded without a specific result (State: {self._state.name})."
        )

    @functools.wraps(BasePlan.done, updated=())
    def done(self) -> bool:
        return self._overall_plan_completion_event.is_set()

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
            return self._state == _PlanState.RUNNING and self._loop_handle is not None
        if name == "ask":
            return self._state in (_PlanState.RUNNING, _PlanState.PAUSED)
        return False

    @functools.wraps(BasePlan.stop, updated=())
    async def stop(self) -> str:
        if not self._is_valid_method("stop"):
            if self.done():
                return await self.result()
            raise RuntimeError(
                f"Plan {self._task_id} cannot be stopped in state {self._state.name}.",
            )

        logger.info(
            f"ToolLoopPlan {self._task_id}: Stopping. Current state: {self._state.name}",
        )
        previous_state = self._state
        self._state = _PlanState.STOPPED
        self._result_str = f"Plan {self._task_id} was stopped."

        if previous_state == _PlanState.PAUSED:
            self._resume_requested_event.set()

        if self._loop_handle and not self._loop_handle.done():
            self._loop_handle.stop()
        elif (
            previous_state == _PlanState.IDLE
            and not self._overall_plan_completion_event.is_set()
        ):
            logger.warning(
                f"ToolLoopPlan {self._task_id}: Stop called in IDLE state. Forcing overall completion.",
            )
            self._overall_plan_completion_event.set()

        await self._overall_plan_completion_event.wait()
        return self._result_str

    @functools.wraps(BasePlan.pause, updated=())
    async def pause(self) -> str:
        if not self._is_valid_method("pause"):
            raise RuntimeError(
                f"Plan {self._task_id} cannot be paused in state {self._state.name}.",
            )
        logger.info(
            f"ToolLoopPlan {self._task_id}: Pausing. Current state: {self._state.name}",
        )
        self._state = _PlanState.PAUSED

        if self._plan_client and self._plan_client.messages:
            self._parent_chat_context_on_pause = copy.deepcopy(
                self._plan_client.messages,
            )
            logger.info(
                f"ToolLoopPlan {self._task_id}: Context saved on pause: {len(self._parent_chat_context_on_pause)} messages.",
            )
        else:
            self._parent_chat_context_on_pause = []
            logger.info(
                f"ToolLoopPlan {self._task_id}: No active LLM context to save on pause.",
            )

        if self._loop_handle and not self._loop_handle.done():
            logger.info(
                f"ToolLoopPlan {self._task_id}: Requesting stop of current internal loop for pause.",
            )
            self._loop_handle.stop()
        else:
            logger.warning(
                f"ToolLoopPlan {self._task_id}: Pause called but no active internal loop_handle to stop.",
            )

        return f"Plan {self._task_id} paused successfully. Awaiting resume."

    @functools.wraps(BasePlan.resume, updated=())
    async def resume(self) -> str:
        if not self._is_valid_method("resume"):
            raise RuntimeError(
                f"Plan {self._task_id} cannot be resumed in state {self._state.name}.",
            )
        logger.info(
            f"ToolLoopPlan {self._task_id}: Requesting resume. Current state: {self._state.name}",
        )
        self._resume_requested_event.set()
        return f"Plan {self._task_id} is resuming."

    @functools.wraps(BasePlan.interject, updated=())
    async def interject(self, message: str) -> str:
        if not self._is_valid_method("interject"):
            if self._state != _PlanState.RUNNING:
                return f"Error: Plan {self._task_id} is not in RUNNING state (current: {self._state.name}), cannot interject."
            if not self._loop_handle:  # Should not happen if RUNNING
                return f"Error: Plan {self._task_id} is RUNNING but has no active internal loop to interject."

        logger.info(
            f"ToolLoopPlan {self._task_id}: Interjecting message: '{message}' into active internal loop.",
        )
        await self._loop_handle.interject(message)  # type: ignore
        return f"Interjection '{message}' sent to plan {self._task_id}."

    @functools.wraps(BasePlan.ask, updated=())
    async def ask(self, question: str) -> str:
        if not self._is_valid_method("ask"):
            raise RuntimeError(
                f"Cannot ask question for plan {self._task_id} in state {self._state.name}.",
            )

        logger.info(f"ToolLoopPlan {self._task_id}: Answering query: '{question}'")
        current_context_to_share = []
        if (
            self._state == _PlanState.RUNNING
            and self._plan_client
            and self._plan_client.messages
        ):
            current_context_to_share = copy.deepcopy(self._plan_client.messages)
        elif self._state == _PlanState.PAUSED and self._parent_chat_context_on_pause:
            current_context_to_share = copy.deepcopy(self._parent_chat_context_on_pause)

        if not current_context_to_share:
            return "No context available to answer the question for the current plan state."

        # Ensure _ask_client uses its own separate message history for each ask
        self._ask_client.reset_messages()
        self._ask_client.reset_system_message()
        self._ask_client.set_system_message(
            "You are answering questions about an ongoing automated task. "
            "The main task's chat history will be provided. Answer concisely based on this history.",
        )
        self._ask_client.append_messages(
            [
                {
                    "role": "system",
                    "content": f"Current task ({self._task_id}, state: {self._state.name}) chat history:\n{json.dumps(current_context_to_share, indent=2)}",
                },
                {"role": "user", "content": question},
            ],
        )

        try:
            response = await self._ask_client.generate()
            return response.strip() if isinstance(response, str) else str(response)
        except Exception as e:
            logger.error(
                f"ToolLoopPlan {self._task_id}: Error during LLM call for ask: {e}",
                exc_info=True,
            )
            return f"Error answering question due to LLM failure: {e}"

    @property
    @functools.wraps(BasePlan.valid_tools, updated=())
    def valid_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        tools = {}
        potential_tools = ["stop", "pause", "resume", "interject", "ask"]
        for method_name in potential_tools:
            if self._is_valid_method(method_name):
                tools[method_name] = getattr(self, method_name)
        return tools


class ToolLoopPlanner(BasePlanner[ToolLoopPlan]):
    def __init__(
        self,
        session_connect_url: Optional[str] = None,
        headless: bool = False,
    ):
        super().__init__()
        self._controller = Controller(
            session_connect_url=session_connect_url,
            headless=headless,
        )
        if not self._controller.is_alive():
            self._controller.start()
        self._coms_manager = ComsManager()
        self._tools_cache: Optional[Dict[str, Callable[..., Awaitable[Any]]]] = None
        self._main_event_loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            self._main_event_loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "ToolLoopPlanner initialized outside of a running asyncio event loop.",
            )

    def _get_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        if self._tools_cache is None:
            self._tools_cache = self._build_tools()
        return self._tools_cache

    def _build_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        # This remains the same as it defines tools for the *internal* loop
        async def act(action: str) -> str:
            logger.info(f"Planner: Calling Controller.act with '{action}'")
            result = await self._controller.act(action)
            if isinstance(result, list):
                res_str = ", ".join(map(str, result))
                return (
                    f"Executed actions: {res_str}"
                    if result
                    else "No action taken by controller."
                )
            return (
                f"Executed action: {result}"
                if result
                else "No action taken by controller."
            )

        async def observe(query: str, response_format_str: str = "str") -> str:
            logger.info(
                f"Planner: Calling Controller.observe with query '{query}', format '{response_format_str}'",
            )
            format_map: Dict[str, Type] = {
                "str": str,
                "bool": bool,
                "int": int,
                "float": float,
            }
            actual_response_format = format_map.get(response_format_str.lower(), str)
            result = await self._controller.observe(
                query,
                response_format=actual_response_format,
            )
            return str(result)

        async def communicate(description: str) -> str:
            logger.info(
                f"Planner: Calling ComsManager.communicate with '{description}'",
            )
            return await self._coms_manager.communicate(description)

        act.__doc__ = "Performs a specified action using the system controller. Input is the action description."
        observe.__doc__ = "Observes the system or environment based on a query. Specify 'response_format_str' as 'str', 'bool', 'int', or 'float'."
        communicate.__doc__ = "Communicates a message or instruction externally."

        import inspect

        act.__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    "action",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=str,
                ),
            ],
            return_annotation=str,
        )
        observe.__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    "query",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=str,
                ),
                inspect.Parameter(
                    "response_format_str",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=str,
                    default="str",
                ),
            ],
            return_annotation=str,
        )
        communicate.__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    "description",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=str,
                ),
            ],
            return_annotation=str,
        )

        return {
            "act": act,
            "observe": observe,
            "communicate": communicate,
        }

    def _make_plan(
        self,
        task_description: str,
        *,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> ToolLoopPlan:
        logger.info(f"ToolLoopPlanner: Planning task: '{task_description}'")

        if not self._main_event_loop:
            try:
                self._main_event_loop = asyncio.get_running_loop()
                logger.info(
                    f"ToolLoopPlanner._make_plan captured event loop: {self._main_event_loop}",
                )
            except RuntimeError:
                logger.error(
                    "ToolLoopPlanner._make_plan: No running event loop to pass to ToolLoopPlan.",
                )

        plan = ToolLoopPlan(
            task_description=task_description,
            tools=self._get_tools(),
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
            main_event_loop=self._main_event_loop,
        )
        return plan

    async def close(self):
        logger.info("ToolLoopPlanner: Closing resources...")
        if self._controller:
            self._controller.stop()
        if self._coms_manager:
            await self._coms_manager.stop()
