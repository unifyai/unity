import asyncio
import enum
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type
import os
import json
import copy
import uuid
from functools import wraps

from unity.common.llm_helpers import (
    start_async_tool_use_loop,
    SteerableToolHandle,
)
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
        await asyncio.sleep(0.1)
        return f"Communication task initiated for: {description}"


class _PlanState(enum.Enum):
    IDLE = enum.auto()
    RUNNING = enum.auto()
    PAUSED = enum.auto()
    COMPLETED = enum.auto()
    STOPPED = enum.auto()
    ERROR = enum.auto()


class ToolLoopPlan(SteerableToolHandle):
    """
    Represents an active plan being executed by the ToolLoopPlanner.
    Inherits from SteerableToolHandle to provide a consistent interface for interaction.
    """

    def __init__(
        self,
        task_description: str,
        initial_client: AsyncUnify,
        tools: Dict[str, Callable[..., Awaitable[Any]]],
        base_system_prompt: Optional[str],
        parent_chat_context: Optional[List[dict]] = None,
    ):
        super().__init__()
        self._task_description = task_description
        self._client = initial_client
        self._tools = tools
        self._base_system_prompt = base_system_prompt
        self._parent_chat_context_on_pause: Optional[List[dict]] = parent_chat_context

        self._clar_up_q_internal: asyncio.Queue[str] = asyncio.Queue()
        self._clar_down_q_internal: asyncio.Queue[str] = asyncio.Queue()

        self._state: _PlanState = _PlanState.IDLE
        self._loop_handle: Optional[SteerableToolHandle] = None
        self._result_str: Optional[str] = None
        self._error_str: Optional[str] = None
        self._completion_event = asyncio.Event()
        self._task_id = str(uuid.uuid4())

        self._ask_client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
            stateful=True,
        )
        self._ask_system_prompt = (
            "You are answering questions about an ongoing automated task. "
            "The main task's chat history will be provided. Answer concisely based on this history."
        )
        self._ask_client.set_system_message(self._ask_system_prompt)

        self._start()

    def _start(self):
        if self._state != _PlanState.IDLE:
            raise RuntimeError("Plan can only be started once.")

        logger.info(
            f"ToolLoopPlan {self._task_id}: Starting with description: '{self._task_description}'",
        )
        self._state = _PlanState.RUNNING
        self._completion_event.clear()

        current_tools = self._tools.copy()

        async def request_clarification_tool(question: str) -> str:
            logger.info(
                f"ToolLoopPlan {self._task_id}: LLM requesting clarification: '{question}'",
            )
            await self.clarification_questions.put(question)
            answer = await self._clar_down_q_internal.get()
            logger.info(
                f"ToolLoopPlan {self._task_id}: User provided clarification: '{answer}'",
            )
            return answer

        current_tools["request_clarification_tool"] = request_clarification_tool

        self._loop_handle = start_async_tool_use_loop(
            client=self._client,
            message=self._task_description,
            tools=current_tools,
            parent_chat_context=self._parent_chat_context_on_pause,
            propagate_chat_context=True,
            interrupt_llm_with_interjections=True,
            log_steps=True,
        )
        asyncio.create_task(self._await_completion())

    async def _await_completion(self):
        if not self._loop_handle:
            return
        try:
            self._result_str = await self._loop_handle.result()
            if (
                self._state == _PlanState.RUNNING
            ):  # Only if not already paused or stopped
                self._state = _PlanState.COMPLETED
            logger.info(
                f"ToolLoopPlan {self._task_id}: Completed/Finished. Final state: {self._state.name}. Result: {self._result_str}",
            )
        except asyncio.CancelledError:
            if self._state == _PlanState.RUNNING:
                pass  # State will be set by stop/pause
            elif self._state != _PlanState.PAUSED and self._state != _PlanState.STOPPED:
                self._state = _PlanState.STOPPED
            logger.info(
                f"ToolLoopPlan {self._task_id}: Execution was cancelled. Current state: {self._state.name}",
            )
            if self._result_str is None:
                self._result_str = f"Task was {self._state.name.lower()}."
        except Exception as e:
            self._state = _PlanState.ERROR
            self._error_str = str(e)
            self._result_str = f"Task failed with error: {self._error_str}"
            logger.error(
                f"ToolLoopPlan {self._task_id}: Failed with error: {e}",
                exc_info=True,
            )
        finally:
            self._completion_event.set()

    async def result(self) -> str:
        await self._completion_event.wait()
        if self._error_str:
            return f"Error: {self._error_str}"
        return (
            self._result_str
            if self._result_str is not None
            else "Plan did not produce a result or was stopped."
        )

    def done(self) -> bool:
        return self._state in (
            _PlanState.COMPLETED,
            _PlanState.STOPPED,
            _PlanState.ERROR,
        )

    @property
    def clarification_questions(self) -> asyncio.Queue[str]:
        return self._clar_up_q_internal

    async def answer_clarification(self, answer: str) -> None:
        await self._clar_down_q_internal.put(answer)

    def _sync_stop(self):
        logger.info(
            f"ToolLoopPlan {self._task_id}: Stopping (sync part). Current state: {self._state.name}",
        )
        self._state = _PlanState.STOPPED
        if self._loop_handle:
            self._loop_handle.stop()
        else:
            self._completion_event.set()
        if self._result_str is None:
            self._result_str = f"Plan {self._task_id} was stopped."

    def _sync_pause(self):
        logger.info(
            f"ToolLoopPlan {self._task_id}: Pausing (sync part). Current state: {self._state.name}",
        )
        self._state = _PlanState.PAUSED
        if self._loop_handle:
            self._loop_handle.stop()
        else:
            logger.warning(
                f"ToolLoopPlan {self._task_id}: Pause called but no active loop_handle.",
            )
            self._completion_event.set()

    def _sync_resume(self):
        logger.info(
            f"ToolLoopPlan {self._task_id}: Resuming (sync part). Current state: {self._state.name}",
        )

        self._state = _PlanState.RUNNING
        self._completion_event.clear()

        self._client = AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        if self._base_system_prompt:
            self._client.set_system_message(self._base_system_prompt)

        if self._parent_chat_context_on_pause:
            messages_to_load = self._parent_chat_context_on_pause
            if (
                self._base_system_prompt
                and messages_to_load
                and messages_to_load[0].get("role") == "system"
            ):
                messages_to_load = messages_to_load[1:]
            if messages_to_load:
                self._client.append_messages(messages_to_load)

        current_tools = self._tools.copy()

        async def request_clarification_tool(question: str) -> str:
            logger.info(
                f"ToolLoopPlan {self._task_id} (resumed): LLM requesting clarification: '{question}'",
            )
            await self.clarification_questions.put(question)
            answer = await self._clar_down_q_internal.get()
            logger.info(
                f"ToolLoopPlan {self._task_id} (resumed): User provided clarification: '{answer}'",
            )
            return answer

        current_tools["request_clarification_tool"] = request_clarification_tool

        self._loop_handle = start_async_tool_use_loop(
            client=self._client,
            message="The task was paused. Please review the history and determine the next best action or tool call.",
            tools=current_tools,
            parent_chat_context=None,
            propagate_chat_context=True,
            interrupt_llm_with_interjections=True,
            log_steps=True,
        )
        asyncio.create_task(self._await_completion())

    async def _async_interject(self, message: str) -> str:
        if not self._loop_handle:
            return "Error: No active loop to interject."
        logger.info(f"ToolLoopPlan {self._task_id}: Interjecting message: '{message}'")
        await self._loop_handle.interject(message)
        return f"Interjection '{message}' sent to plan {self._task_id}."

    async def _async_ask(self, question: str) -> str:
        logger.info(f"ToolLoopPlan {self._task_id}: Answering query: '{question}'")
        current_context_to_share = []
        if self._state == _PlanState.RUNNING and self._client:
            current_context_to_share = copy.deepcopy(self._client.messages or [])
        elif self._state == _PlanState.PAUSED and self._parent_chat_context_on_pause:
            current_context_to_share = copy.deepcopy(self._parent_chat_context_on_pause)

        if not current_context_to_share:
            return "No context available to answer the question."

        self._ask_client.reset_messages()
        self._ask_client.set_system_message(self._ask_system_prompt)
        self._ask_client.append_message(
            {
                "role": "system",
                "content": f"Current task ({self._task_id}) chat history:\n{json.dumps(current_context_to_share, indent=2)}",
            },
        )
        self._ask_client.append_message({"role": "user", "content": question})

        try:
            response = await self._ask_client.generate()
            return response.strip() if isinstance(response, str) else str(response)
        except Exception as e:
            logger.error(
                f"ToolLoopPlan {self._task_id}: Error during ask: {e}",
                exc_info=True,
            )
            return f"Error answering question: {e}"

    def _is_valid_method(self, name: str) -> bool:
        if name == "stop":
            return self._state in (_PlanState.RUNNING, _PlanState.PAUSED)
        if name == "pause":
            return self._state == _PlanState.RUNNING
        if name == "resume":
            return self._state == _PlanState.PAUSED
        if name == "interject":
            return self._state == _PlanState.RUNNING
        if name == "ask":
            return self._state in (_PlanState.RUNNING, _PlanState.PAUSED)
        return False

    def __getattr__(self, name: str) -> Callable[..., Awaitable[Any]]:
        sync_method_map = {
            "stop": self._sync_stop,
            "pause": self._sync_pause,
            "resume": self._sync_resume,
        }
        async_method_map = {
            "interject": self._async_interject,
            "ask": self._async_ask,
        }

        if not self._is_valid_method(name):
            raise AttributeError(
                f"Method '{name}' is not valid in the current plan state '{self._state.name}'. Attempted on ToolLoopPlan {self._task_id}",
            )

        if name in sync_method_map:
            sync_method_to_call = sync_method_map[name]

            @wraps(sync_method_to_call)
            async def async_wrapper(*args, **kwargs) -> str:
                sync_method_to_call(*args, **kwargs)
                if name in ["stop", "pause"]:
                    await self._completion_event.wait()
                if name == "pause":  # Save context after pause is complete
                    if self._client and self._client.messages:
                        self._parent_chat_context_on_pause = copy.deepcopy(
                            self._client.messages,
                        )
                    else:
                        self._parent_chat_context_on_pause = []
                    logger.info(
                        f"ToolLoopPlan {self._task_id}: Context saved on pause: {len(self._parent_chat_context_on_pause)} messages.",
                    )

                return (
                    self._result_str
                    if name == "stop" and self._result_str
                    else f"Plan {self._task_id} {name} action initiated."
                )

            return async_wrapper
        elif name in async_method_map:
            return async_method_map[name]

        # Fallback for SteerableToolHandle's own methods if not overridden
        # Or raise AttributeError if truly not found
        try:
            return super().__getattr__(name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}' (task_id: {self._task_id}, state: {self._state.name})",
            )

    def __dir__(self):
        default_attrs = set(super().__dir__())
        exposed_methods = set()
        if self._is_valid_method("stop"):
            exposed_methods.add("stop")
        if self._is_valid_method("pause"):
            exposed_methods.add("pause")
        if self._is_valid_method("resume"):
            exposed_methods.add("resume")
        if self._is_valid_method("interject"):
            exposed_methods.add("interject")
        if self._is_valid_method("ask"):
            exposed_methods.add("ask")
        # Add properties for clarification
        exposed_methods.add("clarification_questions")
        exposed_methods.add("answer_clarification")
        return sorted(list(default_attrs.union(exposed_methods)))


class ToolLoopPlanner:
    def __init__(
        self,
        base_system_prompt: str = "You are a helpful assistant. Use the available tools to complete the user's request. Prioritize completing the primary request.",
    ):
        self._base_system_prompt = base_system_prompt
        self._controller = Controller()
        if not self._controller.is_alive():
            self._controller.start()
        self._coms_manager = ComsManager()

        self._tools_cache: Optional[Dict[str, Callable[..., Awaitable[Any]]]] = None

    def _get_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        if self._tools_cache is None:
            self._tools_cache = self._build_tools()
        return self._tools_cache

    def _build_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        async def act_tool(action: str) -> str:
            logger.info(f"Planner: Calling Controller.act with '{action}'")
            result = await self._controller.act(action)
            if isinstance(result, list):
                res_str = ", ".join(result)
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

        async def observe_tool(query: str, response_format_str: str = "str") -> str:
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

        async def communicate_tool(description: str) -> str:
            logger.info(
                f"Planner: Calling ComsManager.communicate with '{description}'",
            )
            return await self._coms_manager.communicate(description)

        return {
            "act_tool": act_tool,
            "observe_tool": observe_tool,
            "communicate_tool": communicate_tool,
        }

    def start(
        self,
        task_description: str,
        parent_chat_context: Optional[List[dict]] = None,
    ) -> ToolLoopPlan:
        logger.info(f"ToolLoopPlanner: Starting a new plan for: '{task_description}'")

        plan_client = AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        current_client_messages = []
        if self._base_system_prompt:
            plan_client.set_system_message(self._base_system_prompt)
            current_client_messages.append(
                {"role": "system", "content": self._base_system_prompt},
            )

        if parent_chat_context:
            messages_to_load = parent_chat_context
            if (
                self._base_system_prompt
                and messages_to_load
                and messages_to_load[0].get("role") == "system"
            ):
                messages_to_load = messages_to_load[1:]
            if messages_to_load:
                plan_client.append_messages(messages_to_load)
                current_client_messages.extend(messages_to_load)

        plan = ToolLoopPlan(
            task_description=task_description,
            initial_client=plan_client,
            tools=self._get_tools(),
            base_system_prompt=self._base_system_prompt,
            parent_chat_context=current_client_messages,
        )
        return plan
