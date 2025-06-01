import asyncio
import enum
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type
import os
import json
import copy
import uuid

from unity.common.llm_helpers import (
    start_async_tool_use_loop,
    SteerableToolHandle,
)
from unity.controller.controller import (
    Controller,
)
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
    Methods like stop, pause, resume, interject, ask are always public,
    but will raise errors if called in an invalid state.
    The valid_tools property indicates which methods can be successfully called.
    """

    def __init__(
        self,
        task_description: str,
        initial_client: AsyncUnify,
        tools: Dict[str, Callable[..., Awaitable[Any]]],
        base_system_prompt: Optional[str],
        parent_chat_context: Optional[List[dict]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ):
        super().__init__()
        self._task_description = task_description
        self._client = initial_client  # LLM client for the main tool loop
        self._tools = tools  # Tools available to the main tool loop
        self._base_system_prompt = base_system_prompt
        self._parent_chat_context_on_pause: Optional[List[dict]] = parent_chat_context

        # Clarification queues for interaction with the entity that started this plan
        self._clar_up_q_internal: asyncio.Queue[str] = (
            clarification_up_q or asyncio.Queue()
        )
        self._clar_down_q_internal: asyncio.Queue[str] = (
            clarification_down_q or asyncio.Queue()
        )

        self._state: _PlanState = _PlanState.IDLE
        self._loop_handle: Optional[SteerableToolHandle] = (
            None  # Handle for the underlying async tool loop
        )
        self._result_str: Optional[str] = None  # Final result of the plan
        self._error_str: Optional[str] = None  # Error message if the plan failed
        self._completion_event = asyncio.Event()  # Signals plan completion/stop/error
        self._task_id = str(uuid.uuid4())  # Unique ID for this plan instance

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

        self._start_internal_loop()

    def _start_internal_loop(self):
        """
        Starts the internal async tool use loop that executes the plan's logic.
        This is called once during initialization or resume.
        """
        if (
            self._state == _PlanState.RUNNING
            and self._loop_handle
            and not self._loop_handle.done()
        ):
            logger.warning(
                f"ToolLoopPlan {self._task_id}: Attempted to start internal loop while already running.",
            )
            return

        logger.info(
            f"ToolLoopPlan {self._task_id}: Starting internal loop with description: '{self._task_description}'",
        )
        self._state = _PlanState.RUNNING
        self._completion_event.clear()

        current_tools = self._tools.copy()

        async def request_clarification_tool_for_llm(question: str) -> str:
            logger.info(
                f"ToolLoopPlan {self._task_id}: LLM (internal loop) requesting clarification: '{question}'",
            )
            await self.clarification_questions.put(question)
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

        self._loop_handle = start_async_tool_use_loop(
            client=self._client,
            message=self._task_description,
            tools=current_tools,
            parent_chat_context=(
                self._parent_chat_context_on_pause
                if self._state != _PlanState.RUNNING
                else None
            ),
            propagate_chat_context=True,
            interrupt_llm_with_interjections=True,
            log_steps=False,
        )
        asyncio.create_task(self._await_completion())

    async def _await_completion(self):
        """
        Waits for the internal tool loop to complete and updates plan state.
        """
        if not self._loop_handle:
            return
        try:
            self._result_str = await self._loop_handle.result()
            if self._state == _PlanState.RUNNING:
                self._state = _PlanState.COMPLETED
            logger.info(
                f"ToolLoopPlan {self._task_id}: Completed/Finished. Final state: {self._state.name}. Result: {self._result_str}",
            )
        except asyncio.CancelledError:
            if self._state == _PlanState.RUNNING:
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
        """Waits for the plan to complete and returns its final result string."""
        await self._completion_event.wait()
        if self._error_str:
            return f"Error: {self._error_str}"
        return (
            self._result_str
            if self._result_str is not None
            else "Plan did not produce a result or was stopped."
        )

    def done(self) -> bool:
        """Returns True if the plan has completed, stopped, or encountered an error."""
        return self._state in (
            _PlanState.COMPLETED,
            _PlanState.STOPPED,
            _PlanState.ERROR,
        )

    @property
    def clarification_questions(self) -> asyncio.Queue[str]:
        """Queue for this plan to send clarification questions upwards."""
        return self._clar_up_q_internal

    async def answer_clarification(self, answer: str) -> None:
        """Method for the caller of this plan to provide answers to clarifications."""
        await self._clar_down_q_internal.put(answer)

    def _is_valid_method(self, name: str) -> bool:
        """Checks if a control method is valid in the current plan state."""
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

    # --- Public Control Methods ---
    async def stop(self) -> str:
        """Stops the execution of the current plan."""
        if not self._is_valid_method("stop"):
            raise RuntimeError(
                f"Plan {self._task_id} cannot be stopped in state {self._state.name}.",
            )
        logger.info(
            f"ToolLoopPlan {self._task_id}: Stopping. Current state: {self._state.name}",
        )
        self._state = _PlanState.STOPPED
        if self._loop_handle:
            self._loop_handle.stop()
        else:
            self._completion_event.set()

        await self._completion_event.wait()

        if self._result_str is None:
            self._result_str = f"Plan {self._task_id} was stopped."
        return self._result_str

    async def pause(self) -> str:
        """Pauses the execution of the current plan."""
        if not self._is_valid_method("pause"):
            raise RuntimeError(
                f"Plan {self._task_id} cannot be paused in state {self._state.name}.",
            )
        logger.info(
            f"ToolLoopPlan {self._task_id}: Pausing. Current state: {self._state.name}",
        )
        self._state = _PlanState.PAUSED

        if self._client and self._client.messages:
            self._parent_chat_context_on_pause = copy.deepcopy(self._client.messages)
        else:
            self._parent_chat_context_on_pause = []
        logger.info(
            f"ToolLoopPlan {self._task_id}: Context saved on pause: {len(self._parent_chat_context_on_pause)} messages.",
        )

        if self._loop_handle:
            self._loop_handle.stop()
            await self._completion_event.wait()
        else:
            logger.warning(
                f"ToolLoopPlan {self._task_id}: Pause called but no active loop_handle.",
            )
            self._completion_event.set()

        return f"Plan {self._task_id} paused."

    async def resume(self) -> str:
        """Resumes a paused plan."""
        if not self._is_valid_method("resume"):
            raise RuntimeError(
                f"Plan {self._task_id} cannot be resumed in state {self._state.name}.",
            )
        logger.info(
            f"ToolLoopPlan {self._task_id}: Resuming. Current state: {self._state.name}",
        )

        if (
            not self._parent_chat_context_on_pause
        ):  # Check if there's context to resume with
            logger.warning(
                f"ToolLoopPlan {self._task_id}: Resuming without a saved parent context.",
            )

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

        self._task_description = "The task was paused. Please review the history and determine the next best action or tool call."
        self._start_internal_loop()

        return f"Plan {self._task_id} resuming."

    async def interject(self, message: str) -> str:
        """Sends an interjection to the running plan's internal tool loop."""
        if not self._is_valid_method("interject"):
            raise RuntimeError(
                f"Plan {self._task_id} cannot be interjected in state {self._state.name}.",
            )
        if not self._loop_handle:
            return "Error: No active loop to interject."
        logger.info(f"ToolLoopPlan {self._task_id}: Interjecting message: '{message}'")
        await self._loop_handle.interject(message)
        return f"Interjection '{message}' sent to plan {self._task_id}."

    async def ask(self, question: str) -> str:
        """Asks a question about the current state of the plan."""
        if not self._is_valid_method("ask"):
            raise RuntimeError(
                f"Cannot ask question for plan {self._task_id} in state {self._state.name}.",
            )

        logger.info(f"ToolLoopPlan {self._task_id}: Answering query: '{question}'")
        current_context_to_share = []
        if self._state == _PlanState.RUNNING and self._client and self._client.messages:
            current_context_to_share = copy.deepcopy(self._client.messages)
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

    @property
    def valid_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        """
        Returns a dictionary of currently available public methods on this plan handle.
        """
        tools = {}
        for method_name in ["stop", "pause", "resume", "interject", "ask"]:
            if self._is_valid_method(method_name):
                tools[method_name] = getattr(self, method_name)
        return tools

    def __dir__(self):
        """Ensures that `dir(plan_handle)` includes conditionally available methods."""
        default_attrs = set(super().__dir__())
        exposed_methods = set(self.valid_tools.keys())
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
        """Prepares and caches the tools available for the ToolLoopPlan."""
        if self._tools_cache is None:
            self._tools_cache = self._build_tools()
        return self._tools_cache

    def _build_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        """
        Builds a dictionary of tools that the ToolLoopPlanner can use.
        This includes tools from the controller and communications manager.
        """

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

        act_tool.__name__ = "act_tool"
        act_tool.__qualname__ = "act_tool"
        observe_tool.__name__ = "observe_tool"
        observe_tool.__qualname__ = "observe_tool"
        communicate_tool.__name__ = "communicate_tool"
        communicate_tool.__qualname__ = "communicate_tool"

        return {
            "act_tool": act_tool,
            "observe_tool": observe_tool,
            "communicate_tool": communicate_tool,
        }

    def plan(
        self,
        task_description: str,
        parent_chat_context: Optional[List[dict]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> ToolLoopPlan:
        """
        Initiates a new plan for the given task description.
        Renamed from 'start' to 'plan'.
        """
        logger.info(f"ToolLoopPlanner: Planning task: '{task_description}'")

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
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )
        return plan
