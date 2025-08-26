import asyncio
import enum
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type
from pydantic import create_model, BaseModel, Field
import functools
import os
import json
import copy
import uuid
import inspect
import base64

from unity.common.llm_helpers import (
    start_async_tool_use_loop,
    SteerableToolHandle,
    _strip_image_keys,
)
from ..task_scheduler.base import BaseActiveTask
from .base import BaseActor
from unity.controller.controller import Controller, ActionFailedError
from unify import AsyncUnify
import unify

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class _PlanState(enum.Enum):
    IDLE = enum.auto()
    RUNNING = enum.auto()
    PAUSED = enum.auto()
    COMPLETED = enum.auto()
    STOPPED = enum.auto()
    ERROR = enum.auto()


def create_model_from_schema(
    schema: dict,
    model_name: str = "DynamicModel",
) -> Type[BaseModel]:
    """
    Recursively creates a Pydantic model from a JSON schema dictionary.
    """
    fields = {}

    type_mapping = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
    }

    required_fields = schema.get("required", [])

    for prop_name, prop_details in schema.get("properties", {}).items():
        prop_type_str = prop_details.get("type")
        description = prop_details.get("description")
        is_required = prop_name in required_fields
        field_type = Any

        if prop_type_str == "object":
            field_type = create_model_from_schema(
                prop_details,
                model_name=f"{model_name}_{prop_name}",
            )
        elif prop_type_str == "array":
            items_schema = prop_details.get("items", {})
            item_type_str = items_schema.get("type")
            if item_type_str == "object":
                item_model = create_model_from_schema(
                    items_schema,
                    model_name=f"{model_name}_{prop_name}Item",
                )
                field_type = List[item_model]
            else:
                primitive_item_type = type_mapping.get(item_type_str, str)
                field_type = List[primitive_item_type]
        else:
            field_type = type_mapping.get(prop_type_str, str)

        default_value = ... if is_required else None
        fields[prop_name] = (
            field_type,
            Field(default=default_value, description=description),
        )

    return create_model(model_name, **fields)


TOOL_LOOP_SYSTEM_PROMPT = """### Your Role
You are an expert web automation agent. Your primary function is to achieve a user's goal by executing a precise sequence of tool calls.

### Core Mission
Break down the user's request into a series of logical steps. At each step, choose the single best tool to advance the task. Your final response should be the specific information the user asked for, not just a confirmation like "Task Complete."

### Critical Rules for Operation
1.  **Observe, Then Act**: Always use the `observe` tool to analyze the page before you attempt an action with `act`. You cannot click what you cannot see.
2.  **Be Precise & Complete**: Your calls to tools MUST be exact and include all required arguments. For the `act` tool, both `action` and `expectation` are **always required**.
3.  **Self-Correct**: If an action fails or you are unsure what to do, your first step is to use `get_action_history()` to review your past actions. This is your primary method for self-correction.
4.  **Use Structured Output**: For any `observe` call where you need to extract specific information, you MUST use the `response_schema` argument, providing a valid JSON schema `dict`.

---
### Tools Reference
You can call the following functions. Adhere strictly to the signatures and argument requirements.

1.  `act(action: str, expectation: str) -> str`
    - **Description**: Performs a single, high-level action in the browser (e.g., "click the login button", "type 'hello' into the search bar").
    - **Arguments**:
        - `action` (str, **required**): The natural-language instruction for what to do.
        - `expectation` (str, **required**): A clear, verifiable description of what the page should look like *after* the action is successfully completed.
    - **Returns**: A string confirming the action was performed.

2.  `observe(query: str, response_schema: dict = None) -> Any`
    - **Description**: Asks a question about the current state of the browser page. This is a read-only operation.
    - **Arguments**:
        - `query` (str, **required**): The question to ask about the page (e.g., "What is the page title?", "Is there a 'Submit' button?").
        - `response_schema` (dict, optional): A JSON schema dictionary to structure the output.
    - **Returns**: The answer to the query, either as a string or a JSON object matching the `response_schema`.

3.  `get_action_history() -> list[dict]`
    - **Description**: Retrieves a summary of the browser actions executed so far in this session. Each item includes the command and its timestamp.

4.  `get_screenshots_for_action(timestamp: float) -> dict`
    - **Description**: Retrieves the 'before' and 'after' screenshots for a specific action, identified by its timestamp from the action history.

---
### Usage Examples
Your response must be a JSON object containing a `tool_calls` array, as shown below.

**Example 1: Simple Action**
*User Request*: "Click the 'Images' tab."
*Your Thought*: First, I should see if the tab is visible. Then I will click it and confirm the page has changed.
*Tool Call (Observe, then Act)*:
```json
{
  "tool_calls": [
    {
      "name": "act",
      "arguments": {
        "action": "Click the 'Images' link.",
        "expectation": "The page should now be displaying image search results, and the 'Images' tab should be highlighted."
      }
    }
  ]
}

**Example 2: Structured Observation**
*User Request*: "Find the price on the page."
*Your Thought*: I need to find the price and extract it as a structured object. I will use observe with a response_schema.
*Tool Call*:

JSON

{
  "tool_calls": [
    {
      "name": "observe",
      "arguments": {
        "query": "What is the price of the main item on the page?",
        "response_schema": {
          "type": "object",
          "properties": {
            "price": { "type": "string", "description": "The price of the item, including currency symbol." },
            "currency": { "type": "string", "description": "The ISO currency code, e.g., USD, EUR." }
          },
          "required": ["price"]
        }
      }
    }
  ]
}
"""


class ToolLoopPlan(BaseActiveTask):
    """
    Represents an active plan being executed by the ToolLoopActor.
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
        timeout: Optional[float] = 1000,
        persist: bool = False,
        custom_system_prompt: str | None = None,
        tool_policy: Optional[Callable] = None,
        action_provider: Optional["ActionProvider"] = None,  # type: ignore
    ):
        self._initial_task_description = task_description
        self._tools = tools
        self._parent_chat_context_on_pause: Optional[List[dict]] = parent_chat_context
        self._chat_history: List[Dict[str, Any]] = []
        self._custom_system_prompt = custom_system_prompt

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
            "gemini-2.5-pro@vertex-ai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )

        self._ask_client = unify.AsyncUnify(
            "gemini-2.5-pro@vertex-ai",
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
                f"ToolLoopPlan {self._task_id}: LLM (internal loop) requesting clarification: '{question}'",
            )
            await self._clar_up_q_internal.put(question)
            answer = await self._clar_down_q_internal.get()
            logger.info(
                f"ToolLoopPlan {self._task_id}: User (via plan) provided clarification: '{answer}'",
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
                        f"ToolLoopPlan {self._task_id}: Execution manager exiting due to state {self._state.name}",
                    )
                    break

                self._state = _PlanState.RUNNING
                logger.info(
                    f"ToolLoopPlan {self._task_id}: Starting/Resuming internal loop with description: '{current_task_description}'",
                )

                self._plan_client.reset_messages()
                self._plan_client.reset_system_message()

                system_prompt = (
                    self._custom_system_prompt
                    if self._custom_system_prompt
                    else TOOL_LOOP_SYSTEM_PROMPT
                )
                self._plan_client.set_system_message(system_prompt)

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
                    log_steps=True,
                    max_steps=self.MAX_STEPS,
                    timeout=self._timeout,
                    persist=self._persist,
                    tool_policy=self._tool_policy,
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

                if self._loop_handle and self._loop_handle._client:
                    self._chat_history = list(self._loop_handle._client.messages)

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
    async def stop(self, reason: Optional[str] = None) -> str:
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
        self._result_str = (
            f"Plan {self._task_id} was stopped."
            if not reason
            else f"Plan {self._task_id} was stopped: {reason}"
        )

        if previous_state == _PlanState.PAUSED:
            self._resume_requested_event.set()

        if self._loop_handle and not self._loop_handle.done():
            self._loop_handle.stop(reason)
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

    @functools.wraps(BaseActiveTask.pause, updated=())
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

    @functools.wraps(BaseActiveTask.resume, updated=())
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

    @functools.wraps(BaseActiveTask.interject, updated=())
    async def interject(self, message: str) -> str:
        if not self._is_valid_method("interject"):
            if self.done():
                return f"Error: Plan {self._task_id} is already done, cannot interject."
            return f"Error: Plan {self._task_id} is in state {self._state.name}, cannot interject."

        if not self._loop_handle:
            logger.info(
                f"ToolLoopPlan {self._task_id}: Interject called before loop handle was created. Waiting briefly.",
            )
            for _ in range(5):
                if self._loop_handle:
                    break
                await asyncio.sleep(1)

            if not self._loop_handle:
                return f"Error: Plan {self._task_id} did not initialize in time for interjection."

        logger.info(
            f"ToolLoopPlan {self._task_id}: Interjecting message: '{message}' into active internal loop.",
        )
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

        logger.info(f"ToolLoopPlan {self._task_id}: Answering query: '{question}'")
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

        return start_async_tool_use_loop(
            client=self._ask_client,
            message=question,
            tools={},
            loop_id=f"Question({self._task_id})",
            max_consecutive_failures=1,
            timeout=60,
        )

    @property
    @functools.wraps(BaseActiveTask.valid_tools, updated=())
    def valid_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        tools = {}
        potential_tools = ["stop", "pause", "resume", "interject", "ask"]
        for method_name in potential_tools:
            if self._is_valid_method(method_name):
                tools[method_name] = getattr(self, method_name)
        return tools


class ToolLoopActor(BaseActor):
    def __init__(
        self,
        session_connect_url: Optional[str] = None,
        headless: bool = False,
        controller: Controller = None,
    ):
        self._controller = controller or Controller(
            session_connect_url=session_connect_url,
            headless=headless,
        )
        if not self._controller.is_alive():
            self._controller.start()
        self._tools_cache: Optional[Dict[str, Callable[..., Awaitable[Any]]]] = None
        self._main_event_loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            self._main_event_loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "ToolLoopActor initialized outside of a running asyncio event loop.",
            )

    def _get_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        if self._tools_cache is None:
            self._tools_cache = self._build_tools()
        return self._tools_cache

    def _build_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        async def act(action: str, expectation: str) -> str:
            logger.info(f"Actor: Calling Controller.act with '{action}'")
            try:
                result = await self._controller.act(action, expectation=expectation)
                return result
            except ActionFailedError as e:
                logger.warning(
                    f"ActionFailedError caught and handled within ToolLoopActor: {e.reason}",
                )

                error_message = (
                    "The previous action failed because the browser state did not match the expectation.\n\n"
                    f"**Action Attempted**: `{e.action}`\n"
                    f"**Expected Outcome**: `{e.expectation}`\n"
                    f"**Reason for Failure**: `{e.reason}`\n\n"
                    "Please analyze the 'Reason for Failure' and devise a new, better action to achieve the goal."
                )
                return error_message

        async def observe(query: str, response_schema: dict = None) -> Any:
            """
            Asks a question about the current state of the browser page.
            To get a structured response, provide a valid JSON Schema in the 'response_schema' argument.
            """
            logger.info(f"Actor: Calling Controller.observe with query '{query}'.")

            response_format = str
            if response_schema:
                try:
                    response_format = create_model_from_schema(
                        response_schema,
                        "DynamicResponseModel",
                    )
                    logger.info(
                        "Dynamically created a Pydantic model for structured observation.",
                    )
                except Exception as e:
                    raise e

            result = await self._controller.observe(
                query,
                response_format=response_format,
            )

            if isinstance(result, BaseModel):
                return result.model_dump()

            return result

        async def get_action_history() -> list[dict]:
            return await self._controller.get_action_history()

        async def get_screenshots_for_action(timestamp: float) -> dict:
            return await self._controller.get_screenshots_for_action(timestamp)

        act.__doc__ = self._controller.act.__doc__
        get_action_history.__doc__ = self._controller.get_action_history.__doc__
        get_screenshots_for_action.__doc__ = (
            self._controller.get_screenshots_for_action.__doc__
        )

        act.__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    "action",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=str,
                ),
                inspect.Parameter(
                    "expectation",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=str,
                    default=None,
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
                    "response_schema",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=dict,
                    default=None,
                ),
            ],
            return_annotation=Any,
        )

        get_action_history.__signature__ = inspect.Signature(
            [],
            return_annotation=list[dict],
        )

        get_screenshots_for_action.__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    "timestamp",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=float,
                ),
            ],
            return_annotation=dict,
        )

        return {
            "act": act,
            "observe": observe,
            "get_action_history": get_action_history,
            "get_screenshots_for_action": get_screenshots_for_action,
        }

    async def act(
        self,
        description: str,
        *,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        **kwargs,
    ) -> ToolLoopPlan:
        logger.info(f"ToolLoopActor: Starting work on: '{description}'")

        if not self._main_event_loop:
            try:
                self._main_event_loop = asyncio.get_running_loop()
                logger.info(
                    f"ToolLoopActor.act captured event loop: {self._main_event_loop}",
                )
            except RuntimeError:
                logger.error(
                    "ToolLoopActor.act: No running event loop to pass to ToolLoopPlan.",
                )

        plan = ToolLoopPlan(
            task_description=description,
            tools=self._get_tools(),
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
            main_event_loop=self._main_event_loop,
            persist=kwargs.get("persist", False),
            tool_policy=kwargs.get("tool_policy"),
        )
        return plan

    async def close(self):
        logger.info("ToolLoopActor: Closing resources...")
        if self._controller:
            self._controller.stop()
