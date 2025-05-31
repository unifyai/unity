import asyncio
import enum
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional
import os
import json
import copy
import uuid
from functools import wraps

from browser_use.controller.service import Controller as BrowserUseController
from browser_use import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext as BrowserUseBrowserContext
from unity.common.llm_helpers import (
    start_async_tool_use_loop,
    SteerableToolHandle,
)
from unify import AsyncUnify
import unify

__all__ = ["BrowserUsePlanner"]

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class _BrowserPlannerState(enum.Enum):
    IDLE = enum.auto()
    RUNNING = enum.auto()
    PAUSED = enum.auto()
    COMPLETED = enum.auto()
    STOPPED = enum.auto()
    ERROR = enum.auto()


class BrowserUsePlan(SteerableToolHandle):
    """
    Represents an active plan being executed by the BrowserUsePlanner.
    Inherits from SteerableToolHandle to provide a consistent interface for interaction.
    """

    def __init__(
        self,
        task_description: str,
        initial_client: AsyncUnify,
        tools: Dict[str, Callable[..., Awaitable[str]]],
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

        self._state: _BrowserPlannerState = _BrowserPlannerState.IDLE
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
            "You are answering questions about an ongoing automated web Browse task. "
            "The main task's chat history will be provided. Answer concisely based on this history."
        )
        self._ask_client.set_system_message(self._ask_system_prompt)

        self._start()

    def _start(self):
        if self._state != _BrowserPlannerState.IDLE:
            raise RuntimeError("Plan can only be started once.")

        logger.info(
            f"BrowserUsePlan {self._task_id}: Starting with description: '{self._task_description}'",
        )
        self._state = _BrowserPlannerState.RUNNING
        self._completion_event.clear()

        # Dynamically add request_clarification_tool using the instance's queues
        current_tools = self._tools.copy()

        async def request_clarification_tool(question: str) -> str:
            logger.info(
                f"BrowserUsePlan {self._task_id}: LLM requesting clarification: '{question}'",
            )
            await self.clarification_questions.put(question)  # Use the property
            answer = await self._clar_down_q_internal.get()
            logger.info(
                f"BrowserUsePlan {self._task_id}: User provided clarification: '{answer}'",
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
            # Only set to COMPLETED if not already STOPPED or PAUSED (which are terminal/transitional states set by their respective methods)
            if self._state == _BrowserPlannerState.RUNNING:
                self._state = _BrowserPlannerState.COMPLETED
            logger.info(
                f"BrowserUsePlan {self._task_id}: Completed/Finished. Final state: {self._state.name}. Result: {self._result_str}",
            )
        except asyncio.CancelledError:
            if (
                self._state == _BrowserPlannerState.RUNNING
            ):  # If stop() or pause() was called
                # The state will be set by _stop_sync or _pause_sync
                pass
            elif (
                self._state != _BrowserPlannerState.PAUSED
                and self._state != _BrowserPlannerState.STOPPED
            ):
                self._state = (
                    _BrowserPlannerState.STOPPED
                )  # Default if cancelled for other reasons
            logger.info(
                f"BrowserUsePlan {self._task_id}: Execution was cancelled. Current state: {self._state.name}",
            )
            if self._result_str is None:  # Only set if not already set by stop()
                self._result_str = f"Task was {self._state.name.lower()}."
        except Exception as e:
            self._state = _BrowserPlannerState.ERROR
            self._error_str = str(e)
            self._result_str = f"Task failed with error: {self._error_str}"
            logger.error(
                f"BrowserUsePlan {self._task_id}: Failed with error: {e}",
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
            _BrowserPlannerState.COMPLETED,
            _BrowserPlannerState.STOPPED,
            _BrowserPlannerState.ERROR,
        )

    @property
    def clarification_questions(self) -> asyncio.Queue[str]:
        return self._clar_up_q_internal

    async def answer_clarification(self, answer: str) -> None:
        await self._clar_down_q_internal.put(answer)

    # --- Private Synchronous Logic for Steerable Methods ---
    def _sync_stop(self):
        logger.info(
            f"BrowserUsePlan {self._task_id}: Stopping (sync part). Current state: {self._state.name}",
        )
        self._state = _BrowserPlannerState.STOPPED
        if self._loop_handle:
            self._loop_handle.stop()
        else:  # Was paused or already completed/stopped
            self._completion_event.set()  # Ensure result() unblocks
        # Result string is updated in _await_completion or if already set
        if self._result_str is None:
            self._result_str = f"Plan {self._task_id} was stopped."

    def _sync_pause(self):
        logger.info(
            f"BrowserUsePlan {self._task_id}: Pausing (sync part). Current state: {self._state.name}",
        )
        self._state = _BrowserPlannerState.PAUSED
        if self._loop_handle:
            self._loop_handle.stop()  # This cancels the underlying task's loop
        else:  # Should not happen if called from RUNNING state
            logger.warning(
                f"BrowserUsePlan {self._task_id}: Pause called but no active loop_handle.",
            )
            self._completion_event.set()  # If no loop, pause is effectively immediate

    def _sync_resume(self):
        logger.info(
            f"BrowserUsePlan {self._task_id}: Resuming (sync part). Current state: {self._state.name}",
        )
        if not self._parent_chat_context_on_pause:
            logger.warning(
                f"BrowserUsePlan {self._task_id}: Resuming without a saved parent context.",
            )

        self._state = _BrowserPlannerState.RUNNING
        self._completion_event.clear()

        self._client = AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        if self._base_system_prompt:  # Restore base system prompt
            self._client.set_system_message(self._base_system_prompt)

        if self._parent_chat_context_on_pause:
            # If system prompt is first in context and also set by _base_system_prompt, skip first msg
            messages_to_load = self._parent_chat_context_on_pause
            if (
                self._base_system_prompt
                and messages_to_load
                and messages_to_load[0].get("role") == "system"
            ):
                messages_to_load = messages_to_load[1:]
            if messages_to_load:
                self._client.append_messages(messages_to_load)

        # Dynamically add request_clarification_tool using the instance's queues
        current_tools = self._tools.copy()

        async def request_clarification_tool(question: str) -> str:
            logger.info(
                f"BrowserUsePlan {self._task_id} (resumed): LLM requesting clarification: '{question}'",
            )
            await self.clarification_questions.put(question)
            answer = await self._clar_down_q_internal.get()
            logger.info(
                f"BrowserUsePlan {self._task_id} (resumed): User provided clarification: '{answer}'",
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
        logger.info(
            f"BrowserUsePlan {self._task_id}: Interjecting message: '{message}'",
        )
        await self._loop_handle.interject(message)
        return f"Interjection '{message}' sent to plan {self._task_id}."

    async def _async_ask(self, question: str) -> str:
        logger.info(f"BrowserUsePlan {self._task_id}: Answering query: '{question}'")
        current_context_to_share = []
        if self._state == _BrowserPlannerState.RUNNING and self._client:
            current_context_to_share = copy.deepcopy(self._client.messages or [])
        elif (
            self._state == _BrowserPlannerState.PAUSED
            and self._parent_chat_context_on_pause
        ):
            current_context_to_share = copy.deepcopy(self._parent_chat_context_on_pause)

        if not current_context_to_share:
            return "No context available to answer the question."

        self._ask_client.reset_messages()
        self._ask_client.set_system_message(
            self._ask_system_prompt,
        )  # Re-apply system prompt
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
                f"BrowserUsePlan {self._task_id}: Error during ask: {e}",
                exc_info=True,
            )
            return f"Error answering question: {e}"

    # --- Dynamic Public API ---
    def _is_valid_method(self, name: str) -> bool:
        if name == "stop":
            return self._state in (
                _BrowserPlannerState.RUNNING,
                _BrowserPlannerState.PAUSED,
            )
        if name == "pause":
            return self._state == _BrowserPlannerState.RUNNING
        if name == "resume":
            return self._state == _BrowserPlannerState.PAUSED
        if name == "interject":
            return self._state == _BrowserPlannerState.RUNNING
        if name == "ask":
            return self._state in (
                _BrowserPlannerState.RUNNING,
                _BrowserPlannerState.PAUSED,
            )
        return False

    def __getattr__(self, name: str) -> Callable[..., Awaitable[Any]]:
        # Map public name to private synchronous logic method
        sync_method_map = {
            "stop": self._sync_stop,
            "pause": self._sync_pause,
            "resume": self._sync_resume,
        }
        # Map public name to private asynchronous logic method
        async_method_map = {
            "interject": self._async_interject,
            "ask": self._async_ask,
        }

        if not self._is_valid_method(name):
            raise AttributeError(
                f"Method '{name}' is not valid in the current plan state '{self._state.name}'.",
            )

        if name in sync_method_map:
            sync_method_to_call = sync_method_map[name]

            @wraps(sync_method_to_call)
            async def async_wrapper(*args, **kwargs) -> str:
                sync_method_to_call(*args, **kwargs)
                # For stop and pause, ensure the loop's completion is awaited
                if name in ["stop", "pause"]:
                    await self._completion_event.wait()
                if name == "pause":
                    # After pause's sync logic and completion await, capture context
                    if self._client and self._client.messages:
                        self._parent_chat_context_on_pause = copy.deepcopy(
                            self._client.messages,
                        )
                    else:  # If client was cleared or no messages (e.g. immediate pause)
                        self._parent_chat_context_on_pause = []
                    logger.info(
                        f"BrowserUsePlan {self._task_id}: Context saved on pause: {len(self._parent_chat_context_on_pause)} messages.",
                    )

                # Provide a meaningful return string for actions that don't naturally return one from sync part
                return (
                    self._result_str
                    if name == "stop" and self._result_str
                    else f"Plan {self._task_id} {name} action initiated."
                )

            return async_wrapper
        elif name in async_method_map:
            return async_method_map[name]

        # Should not be reached if _is_valid_method is correct
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'",
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
        return sorted(list(default_attrs.union(exposed_methods)))


class BrowserUsePlanner:
    def __init__(
        self,
        base_system_prompt: str | None = "You are a helpful web-Browser assistant.",
        headless: bool = True,
        disable_browser_security: bool = False,
    ):
        self._base_system_prompt = base_system_prompt
        self._browser = Browser(
            config=BrowserConfig(
                disable_security=disable_browser_security,
                headless=headless,
            ),
        )
        self._browser_context = BrowserUseBrowserContext(browser=self._browser)
        self._bu_controller = BrowserUseController()

        # Initialize extraction_llm once
        self._extraction_llm = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        self._tools_cache: Optional[Dict[str, Callable[..., Awaitable[Any]]]] = None

    def _get_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        if self._tools_cache is None:
            self._tools_cache = self._build_tools()
        return self._tools_cache

    def _build_tools(self) -> Dict[str, Callable[..., Awaitable[str]]]:
        tools: Dict[str, Callable[..., Awaitable[str]]] = {}
        from pydantic import BaseModel
        import inspect

        for (
            action_name,
            action,
        ) in self._bu_controller.registry.registry.actions.items():
            param_model = getattr(action, "param_model", None)
            description = action.description or f"{action_name} browser action."

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

                    # Reset extraction_llm messages before each use if it's stateful and needs a clean slate
                    self._extraction_llm.reset_messages()

                    result = await self._bu_controller.registry.execute_action(
                        _action_name,
                        params,
                        browser=self._browser_context,
                        page_extraction_llm=self._extraction_llm,  # Use the shared instance
                    )
                    return (
                        getattr(result, "extracted_content", None)
                        or getattr(result, "message", "")
                        or "DONE"
                    )
                except Exception as exc:
                    logger.exception(f"BrowserUse Tool {_action_name} failed")
                    return f"ERROR: {exc!s}"

            _make_tool.__name__ = action_name
            _make_tool.__doc__ = description

            if param_model:
                raw_fields = (
                    param_model.model_fields
                    if hasattr(param_model, "model_fields")
                    else param_model.__fields__
                )
                required_params: list[inspect.Parameter] = []
                optional_params: list[inspect.Parameter] = []

                def _is_required_field(field_info) -> bool:  # Renamed to avoid conflict
                    if hasattr(field_info, "is_required"):  # Pydantic v2
                        return field_info.is_required()
                    if hasattr(field_info, "required"):  # Pydantic v1
                        return field_info.required
                    return True  # Default to required

                for (
                    fname,
                    field_info_obj,
                ) in raw_fields.items():  # Renamed to avoid conflict
                    annotation = getattr(field_info_obj, "annotation", Any)
                    param_kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
                    if _is_required_field(field_info_obj):
                        param = inspect.Parameter(
                            fname,
                            param_kind,
                            annotation=annotation,
                        )
                        required_params.append(param)
                    else:
                        default_val = getattr(
                            field_info_obj,
                            "default",
                            inspect.Parameter.empty,
                        )
                        param = inspect.Parameter(
                            fname,
                            param_kind,
                            default=default_val,
                            annotation=annotation,
                        )
                        optional_params.append(param)
                try:
                    _make_tool.__signature__ = inspect.Signature(
                        required_params + optional_params,
                        return_annotation=str,
                    )
                except Exception as e:
                    logger.warning(f"Could not build signature for {action_name}: {e}")
                    _make_tool.__signature__ = inspect.Signature(
                        [inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)],
                        return_annotation=str,
                    )
            tools[action_name] = _make_tool
        return tools

    def start(
        self,
        task_description: str,
        parent_chat_context: Optional[List[dict]] = None,
    ) -> BrowserUsePlan:
        logger.info(f"BrowserUsePlanner: Starting a new plan for: '{task_description}'")

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
            # Avoid duplicating system message if already set and present in parent_chat_context
            if (
                self._base_system_prompt
                and messages_to_load
                and messages_to_load[0].get("role") == "system"
            ):
                messages_to_load = messages_to_load[1:]
            if messages_to_load:
                plan_client.append_messages(messages_to_load)
                current_client_messages.extend(messages_to_load)

        # Ensure the tools are ready, including the dynamically added clarification tool.
        # The Plan will add its own clarification tool wrapper.
        initial_tools_for_plan = self._get_tools()

        plan = BrowserUsePlan(
            task_description=task_description,
            initial_client=plan_client,
            tools=initial_tools_for_plan,  # Planner provides the base browser tools
            base_system_prompt=self._base_system_prompt,  # Pass for resume
            parent_chat_context=current_client_messages,  # Pass current client messages for potential resume
        )
        return plan

    async def close(self):
        logger.info("BrowserUsePlanner: Closing browser...")
        # Ensure browser context and browser are closed if they were initialized
        if hasattr(self, "_browser_context") and self._browser_context:
            await self._browser_context.close()
        if hasattr(self, "_browser") and self._browser:
            await self._browser.close()
