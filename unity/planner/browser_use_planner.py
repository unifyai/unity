import asyncio
import enum
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional
import functools
import os
import json
import copy
import uuid

from browser_use.controller.service import Controller as BrowserUseController
from browser_use import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext as BrowserUseBrowserContext
from unity.common.llm_helpers import (
    start_async_tool_use_loop,
    SteerableToolHandle,
)
from .base import BasePlan, BasePlanner
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


class BrowserUsePlan(BasePlan):
    """
    Represents an active plan being executed by the BrowserUsePlanner.
    Inherits from SteerableToolHandle to provide a consistent interface for interaction.
    Methods like stop, pause, resume, interject, ask are always public,
    but will raise errors if called in an invalid state.
    The valid_tools property indicates which methods can be successfully called.
    """

    MAX_STEPS = 100

    def __init__(
        self,
        task_description: str,
        tools: Dict[str, Callable[..., Awaitable[str]]],
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        main_event_loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self._task_description = task_description
        self._tools = tools  # Tools available to the main tool loop
        self._parent_chat_context_on_pause: Optional[List[dict]] = []
        # Clarification queues for interaction with the entity that started this plan
        self._clar_up_q_internal: asyncio.Queue[str] = (
            clarification_up_q or asyncio.Queue()
        )
        self._clar_down_q_internal: asyncio.Queue[str] = (
            clarification_down_q or asyncio.Queue()
        )

        self._state: _BrowserPlannerState = _BrowserPlannerState.IDLE
        self._loop_handle: Optional[SteerableToolHandle] = (
            None  # Handle for the underlying async tool loop
        )
        self._result_str: Optional[str] = None  # Final result of the plan
        self._error_str: Optional[str] = None  # Error message if the plan failed
        self._completion_event = asyncio.Event()  # Signals plan completion/stop/error
        self._task_id = str(uuid.uuid4())  # Unique ID for this plan instance
        self._main_event_loop = main_event_loop
        self._plan_client = AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        self._plan_client.set_system_message("You are a helpful web-Browser assistant.")
        self._ask_client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )

        logger.info(
            f"BrowserUsePlan {self._task_id}: Scheduling async initialization on loop {self._main_event_loop}.",
        )
        future = asyncio.run_coroutine_threadsafe(
            self._async_init_and_start_internal_loop(),
            self._main_event_loop,
        )
        try:
            future.result(timeout=1000)
            logger.info(
                f"BrowserUsePlan {self._task_id}: Async initialization completed.",
            )
        except Exception as e:
            logger.error(
                f"BrowserUsePlan {self._task_id}: Error during async part of initialization: {e}",
                exc_info=True,
            )
            self._state = _BrowserPlannerState.ERROR
            self._error_str = f"Async initialization failed: {e}"
            self._completion_event.set()
            raise RuntimeError(
                f"BrowserUsePlan async initialization failed: {e}",
            ) from e

    async def _async_init_and_start_internal_loop(self):
        """
        Starts the internal async tool use loop that executes the plan's logic.
        This is called once during initialization.
        """
        if self._state != _BrowserPlannerState.IDLE:
            raise RuntimeError("Plan internal loop can only be started once.")

        logger.info(
            f"BrowserUsePlan {self._task_id}: Starting internal loop with description: '{self._task_description}'",
        )
        self._state = _BrowserPlannerState.RUNNING
        self._completion_event.clear()

        current_tools = self._tools.copy()

        async def request_clarification_tool_for_llm(question: str) -> str:
            logger.info(
                f"BrowserUsePlan {self._task_id}: LLM (internal loop) requesting clarification: '{question}'",
            )
            await self._clar_up_q_internal.put(question)
            answer = await self._clar_down_q_internal.get()
            logger.info(
                f"BrowserUsePlan {self._task_id}: User (via plan) provided clarification: '{answer}'",
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
            client=self._plan_client,
            message=self._task_description,
            tools=current_tools,
            parent_chat_context=self._parent_chat_context_on_pause,
            propagate_chat_context=True,
            interrupt_llm_with_interjections=True,
            log_steps=False,
            max_steps=self.MAX_STEPS,
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
            if self._state == _BrowserPlannerState.RUNNING:
                self._state = _BrowserPlannerState.COMPLETED
            logger.info(
                f"BrowserUsePlan {self._task_id}: Completed/Finished. Final state: {self._state.name}. Result: {self._result_str}",
            )
        except asyncio.CancelledError:
            if self._state == _BrowserPlannerState.RUNNING:
                self._state = _BrowserPlannerState.STOPPED
            logger.info(
                f"BrowserUsePlan {self._task_id}: Execution was cancelled. Current state: {self._state.name}",
            )
            if self._result_str is None:
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

    @functools.wraps(BasePlan.result, updated=())
    async def result(self) -> str:
        await self._completion_event.wait()
        if self._error_str:
            return f"Error: {self._error_str}"
        return (
            self._result_str
            if self._result_str is not None
            else "Plan did not produce a result or was stopped."
        )

    @functools.wraps(BasePlan.done, updated=())
    def done(self) -> bool:
        return self._state in (
            _BrowserPlannerState.COMPLETED,
            _BrowserPlannerState.STOPPED,
            _BrowserPlannerState.ERROR,
        )

    @property
    def clarification_up_q(self) -> Optional[asyncio.Queue[str]]:
        return self._clar_up_q_internal

    @property
    def clarification_down_q(self) -> Optional[asyncio.Queue[str]]:
        return self._clar_down_q_internal

    def _is_valid_method(self, name: str) -> bool:
        """Checks if a control method is valid in the current plan state."""
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

    # --- Public Control Methods ---
    @functools.wraps(BasePlan.stop, updated=())
    async def stop(self) -> str:
        try:
            if not self._is_valid_method("stop"):
                raise RuntimeError(
                    f"Plan {self._task_id} cannot be stopped in state {self._state.name}.",
                )
            logger.info(
                f"BrowserUsePlan {self._task_id}: Stopping. Current state: {self._state.name}",
            )
            self._state = _BrowserPlannerState.STOPPED
            if self._loop_handle:
                self._loop_handle.stop()
            else:
                self._completion_event.set()

            await self._completion_event.wait()

            if self._result_str is None:
                self._result_str = f"Plan {self._task_id} was stopped."
            return self._result_str
        except Exception as e:
            logger.error(
                f"BrowserUsePlan {self._task_id}: Error during stop: {e}",
                exc_info=True,
            )
            raise e

    @functools.wraps(BasePlan.pause, updated=())
    async def pause(self) -> str:
        try:
            if not self._is_valid_method("pause"):
                raise RuntimeError(
                    f"Plan {self._task_id} cannot be paused in state {self._state.name}.",
                )
            logger.info(
                f"BrowserUsePlan {self._task_id}: Pausing. Current state: {self._state.name}",
            )
            self._state = _BrowserPlannerState.PAUSED

            if self._plan_client and self._plan_client.messages:
                self._parent_chat_context_on_pause = copy.deepcopy(
                    self._plan_client.messages,
                )
            else:
                self._parent_chat_context_on_pause = []
            logger.info(
                f"BrowserUsePlan {self._task_id}: Context saved on pause: {len(self._parent_chat_context_on_pause)} messages.",
            )

            if self._loop_handle:
                self._loop_handle.stop()
                await self._completion_event.wait()
            else:
                logger.warning(
                    f"BrowserUsePlan {self._task_id}: Pause called but no active loop_handle.",
                )
                self._completion_event.set()

            return f"Plan {self._task_id} paused."
        except Exception as e:
            logger.error(
                f"BrowserUsePlan {self._task_id}: Error during pause: {e}",
                exc_info=True,
            )
            raise e

    @functools.wraps(BasePlan.resume, updated=())
    async def resume(self) -> str:
        try:
            if not self._is_valid_method("resume"):
                raise RuntimeError(
                    f"Plan {self._task_id} cannot be resumed in state {self._state.name}.",
                )
            logger.info(
                f"BrowserUsePlan {self._task_id}: Resuming. Current state: {self._state.name}",
            )

            if not self._parent_chat_context_on_pause:
                logger.warning(
                    f"BrowserUsePlan {self._task_id}: Resuming without a saved parent context.",
                )

            if self._parent_chat_context_on_pause:
                messages_to_load = self._parent_chat_context_on_pause
                if messages_to_load and messages_to_load[0].get("role") == "system":
                    messages_to_load = messages_to_load[1:]
                if messages_to_load:
                    self._plan_client.append_messages(messages_to_load)

            self._task_description = "The task was paused. Please review the history and determine the next best action or tool call."
            self._start_internal_loop()

            return f"Plan {self._task_id} resuming."
        except Exception as e:
            logger.error(
                f"BrowserUsePlan {self._task_id}: Error during resume: {e}",
                exc_info=True,
            )
            raise e

    @functools.wraps(BasePlan.interject, updated=())
    async def interject(self, message: str) -> str:
        try:
            if not self._is_valid_method("interject"):
                raise RuntimeError(
                    f"Plan {self._task_id} cannot be interjected in state {self._state.name}.",
                )
            if not self._loop_handle:
                return "Error: No active loop to interject."
            logger.info(
                f"BrowserUsePlan {self._task_id}: Interjecting message: '{message}'",
            )
            await self._loop_handle.interject(message)
            return f"Interjection '{message}' sent to plan {self._task_id}."
        except Exception as e:
            logger.error(
                f"BrowserUsePlan {self._task_id}: Error during interject: {e}",
                exc_info=True,
            )
            raise e

    @functools.wraps(BasePlan.ask, updated=())
    async def ask(self, question: str) -> str:
        try:
            if not self._is_valid_method("ask"):
                raise RuntimeError(
                    f"Cannot ask question for plan {self._task_id} in state {self._state.name}.",
                )

            logger.info(
                f"BrowserUsePlan {self._task_id}: Answering query: '{question}'",
            )
            current_context_to_share = []
            if (
                self._state == _BrowserPlannerState.RUNNING
                and self._plan_client
                and self._plan_client.messages
            ):
                current_context_to_share = copy.deepcopy(self._plan_client.messages)
            elif (
                self._state == _BrowserPlannerState.PAUSED
                and self._parent_chat_context_on_pause
            ):
                current_context_to_share = copy.deepcopy(
                    self._parent_chat_context_on_pause,
                )

            if not current_context_to_share:
                return "No context available to answer the question."

            self._ask_client.reset_messages()
            self._ask_client.set_system_message(
                "You are answering questions about an ongoing automated web Browse task. "
                "The main task's chat history will be provided. Answer concisely based on this history.",
            )
            self._ask_client.append_messages(
                [
                    {
                        "role": "system",
                        "content": f"Current task ({self._task_id}) chat history:\n{json.dumps(current_context_to_share, indent=2)}",
                    },
                    {"role": "user", "content": question},
                ],
            )
            try:
                response = await self._ask_client.generate()
                return response.strip() if isinstance(response, str) else str(response)
            except Exception as e:
                logger.error(
                    f"BrowserUsePlan {self._task_id}: Error during ask: {e}",
                    exc_info=True,
                )
                return f"Error answering question: {e}"
        except Exception as e:
            logger.error(
                f"BrowserUsePlan {self._task_id}: Error during ask: {e}",
                exc_info=True,
            )
            raise e

    @property
    @functools.wraps(BasePlan.valid_tools, updated=())
    def valid_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        tools = {}
        for method_name in ["stop", "pause", "resume", "interject", "ask"]:
            if self._is_valid_method(method_name):
                tools[method_name] = getattr(self, method_name)
        return tools


class BrowserUsePlanner(BasePlanner[BrowserUsePlan]):
    def __init__(
        self,
        headless: bool = True,
        disable_browser_security: bool = False,
    ):
        super().__init__()
        self._browser = Browser(
            config=BrowserConfig(
                disable_security=disable_browser_security,
                headless=headless,
            ),
        )
        self._browser_context = BrowserUseBrowserContext(browser=self._browser)
        self._bu_controller = BrowserUseController()

        self._extraction_llm = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        self._tools_cache: Optional[Dict[str, Callable[..., Awaitable[Any]]]] = None
        try:
            self._main_event_loop = asyncio.get_running_loop()
            logger.info(
                f"BrowserUsePlanner captured event loop: {self._main_event_loop}",
            )
        except RuntimeError as e:
            logger.error(
                "BrowserUsePlanner initialized outside of a running asyncio event loop. "
                "This may cause issues if plans are created from non-async contexts or threads "
                "without explicit loop management. Error: %s",
                e,
            )
            raise RuntimeError(
                "BrowserUsePlanner must be initialized within an active asyncio event loop.",
            ) from e

    def _get_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        """Prepares and caches the tools available for the BrowserUsePlan."""
        if self._tools_cache is None:
            self._tools_cache = self._build_tools()
        return self._tools_cache

    def _build_tools(self) -> Dict[str, Callable[..., Awaitable[str]]]:
        """
        Builds a dictionary of tools that the BrowserUsePlanner can use.
        This involves wrapping the actions from BrowserUseController.
        """
        tools: Dict[str, Callable[..., Awaitable[str]]] = {}
        from pydantic import BaseModel
        import inspect

        for (
            action_name,
            action,
        ) in self._bu_controller.registry.registry.actions.items():
            param_model = getattr(action, "param_model", None)
            description = action.description or f"{action_name} browser action."

            async def _make_tool_wrapper(
                _action_name_inner=action_name,
                _param_model_inner=param_model,
                **kwargs: Any,
            ) -> str:
                try:
                    params: Any
                    if _param_model_inner and issubclass(_param_model_inner, BaseModel):
                        params = _param_model_inner(**kwargs).model_dump()
                    else:
                        params = kwargs

                    self._extraction_llm.reset_messages()

                    result = await self._bu_controller.registry.execute_action(
                        _action_name_inner,
                        params,
                        browser=self._browser_context,
                        page_extraction_llm=self._extraction_llm,
                    )
                    return (
                        getattr(result, "extracted_content", None)
                        or getattr(result, "message", "")
                        or "DONE"
                    )
                except Exception as exc:
                    logger.exception(f"BrowserUse Tool {_action_name_inner} failed")
                    return f"ERROR: {exc!s}"

            _make_tool_wrapper.__name__ = action_name
            _make_tool_wrapper.__qualname__ = action_name
            _make_tool_wrapper.__doc__ = description

            if param_model:
                raw_fields = (
                    param_model.model_fields
                    if hasattr(param_model, "model_fields")
                    else param_model.__fields__
                )
                required_params: list[inspect.Parameter] = []
                optional_params: list[inspect.Parameter] = []

                def _is_required_field(field_info) -> bool:
                    if hasattr(field_info, "is_required"):
                        return field_info.is_required()
                    if hasattr(field_info, "required"):
                        return field_info.required
                    return True

                for fname, field_info_obj in raw_fields.items():
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
                    _make_tool_wrapper.__signature__ = inspect.Signature(
                        required_params + optional_params,
                        return_annotation=str,
                    )
                except Exception as e:
                    logger.warning(f"Could not build signature for {action_name}: {e}")
                    _make_tool_wrapper.__signature__ = inspect.Signature(
                        [inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)],
                        return_annotation=str,
                    )
            tools[action_name] = _make_tool_wrapper
        return tools

    def _make_plan(
        self,
        task_description: str,
        *,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> BrowserUsePlan:
        """
        Initiates a new plan for the given task description using browser_use tools.
        """
        logger.info(f"BrowserUsePlanner: Planning task: '{task_description}'")
        try:
            plan = BrowserUsePlan(
                task_description=task_description,
                tools=self._get_tools(),
                clarification_up_q=clarification_up_q,
                clarification_down_q=clarification_down_q,
                main_event_loop=self._main_event_loop,
            )
        except Exception as e:
            logger.error(f"BrowserUsePlanner: Error creating plan: {e}", exc_info=True)
            raise e
        return plan

    async def close(self):
        try:
            """Closes the browser and associated resources."""
            logger.info("BrowserUsePlanner: Closing browser...")
            if hasattr(self, "_browser_context") and self._browser_context:
                await self._browser_context.close()
            if hasattr(self, "_browser") and self._browser:
                await self._browser.close()
        except Exception as e:
            logger.error(f"BrowserUsePlanner: Error during close: {e}", exc_info=True)
            raise e
