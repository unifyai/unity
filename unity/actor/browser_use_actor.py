import asyncio
import enum
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional
import functools
import os
import json
import copy
import uuid

from unity.common.llm_helpers import (
    start_async_tool_use_loop,
    SteerableToolHandle,
)
from .base import BaseActiveTask, BaseActor
from unify import AsyncUnify
import unify

__all__ = ["BrowserUseActor"]

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class _BrowserActorState(enum.Enum):
    IDLE = enum.auto()
    RUNNING = enum.auto()
    PAUSED = enum.auto()
    COMPLETED = enum.auto()
    STOPPED = enum.auto()
    ERROR = enum.auto()


class BrowserUsePlan(BaseActiveTask):
    """
    Represents an active plan being executed by the BrowserUseActor.
    Inherits from SteerableToolHandle to provide a consistent interface for interaction.
    """

    MAX_STEPS = 100

    def __init__(
        self,
        task_description: str,
        tools: Dict[str, Callable[..., Awaitable[str]]],
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

        self._state: _BrowserActorState = _BrowserActorState.IDLE
        self._loop_handle: Optional[SteerableToolHandle] = None
        self._result_str: Optional[str] = None
        self._error_str: Optional[str] = None

        self._overall_plan_completion_event = asyncio.Event()
        self._resume_requested_event = asyncio.Event()

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
                    f"BrowserUsePlan {self._task_id}: Could not get running event loop and none was provided: {e}",
                    exc_info=True,
                )
                self._state = _BrowserActorState.ERROR
                self._error_str = f"Initialization failed: no event loop. {e}"
                self._overall_plan_completion_event.set()
                return

        logger.info(
            f"BrowserUsePlan {self._task_id}: Scheduling main execution manager on loop {self._main_event_loop}.",
        )
        asyncio.run_coroutine_threadsafe(
            self._manage_plan_execution(),
            self._main_event_loop,
        )

    def _get_internal_tools(self) -> Dict[str, Callable[..., Awaitable[str]]]:
        """Prepares tools for the internal LLM loop, including clarification."""
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
        return current_tools

    async def _manage_plan_execution(self):
        """
        Manages the lifecycle of the internal tool loop, including pause, resume,
        and final completion. This is the primary task for the BrowserUsePlan.
        """
        current_task_description = self._initial_task_description
        current_parent_chat_context = None
        self._state = _BrowserActorState.IDLE

        try:
            while True:
                if (
                    self._state == _BrowserActorState.STOPPED
                    or self._state == _BrowserActorState.ERROR
                ):
                    logger.info(
                        f"BrowserUsePlan {self._task_id}: Execution manager exiting due to state {self._state.name}",
                    )
                    break

                self._state = _BrowserActorState.RUNNING
                logger.info(
                    f"BrowserUsePlan {self._task_id}: Starting/Resuming internal loop with description: '{current_task_description}'",
                )

                self._plan_client.reset_messages()
                self._plan_client.reset_system_message()
                self._plan_client.set_system_message(
                    "You are a helpful web browser assistant. Use the available tools to complete the user's request",
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
                    if self._state == _BrowserActorState.RUNNING:
                        self._state = _BrowserActorState.COMPLETED
                        self._result_str = loop_result_str
                        logger.info(
                            f"BrowserUsePlan {self._task_id}: Internal loop COMPLETED. Result: {self._result_str}",
                        )
                    elif self._state == _BrowserActorState.PAUSED:
                        logger.info(
                            f"BrowserUsePlan {self._task_id}: Internal loop stopped for PAUSE.",
                        )
                    elif self._state == _BrowserActorState.STOPPED:
                        logger.info(
                            f"BrowserUsePlan {self._task_id}: Internal loop stopped for STOP.",
                        )
                        if self._result_str is None:
                            self._result_str = f"Plan {self._task_id} was stopped."

                except asyncio.CancelledError:
                    logger.info(
                        f"BrowserUsePlan {self._task_id}: Internal loop task was cancelled. Current state: {self._state.name}",
                    )
                    if self._state == _BrowserActorState.RUNNING:
                        self._state = _BrowserActorState.STOPPED
                    if self._result_str is None:
                        self._result_str = f"Plan {self._task_id} was {self._state.name.lower()} (cancelled)."
                except Exception as e:
                    logger.error(
                        f"BrowserUsePlan {self._task_id}: Internal loop failed: {e}",
                        exc_info=True,
                    )
                    self._state = _BrowserActorState.ERROR
                    self._error_str = str(e)
                    self._result_str = f"Task failed with error: {self._error_str}"

                self._loop_handle = None

                if self._state == _BrowserActorState.PAUSED:
                    logger.info(
                        f"BrowserUsePlan {self._task_id}: Execution PAUSED, awaiting resume signal.",
                    )
                    await self._resume_requested_event.wait()
                    self._resume_requested_event.clear()
                    if self._state == _BrowserActorState.STOPPED:
                        logger.info(
                            f"BrowserUsePlan {self._task_id}: Stop called while paused. Terminating.",
                        )
                        break
                    logger.info(f"BrowserUsePlan {self._task_id}: RESUMING execution.")
                    current_task_description = "The task was paused and is now resumed. Please review the history and continue."
                    current_parent_chat_context = self._parent_chat_context_on_pause
                    self._parent_chat_context_on_pause = None
                    continue
                else:
                    logger.info(
                        f"BrowserUsePlan {self._task_id}: Execution ended with state {self._state.name}. Finalizing.",
                    )
                    break

        except Exception as e:
            logger.error(
                f"BrowserUsePlan {self._task_id}: Unexpected error in _manage_plan_execution: {e}",
                exc_info=True,
            )
            if self._state not in [
                _BrowserActorState.ERROR,
                _BrowserActorState.COMPLETED,
                _BrowserActorState.STOPPED,
            ]:
                self._state = _BrowserActorState.ERROR
            if self._error_str is None:
                self._error_str = str(e)
            if self._result_str is None:
                self._result_str = (
                    f"Plan failed with unexpected error: {self._error_str}"
                )
        finally:
            logger.info(
                f"BrowserUsePlan {self._task_id}: Setting overall completion event. Final state: {self._state.name}",
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
    def clarification_up_q(self) -> Optional[asyncio.Queue[str]]:
        return self._clar_up_q_internal

    @property
    def clarification_down_q(self) -> Optional[asyncio.Queue[str]]:
        return self._clar_down_q_internal

    def _is_valid_method(self, name: str) -> bool:
        """Checks if a control method is valid in the current plan state."""
        if name == "stop":
            return self._state in (
                _BrowserActorState.RUNNING,
                _BrowserActorState.PAUSED,
                _BrowserActorState.IDLE,
            )
        if name == "pause":
            return self._state == _BrowserActorState.RUNNING
        if name == "resume":
            return self._state == _BrowserActorState.PAUSED
        if name == "interject":
            return (
                self._state == _BrowserActorState.RUNNING
                and self._loop_handle is not None
            )
        if name == "ask":
            return self._state in (
                _BrowserActorState.RUNNING,
                _BrowserActorState.PAUSED,
            )
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
            f"BrowserUsePlan {self._task_id}: Stopping. Current state: {self._state.name}",
        )
        previous_state = self._state
        self._state = _BrowserActorState.STOPPED
        self._result_str = (
            f"Plan {self._task_id} was stopped."
            if not reason
            else f"Plan {self._task_id} was stopped: {reason}"
        )

        if previous_state == _BrowserActorState.PAUSED:
            self._resume_requested_event.set()

        if self._loop_handle and not self._loop_handle.done():
            self._loop_handle.stop(reason)
        elif (
            previous_state == _BrowserActorState.IDLE
            and not self._overall_plan_completion_event.is_set()
        ):
            logger.warning(
                f"BrowserUsePlan {self._task_id}: Stop called in IDLE state. Forcing overall completion.",
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
            f"BrowserUsePlan {self._task_id}: Pausing. Current state: {self._state.name}",
        )
        self._state = _BrowserActorState.PAUSED

        if self._plan_client and self._plan_client.messages:
            self._parent_chat_context_on_pause = copy.deepcopy(
                self._plan_client.messages,
            )
            logger.info(
                f"BrowserUsePlan {self._task_id}: Context saved on pause: {len(self._parent_chat_context_on_pause)} messages.",
            )
        else:
            self._parent_chat_context_on_pause = []
            logger.info(
                f"BrowserUsePlan {self._task_id}: No active LLM context to save on pause.",
            )

        if self._loop_handle and not self._loop_handle.done():
            logger.info(
                f"BrowserUsePlan {self._task_id}: Requesting stop of current internal loop for pause.",
            )
            self._loop_handle.stop()
        else:
            logger.warning(
                f"BrowserUsePlan {self._task_id}: Pause called but no active internal loop_handle to stop.",
            )

        return f"Plan {self._task_id} paused successfully. Awaiting resume."

    @functools.wraps(BaseActiveTask.resume, updated=())
    async def resume(self) -> str:
        if not self._is_valid_method("resume"):
            raise RuntimeError(
                f"Plan {self._task_id} cannot be resumed in state {self._state.name}.",
            )

        logger.info(
            f"BrowserUsePlan {self._task_id}: Requesting resume. Current state: {self._state.name}",
        )
        self._resume_requested_event.set()
        return f"Plan {self._task_id} is resuming."

    @functools.wraps(BaseActiveTask.interject, updated=())
    async def interject(self, message: str) -> str:
        if not self._is_valid_method("interject"):
            if self._state != _BrowserActorState.RUNNING:
                return f"Error: Plan {self._task_id} is not in RUNNING state (current: {self._state.name}), cannot interject."
            if not self._loop_handle:
                return f"Error: Plan {self._task_id} is RUNNING but has no active internal loop to interject."

        logger.info(
            f"BrowserUsePlan {self._task_id}: Interjecting message: '{message}' into active internal loop.",
        )
        await self._loop_handle.interject(message)
        return f"Interjection '{message}' sent to plan {self._task_id}."

    @functools.wraps(BaseActiveTask.ask, updated=())
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
                self._state == _BrowserActorState.RUNNING
                and self._plan_client
                and self._plan_client.messages
            ):
                current_context_to_share = copy.deepcopy(self._plan_client.messages)
            elif (
                self._state == _BrowserActorState.PAUSED
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
            return f"Error answering question due to LLM failure: {e}"

    @property
    @functools.wraps(BaseActiveTask.valid_tools, updated=())
    def valid_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        tools = {}
        potential_tools = ["stop", "pause", "resume", "interject", "ask"]
        for method_name in potential_tools:
            if self._is_valid_method(method_name):
                tools[method_name] = getattr(self, method_name)
        return tools


class BrowserUseActor(BaseActor):
    def __init__(
        self,
        headless: bool = True,
        disable_browser_security: bool = False,
    ):
        from browser_use.controller.service import Controller as BrowserUseController
        from browser_use import Browser, BrowserConfig
        from browser_use.browser.context import (
            BrowserContext as BrowserUseBrowserContext,
        )

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
                f"BrowserUseActor captured event loop: {self._main_event_loop}",
            )
        except RuntimeError as e:
            logger.error(
                "BrowserUseActor initialized outside of a running asyncio event loop. "
                "This may cause issues if plans are created from non-async contexts or threads "
                "without explicit loop management. Error: %s",
                e,
            )
            self._main_event_loop = None

    def _get_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        """Prepares and caches the tools available for the BrowserUsePlan's internal loop."""
        if self._tools_cache is None:
            self._tools_cache = self._build_tools()
        return self._tools_cache

    def _build_tools(self) -> Dict[str, Callable[..., Awaitable[str]]]:
        """
        Builds a dictionary of tools that the BrowserUsePlan's internal LLM can use.
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
                    content = (
                        getattr(result, "extracted_content", None)
                        or getattr(result, "message", "")
                        or (str(result) if result is not None else "")
                    )

                    return (
                        content
                        if content
                        else "Action completed without specific content."
                    )

                except Exception as exc:
                    logger.exception(
                        f"BrowserUse Tool {_action_name_inner} failed with args {kwargs}",
                    )
                    return f"ERROR executing tool {_action_name_inner}: {exc!s}"

            _make_tool_wrapper.__name__ = action_name
            _make_tool_wrapper.__qualname__ = action_name
            _make_tool_wrapper.__doc__ = description

            if param_model and hasattr(
                param_model,
                "model_fields",
            ):
                fields = param_model.model_fields
                sig_params: list[inspect.Parameter] = []
                for fname, field_info in fields.items():
                    is_required = field_info.is_required()
                    default_val = (
                        field_info.default
                        if not is_required
                        else inspect.Parameter.empty
                    )
                    annotation = field_info.rebuild_annotation()

                    sig_params.append(
                        inspect.Parameter(
                            fname,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            default=default_val,
                            annotation=annotation,
                        ),
                    )
                try:
                    _make_tool_wrapper.__signature__ = inspect.Signature(
                        parameters=sig_params,
                        return_annotation=str,
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not build signature for {action_name} (Pydantic v2): {e}",
                    )
                    _make_tool_wrapper.__signature__ = inspect.Signature(
                        [inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)],
                        return_annotation=str,
                    )

            elif param_model and hasattr(param_model, "__fields__"):
                fields = param_model.__fields__
                sig_params = []
                for fname, field_info in fields.items():
                    is_required = getattr(field_info, "required", True)
                    default_val = (
                        field_info.default
                        if not is_required
                        else inspect.Parameter.empty
                    )
                    annotation = field_info.outer_type_

                    sig_params.append(
                        inspect.Parameter(
                            fname,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            default=default_val,
                            annotation=annotation,
                        ),
                    )
                try:
                    _make_tool_wrapper.__signature__ = inspect.Signature(
                        parameters=sig_params,
                        return_annotation=str,
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not build signature for {action_name} (Pydantic v1): {e}",
                    )
                    _make_tool_wrapper.__signature__ = inspect.Signature(
                        [inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)],
                        return_annotation=str,
                    )
            else:
                _make_tool_wrapper.__signature__ = inspect.Signature(
                    [inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)],
                    return_annotation=str,
                )

            tools[action_name] = _make_tool_wrapper
        return tools

    async def _execute_task_and_return_handle(
        self,
        task_description: str,
        *,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        **kwargs,
    ) -> BrowserUsePlan:
        """
        Initiates a new plan for the given task description using browser_use tools.
        """
        logger.info(f"BrowserUseActor: Planning task: '{task_description}'")
        if not self._main_event_loop:
            try:
                self._main_event_loop = asyncio.get_running_loop()
                logger.info(
                    f"BrowserUseActor._execute_task_and_return_handle captured event loop: {self._main_event_loop}",
                )
            except RuntimeError:
                logger.error(
                    "BrowserUseActor._execute_task_and_return_handle: No running event loop to pass to BrowserUsePlan.",
                )

        try:
            plan = BrowserUsePlan(
                task_description=task_description,
                tools=self._get_tools(),
                parent_chat_context=parent_chat_context,
                clarification_up_q=clarification_up_q,
                clarification_down_q=clarification_down_q,
                main_event_loop=self._main_event_loop,
            )
        except Exception as e:
            logger.error(f"BrowserUseActor: Error creating plan: {e}", exc_info=True)
            raise e
        return plan

    async def close(self):
        try:
            """Closes the browser and associated resources."""
            logger.info("BrowserUseActor: Closing browser...")
            if hasattr(self, "_browser_context") and self._browser_context:
                await self._browser_context.close()
            if hasattr(self, "_browser") and self._browser:
                await self._browser.close()
        except Exception as e:
            logger.error(f"BrowserUseActor: Error during close: {e}", exc_info=True)
            raise e
