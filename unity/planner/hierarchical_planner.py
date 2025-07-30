from __future__ import annotations

import ast
import asyncio
import base64
import collections
import datetime
import re
import enum
import functools
import inspect
import json
import logging
import sys
import textwrap
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
import typing
import types
import pydantic
import unify
from pydantic import BaseModel, Field
import importlib.util
import importlib.machinery
import uuid
from pathlib import Path

from unity.common.llm_helpers import (
    AsyncToolUseLoopHandle,
    start_async_tool_use_loop,
    SteerableToolHandle,
)
from unity.function_manager.function_manager import FunctionManager
from unity.planner.base import (
    BaseActiveTask,
    BasePlanner,
)
from unity.planner.action_provider import ActionProvider
import unity.planner.prompt_builders as prompt_builders

from unity.controller.browser_backends import BrowserAgentError

logger = logging.getLogger(__name__)


def format_pydantic_model(
    model: BaseModel,
    title: Optional[str] = None,
    indent: int = 0,
) -> str:
    """
    Generic pretty printer for any Pydantic model.

    Args:
        model: The Pydantic model instance to format
        title: Optional title to display above the model data
        indent: Base indentation level (number of spaces)

    Returns:
        A formatted string representation of the model
    """
    lines = []
    indent_str = " " * indent
    if title:
        lines.append(f"{indent_str}{title}:")
    else:
        lines.append(f"{indent_str}{model.__class__.__name__}:")
    model_data = model.model_dump(exclude_none=True)

    for field_name, field_value in model_data.items():
        field_indent = " " * (indent + 2)

        if isinstance(field_value, str):
            if "\n" in field_value:
                lines.append(f"{field_indent}{field_name}:")
                value_lines = field_value.split("\n")
                for value_line in value_lines:
                    lines.append(f"{field_indent}  {value_line}")
            else:
                lines.append(f"{field_indent}{field_name}: {field_value}")
        elif isinstance(field_value, (list, dict)):
            lines.append(f"{field_indent}{field_name}:")
            json_str = json.dumps(field_value, indent=2)
            for json_line in json_str.split("\n"):
                lines.append(f"{field_indent}  {json_line}")
        else:
            lines.append(f"{field_indent}{field_name}: {field_value}")

    return "\n".join(lines)


class ReplanFromParentException(Exception):
    """Raised by the @verify decorator when a function's goal is misguided."""

    def __init__(
        self,
        message,
        reason: Optional[str] = None,
        failed_interactions: Optional[List] = None,
    ):
        super().__init__(message)
        self.reason = reason if reason else message
        self.failed_interactions = failed_interactions


class _ForcedRetryException(Exception):
    """Internal exception to force a retry loop after a successful reimplementation."""


class FatalVerificationError(Exception):
    """Raised when verification results in a fatal, unrecoverable error."""


class VerificationAssessment(BaseModel):
    """Structured output for the _check_state_against_goal LLM call."""

    status: str = Field(
        ...,
        description="Outcome: 'ok', 'reimplement_local', 'replan_parent', 'fatal_error', or 'request_clarification'.",
    )
    reason: str = Field(..., description="A concise explanation for the status.")


class CourseCorrectionDecision(BaseModel):
    """A structured decision on whether course correction is needed."""

    correction_needed: bool = Field(
        ...,
        description="True if the current state deviates from the expected state and a correction script is required.",
    )
    reason: str = Field(
        ...,
        description="A brief explanation of why correction is or is not needed.",
    )
    correction_code: Optional[str] = Field(
        None,
        description="A short, self-contained Python script using the 'action_provider' to restore the browser state. This should only be provided if correction_needed is True. The script must be a single code block and contain no functions, only a sequence of `await action_provider...` calls.",
    )


class PageAnalysis(BaseModel):
    page_title: str = Field(description="The title of the current page.")
    url: str = Field(description="The current URL.")
    visible_headings: List[str] = Field(
        description="A list of all visible headings on the page.",
    )
    visible_links: List[str] = Field(
        description="A list of all visible links and their text.",
    )
    interactive_elements: List[str] = Field(
        description="A list of all buttons, input fields, and other interactive elements.",
    )


class ImplementationDecision(BaseModel):
    """A structured decision for how to proceed with a function implementation."""

    action: typing.Literal["implement_function", "replan_parent", "skip_function"] = (
        Field(
            ...,
            description="The chosen action: 'implement_function' to provide new code, 'replan_parent' to escalate the failure, or 'skip_function' to bypass the current step.",
        )
    )
    code: Optional[str] = Field(
        None,
        description="The Python code for the function. Required if action is 'implement_function'.",
    )
    reason: str = Field(
        ...,
        description="A concise justification for the chosen action. If replanning the parent, this reason will be passed up.",
    )


class _HierarchicalPlanState(enum.Enum):
    """Manages the detailed lifecycle state of a hierarchical plan."""

    IDLE = enum.auto()
    EXPLORING = enum.auto()
    RUNNING = enum.auto()
    PAUSED = enum.auto()
    PAUSED_FOR_MODIFICATION = enum.auto()
    PAUSED_FOR_ESCALATION = enum.auto()
    COMPLETED = enum.auto()
    STOPPED = enum.auto()
    ERROR = enum.auto()


async def llm_call(
    client: unify.AsyncUnify,
    prompt: str,
    screenshot: bytes | str | None = None,
) -> str:
    """
    Convenience wrapper for a simple, stateless LLM call.

    This helper automatically resets the client's message history before making
    the call to ensure no context is leaked from previous interactions.
    """
    client.reset_messages()
    content = [{"type": "text", "text": prompt}]
    if screenshot:
        if isinstance(screenshot, str):
            screenshot_b64 = screenshot
        else:
            screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{screenshot_b64}",
                },
            },
        )
    messages_to_send = [{"role": "user", "content": content}]
    return await client.generate(messages=messages_to_send)


class PlanSanitizer(ast.NodeTransformer):
    """
    AST transformer to enforce security and correctness of plan code.

    Ensures every `async def` function is decorated with `@verify`.
    """

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> ast.AsyncFunctionDef:
        """Ensures all async functions have a @verify decorator."""
        has_verify = any(
            isinstance(d, ast.Name) and d.id == "verify" for d in node.decorator_list
        )
        if not has_verify:
            node.decorator_list.insert(0, ast.Name(id="verify", ctx=ast.Load()))
        return self.generic_visit(node)


class FunctionReplacer(ast.NodeTransformer):
    """AST transformer to replace a function definition in a module."""

    def __init__(self, target_name: str, new_function_node: ast.FunctionDef):
        """
        Initializes the transformer.

        Args:
            target_name: The name of the function to replace.
            new_function_node: The new AST node for the function.
        """
        self.target_name = target_name
        self.new_function_node = new_function_node
        self.replaced = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Visits and potentially replaces a synchronous function definition."""
        if node.name == self.target_name:
            self.replaced = True
            return self.new_function_node
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        """Visits and potentially replaces an asynchronous function definition."""
        if node.name == self.target_name:
            self.replaced = True
            return self.new_function_node
        return self.generic_visit(node)


class _SteerableToolHandleProxy:
    """
    A proxy for SteerableToolHandle to intercept its method calls and log
    them for the @verify decorator. This ensures that interactions with
    handles (e.g., call_handle.ask()) are visible to the verification process.
    """

    def __init__(
        self,
        real_handle: SteerableToolHandle,
        plan: "HierarchicalPlan",
        handle_name: str,
    ):
        self._real_handle = real_handle
        self._plan = plan
        self._handle_name = handle_name

    def __getattr__(self, name: str) -> Any:
        """
        Intercepts attribute access on the handle (e.g., call_handle.ask).
        """
        real_attr = getattr(self._real_handle, name)

        if not callable(real_attr):
            return real_attr

        @functools.wraps(real_attr)
        async def async_method_wrapper(*args, **kwargs):
            interactions_log = self._plan.interaction_stack[-1]
            arg_str = ", ".join(map(repr, args))
            kwarg_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            call_repr = f"{self._handle_name}.{name}({arg_str}, {kwarg_str})"

            output = await real_attr(*args, **kwargs)

            if isinstance(output, SteerableToolHandle):
                interactions_log.append(
                    (
                        "handle_method_call",
                        call_repr,
                        f"Returned new handle: {output.__class__.__name__}",
                    ),
                )
                new_handle_name = f"{self._handle_name}_{name}"
                return _SteerableToolHandleProxy(output, self._plan, new_handle_name)
            else:
                interactions_log.append(("handle_method_call", call_repr, str(output)))
                return output

        @functools.wraps(real_attr)
        def sync_method_wrapper(*args, **kwargs):
            interactions_log = self._plan.interaction_stack[-1]
            arg_str = ", ".join(map(repr, args))
            kwarg_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            call_repr = f"{self._handle_name}.{name}({arg_str}, {kwarg_str})"

            output = real_attr(*args, **kwargs)

            if isinstance(output, SteerableToolHandle):
                interactions_log.append(
                    (
                        "handle_method_call",
                        call_repr,
                        f"Returned new handle: {output.__class__.__name__}",
                    ),
                )
                new_handle_name = f"{self._handle_name}_{name}"
                return _SteerableToolHandleProxy(output, self._plan, new_handle_name)
            else:
                interactions_log.append(("handle_method_call", call_repr, str(output)))
                return output

        if inspect.iscoroutinefunction(real_attr):
            return async_method_wrapper
        else:
            return sync_method_wrapper


class _ActionProviderProxy:
    """
    A generic proxy that wraps the real ActionProvider to intercept all tool
    calls and log them for the @verify decorator. It correctly
    handles both synchronous and asynchronous tools and ensures that handles
    returned by tools are also proxied to log subsequent interactions.
    """

    def __init__(self, real_action_provider: ActionProvider, plan: "HierarchicalPlan"):
        self._real_action_provider = real_action_provider
        self._plan = plan

    def __getattr__(self, name: str) -> Any:
        """
        This magic method is called whenever an attribute (like a tool method)
        is accessed on the proxy instance.
        """
        real_attr = getattr(self._real_action_provider, name)

        if not callable(real_attr):
            return real_attr

        @functools.wraps(real_attr)
        async def async_wrapper(*args, **kwargs):
            """Asynchronous wrapper for logging and calling async tools."""
            interactions_log = self._plan.interaction_stack[-1]

            arg_str = ", ".join(map(repr, args))
            kwarg_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            call_repr = f"action_provider.{name}({arg_str}, {kwarg_str})"

            tool_output = await real_attr(*args, **kwargs)

            if isinstance(tool_output, SteerableToolHandle):
                interactions_log.append(
                    (
                        "tool_call",
                        call_repr,
                        f"Returned handle: {tool_output.__class__.__name__}",
                    ),
                )
                handle_name = f"{name}_handle"
                return _SteerableToolHandleProxy(tool_output, self._plan, handle_name)
            else:
                if isinstance(tool_output, SteerableToolHandle):
                    final_result = await tool_output.result()
                    interactions_log.append(("tool_call", call_repr, str(final_result)))
                else:
                    interactions_log.append(("tool_call", call_repr, str(tool_output)))

                return tool_output

        @functools.wraps(real_attr)
        def sync_wrapper(*args, **kwargs):
            """Synchronous wrapper for logging and calling sync tools."""
            interactions_log = self._plan.interaction_stack[-1]

            arg_str = ", ".join(map(repr, args))
            kwarg_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            call_repr = f"action_provider.{name}({arg_str}, {kwarg_str})"

            result = real_attr(*args, **kwargs)

            if isinstance(result, SteerableToolHandle):
                interactions_log.append(
                    (
                        "tool_call",
                        call_repr,
                        f"Returned handle: {result.__class__.__name__}",
                    ),
                )
                handle_name = f"{name}_handle"
                return _SteerableToolHandleProxy(result, self._plan, handle_name)
            else:
                interactions_log.append(("tool_call", call_repr, str(result)))
                return result

        if inspect.iscoroutinefunction(real_attr):
            return async_wrapper
        else:
            return sync_wrapper


class HierarchicalPlan(BaseActiveTask):
    """
    Represents and executes a single, dynamically generated hierarchical plan.

    This class is a steerable handle managing the plan's lifecycle, including
    generation, execution, self-correction, and modification.
    """

    def __init__(
        self,
        planner: "HierarchicalPlanner",
        goal: str,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        parent_chat_context: Optional[str] = None,
        max_escalations: Optional[int] = None,
        max_local_retries: Optional[int] = None,
    ):
        """
        Initializes the Hierarchical Plan active task.

        Args:
            planner: The parent HierarchicalPlanner instance.
            goal: The high-level user goal for this plan.
            clarification_up_q: Queue for sending clarification questions to the user.
            clarification_down_q: Queue for receiving answers from the user.
            parent_chat_context: The chat context from a parent process, if any.
            max_escalations: Max number of strategic replans before pausing.
            max_local_retries: Max number of tactical retries for a function.
        """
        self.planner = planner
        self.goal = goal
        self.exploration_summary: Optional[str] = None
        self.plan_source_code: Optional[str] = None
        self.execution_namespace: Dict[str, Any] = {}
        self.call_stack: List[str] = []
        self.action_log: List[str] = []
        self.last_verified_function_name: Optional[str] = None
        self.last_verified_url: Optional[str] = None
        self.last_verified_screenshot: Optional[str | bytes] = None
        self.last_verified_page_analysis: Optional[PageAnalysis] = None
        self.function_source_map: Dict[str, str] = {}
        self.interaction_stack: List[List[Tuple[str, str, Optional[str]]]] = []
        self.escalation_count = 0
        self._is_complete = False
        self.main_loop_handle: Optional[AsyncToolUseLoopHandle] = None
        self._execution_task: Optional[asyncio.Task] = None
        self._state = _HierarchicalPlanState.IDLE
        self._completion_event = asyncio.Event()
        self._final_result_str: Optional[str] = None
        self.clarification_up_q = clarification_up_q or asyncio.Queue()
        self.clarification_down_q = clarification_down_q or asyncio.Queue()
        self.completed_functions: dict = {}
        self.skipped_functions: set = set()
        self._execution_task = asyncio.create_task(self._initialize_and_run())
        self.MAX_ESCALATIONS = max_escalations or 2
        self.MAX_LOCAL_RETRIES = max_local_retries or 3

        self._temp_file_path: Optional[Path] = None
        self._module_name: str = f"hp_plan_{uuid.uuid4().hex}"
        self._module: Optional[types.ModuleType] = None
        self._module_spec: Optional[importlib.machinery.ModuleSpec] = None

        self.main_loop_client: unify.AsyncUnify = unify.AsyncUnify("gpt-4o-mini@openai")
        self.plan_generation_client: unify.AsyncUnify = unify.AsyncUnify(
            "gemini-2.5-pro@vertex-ai",
        )
        self.verification_client: unify.AsyncUnify = unify.AsyncUnify(
            "gemini-2.5-pro@vertex-ai",
        )
        self.implementation_client: unify.AsyncUnify = unify.AsyncUnify(
            "gemini-2.5-pro@vertex-ai",
        )
        self.summarization_client: unify.AsyncUnify = unify.AsyncUnify(
            "gemini-2.5-pro@vertex-ai",
        )
        self.course_correction_client: unify.AsyncUnify = unify.AsyncUnify(
            "gemini-2.5-pro@vertex-ai",
        )
        self.modification_client: unify.AsyncUnify = unify.AsyncUnify("o4-mini@openai")
        self.exploration_client: unify.AsyncUnify = unify.AsyncUnify("o4-mini@openai")
        self.ask_client: unify.AsyncUnify = unify.AsyncUnify("gpt-4o-mini@openai")

    def _set_final_result(self, result: str):
        """Sets the final result and the completion event."""
        if not self._completion_event.is_set():
            self._final_result_str = result
            self._is_complete = True
            self._completion_event.set()

    def _set_state(self, new_state: _HierarchicalPlanState):
        """Sets the plan state and logs the transition."""
        old_state = self._state
        self._state = new_state
        self.action_log.append(f"STATE CHANGE: {old_state.name} -> {new_state.name}")

    async def _initialize_and_run(self):
        """
        Manages the entire lifecycle of the plan from initialization to completion.
        """
        self.action_log.append("Initializing plan...")
        try:
            if await self.planner._should_explore(self.goal):
                await self._perform_exploration()

            if not self._is_complete:
                self._set_state(_HierarchicalPlanState.RUNNING)

            if self.plan_source_code is None:
                self.action_log.append("Generating new plan from goal...")
                self.plan_source_code = await self.planner._generate_initial_plan(
                    plan=self,
                    goal=self.goal,
                    exploration_summary=self.exploration_summary,
                )
                self.action_log.append("Initial plan generated successfully.")
            else:
                self.action_log.append("Proceeding with existing plan source code.")

            await self.planner._prepare_execution_environment(self)
            await self._start_main_execution_loop()
        except Exception as e:
            logger.error(f"Plan initialization failed: {e}", exc_info=True)
            self._set_state(_HierarchicalPlanState.ERROR)
            self._set_final_result(f"ERROR: Plan initialization failed: {e}")


    def _create_main_loop_iterator(self):
        """
        Creates a generator that yields the main plan coroutine to be executed.

        Yields:
            The main plan's coroutine object.
        """
        main_fn_name = self._get_main_function_name()
        if not main_fn_name:
            raise RuntimeError("Could not determine main entry point 'main_plan'.")
        main_fn = self.execution_namespace[main_fn_name]
        yield main_fn()

    async def _start_main_execution_loop(self):
        """
        Starts the primary execution loop, driven by `start_async_tool_use_loop`.

        This loop uses a single tool, `_run_one_plan_step`, to advance the plan's
        execution, allowing for pausing, interjection, and control.
        """
        client = self.main_loop_client
        client.reset_messages()
        plan_iterator = self._create_main_loop_iterator()

        async def _run_one_plan_step():
            """Executes a single step of the plan, handling state transitions."""
            nonlocal plan_iterator
            if self._is_complete:
                return {
                    "status": self._state.name.lower(),
                    "message": "Plan has concluded.",
                    "force_stop": True,
                }

            if self._state in (
                _HierarchicalPlanState.PAUSED_FOR_MODIFICATION,
                _HierarchicalPlanState.PAUSED_FOR_ESCALATION,
            ):
                return {
                    "status": "paused",
                    "message": f"Execution paused for {self._state.name.lower()}.",
                }

            try:
                main_coro = next(plan_iterator)
                result = await main_coro
                self._set_state(_HierarchicalPlanState.COMPLETED)
                self.action_log.append(f"Plan completed. Result: {result}")
                self._set_final_result(f"Plan completed. Result: {result}")
                return {
                    "status": "completed",
                    "message": f"Plan finished. Result: {result}",
                    "force_stop": True,
                }
            except StopIteration:
                self._set_state(_HierarchicalPlanState.COMPLETED)
                self.action_log.append("Plan finished.")
                self._set_final_result("Plan finished.")
                return {
                    "status": "completed",
                    "message": "Plan finished.",
                    "force_stop": True,
                }
            except NotImplementedError as e:
                try:
                    function_name = self._get_unimplemented_function_name()
                    not_implemented_reason = str(e)
                    replan_reason = None
                    if not_implemented_reason:
                        replan_reason = f"Implement the function as described in its stub: '{not_implemented_reason}'"
                    await self._handle_dynamic_implementation(
                        function_name,
                        replan_reason=replan_reason,
                    )
                    plan_iterator = self._create_main_loop_iterator()
                    self.action_log.append(
                        f"Restarting main execution loop after implementing '{function_name}'",
                    )
                    return {
                        "status": "in_progress",
                        "message": f"Implemented {function_name}, retrying.",
                    }
                except Exception as e:
                    logger.error(
                        f"Failed to implement stub function: {e}",
                        exc_info=True,
                    )
                    self._set_state(_HierarchicalPlanState.ERROR)
                    self.action_log.append(
                        f"ERROR: Failed during dynamic implementation: {e}",
                    )
                    self._set_final_result(
                        f"ERROR: Failed during dynamic implementation: {e}",
                    )
                    return {"status": "error", "message": str(e), "force_stop": True}
            except ReplanFromParentException as e:
                if self.call_stack:
                    failed_function = self.call_stack.pop()
                    self.action_log.append(
                        f"Popping failed function '{failed_function}' from call stack before replan.",
                    )
                if self.interaction_stack:
                    self.interaction_stack.pop()

                self.escalation_count += 1
                self.action_log.append(
                    f"Escalation ({self.escalation_count}/{self.MAX_ESCALATIONS}): {e}",
                )

                parent_to_replan = None
                if len(self.call_stack) > 0:
                    parent_to_replan = self.call_stack[-1]
                else:
                    parent_to_replan = self._get_main_function_name()

                if not parent_to_replan:
                    raise RuntimeError("Could not determine a function to replan.")

                if self.escalation_count > self.MAX_ESCALATIONS:
                    self._set_state(_HierarchicalPlanState.PAUSED_FOR_ESCALATION)
                    err_msg = f"ESCALATION LIMIT: Max escalations ({self.MAX_ESCALATIONS}) reached. Pausing for intervention. Final reason: {e.reason}"
                    self.action_log.append(err_msg)
                    await self.clarification_up_q.put(err_msg)
                    self._set_final_result(err_msg)
                    return {
                        "status": "paused_for_escalation",
                        "message": err_msg,
                        "force_stop": True,
                    }

                await self._handle_dynamic_implementation(
                    parent_to_replan,
                    is_strategic_replan=True,
                    replan_reason=e.reason,
                    failed_interactions=e.failed_interactions,
                )
                plan_iterator = self._create_main_loop_iterator()
                return {
                    "status": "in_progress",
                    "message": f"Strategically replanned '{parent_to_replan}' due to failure in child. Retrying.",
                }
            except Exception as e:
                logger.error(f"Plan step execution failed: {e}", exc_info=True)
                self._set_state(_HierarchicalPlanState.ERROR)
                self.action_log.append(f"ERROR: Plan execution failed: {e}")
                self._set_final_result(f"ERROR: Plan execution failed: {e}")
                return {"status": "error", "message": str(e), "force_stop": True}

        def dynamic_tool_policy(step_index, tools):
            """Defines the tool usage policy for the main execution loop."""
            if self._is_complete or self._state in (
                _HierarchicalPlanState.PAUSED_FOR_MODIFICATION,
                _HierarchicalPlanState.PAUSED_FOR_ESCALATION,
            ):
                return "auto", {}
            else:
                return "required", {"_run_one_plan_step": _run_one_plan_step}

        self.main_loop_handle = start_async_tool_use_loop(
            client=client,
            message="Executing hierarchical plan...",
            tools={"_run_one_plan_step": _run_one_plan_step},
            loop_id=f"HierarchicalPlan-{self.goal[:50]}",
            max_steps=100,
            tool_policy=dynamic_tool_policy,
            interrupt_llm_with_interjections=True,
            timeout=self.planner.timeout,
        )
        await self.main_loop_handle.result()

    async def _handle_dynamic_implementation(self, function_name: str, **kwargs):
        """
        Orchestrates the dynamic implementation of a stub function based on the LLM's decision.

        Args:
            function_name: The name of the function to implement.
            **kwargs: Additional context for implementation (e.g., replan reason).
        """
        reason = kwargs.get(
            "replan_reason",
            "First-time implementation from NotImplementedError.",
        )
        self.action_log.append(
            f"IMPLEMENTATION CONTEXT for '{function_name}': {reason}",
        )

        decision = await self.planner._dynamic_implement(
            plan=self,
            function_name=function_name,
            **kwargs,
        )

        if decision.action == "implement_function":
            self.action_log.append(
                f"Decision: Implementing function '{function_name}'. Reason: {decision.reason}",
            )
            self._update_plan_with_new_code(function_name, decision.code)

        elif decision.action == "skip_function":
            self.action_log.append(
                f"Decision: Skipping function '{function_name}'. Reason: {decision.reason}",
            )
            self.skipped_functions.add(function_name)

        elif decision.action == "replan_parent":
            self.action_log.append(
                f"Decision: Escalating to replan parent of '{function_name}'. Reason: {decision.reason}",
            )

            try:
                current_index = self.call_stack.index(function_name)
                if current_index > 0:
                    parent_function_name = self.call_stack[current_index - 1]
                    self.action_log.append(
                        f"Now attempting to replan '{parent_function_name}'...",
                    )
                    await self._handle_dynamic_implementation(
                        parent_function_name,
                        is_strategic_replan=True,
                        replan_reason=decision.reason,
                    )
                else:
                    self.action_log.append(
                        f"Preparing a trace summary for full strategic replan of '{function_name}'...",
                    )

                    full_trace = "\n".join(self.action_log)
                    summary_prompt = prompt_builders.build_trace_summary_prompt(
                        goal=self.goal,
                        action_log=full_trace,
                    )
                    trace_summary = await llm_call(
                        self.summarization_client,
                        summary_prompt,
                    )
                    logger.info(f"TRACE SUMMARY:\n{trace_summary}")
                    self.action_log.append(f"TRACE SUMMARY:\n{trace_summary}")

                    await self._handle_dynamic_implementation(
                        function_name,
                        is_strategic_replan=True,
                        replan_reason=trace_summary,
                    )
            except ValueError:
                raise FatalVerificationError(
                    f"Could not find function '{function_name}' in the current call stack: {self.call_stack}",
                )

    def _get_unimplemented_function_name(self) -> str:
        """
        Inspects the traceback to find the name of the unimplemented function.

        Returns:
            The name of the function that raised NotImplementedError.
        """
        _, _, exc_tb = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_tb)[-1]
        return frame_summary.name

    def _get_main_function_name(self) -> str | None:
        """
        Parses the plan's source code to find the main entry point.

        Returns:
            The name of the main function ('main_plan') or None if not found.
        """
        try:
            tree = ast.parse(self.plan_source_code or "")
            for node in ast.walk(tree):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "main_plan"
                ):
                    return node.name
        except SyntaxError:
            return None
        return None

    def _get_node_key(self, node: ast.AST) -> Optional[tuple]:
        """
        Generates a unique, comparable key for a top-level AST node.

        This key is used to identify and replace nodes during a code merge.
        - Definitions (functions, classes) are keyed by type and name.
        - Imports are keyed by their full string representation.
        - Global assignments are keyed by the variable name.
        - Other nodes (like module-level docstrings) are keyed by their content.
        """
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ("def", node.name)
        elif isinstance(node, ast.ClassDef):
            return ("class", node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            return ("import", ast.unparse(node))
        elif isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                return ("assign", node.targets[0].id)
        return ("other", ast.unparse(node))

    def _update_plan_with_new_code(self, function_name: str, new_code: str):
        """
        Updates the plan's source code with a new function implementation using a
        generic, lossless AST merge strategy.

        This function merges all top-level statements, including
        functions, classes, imports, and global assignments, ensuring that the
        new code is seamlessly integrated without losing existing code.

        Args:
            function_name: The name of the function to replace or add.
            new_code: The full source code of the new function implementation.
        """
        self.action_log.append(
            f"Updating implementation of '{function_name}' using robust merge.",
        )
        keys_to_remove = [
            key for key in self.completed_functions if key[0] == function_name
        ]
        for key in keys_to_remove:
            self.completed_functions.pop(key, None)
        if keys_to_remove:
            logger.info(
                f"CACHE INVALIDATE: Removed {len(keys_to_remove)} cache entries for '{function_name}'.",
            )

        try:
            old_tree = ast.parse(self.plan_source_code or "pass")
            new_tree = ast.parse(textwrap.dedent(new_code))

            final_nodes = {}

            for node in old_tree.body:
                key = self._get_node_key(node)
                if key:
                    final_nodes[key] = node

            for node in new_tree.body:
                key = self._get_node_key(node)
                if key:
                    final_nodes[key] = node

            final_tree = ast.Module(body=list(final_nodes.values()), type_ignores=[])
            ast.fix_missing_locations(final_tree)

            self.plan_source_code = ast.unparse(final_tree)
            self.planner._load_plan_module(self)

        except (SyntaxError, ValueError, RuntimeError) as e:
            logger.error(
                f"Robust AST-based code update for '{function_name}' failed: {e}",
                exc_info=True,
            )
            raise

    async def resolve_escalation_with_new_goal(self, new_goal: str) -> str:
        """
        Resolves a plan that is paused due to excessive escalations by restarting with a new goal.

        Args:
            new_goal: The new, revised goal from the user.

        Returns:
            A status message.
        """
        if self._state != _HierarchicalPlanState.PAUSED_FOR_ESCALATION:
            return f"Error: Plan is not paused for escalation. Current state: {self._state.name}"

        self.action_log.append(f"Resolving escalation with new goal: '{new_goal}'")
        self.goal = new_goal
        self.escalation_count = 0
        self._is_complete = False

        if self.main_loop_handle and not self.main_loop_handle.done():
            self.main_loop_handle.stop()
            self.main_loop_handle = None

        await self._initialize_and_run()
        return f"Plan restarted with new goal: '{new_goal}'"

    async def result(self) -> str:
        """
        Waits for the plan to complete and returns its final result.

        Returns:
            The final result string of the plan.
        """
        await self._completion_event.wait()
        return (
            self._final_result_str
            or f"Plan finished in state {self._state.name} without a result."
        )

    def done(self) -> bool:
        """
        Checks if the plan has completed.

        Returns:
            True if the plan is in a terminal state, False otherwise.
        """
        return self._is_complete

    def _cleanup_temp_file(self):
        """
        Clean up temporary file and module references.

        This method ensures proper resource cleanup to prevent accumulation
        of temporary files and modules in memory.
        """
        try:
            if self._temp_file_path and self._temp_file_path.exists():
                self._temp_file_path.unlink(missing_ok=True)
                logger.debug(f"Deleted temporary plan file: {self._temp_file_path}")
                self._temp_file_path = None

            if self._module_name and self._module_name in sys.modules:
                del sys.modules[self._module_name]
                logger.debug(
                    f"Removed plan module from sys.modules: {self._module_name}",
                )

            self._module = None
            self._module_spec = None

        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    async def interject(self, message: str) -> str:
        """
        Sends an interjection message to the running plan's execution loop.

        Args:
            message: The user's interjection.

        Returns:
            A status message.
        """
        if not self._is_valid_method("interject"):
            return "Cannot interject: plan not running."
        if self.main_loop_handle:
            await self.main_loop_handle.interject(message)
            self.action_log.append(f"User interjected: '{message}'")
            return "Interjection sent."
        return "Error: No active loop to interject into."

    async def stop(self) -> str:
        """
        Stops the plan's execution permanently.

        Returns:
            A status message.
        """
        if not self._is_complete:
            self._set_state(_HierarchicalPlanState.STOPPED)
            result_str = final_result or "Plan was stopped."
            if self.main_loop_handle and not self.main_loop_handle.done():
                self.main_loop_handle.stop()
            if self._execution_task and not self._execution_task.done():
                self._execution_task.cancel()

            self.action_log.append("Plan stopped by user.")
            self._cleanup_temp_file()
            self._set_final_result(result_str)
            return result_str
        return f"Plan already in terminal state: {self._state.name}."

    async def pause(self) -> str:
        """
        Pauses the plan's execution.

        Returns:
            A status message.
        """
        if self._state == _HierarchicalPlanState.RUNNING:
            self._set_state(_HierarchicalPlanState.PAUSED)
            if self.main_loop_handle:
                self.main_loop_handle.pause()
            self.action_log.append("Plan paused by user.")
            return "Plan paused."
        return f"Cannot pause in state {self._state.name}."

    async def resume(self) -> str:
        """
        Resumes a paused plan.

        Returns:
            A status message.
        """
        if self._state == _HierarchicalPlanState.PAUSED:
            self._set_state(_HierarchicalPlanState.RUNNING)
            if self.main_loop_handle:
                self.main_loop_handle.resume()
            self.action_log.append("Plan resumed by user.")
            return "Plan resumed."
        return f"Cannot resume from state {self._state.name}."

    async def ask(self, question: str) -> str:
        """
        Asks a question about the current state of the plan.

        Args:
            question: The user's question.

        Returns:
            An answer generated by an LLM based on the plan's current context.
        """
        if not self._is_valid_method("ask"):
            return "Cannot ask: plan is not in a suitable state."

        try:
            try:
                browser_context = await self.planner.action_provider.browser.observe(
                    "Analyze the current page state and provide a structured summary of visible headings, links, and interactive elements.",
                    response_format=PageAnalysis,
                )
            except Exception as e:
                logger.warning(
                    f"Could not get browser state: {e}. No browser state will be available during ask",
                )
                browser_context = None

            context_log = "\n".join(f"- {log}" for log in self.action_log[-10:])
            prompt = prompt_builders.build_ask_prompt(
                goal=self.goal,
                state=self._state.name,
                call_stack=" -> ".join(self.call_stack) or "None",
                browser_context=browser_context,
                context_log=context_log,
                question=question,
            )
            return await llm_call(self.ask_client, prompt)
        except Exception as e:
            return f"Could not answer question. Current state: {self._state.name}. Error: {e}"

    def _is_valid_method(self, name: str) -> bool:
        """
        Checks if a given control method is valid in the current plan state.

        Args:
            name: The name of the method to check.

        Returns:
            True if the method is valid, False otherwise.
        """
        if name == "stop":
            return not self._is_complete
        if name == "pause":
            return self._state == _HierarchicalPlanState.RUNNING
        if name == "resume":
            return self._state == _HierarchicalPlanState.PAUSED
        if name == "ask":
            return self._state not in (
                _HierarchicalPlanState.IDLE,
                _HierarchicalPlanState.EXPLORING,
            )
        if name == "interject":
            return self._state == _HierarchicalPlanState.RUNNING
        if name == "modify_plan":
            return self._state in (
                _HierarchicalPlanState.PAUSED,
                _HierarchicalPlanState.RUNNING,
            )
        if name == "resolve_escalation_with_new_goal":
            return self._state == _HierarchicalPlanState.PAUSED_FOR_ESCALATION
        return False

    @property
    def valid_tools(self) -> Dict[str, Callable]:
        """
        Gets a dictionary of currently valid user-accessible controls.

        Returns:
            A mapping of public tool names to their callable methods.
        """
        tools = {}
        potential_tools = [
            "stop",
            "pause",
            "resume",
            "ask",
            "modify_plan",
            "interject",
            "resolve_escalation_with_new_goal",
        ]
        for method_name in potential_tools:
            if self._is_valid_method(method_name):
                tools[method_name] = getattr(self, method_name)
        return tools


class HierarchicalPlanner(BasePlanner):
    """
    Orchestrates task execution by generating and managing Python code.

    This planner takes a high-level goal, generates a Python script representing
    the plan, and then executes it in a controlled, self-correcting manner.
    """

    def __init__(
        self,
        function_manager: Optional["FunctionManager"] = None,
        session_connect_url: Optional[str] = None,
        headless: bool = False,
        max_escalations: Optional[int] = None,
        max_local_retries: Optional[int] = None,
        timeout: Optional[int] = 300,
        browser_mode: str = "magnitude",
    ):
        """
        Initializes the HierarchicalPlanner.

        Args:
            function_manager: Manages a library of reusable functions.
            controller: The browser controller for executing `act` and `observe`.
            session_connect_url: URL for connecting to an existing browser session.
            headless: Whether to run the browser in headless mode.
            max_escalations: Default max number of strategic replans for plans.
            max_local_retries: Default max number of tactical retries for plans.
            timeout: Default timeout for plan execution.
            browser_mode: The browser mode to use. Can be "legacy" or "magnitude".
        """
        super().__init__()
        self.function_manager = function_manager or FunctionManager()
        self.action_provider = ActionProvider(
            session_connect_url=session_connect_url,
            headless=headless,
            browser_mode=browser_mode,
        )
        self.tools = {
            name: attr
            for name, attr in inspect.getmembers(self.action_provider)
            if not name.startswith("_") and callable(attr)
        }
        self.max_escalations = max_escalations or 2
        self.max_local_retries = max_local_retries or 3
        self.timeout = timeout


    def _sanitize_code(self, code: str) -> str:
        """
        Parses, sanitizes, and unparses code to enforce security.

        Args:
            code: The Python code string to sanitize.

        Returns:
            The sanitized code string.
        """
        try:
            tree = ast.parse(code)
            sanitizer = PlanSanitizer()
            sanitized_tree = sanitizer.visit(tree)
            ast.fix_missing_locations(sanitized_tree)
            return ast.unparse(sanitized_tree)
        except SyntaxError as e:
            logger.error(f"Generated code failed sanitization: {e}")
            raise


    async def _execute_task_and_return_handle(
        self,
        task_description: str,
        *,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> HierarchicalPlan:
        """
        Creates and starts a new HierarchicalPlan active task.

        Args:
            task_description: The high-level goal for the task.
            parent_chat_context: Chat context from a parent process.
            clarification_up_q: Queue for sending clarification questions.
            clarification_down_q: Queue for receiving clarification answers.

        Returns:
            An active handle to the running HierarchicalPlan.
        """
        return HierarchicalPlan(
            planner=self,
            goal=task_description,
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
            max_escalations=self.max_escalations,
            max_local_retries=self.max_local_retries,
        )

    def _load_plan_module(self, plan: HierarchicalPlan):
        """
        Load plan source code as a module from a temporary file.
        """
        plans_dir = Path.cwd() / ".unity_plans"
        plans_dir.mkdir(exist_ok=True)

        if plan._temp_file_path is None:
            plan._temp_file_path = plans_dir / f"{plan._module_name}.py"

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = textwrap.dedent(
            f"""
        # Hierarchical Plan Script
        # Plan ID: {plan._module_name}
        # Goal: {plan.goal}
        # Last Updated: {timestamp}
        #
        # This script is auto-generated and executed by the HierarchicalPlanner.
        # It is updated dynamically during the execution lifecycle.

        """,
        )

        full_script_content = f"{header}\n{plan.plan_source_code or 'pass'}"

        plan._temp_file_path.write_text(full_script_content)
        logger.info(f"Plan source code written to: {plan._temp_file_path}")

        if plan._module is None:
            spec = importlib.util.spec_from_file_location(
                plan._module_name,
                plan._temp_file_path,
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(
                    f"Failed to create module spec for {plan._temp_file_path}",
                )

            plan._module = importlib.util.module_from_spec(spec)
            plan._module_spec = spec

            plan._module.__name__ = plan._module_name
            plan._module.__file__ = str(plan._temp_file_path)

            sys.modules[plan._module_name] = plan._module

            module_dict = plan._module.__dict__
            module_dict.update(plan.execution_namespace)

            spec.loader.exec_module(plan._module)
        else:
            if plan._module_name in sys.modules:
                del sys.modules[plan._module_name]

            spec = importlib.util.spec_from_file_location(
                plan._module_name,
                plan._temp_file_path,
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(
                    f"Failed to recreate module spec for {plan._temp_file_path}",
                )

            plan._module = importlib.util.module_from_spec(spec)
            plan._module_spec = spec

            plan._module.__name__ = plan._module_name
            plan._module.__file__ = str(plan._temp_file_path)

            plan._module.__dict__.clear()
            plan._module.__dict__.update(plan.execution_namespace)

            sys.modules[plan._module_name] = plan._module
            spec.loader.exec_module(plan._module)

        plan.execution_namespace = plan._module.__dict__

        plan.function_source_map.clear()
        if plan.plan_source_code:
            try:
                tree = ast.parse(plan.plan_source_code)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        plan.function_source_map[node.name] = ast.get_source_segment(
                            plan.plan_source_code,
                            node,
                        )
            except Exception:
                pass

    def _create_sandbox_globals(self) -> Dict[str, Any]:
        """
        Creates a dictionary of safe, sandboxed global functions for plan execution.

        Returns:
            A dictionary of globals allowed within the execution environment.
        """
        safe_builtins = {
            k: __builtins__.get(k)
            for k in [
                "print",
                "len",
                "str",
                "int",
                "float",
                "bool",
                "list",
                "dict",
                "set",
                "tuple",
                "range",
                "type",
                "bytes",
                "frozenset",
                "isinstance",
                "hasattr",
                "getattr",
                "setattr",
                "callable",
                "dir",
                "vars",
                "iter",
                "next",
                "filter",
                "map",
                "reversed",
                "enumerate",
                "zip",
                "any",
                "all",
                "sum",
                "min",
                "max",
                "abs",
                "round",
                "pow",
                "divmod",
                "sorted",
                "format",
                "chr",
                "ord",
                "Exception",
                "NotImplementedError",
                "ValueError",
                "TypeError",
                "KeyError",
                "IndexError",
                "AttributeError",
                "RuntimeError",
                "StopIteration",
                "AssertionError",
                "super",
                "property",
                "classmethod",
                "staticmethod",
                "__build_class__",
                "__name__",
                "__import__",
            ]
            if __builtins__.get(k) is not None
        }
        return {
            "__builtins__": safe_builtins,
            "asyncio": asyncio,
            "re": re,
            "json": json,
            "datetime": datetime,
            "collections": collections,
            "pydantic": pydantic,
            "BaseModel": BaseModel,
            "Field": Field,
            "typing": typing,
            "Any": Any,
            "Callable": Callable,
            "Dict": Dict,
            "List": List,
            "Optional": Optional,
            "Tuple": Tuple,
            "Set": Set,
        }

    async def _prepare_execution_environment(self, plan: HierarchicalPlan):
        """
        Prepares the sandboxed execution environment for a plan.

        This involves setting up global functions (`coms_manager`, `verify`)
        and compiling the plan's source code into the execution namespace.

        Args:
            plan: The HierarchicalPlan instance.
        """
        sandbox_globals = self._create_sandbox_globals()

        async def request_clarification_primitive(question: str) -> str:
            """Allows the plan to ask for clarification during execution."""
            await plan.clarification_up_q.put(question)
            return await plan.clarification_down_q.get()

        plan.execution_namespace.clear()
        plan.execution_namespace.update(sandbox_globals)
        plan.execution_namespace.update(
            {
                "action_provider": _ActionProviderProxy(self.action_provider, plan),
                "request_clarification": request_clarification_primitive,
                "verify": self._create_verify_decorator(plan),
                "ReplanFromParentException": ReplanFromParentException,
                "_ForcedRetryException": _ForcedRetryException,
                "FatalVerificationError": FatalVerificationError,
            },
        )

        self._load_plan_module(plan)

    def _create_verify_decorator(self, plan: HierarchicalPlan):
        """
        Creates the @verify decorator for a given plan instance.

        The decorator wraps each function in the plan to implement the
        execution, verification, and correction loop.

        Args:
            plan: The HierarchicalPlan this decorator is associated with.

        Returns:
            The configured `verify` decorator.
        """

        def verify(fn):
            """The actual decorator that wraps plan functions."""

            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                """The wrapper that performs verification and correction."""
                func_name = fn.__name__
                if func_name in plan.skipped_functions:
                    plan.action_log.append(
                        f"SKIPPING function '{func_name}' as per previous decision.",
                    )
                    plan.skipped_functions.remove(func_name)
                    return

                current_fn = plan.execution_namespace[func_name]
                try:
                    sig = inspect.signature(current_fn)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    cache_key = (func_name, frozenset(bound_args.arguments.items()))
                except (TypeError, ValueError):
                    cache_key = (func_name, str(args), str(kwargs))

                if cache_key in plan.completed_functions:
                    logger.info(
                        f"CACHE HIT: Returning cached result for '{func_name}' with args {args}, {kwargs}",
                    )
                    plan.action_log.append(
                        f"CACHE HIT: Returning cached result for '{func_name}'.",
                    )
                    return plan.completed_functions[cache_key]
                logger.info(
                    f"CACHE MISS: Proceeding with execution for '{func_name}'.",
                )

                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
                all_args = ", ".join(args_repr + kwargs_repr)
                plan.action_log.append(
                    f"-> Entering '{func_name}' with args: ({all_args})",
                )
                plan.call_stack.append(func_name)
                plan.interaction_stack.append([])
                logger.info(f"VERIFY: Entering '{func_name}'")
                try:
                    last_error_traceback = ""
                    for i in range(plan.MAX_LOCAL_RETRIES):
                        try:
                            current_fn_for_execution = plan.execution_namespace[
                                func_name
                            ]
                            func_source = plan.function_source_map.get(func_name)
                            return await self._execute_and_verify_step(
                                plan,
                                inspect.unwrap(current_fn_for_execution),
                                func_source,
                                args,
                                kwargs,
                                plan.interaction_stack[-1],
                            )
                        except _ForcedRetryException:
                            plan.action_log.append(
                                f"Retrying '{func_name}' after reimplementation.",
                            )
                            if plan.interaction_stack:
                                plan.interaction_stack[-1].clear()
                            continue
                        except (
                            ReplanFromParentException,
                            NotImplementedError,
                            FatalVerificationError,
                        ):
                            raise
                        except (BrowserAgentError, Exception) as e:
                            logger.error(
                                f"Function '{func_name}' failed on attempt {i+1}: {e}",
                                exc_info=True,
                            )
                            last_error_traceback = traceback.format_exc()

                            error_details = {
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                                "attempt": i + 1,
                                "traceback": last_error_traceback,
                            }
                            plan.action_log.append(
                                f"TOOL ERROR in '{func_name}': {json.dumps(error_details, indent=2)}",
                            )
                            try:
                                logger.info(
                                    f"Performing failure analysis for '{func_name}'...",
                                )
                                page_analysis = await self.action_provider.browser.observe(
                                    "Analyze the current page state to help debug a failure. Provide a structured summary of visible headings, links, and interactive elements.",
                                    response_format=PageAnalysis,
                                )
                                visual_context = f"**Current Page Analysis:**\n{page_analysis.model_dump_json(indent=2)}"
                                logger.info(
                                    f"Failure analysis complete. Visual context captured.",
                                )
                            except Exception as analysis_exc:
                                logger.warning(
                                    f"Could not perform visual failure analysis: {analysis_exc}",
                                )
                                visual_context = (
                                    "Could not retrieve page state for analysis."
                                )

                            replan_reason = (
                                f"The function '{func_name}' failed with an unexpected code error. "
                                f"Analyze the following traceback AND the current page state to fix the bug.\n\n"
                                f"**Traceback:**\n{traceback.format_exc()}\n\n"
                                f"**Visual Context from Browser:**\n{visual_context}"
                            )

                            await plan._handle_dynamic_implementation(
                                func_name,
                                replan_reason=replan_reason,
                            )
                            raise _ForcedRetryException(
                                "Forced retry after unexpected exception.",
                            )
                    raise ReplanFromParentException(
                        f"Function '{func_name}' failed after multiple retries.",
                        reason=last_error_traceback,
                        # TODO: failed_interaction ?
                    )
                finally:
                    completed_successfully = any(
                        key[0] == func_name for key in plan.completed_functions
                    )

                    if completed_successfully:
                        if len(plan.interaction_stack) > 1:
                            child_interactions = plan.interaction_stack.pop()
                            parent_interactions = plan.interaction_stack[-1]
                            parent_interactions.extend(child_interactions)
                            logger.debug(
                                f"Aggregated {len(child_interactions)} interactions from '{func_name}' to its parent.",
                            )
                        elif plan.interaction_stack:
                            plan.interaction_stack.pop()

                        if plan.call_stack:
                            plan.call_stack.pop()

                    exit_status = "completed" if completed_successfully else "failed"
                    plan.action_log.append(
                        f"<- Exiting '{func_name}' (status={exit_status})",
                    )

            return wrapper

        return verify

    async def _execute_and_verify_step(
        self,
        plan: HierarchicalPlan,
        fn: Callable,
        func_source: str,
        args,
        kwargs,
        interactions: list,
    ):
        """
        Executes one function call and verifies its outcome.

        Args:
            plan: The active plan instance.
            fn: The function to execute.
            func_source: The source code of the function.
            args: Positional arguments for the function.
            kwargs: Keyword arguments for the function.
            interactions: A list to log interactions within this step.
        """
        result = await fn(*args, **kwargs)
        if inspect.isawaitable(result):
            logger.warning(
                f"Function '{fn.__name__}' returned a coroutine. "
                f"This suggests a missing 'await' in the generated code. "
                f"Awaiting it now to recover.",
            )
            result = await result
        interactions_for_this_step = plan.interaction_stack[-1]
        logger.info(
            f"🕵️ VERIFICATION INPUT for '{fn.__name__}':\n"
            f"   - Purpose: {fn.__doc__ or 'N/A'}\n"
            f"   - Interactions:\n{json.dumps(interactions_for_this_step, indent=4)}",
        )
        interactions_str = json.dumps(interactions_for_this_step, indent=2)
        plan.action_log.append(
            f"VERIFICATION EVIDENCE for '{fn.__name__}':\n{interactions_str}",
        )

        final_screenshot = None
        if "action_provider.browser" in plan.plan_source_code:
            final_screenshot = await self.action_provider.browser.get_screenshot()
        assessment = await self._check_state_against_goal(
            plan,
            fn.__name__,
            fn.__doc__,
            function_source_code=func_source,
            interactions=interactions_for_this_step,
            screenshot=final_screenshot,
            function_return_value=result,
        )
        logger.info(
            f"🕵️ VERIFICATION ASSESSMENT for '{fn.__name__}':\n{format_pydantic_model(assessment, indent=2)}",
        )
        plan.action_log.append(
            f"Verification for {fn.__name__}: {assessment.status} - '{assessment.reason}'",
        )

        if assessment.status == "ok":
            try:
                sig = inspect.signature(fn)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                cache_key = (fn.__name__, frozenset(bound_args.arguments.items()))
            except (TypeError, ValueError):
                cache_key = (fn.__name__, str(args), str(kwargs))

            plan.completed_functions[cache_key] = result
            logger.info(
                f"CACHE ADD: Stored result for '{fn.__name__}' in cache.",
            )

            try:
                current_url = await self.action_provider.browser.get_current_url()
                plan.last_verified_function_name = fn.__name__
                plan.last_verified_url = current_url
                plan.last_verified_screenshot = final_screenshot
                plan.last_verified_page_analysis = await self.action_provider.browser.observe(
                    "Analyze the current page state and provide a structured summary of visible headings, links, and interactive elements.",
                    response_format=PageAnalysis,
                )
                plan.action_log.append(
                    f"STATE CAPTURE: Stored successful state after '{fn.__name__}' at URL {current_url}.",
                )
                logger.info(
                    f"STATE CAPTURE: Stored successful state after '{fn.__name__}' at URL {current_url}.",
                )
            except Exception as e:
                logger.warning(
                    f"Could not capture successful state after '{fn.__name__}': {e}",
                    exc_info=True,
                )

            if func_source and self.function_manager and fn.__name__ != "main_plan":
                try:
                    func_tree = ast.parse(func_source)
                    func_node = func_tree.body[0]

                    if isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        func_node.decorator_list = [
                            d
                            for d in func_node.decorator_list
                            if not (isinstance(d, ast.Name) and d.id == "verify")
                        ]

                    clean_func_source = ast.unparse(func_tree)
                    existing_funcs = self.function_manager.list_functions(
                        include_implementations=True,
                    )
                    is_duplicate = any(
                        data.get("implementation") == func_source
                        for data in existing_funcs.values()
                    )

                    if not is_duplicate:
                        plan.action_log.append(
                            f"Persisting verified function '{fn.__name__}' as a new skill.",
                        )
                        logger.info(
                            f"Adding function '{fn.__name__}' to FunctionManager.",
                        )
                        self.function_manager.add_functions(
                            implementations=[clean_func_source],
                        )
                    else:
                        plan.action_log.append(
                            f"Skipping persistence for '{fn.__name__}'; identical skill already exists.",
                        )
                        logger.info(
                            f"Skipping adding function '{fn.__name__}' to FunctionManager; identical function already exists.",
                        )
                except Exception as e:
                    plan.action_log.append(
                        f"WARNING: Could not persist function '{fn.__name__}': {e}",
                    )
                    logger.warning(
                        f"Could not add function '{fn.__name__}' to FunctionManager: {e}",
                    )
            return result
        elif assessment.status in ("reimplement_local", "replan_parent"):
            if plan.last_verified_screenshot:
                logger.info(
                    "Function verification failed. Assessing if course correction is needed.",
                )
                plan.action_log.append(
                    "ASSESSING STATE: Verification failed. Checking if browser state is corrupted.",
                )

                try:
                    current_url = await self.action_provider.browser.get_current_url()
                    current_page_analysis = await self.action_provider.browser.observe(
                        "Analyze the current page state for deviation assessment.",
                        response_format=PageAnalysis,
                    )

                    correction_prompt = prompt_builders.build_course_correction_prompt(
                        last_verified_function_name=plan.last_verified_function_name,
                        last_verified_url=plan.last_verified_url,
                        last_verified_page_analysis=plan.last_verified_page_analysis,
                        has_last_verified_screenshot=plan.last_verified_screenshot
                        is not None,
                        current_url=current_url,
                        current_page_analysis=current_page_analysis,
                        has_current_screenshot=final_screenshot is not None,
                        failed_function_name=fn.__name__,
                        failed_function_docstring=fn.__doc__,
                        tools=self.tools,
                    )

                    plan.course_correction_client.reset_messages()
                    content = [{"type": "text", "text": correction_prompt}]
                    if plan.last_verified_screenshot:
                        if isinstance(plan.last_verified_screenshot, str):
                            b64_before = plan.last_verified_screenshot
                        else:
                            b64_before = base64.b64encode(
                                plan.last_verified_screenshot,
                            ).decode("utf-8")
                        content.insert(
                            0,
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_before}",
                                },
                            },
                        )
                        content.insert(
                            0,
                            {
                                "type": "text",
                                "text": "This is the screenshot of the state BEFORE the failure (the last known good state):",
                            },
                        )

                    if final_screenshot:
                        if isinstance(final_screenshot, str):
                            b64_after = final_screenshot
                        else:
                            b64_after = base64.b64encode(final_screenshot).decode(
                                "utf-8",
                            )
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_after}",
                                },
                            },
                        )
                        content.insert(
                            len(content) - 1,
                            {
                                "type": "text",
                                "text": "This is the screenshot of the state AFTER the failure (the current, potentially corrupted state):",
                            },
                        )

                    plan.course_correction_client.set_response_format(
                        CourseCorrectionDecision,
                    )
                    response_str = await plan.course_correction_client.generate(
                        messages=[{"role": "user", "content": content}],
                    )
                    correction_decision = CourseCorrectionDecision.model_validate_json(
                        response_str,
                    )

                    if (
                        correction_decision.correction_needed
                        and correction_decision.correction_code
                    ):
                        plan.action_log.append(
                            f"COURSE CORRECTION: State deviated. Reason: {correction_decision.reason}. Running correction script.",
                        )
                        logger.info(
                            f"COURSE CORRECTION: State deviated. Reason: {correction_decision.reason}. Correction script generated:\n{correction_decision.correction_code}",
                        )

                        try:
                            await self._execute_course_correction(
                                plan,
                                correction_decision.correction_code,
                            )
                            plan.action_log.append(
                                "COURSE CORRECTION: State restored successfully.",
                            )
                            logger.info(
                                "COURSE CORRECTION: State restored successfully.",
                            )
                        except Exception as e:
                            logger.error(
                                f"Course correction script FAILED to execute: {e}",
                                exc_info=True,
                            )
                            plan.action_log.append(
                                f"CRITICAL: Course correction script FAILED. Proceeding from potentially corrupted state. Error: {e}",
                            )
                    else:
                        plan.action_log.append(
                            "COURSE CORRECTION: No state deviation detected. Proceeding with reimplementation directly.",
                        )

                except Exception as e:
                    logger.error(
                        f"Course correction assessment failed: {e}",
                        exc_info=True,
                    )
                    plan.action_log.append(
                        f"WARNING: Course correction assessment failed: {e}. Proceeding with reimplementation from current state.",
                    )
                finally:
                    plan.course_correction_client.reset_response_format()

            if assessment.status == "replan_parent":
                raise ReplanFromParentException(
                    f"Strategic failure in '{fn.__name__}': {assessment.reason}",
                    failed_interactions=interactions,
                )
            else:
                await plan._handle_dynamic_implementation(
                    fn.__name__,
                    replan_reason=assessment.reason,
                    failed_interactions=interactions,
                )
                raise _ForcedRetryException("Forced retry after local reimplementation")

        elif assessment.status == "fatal_error":
            raise FatalVerificationError(
                f"Fatal error in '{fn.__name__}': {assessment.reason}",
            )

    async def _generate_initial_plan(
        self,
        plan: HierarchicalPlan,
        goal: str,
    ) -> str:
        """
        Generates the initial Python script for the plan from a user goal.

        Args:
            plan: The HierarchicalPlan instance to generate the initial plan for.
            goal: The high-level user goal.

        Returns:
            A string containing the generated Python code for the plan.
        """
        max_retries = 3
        last_error = ""
        for attempt in range(max_retries):
            try:
                if self.function_manager:
                    try:
                        relevant_functions = (
                            self.function_manager.search_functions_by_similarity(
                                query=goal,
                                n=5,
                            )
                        )
                        existing_functions = {f["name"]: f for f in relevant_functions}
                    except Exception as e:
                        logger.warning(
                            f"Could not retrieve functions from FunctionManager: {e}",
                        )
                        existing_functions = {}
                else:
                    existing_functions = {}

                prompt = prompt_builders.build_initial_plan_prompt(
                    goal=goal,
                    tools=self.tools,
                    existing_functions=existing_functions,
                    retry_msg=(
                        ""
                        if attempt == 0
                        else f"Last attempt failed: {last_error}. Please fix."
                    ),
                )
                response = await llm_call(plan.plan_generation_client, prompt)
                code = (
                    response.strip().replace("```python", "").replace("```", "").strip()
                )
                logger.debug(
                    f"LLM response for initial plan (attempt {attempt+1}):\n\n--- LLM RAW RESPONSE START ---\n{response}\n--- LLM RAW RESPONSE END ---\n\n",
                )

                return self._sanitize_code(code)

            except SyntaxError as e:
                last_error = f"{e}\nProblematic Code:\n---\n{code}\n---"
                logger.error(
                    f"Attempt {attempt+1} to generate plan failed. Reason: {last_error}",
                )
                if attempt == max_retries - 1:
                    raise
        raise RuntimeError("Failed to generate a valid plan after multiple retries.")

    async def _dynamic_implement(
        self,
        plan: HierarchicalPlan,
        function_name: str,
        **kwargs,
    ) -> ImplementationDecision:
        """
        Generates and returns an ImplementationDecision for a stub function.
        Includes a retry loop to handle LLM-generated syntax errors.
        """
        is_browser_task = "action_provider.browser" in plan.plan_source_code

        max_retries = 3
        last_syntax_error = ""

        for attempt in range(max_retries):
            replan_reason = kwargs.get(
                "replan_reason",
                "First-time implementation from NotImplementedError.",
            )
            if last_syntax_error:
                replan_reason += (
                    "\n\nCRITICAL: Your previous attempt to generate code failed with a "
                    f"SyntaxError. You MUST fix this error. Details:\n{last_syntax_error}"
                )

            browser_state = None
            browser_screenshot = None
            if is_browser_task:
                try:
                    browser_state = await self.action_provider.browser.observe(
                        "Analyze the current page and provide a structured summary of its content.",
                        response_format=PageAnalysis,
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not get browser state: {e}. No browser state will be available during reimplementation of {function_name}",
                    )
                    browser_state = None

                plan.action_log.append(
                    f"Browser State during dynamic implementation: {browser_state}",
                )
                browser_screenshot = await self.action_provider.browser.get_screenshot()

            docstring = (
                inspect.getdoc(plan.execution_namespace[function_name])
                or "No docstring provided."
            )
            func_sig = inspect.signature(plan.execution_namespace[function_name])
            parent_code = (
                plan.function_source_map.get(plan.call_stack[-2], "")
                if len(plan.call_stack) > 1
                else "N/A (This is a top-level function call)"
            )

            prompt = prompt_builders.build_dynamic_implement_prompt(
                full_plan_source=plan.plan_source_code or "",
                call_stack=plan.call_stack,
                function_name=function_name,
                function_sig=func_sig,
                function_docstring=docstring,
                parent_code=parent_code,
                browser_state=browser_state,
                has_browser_screenshot=browser_screenshot is not None,
                replan_context=replan_reason,
                tools=self.tools,
            )
            plan.implementation_client.set_response_format(ImplementationDecision)

            try:
                response_str = await llm_call(
                    plan.implementation_client,
                    prompt,
                    screenshot=browser_screenshot,
                )
                decision = ImplementationDecision.model_validate_json(response_str)
                logger.debug(
                    f"\n{format_pydantic_model(decision, title='IMPLEMENTATION DECISION', indent=2)}",
                )
                if decision.action == "implement_function":
                    if not decision.code:
                        raise ValueError(
                            "Action 'implement_function' requires the 'code' field.",
                        )

                    try:
                        clean_code = (
                            decision.code.strip()
                            .replace("```python", "")
                            .replace("```", "")
                            .strip()
                        )
                        decision.code = self._sanitize_code(clean_code)
                        return decision
                    except SyntaxError as e:
                        last_syntax_error = f"Invalid Python code provided.\nError: {e}\nProblematic Code Snippet:\n---\n{decision.code}\n---"
                        logger.error(
                            f"Attempt {attempt + 1} failed: {last_syntax_error}",
                        )
                        if attempt == max_retries - 1:
                            raise e
                        continue

                return decision

            finally:
                plan.implementation_client.reset_response_format()

        raise RuntimeError(
            "Failed to generate a valid implementation after multiple retries.",
        )

    async def _check_state_against_goal(
        self,
        plan: HierarchicalPlan,
        function_name: str,
        function_docstring: str | None,
        function_source_code: str | None,
        interactions: list,
        screenshot: bytes | str | None = None,
        function_return_value: Any = None,
    ) -> VerificationAssessment:
        """
        Uses an LLM to assess if a function's execution achieved its goal.

        Args:
            plan: The active plan instance.
            function_name: The name of the function being verified.
            function_docstring: The docstring of the function.
            function_source_code: The source code of the function.
            interactions: A log of interactions that occurred.
            function_return_value: The return value of the function.
            screenshot: The screenshot of the current state of the browser.

        Returns:
            A VerificationAssessment object with the outcome.
        """
        prompt = prompt_builders.build_verification_prompt(
            goal=plan.goal,
            function_name=function_name,
            function_docstring=function_docstring,
            function_source_code=function_source_code,
            interactions=interactions,
            has_browser_screenshot=screenshot is not None,
            function_return_value=function_return_value,
        )

        plan.verification_client.set_response_format(VerificationAssessment)

        try:
            response_str = await llm_call(
                plan.verification_client,
                prompt,
                screenshot=screenshot,
            )
            assessment = VerificationAssessment.model_validate_json(response_str)
            return assessment
        except Exception as e:
            logger.error(
                f"Failed to parse verification assessment: {e}. Raw response: {response_str if 'response_str' in locals() else 'N/A'}",
                exc_info=True,
            )
            return VerificationAssessment(
                status="fatal_error",
                reason=f"LLM provided malformed assessment: {str(e)}",
            )
        finally:
            plan.verification_client.reset_response_format()

    async def _execute_course_correction(self, plan: HierarchicalPlan, code: str):
        """
        Executes a temporary, dynamically generated script from a file to correct the browser state.

        Args:
            plan: The active HierarchicalPlan instance.
            code: The Python code snippet to execute.
        """
        failed_function_name = (
            plan.call_stack[-1] if plan.call_stack else "unknown_function"
        )
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        correction_filename = f"correction_for_{failed_function_name}_{timestamp}.py"
        plans_dir = Path.cwd() / ".unity_plans"
        plans_dir.mkdir(exist_ok=True)
        correction_file_path = plans_dir / correction_filename

        script_to_write = f"""
# Auto-generated course correction script
# Triggered by failure in: {failed_function_name}
# Executed at: {timestamp}
#
# Goal: Restore browser state to the one left by '{plan.last_verified_function_name}'
# Target URL: {plan.last_verified_url}

import asyncio
import textwrap

# The 'action_provider' is injected into the execution namespace by the planner.

async def course_correction_plan():
    # This is a sequence of actions to restore the state.
    print("--- Starting Course Correction ---")
{textwrap.indent(code, '    ')}
    print("--- Course Correction Finished ---")

"""
        correction_file_path.write_text(textwrap.dedent(script_to_write).strip())

        logger.info(
            f"Executing course correction script. See '{correction_file_path}' for details.",
        )
        plan.action_log.append(
            f"Saved course correction script to '{correction_file_path}'.",
        )

        exec_namespace = plan.execution_namespace

        script_content = correction_file_path.read_text()
        exec(script_content, exec_namespace)

        correction_func = exec_namespace["course_correction_plan"]
        await correction_func()

    async def close(self):
        """Shuts down the planner and its associated resources gracefully."""
        if self._active_task and not self._active_task.done():
            if hasattr(self._active_task, "_cleanup_temp_file"):
                self._active_task._cleanup_temp_file()
            await self._active_task.stop()
        self.action_provider.browser.stop()
