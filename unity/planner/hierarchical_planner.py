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
import pydantic
import unify
from pydantic import BaseModel, Field

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

from unity.controller.controller import InvalidActionError
from unity.controller.browser_backends import BrowserAgentError

logger = logging.getLogger(__name__)


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

    1. Disallows `import` and `import from` statements.
    2. Ensures every `async def` function is decorated with `@verify`.
    """

    def visit_Import(self, node: ast.Import) -> Any:
        """Removes all `import <module>` statements."""
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        """Removes all `from <module> import ...` statements."""
        return None

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
    calls and log them for the @verify decorator. Itcorrectly
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
        self.completed_functions: set = set()
        self.skipped_functions: set = set()
        self._execution_task = asyncio.create_task(self._initialize_and_run())
        self.MAX_ESCALATIONS = max_escalations
        self.MAX_LOCAL_RETRIES = max_local_retries

    def _set_final_result(self, result: str):
        """Sets the final result and the completion event."""
        if not self._completion_event.is_set():
            self._final_result_str = result
            self._is_complete = True
            self._completion_event.set()

    async def _initialize_and_run(self):
        """
        Manages the entire lifecycle of the plan from initialization to completion.
        """
        self.action_log.append("Initializing plan...")
        try:
            if await self.planner._should_explore(self.goal):
                await self._perform_exploration()

            if not self._is_complete:
                self._state = _HierarchicalPlanState.RUNNING

            if self.plan_source_code is None:
                self.action_log.append("Generating new plan from goal...")
                self.plan_source_code = await self.planner._generate_initial_plan(
                    self.goal,
                    self.exploration_summary,
                )
                self.action_log.append("Initial plan generated successfully.")
            else:
                self.action_log.append("Proceeding with existing plan source code.")

            await self.planner._prepare_execution_environment(self)
            await self._start_main_execution_loop()
        except Exception as e:
            logger.error(f"Plan initialization failed: {e}", exc_info=True)
            self._state = _HierarchicalPlanState.ERROR
            self._set_final_result(f"ERROR: Plan initialization failed: {e}")

    async def _perform_exploration(self, function_purpose: str):
        """
        Runs an interactive conversational loop to gather information for a specific function.
        """
        self._state = _HierarchicalPlanState.EXPLORING
        self.action_log.append(
            f"Starting exploration for task: '{function_purpose}'...",
        )
        try:
            research_prompt = prompt_builders.build_exploration_prompt(
                function_purpose=function_purpose,
                overall_goal=self.goal,
                tools=self.planner.tools,
            )

            client = self.planner.exploration_client
            client.reset_messages()
            client.set_system_message(research_prompt)

            exploration_loop_handle = start_async_tool_use_loop(
                client=client,
                message="Begin your research based on the main objective.",
                tools=self.planner.tools,
                loop_id="FunctionExplorationPhase",
                max_steps=10,
                timeout=self.planner.timeout,
            )

            summary = await exploration_loop_handle.result()
            self.action_log.append("Exploration completed.")
            self.action_log.append(f"Exploration Summary: {summary}")
            return summary

        except Exception as e:
            logger.error(f"Exploration phase failed: {e}", exc_info=True)
            self.action_log.append(
                f"WARNING: Exploration failed: {e}. Proceeding without extra context.",
            )
            return None
        finally:
            self._state = _HierarchicalPlanState.RUNNING

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
        client = self.planner.main_loop_client
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
                self._state = _HierarchicalPlanState.COMPLETED
                self.action_log.append(f"Plan completed. Result: {result}")
                self._set_final_result(f"Plan completed. Result: {result}")
                return {
                    "status": "completed",
                    "message": f"Plan finished. Result: {result}",
                    "force_stop": True,
                }
            except StopIteration:
                self._state = _HierarchicalPlanState.COMPLETED
                self.action_log.append("Plan finished.")
                self._set_final_result("Plan finished.")
                return {
                    "status": "completed",
                    "message": "Plan finished.",
                    "force_stop": True,
                }
            except NotImplementedError:
                try:
                    function_name = self._get_unimplemented_function_name()
                    await self._handle_dynamic_implementation(function_name)
                    plan_iterator = self._create_main_loop_iterator()
                    return {
                        "status": "in_progress",
                        "message": f"Implemented {function_name}, retrying.",
                    }
                except Exception as e:
                    logger.error(
                        f"Failed to implement stub function: {e}",
                        exc_info=True,
                    )
                    self._state = _HierarchicalPlanState.ERROR
                    self.action_log.append(
                        f"ERROR: Failed during dynamic implementation: {e}",
                    )
                    self._set_final_result(
                        f"ERROR: Failed during dynamic implementation: {e}",
                    )
                    return {"status": "error", "message": str(e), "force_stop": True}
            except ReplanFromParentException as e:
                self.escalation_count += 1
                self.action_log.append(
                    f"Escalation ({self.escalation_count}/{self.MAX_ESCALATIONS}): {e}",
                )

                parent_to_replan = None
                if len(self.call_stack) > 1:
                    parent_to_replan = self.call_stack[-2]
                else:
                    parent_to_replan = self._get_main_function_name()

                if not parent_to_replan:
                    raise RuntimeError("Could not determine a function to replan.")

                if self.escalation_count > self.MAX_ESCALATIONS:
                    self._state = _HierarchicalPlanState.PAUSED_FOR_ESCALATION
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
                self._state = _HierarchicalPlanState.ERROR
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
                    raise FatalVerificationError(
                        f"Cannot replan parent of '{function_name}' as it is a top-level function. Reason: {decision.reason}",
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

    def _update_plan_with_new_code(self, function_name: str, new_code: str):
        """
        Updates the plan's source code with a new function implementation using AST.

        Args:
            function_name: The name of the function to replace or add.
            new_code: The full source code of the new function implementation.
        """
        keys_to_remove = {
            key for key in self.completed_functions if key[0] == function_name
        }
        for key in keys_to_remove:
            self.completed_functions.remove(key)
        if keys_to_remove:
            logger.info(
                f"CACHE INVALIDATE: Removed {len(keys_to_remove)} cache entries for '{function_name}'.",
            )

        try:
            old_tree = ast.parse(self.plan_source_code or "pass")
            new_tree = ast.parse(textwrap.dedent(new_code))

            old_defs = {
                node.name: node
                for node in old_tree.body
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                )
            }

            new_defs = {
                node.name: node
                for node in new_tree.body
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                )
            }

            if function_name not in new_defs:
                raise ValueError(
                    f"The new code block from the LLM does not contain the required "
                    f"function '{function_name}'.",
                )

            old_defs.update(new_defs)

            final_body = list(old_defs.values())
            final_tree = ast.Module(body=final_body, type_ignores=[])
            ast.fix_missing_locations(final_tree)

            self.plan_source_code = ast.unparse(final_tree)

            self.function_source_map.clear()
            fresh_tree = ast.parse(self.plan_source_code)
            for node in ast.walk(fresh_tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    try:
                        self.function_source_map[node.name] = ast.get_source_segment(
                            self.plan_source_code,
                            node,
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to get source segment for '{node.name}': {e}",
                            exc_info=True,
                        )
                        raise e

            exec(
                compile(self.plan_source_code, "<string>", "exec"),
                self.execution_namespace,
            )
        except (SyntaxError, ValueError, RuntimeError) as e:
            logger.error(
                f"AST-based code update for '{function_name}' failed: {e}",
                exc_info=True,
            )
            raise

    async def modify_plan(self, modification_request: str) -> str:
        """
        Modifies the current plan based on a user request.

        This involves pausing, rewriting the code, generating a course-correction
        script, and resuming execution.

        Args:
            modification_request: The user's instruction for how to change the plan.

        Returns:
            A status message indicating the outcome of the modification.
        """
        if not self._is_valid_method("modify_plan"):
            return f"Plan cannot be modified in state: {self._state.name}"

        original_source_code = self.plan_source_code
        self.action_log.append(f"Modification requested: '{modification_request}'")
        self._state = _HierarchicalPlanState.PAUSED_FOR_MODIFICATION

        if self._execution_task and not self._execution_task.done():
            self._execution_task.cancel()
            try:
                await self._execution_task
            except asyncio.CancelledError:
                pass
        if self.main_loop_handle:
            self.main_loop_handle.stop()
            self.main_loop_handle = None

        try:
            new_source_code = await self.planner._perform_plan_surgery(
                self.plan_source_code or "",
                modification_request,
            )
            correction_script = await self.planner._generate_course_correction_script(
                self.plan_source_code or "",
                new_source_code,
            )
            if correction_script:
                await self._execute_correction_script(
                    correction_script,
                    new_source_code,
                )

            self.plan_source_code = new_source_code
            self.action_log.append("Plan successfully modified.")

            if self.completed_functions:
                logger.info(
                    "CACHE INVALIDATE: Clearing entire cache due to plan modification.",
                )
                self.completed_functions.clear()

            self.escalation_count = 0
            self._is_complete = False
            self.exploration_summary = None

            self._execution_task = asyncio.create_task(self._initialize_and_run())

            return "Plan modified and resumed successfully."

        except Exception as e:
            logger.error(f"Failed to modify plan, rolling back: {e}", exc_info=True)
            self.action_log.append(f"ERROR: Failed to modify plan, rolling back. {e}")

            self.plan_source_code = original_source_code

            self.escalation_count = 0
            self._is_complete = False
            self._state = _HierarchicalPlanState.RUNNING
            self._execution_task = asyncio.create_task(self._initialize_and_run())

            return "Failed to modify the plan. Rolled back to previous version and resumed."

    async def _execute_correction_script(self, script: str, new_plan_code: str):
        """
        Executes a course-correction script to align the browser state with a new plan.

        Args:
            script: The Python script to execute for course correction.
            new_plan_code: The source code of the new plan for context.
        """
        logger.info("Executing course correction script...")
        self.action_log.append("Executing course correction script.")

        if self.completed_functions:
            logger.info(
                "CACHE INVALIDATE: Clearing entire cache due to course correction.",
            )
            self.completed_functions.clear()

        interactions = []
        try:
            correction_namespace = self.planner._create_sandbox_globals()
            correction_namespace["action_provider"] = self.planner.action_provider

            correction_namespace["verify"] = self.planner._create_verify_decorator(self)
            correction_namespace["ReplanFromParentException"] = (
                ReplanFromParentException
            )
            correction_namespace["_ForcedRetryException"] = _ForcedRetryException

            sanitized_script = self.planner._sanitize_code(script)

            exec(
                compile(sanitized_script, "<correction_script>", "exec"),
                correction_namespace,
            )
            correction_fn = correction_namespace.get("course_correction_main")
            if not correction_fn or not asyncio.iscoroutinefunction(correction_fn):
                raise RuntimeError(
                    "Script must define 'async def course_correction_main'.",
                )

            self.interaction_stack.append(interactions)
            await asyncio.wait_for(correction_fn(), timeout=60.0)

            assessment = await self.planner._check_state_against_goal(
                self,
                function_name="course_correction",
                function_docstring=f"Align state for new plan: {new_plan_code[:200]}...",
                interactions=interactions,
            )
            if assessment.status != "ok":
                raise RuntimeError(
                    f"Course correction failed verification: {assessment.reason}",
                )

            self.action_log.append("Course correction finished and verified.")
        except Exception as e:
            self.action_log.append(f"ERROR: Course correction failed: {e}")
            raise RuntimeError(f"Course correction script failed: {e}") from e
        finally:
            if self.interaction_stack:
                self.interaction_stack.pop()

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
            self._state = _HierarchicalPlanState.STOPPED
            result_str = "Plan was stopped."
            if self.main_loop_handle and not self.main_loop_handle.done():
                self.main_loop_handle.stop()
            if self._execution_task and not self._execution_task.done():
                self._execution_task.cancel()

            self.action_log.append("Plan stopped by user.")
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
            self._state = _HierarchicalPlanState.PAUSED
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
            self._state = _HierarchicalPlanState.RUNNING
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
            async with self.planner.coms_manager.start_browser_session() as browser_handle:
                browser_context = await browser_handle.observe(
                    "Summarize the current page.",
                )

            context_log = "\n".join(f"- {log}" for log in self.action_log[-10:])
            prompt = prompt_builders.build_ask_prompt(
                goal=self.goal,
                state=self._state.name,
                call_stack=" -> ".join(self.call_stack) or "None",
                browser_context=browser_context,
                context_log=context_log,
                question=question,
            )
            return await llm_call(self.planner.ask_client, prompt)
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
        self.max_escalations = max_escalations or 3
        self.max_local_retries = max_local_retries or 2
        self.timeout = timeout

        self.main_loop_client: unify.AsyncUnify = unify.AsyncUnify("gpt-4o-mini@openai")
        self.plan_generation_client: unify.AsyncUnify = unify.AsyncUnify(
            "o4-mini@openai",
        )
        self.verification_client: unify.AsyncUnify = unify.AsyncUnify("o4-mini@openai")
        self.implementation_client: unify.AsyncUnify = unify.AsyncUnify(
            "o4-mini@openai",
        )
        self.modification_client: unify.AsyncUnify = unify.AsyncUnify("o4-mini@openai")
        self.exploration_client: unify.AsyncUnify = unify.AsyncUnify("o4-mini@openai")
        self.ask_client: unify.AsyncUnify = unify.AsyncUnify("gpt-4o-mini@openai")

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

    async def _should_explore(self, goal: str) -> bool:
        """
        Uses an LLM to assess if the goal is ambiguous and requires exploration.

        Args:
            goal: The user's goal.

        Returns:
            True if exploration is needed, False otherwise.
        """
        return False
        prompt = build_should_explore_prompt(goal)
        response = await llm_call(self.exploration_client, prompt)
        logger.info(f"Exploration assessment for goal '{goal}': {response.strip()}")
        return "EXPLORE" in response.strip().upper()

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

        exec(
            compile(plan.plan_source_code or "pass", "<string>", "exec"),
            plan.execution_namespace,
        )

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
                        f"CACHE HIT: Skipping already completed call to '{func_name}' with args {args}, {kwargs}",
                    )
                    return
                logger.info(
                    f"CACHE MISS: Proceeding with execution for '{func_name}'.",
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
                        except InvalidActionError as e:
                            logger.warning(
                                f"Caught InvalidActionError in '{func_name}'. Forcing local reimplementation.",
                            )
                            replan_reason = (
                                f"The function failed because it tried to execute an invalid browser action. "
                                f"The instruction it gave resulted in the error: '{e}'.\n\n"
                                f"The 'browser_act' tool can only perform atomic actions like clicking, typing, navigating, or scrolling. "
                                f"Please rewrite the function to use only valid, direct actions."
                            )
                            await plan._handle_dynamic_implementation(
                                func_name,
                                replan_reason=replan_reason,
                            )
                            raise _ForcedRetryException(
                                "Forced retry after invalid action.",
                            )
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
                    if plan.call_stack:
                        plan.call_stack.pop()
                    if plan.interaction_stack:
                        plan.interaction_stack.pop()

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
        all_interactions = [
            item for sublist in plan.interaction_stack for item in sublist
        ]
        logger.info(
            f"🕵️ VERIFICATION INPUT for '{fn.__name__}':\n"
            f"   - Purpose: {fn.__doc__ or 'N/A'}\n"
            f"   - Interactions:\n{json.dumps(all_interactions, indent=4)}",
        )
        final_screenshot = None
        if "action_provider.browser" in plan.plan_source_code:
            final_screenshot = await self.action_provider.browser.get_screenshot()
        assessment = await self._check_state_against_goal(
            plan,
            fn.__name__,
            fn.__doc__,
            all_interactions,
            screenshot=final_screenshot,
        )
        logger.info(
            f"🕵️ VERIFICATION ASSESSMENT for '{fn.__name__}': {assessment.model_dump_json(indent=2)}",
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

            plan.completed_functions.add(cache_key)
            logger.info(
                f"CACHE ADD: Added call to '{fn.__name__}' to completed functions cache.",
            )

            if len(plan.interaction_stack) > 1:
                child_interactions = plan.interaction_stack[-1]
                parent_interactions = plan.interaction_stack[-2]
                parent_interactions.extend(child_interactions)

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
        elif assessment.status == "reimplement_local":
            await plan._handle_dynamic_implementation(
                fn.__name__,
                replan_reason=assessment.reason,
                failed_interactions=interactions,
            )
            raise _ForcedRetryException("Forced retry after local reimplementation")
        elif assessment.status == "replan_parent":
            raise ReplanFromParentException(
                f"Strategic failure in '{fn.__name__}': {assessment.reason}",
                failed_interactions=interactions,
            )
        else:
            raise FatalVerificationError(
                f"Fatal error in '{fn.__name__}': {assessment.reason}",
            )

    async def _generate_initial_plan(
        self,
        goal: str,
        exploration_summary: Optional[str] = None,
    ) -> str:
        """
        Generates the initial Python script for the plan from a user goal.

        Args:
            goal: The high-level user goal.
            exploration_summary: A summary from a preceding exploration phase.

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
                    exploration_summary=exploration_summary,
                )
                response = await llm_call(self.plan_generation_client, prompt)
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
        Generates and returns an ImplementationDecision for a stub function in a single LLM call.

        Args:
            plan: The active plan instance.
            function_name: The name of the function to implement.
            **kwargs: Additional context for the implementation prompt.

        Returns:
            An ImplementationDecision indicating the action to take.
        """
        is_browser_task = "action_provider.browser" in plan.plan_source_code
        replan_reason = kwargs.get("replan_reason")
        browser_state = None
        browser_screenshot = None
        if is_browser_task:
            browser_state = await self.action_provider.browser.observe(
                "Analyze the current page and provide a structured summary of its content.",
                response_format=PageAnalysis,
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
        full_plan_source = plan.plan_source_code or ""
        call_stack = plan.call_stack
        replan_context = replan_reason
        if not replan_context:
            replan_context = f"This is the first time the function '{function_name}' is being implemented. The function was defined as a stub with 'raise NotImplementedError' and now needs to be implemented to achieve its purpose: {docstring}"

        prompt = prompt_builders.build_dynamic_implement_prompt(
            full_plan_source=full_plan_source,
            call_stack=call_stack,
            function_name=function_name,
            function_sig=func_sig,
            function_docstring=docstring,
            parent_code=parent_code,
            browser_state=browser_state,
            has_browser_screenshot=browser_screenshot is not None,
            replan_context=replan_context,
            implementation_strategy=None,
            tools=self.tools,
        )
        self.implementation_client.set_response_format(ImplementationDecision)

        try:
            response_str = await llm_call(
                self.implementation_client,
                prompt,
                screenshot=browser_screenshot,
            )
            decision = ImplementationDecision.model_validate_json(response_str)

            if decision.action == "implement_function":
                if not decision.code:
                    raise ValueError(
                        "Action 'implement_function' requires the 'code' field.",
                    )
                decision.code = self._sanitize_code(
                    decision.code.strip()
                    .replace("```python", "")
                    .replace("```", "")
                    .strip(),
                )
            return decision
        finally:
            self.implementation_client.reset_response_format()

    async def _check_state_against_goal(
        self,
        plan: HierarchicalPlan,
        function_name: str,
        function_docstring: str | None,
        interactions: list,
        screenshot: bytes | str | None = None,
    ) -> VerificationAssessment:
        """
        Uses an LLM to assess if a function's execution achieved its goal.

        Args:
            plan: The active plan instance.
            function_name: The name of the function being verified.
            function_docstring: The docstring of the function.
            interactions: A log of interactions that occurred.
            screenshot: The screenshot of the current state of the browser.

        Returns:
            A VerificationAssessment object with the outcome.
        """
        prompt = prompt_builders.build_verification_prompt(
            goal=plan.goal,
            function_name=function_name,
            function_docstring=function_docstring,
            interactions=interactions,
            has_browser_screenshot=screenshot is not None,
        )

        self.verification_client.set_response_format(VerificationAssessment)

        try:
            response_str = await llm_call(
                self.verification_client,
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
            self.verification_client.reset_response_format()

    async def _perform_plan_surgery(self, current_code: str, request: str) -> str:
        """
        Uses an LLM to rewrite the plan's source code based on a request.

        Args:
            current_code: The current source code of the plan.
            request: The user's modification request.

        Returns:
            The new, sanitized source code for the plan.
        """
        prompt = prompt_builders.build_plan_surgery_prompt(
            current_code,
            request,
            tools=self.tools,
        )
        new_code = await llm_call(self.modification_client, prompt)
        return self._sanitize_code(new_code)

    async def _generate_course_correction_script(
        self,
        old_code: str,
        new_code: str,
    ) -> str | None:
        """
        Generates a script to transition from an old plan state to a new one.

        Args:
            old_code: The old plan's source code.
            new_code: The new plan's source code.

        Returns:
            A sanitized Python script for course correction, or None if not needed.
        """
        current_state = await self.action_provider.browser.observe(
            "Describe current page.",
        )

        prompt = prompt_builders.build_course_correction_prompt(
            old_code,
            new_code,
            current_state,
            tools=self.tools,
        )
        script = await llm_call(self.modification_client, prompt)
        if "None" in script:
            return None

        return self._sanitize_code(
            script.strip().replace("```python", "").replace("```", "").strip(),
        )

    async def close(self):
        """Shuts down the planner and its associated resources gracefully."""
        if self._active_task and not self._active_task.done():
            await self._active_task.stop()
        self.action_provider.browser.stop()
