from __future__ import annotations

import asyncio
import enum
import functools
import inspect
import json
import logging
import os
import sys
import textwrap
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple
import ast

import unify
from pydantic import BaseModel, Field

from unity.common.llm_helpers import (
    AsyncToolUseLoopHandle,
    start_async_tool_use_loop,
)
from unity.controller.controller import Controller
from unity.function_manager.function_manager import FunctionManager
from unity.planner.base import BaseActiveTask, BasePlanner
from unity.planner.tool_loop_planner import ComsManager

logger = logging.getLogger(__name__)


class ReplanFromParentException(Exception):
    """Raised by the @verify decorator when a function's goal is misguided."""


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


async def llm_call(client: unify.AsyncUnify, prompt: str) -> str:
    """
    Convenience wrapper for a simple, stateless LLM call.

    This helper automatically resets the client's message history before making
    the call to ensure no context is leaked from previous interactions.
    """
    client.reset_messages()
    return await client.generate(prompt)


class PlanSanitizer(ast.NodeTransformer):
    """
    AST transformer to enforce security and correctness of plan code.

    1. Disallows `import` and `import from` statements.
    2. Ensures every `async def` function is decorated with `@verify`.
    """

    def visit_Import(self, node: ast.Import) -> Any:
        """Blocks `import` statements."""
        raise SyntaxError("Import statements are not allowed in plans.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        """Blocks `from ... import` statements."""
        raise SyntaxError("Import statements are not allowed in plans.")

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

    async def _perform_exploration(self):
        """
        Runs an interactive conversational loop to gather information before planning.
        """
        self._state = _HierarchicalPlanState.EXPLORING
        self.action_log.append("Starting interactive exploratory phase...")
        try:

            async def request_clarification(question: str) -> str:
                """Asks the user for clarification when blocked or needs more input."""
                self.action_log.append(
                    f"Exploration: Asking for clarification: '{question}'",
                )
                await self.clarification_up_q.put(question)
                answer = await self.clarification_down_q.get()
                self.action_log.append(f"Exploration: Received answer: '{answer}'")
                return f"Received user clarification: '{answer}'"

            exploratory_tools = {
                "observe": self.planner.controller.observe,
                "request_clarification": request_clarification,
            }

            research_prompt = textwrap.dedent(
                f"""
                You are an intelligent Research Assistant. Your goal is to gather critical information to create a robust plan for the main objective.

                **Main Objective:** "{self.goal}"

                **Your Tools:**
                - `observe(query: str)`: Use this to inspect the current environment and gather information.
                - `request_clarification(question: str)`: If you are blocked, missing information, or need user input, use this tool to ask a question.

                **Your Task:**
                1. Think step-by-step to determine what information is needed.
                2. Use the `observe` tool to gather this information.
                3. If necessary, use `request_clarification` to ask for guidance.
                4. When you have gathered all necessary information, provide a final, concise summary of your findings. This summary will be used to generate the main plan. DO NOT say that you are ready; your final output MUST BE the summary itself.
                """,
            )

            client = self.planner.exploration_client
            client.reset_messages()
            client.set_system_message(research_prompt)

            exploration_loop_handle = start_async_tool_use_loop(
                client=client,
                message="Begin your research based on the main objective.",
                tools=exploratory_tools,
                loop_id="ExploratoryPhase",
                max_steps=10,
            )

            summary = await exploration_loop_handle.result()
            self.exploration_summary = summary
            self.action_log.append("Exploratory phase completed.")
            self.action_log.append(f"Exploration Summary: {summary}")

        except Exception as e:
            logger.error(f"Exploration phase failed: {e}", exc_info=True)
            self.action_log.append(
                f"WARNING: Exploration failed: {e}. Proceeding without extra context.",
            )
            self.exploration_summary = None
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

                parent_to_replan = self._get_main_function_name()
                if not parent_to_replan:
                    raise RuntimeError("Could not determine main_plan to replan.")

                if self.escalation_count > self.MAX_ESCALATIONS:
                    self._state = _HierarchicalPlanState.PAUSED_FOR_ESCALATION
                    err_msg = f"ESCALATION LIMIT: Max escalations ({self.MAX_ESCALATIONS}) reached. Pausing for intervention. Final reason: {e}"
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
                    replan_reason=str(e),
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
            loop_id=f"HierarchicalPlan-{self.goal[:20]}",
            max_steps=100,
            tool_policy=dynamic_tool_policy,
            interrupt_llm_with_interjections=True,
        )
        await self.main_loop_handle.result()

    async def _handle_dynamic_implementation(self, function_name: str, **kwargs):
        """
        Orchestrates the dynamic implementation of a stub function.

        Args:
            function_name: The name of the function to implement.
            **kwargs: Additional context for implementation (e.g., replan reason).
        """
        new_code = await self.planner._dynamic_implement(
            plan=self,
            function_name=function_name,
            **kwargs,
        )
        self._update_plan_with_new_code(function_name, new_code)
        self.action_log.append(f"Implemented function: {function_name}")

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
            new_code_module = ast.parse(textwrap.dedent(new_code))
            if not new_code_module.body or not isinstance(
                new_code_module.body[0],
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                raise ValueError("New code does not define a function.")
            new_func_node = new_code_module.body[0]

            old_tree = ast.parse(self.plan_source_code or "pass")
            transformer = FunctionReplacer(function_name, new_func_node)
            new_tree = transformer.visit(old_tree)

            if not transformer.replaced:
                old_tree.body.append(new_func_node)
                new_tree = old_tree

            self.plan_source_code = ast.unparse(new_tree)
            self.function_source_map[function_name] = ast.get_source_segment(
                self.plan_source_code,
                new_func_node,
            )

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

            async def act_wrapper(instruction: str):
                """Wraps the act primitive to log interactions."""
                interactions.append(("act", instruction, None))
                result = await self.planner.controller.act(instruction)
                return result

            correction_namespace["act"] = act_wrapper
            correction_namespace["observe"] = self.planner.controller.observe
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
            browser_context = await self.planner.controller.observe(
                "Summarize the current page.",
            )
            context_log = "\n".join(f"- {log}" for log in self.action_log[-10:])
            prompt = textwrap.dedent(
                f"""
                You are an assistant analyzing an agent's state. Answer the user's question concisely based *only* on the provided context.

                **Goal:** {self.goal}
                **State:** {self._state.name}
                **Call Stack:** {' -> '.join(self.call_stack) or 'None'}
                **Browser State:** {browser_context}
                **Recent Log:**
                {context_log}

                **Question:** "{question}"
                **Answer:**
            """,
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
        controller: Optional["Controller"] = None,
        coms_manager: Optional["ComsManager"] = None,
        session_connect_url: Optional[str] = None,
        headless: bool = False,
        max_escalations: Optional[int] = None,
        max_local_retries: Optional[int] = None,
    ):
        """
        Initializes the HierarchicalPlanner.

        Args:
            function_manager: Manages a library of reusable functions.
            controller: The browser controller for executing `act` and `observe`.
            coms_manager: Manages communication with the user.
            session_connect_url: URL for connecting to an existing browser session.
            headless: Whether to run the browser in headless mode.
            max_escalations: Default max number of strategic replans for plans.
            max_local_retries: Default max number of tactical retries for plans.
        """
        super().__init__()
        self.function_manager = function_manager or FunctionManager()
        self.controller = controller or Controller(
            session_connect_url=session_connect_url,
            headless=headless,
        )
        if not self.controller.is_alive():
            self.controller.start()
        self.coms_manager = coms_manager or ComsManager()
        self.max_escalations = max_escalations or 3
        self.max_local_retries = max_local_retries or 2

        model = os.environ.get("UNIFY_MODEL", "gpt-4o-mini@openai")
        self.main_loop_client: unify.AsyncUnify = unify.AsyncUnify(model)
        self.plan_generation_client: unify.AsyncUnify = unify.AsyncUnify(model)
        self.verification_client: unify.AsyncUnify = unify.AsyncUnify(model)
        self.implementation_client: unify.AsyncUnify = unify.AsyncUnify(model)
        self.modification_client: unify.AsyncUnify = unify.AsyncUnify(model)
        self.exploration_client: unify.AsyncUnify = unify.AsyncUnify(model)
        self.ask_client: unify.AsyncUnify = unify.AsyncUnify(model)

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
        prompt = textwrap.dedent(
            f"""
            You are a web browser agent assessing a task description from a user.
            The agent's goal is to generate a complete Python script to accomplish a task.
            The available tools are high-level: `act(instruction)` and `observe(query)`.
            The agent will be using the `act` tool to navigate the web and the `observe` tool to get information about the page.

            Analyze the following goal:
            **Goal:** "{goal}"

            Is the goal specific and actionable enough to directly write a Python script?
            Or is the goal ambiguous, broad, or lacking key details (like URLs, exact button text, or a clear workflow)
            that the agent would need to discover first using the `observe` and `request_clarification` tools?

            - If the goal is **clear and specific**, respond with the single word: **EXECUTE**.
            - If the goal is **ambiguous or requires information gathering**, respond with the single word: **EXPLORE**.
            """,
        )
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
                "Exception",
                "NotImplementedError",
                "isinstance",
                "any",
                "all",
                "sum",
                "min",
                "max",
                "abs",
                "round",
                "sorted",
                "enumerate",
                "zip",
                "ValueError",
                "TypeError",
                "KeyError",
                "IndexError",
            ]
            if __builtins__.get(k) is not None
        }
        return {"__builtins__": safe_builtins, "asyncio": asyncio}

    async def _prepare_execution_environment(self, plan: HierarchicalPlan):
        """
        Prepares the sandboxed execution environment for a plan.

        This involves setting up global functions (`act`, `observe`, `verify`)
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
                "act": self.controller.act,
                "observe": self.controller.observe,
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
                try:
                    sig = inspect.signature(fn)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    cache_key = (fn.__name__, frozenset(bound_args.arguments.items()))
                except (TypeError, ValueError):
                    cache_key = (fn.__name__, str(args), str(kwargs))

                if cache_key in plan.completed_functions:

                    logger.info(
                        f"CACHE HIT: Skipping already completed call to '{fn.__name__}' with args {args}, {kwargs}",
                    )
                    return
                logger.info(
                    f"CACHE MISS: Proceeding with execution for '{fn.__name__}'.",
                )
                plan.call_stack.append(fn.__name__)
                plan.interaction_stack.append([])
                logger.info(f"VERIFY: Entering '{fn.__name__}'")
                try:
                    for _ in range(plan.MAX_LOCAL_RETRIES):
                        try:
                            func_source = plan.function_source_map.get(fn.__name__)
                            return await self._execute_and_verify_step(
                                plan,
                                fn,
                                func_source,
                                args,
                                kwargs,
                                plan.interaction_stack[-1],
                            )
                        except _ForcedRetryException:
                            plan.action_log.append(
                                f"Retrying '{fn.__name__}' after reimplementation.",
                            )
                            continue
                        except (
                            ReplanFromParentException,
                            NotImplementedError,
                            FatalVerificationError,
                        ):
                            raise
                        except Exception as e:
                            logger.error(
                                f"Function '{fn.__name__}' failed: {e}",
                                exc_info=True,
                            )
                            await asyncio.sleep(1)
                            continue
                    raise ReplanFromParentException(
                        f"Function '{fn.__name__}' failed after multiple retries.",
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

        Returns:
            The result of the function call if verification passes.
        """

        async def act_wrapper(instruction: str):
            """Wraps the act primitive to log interactions."""
            interactions.append(("act", instruction, None))
            result = await self.controller.act(instruction)
            return result

        async def observe_wrapper(query: str, **opts):
            """Wraps the observe primitive to log interactions."""
            res = await self.controller.observe(query, **opts)
            interactions.append(("observe", query, res))
            return res

        original_act, original_observe = plan.execution_namespace.get(
            "act",
        ), plan.execution_namespace.get("observe")
        plan.execution_namespace["act"], plan.execution_namespace["observe"] = (
            act_wrapper,
            observe_wrapper,
        )
        try:
            result = await fn(*args, **kwargs)
        finally:
            plan.execution_namespace["act"], plan.execution_namespace["observe"] = (
                original_act,
                original_observe,
            )

        all_interactions = [
            item for sublist in plan.interaction_stack for item in sublist
        ]
        logger.info(
            f"🕵️ VERIFICATION INPUT for '{fn.__name__}':\n"
            f"   - Purpose: {fn.__doc__ or 'N/A'}\n"
            f"   - Interactions:\n{json.dumps(all_interactions, indent=4)}",
        )
        assessment = await self._check_state_against_goal(
            plan,
            fn.__name__,
            fn.__doc__,
            all_interactions,
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
                logger.debug(
                    f"Rolled up {len(child_interactions)} interactions from '{fn.__name__}' to its parent.",
                )

            if func_source and self.function_manager:
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
                        logger.info("clean_func_source: %s", clean_func_source)
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
            )
            raise _ForcedRetryException("Forced retry after local reimplementation")
        elif assessment.status == "replan_parent":
            raise ReplanFromParentException(
                f"Strategic failure in '{fn.__name__}': {assessment.reason}",
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

                prompt = self._build_initial_plan_prompt(
                    goal,
                    existing_functions,
                    (
                        ""
                        if attempt == 0
                        else f"Last attempt failed: {last_error}. Please fix."
                    ),
                    exploration_summary,
                )
                response = await llm_call(self.plan_generation_client, prompt)
                code = (
                    response.strip().replace("```python", "").replace("```", "").strip()
                )
                logger.debug(
                    f"LLM response for initial plan (attempt {attempt+1}):\n--- LLM RAW RESPONSE START ---\n{response}\n--- LLM RAW RESPONSE END ---",
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

    def _build_initial_plan_prompt(
        self,
        goal: str,
        existing_functions: dict,
        retry_msg: str,
        exploration_summary: str | None,
    ) -> str:
        """
        Builds the system prompt for generating an initial plan.

        Args:
            goal: The user's goal.
            existing_functions: A dictionary of reusable functions from the library.
            retry_msg: A message to include if a previous attempt failed.
            exploration_summary: Context from the exploration phase.

        Returns:
            The complete prompt string.
        """
        primitives_doc = (
            "- `await act(instruction: str)`: Executes a high-level action in the browser (e.g., 'click the login button', 'type 'hello' into the search bar').\n"
            "- `await observe(query: str)`: Asks a question about the current browser state (e.g., 'what is the text of the main heading?', 'are there any error messages?')."
        )
        existing_functions_doc = (
            "\n".join(
                f'- `{name}{data["argspec"]}`: {data["docstring"]}'
                for name, data in existing_functions.items()
            )
            or "None."
        )
        full_existing_code = "\n\n".join(
            data["implementation"] for data in existing_functions.values()
        )

        exploration_context = (
            f"**Context from Initial Exploration:**\n{exploration_summary}"
            if exploration_summary
            else ""
        )
        return textwrap.dedent(
            f"""
        You are an expert Python programmer tasked with generating a complete, single-file script to achieve a user's goal in a web browser.

        **Primary Goal:** "{goal}"
        {exploration_context}
        {retry_msg}

        ---
        ### Instructions & Rules
        1.  **Single Code Block:** Your entire response MUST be a single, valid Python code block. Do NOT include any preamble, explanations, or markdown fences.
        2.  **Entry Point:** The main entry point for the script MUST be a function named `async def main_plan()`.
        3.  **Docstrings Required:** Each function you define MUST include a concise one-line docstring explaining its purpose.
        4.  **Required Decorator:** All functions you define MUST be `async def` and MUST be decorated with `@verify`.
        5.  **No Imports:** You MUST NOT use any `import` statements. The execution environment is sandboxed.
        6.  **Asynchronous Calls:** You MUST use the `await` keyword before any call to an `async def` function, including the primitives (`act`, `observe`) and any helper functions you define.
        7.  **Decomposition:** Break down complex problems into smaller, logical helper functions. If a suitable function exists in the library, use it. If not, you may define it, or if its implementation is not immediately obvious, you may leave it as a stub (e.g., `raise NotImplementedError`).
        8.  **Final Output:** The `main_plan` function MUST return the final answer as a string. It MUST NOT use `print()` for the final output.

        ---
        ### Primitives Reference
        You have ONLY TWO primitive tools for browser interaction:
        {primitives_doc}
        Primitive Signatures: The act and observe primitives each take only one string argument (e.g., await act("click the login button")). You MUST NOT pass dictionaries or multiple arguments.

        ---
        ### Existing Functions Library
        You have access to the following pre-existing functions. You should use them if they help achieve the goal.
        {existing_functions_doc}

        ---
        ### Code of Available Functions
        ```python
        {full_existing_code or "# No pre-existing functions were provided."}
        ```

        Begin your response now. Your response must start immediately with the code.
        """,
        )

    def _build_dynamic_implement_prompt(
        self,
        plan: HierarchicalPlan,
        function_name: str,
        browser_state: str,
        **kwargs,
    ) -> str:
        """
        Builds the system prompt for dynamically implementing a function.

        Args:
            plan: The active plan instance.
            function_name: The name of the function to implement.
            browser_state: A description of the current browser state.
            **kwargs: Additional context, like a replan reason.

        Returns:
            The complete prompt string.
        """
        replan_context = (
            f"**REPLANNING NOTE:** The previous attempt failed because: '{kwargs.get('replan_reason', 'No reason provided.')}'. Please devise a new and improved strategy."
            if kwargs.get("is_strategic_replan")
            else ""
        )
        func_sig = inspect.signature(plan.execution_namespace[function_name])
        parent_code = (
            plan.function_source_map.get(plan.call_stack[-2], "")
            if len(plan.call_stack) > 1
            else "N/A (This is a top-level function call)"
        )

        return textwrap.dedent(
            f"""
            You are an expert Python programmer. Your task is to write the implementation for the function `{function_name}`.

            **Overall Goal:** "{plan.goal}"
            **Function to Implement:** `async def {function_name}{func_sig}`
            **Parent Function (for context):**
            ```python
            {parent_code}
            ```
            **Current Browser State:**
            {browser_state}

            {replan_context}

            ---
            ### Instructions & Rules
            1.  **Code Only:** Your output MUST be ONLY the Python code for the function, starting with the `@verify` decorator and the function definition. Do not include explanations or markdown.
            2.  **Add a Docstring:** The function implementation MUST include a concise one-line docstring explaining its purpose.
            3.  **Primitives Only:** For browser interaction, use ONLY `await act(...)` and `await observe(...)`. Primitive Signatures: The act and observe primitives each take only one string argument (e.g., await act("click the login button")). You MUST NOT pass dictionaries or multiple arguments.
            4.  **No Imports:** `import` statements are forbidden.

            Begin your response now.
            """,
        )

    async def _dynamic_implement(
        self,
        plan: HierarchicalPlan,
        function_name: str,
        **kwargs,
    ) -> str:
        """
        Generates and returns the implementation for a stub function.

        Args:
            plan: The active plan instance.
            function_name: The name of the function to implement.
            **kwargs: Additional context for the implementation prompt.

        Returns:
            The sanitized source code for the new function implementation.
        """
        browser_state = await self.controller.observe(
            "Describe current page for context.",
        )
        prompt = self._build_dynamic_implement_prompt(
            plan,
            function_name,
            browser_state,
            **kwargs,
        )
        code = await llm_call(self.implementation_client, prompt)
        logger.debug(
            f"LLM response for dynamic implementation of '{function_name}':\n--- LLM RAW RESPONSE START ---\n{code}\n--- LLM RAW RESPONSE END ---",
        )

        try:
            sanitized_code = self._sanitize_code(
                code.strip().replace("```python", "").replace("```", "").strip(),
            )
            logger.debug(
                f"Sanitized code for '{function_name}':\n--- CODE START ---\n{sanitized_code}\n--- CODE END ---",
            )
            return sanitized_code
        except SyntaxError as e:
            logger.error(
                f"Syntax error implementing '{function_name}'. Reason: {e}\nProblematic Code:\n---\n{code}\n---",
            )
            raise

    def _build_verification_prompt(
        self,
        plan: HierarchicalPlan,
        function_name: str,
        function_docstring: str | None,
        interactions: list,
    ) -> str:
        """
        Builds the prompt for verifying a function's execution.

        Args:
            plan: The active plan instance.
            function_name: The name of the function being verified.
            function_docstring: The docstring of the function.
            interactions: A log of `act` and `observe` calls made.

        Returns:
            The complete prompt string for the verification LLM call.
        """
        interactions_log = (
            "\n".join(
                (
                    f"- Action: `{act}`, Observation: `{obs or 'N/A'}`"
                    if kind == "observe"
                    else f"- Action: `{act}`"
                )
                for kind, act, obs in interactions
            )
            or "No browser actions were recorded."
        )

        return textwrap.dedent(
            f"""
        You are a meticulous verification agent. Your task is to assess if the executed actions successfully achieved the function's intended purpose, in the context of the overall goal.

        **Overall User Goal:** "{plan.goal}"
        **Function Under Review:** `{function_name}`
        **Purpose of this function:** {function_docstring or 'No docstring provided.'}

        **Execution Log (Primitives Used):**
        {interactions_log}

        ---
        ### Assessment Task
        Based on the function's purpose and the execution log, provide your assessment as a single JSON object.
        - **Be pragmatic:** If the function's purpose is to gather data (like search results), and the log shows that the data was successfully retrieved, this should be considered a success (`ok`). The function does not need to perform extra analysis unless explicitly asked.
        - **Consider the overall goal:** If a function's individual purpose is unclear but its actions logically progress toward the overall user goal, you should also consider it a success (`ok`).

        **Response Schema:**
        `{{"status": "...", "reason": "..."}}`

        **Valid Statuses:**
        - `ok`: The function's purpose was fully and correctly achieved.
        - `reimplement_local`: A tactical error occurred. The goal is correct, but the actions were wrong. The function needs to be re-written.
        - `replan_parent`: A strategic error occurred. The function itself is flawed or was called at the wrong time. The parent function needs to be replanned.
        - `fatal_error`: An unrecoverable error occurred that prevents any further progress.
        """,
        )

    def _build_plan_surgery_prompt(self, current_code: str, request: str) -> str:
        """
        Builds the prompt for modifying an existing plan script.

        Args:
            current_code: The current source code of the plan.
            request: The user's modification request.

        Returns:
            The complete prompt string.
        """
        return textwrap.dedent(
            f"""
            You are an expert Python programmer specializing in code modification. Your task is to rewrite an entire script to incorporate a user's change request.

            **Modification Request:**
            "{request}"

            ---
            ### Current Script
            ```python
            {current_code}
            ```

            ---
            ### Instructions & Rules
            1.  **Rewrite the Whole Script:** You must return a new, complete, and valid Python script that incorporates the requested change.
            2.  **Code Only:** Your response MUST be a single Python code block. Do NOT include any preamble, explanations, or markdown fences.
            3.  **Preserve Docstrings:** All functions should have a concise one-line docstring explaining their purpose.
            4.  **Preserve Decorators:** All `async def` functions MUST retain their `@verify` decorator.
            5.  **No Imports:** You MUST NOT use any `import` statements.

            Begin your response now. Your response must start immediately with the code.
            """,
        )

    def _build_course_correction_prompt(
        self,
        old_code: str,
        new_code: str,
        current_state: str,
    ) -> str:
        """
        Builds the prompt to generate a course-correction script.

        Args:
            old_code: The previous version of the plan's code.
            new_code: The new version of the plan's code.
            current_state: A description of the current browser state.

        Returns:
            The complete prompt string.
        """
        return textwrap.dedent(
            f"""
            You are a state transition analyst. An agent's plan has been modified, and you must determine if its current state is compatible with the new plan. If not, you must generate a Python script to fix it.

            ---
            ### Context
            **Current Browser State:**
            {current_state}

            **Old Plan Snippet (for context):**
            ```python
            {old_code[:1000]}...
            ```

            **New Plan Code:**
            ```python
            {new_code}
            ```

            ---
            ### Task
            1.  Analyze if the **Current Browser State** is a suitable starting point for executing the **New Plan Code**.
            2.  If it is NOT suitable, write a script containing an `async def course_correction_main()` function. This script must use `await act(...)` and `await observe(...)` to navigate to the correct starting state for the new plan. The script must be a single code block.
            3.  If the current state is already suitable, respond ONLY with the single word: `None`.
            4.  **CRITICAL**: The script MUST NOT use any `import` statements.

            Begin your response now.
            """,
        )

    async def _check_state_against_goal(
        self,
        plan: HierarchicalPlan,
        function_name: str,
        function_docstring: str | None,
        interactions: list,
    ) -> VerificationAssessment:
        """
        Uses an LLM to assess if a function's execution achieved its goal.

        Args:
            plan: The active plan instance.
            function_name: The name of the function being verified.
            function_docstring: The docstring of the function.
            interactions: A log of interactions that occurred.

        Returns:
            A VerificationAssessment object with the outcome.
        """
        prompt = self._build_verification_prompt(
            plan,
            function_name,
            function_docstring,
            interactions,
        )
        response_str = await llm_call(self.verification_client, prompt)
        try:
            clean_response = (
                response_str.strip().replace("```json", "").replace("```", "")
            )
            return VerificationAssessment(**json.loads(clean_response))
        except (json.JSONDecodeError, TypeError):
            return VerificationAssessment(
                status="fatal_error",
                reason="LLM provided malformed JSON assessment.",
            )

    async def _perform_plan_surgery(self, current_code: str, request: str) -> str:
        """
        Uses an LLM to rewrite the plan's source code based on a request.

        Args:
            current_code: The current source code of the plan.
            request: The user's modification request.

        Returns:
            The new, sanitized source code for the plan.
        """
        prompt = self._build_plan_surgery_prompt(current_code, request)
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
        current_state = await self.controller.observe("Describe current page.")
        prompt = self._build_course_correction_prompt(old_code, new_code, current_state)
        script = await llm_call(self.modification_client, prompt)
        if "None" in script:
            return None

        return self._sanitize_code(
            script.strip().replace("```python", "").replace("```", "").strip(),
        )

    async def close(self):
        """Shuts down the planner and its associated resources gracefully."""
        if self._active_task:
            await self._active_task.stop()
        if self.controller:
            self.controller.stop()
        if self.coms_manager and hasattr(self.coms_manager, "stop"):
            if inspect.iscoroutinefunction(self.coms_manager.stop):
                await self.coms_manager.stop()
            else:
                self.coms_manager.stop()
