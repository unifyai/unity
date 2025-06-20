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
from unity.planner.base import ActiveTask, BasePlanner
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
        description="Outcome: 'ok', 'reimplement_local', 'replan_parent', or 'fatal_error'.",
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
    """Convenience wrapper for a simple LLM call."""
    return await client.generate(prompt)


class PlanSanitizer(ast.NodeTransformer):
    """
    AST transformer to enforce security and correctness of plan code.
    1. Disallows `import` and `import from` statements.
    2. Ensures every `async def` function is decorated with `@verify`.
    """

    def visit_Import(self, node: ast.Import) -> Any:
        raise SyntaxError("Import statements are not allowed in plans.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        raise SyntaxError("Import statements are not allowed in plans.")

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> ast.AsyncFunctionDef:
        has_verify = any(
            isinstance(d, ast.Name) and d.id == "verify" for d in node.decorator_list
        )
        if not has_verify:
            node.decorator_list.insert(0, ast.Name(id="verify", ctx=ast.Load()))
        return self.generic_visit(node)


class FunctionReplacer(ast.NodeTransformer):
    """AST transformer to replace a function definition in a module."""

    def __init__(self, target_name: str, new_function_node: ast.FunctionDef):
        self.target_name = target_name
        self.new_function_node = new_function_node
        self.replaced = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        if node.name == self.target_name:
            self.replaced = True
            return self.new_function_node
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        if node.name == self.target_name:
            self.replaced = True
            return self.new_function_node
        return self.generic_visit(node)


class HierarchicalPlan(ActiveTask):
    """
    Represents and executes a single, dynamically generated hierarchical plan.
    This class is a steerable handle managing the plan's lifecycle.
    """

    MAX_ESCALATIONS = 3
    MAX_LOCAL_RETRIES = 2

    def __init__(
        self,
        planner: "HierarchicalPlanner",
        goal: str,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        parent_chat_context: Optional[str] = None,
        exploratory_mode: bool = False,
    ):
        self.planner = planner
        self.goal = goal
        self.exploratory_mode = exploratory_mode
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
        self._execution_task = asyncio.create_task(self._initialize_and_run())

    def _set_final_result(self, result: str):
        """Sets the final result and the completion event."""
        if not self._completion_event.is_set():
            self._final_result_str = result
            self._is_complete = True
            self._completion_event.set()

    async def _initialize_and_run(self):
        self.action_log.append("Initializing plan...")
        try:
            if self.exploratory_mode:
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
        self._state = _HierarchicalPlanState.EXPLORING
        self.action_log.append("Starting exploratory phase...")
        try:
            initial_state = await self.planner.controller.observe(
                "Describe the current page and its URL for initial context.",
            )
            prompt = textwrap.dedent(
                f"""
                You are an exploration agent. Your goal is to gather information to help create a better plan.
                **Main Goal:** "{self.goal}"
                **Current Page State:** "{initial_state}"
                **Task:** Formulate a single, concise `observe` query to gather the most critical information from this page to achieve the main goal. Your query should be a simple string.
                **Example Query:** "What are the main navigation links in the header?"
                """,
            )
            exploration_query = await llm_call(self.planner.llm_client, prompt)
            self.action_log.append(f"Exploration query: '{exploration_query}'")
            observation_result = await self.planner.controller.observe(
                exploration_query,
            )
            self.exploration_summary = f"Initial observation of the page for the goal '{self.goal}' revealed: {observation_result}"
            self.action_log.append("Exploratory phase completed.")
        except Exception as e:
            logger.error(f"Exploration phase failed: {e}", exc_info=True)
            self.action_log.append(
                f"WARNING: Exploration failed: {e}. Proceeding without extra context.",
            )
            self.exploration_summary = None
        finally:
            self._state = _HierarchicalPlanState.RUNNING

    def _create_main_loop_iterator(self):
        main_fn_name = self._get_main_function_name()
        if not main_fn_name:
            raise RuntimeError("Could not determine main entry point 'main_plan'.")
        main_fn = self.execution_namespace[main_fn_name]
        yield main_fn()

    async def _start_main_execution_loop(self):
        self.planner.llm_client.reset_messages()
        plan_iterator = self._create_main_loop_iterator()

        async def _run_one_plan_step():
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
                function_name = self._get_unimplemented_function_name()
                await self._handle_dynamic_implementation(function_name)
                plan_iterator = self._create_main_loop_iterator()
                return {
                    "status": "in_progress",
                    "message": f"Implemented {function_name}, retrying.",
                }
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
            if self._is_complete or self._state in (
                _HierarchicalPlanState.PAUSED,
                _HierarchicalPlanState.PAUSED_FOR_MODIFICATION,
                _HierarchicalPlanState.PAUSED_FOR_ESCALATION,
            ):
                return "auto", {}
            else:
                return "required", {"_run_one_plan_step": _run_one_plan_step}

        self.main_loop_handle = start_async_tool_use_loop(
            client=self.planner.llm_client,
            message="Executing hierarchical plan...",
            tools={"_run_one_plan_step": _run_one_plan_step},
            loop_id=f"HierarchicalPlan-{self.goal[:20]}",
            max_steps=100,
            tool_policy=dynamic_tool_policy,
            interrupt_llm_with_interjections=True,
        )

    async def _handle_dynamic_implementation(self, function_name: str, **kwargs):
        new_code = await self.planner._dynamic_implement(
            plan=self,
            function_name=function_name,
            **kwargs,
        )
        self._update_plan_with_new_code(function_name, new_code)
        self.action_log.append(f"Implemented function: {function_name}")

    def _get_unimplemented_function_name(self) -> str:
        _, _, exc_tb = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_tb)[-1]
        return frame_summary.name

    def _get_main_function_name(self) -> str | None:
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

            self.escalation_count = 0
            self._is_complete = False
            self.exploratory_mode = False
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
        logger.info("Executing course correction script...")
        self.action_log.append("Executing course correction script.")
        interactions = []
        try:
            correction_namespace = self.planner._create_sandbox_globals()

            async def act_wrapper(instruction: str):
                interactions.append(("act", instruction, None))
                return await self.planner.controller.act(instruction)

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
        await self._completion_event.wait()
        return (
            self._final_result_str
            or f"Plan finished in state {self._state.name} without a result."
        )

    def done(self) -> bool:
        return self._is_complete

    async def interject(self, message: str) -> str:
        if not self._is_valid_method("interject"):
            return "Cannot interject: plan not running."
        if self.main_loop_handle:
            await self.main_loop_handle.interject(message)
            self.action_log.append(f"User interjected: '{message}'")
            return "Interjection sent."
        return "Error: No active loop to interject into."

    async def stop(self) -> str:
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
        if self._state == _HierarchicalPlanState.RUNNING:
            self._state = _HierarchicalPlanState.PAUSED
            if self.main_loop_handle:
                self.main_loop_handle.pause()
            self.action_log.append("Plan paused by user.")
            return "Plan paused."
        return f"Cannot pause in state {self._state.name}."

    async def resume(self) -> str:
        if self._state == _HierarchicalPlanState.PAUSED:
            self._state = _HierarchicalPlanState.RUNNING
            if self.main_loop_handle:
                self.main_loop_handle.resume()
            self.action_log.append("Plan resumed by user.")
            return "Plan resumed."
        return f"Cannot resume from state {self._state.name}."

    async def ask(self, question: str) -> str:
        if self._state in (
            _HierarchicalPlanState.IDLE,
            _HierarchicalPlanState.EXPLORING,
        ):
            return "Plan has not started its main execution loop."
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
            return await llm_call(self.planner.llm_client, prompt)
        except Exception as e:
            return f"Could not answer question. Current state: {self._state.name}. Error: {e}"

    def _is_valid_method(self, name: str) -> bool:
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
    """Orchestrates task execution by generating and managing Python code."""

    def __init__(
        self,
        function_manager: "FunctionManager",
        controller: "Controller",
        coms_manager: "ComsManager",
    ):
        super().__init__()
        self.function_manager = function_manager
        self.controller = controller
        self.coms_manager = coms_manager
        self.llm_client: unify.AsyncUnify = unify.AsyncUnify(
            os.environ.get("UNIFY_MODEL", "gpt-4o-mini@openai"),
        )

    def _sanitize_code(self, code: str) -> str:
        """Parses, sanitizes, and unparses code to enforce security."""
        try:
            tree = ast.parse(code)
            sanitizer = PlanSanitizer()
            sanitized_tree = sanitizer.visit(tree)
            ast.fix_missing_locations(sanitized_tree)
            return ast.unparse(sanitized_tree)
        except SyntaxError as e:
            logger.error(f"Generated code failed sanitization: {e}")
            raise

    def _make_plan(
        self,
        task_description: str,
        *,
        exploratory_mode: bool = False,
        **kwargs,
    ) -> HierarchicalPlan:
        return HierarchicalPlan(
            planner=self,
            goal=task_description,
            exploratory_mode=exploratory_mode,
            **kwargs,
        )

    def _create_sandbox_globals(self) -> Dict[str, Any]:
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
        sandbox_globals = self._create_sandbox_globals()

        async def request_clarification_primitive(question: str) -> str:
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
        def verify(fn):
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
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
        async def act_wrapper(instruction: str):
            interactions.append(("act", instruction, None))
            return await self.controller.act(instruction)

        async def observe_wrapper(query: str, **opts):
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
        assessment = await self._check_state_against_goal(
            fn.__name__,
            fn.__doc__,
            all_interactions,
        )
        plan.action_log.append(
            f"Verification for {fn.__name__}: {assessment.status} - '{assessment.reason}'",
        )

        if assessment.status == "ok":
            if func_source and self.function_manager:
                self.function_manager.add_functions(implementations=[func_source])
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
        max_retries = 3
        last_error = ""
        for attempt in range(max_retries):
            try:
                existing_functions = (
                    self.function_manager.list_functions(include_implementations=True)
                    if self.function_manager
                    else {}
                )
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
                response = await llm_call(self.llm_client, prompt)
                code = (
                    response.strip().replace("```python", "").replace("```", "").strip()
                )

                return self._sanitize_code(code)

            except SyntaxError as e:
                last_error = str(e)
                if attempt == max_retries - 1:
                    raise
        raise RuntimeError("Failed to generate a valid plan after multiple retries.")

    def _build_initial_plan_prompt(
        self,
        goal,
        existing_functions,
        retry_msg,
        exploration_summary,
    ):
        primitives_doc = (
            "- `await act(instruction: str)`\n- `await observe(query: str)`"
        )
        existing_functions_doc = "\n".join(
            f'- `{name}{data["argspec"]}`: {data["docstring"]}'
            for name, data in existing_functions.items()
        )
        full_existing_code = "\n\n".join(
            data["implementation"] for data in existing_functions.values()
        )
        exploration_context = (
            f"\n**Context from Initial Exploration:**\n{exploration_summary}\n"
            if exploration_summary
            else ""
        )
        return textwrap.dedent(
            f"""
            You are an expert Python programmer. Decompose the goal into a Python script.
            {retry_msg}
            **Goal:** "{goal}"
            {exploration_context}
            **Primitives:**\n{primitives_doc}
            **Existing Functions:**\n{existing_functions_doc or "None"}
            **Instructions:**
            1. Main entry MUST be `async def main_plan()`.
            2. All functions MUST be `async def` and decorated with `@verify`.
            3. Stub complex new tasks with `raise NotImplementedError`.
            4. The script must be complete, including any reused function code.
            5. CRITICAL: You MUST NOT use any `import` statements. Rely only on provided primitives.
            **Code of available functions:**
            ```python
            {full_existing_code or "# None"}
            ```
            Generate the complete script.
        """,
        )

    async def _dynamic_implement(
        self,
        plan: HierarchicalPlan,
        function_name: str,
        **kwargs,
    ) -> str:
        browser_state = await self.controller.observe(
            "Describe current page for context.",
        )
        replan_context = (
            f"**REPLANNING:** Previous attempt failed: '{kwargs.get('replan_reason', '')}'. Devise a new strategy."
            if kwargs.get("is_strategic_replan")
            else ""
        )
        func_sig = inspect.signature(plan.execution_namespace[function_name])
        parent_code = (
            plan.function_source_map.get(plan.call_stack[-2], "")
            if len(plan.call_stack) > 1
            else ""
        )
        prompt = textwrap.dedent(
            f"""
            You are an expert Python programmer. Implement the function below.
            {replan_context}
            **Goal:** "{plan.goal}"
            **Function to Implement:** `async def {function_name}{func_sig}:`
            **Parent Function Code:**\n```python\n{parent_code}\n```
            **Current Browser State:** "{browser_state}"
            **Instructions:**
            1. Write the full `async def` implementation for `{function_name}`.
            2. The function MUST be decorated with `@verify`.
            3. Provide a complete function (decorator, signature, docstring, body).
            4. CRITICAL: You MUST NOT use any `import` statements.
        """,
        )
        code = await llm_call(self.llm_client, prompt)
        return self._sanitize_code(code)

    async def _check_state_against_goal(
        self,
        function_name,
        function_docstring,
        interactions,
    ):
        interactions_log = "\n".join(
            (
                f"- Action: `{act}`, Observation: `{obs or 'N/A'}`"
                if kind == "observe"
                else f"- Action: `{act}`"
            )
            for kind, act, obs in interactions
        )
        prompt = textwrap.dedent(
            f"""
            You are a verification agent. Assess if the actions met the function's goal.
            **Function:** `{function_name}` (Purpose: {function_docstring or 'N/A'})
            **Action Log:**\n{interactions_log or "No actions recorded."}
            **Task:** Respond with a single JSON object: {{"status": "...", "reason": "..."}}.
            Valid statuses: "ok", "reimplement_local" (tactical error), "replan_parent" (strategic error), "fatal_error".
        """,
        )
        response_str = await llm_call(self.llm_client, prompt)
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

    async def _perform_plan_surgery(self, current_code, request):
        prompt = textwrap.dedent(
            f"""
            You are a master programmer modifying a script.
            **Modification Request:** "{request}"
            **Current Code:**\n```python\n{current_code}\n```
            **Instructions:** Rewrite the entire script to incorporate the change. Ensure all functions retain their `@verify` decorators. Output only the new, complete Python script. Do not include any explanation.
            CRITICAL: You MUST NOT use any `import` statements.
        """,
        )
        new_code = await llm_call(self.llm_client, prompt)
        return self._sanitize_code(new_code)

    async def _generate_course_correction_script(self, old_code, new_code):
        current_state = await self.controller.observe("Describe current page.")
        prompt = textwrap.dedent(
            f"""
            You are a state transition analyst. An agent's plan has changed. If the agent's current state is incompatible with the start of the new plan, generate a Python script to fix it.
            **Current Browser State:** {current_state}
            **Old Plan Snippet:**\n```python\n{old_code[:1000]}...\n```
            **New Plan Code:**\n```python\n{new_code}\n```
            **Task:**
            1. Analyze if the **Current Browser State** is a suitable starting point for the **New Plan Code**.
            2. If it is NOT suitable, write a script with an `async def course_correction_main()` function that uses `await act()` and `await observe()` to navigate to the correct state.
            3. If the state is already suitable, respond ONLY with the word "None".
            4. CRITICAL: The script MUST NOT use any `import` statements.
        """,
        )
        script = await llm_call(self.llm_client, prompt)
        if "None" in script:
            return None

        return self._sanitize_code(
            script.strip().replace("```python", "").replace("```", "").strip(),
        )

    async def close(self):
        if self._active_task:
            await self._active_task.stop()
        if self.controller:
            self.controller.stop()
        if self.coms_manager and hasattr(self.coms_manager, "stop"):
            if inspect.iscoroutinefunction(self.coms_manager.stop):
                await self.coms_manager.stop()
            else:
                self.coms_manager.stop()
