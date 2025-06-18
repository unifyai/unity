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
from unity.planner.base import BasePlan, BasePlanner
from unity.planner.tool_loop_planner import ComsManager

logger = logging.getLogger(__name__)


class ReplanFromParentException(Exception):
    """Raised by the @verify decorator when a function's goal is misguided."""


class VerificationAssessment(BaseModel):
    """Structured output for the _check_state_against_goal LLM call."""

    status: str = Field(
        ...,
        description="Outcome status: 'ok', 'reimplement_local', 'replan_parent', or 'fatal_error'.",
    )
    reason: str = Field(..., description="A concise explanation for the status.")


class _HierarchicalPlanState(enum.Enum):
    """Manages the detailed lifecycle state of a hierarchical plan."""

    IDLE = enum.auto()
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


class HierarchicalPlan(BasePlan):
    """
    Represents and executes a single, dynamically generated hierarchical plan.
    This class is a steerable handle managing the plan's lifecycle.
    """

    def __init__(
        self,
        planner: "HierarchicalPlanner",
        goal: str,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ):
        self.planner = planner
        self.goal = goal
        self.plan_source_code: Optional[str] = None
        self.execution_namespace: Dict[str, Any] = {}
        self.call_stack: List[str] = []
        self.action_log: List[str] = []
        self.function_source_map: Dict[str, str] = {}

        self.main_loop_handle: Optional[AsyncToolUseLoopHandle] = None
        self._execution_task = asyncio.create_task(self._initialize_and_run())

        self._state = _HierarchicalPlanState.IDLE
        self.clarification_up_q = clarification_up_q or asyncio.Queue()
        self.clarification_down_q = clarification_down_q or asyncio.Queue()

    async def _initialize_and_run(self):
        """Initializes the plan and starts the async execution loop."""
        self._state = _HierarchicalPlanState.RUNNING
        self.action_log.append("Initializing plan...")
        try:
            self.plan_source_code = await self.planner._generate_initial_plan(self.goal)
            logger.info(f"Initial plan generated for goal: '{self.goal}'")
            logger.debug(f"Plan Source Code:\n{self.plan_source_code}")
            self.action_log.append("Initial plan generated successfully.")

            await self.planner._prepare_execution_environment(self)
            await self._start_main_execution_loop()

        except Exception as e:
            logger.error(f"Plan initialization failed with error: {e}", exc_info=True)
            self.action_log.append(f"ERROR: Plan initialization failed: {e}")
            self._state = _HierarchicalPlanState.ERROR
            if self.main_loop_handle and not self.main_loop_handle.done():
                self.main_loop_handle.stop()

    def _create_main_loop_iterator(self):
        """Creates a fresh generator for executing the plan."""
        # Re-exec the code to ensure the namespace is up-to-date with any modifications
        exec(self.plan_source_code, self.execution_namespace)
        main_fn_name = self._get_main_function_name()
        if not main_fn_name:
            raise RuntimeError("Could not determine the main entry point of the plan.")
        main_fn = self.execution_namespace[main_fn_name]
        yield main_fn()

    async def _start_main_execution_loop(self):
        """Sets up and runs the main execution loop for the plan."""
        plan_iterator = self._create_main_loop_iterator()

        async def _run_one_plan_step():
            """The core 'tool' for the SteerableToolLoopHandle."""
            nonlocal plan_iterator

            # This check ensures that the loop doesn't advance while a modification is in progress.
            if self._state == _HierarchicalPlanState.PAUSED_FOR_MODIFICATION:
                return {
                    "status": "paused",
                    "message": "Execution paused for plan modification.",
                }
            try:
                main_coro = next(plan_iterator)
                result = await main_coro
                self._state = _HierarchicalPlanState.COMPLETED
                self.action_log.append(f"Plan completed successfully. Result: {result}")
                return {
                    "status": "completed",
                    "message": f"Plan finished successfully. Result: {result}",
                }
            except StopIteration:
                self._state = _HierarchicalPlanState.COMPLETED
                self.action_log.append("Plan finished successfully.")
                return {"status": "completed", "message": "Plan finished successfully."}
            except NotImplementedError:
                function_name = self._get_unimplemented_function_name()
                logger.info(f"Handling dynamic implementation for: {function_name}")
                self.action_log.append(
                    f"Attempting to dynamically implement function: {function_name}",
                )
                await self._handle_dynamic_implementation(function_name)
                # Restart the iterator with the updated code
                plan_iterator = self._create_main_loop_iterator()
                return {
                    "status": "in_progress",
                    "message": f"Retrying after implementing {function_name}",
                }
            except ReplanFromParentException as e:
                # IMPROVEMENT: Use a dedicated state and externalize the failure to the user.
                self._state = _HierarchicalPlanState.PAUSED_FOR_ESCALATION
                escalation_message = f"ESCALATION: Plan has failed strategically and requires intervention. Reason: {e}. Please use 'modify_plan' to provide new instructions or 'stop' to terminate."
                self.action_log.append(escalation_message)
                logger.critical(escalation_message)
                await self.clarification_up_q.put(escalation_message)  # Notify client
                return {"status": "paused_for_escalation", "details": str(e)}
            except Exception as e:
                logger.error(f"Error during plan step execution: {e}", exc_info=True)
                self._state = _HierarchicalPlanState.ERROR
                self.action_log.append(f"ERROR: Plan execution failed: {e}")
                return {"status": "error", "message": str(e)}

        self.main_loop_handle = start_async_tool_use_loop(
            client=self.planner.llm_client,
            message="Executing hierarchical plan...",
            tools={"_run_one_plan_step": _run_one_plan_step},
            loop_id=f"HierarchicalPlan-{self.goal[:20]}",
            max_steps=100,
            tool_policy=lambda i, _: (
                "required",
                {"_run_one_plan_step": _run_one_plan_step},
            ),
            interrupt_llm_with_interjections=True,  # Allow interruption
        )

    async def _handle_dynamic_implementation(
        self,
        function_name: str,
        is_strategic_replan: bool = False,
        replan_reason: str = "",
    ):
        """Orchestrates the async dynamic implementation of a function."""
        if function_name in self.execution_namespace:
            func_to_implement = self.execution_namespace[function_name]
            signature = inspect.signature(func_to_implement)
            parent_name = self.call_stack[-1] if self.call_stack else None
            parent_code = (
                self.function_source_map.get(parent_name, "") if parent_name else ""
            )

            new_code = await self.planner._dynamic_implement(
                plan=self,
                function_name=function_name,
                function_signature=str(signature),
                parent_code=parent_code,
                is_strategic_replan=is_strategic_replan,
                replan_reason=replan_reason,
            )
            self._update_plan_with_new_code(function_name, new_code)
            logger.info(f"Dynamically implemented function: '{function_name}'.")
            self.action_log.append(
                f"Successfully implemented function: {function_name}",
            )
        else:
            raise RuntimeError(
                f"Could not find function '{function_name}' to implement.",
            )

    def _get_unimplemented_function_name(self) -> str:
        """Extracts the function name from the latest traceback."""
        _, _, exc_tb = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_tb)[-1]
        return frame_summary.name

    def _get_main_function_name(self) -> str | None:
        """Parses the source code to find the function named 'main_plan'."""
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
        """Replaces a function stub or existing function with new code, validating first."""
        try:
            ast.parse(new_code)
        except SyntaxError as e:
            logger.error(
                f"Generated code for '{function_name}' has a syntax error: {e}",
            )
            raise ValueError(
                f"Invalid syntax in generated code for {function_name}.",
            ) from e

        import re

        pattern = re.compile(
            rf"(?:@verify\s*\n)?(?:async\s+)?def\s+{function_name}\(.*\):(?:\n(.|\n)*?)(?=\n(?:@verify|async def|def)\s|\Z)",
            re.MULTILINE,
        )
        if pattern.search(self.plan_source_code):
            new_source = pattern.sub(
                textwrap.dedent(new_code).strip(),
                self.plan_source_code or "",
                1,
            )
        else:
            # Fallback for functions that might be at the end of the file without a following function
            pattern_eof = re.compile(
                rf"(?:@verify\s*\n)?(?:async\s+)?def\s+{function_name}\(.*\):.*",
                re.DOTALL,
            )
            if pattern_eof.search(self.plan_source_code):
                new_source = pattern_eof.sub(
                    textwrap.dedent(new_code).strip(),
                    self.plan_source_code or "",
                    1,
                )
            else:
                raise RuntimeError(
                    f"Could not find function '{function_name}' to update in plan.",
                )

        self.plan_source_code = new_source
        self.function_source_map[function_name] = new_code
        logger.debug(f"Updated plan source code:\n{self.plan_source_code}")
        exec(self.plan_source_code, self.execution_namespace)

    async def modify_plan(self, modification_request: str) -> str:
        """
        Handles 'Modifying Mode' with robust rollback. This is the primary way to correct
        a plan that is paused for escalation or needs a new direction.
        """
        if not self._is_valid_method("modify_plan"):
            return f"Plan cannot be modified in its current state: {self._state.name}"

        logger.info(f"Starting plan modification: '{modification_request}'")
        self.action_log.append(f"Starting plan modification: {modification_request}")

        original_source_code = self.plan_source_code
        self._state = _HierarchicalPlanState.PAUSED_FOR_MODIFICATION

        if self.main_loop_handle:
            self.main_loop_handle.stop()
            self.main_loop_handle = None

        try:
            # 1. Plan Surgery
            new_source_code = await self.planner._perform_plan_surgery(
                self.plan_source_code,
                modification_request,
            )
            self.action_log.append("Plan surgery completed.")

            # 2. Course Correction
            correction_script = await self.planner._generate_course_correction_script(
                self.plan_source_code,
                new_source_code,
            )
            if correction_script:
                logger.info("Executing course correction script...")
                self.action_log.append("Executing course correction script.")
                await self._execute_correction_script(correction_script)
                logger.info("Course correction finished.")
                self.action_log.append("Course correction finished.")

            # 3. Update and Resume
            self.plan_source_code = new_source_code
            logger.info("Plan successfully modified. Restarting execution loop.")
            self.action_log.append("Plan successfully modified. Restarting execution.")
            self._state = _HierarchicalPlanState.RUNNING
            await self._start_main_execution_loop()
            return "Plan modified and resumed successfully."

        except Exception as e:
            logger.error(
                f"Failed to modify plan, rolling back to previous version: {e}",
                exc_info=True,
            )
            self.action_log.append(f"ERROR: Failed to modify plan, rolling back: {e}")
            self.plan_source_code = original_source_code
            self._state = _HierarchicalPlanState.RUNNING
            await self._start_main_execution_loop()  # Restart with old code
            return "Failed to modify the plan. Rolled back to the previous version and resumed."

    async def _execute_correction_script(self, script: str):
        """Executes a short-lived correction script with a timeout."""
        correction_namespace = self.execution_namespace.copy()
        try:
            ast.parse(script)
            exec(script, correction_namespace)
        except SyntaxError as e:
            logger.error(f"Syntax error in course correction script: {e}")
            raise ValueError("Course correction script has invalid syntax.") from e

        correction_fn = correction_namespace.get("course_correction_main")
        if not correction_fn or not asyncio.iscoroutinefunction(correction_fn):
            raise RuntimeError(
                "Course correction script did not define 'async def course_correction_main'.",
            )

        try:
            logger.info("Executing course correction script with a 30s timeout.")
            await asyncio.wait_for(correction_fn(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error("Course correction script timed out.")
            raise RuntimeError("Course correction script took too long to execute.")
        except Exception as e:
            logger.error(
                f"Error executing course correction script: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Course correction script failed: {e}") from e

    async def result(self) -> str:
        """Waits for the plan to complete and returns its final result."""
        if self._execution_task:
            await self._execution_task
        if not self.main_loop_handle:
            return f"Error: Plan concluded in state {self._state.name} without a final result."
        final_result = await self.main_loop_handle.result()
        return final_result

    def done(self) -> bool:
        """Returns True if the plan has finished executing."""
        return self._state in (
            _HierarchicalPlanState.COMPLETED,
            _HierarchicalPlanState.STOPPED,
            _HierarchicalPlanState.ERROR,
        )

    async def stop(self) -> str:
        """Stops the plan's execution and waits for termination."""
        if self._state not in (
            _HierarchicalPlanState.STOPPED,
            _HierarchicalPlanState.COMPLETED,
            _HierarchicalPlanState.ERROR,
        ):
            logger.info(f"HierarchicalPlan stopping. Current state: {self._state.name}")
            self._state = _HierarchicalPlanState.STOPPED
            if self.main_loop_handle:
                self.main_loop_handle.stop()
            if self._execution_task and not self._execution_task.done():
                try:
                    await asyncio.wait_for(self._execution_task, timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for execution task to stop.")
            self.action_log.append("Plan stopped by user.")
            return "Plan was stopped."
        return f"Plan already in a terminal state: {self._state.name}."

    async def pause(self) -> str:
        """Pauses the plan's execution."""
        if self._state == _HierarchicalPlanState.RUNNING:
            logger.info("HierarchicalPlan pausing.")
            self._state = _HierarchicalPlanState.PAUSED
            if self.main_loop_handle:
                self.main_loop_handle.pause()
            self.action_log.append("Plan paused.")
            return "Plan paused successfully."
        return f"Plan cannot be paused in state {self.state.name}."

    async def resume(self) -> str:
        """Resumes a paused plan."""
        if self._state == _HierarchicalPlanState.PAUSED:
            logger.info("HierarchicalPlan resuming.")
            self._state = _HierarchicalPlanState.RUNNING
            if self.main_loop_handle:
                self.main_loop_handle.resume()
            self.action_log.append("Plan resumed.")
            return "Plan resumed successfully."
        return f"Plan cannot be resumed from state {self._state.name}. It might need modification."

    async def ask(self, question: str) -> str:
        """Asks a question about the current state of the plan using its action log."""
        if not self.action_log:
            return "No actions have been logged yet."
        context = "\n".join(f"- {log}" for log in self.action_log)
        prompt = f'You are an intelligent assistant analyzing an execution log. Based on the log, answer the user\'s question concisely.\n\n**Log:**\n{context}\n\n**Question:** "{question}"\n\n**Answer:**'
        try:
            return await llm_call(self.planner.llm_client, textwrap.dedent(prompt))
        except Exception as e:
            logger.error(f"Failed to answer 'ask' with LLM: {e}", exc_info=True)
            return f"Could not answer the question. The current plan state is {self._state.name}."

    def _is_valid_method(self, name: str) -> bool:
        """Checks if a control method is valid in the current plan state."""
        if name == "stop":
            return self._state in (
                _HierarchicalPlanState.RUNNING,
                _HierarchicalPlanState.PAUSED,
                _HierarchicalPlanState.PAUSED_FOR_ESCALATION,
            )
        if name == "pause":
            return self._state == _HierarchicalPlanState.RUNNING
        if name == "resume":
            return self._state == _HierarchicalPlanState.PAUSED
        if name == "ask":
            return self._state not in (
                _HierarchicalPlanState.IDLE,
                _HierarchicalPlanState.COMPLETED,
            )
        if name == "modify_plan":
            return self._state in (
                _HierarchicalPlanState.PAUSED,
                _HierarchicalPlanState.PAUSED_FOR_ESCALATION,
                _HierarchicalPlanState.RUNNING,
            )
        return False

    @property
    def valid_tools(self) -> Dict[str, Callable]:
        """Dynamically exposes steerable methods based on the plan's state."""
        tools = {}
        potential_tools = ["stop", "pause", "resume", "ask", "modify_plan"]
        for method_name in potential_tools:
            if self._is_valid_method(method_name):
                tools[method_name] = getattr(self, method_name)
        return tools


class HierarchicalPlanner(BasePlanner[HierarchicalPlan]):
    """
    Orchestrates task decomposition, execution, and dynamic modification
    by generating and running Python code.
    """

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

    def _make_plan(
        self,
        task_description: str,
        *,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> HierarchicalPlan:
        return HierarchicalPlan(
            planner=self,
            goal=task_description,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

    async def _generate_initial_plan(self, goal: str) -> str:
        """Generates the initial Python code for the plan, with retries for syntax errors."""
        max_retries = 3
        last_error = ""
        for attempt in range(max_retries):
            existing_functions = (
                self.function_manager.list_functions(include_implementations=True)
                if self.function_manager
                else {}
            )
            primitives_doc = "- `await act(instruction: str) -> str`\n- `await observe(query: str) -> str`\n- `await request_clarification(question: str) -> str`"
            existing_functions_doc = "\n".join(
                f'- `{name}{data["argspec"]}`: {data["docstring"]}'
                for name, data in existing_functions.items()
            )
            full_existing_code = "\n\n".join(
                data["implementation"] for data in existing_functions.values()
            )

            retry_prompt = (
                f"The previous attempt failed with a syntax error:\n---\n{last_error}\n---\nPlease correct the code."
                if last_error
                else ""
            )
            prompt = f"""
You are an expert Python programmer. Decompose the high-level user goal into a Python script.

{retry_prompt}

**User Goal:** "{goal}"

**Available Primitives:**
{primitives_doc}

**Available Pre-Implemented Functions:**
{existing_functions_doc or "None"}

**Instructions:**
1.  The main entry point function MUST be named `main_plan`.
2.  All generated functions MUST be `async def` and decorated with `@verify`.
3.  Reuse existing functions where possible by calling them.
4.  For new, complex sub-tasks, define a helper function stubbed with `raise NotImplementedError`.
5.  The final script must be complete, including any reused function code followed by your new functions.

**Full code of available functions (for injection):**
```python
{full_existing_code or "# No existing functions to inject."}
```
Now, generate the complete `async` Python script.
"""
            response = await llm_call(self.llm_client, textwrap.dedent(prompt))
            code = response.strip().replace("```python", "").replace("```", "").strip()
            try:
                ast.parse(code)
                return code
            except SyntaxError as e:
                logger.warning(
                    f"Generated code syntax error (attempt {attempt + 1}/{max_retries}): {e}",
                )
                last_error = str(e)
                if attempt == max_retries - 1:
                    raise
        raise RuntimeError("Failed to generate a valid plan after multiple attempts.")

    async def _dynamic_implement(
        self,
        plan: HierarchicalPlan,
        function_name: str,
        function_signature: str,
        parent_code: str,
        is_strategic_replan: bool = False,
        replan_reason: str = "",
    ) -> str:
        """Generates the async implementation for a function."""
        browser_state = await self.controller.observe(
            "Describe the current page for context.",
        )
        replan_context = (
            f"**REPLANNING REQUIRED:** The previous approach failed: '{replan_reason}'. Devise a new strategy."
            if is_strategic_replan
            else ""
        )
        prompt = f"""
You are an expert Python programmer. Implement the following `async` function.
{replan_context}

**Overall Task Goal:** "{plan.goal}"
**Function to Implement:** `async def {function_name}{function_signature}:`
**Context:** This function is called from:\n```python\n{parent_code}\n```
**Current Browser State:**\n"{browser_state}"

**Instructions:**
1.  Write the full `async def` implementation for `{function_name}`, decorated with `@verify`.
2.  Use only the available `async` primitives (`act`, `observe`, etc.).
3.  Provide a complete function: decorator, signature, docstring, and body.
"""
        response = await llm_call(self.llm_client, textwrap.dedent(prompt))
        return response.strip().replace("```python", "").replace("```", "").strip()

    async def _prepare_execution_environment(self, plan: HierarchicalPlan):
        """Prepares the namespace for executing the generated code."""

        async def request_clarification_primitive(question: str) -> str:
            await plan.clarification_up_q.put(question)
            return await plan.clarification_down_q.get()

        plan.execution_namespace.update(
            {
                "act": self.controller.act,
                "observe": self.controller.observe,
                "request_clarification": request_clarification_primitive,
                "verify": self._create_verify_decorator(plan),
                "ReplanFromParentException": ReplanFromParentException,
                "asyncio": asyncio,
            },
        )

    async def _check_state_against_goal(
        self,
        function_name: str,
        function_docstring: str | None,
        interactions: List[
            Tuple[str, str, Optional[str]]
        ],  # IMPROVEMENT: Now takes a list of interactions
    ) -> VerificationAssessment:
        """Uses an LLM to assess if a sequence of actions achieved the function's goal."""
        interactions_log = "\n".join(
            f"- Action: `{action}`, Observation: `{obs or 'N/A'}`"
            if kind == "observe"
            else f"- Action: `{action}`"
            for kind, action, obs in interactions
        )
        prompt = f"""
You are a verification agent. Assess if a sequence of actions successfully met a function's goal.

**Function Goal:**
- Name: `{function_name}`
- Purpose: `{function_docstring or 'No docstring provided.'}`

**Interaction Log:**
{interactions_log or "No actions or observations were recorded."}

**Assessment Task:**
Respond with a JSON object: {{"status": "...", "reason": "..."}}.
**Possible Statuses:**
- "ok": The actions were successful and the function's goal is met.
- "reimplement_local": A tactical error. The implementation is flawed but the goal is sound.
- "replan_parent": A strategic error. The function's premise is wrong or unachievable.
- "fatal_error": An unrecoverable system error occurred.
"""
        response_str = await llm_call(self.llm_client, textwrap.dedent(prompt))
        try:
            return VerificationAssessment(**json.loads(response_str))
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(
                f"Failed to decode assessment from LLM: {e}\nResponse: {response_str}",
            )
            return VerificationAssessment(
                status="fatal_error",
                reason="LLM provided a malformed assessment.",
            )

    def _create_verify_decorator(self, plan: HierarchicalPlan):
        """Creates the @verify decorator for async functions."""

        def verify(fn):
            try:
                plan.function_source_map[fn.__name__] = inspect.getsource(fn)
            except (TypeError, OSError):
                pass

            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                max_retries = 3
                last_exception = None
                plan.call_stack.append(fn.__name__)
                logger.info(
                    f"VERIFY: Entering '{fn.__name__}' (stack: {plan.call_stack})",
                )
                plan.action_log.append(f"Entering function: {fn.__name__}")

                for attempt in range(max_retries):
                    try:
                        func_source = plan.function_source_map.get(fn.__name__)
                        return await self._execute_and_verify_step(
                            plan,
                            fn,
                            func_source,
                            args,
                            kwargs,
                        )
                    except ReplanFromParentException as e:
                        logger.warning(
                            f"VERIFY: Caught strategic failure from child of '{fn.__name__}'. Replanning '{fn.__name__}'. Reason: {e}",
                        )
                        plan.action_log.append(
                            f"Strategic failure in child of {fn.__name__}. Replanning. Reason: {e}",
                        )
                        await plan._handle_dynamic_implementation(
                            fn.__name__,
                            is_strategic_replan=True,
                            replan_reason=str(e),
                        )
                        logger.info(
                            f"VERIFY: Retrying '{fn.__name__}' after strategic replan (attempt {attempt + 2}).",
                        )
                        continue
                    except NotImplementedError:
                        plan.call_stack.pop()
                        raise
                    except Exception as e:
                        last_exception = e
                        logger.error(
                            f"VERIFY: Exception in '{fn.__name__}' (attempt {attempt + 1}): {e}",
                            exc_info=False,
                        )
                        plan.action_log.append(
                            f"Handled exception in {fn.__name__} (attempt {attempt + 1}): {e}",
                        )
                        await asyncio.sleep(1)  # Brief pause before retry
                        continue

                plan.call_stack.pop()
                plan.action_log.append(
                    f"Function {fn.__name__} failed after {max_retries} attempts.",
                )
                raise ReplanFromParentException(
                    f"Function '{fn.__name__}' failed after {max_retries} attempts. Last error: {last_exception}",
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
    ) -> Any:
        """Helper to encapsulate the execution and verification of a single step."""
        interactions: List[Tuple[str, str, Optional[str]]] = []

        async def act_wrapper(instruction: str):
            interactions.append(("act", instruction, None))
            plan.action_log.append(f'Executing action: act("{instruction}")')
            return await self.controller.act(instruction)

        async def observe_wrapper(query: str):
            res = await self.controller.observe(query)
            interactions.append(("observe", query, res))
            plan.action_log.append(f"Observed state for '{query}': {str(res)[:100]}...")
            return res

        original_act = plan.execution_namespace.get("act")
        original_observe = plan.execution_namespace.get("observe")
        plan.execution_namespace["act"] = act_wrapper
        plan.execution_namespace["observe"] = observe_wrapper

        result = await fn(*args, **kwargs)

        plan.execution_namespace["act"] = original_act
        plan.execution_namespace["observe"] = original_observe

        assessment = await self._check_state_against_goal(
            function_name=fn.__name__,
            function_docstring=fn.__doc__,
            interactions=interactions,
        )
        logger.info(
            f"VERIFY Assessment for '{fn.__name__}': {assessment.status} - {assessment.reason}",
        )
        plan.action_log.append(
            f"Verification for {fn.__name__}: {assessment.status} - {assessment.reason}",
        )

        if assessment.status == "ok":
            if func_source and self.function_manager:
                try:
                    self.function_manager.add_functions(implementations=[func_source])
                except Exception as e:
                    logger.warning(f"Could not save function '{fn.__name__}': {e}")
            plan.call_stack.pop()
            return result
        elif assessment.status == "reimplement_local":
            await plan._handle_dynamic_implementation(
                fn.__name__,
                replan_reason=assessment.reason,
            )
            raise RuntimeError("Forced retry after local reimplementation")
        elif assessment.status == "replan_parent":
            raise ReplanFromParentException(
                f"Strategic failure in '{fn.__name__}': {assessment.reason}",
            )
        elif assessment.status == "fatal_error":
            raise RuntimeError(f"Fatal error in '{fn.__name__}': {assessment.reason}")

    async def _perform_plan_surgery(self, current_code: str, request: str) -> str:
        """Uses an LLM to rewrite a part of the plan's source code."""
        prompt = f"""
You are a master programmer modifying a Python script for an AI agent.
**User's Modification Request:** "{request}"
**Current Plan Source Code:**\n```python\n{current_code}\n```
**Instructions:** Rewrite the script to incorporate the change. Ensure all functions remain decorated with `@verify`. Output only the complete, new Python script.
"""
        response = await llm_call(self.llm_client, textwrap.dedent(prompt))
        return response.strip().replace("```python", "").replace("```", "").strip()

    async def _generate_course_correction_script(
        self,
        old_code: str,
        new_code: str,
    ) -> Optional[str]:
        """Generates a Python script to bridge the state gap between an old and new plan."""
        current_state = await self.controller.observe(
            "Describe current page for state analysis.",
        )
        prompt = f"""
You are a state transition analyst. An AI agent's plan was modified. Generate a 'course correction' script if its current state doesn't match the new plan's starting point.

**Current Browser State:**\n{current_state}
**Old Plan (context):**\n```python\n{old_code[:1000]}...\n```
**New Plan:**\n```python\n{new_code}\n```
**Task:** If the current state is wrong for the new plan, generate a Python script `course_correction_main` using `await act()` to fix it. Otherwise, respond with "None".
"""
        script = await llm_call(self.llm_client, textwrap.dedent(prompt))
        if "None" in script:
            return None
        return script.strip().replace("```python", "").replace("```", "").strip()

    async def close(self):
        """Gracefully shuts down the planner and its resources."""
        logger.info("HierarchicalPlanner: Closing resources...")
        if self.controller:
            self.controller.stop()
        if self.coms_manager and hasattr(self.coms_manager, "stop"):
            stop_coro = self.coms_manager.stop()
            if inspect.iscoroutine(stop_coro):
                await stop_coro
