from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import os
import sys
import textwrap
import traceback
from typing import Any, Callable, Coroutine, Dict, List, Optional

import unify
from pydantic import BaseModel, Field

from unity.common.llm_helpers import (
    AsyncToolUseLoopHandle,
    start_async_tool_use_loop,
)
from unity.controller.controller import Controller
from unity.function_manager.function_manager import FunctionManager
from unity.planner.base import BasePlan, BasePlanner
from unity.task_manager.task_manager import TaskManager
from unity.planner.tool_loop_planner import ComsManager

logger = logging.getLogger(__name__)

# Custom exception for escalating strategic failures, as per the spec.
class ReplanFromParentException(Exception):
    """Raised by the @verify decorator when a function's goal is misguided."""
    pass

class VerificationAssessment(BaseModel):
    """Structured output for the _check_state_against_goal LLM call."""
    status: str = Field(..., description="Outcome status: 'ok', 'reimplement_local', 'replan_parent', or 'fatal_error'.")
    reason: str = Field(..., description="A concise explanation for the status.")

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
    ):
        self.planner = planner
        self.goal = goal
        self.plan_source_code: Optional[str] = None
        self.execution_namespace: Dict[str, Any] = {}
        self.call_stack: List[str] = []

        self.main_loop_handle: Optional[AsyncToolUseLoopHandle] = None
        self._execution_task = asyncio.create_task(self._initialize_and_run())
        self._is_paused_for_modification = False

    
    async def _initialize_and_run(self):
        """Initializes the plan and starts the async execution loop."""
        try:
            self.plan_source_code = await self.planner._generate_initial_plan(self.goal)
            logger.info(f"Initial plan generated for goal: '{self.goal}'")
            logger.debug(f"Plan Source Code:\n{self.plan_source_code}")

            await self.planner._prepare_execution_environment(self)
            await self._start_main_execution_loop()

        except Exception as e:
            logger.error(f"Plan initialization failed with error: {e}", exc_info=True)
            if self.main_loop_handle and not self.main_loop_handle.done():
                self.main_loop_handle.stop()

    def _create_main_loop_iterator(self):
        """Creates a fresh generator for executing the plan."""
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
            if self._is_paused_for_modification:
                return {"status": "paused", "message": "Execution paused for plan modification."}
            try:
                main_coro = next(plan_iterator)
                result = await main_coro
                return {"status": "completed", "message": f"Plan finished successfully. Result: {result}"}
            except StopIteration:
                return {"status": "completed", "message": "Plan finished successfully."}
            except NotImplementedError:
                function_name = self._get_unimplemented_function_name()
                logger.info(f"Handling dynamic implementation for: {function_name}")
                await self._handle_dynamic_implementation(function_name)
                # Restart the iterator with the updated code
                nonlocal plan_iterator
                plan_iterator = self._create_main_loop_iterator()
                return {"status": "in_progress", "message": f"Retrying after implementing {function_name}"}
            except ReplanFromParentException as e:
                return {"status": "paused_for_escalation", "details": str(e)}
            except Exception as e:
                logger.error(f"Error during plan step execution: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}

        self.main_loop_handle = start_async_tool_use_loop(
            client=self.planner.llm_client,
            message="Executing hierarchical plan...",
            tools={"_run_one_plan_step": _run_one_plan_step},
            tool_policy=lambda i, _: ("required", {"_run_one_plan_step": _run_one_plan_step})
        )

    async def _handle_dynamic_implementation(self, function_name: str, is_strategic_replan: bool = False, replan_reason: str = ""):
        """Orchestrates the async dynamic implementation of a function."""
        if function_name in self.execution_namespace:
            func_to_implement = self.execution_namespace[function_name]
            signature = inspect.signature(func_to_implement)
            # Get parent code from the call stack for context.
            parent_name = self.call_stack[-1] if self.call_stack else None
            parent_code = inspect.getsource(self.execution_namespace[parent_name]) if parent_name and parent_name in self.execution_namespace else ""

            new_code = await self.planner._dynamic_implement(
                plan=self,
                function_name=function_name,
                function_signature=str(signature),
                parent_code=parent_code,
                is_strategic_replan=is_strategic_replan,
                replan_reason=replan_reason
            )
            self._update_plan_with_new_code(function_name, new_code)
            logger.info(f"Dynamically implemented function: '{function_name}'.")
        else:
            raise RuntimeError(f"Could not find function '{function_name}' to implement.")

    def _get_unimplemented_function_name(self) -> str:
        """Extracts the function name from the latest traceback."""
        _, _, exc_tb = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_tb)[-1]
        return frame_summary.name

    def _get_main_function_name(self) -> str | None:
        """Parses the source code to find the first defined function."""
        import ast
        try:
            tree = ast.parse(self.plan_source_code or "")
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return node.name
        except SyntaxError:
            return None
        return None

    def _update_plan_with_new_code(self, function_name: str, new_code: str):
        """Replaces a function stub or existing function with new code."""
        import re
        # Regex to find the whole function definition, decorated or not.
        pattern = re.compile(
            rf"(?:@verify\s*\n)?(?:async\s+)?def\s+{function_name}\(.*\):(?:\n\s+.*)*",
            re.MULTILINE
        )
        if pattern.search(self.plan_source_code):
             new_source = pattern.sub(textwrap.dedent(new_code).strip(), self.plan_source_code or "", 1)
        else: # Fallback for simple stubs
            pattern = re.compile(
                rf"def\s+{function_name}\(.*\):\s*\n(?:\s+.*?\n)*?\s+raise NotImplementedError",
                re.MULTILINE
            )
            new_source = pattern.sub(textwrap.dedent(new_code).strip(), self.plan_source_code or "", 1)

        self.plan_source_code = new_source
        logger.debug(f"Updated plan source code:\n{self.plan_source_code}")
        exec(self.plan_source_code, self.execution_namespace)

    async def modify_plan(self, modification_request: str):
        """Handles 'Modifying Mode' workflow as per Phase 4 spec."""
        logger.info(f"Starting plan modification: '{modification_request}'")
        self.pause()
        self._is_paused_for_modification = True

        try:
            # 1. Plan Surgery
            new_source_code = await self.planner._perform_plan_surgery(self.plan_source_code, modification_request)
            
            # 2. Course Correction
            correction_script = await self.planner._generate_course_correction_script(self.plan_source_code, new_source_code)
            if correction_script:
                logger.info("Executing course correction script...")
                await self._execute_correction_script(correction_script)
                logger.info("Course correction finished.")

            # 3. Update and Resume
            self.plan_source_code = new_source_code
            logger.info("Plan successfully modified.")

        except Exception as e:
            logger.error(f"Failed to modify plan: {e}", exc_info=True)
        finally:
            self._is_paused_for_modification = False
            self.resume()
            # We need to restart the execution loop with the new code
            await self._start_main_execution_loop()

    async def _execute_correction_script(self, script: str):
        """Executes a short-lived correction script in its own isolated handle."""
        correction_namespace = self.execution_namespace.copy()
        exec(script, correction_namespace)
        correction_fn = correction_namespace.get('course_correction_main')
        if not correction_fn:
            raise RuntimeError("Course correction script did not define 'course_correction_main' function.")
        
        # We don't need a full steerable loop, just execute the coroutine
        await correction_fn()


    async def result(self) -> str:
        await self._execution_task
        if not self.main_loop_handle:
            return "Error: Main execution loop was not created."
        return await self.main_loop_handle.result()

    def done(self) -> bool:
        return self.main_loop_handle.done() if self.main_loop_handle else False

    def stop(self):
        if self.main_loop_handle: self.main_loop_handle.stop()

    def pause(self):
        if self.main_loop_handle: self.main_loop_handle.pause()

    def resume(self):
        if self.main_loop_handle: self.main_loop_handle.resume()

    async def interject(self, message: str):
        if self.main_loop_handle: await self.main_loop_handle.interject(message)

    async def ask(self, question: str) -> str:
        return f"Plan state: {self.call_stack}. Goal: {self.goal}"

    @property
    def valid_tools(self) -> Dict[str, Callable]:
        return {}


class HierarchicalPlanner(BasePlanner[HierarchicalPlan]):
    """
    Orchestrates task decomposition, execution, and dynamic modification
    by generating and running Python code.
    """
    def __init__(
        self,
        task_manager: "TaskManager",
        function_manager: "FunctionManager",
        controller: "Controller",
        coms_manager: "ComsManager",
    ):
        super().__init__()
        self.task_manager = task_manager
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
        """Factory method to create and start a new hierarchical plan."""
        plan = HierarchicalPlan(planner=self, goal=task_description)
        return plan

    async def _generate_initial_plan(self, goal: str) -> str:
        """
        Generates the initial Python code for the plan, querying the
        FunctionManager for existing implementations.
        """
        # Phase 3: Query FunctionManager
        try:
            # A real implementation would do a semantic search on the goal
            existing_functions = self.function_manager.list_functions(include_implementations=True)
            if existing_functions:
                 logger.info(f"Found {len(existing_functions)} existing functions in FunctionManager.")
            else:
                 logger.info("No existing functions found in FunctionManager.")
        except Exception as e:
            logger.warning(f"Could not query FunctionManager: {e}")
            existing_functions = {}

        primitives_doc = """
- `await act(instruction: str) -> str`: Instructs the browser controller to perform a complex action.
- `await observe(query: str) -> str`: Observes the browser state.
- `await reason(context: str, question: str) -> str`: A general-purpose LLM call for reasoning.
- `await start_call(number: str, purpose: str) -> CallHandle`: Initiates a voice call.
- `await send_email(recipient: str, subject: str, body: str) -> bool`: Sends an email.
"""

        existing_functions_doc = "\n".join(
            f'- `{name}{data["argspec"]}`: {data["docstring"]}'
            for name, data in existing_functions.items()
        )
        full_existing_code = "\n\n".join(
            data["implementation"] for data in existing_functions.values()
        )

        prompt = f"""
You are an expert Python programmer and a meticulous planner. Your task is to take a high-level user goal and decompose it into a Python script.

**User Goal:** "{goal}"

**Available Primitives:**
{primitives_doc}

**Available Pre-Implemented Functions from FunctionManager:**
{existing_functions_doc if existing_functions else "None"}

**Instructions:**
1.  Structure the plan as a set of `async def` functions with a single main entry point.
2.  **Reuse existing functions where possible.** If a suitable function from the FunctionManager exists, call it directly. Do NOT redefine it.
3.  For any new, complex sub-task, define a helper function. If its implementation is not immediately obvious, stub it with `raise NotImplementedError`.
4.  All functions you generate (new stubs or new fully implemented functions) MUST be decorated with `@verify`.
5.  The final script should contain the full code for any reused functions, followed by your newly generated functions.

**Full code of available functions (for injection):**
```python
{full_existing_code if full_existing_code else "# No existing functions to inject."}
```

Now, generate the complete `async` Python script for the user goal.
"""
        response = await llm_call(self.llm_client, textwrap.dedent(prompt))
        code = response.strip().replace("```python", "").replace("```", "").strip()
        return code

    async def _dynamic_implement(
        self,
        plan: HierarchicalPlan,
        function_name: str,
        function_signature: str,
        parent_code: str,
        is_strategic_replan: bool = False,
        replan_reason: str = ""
    ) -> str:
        """Generates the async implementation for a function."""
        try:
            browser_state = await self.controller.observe("Describe the current page for context.")
        except Exception:
            browser_state = "Could not observe browser state."

        replan_context = ""
        if is_strategic_replan:
            replan_context = f"""
**REPLANNING REQUIRED:**
The previous attempt to implement this function failed for a strategic reason: '{replan_reason}'.
You must come up with a different approach to achieve the function's goal. Do not repeat the previous mistake.
"""

        prompt = f"""
You are an expert Python programmer. Your task is to implement a single `async` Python function.
{replan_context}
**Overall Task Goal:** "{plan.goal}"
**Function to Implement:** `async def {function_name}{function_signature}:`
**Context:** This function is called from:
```python
{parent_code}
```
**Current Browser State:**
"{browser_state}"
**Instructions:**
1.  Write the full `async def` implementation for `{function_name}`.
2.  The implementation MUST be decorated with `@verify`.
3.  Only use async primitives: `await act(...)`, `await observe(...)`, etc.
4.  Provide a complete function: signature, decorator, docstring, and body.
"""
        response = await llm_call(self.llm_client, textwrap.dedent(prompt))
        code = response.strip().replace("```python", "").replace("```", "").strip()
        return code

    async def _prepare_execution_environment(self, plan: HierarchicalPlan):
        """Prepares the namespace for executing the generated code."""
        # This now includes ComsManager primitives as per Phase 3.
        plan.execution_namespace.update({
            'act': self.controller.act,
            'observe': self.controller.observe,
            'reason': self._reason_primitive,
            'start_call': self.coms_manager.start_call,
            'send_email': self.coms_manager.send_email,
            'verify': self._create_verify_decorator(plan),
            'ReplanFromParentException': ReplanFromParentException,
            'asyncio': asyncio
        })

    async def _reason_primitive(self, context: str, question: str) -> str:
        prompt = f"Given the context:\n{context}\n\nPlease answer the following question:\n{question}"
        return await llm_call(self.llm_client, prompt)

    async def _check_state_against_goal(
        self, function_name: str, function_docstring: str | None,
        last_action_description: str, observed_state: str
    ) -> VerificationAssessment:
        """Uses an LLM to assess if the last action achieved the function's goal."""
        prompt = f"""
You are a verification agent. Your job is to assess if an action was successful by comparing the outcome to the intended goal.

**Function Goal:**
- Name: `{function_name}`
- Purpose: `{function_docstring or 'No docstring provided.'}`

**Action Taken:** `{last_action_description}`
**Observed Outcome:** `{observed_state}`

**Assessment Task:**
Respond with a JSON object: {{"status": "...", "reason": "..."}}.
**Possible Statuses:**
- "ok": The action was successful and the function's goal appears to be met.
- "reimplement_local": A tactical error occurred. The implementation is flawed but the goal is sound.
- "replan_parent": A strategic error occurred. The function's entire premise is wrong or unachievable.
- "fatal_error": An unrecoverable system error occurred.
"""
        response_str = await llm_call(self.llm_client, textwrap.dedent(prompt))
        try:
            return VerificationAssessment(**json.loads(response_str))
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to decode assessment from LLM: {e}\nResponse: {response_str}")
            return VerificationAssessment(status="fatal_error", reason="LLM provided a malformed assessment.")

    def _create_verify_decorator(self, plan: HierarchicalPlan):
        """Creates the @verify decorator for async functions, with strategic replan logic."""
        def verify(fn):
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                max_retries = 3
                last_exception = None
                plan.call_stack.append(fn.__name__)
                logger.info(f"VERIFY: Entering '{fn.__name__}' (stack: {plan.call_stack})")

                for attempt in range(max_retries):
                    try:
                        # Phase 4: Handle strategic replan from a child call
                        return await self._execute_and_verify_step(plan, fn, args, kwargs)
                    except ReplanFromParentException as e:
                        logger.warning(f"VERIFY: Caught strategic failure from child of '{fn.__name__}'. Replanning '{fn.__name__}'. Reason: {e}")
                        # Re-implement the CURRENT function due to its child's strategic failure
                        await plan._handle_dynamic_implementation(fn.__name__, is_strategic_replan=True, replan_reason=str(e))
                        logger.info(f"VERIFY: Retrying '{fn.__name__}' after strategic replan (attempt {attempt + 2}).")
                        continue # Retry the current function with its new implementation
                    except NotImplementedError:
                        plan.call_stack.pop()
                        raise
                    except Exception as e:
                        last_exception = e
                        logger.error(f"VERIFY: Exception in '{fn.__name__}' (attempt {attempt + 1}): {e}", exc_info=False)
                        continue
                
                plan.call_stack.pop()
                raise ReplanFromParentException(f"Function '{fn.__name__}' failed after {max_retries} attempts. Last error: {last_exception}")

            return wrapper
        return verify

    async def _execute_and_verify_step(self, plan: HierarchicalPlan, fn: Callable, args, kwargs) -> Any:
        """Helper to encapsulate the execution and verification of a single step."""
        last_action = {"desc": "No action taken"}
        last_observation = {"state": "No observation made"}

        async def act_wrapper(instruction: str):
            last_action['desc'] = f'act("{instruction}")'
            return await self.controller.act(instruction)

        async def observe_wrapper(query: str):
            res = await self.controller.observe(query)
            last_observation['state'] = res
            return res

        original_act = plan.execution_namespace.get('act')
        original_observe = plan.execution_namespace.get('observe')
        plan.execution_namespace['act'] = act_wrapper
        plan.execution_namespace['observe'] = observe_wrapper

        result = await fn(*args, **kwargs)

        plan.execution_namespace['act'] = original_act
        plan.execution_namespace['observe'] = original_observe
        
        assessment = await self._check_state_against_goal(
            function_name=fn.__name__,
            function_docstring=fn.__doc__,
            last_action_description=last_action['desc'],
            observed_state=last_observation['state']
        )
        logger.info(f"VERIFY Assessment for '{fn.__name__}': {assessment.status} - {assessment.reason}")
        
        if assessment.status == "ok":
            logger.info(f"VERIFY: Success for '{fn.__name__}'. Saving to FunctionManager.")
            # Phase 3: Persist successful function
            try:
                func_code = inspect.getsource(fn)
                self.function_manager.add_functions(implementations=[func_code])
            except Exception as e:
                logger.warning(f"Could not save function '{fn.__name__}' to FunctionManager: {e}")
            plan.call_stack.pop()
            return result

        elif assessment.status == "reimplement_local":
            logger.warning(f"VERIFY: Tactical error in '{fn.__name__}'. Triggering re-implementation.")
            await plan._handle_dynamic_implementation(fn.__name__)
            raise RuntimeError("Forced retry after local reimplementation") # This will be caught by the outer loop to retry

        elif assessment.status == "replan_parent":
            raise ReplanFromParentException(f"Strategic failure in '{fn.__name__}': {assessment.reason}")

        elif assessment.status == "fatal_error":
            raise RuntimeError(f"Fatal error in '{fn.__name__}': {assessment.reason}")

    async def _perform_plan_surgery(self, current_code: str, request: str) -> str:
        """Uses an LLM to rewrite a part of the plan's source code."""
        prompt = f"""
You are a master programmer tasked with modifying a Python script for an AI agent.

**User's Modification Request:**
"{request}"

**Current Plan Source Code:**
```python
{current_code}
```

**Instructions:**
Rewrite the Python script to incorporate the user's change.
- Identify the correct function(s) to modify.
- You can add, remove, or change functions.
- If you add a new function that needs implementation, stub it with `raise NotImplementedError`.
- Ensure all functions remain decorated with `@verify`.
- Output only the complete, new Python script.
"""
        response = await llm_call(self.llm_client, textwrap.dedent(prompt))
        return response.strip().replace("```python", "").replace("```", "").strip()

    async def _generate_course_correction_script(self, old_code: str, new_code: str) -> Optional[str]:
        """Generates a Python script to bridge the state gap between an old and new plan."""
        current_state = await self.controller.observe("Describe the current page, including URL and main elements, for state analysis.")
        
        prompt = f"""
You are an expert state transition analyst for an AI agent. The agent's plan has been modified, and it might be in the wrong state to continue. Your task is to generate a 'course correction' script if necessary.

**Current Browser State:**
{current_state}

**Old Plan (partial for context):**
```python
{old_code[:1000]}...
```

**New Plan:**
```python
{new_code}
```

**Analysis Task:**
1.  Analyze the `new_plan` to determine the expected starting state for its execution (e.g., "should be on the LinkedIn search results page").
2.  Compare this required state with the `current_browser_state`.
3.  If they are different, generate a short, self-contained Python script named `course_correction_main` that uses `await act()` and `await observe()` to navigate from the current state to the required state.
4.  If no correction is needed, respond with "None".

**Example Correction Script:**
```python
async def course_correction_main():
    \"\"\"Navigates back to the search results page.\"\"\"
    await act("Click the browser's back button twice.")
```

**Your Response:**
Generate either the Python script or the word "None".
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
            # Assuming coms_manager might have an async stop method
            stop_coro = self.coms_manager.stop()
            if inspect.iscoroutine(stop_coro):
                await stop_coro
