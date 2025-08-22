import asyncio
import io
import traceback
import json
import ast
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, Optional, Callable, Awaitable

from unity.actor.base import BaseActor
from unity.actor.tool_loop_actor import ToolLoopPlan
from unity.actor.action_provider import ActionProvider
from unity.actor.prompt_builders import _build_code_act_rules_and_examples


def build_code_act_system_prompt(
    action_provider: ActionProvider,
) -> str:
    """Builds the rich system prompt for the CodeActActor with enhanced quality."""

    rules_and_examples = _build_code_act_rules_and_examples(action_provider)

    execute_tool = {
        "execute_python_code": {
            "signature": "async def execute_python_code(code: str) -> Any",
            "docstring": (
                "Executes a block of Python code in a stateful sandbox and returns the result.\n"
                "You have access to an `action_provider` object to control a web browser and send communications.\n"
                "All variables are preserved between calls. The sandbox is asynchronous - use await for all action_provider methods."
            ),
        },
    }
    execute_tool_reference = json.dumps(execute_tool, indent=4)

    return f"""
### Your Role: Code-First Automation Agent
You are an expert agent that solves tasks by writing and executing Python code. Your primary tool is a stateful code execution sandbox where you can control browsers, send communications, and perform complex automation tasks.

### Primary Execution Tool
You MUST use this tool for ALL code execution tasks such as
- Browser automation (navigate, act, observe)
- Data extraction and processing
- Sending communications (SMS, email, WhatsApp)
- Complex calculations
- Multi-step workflows


```json
{execute_tool_reference}
```

{rules_and_examples}

### CRITICAL REMINDER
Every response MUST be a tool call to `execute_python_code` **UNLESS** you are providing the final answer to the user. Use the code execution sandbox for everything until you are ready to conclude the task.

Begin your response now. You must use the `execute_python_code` tool.
"""


class CodeExecutionSandbox:
    """
    A stateful sandbox for executing Python code asynchronously.

    This class maintains a persistent global state across multiple executions,
    capturing stdout, stderr, return values, and exceptions in a structured format.
    """

    def __init__(self, action_provider: Optional[ActionProvider] = None):
        """
        Initializes the sandbox.

        Args:
            action_provider: An instance of ActionProvider to be injected into the
                             sandbox's global state, making browser tools available.
        """
        self.global_state: Dict[str, Any] = {}
        if action_provider:
            self.global_state["action_provider"] = action_provider
            from pydantic import BaseModel, Field

            self.global_state["BaseModel"] = BaseModel
            self.global_state["Field"] = Field

    async def execute(self, code: str) -> Dict[str, Any]:
        """
        Executes a string of Python code within the sandbox's stateful environment.
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        result = None
        error = None

        try:
            is_empty_or_comment_only = all(
                line.strip() == "" or line.strip().startswith("#")
                for line in code.splitlines()
            )
            if is_empty_or_comment_only:
                code += "\npass"

            tree = ast.parse(code)
            top_level_assign_targets = set()
            for node in tree.body:
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = []
                    if isinstance(node, ast.Assign):
                        targets.extend(node.targets)
                    else:
                        targets.append(node.target)

                    for target in targets:
                        if isinstance(target, ast.Name):
                            top_level_assign_targets.add(target.id)
                        elif isinstance(target, ast.Tuple):
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    top_level_assign_targets.add(elt.id)

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        top_level_assign_targets.add(
                            alias.asname or alias.name.split(".")[0]
                        )

                elif isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    top_level_assign_targets.add(node.name)

            async_code = "async def __exec_wrapper():\n"
            if top_level_assign_targets:
                async_code += (
                    f"    global {', '.join(sorted(list(top_level_assign_targets)))}\n"
                )

            async_code += "".join(f"    {line}\n" for line in code.splitlines())

            exec(async_code, self.global_state)

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                result = await self.global_state["__exec_wrapper"]()

        except Exception:
            error = traceback.format_exc()
        finally:
            if "__exec_wrapper" in self.global_state:
                del self.global_state["__exec_wrapper"]

        return {
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "result": result,
            "error": error,
        }


class CodeActActor(BaseActor):
    """
    An actor that uses a conversational tool loop and a stateful code execution
    sandbox to accomplish tasks. It acts as a baseline for code-centric agents.
    """

    def __init__(
        self,
        session_connect_url: Optional[str] = None,
        headless: bool = False,
        browser_mode: str = "magnitude",
        timeout: float = 1000,
    ):
        """
        Initializes the CodeActActor.
        """
        self._action_provider = ActionProvider(
            session_connect_url=session_connect_url,
            headless=headless,
            browser_mode=browser_mode,
        )
        self._sandbox = CodeExecutionSandbox(action_provider=self._action_provider)
        self._timeout = timeout
        self._browser_tools = self._get_browser_tools()
        self._tools = self._build_tools()
        self.system_prompt = build_code_act_system_prompt(self._action_provider)

        self._main_event_loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            self._main_event_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

    def _get_browser_tools(self) -> Dict[str, Callable]:
        """Extracts browser-related methods from the ActionProvider."""
        return {
            "browser_navigate": self._action_provider.browser_navigate,
            "browser_act": self._action_provider.browser_act,
            "browser_observe": self._action_provider.browser_observe,
        }

    def _build_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        """Builds the dictionary of tools available to the LLM."""

        async def execute_python_code(code: str) -> Any:
            """
            Executes a block of Python code in a stateful sandbox and returns the result.
            You have access to an `action_provider` object to control a web browser.
            All variables are preserved between calls.
            """
            execution_result = await self._sandbox.execute(code)

            output_parts = []
            if execution_result["stdout"]:
                output_parts.append(f"--- STDOUT ---\n{execution_result['stdout']}")
            if execution_result["stderr"]:
                output_parts.append(f"--- STDERR ---\n{execution_result['stderr']}")
            if execution_result["error"]:
                output_parts.append(f"--- ERROR ---\n{execution_result['error']}")
            if execution_result["result"] is not None:
                output_parts.append(
                    f"--- RESULT ---\n{repr(execution_result['result'])}"
                )

            text_summary = "\n\n".join(output_parts)
            if not text_summary:
                text_summary = "Code executed successfully with no output."

            browser_action_keywords = [
                "browser_navigate",
                "browser_act",
                "browser_observe",
            ]
            if any(keyword in code for keyword in browser_action_keywords):
                try:
                    url = await self._action_provider.browser.get_current_url()
                    screenshot_b64 = (
                        await self._action_provider.browser.get_screenshot()
                    )

                    browser_state_summary = f"--- BROWSER STATE ---\nURL: {url}"
                    text_summary += f"\n\n{browser_state_summary}"

                    return {"summary": text_summary, "image": screenshot_b64}
                except Exception as e:
                    text_summary += f"\n\n--- BROWSER STATE ERROR ---\nCould not retrieve browser state: {e}"

            return text_summary

        return {"execute_python_code": execute_python_code}

    async def act(
        self,
        description: str,
        *,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        **kwargs,
    ) -> ToolLoopPlan:
        """
        Creates and starts a new ToolLoopPlan for the CodeAct agent.
        """
        if not self._main_event_loop:
            self._main_event_loop = asyncio.get_running_loop()

        is_interactive_session = not description

        initial_prompt = (
            "This is an interactive session. Acknowledge that you are ready and "
            "wait for the user to provide instructions via interjection."
        )

        def dynamic_tool_policy(step_index: int, tools: Dict[str, Callable]):
            """
            Allows a text response on the first turn of an interactive session,
            but requires tool use for all subsequent turns.
            """
            if is_interactive_session:
                return ("auto", tools)
            return ("required", tools)

        plan = ToolLoopPlan(
            task_description=description or initial_prompt,
            tools=self._tools,
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
            main_event_loop=self._main_event_loop,
            timeout=self._timeout,
            persist=is_interactive_session,
            custom_system_prompt=self.system_prompt,
            tool_policy=dynamic_tool_policy,
        )
        return plan

    async def close(self):
        """Shuts down the actor and its associated resources gracefully."""
        if self._action_provider:
            self._action_provider.browser.stop()
