import asyncio
import io
import traceback
import json
import ast
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, Optional, Callable, Awaitable

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

