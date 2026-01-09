import asyncio
import io
import traceback
import json
import ast
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, Optional, Callable, Awaitable, Type, TYPE_CHECKING
from pydantic import BaseModel

from unity.actor.base import BaseActor
from unity.actor.handle import ActorHandle
from unity.function_manager.primitives import ComputerPrimitives
from unity.actor.prompt_builders import _build_code_act_rules_and_examples
from unity.image_manager.types.image_refs import ImageRefs
from unity.image_manager.types.raw_image_ref import RawImageRef
from unity.image_manager.types.annotated_image_ref import AnnotatedImageRef

if TYPE_CHECKING:
    from unity.actor.environments.base import BaseEnvironment
    from unity.function_manager.function_manager import FunctionManager


def build_code_act_system_prompt(
    environments: Dict[str, "BaseEnvironment"],
    tools: Optional[Dict[str, Callable]] = None,
) -> str:
    """Builds the rich system prompt for the CodeActActor with enhanced quality.

    Args:
        environments: Active execution environments (computer_primitives, primitives, etc.)
        tools: The tools dict - used to dynamically render additional tools (e.g. FunctionManager).
    """
    from unity.common.prompt_helpers import render_tools_block

    rules_and_examples = _build_code_act_rules_and_examples(environments=environments)

    # Primary execution tool - always present and deserves its own dedicated section
    execute_tool = {
        "execute_python_code": {
            "signature": "async def execute_python_code(thought: str, code: str) -> Any",
            "docstring": (
                "Executes a block of Python code in a stateful sandbox and returns the result.\n"
                "You have access to environment globals injected into the sandbox (e.g. `computer_primitives`, `primitives`).\n"
                "All variables are preserved between calls. The sandbox is asynchronous - use await for all async methods."
            ),
        },
    }
    execute_tool_reference = json.dumps(execute_tool, indent=4)

    has_browser_env = "computer_primitives" in environments
    role_line = "You are an expert agent that solves tasks by writing and executing Python code."
    capabilities_line = (
        "Your primary tool is a stateful code execution sandbox where you can control browsers, "
        "send communications, and perform complex automation tasks."
        if has_browser_env
        else "Your primary tool is a stateful code execution sandbox where you can use whatever tool "
        "domains are available via injected environment globals (e.g. state managers, and optionally browser/desktop)."
    )

    prompt = f"""
### Your Role: Code-First Automation Agent
{role_line} {capabilities_line}

### Primary Execution Tool
```json
{execute_tool_reference}
```

{rules_and_examples}
"""

    # Dynamically render additional tools (e.g. FunctionManager tools) if present
    if tools:
        # Filter out execute_python_code since it has its own dedicated section above
        additional_tools = {
            k: v for k, v in tools.items() if k != "execute_python_code"
        }
        if additional_tools:
            prompt += (
                f"\n### Additional Tools (JSON Tool Calls)\n"
                f"These tools are called via **structured JSON tool calls**, NOT inside Python code.\n\n"
                f"{render_tools_block(additional_tools)}\n"
            )

        # Add FunctionManager guidance if its tools are present
        has_fm_tools = any(k.startswith("FunctionManager_") for k in tools)
        if has_fm_tools:
            prompt += """
### Function Library (CRITICAL)

You have access to a catalogue of **pre-stored reusable functions** via the FunctionManager tools listed above.

**🎯 FUNCTION-FIRST WORKFLOW:**

1. **ALWAYS search first** using FunctionManager tools (structured tool calls, NOT Python code):
   - `FunctionManager_search_functions_by_similarity` - semantic search for functions
   - `FunctionManager_search_functions` - filter-based search
   - `FunctionManager_list_functions` - list all available functions

2. **Functions are automatically injected into your Python sandbox** after searching.
   - The function name(s) returned by the tool become available immediately in Python.
   - Dependencies are injected automatically (including nested helper functions).
   - Venv-backed functions work transparently (subprocess RPC hidden behind an awaitable callable).

3. **If found → USE IT**: Pre-saved functions are tested, optimized, and handle edge cases.
   Don't re-explore tables/schemas when a function already does the job.

4. **Read signatures carefully**: Check `argspec` in the search results for parameter options
   like `group_by`, `include_plots`, date filters, etc.

5. **Execute found functions** in your Python code:

```python
# Step 1: Search (JSON tool call)
FunctionManager_search_functions_by_similarity(query="analyze sales data")

# Step 2: Execute in Python code (functions now available)
result = await analyze_sales_data(tools=primitives.files.get_tools())
```

**❌ ANTI-PATTERN (AVOID THIS):**
```python
# DON'T explore tables when a function already exists!
tables = await primitives.files.tables_overview()  # Unnecessary!
schema = await primitives.files.schema_explain(...)  # Unnecessary!
```

**✅ CORRECT WORKFLOW:**
1. Call `FunctionManager_search_functions_by_similarity` tool with your query
2. Review the returned functions and their `argspec`
3. Execute the function in Python code with appropriate parameters

**When passing tools to functions:**
- Functions accepting `tools: FileTools` need: `tools = primitives.files.get_tools()`
- For direct data operations, use: `await primitives.files.reduce(...)`
"""

    return prompt


class CodeExecutionSandbox:
    """
    A stateful sandbox for executing Python code asynchronously.

    This class maintains a persistent global state across multiple executions,
    capturing stdout, stderr, return values, and exceptions in a structured format.
    """

    def __init__(
        self,
        computer_primitives: Optional[ComputerPrimitives] = None,
        environments: Optional[Dict[str, "BaseEnvironment"]] = None,
    ):
        """
        Initializes the sandbox.

        Args:
            computer_primitives: An instance of ComputerPrimitives to be injected into the
                             sandbox's global state, making browser tools available.
            environments: Optional mapping of environment namespaces to environments. If
                provided, each environment instance is injected into the sandbox globals.
        """
        from unity.function_manager.execution_env import create_execution_globals

        self.global_state: Dict[str, Any] = create_execution_globals()
        self._browser_used: bool = False

        class _UsageTrackingProxy:
            def __init__(self, target: Any, on_use: Callable[[], None]):
                self._target = target
                self._on_use = on_use

            def __getattr__(self, name: str) -> Any:
                # Treat any access as potential "use" since callers may invoke nested objects
                # like `computer_primitives.browser.get_screenshot()`.
                self._on_use()
                attr = getattr(self._target, name)
                if callable(attr):

                    async def _async_wrapper(*args, **kwargs):
                        self._on_use()
                        return await attr(*args, **kwargs)

                    def _sync_wrapper(*args, **kwargs):
                        self._on_use()
                        return attr(*args, **kwargs)

                    # Preserve sync vs async callable behavior.
                    if asyncio.iscoroutinefunction(attr):
                        return _async_wrapper
                    return _sync_wrapper
                return attr

        def _mark_browser_used() -> None:
            self._browser_used = True

        if environments:
            for namespace, env in environments.items():
                try:
                    # Use get_sandbox_instance() if available (for filtered primitives),
                    # otherwise fall back to get_instance()
                    if hasattr(env, "get_sandbox_instance"):
                        instance = env.get_sandbox_instance()
                    else:
                        instance = env.get_instance()
                    if namespace == "computer_primitives":
                        instance = _UsageTrackingProxy(instance, _mark_browser_used)
                    self.global_state[namespace] = instance
                except Exception:
                    # Keep sandbox usable even if a non-critical environment fails to inject.
                    continue

        # Backward-compat: allow direct injection when environments weren't provided.
        if computer_primitives and "computer_primitives" not in self.global_state:
            self.global_state["computer_primitives"] = _UsageTrackingProxy(
                computer_primitives,
                _mark_browser_used,
            )

    async def execute(self, code: str) -> Dict[str, Any]:
        """
        Executes a string of Python code within the sandbox's stateful environment.
        """
        # Reset per-execution usage flags.
        self._browser_used = False
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
                            alias.asname or alias.name.split(".")[0],
                        )

                elif isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
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
            "browser_used": self._browser_used,
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
        agent_mode: str = "browser",
        agent_server_url: str = "http://localhost:3000",
        computer_primitives: Optional["ComputerPrimitives"] = None,
        environments: Optional[list["BaseEnvironment"]] = None,
        function_manager: Optional["FunctionManager"] = None,
    ):
        """
        Initializes the CodeActActor.

        Args:
            computer_primitives: Optional existing ComputerPrimitives instance to reuse.
                           If provided, other browser-related params are ignored.
            environments: Optional list of execution environments. If None, defaults to
                [ComputerEnvironment, StateManagerEnvironment].
            function_manager: Manages a library of reusable functions. Exposes read-only tools
                (list_functions, search_functions, search_functions_by_similarity) to the LLM.
                The LLM can call these tools to discover and retrieve reusable function implementations.
        """
        super().__init__(
            environments=environments,
            computer_primitives=computer_primitives,
            function_manager=function_manager,
            session_connect_url=session_connect_url,
            headless=headless,
            browser_mode=browser_mode,
            agent_mode=agent_mode,
            agent_server_url=agent_server_url,
        )

        self._sandbox = CodeExecutionSandbox(
            computer_primitives=self._computer_primitives,
            environments=self.environments,
        )
        self._timeout = timeout
        self._browser_tools = self._get_browser_tools()
        self._tools = self._build_tools()

        self._main_event_loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            self._main_event_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

    def _get_browser_tools(self) -> Dict[str, Callable]:
        """Extracts browser-related methods from the ComputerPrimitives."""
        if not self._computer_primitives:
            return {}
        return {
            "navigate": self._computer_primitives.navigate,
            "act": self._computer_primitives.act,
            "observe": self._computer_primitives.observe,
        }

    def _build_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        """Builds the dictionary of tools available to the LLM."""

        async def execute_python_code(thought: str, code: Optional[str] = None) -> Any:
            """
            Executes a block of Python code in a stateful sandbox after reasoning about the step.

            Args:
                thought: A detailed, step-by-step reasoning of what you are about to do and why.
                code: The Python code to execute. Can be None if only thinking is required.
            """
            if code is None or code.strip() == "":
                return "Acknowledged thought. No code to execute."

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
                    f"--- RESULT ---\n{repr(execution_result['result'])}",
                )

            text_summary = "\n\n".join(output_parts)
            if not text_summary:
                text_summary = "Code executed successfully with no output."

            # Only append browser state when a browser environment is active.
            # Avoid any heuristics based on code substring matching.
            if (
                "computer_primitives" in self.environments
                and self._computer_primitives is not None
                and execution_result.get("browser_used")
            ):
                try:
                    url = await self._computer_primitives.browser.get_current_url()
                    screenshot_b64 = (
                        await self._computer_primitives.browser.get_screenshot()
                    )

                    browser_state_summary = f"--- BROWSER STATE ---\nURL: {url}"
                    text_summary += f"\n\n{browser_state_summary}"

                    # Only attach an image if we received non-empty base64.
                    # Some providers reject empty image payloads.
                    if screenshot_b64:
                        return {"summary": text_summary, "image": screenshot_b64}
                    return {"summary": text_summary}
                except Exception as e:
                    text_summary += f"\n\n--- BROWSER STATE ERROR ---\nCould not retrieve browser state: {e}"

            return text_summary

        tools: Dict[str, Callable[..., Awaitable[Any]]] = {
            "execute_python_code": execute_python_code,
        }

        # Add FunctionManager tools (auto-inject callables into sandbox) if available.
        #
        # IMPORTANT:
        # These tools are called via JSON tool calls (not inside Python). They return
        # metadata to the LLM while injecting the matching function callables into the
        # sandbox global namespace so they can be executed immediately in Python code.
        if self.function_manager:

            async def FunctionManager_search_functions_by_similarity(
                query: str,
                n: int = 5,
            ) -> Any:
                """
                Search for functions by semantic similarity to a natural-language query.

                Functions are automatically injected into your Python sandbox namespace,
                so you can execute them immediately after searching.
                """
                result = self.function_manager.search_functions_by_similarity(
                    query=query,
                    n=n,
                    return_callable=True,
                    namespace=self._sandbox.global_state,
                    also_return_metadata=True,
                )
                return result["metadata"]

            async def FunctionManager_search_functions(
                filter: Optional[str] = None,
                offset: int = 0,
                limit: int = 100,
            ) -> Any:
                """
                Search for functions using a Python-like filter expression.

                Functions are automatically injected into your Python sandbox namespace,
                so you can execute them immediately after searching.
                """
                result = self.function_manager.search_functions(
                    filter=filter,
                    offset=offset,
                    limit=limit,
                    return_callable=True,
                    namespace=self._sandbox.global_state,
                    also_return_metadata=True,
                )
                return result["metadata"]

            async def FunctionManager_list_functions(
                include_implementations: bool = False,
            ) -> Any:
                """
                List available functions.

                Functions are automatically injected into your Python sandbox namespace,
                so you can execute them immediately after listing.
                """
                result = self.function_manager.list_functions(
                    include_implementations=include_implementations,
                    return_callable=True,
                    namespace=self._sandbox.global_state,
                    also_return_metadata=True,
                )
                return result["metadata"]

            tools["FunctionManager_search_functions_by_similarity"] = (
                FunctionManager_search_functions_by_similarity
            )
            tools["FunctionManager_search_functions"] = FunctionManager_search_functions
            tools["FunctionManager_list_functions"] = FunctionManager_list_functions

        return tools

    async def act(
        self,
        description: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        images: Optional[ImageRefs | list[RawImageRef | AnnotatedImageRef]] = None,
        **kwargs,
    ) -> ActorHandle:
        """
        Creates and starts a new ActorHandle for the CodeAct agent.
        """
        if not self._main_event_loop:
            self._main_event_loop = asyncio.get_running_loop()

        is_interactive_session = not description

        initial_prompt = (
            "This is an interactive session. Acknowledge that you are ready and "
            "wait for the user to provide instructions via interjection."
        )

        system_prompt = build_code_act_system_prompt(
            self.environments,
            tools=self._tools,
        )
        handle = ActorHandle(
            task_description=description or initial_prompt,
            tools=self._tools,
            parent_chat_context=_parent_chat_context,
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
            main_event_loop=self._main_event_loop,
            timeout=self._timeout,
            persist=is_interactive_session,
            custom_system_prompt=system_prompt,
            tool_policy=None,
            computer_primitives=self._computer_primitives,
            images=images,
        )
        return handle

    async def close(self):
        """Shuts down the actor and its associated resources gracefully."""
        if self._computer_primitives:
            self._computer_primitives.browser.stop()
