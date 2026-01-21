import asyncio
import contextvars
import functools
import inspect
import io
import traceback
import json
import ast
import copy
import uuid
from datetime import datetime, timezone
import logging
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, Optional, Callable, Awaitable, Type, TYPE_CHECKING
from pydantic import BaseModel

from unity.actor.base import BaseCodeActActor
from unity.actor.handle import ActorHandle
from unity.common.async_tool_loop import SteerableToolHandle, start_async_tool_loop
from unity.common.llm_client import new_llm_client
from unity.function_manager.primitives import ComputerPrimitives
from unity.actor.prompt_builders import build_code_act_prompt
from unity.events.manager_event_logging import log_manager_call
from unity.image_manager.types.image_refs import ImageRefs
from unity.image_manager.types.raw_image_ref import RawImageRef
from unity.image_manager.types.annotated_image_ref import AnnotatedImageRef

if TYPE_CHECKING:
    from unity.actor.environments.base import BaseEnvironment
    from unity.function_manager.function_manager import FunctionManager


_CURRENT_SANDBOX: contextvars.ContextVar["CodeExecutionSandbox"] = contextvars.ContextVar(
    "code_act_current_sandbox"
)

logger = logging.getLogger(__name__)


class _CodeActEntrypointHandle(SteerableToolHandle):  # type: ignore[abstract-method]
    """Execute a FunctionManager entrypoint function without invoking the CodeAct LLM loop.

    TaskScheduler delegates task execution to an actor via:
    `actor.act(task_description, entrypoint=<function_id>, persist=False)`.

    When an `entrypoint` is provided, CodeActActor resolves the function by id,
    injects it into the sandbox namespace, and executes it in an asyncio task.
    """

    def __init__(
        self,
        *,
        entrypoint_id: int,
        execution_task: asyncio.Task[Any],
        on_finally: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        self._entrypoint_id = int(entrypoint_id)
        self._execution_task = execution_task
        self._completion_event = asyncio.Event()
        self._result_str: Optional[str] = None
        self._stopped = False
        self._on_finally = on_finally

        asyncio.create_task(self._monitor_execution())

    async def _monitor_execution(self) -> None:
        try:
            out = await self._execution_task
            if not self._stopped:
                self._result_str = str(out) if out is not None else ""
        except asyncio.CancelledError:
            self._stopped = True
            self._result_str = f"Entrypoint {self._entrypoint_id} was cancelled."
        except Exception as e:
            self._result_str = f"Error: {e}"
        finally:
            if self._on_finally is not None:
                try:
                    await self._on_finally()
                except Exception:
                    pass
            self._completion_event.set()

    async def ask(
        self,
        question: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
        images: list | dict | None = None,
    ) -> SteerableToolHandle:
        status = "completed" if self.done() else "still running"
        client = new_llm_client()
        client.set_system_message(
            "You are an AI assistant answering a status question about an in-flight entrypoint execution. "
            "Be brief and factual."
        )
        msg = (
            f"Entrypoint {self._entrypoint_id} status: {status}.\n\n"
            f"User question: {question}"
        )
        return start_async_tool_loop(
            client=client,
            message=msg,
            tools={},
            loop_id=f"EntrypointQuestion({self._entrypoint_id})",
            max_consecutive_failures=1,
            timeout=30,
        )

    async def interject(
        self,
        message: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
        images: list | dict | None = None,
    ) -> Optional[str]:
        # No-op for non-LLM entrypoint execution.
        return None

    async def stop(
        self,
        reason: Optional[str] = None,
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ) -> Optional[str]:
        if self._completion_event.is_set():
            return self._result_str
        self._stopped = True
        self._execution_task.cancel()
        try:
            await asyncio.wait_for(self._completion_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
        return (
            f"Entrypoint {self._entrypoint_id} stopped."
            if not reason
            else f"Entrypoint {self._entrypoint_id} stopped: {reason}"
        )

    async def pause(self) -> Optional[str]:
        return None

    async def resume(self) -> Optional[str]:
        return None

    def done(self) -> bool:
        return self._completion_event.is_set()

    async def result(self) -> str:
        await self._completion_event.wait()
        return self._result_str or ""

    async def next_clarification(self) -> dict:
        await asyncio.Event().wait()
        return {}

    async def next_notification(self) -> dict:
        await asyncio.Event().wait()
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        return None


class CodeExecutionSandbox:
    """
    A stateful execution environment for running Python code asynchronously.

    This class maintains a persistent global state across multiple executions,
    capturing stdout, stderr, return values, and exceptions in a structured format.

    It can optionally use pools for persistent subprocess connections (VenvPool
    for Python venvs, ShellPool for shell sessions), enabling state to be
    preserved across multiple function calls.
    """

    def __init__(
        self,
        computer_primitives: Optional[ComputerPrimitives] = None,
        environments: Optional[Dict[str, "BaseEnvironment"]] = None,
        venv_pool: Optional[Any] = None,
        shell_pool: Optional[Any] = None,
    ):
        """
        Initializes the execution environment.

        Args:
            computer_primitives: An instance of ComputerPrimitives to be injected into the
                             global state, making browser tools available.
            environments: Optional mapping of environment namespaces to environments. If
                provided, each environment instance is injected into globals.
            venv_pool: Optional VenvPool for persistent Python venv connections.
                If provided, venv-backed functions will use persistent connections
                that maintain state across calls.
            shell_pool: Optional ShellPool for persistent shell session connections.
                If provided, shell functions will use persistent sessions.
        """
        from unity.function_manager.execution_env import create_execution_globals

        self.id: str = str(uuid.uuid4())
        self.global_state: Dict[str, Any] = create_execution_globals()
        self._browser_used: bool = False

        # Expose sandbox metadata to user code (best-effort; callers may ignore).
        self.global_state["__sandbox_id__"] = self.id
        # Notification queue is injected per-call by CodeActActor via:
        # sandbox.global_state["__notification_up_q__"] = <asyncio.Queue>
        #
        # Provide a user-driven progress helper:
        #   notify({"type": "...", ...})
        # This helper is intentionally synchronous; it uses put_nowait.
        def notify(payload: dict) -> None:
            try:
                q = self.global_state.get("__notification_up_q__")
                if q is None:
                    return
                # Queue is expected to be an asyncio.Queue[dict]
                q.put_nowait(payload)
            except Exception:
                return

        self.global_state["notify"] = notify

        # Inject pools into namespace (for function proxies to use)
        if venv_pool is not None:
            self.global_state["__venv_pool__"] = venv_pool
        if shell_pool is not None:
            self.global_state["__shell_pool__"] = shell_pool

        class _UsageTrackingProxy:
            def __init__(self, target: Any, on_use: Callable[[], None]):
                self._target = target
                self._on_use = on_use

            def __getattr__(self, name: str) -> Any:
                # Treat any access as potential "use" since callers may invoke nested objects
                # like `computer_primitives.computer.get_screenshot()`.
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

    async def close(self) -> None:
        """
        Best-effort cleanup for an ephemeral sandbox instance.

        Notes
        -----
        - Pools (venv/shell) are owned by the actor and are not closed here.
        - This method is safe to call multiple times.
        """
        try:
            self.global_state.clear()
        except Exception as e:
            try:
                logger.warning(f"CodeExecutionSandbox.close() failed: {e}", exc_info=True)
            except Exception:
                pass

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
        builtins_dict: Any = None
        original_print: Any = None
        print_patched = False

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

            # Robust stdout capture for agent code:
            #
            # In practice, some nested tool loops may temporarily replace `sys.stdout`
            # (e.g. for their own logging/capture), which can bypass `redirect_stdout`
            # and cause `print(...)` output to leak to the outer process instead of
            # appearing in the tool result.
            #
            # To keep CodeAct reliable, we patch the sandbox built-in `print` so that
            # it always writes to our per-execution `stdout_capture`, regardless of
            # any temporary `sys.stdout` replacements inside awaited tool calls.
            try:
                builtins_dict = self.global_state.get("__builtins__")
                if (
                    isinstance(builtins_dict, dict)
                    and builtins_dict.get("print") is not None
                ):
                    original_print = builtins_dict.get("print")

                    def _captured_print(
                        *args: Any,
                        sep: str = " ",
                        end: str = "\n",
                        file: Any = None,
                        flush: bool = False,
                    ) -> None:
                        text = sep.join(str(a) for a in args) + end
                        if file is None:
                            stdout_capture.write(text)
                            if flush:
                                try:
                                    stdout_capture.flush()
                                except Exception:
                                    pass
                            return

                        # Respect explicit `file=` when possible, but fall back to
                        # stdout_capture to avoid losing output.
                        try:
                            file.write(text)
                            if flush:
                                try:
                                    file.flush()
                                except Exception:
                                    pass
                        except Exception:
                            stdout_capture.write(text)
                            if flush:
                                try:
                                    stdout_capture.flush()
                                except Exception:
                                    pass

                    builtins_dict["print"] = _captured_print
                    print_patched = True
            except Exception:
                # Best-effort: if patching fails, keep executing.
                print_patched = False

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                result = await self.global_state["__exec_wrapper"]()

        except Exception:
            error = traceback.format_exc()
        finally:
            if print_patched and isinstance(builtins_dict, dict):
                try:
                    builtins_dict["print"] = original_print
                except Exception:
                    pass
            if "__exec_wrapper" in self.global_state:
                del self.global_state["__exec_wrapper"]

        return {
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "result": result,
            "error": error,
            "browser_used": self._browser_used,
        }


class CodeActActor(BaseCodeActActor):
    """
    An actor that uses a conversational tool loop and a stateful code execution
    sandbox to accomplish tasks. It acts as a baseline for code-centric agents.
    """

    def __init__(
        self,
        session_connect_url: Optional[str] = None,
        headless: bool = False,
        computer_mode: str = "magnitude",
        timeout: float = 1000,
        agent_mode: str = "browser",
        agent_server_url: str = "http://localhost:3000",
        computer_primitives: Optional["ComputerPrimitives"] = None,
        environments: Optional[list["BaseEnvironment"]] = None,
        function_manager: Optional["FunctionManager"] = None,
        can_compose: bool = True,
        can_store: bool = True,
    ):
        """
        Initializes the CodeActActor.

        Args:
            computer_primitives: Optional existing ComputerPrimitives instance to reuse.
                           If provided, other browser-related params are ignored.
            environments: Optional list of execution environments. If None, defaults to
                [ComputerEnvironment, StateManagerEnvironment].
            function_manager: Manages a library of reusable functions. Exposes read-only tools
                (list_functions, search_functions, filter_functions) to the LLM.
                The LLM can call these tools to discover and retrieve reusable function implementations.
        """
        super().__init__(
            environments=environments,
            computer_primitives=computer_primitives,
            function_manager=function_manager,
            session_connect_url=session_connect_url,
            headless=headless,
            computer_mode=computer_mode,
            agent_mode=agent_mode,
            agent_server_url=agent_server_url,
        )

        # Create persistent pools that survive across act() calls
        from unity.function_manager.function_manager import VenvPool
        from unity.function_manager.shell_pool import ShellPool

        self._venv_pool = VenvPool()
        self._shell_pool = ShellPool()

        self._timeout = timeout
        self.can_compose: bool = bool(can_compose)
        self.can_store: bool = bool(can_store)
        self._browser_tools = self._get_browser_tools()
        self._tools = self._build_tools()

        self._main_event_loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            self._main_event_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        # Concurrency guard: limit active sandboxes per actor instance.
        self._act_semaphore = asyncio.Semaphore(20)
        # Timeout used when acquiring the semaphore (prevents unbounded waits).
        self._act_semaphore_timeout_s: float = 30.0

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

            try:
                sandbox = _CURRENT_SANDBOX.get()
            except Exception as e:
                err = f"CodeAct sandbox is not bound for this call: {type(e).__name__}"
                try:
                    logger.warning(err, exc_info=True)
                except Exception:
                    pass
                return {
                    "stdout": "",
                    "stderr": "",
                    "result": None,
                    "error": err,
                    "browser_used": False,
                }
            notification_q = sandbox.global_state.get("__notification_up_q__")
            if notification_q is not None:
                try:
                    await notification_q.put(
                        {
                            "type": "execution_started",
                            "sandbox_id": sandbox.id,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                except Exception:
                    pass

            try:
                execution_result = await sandbox.execute(code)
            except Exception as e:
                tb = traceback.format_exc()
                try:
                    logger.info(
                        f"Sandbox execution error: {type(e).__name__}",
                        exc_info=True,
                    )
                except Exception:
                    pass
                if notification_q is not None:
                    try:
                        await notification_q.put(
                            {
                                "type": "execution_error",
                                "sandbox_id": sandbox.id,
                                "error_kind": "exception",
                                "traceback_preview": tb[:2000],
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                    except Exception:
                        pass
                return {
                    "stdout": "",
                    "stderr": "",
                    "result": None,
                    "error": tb,
                    "browser_used": False,
                }

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
                    url = await self._computer_primitives.computer.get_current_url()
                    screenshot_b64 = (
                        await self._computer_primitives.computer.get_screenshot()
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

            if notification_q is not None:
                try:
                    await notification_q.put(
                        {
                            "type": "execution_finished",
                            "sandbox_id": sandbox.id,
                            "status": "ok" if not execution_result.get("error") else "error",
                            "stdout_len": len(execution_result.get("stdout") or ""),
                            "stderr_len": len(execution_result.get("stderr") or ""),
                            "browser_used": bool(execution_result.get("browser_used")),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                except Exception:
                    pass

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

            async def FunctionManager_search_functions(
                query: str,
                n: int = 5,
            ) -> Any:
                """
                Search for functions by semantic similarity to a natural-language query.

                Functions are automatically injected into your Python sandbox namespace,
                so you can execute them immediately after searching.
                """
                result = self.function_manager.search_functions(
                    query=query,
                    n=n,
                    return_callable=True,
                    namespace=_CURRENT_SANDBOX.get().global_state,
                    also_return_metadata=True,
                )
                return result["metadata"]

            async def FunctionManager_filter_functions(
                filter: Optional[str] = None,
                offset: int = 0,
                limit: int = 100,
            ) -> Any:
                """
                Filter functions using a Python-like filter expression.

                Functions are automatically injected into your Python sandbox namespace,
                so you can execute them immediately after filtering.
                """
                result = self.function_manager.filter_functions(
                    filter=filter,
                    offset=offset,
                    limit=limit,
                    return_callable=True,
                    namespace=_CURRENT_SANDBOX.get().global_state,
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
                    namespace=_CURRENT_SANDBOX.get().global_state,
                    also_return_metadata=True,
                )
                return result["metadata"]

            tools["FunctionManager_search_functions"] = FunctionManager_search_functions
            tools["FunctionManager_filter_functions"] = FunctionManager_filter_functions
            tools["FunctionManager_list_functions"] = FunctionManager_list_functions

            async def FunctionManager_add_functions(
                implementations: str | list[str],
                *,
                language: str = "python",
                overwrite: bool = False,
                verify: Optional[dict[str, bool]] = None,
                preconditions: Optional[dict[str, dict]] = None,
            ) -> Any:
                """
                Add/store new functions into the FunctionManager.

                Notes
                -----
                - This tool is gated by CodeActActor's `can_store` flag (and can be disabled per-call).
                - Prefer using existing functions (search first) before adding new ones.
                """
                fm = self.function_manager
                if fm is None:
                    raise RuntimeError("FunctionManager is not configured on this actor.")
                return fm.add_functions(
                    implementations=implementations,
                    language=language,  # type: ignore[arg-type]
                    overwrite=bool(overwrite),
                    verify=(verify or {}),
                    preconditions=(preconditions or {}),
                )

            tools["FunctionManager_add_functions"] = FunctionManager_add_functions

            async def inspect_state() -> dict:
                """
                Inspect persistent state across all execution contexts.

                Use this tool to understand what variables exist before deciding
                how to call a function (stateful vs stateless vs read_only).

                Returns:
                    Dict with keys:
                    - contexts: Dict mapping context names to their variables
                      - "default": The main Python execution environment
                      - "venv_{id}": Python venv contexts (session 0 implicit)
                      - "venv_{id}_session_{n}": Additional venv sessions (n > 0)
                      - "{shell}_0": Shell contexts (bash, zsh, sh, powershell)
                    - summary: Human-readable overview of active contexts

                Notes:
                    - Each context maintains independent state across function calls.
                    - Use this to decide: should the next function call be stateful
                      (extend state), stateless (fresh), or read_only (preview)?
                """
                contexts: dict[str, dict] = {}
                summary_parts: list[str] = []

                # Default context (in-process Python execution environment)
                sandbox = _CURRENT_SANDBOX.get()
                default_state = {}
                for name, value in sandbox.global_state.items():
                    # Skip internal names and infrastructure
                    if name.startswith("_"):
                        continue
                    if name in (
                        "asyncio",
                        "typing",
                        "pydantic",
                        "json",
                        "re",
                        "os",
                        "sys",
                        "math",
                        "datetime",
                        "collections",
                        "itertools",
                        "functools",
                        "pathlib",
                        "primitives",
                        "computer_primitives",
                    ):
                        continue
                    # Skip modules, classes, and functions (show only data)
                    if isinstance(value, type) or callable(value):
                        # But include function proxies - show their names
                        if hasattr(value, "__name__") and not isinstance(value, type):
                            default_state[name] = f"<callable: {value.__name__}>"
                        continue
                    # Try to represent the value
                    try:
                        repr_val = repr(value)
                        if len(repr_val) > 500:
                            repr_val = repr_val[:500] + "..."
                        default_state[name] = repr_val
                    except Exception:
                        default_state[name] = f"<{type(value).__name__}>"

                contexts["default"] = default_state
                if default_state:
                    summary_parts.append(f"default: {len(default_state)} variables")

                # Python venv contexts
                if self._venv_pool is not None:
                    active_venv_sessions = self._venv_pool.list_active_sessions()

                    if active_venv_sessions:
                        all_venv_states = await self._venv_pool.get_all_states(
                            function_manager=self.function_manager,
                            timeout=10.0,
                        )
                        for (venv_id, session_id), state in all_venv_states.items():
                            # Use simplified key: venv_{id} for session 0, venv_{id}_session_{n} for others
                            if session_id == 0:
                                key = f"venv_{venv_id}"
                            else:
                                key = f"venv_{venv_id}_session_{session_id}"

                            # Filter state
                            filtered_state = {}
                            for name, value in state.items():
                                if name.startswith("_"):
                                    continue
                                try:
                                    repr_val = (
                                        repr(value)
                                        if not isinstance(value, str)
                                        else value
                                    )
                                    if len(str(repr_val)) > 500:
                                        repr_val = str(repr_val)[:500] + "..."
                                    filtered_state[name] = repr_val
                                except Exception:
                                    filtered_state[name] = "<unserializable>"
                            contexts[key] = filtered_state
                            if filtered_state:
                                summary_parts.append(
                                    f"{key}: {len(filtered_state)} variables",
                                )

                # Shell contexts
                if self._shell_pool is not None:
                    active_shell_sessions = self._shell_pool.get_active_sessions()

                    for language, session_id in active_shell_sessions:
                        # Use simplified key: {lang}_0 for session 0, {lang}_{n} for others
                        key = f"{language}_{session_id}"
                        # Shell state is not easily inspectable like Python state
                        # Just indicate the context exists
                        contexts[key] = {
                            "_note": "Shell session active (state not inspectable)",
                        }
                        summary_parts.append(f"{key}: active")

                return {
                    "contexts": contexts,
                    "summary": (
                        f"{len(contexts)} contexts"
                        + (f" ({', '.join(summary_parts)})" if summary_parts else "")
                        if contexts
                        else "No active contexts"
                    ),
                }

            tools["inspect_state"] = inspect_state

        return tools

    @functools.wraps(BaseCodeActActor.act, updated=())
    @log_manager_call("CodeActActor", "act", payload_key="description")
    async def act(
        self,
        description: str,
        *,
        clarification_enabled: bool = True,
        response_format: Optional[Type[BaseModel]] = None,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        _notification_up_q: Optional[asyncio.Queue[dict]] = None,
        _call_id: Optional[str] = None,
        images: Optional[ImageRefs | list[RawImageRef | AnnotatedImageRef]] = None,
        entrypoint: Optional[int] = None,
        entrypoint_args: Optional[list[Any]] = None,
        entrypoint_kwargs: Optional[dict[str, Any]] = None,
        can_compose: Optional[bool] = None,
        can_store: Optional[bool] = None,
        **kwargs,
    ) -> SteerableToolHandle:
        if not self._main_event_loop:
            self._main_event_loop = asyncio.get_running_loop()

        effective_can_compose = self.can_compose if can_compose is None else bool(can_compose)
        effective_can_store = self.can_store if can_store is None else bool(can_store)

        # can_compose=False mode: do not run an LLM tool loop or allow arbitrary code execution.
        # Instead, semantic-search for a stored function and execute it directly.
        if entrypoint is None and not effective_can_compose:
            from unity.actor.single_function_actor import SingleFunctionActorHandle

            fm = self.function_manager
            if fm is None:
                raise RuntimeError(
                    "CodeActActor cannot run with can_compose=False: function_manager is None",
                )

            matches = fm.search_functions(
                query=str(description or ""),
                n=1,
                include_implementations=True,
            )
            if not matches:

                async def _fail() -> Any:
                    raise RuntimeError(
                        "can_compose=False: no matching functions found via semantic search.",
                    )

                return SingleFunctionActorHandle(
                    function_name="(no_match)",
                    function_id=None,
                    execution_task=asyncio.create_task(_fail()),
                    is_primitive=False,
                    verify=False,
                    goal=description,
                )

            fn_name = matches[0].get("name")
            if not isinstance(fn_name, str) or not fn_name.strip():
                raise RuntimeError(
                    "can_compose=False: semantic search returned a function without a valid name.",
                )

            primitives = None
            try:
                env = self.environments.get("primitives")
                if env is not None:
                    primitives = env.get_instance()
            except Exception:
                primitives = None

            async def _run_found() -> Any:
                out = await fm.execute_function(
                    function_name=fn_name,
                    primitives=primitives,
                    computer_primitives=self._computer_primitives,
                    venv_pool=self._venv_pool,
                    shell_pool=self._shell_pool,
                    state_mode="stateless",
                )
                if isinstance(out, dict) and out.get("error"):
                    raise RuntimeError(str(out.get("error")))
                if isinstance(out, dict):
                    return out.get("result")
                return out

            return SingleFunctionActorHandle(
                function_name=fn_name,
                function_id=matches[0].get("function_id") if isinstance(matches[0], dict) else None,
                execution_task=asyncio.create_task(_run_found()),
                is_primitive=False,
                verify=False,
                goal=description,
                docstring=(matches[0].get("docstring") if isinstance(matches[0], dict) else None),
            )

        
        initial_prompt = (
            "This is an interactive session. Acknowledge that you are ready and "
            "wait for the user to provide instructions via interjection."
        )

        # Clarification queues:
        # - When enabled, we ensure the handle has queues (either provided by caller or newly created).
        # - When disabled, we do not provide queues and we do not wire queue injection into environments.
        clarification_up_q: Optional[asyncio.Queue[str]]
        clarification_down_q: Optional[asyncio.Queue[str]]
        if clarification_enabled:
            clarification_up_q = _clarification_up_q or asyncio.Queue()
            clarification_down_q = _clarification_down_q or asyncio.Queue()
        else:
            clarification_up_q = None
            clarification_down_q = None

        # Create per-call environments so clarification queues are not stored on shared actor environments.
        sandbox_envs: Dict[str, "BaseEnvironment"] = {}
        try:
            from unity.actor.environments import (
                ComputerEnvironment as _ComputerEnvironment,
                StateManagerEnvironment as _StateManagerEnvironment,
            )
        except Exception:
            _ComputerEnvironment = None  # type: ignore
            _StateManagerEnvironment = None  # type: ignore

        for ns, env in self.environments.items():
            # Prefer explicit reconstruction for known env types.
            try:
                if _ComputerEnvironment is not None and isinstance(env, _ComputerEnvironment):
                    sandbox_envs[ns] = _ComputerEnvironment(
                        env.get_instance(),
                        clarification_up_q=clarification_up_q,
                        clarification_down_q=clarification_down_q,
                    )
                    continue
                if _StateManagerEnvironment is not None and isinstance(env, _StateManagerEnvironment):
                    sandbox_envs[ns] = _StateManagerEnvironment(
                        env.get_instance(),
                        exposed_managers=getattr(env, "_exposed_managers", None),
                        clarification_up_q=clarification_up_q,
                        clarification_down_q=clarification_down_q,
                    )
                    continue
            except Exception:
                pass

            # Fallback: shallow-copy and set private queue attrs on the copy only.
            try:
                env_copy = copy.copy(env)
                if hasattr(env_copy, "_clarification_up_q"):
                    setattr(env_copy, "_clarification_up_q", clarification_up_q)
                if hasattr(env_copy, "_clarification_down_q"):
                    setattr(env_copy, "_clarification_down_q", clarification_down_q)
                sandbox_envs[ns] = env_copy
            except Exception:
                sandbox_envs[ns] = env

        # Concurrency/backpressure guard. If we can't acquire within 30s, treat as resource exhaustion.
        try:
            await asyncio.wait_for(
                self._act_semaphore.acquire(),
                timeout=float(getattr(self, "_act_semaphore_timeout_s", 30.0)),
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                "CodeActActor is at capacity (too many concurrent sessions). "
                "Try again later or reduce concurrency.",
            )
        sandbox = CodeExecutionSandbox(
            computer_primitives=self._computer_primitives,
            environments=sandbox_envs,
            venv_pool=self._venv_pool,
            shell_pool=self._shell_pool,
        )
        if _notification_up_q is not None:
            sandbox.global_state["__notification_up_q__"] = _notification_up_q
        token = _CURRENT_SANDBOX.set(sandbox)

        async def _cleanup() -> None:
            try:
                # Best-effort cleanup; CodeExecutionSandbox will grow a close() in a later ticket.
                if hasattr(sandbox, "close") and callable(getattr(sandbox, "close")):
                    await sandbox.close()  # type: ignore[misc]
            except Exception:
                pass
            try:
                _CURRENT_SANDBOX.reset(token)
            except Exception:
                pass
            try:
                self._act_semaphore.release()
            except Exception:
                pass

        # If an explicit FunctionManager entrypoint is provided (e.g., TaskScheduler task execution),
        # bypass the CodeAct LLM loop and run the function directly.
        if entrypoint is not None:
            entrypoint_id = int(entrypoint)
            args = list(entrypoint_args or [])
            kwargs_for_entrypoint = dict(entrypoint_kwargs or {})

            async def _run_entrypoint() -> Any:
                fm = self.function_manager
                if fm is None:
                    raise RuntimeError(
                        "CodeActActor cannot execute entrypoint: function_manager is None",
                    )

                out = fm.filter_functions(
                    filter=f"function_id == {entrypoint_id}",
                    return_callable=True,
                    namespace=sandbox.global_state,
                    also_return_metadata=True,
                )
                metadata = []
                if isinstance(out, dict):
                    metadata = list(out.get("metadata") or [])
                if not metadata:
                    raise ValueError(
                        f"Entrypoint function_id {entrypoint_id} not found in FunctionManager.",
                    )
                fn_name = metadata[0].get("name")
                if not isinstance(fn_name, str) or not fn_name.strip():
                    raise ValueError(
                        f"Entrypoint {entrypoint_id} has no valid function name.",
                    )
                fn = sandbox.global_state.get(fn_name)
                if fn is None:
                    raise ValueError(
                        f"Entrypoint {entrypoint_id} ({fn_name}) was not injected into the sandbox namespace.",
                    )

                res = fn(*args, **kwargs_for_entrypoint)
                if inspect.isawaitable(res):
                    res = await res
                return res

            entry_task = asyncio.create_task(_run_entrypoint())
            entry_handle = _CodeActEntrypointHandle(
                entrypoint_id=entrypoint_id,
                execution_task=entry_task,
                on_finally=_cleanup,
            )
            return entry_handle

        system_prompt = build_code_act_system_prompt(
            self.environments,
            tools=self._tools,
        )
        handle = ActorHandle(
            task_description=description or initial_prompt,
            tools=self._tools,
            parent_chat_context=_parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
            notification_up_q=_notification_up_q,
            call_id=_call_id,
            on_finally=_cleanup,
            main_event_loop=self._main_event_loop,
            timeout=self._timeout,
            persist=is_interactive_session,
            custom_system_prompt=system_prompt,
            tool_policy=None,
            computer_primitives=self._computer_primitives,
            images=images,
            response_format=response_format,
        )
        return handle

    async def close(self):
        """Shuts down the actor and its associated resources gracefully."""
        # Close the pools (terminates persistent subprocess/session connections)
        await self._venv_pool.close()
        await self._shell_pool.close()

        if self._computer_primitives:
            self._computer_primitives.computer.stop()
