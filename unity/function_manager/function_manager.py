import ast
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import inspect
import functools
import json
import os
import re
import signal
import socket
import sys
import tempfile
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, Union
import unify
from .shell_pool import ShellPool
from unify.utils.http import RequestError as _UnifyRequestError
from ..common.log_utils import create_logs as unity_create_logs
from ..common.embed_utils import list_private_fields
from ..common.search_utils import table_search_top_k
from .execution_env import create_base_globals
from .dependency_analysis import collect_dependencies_from_function_node
from .types.function import Function
from .types.meta import FunctionsMeta
from .types.venv import VirtualEnv
from .base import BaseFunctionManager
from ..common.model_to_fields import model_to_fields
from ..file_manager.managers.local import LocalFileManager
from ..image_manager.image_manager import ImageHandle
from ..manager_registry import ManagerRegistry
from ..common.filter_utils import normalize_filter_expr
from ..common.context_registry import ContextRegistry, TableContext
from .primitives import collect_primitives, compute_primitives_hash
from .custom_functions import (
    collect_custom_functions,
    compute_custom_functions_hash,
    collect_custom_venvs,
    compute_custom_venvs_hash,
)


logger = logging.getLogger(__name__)


class _DependencyVisitor(ast.NodeVisitor):
    """
    Statefully analyzes function AST to find direct calls and indirect calls
    via variables assigned function names, specifically looking for names
    known to the FunctionManager.
    """

    def __init__(self, known_function_names: Set[str]):
        self.known_function_names = known_function_names
        self.dependencies: Set[str] = set()
        self._assignment_map: Dict[str, str] = {}

    def visit_Assign(self, node: ast.Assign):
        # Only track simple assignments: target_var = potential_func_name
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_var = node.targets[0].id
            if isinstance(node.value, ast.Name):
                assigned_name = node.value.id
                # Check if the assigned name is one of the functions we manage
                if assigned_name in self.known_function_names:
                    # Record the mapping for the current scope
                    self._assignment_map[target_var] = assigned_name
                # If variable is assigned something else, remove mapping
                elif target_var in self._assignment_map:
                    del self._assignment_map[target_var]
            # If variable is assigned non-Name, remove mapping
            elif target_var in self._assignment_map:
                del self._assignment_map[target_var]

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func_node = node.func
        called_name: Optional[str] = None

        # Case 1: Direct call -> func_name()
        if isinstance(func_node, ast.Name):
            func_name = func_node.id
            # Check if it's a direct call to a known library function
            if func_name in self.known_function_names:
                called_name = func_name
            # Check if it's an indirect call via a mapped variable -> var()
            elif func_name in self._assignment_map:
                called_name = self._assignment_map[func_name]

        # Case 2: Method call -> obj.method() - generally ignore for dependency injection
        # (We assume obj like computer_primitives is globally available)

        if called_name:
            self.dependencies.add(called_name)

        self.generic_visit(node)  # Continue traversal

    def visit_Return(self, node: ast.Return):
        # Case 3: Return statement -> return func_name or return var
        if isinstance(node.value, ast.Name):
            returned_name = node.value.id
            # Check if returning a known function name directly
            if returned_name in self.known_function_names:
                self.dependencies.add(returned_name)
            # Also check if returning a variable that was assigned a function
            elif returned_name in self._assignment_map:
                self.dependencies.add(self._assignment_map[returned_name])
        self.generic_visit(node)


def _strip_custom_function_decorators(source: str) -> str:
    """
    Remove @custom_function decorators from a function source string.

    The @custom_function decorator is used for sync metadata only (it is effectively
    a no-op at runtime), but the symbol is not guaranteed to exist inside execution
    environments (e.g., Actor sandboxes or venv runner subprocesses).
    """
    try:
        lines = source.splitlines(keepends=True)
    except Exception:
        return source

    out: List[str] = []
    seen_def = False
    for line in lines:
        stripped = line.lstrip()
        if not seen_def and stripped.startswith("@custom_function"):
            continue
        if stripped.startswith("def ") or stripped.startswith("async def "):
            seen_def = True
        out.append(line)
    return "".join(out)


# Pattern for shell script metadata comments
_SHELL_NAME_PATTERN = re.compile(r"^#\s*@name:\s*(.+?)\s*$", re.MULTILINE)
_SHELL_ARGS_PATTERN = re.compile(r"^#\s*@args:\s*(.+?)\s*$", re.MULTILINE)
_SHELL_DESC_PATTERN = re.compile(r"^#\s*@description:\s*(.+?)\s*$", re.MULTILINE)


def _parse_shell_script_metadata(source: str) -> Dict[str, Optional[str]]:
    """
    Parse metadata from shell script comments.

    Expected format at the top of the script::

        #!/bin/sh
        # @name: my_function
        # @args: (input_file output_file --verbose)
        # @description: Brief description of what the function does

    Returns:
        Dict with keys: name, argspec, docstring (any may be None if not found)
    """
    name_match = _SHELL_NAME_PATTERN.search(source)
    args_match = _SHELL_ARGS_PATTERN.search(source)
    desc_match = _SHELL_DESC_PATTERN.search(source)

    return {
        "name": name_match.group(1).strip() if name_match else None,
        "argspec": args_match.group(1).strip() if args_match else "()",
        "docstring": desc_match.group(1).strip() if desc_match else "",
    }


class _VenvConnection:
    """
    Manages a persistent connection to a venv subprocess in server mode.

    The subprocess maintains state across calls, enabling variables to persist
    between function executions within the same venv.
    """

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        venv_id: int,
        function_manager: "FunctionManager",
    ):
        self._process = process
        self._venv_id = venv_id
        self._function_manager = function_manager
        self._lock = asyncio.Lock()  # Serialize calls to same venv
        self._closed = False
        self._tainted = False  # Set to True after timeout or other corruption

    @classmethod
    async def create(
        cls,
        venv_id: int,
        function_manager: "FunctionManager",
        timeout: float = 30.0,
    ) -> "_VenvConnection":
        """
        Create a new persistent venv connection.

        Args:
            venv_id: The virtual environment to connect to.
            function_manager: The FunctionManager instance for venv preparation.
            timeout: Timeout for subprocess startup.

        Returns:
            A new _VenvConnection instance.

        Raises:
            RuntimeError: If the subprocess fails to start or send ready signal.
        """
        python_path = await function_manager.prepare_venv(venv_id=venv_id)
        runner_path = function_manager._get_venv_runner_path(venv_id)

        use_process_group = sys.platform != "win32"
        process = await asyncio.create_subprocess_exec(
            str(python_path),
            str(runner_path),
            "--server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=use_process_group,
        )

        conn = cls(process, venv_id, function_manager)

        # Wait for ready signal from subprocess
        try:
            ready_msg = await asyncio.wait_for(
                conn._read_message(),
                timeout=timeout,
            )
            if ready_msg.get("type") != "ready":
                raise RuntimeError(
                    f"Venv {venv_id} subprocess sent unexpected message: {ready_msg}",
                )
        except asyncio.TimeoutError:
            await conn.shutdown()
            raise RuntimeError(
                f"Venv {venv_id} subprocess did not send ready signal within {timeout}s",
            )
        except Exception as e:
            await conn.shutdown()
            raise RuntimeError(
                f"Venv {venv_id} subprocess failed to start: {e}",
            ) from e

        return conn

    async def _read_message(self) -> dict:
        """Read a JSON message from the subprocess stdout."""
        if self._process.stdout is None:
            raise RuntimeError("Subprocess stdout is None")
        line = await self._process.stdout.readline()
        if not line:
            raise EOFError("Subprocess stdout closed")
        return json.loads(line.decode().strip())

    async def _write_message(self, msg: dict) -> None:
        """Write a JSON message to the subprocess stdin."""
        if self._process.stdin is None:
            raise RuntimeError("Subprocess stdin is None")
        data = json.dumps(msg) + "\n"
        self._process.stdin.write(data.encode())
        await self._process.stdin.drain()

    def is_alive(self) -> bool:
        """Check if the subprocess is still running and usable."""
        return (
            not self._closed and not self._tainted and self._process.returncode is None
        )

    async def execute(
        self,
        implementation: str,
        call_kwargs: dict,
        is_async: bool,
        primitives: Optional[Any] = None,
        computer_primitives: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """
        Execute a function in the persistent venv subprocess.

        Args:
            implementation: The function source code.
            call_kwargs: Keyword arguments to pass to the function.
            is_async: Whether the function is async.
            primitives: The Primitives instance for RPC access.
            computer_primitives: The ComputerPrimitives instance for RPC access.
            timeout: Execution timeout in seconds (None for no timeout).

        Returns:
            Dict with keys: result, error, stdout, stderr

        Raises:
            RuntimeError: If the subprocess has died or execution fails.
            asyncio.TimeoutError: If execution exceeds timeout.
        """
        async with self._lock:
            if not self.is_alive():
                raise RuntimeError(
                    f"Venv {self._venv_id} subprocess has died (returncode={self._process.returncode})",
                )

            # Send execute request
            await self._write_message(
                {
                    "type": "execute",
                    "implementation": implementation,
                    "call_kwargs": call_kwargs,
                    "is_async": is_async,
                },
            )

            # Handle bidirectional RPC until we get a complete message
            async def handle_rpc_loop() -> dict:
                while True:
                    msg = await self._read_message()
                    msg_type = msg.get("type")

                    if msg_type == "complete":
                        return msg

                    if msg_type == "rpc_call":
                        # Handle RPC call from subprocess
                        rpc_result = await self._handle_rpc_call(
                            msg,
                            primitives=primitives,
                            computer_primitives=computer_primitives,
                        )
                        await self._write_message(rpc_result)
                    else:
                        logger.warning(
                            f"Venv {self._venv_id}: unexpected message type '{msg_type}'",
                        )

            if timeout is not None:
                try:
                    return await asyncio.wait_for(handle_rpc_loop(), timeout=timeout)
                except asyncio.TimeoutError:
                    # After a timeout, the subprocess is in an unknown state.
                    # Mark it as tainted so the pool recreates it on next use.
                    self._tainted = True
                    raise
            return await handle_rpc_loop()

    async def _handle_rpc_call(
        self,
        msg: dict,
        primitives: Optional[Any],
        computer_primitives: Optional[Any],
    ) -> dict:
        """Handle an RPC call from the subprocess."""
        request_id = msg.get("id")
        path = msg.get("path", "")
        kwargs = msg.get("kwargs", {})

        try:
            parts = path.split(".")
            if len(parts) == 2:
                namespace, method = parts
                if namespace == "computer" and computer_primitives is not None:
                    fn = getattr(computer_primitives, method, None)
                elif primitives is not None:
                    manager = getattr(primitives, namespace, None)
                    fn = getattr(manager, method, None) if manager else None
                else:
                    fn = None

                if fn is not None and callable(fn):
                    result = fn(**kwargs)
                    if asyncio.iscoroutine(result):
                        result = await result
                    return {"type": "rpc_result", "id": request_id, "result": result}

            return {
                "type": "rpc_error",
                "id": request_id,
                "error": f"Unknown RPC path: {path}",
            }
        except Exception as e:
            return {"type": "rpc_error", "id": request_id, "error": str(e)}

    async def get_state(self, timeout: float = 30.0) -> Dict[str, Any]:
        """
        Get serialized user-defined state from the persistent subprocess.

        This is used for read_only mode to capture the current state before
        executing in an ephemeral subprocess.

        Args:
            timeout: Timeout for state retrieval.

        Returns:
            Dict of serialized state variables.

        Raises:
            RuntimeError: If the subprocess has died or retrieval fails.
            asyncio.TimeoutError: If retrieval exceeds timeout.
        """
        async with self._lock:
            if not self.is_alive():
                raise RuntimeError(
                    f"Venv {self._venv_id} subprocess has died (returncode={self._process.returncode})",
                )

            await self._write_message({"type": "get_state"})

            async def wait_for_state() -> Dict[str, Any]:
                while True:
                    msg = await self._read_message()
                    if msg.get("type") == "state":
                        return msg.get("state", {})
                    # Ignore other message types while waiting

            if timeout is not None:
                return await asyncio.wait_for(wait_for_state(), timeout=timeout)
            return await wait_for_state()

    async def shutdown(self, timeout: float = 5.0) -> None:
        """
        Gracefully shut down the subprocess.

        Args:
            timeout: Timeout for graceful shutdown before force-killing.
        """
        if self._closed:
            return
        self._closed = True

        if self._process.returncode is not None:
            return

        try:
            # Try graceful shutdown
            await self._write_message({"type": "shutdown"})
            await asyncio.wait_for(self._process.wait(), timeout=timeout)
        except (asyncio.TimeoutError, Exception):
            # Force kill if graceful shutdown fails
            try:
                if sys.platform != "win32":
                    # Kill entire process group
                    os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                else:
                    self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass


@dataclass
class SessionMetadata:
    venv_id: int
    session_id: int
    created_at: datetime
    last_used: datetime


class SessionLimitError(RuntimeError):
    def __init__(self, *, message: str):
        super().__init__(message)
        self.message = message

    def to_error_dict(self) -> dict:
        return {"error": self.message, "error_type": "resource_limit"}


class VenvPool:
    """
    Manages a pool of persistent venv subprocess connections.

    Each sandbox gets its own VenvPool, ensuring state isolation between
    different actors/sandboxes while preserving state across function calls
    within the same sandbox.

    Connections are keyed by (venv_id, session_id), allowing multiple independent
    stateful sessions per venv. Each session has its own subprocess and globals.
    """

    def __init__(self, *, max_total_sessions: int = 20) -> None:
        # Key: (venv_id, session_id) -> _VenvConnection
        self._connections: Dict[Tuple[int, int], _VenvConnection] = {}
        self._metadata: Dict[Tuple[int, int], SessionMetadata] = {}
        self._lock = asyncio.Lock()
        self._closed = False
        self._max_total_sessions = int(max_total_sessions)

    async def get_or_create_connection(
        self,
        venv_id: int,
        function_manager: "FunctionManager",
        session_id: int = 0,
        timeout: float = 30.0,
    ) -> _VenvConnection:
        """
        Get an existing connection or create a new one for the given venv/session.

        Args:
            venv_id: The virtual environment ID.
            function_manager: The FunctionManager for venv preparation.
            session_id: The session ID within the venv (default 0).
            timeout: Timeout for creating a new connection.

        Returns:
            A _VenvConnection instance.
        """
        key = (venv_id, session_id)
        async with self._lock:
            if self._closed:
                raise RuntimeError("VenvPool has been closed")

            if key in self._connections:
                conn = self._connections[key]
                if conn.is_alive():
                    md = self._metadata.get(key)
                    if md is not None:
                        md.last_used = datetime.now(timezone.utc)
                    return conn
                # Connection died, remove it and create a new one
                logger.warning(
                    f"VenvPool: connection for venv {venv_id} session {session_id} died, creating new one",
                )
                del self._connections[key]
                self._metadata.pop(key, None)

            # Enforce global session cap (across all venv_id/session_id combinations).
            active = sum(1 for c in self._connections.values() if c.is_alive())
            if active >= self._max_total_sessions:
                raise SessionLimitError(
                    message=f"Maximum sessions reached for python ({active}/{self._max_total_sessions})",
                )

            # Create new connection
            conn = await _VenvConnection.create(
                venv_id=venv_id,
                function_manager=function_manager,
                timeout=timeout,
            )
            self._connections[key] = conn
            now = datetime.now(timezone.utc)
            self._metadata[key] = SessionMetadata(
                venv_id=int(venv_id),
                session_id=int(session_id),
                created_at=now,
                last_used=now,
            )
            return conn

    async def execute_in_venv(
        self,
        *,
        venv_id: int,
        implementation: str,
        call_kwargs: dict,
        is_async: bool,
        session_id: int = 0,
        primitives: Optional[Any] = None,
        computer_primitives: Optional[Any] = None,
        function_manager: "FunctionManager",
        timeout: Optional[float] = None,
    ) -> dict:
        """
        Execute a function in a persistent venv subprocess.

        Args:
            venv_id: The virtual environment to use.
            implementation: The function source code.
            call_kwargs: Keyword arguments to pass to the function.
            is_async: Whether the function is async.
            session_id: The session ID within the venv (default 0).
            primitives: The Primitives instance for RPC access.
            computer_primitives: The ComputerPrimitives instance for RPC access.
            function_manager: The FunctionManager for venv preparation.
            timeout: Execution timeout in seconds.

        Returns:
            Dict with keys: result, error, stdout, stderr
        """
        key = (venv_id, session_id)
        try:
            conn = await self.get_or_create_connection(
                venv_id=venv_id,
                function_manager=function_manager,
                session_id=session_id,
            )
        except SessionLimitError as e:
            return {
                "result": None,
                "stdout": "",
                "stderr": "",
                "error": e.message,
                "error_type": "resource_limit",
            }

        try:
            out = await conn.execute(
                implementation=implementation,
                call_kwargs=call_kwargs,
                is_async=is_async,
                primitives=primitives,
                computer_primitives=computer_primitives,
                timeout=timeout,
            )
            # Update last_used best-effort
            md = self._metadata.get(key)
            if md is not None:
                md.last_used = datetime.now(timezone.utc)
            return out
        except RuntimeError as e:
            if "subprocess has died" in str(e):
                # Try to recreate and retry once
                logger.warning(
                    f"VenvPool: retrying after subprocess death for venv {venv_id} session {session_id}",
                )
                async with self._lock:
                    if key in self._connections:
                        del self._connections[key]

                conn = await self.get_or_create_connection(
                    venv_id=venv_id,
                    function_manager=function_manager,
                    session_id=session_id,
                )
                return await conn.execute(
                    implementation=implementation,
                    call_kwargs=call_kwargs,
                    is_async=is_async,
                    primitives=primitives,
                    computer_primitives=computer_primitives,
                    timeout=timeout,
                )
            raise

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Return list of all active python venv sessions with metadata."""
        out: List[Dict[str, Any]] = []
        for (venv_id, session_id), conn in list(self._connections.items()):
            if not conn.is_alive():
                continue
            md = self._metadata.get((venv_id, session_id))
            if md is None:
                now = datetime.now(timezone.utc)
                md = SessionMetadata(
                    venv_id=int(venv_id),
                    session_id=int(session_id),
                    created_at=now,
                    last_used=now,
                )
                self._metadata[(venv_id, session_id)] = md
            out.append(
                {
                    "language": "python",
                    "session_id": int(session_id),
                    "venv_id": int(venv_id),
                    "created_at": md.created_at.isoformat(),
                    "last_used": md.last_used.isoformat(),
                    "state_summary": "active",
                },
            )
        return out

    async def get_session_state(
        self,
        *,
        venv_id: int,
        session_id: int,
        function_manager: "FunctionManager",
        detail: str = "summary",
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Inspect state of a python venv-backed session.
        """
        key = (int(venv_id), int(session_id))
        if key not in self._connections or not self._connections[key].is_alive():
            return {
                "error": f"Python venv session {(int(venv_id), int(session_id))} not found",
                "error_type": "validation",
            }
        state = await self.get_connection_state(
            venv_id=int(venv_id),
            function_manager=function_manager,
            session_id=int(session_id),
            timeout=timeout,
        )

        def _is_secret_name(n: str) -> bool:
            nn = n.lower()
            return any(
                tok in nn
                for tok in ("token", "secret", "apikey", "api_key", "password", "key")
            )

        def _safe_repr(name: str, value: Any) -> str:
            if _is_secret_name(name):
                return "<redacted>"
            try:
                s = repr(value)
            except Exception:
                s = f"<{type(value).__name__}>"
            if len(s) > 500:
                s = s[:500] + "..."
            return s

        names = sorted(
            [k for k in state.keys() if isinstance(k, str) and not k.startswith("_")],
        )
        if detail in ("summary", "names"):
            return {
                "names": names,
                "count": len(names),
            }
        if detail == "full":
            return {name: _safe_repr(name, state.get(name)) for name in names}
        return {
            "error": f"Unsupported detail level: {detail!r}",
            "error_type": "validation",
        }

    async def close_session(self, *, venv_id: int, session_id: int) -> bool:
        """Close a specific venv session and free resources."""
        key = (int(venv_id), int(session_id))
        async with self._lock:
            conn = self._connections.get(key)
            if conn is None:
                return False
            try:
                await conn.shutdown()
            except Exception:
                pass
            self._connections.pop(key, None)
            self._metadata.pop(key, None)
            return True

    async def get_connection_state(
        self,
        venv_id: int,
        function_manager: "FunctionManager",
        session_id: int = 0,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Get serialized state from a venv connection.

        Used for read_only mode to snapshot current state before ephemeral execution.

        Args:
            venv_id: The virtual environment ID.
            function_manager: The FunctionManager for venv preparation.
            session_id: The session ID within the venv (default 0).
            timeout: Timeout for state retrieval.

        Returns:
            Dict of serialized state variables.
        """
        conn = await self.get_or_create_connection(
            venv_id=venv_id,
            function_manager=function_manager,
            session_id=session_id,
        )
        return await conn.get_state(timeout=timeout)

    def list_active_sessions(self) -> List[Tuple[int, int]]:
        """
        List all active venv sessions in the pool.

        Returns:
            List of (venv_id, session_id) tuples for sessions with live connections.
        """
        return [key for key, conn in self._connections.items() if conn.is_alive()]

    async def get_all_states(
        self,
        function_manager: "FunctionManager",
        timeout: float = 30.0,
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """
        Get serialized state from all active venv connections.

        Args:
            function_manager: The FunctionManager for venv preparation.
            timeout: Timeout for state retrieval per connection.

        Returns:
            Dict mapping (venv_id, session_id) -> state dict for each active session.
        """
        results: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for key, conn in list(self._connections.items()):
            if conn.is_alive():
                try:
                    state = await conn.get_state(timeout=timeout)
                    results[key] = state
                except Exception as e:
                    # Connection may have died during iteration
                    results[key] = {"__error__": str(e)}
        return results

    async def close(self) -> None:
        """Close all connections in the pool."""
        async with self._lock:
            self._closed = True
            for conn in self._connections.values():
                await conn.shutdown()
            self._connections.clear()
            self._metadata.clear()

    def __del__(self) -> None:
        """Ensure cleanup on garbage collection."""
        if self._connections and not self._closed:
            # Can't run async cleanup in __del__, but we can try to kill processes
            for conn in self._connections.values():
                try:
                    if conn._process.returncode is None:
                        conn._process.kill()
                except Exception:
                    pass


class _InProcessFunctionProxy:
    """Proxy that wraps an in-process function with state mode support.

    This proxy enables in-process functions (no venv) to be called with the same
    state mode API as venv-backed functions. It supports three execution modes
    for fine-grained control over state management:

    Execution Modes
    ---------------
    **stateful** (default via ``__call__``, or explicit via ``.stateful()``):
        Executes in a persistent in-process session. Variables defined in previous
        calls persist across executions within the same session_id. Use this for
        iterative workflows where you want to build up state incrementally.

    **stateless** (via ``.stateless()``):
        Executes in a fresh globals dict with no inherited state. Each call starts
        with a clean environment. Use this for pure functions that should not
        depend on or affect any global state - guarantees reproducible results.

    **read_only** (via ``.read_only()``):
        Reads the current global state from the persistent session but executes
        in a fresh globals dict. Changes made during execution are NOT persisted
        back to the session. Use this for "what-if" exploration.

    Usage Examples
    --------------
    ```python
    # Stateful (default) - state persists between calls
    await set_config(key="debug", value=True)
    await run_analysis()  # can access 'config' from previous call

    # Explicit stateful (equivalent to default __call__)
    result = await compute.stateful(x=1, y=2)

    # Stateless - fresh environment each time
    result = await compute.stateless(x=1, y=2)

    # Read-only - see current state but don't modify it
    preview = await transform.read_only(factor=2)
    ```

    Design Note
    -----------
    The proxy is returned to callers (e.g., CodeActActor) for state mode control,
    but the **raw function** remains in the execution namespace. This allows:
    - Inter-function calls to work naturally (``await b()`` calls raw ``b``)
    - ``typing.get_type_hints(fn_name)`` to resolve correctly
    - Custom decorators (``@my_decorator``) to work during exec()

    The proxy exposes ``__wrapped__`` pointing to the raw function for introspection.
    """

    def __init__(
        self,
        *,
        function_manager: "FunctionManager",
        func_data: Dict[str, Any],
        namespace: Dict[str, Any],
        raw_callable: Callable[..., Any],
    ):
        self._function_manager = function_manager
        self._func_data = func_data
        self._namespace = namespace
        self._raw_callable = raw_callable

        # Copy key attributes from raw callable for introspection
        self.__name__ = str(func_data.get("name") or "unknown")
        self.__doc__ = str(func_data.get("docstring") or "")
        self.__wrapped__ = raw_callable  # Standard Python convention for wrapper chains

    async def _execute_with_mode(
        self,
        state_mode: Literal["stateful", "read_only", "stateless"],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute the function with the specified state mode.

        Args:
            state_mode: How to handle global state during execution.
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            The function's return value.
        """
        if state_mode == "stateful":
            # Execute directly using the raw callable in the shared namespace.
            # This is the existing behavior - state naturally persists in the namespace.
            result = self._raw_callable(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        # For stateless and read_only, use execute_function with appropriate mode
        result = await self._function_manager.execute_function(
            function_name=self.__name__,
            call_kwargs=kwargs,
            target_venv_id=None,  # Force in-process execution
            state_mode=state_mode,
            session_id=0,  # Default session for read_only state source
            primitives=self._namespace.get("primitives"),
            computer_primitives=self._namespace.get("computer_primitives"),
        )

        if result.get("error"):
            raise RuntimeError(str(result.get("error")))
        return result.get("result")

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the function in stateful mode (default).

        State persists across calls within the shared namespace. Variables
        defined in previous executions remain accessible. This is the default
        behavior, suitable for iterative/interactive workflows.

        Equivalent to calling ``.stateful()`` explicitly.

        Args:
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            The function's return value.
        """
        return await self._execute_with_mode("stateful", *args, **kwargs)

    def stateful(self, *args: Any, **kwargs: Any):
        """
        Execute the function in stateful mode (explicit form of default ``__call__``).

        State persists across calls within the shared namespace. Variables
        defined in previous executions remain accessible. Use this when you want
        to be explicit about the execution mode in your code.

        Equivalent to ``await fn()`` but more self-documenting.

        Args:
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            Awaitable that resolves to the function's return value.
        """
        return self._execute_with_mode("stateful", *args, **kwargs)

    def stateless(self, *args: Any, **kwargs: Any):
        """
        Execute the function in stateless mode (fresh environment).

        Each call executes with fresh globals and no inherited state.
        The function cannot see or modify any variables from previous executions.
        Use this for pure functions that should produce identical results
        regardless of execution history.

        Args:
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            Awaitable that resolves to the function's return value.
        """
        return self._execute_with_mode("stateless", *args, **kwargs)

    def read_only(self, *args: Any, **kwargs: Any):
        """
        Execute the function in read-only mode (sees state, no persistence).

        Reads the current global state from the persistent in-process session but
        executes in a fresh globals dict. Any modifications during execution are
        discarded - the persistent session state remains unchanged.

        Args:
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            Awaitable that resolves to the function's return value.
        """
        return self._execute_with_mode("read_only", *args, **kwargs)


class _VenvFunctionProxy:
    """Proxy that wraps a venv-backed function as an awaitable callable.

    This proxy enables venv-isolated functions to be called transparently from
    the CodeActActor sandbox. It supports three execution modes for fine-grained
    control over state management:

    Execution Modes
    ---------------
    **stateful** (default via ``__call__``, or explicit via ``.stateful()``):
        Executes in a persistent subprocess connection via VenvPool. Variables
        defined in previous calls persist across executions. Use this for
        iterative workflows where you want to build up state incrementally
        (e.g., loading data once, then running multiple analyses).

    **stateless** (via ``.stateless()``):
        Executes in a fresh subprocess with no inherited state. Each call starts
        with a clean globals dict. Use this for pure functions that should not
        depend on or affect any global state - guarantees reproducible results
        regardless of prior execution history.

    **read_only** (via ``.read_only()``):
        Reads the current global state from the persistent connection but executes
        in an ephemeral subprocess. Changes made during execution are NOT persisted
        back to the session. Use this for "what-if" exploration - you can inspect
        or transform session state without side effects.

    Usage Examples
    --------------
    ```python
    # Stateful (default) - state persists between calls
    # First call: loads data into session globals
    await load_dataset(path="data.csv")
    # Second call: can access the loaded data
    await analyze_dataset()

    # Explicit stateful (equivalent to default __call__)
    result = await my_func.stateful(x=1, y=2)

    # Stateless - fresh environment each time, no side effects
    # Useful for pure computations that shouldn't depend on session state
    result = await my_func.stateless(x=1, y=2)

    # Read-only - see current state but don't modify it
    # Useful for exploratory queries without affecting the main session
    preview = await transform_data.read_only(sample_size=100)
    ```

    When to Use Each Mode
    ---------------------
    - **stateful**: Default for most use cases. Enables Jupyter-notebook-style
      workflows where you iteratively build up state.
    - **stateless**: When you need guaranteed isolation - the function's behavior
      depends only on its explicit arguments, never on hidden global state.
    - **read_only**: When you want to "peek" at what a transformation would do
      without committing the changes, or run exploratory analysis without
      polluting the session namespace.
    """

    def __init__(
        self,
        *,
        function_manager: "FunctionManager",
        func_data: Dict[str, Any],
        namespace: Dict[str, Any],
    ):
        self._function_manager = function_manager
        self._func_data = func_data
        self._namespace = namespace

        self.__name__ = str(func_data.get("name") or "unknown")
        self.__doc__ = str(func_data.get("docstring") or "")
        # Note: venv functions don't have a raw_callable since they run in subprocess

    @staticmethod
    def _map_positional_args(
        *,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        implementation: str,
        func_name: str,
    ) -> dict[str, Any]:
        """
        Map positional args to kwargs using AST-extracted parameter names.

        Note: the venv runner currently executes with ``fn(**call_kwargs)``, so we can
        only support positional args by mapping them onto non-positional-only params.
        """
        if not args:
            return kwargs

        try:
            tree = ast.parse(implementation)
        except Exception as e:
            raise TypeError(
                f"Cannot map positional args for venv function '{func_name}': failed to parse implementation",
            ) from e

        if not tree.body or not isinstance(
            tree.body[0],
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            raise TypeError(
                f"Cannot map positional args for venv function '{func_name}': implementation must contain exactly one top-level function",
            )

        node: ast.FunctionDef | ast.AsyncFunctionDef = tree.body[0]
        if node.args.posonlyargs:
            raise TypeError(
                f"Cannot call venv function '{func_name}' with positional-only args; use keyword arguments",
            )
        if node.args.vararg is not None:
            raise TypeError(
                f"Cannot call venv function '{func_name}' with *args; use keyword arguments",
            )

        param_names = [a.arg for a in node.args.args]
        if len(args) > len(param_names):
            raise TypeError(
                f"Too many positional arguments for venv function '{func_name}'",
            )

        mapped: dict[str, Any] = dict(kwargs)
        for k, v in zip(param_names[: len(args)], args):
            if k in mapped:
                raise TypeError(
                    f"Multiple values for argument '{k}' in venv function '{func_name}'",
                )
            mapped[k] = v
        return mapped

    async def _execute_with_mode(
        self,
        state_mode: Literal["stateful", "read_only", "stateless"],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute the function with the specified state mode.

        Args:
            state_mode: How to handle global state during execution.
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            The function's return value.

        Raises:
            ValueError: If venv_id is missing or implementation is invalid.
            RuntimeError: If execution fails (error from subprocess).
        """
        venv_id = self._func_data.get("venv_id")
        if venv_id is None:
            raise ValueError(f"Venv proxy '{self.__name__}' missing venv_id")

        implementation = self._func_data.get("implementation")
        if not isinstance(implementation, str) or not implementation.strip():
            raise ValueError(f"Venv function '{self.__name__}' has no implementation")

        # Strip @custom_function decorators (not available in subprocess runner).
        implementation = _strip_custom_function_decorators(implementation)

        # Determine async-ness based on source.
        is_async = "async def" in implementation

        # Resolve RPC targets from the injected namespace (caller-controlled).
        primitives = self._namespace.get("primitives")
        computer_primitives = self._namespace.get("computer_primitives")

        call_kwargs = self._map_positional_args(
            args=args,
            kwargs=kwargs,
            implementation=implementation,
            func_name=self.__name__,
        )

        # Check if a persistent venv pool is available (injected by PythonExecutionSession)
        venv_pool = self._namespace.get("__venv_pool__")
        venv_id_int = int(venv_id)

        if state_mode == "stateful":
            # Use persistent connection via VenvPool - state persists across calls
            if venv_pool is not None:
                result = await venv_pool.execute_in_venv(
                    venv_id=venv_id_int,
                    implementation=implementation,
                    call_kwargs=call_kwargs,
                    is_async=is_async,
                    primitives=primitives,
                    computer_primitives=computer_primitives,
                    function_manager=self._function_manager,
                )
            else:
                # No pool available - fall back to stateless (one-shot) execution
                # This maintains backward compatibility when VenvPool isn't injected
                result = await self._function_manager.execute_in_venv(
                    venv_id=venv_id_int,
                    implementation=implementation,
                    call_kwargs=call_kwargs,
                    is_async=is_async,
                    primitives=primitives,
                    computer_primitives=computer_primitives,
                )

        elif state_mode == "read_only":
            # Read current state from persistent connection, execute in ephemeral subprocess
            # Changes are NOT persisted back to the session
            if venv_pool is None:
                raise ValueError(
                    f"read_only mode for '{self.__name__}' requires a VenvPool to read "
                    f"existing state. Use stateless mode if you don't need to read session state.",
                )
            # Get current state from the persistent connection
            initial_state = await venv_pool.get_connection_state(
                venv_id=venv_id_int,
                function_manager=self._function_manager,
            )
            # Execute in fresh subprocess with that state (not modifying persistent state)
            result = await self._function_manager.execute_in_venv(
                venv_id=venv_id_int,
                implementation=implementation,
                call_kwargs=call_kwargs,
                is_async=is_async,
                initial_state=initial_state,
                primitives=primitives,
                computer_primitives=computer_primitives,
            )

        else:  # state_mode == "stateless"
            # Fresh subprocess with no inherited state - pure function behavior
            result = await self._function_manager.execute_in_venv(
                venv_id=venv_id_int,
                implementation=implementation,
                call_kwargs=call_kwargs,
                is_async=is_async,
                primitives=primitives,
                computer_primitives=computer_primitives,
            )

        if result.get("error"):
            raise RuntimeError(str(result.get("error")))
        return result.get("result")

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the function in stateful mode (default).

        State persists across calls within the same VenvPool session. Variables
        defined in previous executions remain accessible. This is the default
        behavior, suitable for iterative/interactive workflows.

        Equivalent to calling ``.stateful()`` explicitly.

        Args:
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            The function's return value.

        Example:
            ```python
            # First call - defines 'data' in session globals
            await load_data(path="input.csv")
            # Second call - can access 'data' from previous call
            await process_data()
            ```
        """
        return await self._execute_with_mode("stateful", *args, **kwargs)

    def stateful(self, *args: Any, **kwargs: Any):
        """
        Execute the function in stateful mode (explicit form of default ``__call__``).

        State persists across calls within the same VenvPool session. Variables
        defined in previous executions remain accessible. Use this when you want
        to be explicit about the execution mode in your code.

        Equivalent to ``await fn()`` but more self-documenting.

        Args:
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            Awaitable that resolves to the function's return value.
        """
        return self._execute_with_mode("stateful", *args, **kwargs)

    def stateless(self, *args: Any, **kwargs: Any):
        """
        Execute the function in stateless mode (fresh environment).

        Each call executes in a fresh subprocess with no inherited global state.
        The function cannot see or modify any variables from previous executions.
        Use this for pure functions that should produce identical results
        regardless of execution history.

        Args:
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            Awaitable that resolves to the function's return value.

        Example:
            ```python
            # Each call is completely independent - no shared state
            result1 = await compute_score.stateless(data=[1, 2, 3])
            result2 = await compute_score.stateless(data=[4, 5, 6])
            # result1 and result2 computed in isolated environments
            ```

        When to use:
            - Pure computations that shouldn't depend on hidden state
            - Functions where reproducibility is critical
            - Avoiding accidental state pollution from prior calls
        """
        return self._execute_with_mode("stateless", *args, **kwargs)

    def read_only(self, *args: Any, **kwargs: Any):
        """
        Execute the function in read-only mode (sees state, no persistence).

        Reads the current global state from the persistent VenvPool session but
        executes in an ephemeral subprocess. Any modifications to globals during
        execution are discarded - the persistent session state remains unchanged.

        This is useful for "what-if" exploration: you can inspect or transform
        the current session state without committing changes.

        Args:
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            Awaitable that resolves to the function's return value.

        Raises:
            ValueError: If no VenvPool is available (read_only requires existing state).

        Example:
            ```python
            # Session has 'df' DataFrame from prior stateful calls
            await load_data(path="sales.csv")  # stateful: df now in session

            # Preview a transformation without modifying the session
            preview = await filter_data.read_only(min_value=100)
            # 'df' in session is unchanged - filter was applied to a copy

            # If the preview looks good, run it statefully to persist
            await filter_data(min_value=100)  # now session 'df' is filtered
            ```

        When to use:
            - Exploratory analysis without side effects
            - Previewing transformations before committing
            - Running queries against session state without modification
        """
        return self._execute_with_mode("read_only", *args, **kwargs)


class FunctionManager(BaseFunctionManager):
    """
    Keeps a catalogue of user-supplied Python functions and system primitives.

    User-defined functions are stored in `Functions/Compositional` with auto-incrementing
    IDs. System primitives (state manager methods) are stored in `Functions/Primitives`
    with explicit stable IDs that are consistent across all users.

    This separation ensures:
    - User function IDs are stable (adding/removing primitives doesn't affect them)
    - Primitive IDs are consistent across all users (hash-based stable IDs)
    - No ID collisions between the two namespaces
    """

    class Config:
        required_contexts = [
            TableContext(
                name="Functions/VirtualEnvs",
                description="Virtual environment configurations (pyproject.toml content).",
                fields=model_to_fields(VirtualEnv),
                unique_keys={"venv_id": "int"},
                auto_counting={"venv_id": None},
            ),
            TableContext(
                name="Functions/Compositional",
                description="User-defined functions with auto-incrementing IDs.",
                fields=model_to_fields(Function),
                unique_keys={"function_id": "int"},
                auto_counting={"function_id": None},
                foreign_keys=[
                    {
                        "name": "guidance_ids[*]",
                        "references": "Guidance.guidance_id",
                        "on_delete": "CASCADE",
                        "on_update": "CASCADE",
                    },
                    {
                        "name": "venv_id",
                        "references": "Functions/VirtualEnvs.venv_id",
                        "on_delete": "SET NULL",
                        "on_update": "CASCADE",
                    },
                ],
            ),
            TableContext(
                name="Functions/Primitives",
                description="System action primitives with stable explicit IDs.",
                fields=model_to_fields(Function),
                unique_keys={"function_id": "int"},
                # No auto_counting - primitives get explicit IDs from collect_primitives()
            ),
            TableContext(
                name="Functions/Meta",
                description="Metadata for primitives sync state.",
                fields=model_to_fields(FunctionsMeta),
                unique_keys={"meta_id": "int"},
            ),
        ]

    # ------------------------------------------------------------------ #
    #  Construction                                                      #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        daemon: bool = True,
        file_manager: Optional[LocalFileManager] = None,
    ) -> None:
        # No thread behavior; keep parameter for backward compatibility
        self._daemon = daemon
        # ToDo: expose tools to LLM once needed
        self._tools: Dict[str, callable] = {}
        self.include_in_multi_assistant_table = True

        # Internal monotonically-increasing function-id counter.  We keep it local
        # to the manager to avoid an expensive scan across *all* logs every
        # time we create a function.  Initialised lazily on first use.
        self._next_id: Optional[int] = None

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        if not read_ctx:
            # Ensure the global assistant/context is selected before we derive our sub-context
            try:
                from .. import (
                    ensure_initialised as _ensure_initialised,
                )  # local to avoid cycles

                _ensure_initialised()
                ctxs = unify.get_active_context()
                read_ctx, write_ctx = ctxs["read"], ctxs["write"]
            except Exception:
                # If ensure fails (e.g. offline tests), proceed; downstream will fall back safely
                pass
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a FunctionManager."
        self._venvs_ctx = ContextRegistry.get_context(self, "Functions/VirtualEnvs")
        self._compositional_ctx = ContextRegistry.get_context(
            self,
            "Functions/Compositional",
        )
        self._primitives_ctx = ContextRegistry.get_context(self, "Functions/Primitives")
        self._meta_ctx = ContextRegistry.get_context(self, "Functions/Meta")

        # Track whether primitives, custom venvs, and custom functions have been synced
        self._primitives_synced = False
        self._custom_venvs_synced = False
        self._custom_functions_synced = False

        # ------------------------------------------------------------------ #
        #  File system mirroring (functions folder under FileManager root)    #
        # ------------------------------------------------------------------ #
        try:
            # Resolve a LocalFileManager instance (DI preferred, else via registry)
            self._fm: Optional[LocalFileManager] = (
                file_manager if file_manager is not None else LocalFileManager()
            )
        except Exception:
            self._fm = None

        self._functions_dir: Optional[Path] = None
        if self._fm is not None:
            try:
                # Access adapter root directly (LocalFileSystemAdapter._root)
                adapter = getattr(self._fm, "_adapter", None)
                root_dir = getattr(adapter, "_root", None) if adapter else None

                if root_dir is not None and isinstance(root_dir, Path):
                    functions_dir = root_dir / "functions"
                    functions_dir.mkdir(parents=True, exist_ok=True)
                    self._functions_dir = functions_dir
                    # Bootstrap: mirror existing functions from context to disk (idempotent)
                    self._bootstrap_functions_to_disk()
            except Exception:
                # Non-fatal – tests without FileManager still pass
                self._functions_dir = None

        # ------------------------------------------------------------------ #
        #  In-process session state (for stateful/read_only modes)           #
        # ------------------------------------------------------------------ #
        # Dict[session_id, Dict[str, Any]] - persistent globals per session
        self._in_process_sessions: Dict[int, Dict[str, Any]] = {}

    @property
    def _dangerous_builtins(self) -> Set[str]:
        """
        A minimal set of truly dangerous built-ins that should never be allowed.
        These could compromise security or system integrity.
        """
        return {
            "eval",
            "exec",
            "compile",
            "__import__",
            "open",  # File system access should go through proper APIs
            "input",  # No interactive input in automated functions
            "breakpoint",  # No debugging breakpoints
            "exit",
            "quit",
        }

    def _parse_implementation(
        self,
        source: str,
    ) -> Tuple[str, ast.Module, ast.FunctionDef, str]:
        """
        Common syntactic checks (unchanged, but now returns the stripped
        source verbatim so we can persist it later).
        """
        stripped = source.lstrip("\n")
        first_line = stripped.splitlines()[0] if stripped else ""
        if first_line.startswith((" ", "\t")):
            raise ValueError(
                "Function definition must start at column 0 (no indentation).",
            )

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise ValueError(f"Syntax error:\n{e.text}") from e

        if len(tree.body) != 1 or not isinstance(
            tree.body[0],
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            raise ValueError(
                "Each implementation must contain exactly one top-level function.",
            )

        fn_node: Union[ast.FunctionDef, ast.AsyncFunctionDef] = tree.body[0]
        if fn_node.col_offset != 0:
            raise ValueError(
                f"Function {fn_node.name!r} must start at column 0 (no indentation).",
            )

        return fn_node.name, tree, fn_node, source

    def _collect_verified_dependencies(
        self,
        fn_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        all_known_function_names: Set[str],
    ) -> Set[str]:
        """
        Uses the stateful _DependencyVisitor to find verified direct calls,
        indirect calls via variables, and returned function name references
        to other known library functions.
        """
        return collect_dependencies_from_function_node(
            fn_node,
            all_known_function_names,
        )

    def _collect_function_calls(
        self,
        fn_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    ) -> Set[str]:
        calls: Set[str] = set()
        for node in ast.walk(fn_node):
            if isinstance(node, ast.Call):
                name = self._format_callable_name(node.func)
                if name:
                    calls.add(name)
        return calls

    @staticmethod
    def _format_callable_name(callable_node: ast.AST) -> Optional[str]:
        """Return a best-effort fully qualified name for a callable.

        Handles both simple names (e.g., ``foo()``) and nested attributes
        (e.g., ``a.b.c()``). If the base of the attribute chain is not a simple
        ``ast.Name`` (e.g., ``get().b()``), this falls back to ``ast.unparse``
        when available.
        """
        # Simple function call: foo()
        if isinstance(callable_node, ast.Name):
            return callable_node.id

        # Attribute access: a.b.c()
        if isinstance(callable_node, ast.Attribute):
            parts: List[str] = []
            current: ast.AST = callable_node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))
            # Fallback to unparse for complex bases like calls/subscripts
            try:
                return ast.unparse(callable_node)
            except Exception:
                pass
            return ".".join(reversed(parts)) if parts else None

        try:
            return ast.unparse(callable_node)
        except Exception:
            return None

    def _validate_function_calls(
        self,
        fn_name: str,
        calls: Set[str],
        provided_names: Set[str],
    ) -> None:
        """
        Validates function calls to prevent dangerous operations.

        Allows:
        - Built-in functions (except dangerous ones)
        - Any method calls on objects (e.g., computer_primitives.*, call_handle.*, call.*)
        - User-defined functions (tracked as dependencies)

        Disallows:
        - Dangerous built-in functions (eval, exec, etc.)
        """
        dangerous = self._dangerous_builtins

        for called in calls:
            # Allow all method calls (anything with a dot)
            # This includes computer_primitives.*, call_handle.*, obj.method(), etc.
            if "." in called:
                continue

            # Block only truly dangerous built-ins
            if called in dangerous:
                raise ValueError(
                    f"Dangerous built-in '{called}' is not permitted in {fn_name}(). "
                    f"Functions cannot use: {', '.join(sorted(dangerous))}",
                )

    # ------------------------------------------------------------------ #
    #  Private helpers for persistence                                    #
    # ------------------------------------------------------------------ #

    def _get_log_by_function_id(
        self,
        *,
        function_id: int,
        raise_if_missing: bool = True,
    ) -> Optional[unify.Log]:
        logs = unify.get_logs(
            context=self._compositional_ctx,
            filter=f"function_id == {function_id}",
            exclude_fields=list_private_fields(self._compositional_ctx),
        )
        if len(logs) == 0:
            if raise_if_missing:
                raise ValueError(f"No function with id {function_id!r} exists.")
            return None
        assert len(logs) == 1, f"Multiple functions found with id {function_id!r}."
        return logs[0]

    # ------------------------------------------------------------------ #
    #  Filesystem helpers                                                #
    # ------------------------------------------------------------------ #

    def _function_filename(self, name: str) -> str:
        """Return canonical filename for a function (no extensions in name)."""
        safe = name.strip().replace(os.sep, "_")
        return f"{safe}.py"

    def _function_path(self, name: str) -> Optional[Path]:
        if self._functions_dir is None:
            return None
        return self._functions_dir / self._function_filename(name)

    def _write_function_file(self, name: str, source: str) -> Optional[Path]:
        """Atomically write the function source into the functions folder."""
        p = self._function_path(name)
        if p is None:
            return None
        try:
            tmp = p.with_suffix(p.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(source)
            os.replace(tmp, p)
            return p
        except Exception:
            return None

    def _register_function_file(self, name: str, path: Path) -> None:
        """Register function file with FileManager as protected and visible."""
        if self._fm is None:
            return
        display = f"functions/{path.name}"
        try:
            # Idempotent: if already registered under same display, keep it
            if not self._fm.exists(display):
                self._fm.register_existing_file(
                    path,
                    display_name=display,
                    protected=True,
                )
        except Exception:
            # Best-effort registration only
            pass

    def _bootstrap_functions_to_disk(self) -> None:
        """Ensure all existing functions have a file on disk and are registered."""
        if self._functions_dir is None:
            return
        try:
            logs = unify.get_logs(
                context=self._compositional_ctx,
                exclude_fields=list_private_fields(self._compositional_ctx),
            )
            for lg in logs:
                name = lg.entries.get("name")
                impl = lg.entries.get("implementation") or ""
                if not isinstance(name, str) or not impl:
                    continue
                p = self._function_path(name)
                if p is None:
                    continue
                if not p.exists():
                    wrote = self._write_function_file(name, impl)
                    if wrote is not None:
                        self._register_function_file(name, wrote)
                else:
                    # Ensure it's registered as protected even if file already exists
                    self._register_function_file(name, p)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    @functools.wraps(BaseFunctionManager.clear, updated=())
    def clear(self) -> None:
        unify.delete_context(self._compositional_ctx)
        unify.delete_context(self._primitives_ctx)
        unify.delete_context(self._venvs_ctx)
        unify.delete_context(self._meta_ctx)

        # Reset any manager-local counters or caches
        try:
            self._next_id = None
            self._primitives_synced = False
            self._custom_venvs_synced = False
            self._custom_functions_synced = False
            # Clear in-process session state
            self._in_process_sessions.clear()
        except Exception:
            pass

        # Force re-provisioning
        ContextRegistry.refresh(self, "Functions/VirtualEnvs")
        ContextRegistry.refresh(self, "Functions/Compositional")
        ContextRegistry.refresh(self, "Functions/Primitives")
        ContextRegistry.refresh(self, "Functions/Meta")

        # Verify visibility before proceeding
        try:
            import time as _time  # local import to avoid polluting module namespace

            for _ in range(3):
                try:
                    unify.get_fields(context=self._compositional_ctx)
                    break
                except Exception:
                    _time.sleep(0.05)
        except Exception:
            pass

    def clear_in_process_sessions(self, session_id: Optional[int] = None) -> None:
        """
        Clear in-process session state.

        Parameters
        ----------
        session_id : int | None, default ``None``
            If provided, clear only the specified session. If None, clear all sessions.
        """
        if session_id is not None:
            self._in_process_sessions.pop(session_id, None)
        else:
            self._in_process_sessions.clear()

    # ------------------------------------------------------------------ #
    #  Primitives sync                                                   #
    # ------------------------------------------------------------------ #

    def _get_stored_primitives_hash(self) -> Optional[str]:
        """Retrieve the primitives hash from the Meta context."""
        try:
            logs = unify.get_logs(
                context=self._meta_ctx,
                filter="meta_id == 1",
                limit=1,
            )
            if logs:
                return logs[0].entries.get("primitives_hash")
        except Exception:
            pass
        return None

    def _store_primitives_hash(self, hash_value: str) -> None:
        """Store the primitives hash in the Meta context."""
        try:
            logs = unify.get_logs(
                context=self._meta_ctx,
                filter="meta_id == 1",
                limit=1,
            )
            if logs:
                unify.update_logs(
                    logs=[logs[0].id],
                    context=self._meta_ctx,
                    entries={"primitives_hash": hash_value},
                    overwrite=True,
                )
            else:
                unity_create_logs(
                    context=self._meta_ctx,
                    entries=[{"meta_id": 1, "primitives_hash": hash_value}],
                    add_to_all_context=self.include_in_multi_assistant_table,
                )
        except Exception as e:
            logger.warning(f"Failed to store primitives hash: {e}")

    def _delete_all_primitives(self) -> None:
        """Delete all rows from the Primitives context."""
        try:
            logs = unify.get_logs(
                context=self._primitives_ctx,
                exclude_fields=list_private_fields(self._primitives_ctx),
            )
            if logs:
                unify.delete_logs(
                    context=self._primitives_ctx,
                    logs=[lg.id for lg in logs],
                )
                logger.debug(f"Deleted {len(logs)} primitive rows")
        except Exception as e:
            logger.warning(f"Failed to delete primitives: {e}")

    def _insert_primitives(self, primitives: Dict[str, Dict[str, Any]]) -> None:
        """Insert primitive rows into the Primitives context with explicit IDs."""
        if not primitives:
            return

        entries = []
        for name, data in primitives.items():
            entry = {
                "name": data["name"],
                "function_id": data[
                    "function_id"
                ],  # Explicit stable ID from collect_primitives()
                "argspec": data["argspec"],
                "docstring": data["docstring"],
                "embedding_text": data["embedding_text"],
                "implementation": None,
                "depends_on": [],
                "precondition": None,
                "verify": False,
                "is_primitive": True,
                "guidance_ids": [],
                "primitive_class": data.get("primitive_class"),
                "primitive_method": data.get("primitive_method"),
            }
            entries.append(entry)

        try:
            unity_create_logs(
                context=self._primitives_ctx,
                entries=entries,
                batched=True,
                add_to_all_context=self.include_in_multi_assistant_table,
            )
            logger.info(f"Inserted {len(entries)} primitives")
        except Exception as e:
            logger.error(f"Failed to insert primitives: {e}")

    def sync_primitives(self) -> bool:
        """
        Ensure primitives in the database match current Python definitions.

        Uses hash comparison to avoid unnecessary writes. Safe to call
        multiple times; will only perform sync if primitives have changed.

        Returns:
            True if sync was performed, False if already up-to-date.
        """
        if self._primitives_synced:
            return False

        expected = collect_primitives()
        expected_hash = compute_primitives_hash(expected)

        current_hash = self._get_stored_primitives_hash()

        if current_hash == expected_hash:
            logger.debug("Primitives hash matches, skipping sync")
            self._primitives_synced = True
            return False

        logger.info(
            f"Primitives hash mismatch (current={current_hash}, expected={expected_hash}), syncing...",
        )
        self._delete_all_primitives()
        self._insert_primitives(expected)
        self._store_primitives_hash(expected_hash)

        self._primitives_synced = True
        return True

    # ------------------------------------------------------------------ #
    #  Custom Functions Sync                                              #
    # ------------------------------------------------------------------ #

    def _get_stored_custom_functions_hash(self) -> str:
        """Retrieve the stored custom functions hash from the Meta context."""
        try:
            logs = unify.get_logs(
                context=self._meta_ctx,
                filter="meta_id == 1",
                limit=1,
            )
            if logs:
                return logs[0].entries.get("custom_functions_hash", "")
        except Exception as e:
            logger.warning(f"Failed to retrieve custom functions hash: {e}")
        return ""

    def _store_custom_functions_hash(self, hash_value: str) -> None:
        """Store the custom functions hash in the Meta context."""
        try:
            logs = unify.get_logs(
                context=self._meta_ctx,
                filter="meta_id == 1",
                limit=1,
            )
            if logs:
                unify.update_logs(
                    context=self._meta_ctx,
                    logs=[logs[0].id],
                    entries={"custom_functions_hash": hash_value},
                    overwrite=True,
                )
            else:
                # Create the meta row if it doesn't exist
                unity_create_logs(
                    context=self._meta_ctx,
                    entries=[{"meta_id": 1, "custom_functions_hash": hash_value}],
                    add_to_all_context=self.include_in_multi_assistant_table,
                )
        except Exception as e:
            logger.warning(f"Failed to store custom functions hash: {e}")

    def _get_custom_functions_from_db(self) -> Dict[str, Dict[str, Any]]:
        """Get all custom functions from the database (those with custom_hash set)."""
        logs = unify.get_logs(
            context=self._compositional_ctx,
            filter="custom_hash != None",
            exclude_fields=list_private_fields(self._compositional_ctx),
        )
        return {
            lg.entries.get("name"): lg.entries for lg in logs if lg.entries.get("name")
        }

    def _delete_custom_function_by_name(self, name: str) -> bool:
        """Delete a custom function by name."""
        logs = unify.get_logs(
            context=self._compositional_ctx,
            filter=f"name == '{name}' and custom_hash != None",
            limit=1,
        )
        if not logs:
            return False
        unify.delete_logs(
            context=self._compositional_ctx,
            logs=[logs[0].id],
        )
        return True

    def _update_custom_function(
        self,
        function_id: int,
        data: Dict[str, Any],
    ) -> None:
        """Update an existing custom function."""
        log = self._get_log_by_function_id(
            function_id=function_id,
            raise_if_missing=True,
        )
        # Update all fields except function_id (preserve it)
        update_data = {k: v for k, v in data.items() if k != "function_id"}
        unify.update_logs(
            context=self._compositional_ctx,
            logs=[log.id],
            entries=update_data,
            overwrite=True,
        )

    def _insert_custom_function(self, data: Dict[str, Any]) -> int:
        """Insert a new custom function."""
        # Remove function_id if present - let it be auto-assigned
        insert_data = {k: v for k, v in data.items() if k != "function_id"}
        result = unity_create_logs(
            context=self._compositional_ctx,
            entries=[insert_data],
            add_to_all_context=self.include_in_multi_assistant_table,
        )
        # unity_create_logs can return either a dict or a list of Log objects
        if isinstance(result, list) and len(result) > 0:
            log = result[0]
            if hasattr(log, "entries"):
                return log.entries.get("function_id", -1)
        elif isinstance(result, dict):
            log_ids = result.get("log_event_ids", [])
            if log_ids:
                logs = unify.get_logs(
                    context=self._compositional_ctx,
                    filter=f"id == {log_ids[0]}",
                    limit=1,
                )
                if logs and hasattr(logs[0], "entries"):
                    return logs[0].entries.get("function_id")
        return -1

    # ------------------------------------------------------------------ #
    #  Custom Venvs Sync                                                  #
    # ------------------------------------------------------------------ #

    def _get_stored_custom_venvs_hash(self) -> str:
        """Retrieve the stored custom venvs hash from the Meta context."""
        try:
            logs = unify.get_logs(
                context=self._meta_ctx,
                filter="meta_id == 1",
                limit=1,
            )
            if logs:
                return logs[0].entries.get("custom_venvs_hash", "")
        except Exception as e:
            logger.warning(f"Failed to retrieve custom venvs hash: {e}")
        return ""

    def _store_custom_venvs_hash(self, hash_value: str) -> None:
        """Store the custom venvs hash in the Meta context."""
        try:
            logs = unify.get_logs(
                context=self._meta_ctx,
                filter="meta_id == 1",
                limit=1,
            )
            if logs:
                unify.update_logs(
                    context=self._meta_ctx,
                    logs=[logs[0].id],
                    entries={"custom_venvs_hash": hash_value},
                    overwrite=True,
                )
            else:
                unity_create_logs(
                    context=self._meta_ctx,
                    entries=[{"meta_id": 1, "custom_venvs_hash": hash_value}],
                    add_to_all_context=self.include_in_multi_assistant_table,
                )
        except Exception as e:
            logger.warning(f"Failed to store custom venvs hash: {e}")

    def _get_custom_venvs_from_db(self) -> Dict[str, Dict[str, Any]]:
        """Get all custom venvs from the database (those with custom_hash set)."""
        logs = unify.get_logs(
            context=self._venvs_ctx,
            filter="custom_hash != None",
            exclude_fields=list_private_fields(self._venvs_ctx),
        )
        return {
            lg.entries.get("name"): lg.entries for lg in logs if lg.entries.get("name")
        }

    def _delete_custom_venv_by_name(self, name: str) -> bool:
        """Delete a custom venv by name."""
        logs = unify.get_logs(
            context=self._venvs_ctx,
            filter=f"name == '{name}' and custom_hash != None",
            limit=1,
        )
        if not logs:
            return False
        unify.delete_logs(
            context=self._venvs_ctx,
            logs=[logs[0].id],
        )
        return True

    def _update_custom_venv(self, venv_id: int, data: Dict[str, Any]) -> None:
        """Update an existing custom venv."""
        logs = unify.get_logs(
            context=self._venvs_ctx,
            filter=f"venv_id == {venv_id}",
            limit=1,
        )
        if not logs:
            raise ValueError(f"VirtualEnv with ID {venv_id} not found")
        update_data = {k: v for k, v in data.items() if k != "venv_id"}
        unify.update_logs(
            context=self._venvs_ctx,
            logs=[logs[0].id],
            entries=update_data,
            overwrite=True,
        )

    def _insert_custom_venv(self, data: Dict[str, Any]) -> int:
        """Insert a new custom venv."""
        insert_data = {k: v for k, v in data.items() if k != "venv_id"}
        result = unity_create_logs(
            context=self._venvs_ctx,
            entries=[insert_data],
            add_to_all_context=self.include_in_multi_assistant_table,
        )
        # unity_create_logs can return either a dict or a list of Log objects
        if isinstance(result, list) and len(result) > 0:
            log = result[0]
            if hasattr(log, "entries"):
                return log.entries.get("venv_id", -1)
        elif isinstance(result, dict):
            log_ids = result.get("log_event_ids", [])
            if log_ids:
                logs = unify.get_logs(
                    context=self._venvs_ctx,
                    filter=f"id == {log_ids[0]}",
                    limit=1,
                )
                if logs and hasattr(logs[0], "entries"):
                    return logs[0].entries.get("venv_id")
        return -1

    def sync_custom_venvs(self) -> Dict[str, int]:
        """
        Ensure custom venvs in the database match source definitions.

        Scans the custom/venvs/ folder for .toml files and syncs them
        to Functions/VirtualEnvs. Uses hash comparison to minimize writes.

        Behavior:
        - New venvs: inserted with auto-assigned venv_id
        - Changed venvs: updated in place (preserves venv_id)
        - Deleted venvs (in source): deleted from database
        - User-added venvs with same name: overwritten by source version

        Returns:
            Dict mapping venv name to venv_id (for use by sync_custom_functions).
        """
        if self._custom_venvs_synced:
            # Return existing name→id mapping
            db_venvs = self._get_custom_venvs_from_db()
            return {name: v["venv_id"] for name, v in db_venvs.items()}

        source_venvs = collect_custom_venvs()
        expected_hash = compute_custom_venvs_hash()
        current_hash = self._get_stored_custom_venvs_hash()

        # Quick check: if aggregate hash matches, skip detailed sync
        if current_hash == expected_hash:
            logger.debug("Custom venvs hash matches, skipping sync")
            self._custom_venvs_synced = True
            db_venvs = self._get_custom_venvs_from_db()
            return {name: v["venv_id"] for name, v in db_venvs.items()}

        logger.info(
            f"Custom venvs hash mismatch "
            f"(current={current_hash}, expected={expected_hash}), syncing...",
        )

        db_venvs = self._get_custom_venvs_from_db()
        processed_names: Set[str] = set()
        name_to_id: Dict[str, int] = {}

        for name, source_data in source_venvs.items():
            processed_names.add(name)

            if name in db_venvs:
                db_entry = db_venvs[name]
                if db_entry.get("custom_hash") != source_data["custom_hash"]:
                    logger.info(f"Updating custom venv: {name}")
                    self._update_custom_venv(
                        venv_id=db_entry["venv_id"],
                        data=source_data,
                    )
                else:
                    logger.debug(f"Custom venv unchanged: {name}")
                name_to_id[name] = db_entry["venv_id"]
            else:
                # Check for user-added venv with same name
                existing = unify.get_logs(
                    context=self._venvs_ctx,
                    filter=f"name == '{name}'",
                    limit=1,
                )
                if existing:
                    logger.info(f"Overwriting user-added venv with custom: {name}")
                    unify.delete_logs(
                        context=self._venvs_ctx,
                        logs=[existing[0].id],
                    )

                logger.info(f"Inserting custom venv: {name}")
                new_id = self._insert_custom_venv(source_data)
                name_to_id[name] = new_id

        # Delete venvs that are in DB but not in source
        for name in db_venvs:
            if name not in processed_names:
                logger.info(f"Deleting removed custom venv: {name}")
                self._delete_custom_venv_by_name(name)

        self._store_custom_venvs_hash(expected_hash)
        self._custom_venvs_synced = True

        return name_to_id

    def sync_custom_functions(
        self,
        venv_name_to_id: Optional[Dict[str, int]] = None,
    ) -> bool:
        """
        Ensure custom functions in the database match source definitions.

        Scans the custom/functions/ folder for functions decorated with
        @custom_function and syncs them to Functions/Compositional. Uses
        per-function hash comparison to minimize database writes.

        Args:
            venv_name_to_id: Optional mapping from venv name to venv_id.
                             Used to resolve venv_name in decorators.
                             If not provided, venv_name resolution is skipped.

        Behavior:
        - New functions: inserted with auto-assigned function_id
        - Changed functions: updated in place (preserves function_id)
        - Deleted functions (in source): deleted from database
        - User-added functions with same name: overwritten by source version
        - venv_name: resolved to venv_id using venv_name_to_id mapping

        Returns:
            True if sync was performed, False if already up-to-date.
        """
        if self._custom_functions_synced:
            return False

        # Collect source-defined custom functions
        source_functions = collect_custom_functions()
        expected_hash = compute_custom_functions_hash()
        current_hash = self._get_stored_custom_functions_hash()

        # Quick check: if aggregate hash matches, skip detailed sync
        if current_hash == expected_hash:
            logger.debug("Custom functions hash matches, skipping sync")
            self._custom_functions_synced = True
            return False

        logger.info(
            f"Custom functions hash mismatch "
            f"(current={current_hash}, expected={expected_hash}), syncing...",
        )

        venv_name_to_id = venv_name_to_id or {}

        # Get existing custom functions from DB
        db_functions = self._get_custom_functions_from_db()

        # Track what we've processed
        processed_names: Set[str] = set()

        # Sync each source function
        for name, source_data in source_functions.items():
            processed_names.add(name)

            # Resolve venv_name to venv_id
            venv_name = source_data.get("venv_name")
            if venv_name and venv_name in venv_name_to_id:
                source_data["venv_id"] = venv_name_to_id[venv_name]
                logger.debug(
                    f"Resolved venv_name={venv_name} to "
                    f"venv_id={source_data['venv_id']} for {name}",
                )
            # Remove venv_name from source_data (not stored in DB)
            source_data.pop("venv_name", None)

            if name in db_functions:
                db_entry = db_functions[name]
                # Check if hash changed
                if db_entry.get("custom_hash") != source_data["custom_hash"]:
                    logger.info(f"Updating custom function: {name}")
                    self._update_custom_function(
                        function_id=db_entry["function_id"],
                        data=source_data,
                    )
                else:
                    logger.debug(f"Custom function unchanged: {name}")
            else:
                # Check if there's a user-added function with same name
                # (no custom_hash) - if so, we need to delete it first
                existing = unify.get_logs(
                    context=self._compositional_ctx,
                    filter=f"name == '{name}'",
                    limit=1,
                )
                if existing:
                    logger.info(
                        f"Overwriting user-added function with custom: {name}",
                    )
                    unify.delete_logs(
                        context=self._compositional_ctx,
                        logs=[existing[0].id],
                    )

                # Insert new custom function
                logger.info(f"Inserting custom function: {name}")
                self._insert_custom_function(source_data)

        # Delete functions that are in DB but not in source
        for name in db_functions:
            if name not in processed_names:
                logger.info(f"Deleting removed custom function: {name}")
                self._delete_custom_function_by_name(name)

        # Store the new hash
        self._store_custom_functions_hash(expected_hash)

        self._custom_functions_synced = True
        return True

    def sync_custom(self) -> bool:
        """
        Sync all custom venvs and functions from source.

        This is the recommended method for syncing custom definitions.
        It ensures venvs are synced first (so venv_name can be resolved),
        then syncs functions.

        Returns:
            True if any sync was performed, False if everything up-to-date.
        """
        # Sync venvs first to get name→id mapping
        venv_name_to_id = self.sync_custom_venvs()

        # Then sync functions with the mapping
        functions_changed = self.sync_custom_functions(venv_name_to_id)

        # Return True if venvs were newly synced OR functions changed
        # (venv sync always returns a dict, not a bool, so check if hash changed)
        venvs_hash_changed = (
            self._get_stored_custom_venvs_hash() != compute_custom_venvs_hash()
            if not self._custom_venvs_synced
            else False
        )

        return venvs_hash_changed or functions_changed

    def list_primitives(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a mapping of primitive name to primitive metadata.

        Returns primitives from the Primitives context. Call sync_primitives()
        first to ensure the database is up-to-date.

        Returns:
            Dict mapping primitive name to metadata dict (includes function_id).
        """
        entries: Dict[str, Dict[str, Any]] = {}
        try:
            logs = unify.get_logs(
                context=self._primitives_ctx,
                exclude_fields=list_private_fields(self._primitives_ctx),
            )
            for log in logs:
                data = {
                    "function_id": log.entries.get("function_id"),
                    "name": log.entries["name"],
                    "argspec": log.entries.get("argspec", ""),
                    "docstring": log.entries.get("docstring", ""),
                    "is_primitive": True,
                    "primitive_class": log.entries.get("primitive_class"),
                    "primitive_method": log.entries.get("primitive_method"),
                }
                entries[log.entries["name"]] = data
        except Exception as e:
            logger.warning(f"Failed to list primitives: {e}")
        return entries

    # 1. Add / register ------------------------------------------------- #

    @functools.wraps(BaseFunctionManager.add_functions, updated=())
    def add_functions(
        self,
        *,
        implementations: Union[str, List[str]],
        language: Literal["python", "bash", "zsh", "sh", "powershell"] = "python",
        preconditions: Optional[Dict[str, Dict]] = None,
        verify: Optional[Dict[str, bool]] = None,
        overwrite: bool = False,
    ) -> Dict[str, str]:
        """
        Add or update functions in batch.

        Args:
            implementations: Function source code (single string or list of strings).
            language: The language/interpreter for the function(s). Default is "python".
            preconditions: Optional preconditions for functions.
            verify: Optional verification settings (name -> bool).
            overwrite: If True, update existing functions; if False, skip duplicates.

        Returns:
            Dictionary mapping function names to status ("added", "updated", "skipped", or "error").
        """

        if preconditions is None:
            preconditions = {}
        if verify is None:
            verify = {}
        if isinstance(implementations, str):
            implementations = [implementations]

        # Branch based on language
        if language != "python":
            return self._add_shell_functions(
                implementations=implementations,
                language=language,
                preconditions=preconditions,
                verify=verify,
                overwrite=overwrite,
            )

        # Python-specific parsing and validation
        parsed: List[Tuple[str, ast.Module, ast.FunctionDef, str]] = []
        parse_errors: Dict[str, str] = {}
        temp_names: Set[str] = set()

        # Parse all implementations
        for i, source in enumerate(implementations):
            try:
                # _parse_implementation validates basic structure (one func at col 0)
                name, tree, node, src = self._parse_implementation(source)
                parsed.append((name, tree, node, src))
                temp_names.add(name)
            except ValueError as e:
                # Associate error with name or index
                potential_name = f"implementation_{i+1}"
                try:
                    name_in_error = ast.parse(source).body[0].name
                except:
                    name_in_error = None
                key = name_in_error or potential_name
                parse_errors[key] = f"error: {e}"

        results: Dict[str, str] = parse_errors

        # Get existing functions for duplicate detection and dependency checking
        try:
            existing_functions = self.list_functions()
            existing_names = set(existing_functions.keys())
            all_known_function_names = existing_names.union(temp_names)
        except Exception as e:
            logger.warning(
                f"Failed to list existing functions for dependency check: {e}",
            )
            existing_functions = {}
            existing_names = set()
            all_known_function_names = temp_names

        # Check for duplicates and separate into new vs. existing functions
        duplicates_to_skip: Set[str] = set()
        existing_to_update: Set[str] = set()

        for name in temp_names:
            if name in existing_names:
                if overwrite:
                    # Mark for in-place update
                    existing_to_update.add(name)
                else:
                    # Skip this function - already exists
                    duplicates_to_skip.add(name)
                    results[name] = "skipped: already exists"

        # Validate dependencies and prepare entries for batch operations
        entries_to_create: List[Dict[str, Any]] = []
        entries_to_update: List[Dict[str, Any]] = []
        log_ids_to_update: List[int] = []
        log_id_to_name: Dict[int, str] = {}
        functions_to_write: List[Tuple[str, str]] = []

        for name, tree, node, source in parsed:
            if name in duplicates_to_skip:
                continue

            try:
                dependencies = self._collect_verified_dependencies(
                    node,
                    all_known_function_names,
                )
                dependencies_list = sorted(list(dependencies))

                all_calls = self._collect_function_calls(node)
                self._validate_function_calls(name, all_calls, temp_names)
                namespace = create_base_globals()
                exec(source, namespace)
                fn_obj = namespace[name]
                signature = str(inspect.signature(fn_obj))
                docstring = inspect.getdoc(fn_obj) or ""
                embedding_text = f"Function Name: {name}\nSignature: {signature}\nDocstring: {docstring}"
                precondition = preconditions.get(name)
                should_verify = verify.get(name, True)

                entry_data = {
                    "language": "python",
                    "argspec": signature,
                    "docstring": docstring,
                    "implementation": source,
                    "depends_on": dependencies_list,
                    "embedding_text": embedding_text,
                    "precondition": precondition,
                    "verify": should_verify,
                }

                if name in existing_to_update:
                    # Update existing function
                    log_id = self._get_log_by_function_id(
                        function_id=existing_functions[name]["function_id"],
                        raise_if_missing=True,
                    ).id
                    log_ids_to_update.append(log_id)
                    log_id_to_name[log_id] = name
                    entries_to_update.append(entry_data)
                    results[name] = "updated"
                else:
                    # Create new function
                    entry_data["name"] = name
                    entry_data["guidance_ids"] = []
                    entries_to_create.append(entry_data)
                    results[name] = "added"

                functions_to_write.append((name, source))
            except ValueError as e:
                results[name] = f"error: {e}"
            except Exception as e:
                results[name] = f"error: Unexpected error - {e}"
                logger.error(
                    f"Unexpected error processing function {name}: {e}",
                    exc_info=True,
                )

        # Batch create new functions
        if entries_to_create:
            try:
                unity_create_logs(
                    context=self._compositional_ctx,
                    entries=entries_to_create,
                    batched=True,
                    add_to_all_context=self.include_in_multi_assistant_table,
                )
            except Exception as e:
                logger.error(
                    f"Failed to batch create function logs: {e}",
                    exc_info=True,
                )
                for entry in entries_to_create:
                    name = entry["name"]
                    if results.get(name) == "added":
                        results[name] = f"error: Failed to create log - {e}"
                        functions_to_write = [
                            (n, s) for n, s in functions_to_write if n != name
                        ]

        # Batch update existing functions
        if log_ids_to_update and entries_to_update:
            try:
                unify.update_logs(
                    logs=log_ids_to_update,
                    context=self._compositional_ctx,
                    entries=entries_to_update,
                    overwrite=True,
                )
            except Exception as e:
                logger.error(
                    f"Failed to batch update function logs: {e}",
                    exc_info=True,
                )
                for log_id in log_ids_to_update:
                    name = log_id_to_name.get(log_id)
                    if name and results.get(name) == "updated":
                        results[name] = f"error: Failed to update log - {e}"
                        functions_to_write = [
                            (n, s) for n, s in functions_to_write if n != name
                        ]

        # Write function files to disk
        for name, source in functions_to_write:
            p = self._write_function_file(name, source)
            if p is not None:
                self._register_function_file(name, p)

        return results

    def _add_shell_functions(
        self,
        *,
        implementations: List[str],
        language: Literal["bash", "zsh", "sh", "powershell"],
        preconditions: Dict[str, Dict],
        verify: Dict[str, bool],
        overwrite: bool,
    ) -> Dict[str, str]:
        """
        Add shell script functions (bash, zsh, sh, powershell).

        Shell scripts must include metadata comments at the top:
            # @name: my_function
            # @args: (input_file output_file --verbose)
            # @description: Brief description

        The @name comment is required. @args and @description are optional.
        """
        results: Dict[str, str] = {}
        parsed: List[Tuple[str, str, str, str, str]] = (
            []
        )  # (name, argspec, docstring, source, language)
        temp_names: Set[str] = set()

        # Parse metadata from all implementations
        for i, source in enumerate(implementations):
            metadata = _parse_shell_script_metadata(source)
            name = metadata["name"]

            if not name:
                key = f"implementation_{i+1}"
                results[key] = (
                    "error: Shell script must include '# @name: <function_name>' comment"
                )
                continue

            parsed.append(
                (
                    name,
                    metadata["argspec"],
                    metadata["docstring"],
                    source,
                    language,
                ),
            )
            temp_names.add(name)

        # Get existing functions for duplicate detection
        try:
            existing_functions = self.list_functions()
            existing_names = set(existing_functions.keys())
        except Exception as e:
            logger.warning(f"Failed to list existing functions: {e}")
            existing_functions = {}
            existing_names = set()

        # Check for duplicates
        duplicates_to_skip: Set[str] = set()
        existing_to_update: Set[str] = set()

        for name in temp_names:
            if name in existing_names:
                if overwrite:
                    existing_to_update.add(name)
                else:
                    duplicates_to_skip.add(name)
                    results[name] = "skipped: already exists"

        # Prepare entries for batch operations
        entries_to_create: List[Dict[str, Any]] = []
        entries_to_update: List[Dict[str, Any]] = []
        log_ids_to_update: List[int] = []
        log_id_to_name: Dict[int, str] = {}
        functions_to_write: List[Tuple[str, str]] = []

        for name, argspec, docstring, source, lang in parsed:
            if name in duplicates_to_skip:
                continue

            try:
                embedding_text = f"Function Name: {name}\nLanguage: {lang}\nSignature: {argspec}\nDocstring: {docstring}"
                precondition = preconditions.get(name)
                should_verify = verify.get(name, True)

                entry_data = {
                    "argspec": argspec,
                    "docstring": docstring,
                    "implementation": source,
                    "language": lang,
                    "depends_on": [],  # Shell scripts don't have auto-detected dependencies
                    "embedding_text": embedding_text,
                    "precondition": precondition,
                    "verify": should_verify,
                }

                if name in existing_to_update:
                    # Update existing function
                    log_id = self._get_log_by_function_id(
                        function_id=existing_functions[name]["function_id"],
                        raise_if_missing=True,
                    ).id
                    log_ids_to_update.append(log_id)
                    log_id_to_name[log_id] = name
                    entries_to_update.append(entry_data)
                    results[name] = "updated"
                else:
                    # Create new function
                    entry_data["name"] = name
                    entry_data["guidance_ids"] = []
                    entries_to_create.append(entry_data)
                    results[name] = "added"

                functions_to_write.append((name, source))

            except Exception as e:
                results[name] = f"error: {e}"
                logger.error(
                    f"Error processing shell function {name}: {e}",
                    exc_info=True,
                )

        # Batch create new functions
        if entries_to_create:
            try:
                unity_create_logs(
                    context=self._compositional_ctx,
                    entries=entries_to_create,
                    batched=True,
                    add_to_all_context=self.include_in_multi_assistant_table,
                )
            except Exception as e:
                logger.error(
                    f"Failed to batch create shell function logs: {e}",
                    exc_info=True,
                )
                for entry in entries_to_create:
                    name = entry["name"]
                    if results.get(name) == "added":
                        results[name] = f"error: Failed to create log - {e}"
                        functions_to_write = [
                            (n, s) for n, s in functions_to_write if n != name
                        ]

        # Batch update existing functions
        if log_ids_to_update and entries_to_update:
            try:
                unify.update_logs(
                    logs=log_ids_to_update,
                    context=self._compositional_ctx,
                    entries=entries_to_update,
                    overwrite=True,
                )
            except Exception as e:
                logger.error(
                    f"Failed to batch update shell function logs: {e}",
                    exc_info=True,
                )
                for log_id in log_ids_to_update:
                    name = log_id_to_name.get(log_id)
                    if name and results.get(name) == "updated":
                        results[name] = f"error: Failed to update log - {e}"
                        functions_to_write = [
                            (n, s) for n, s in functions_to_write if n != name
                        ]

        # Write function files to disk (with appropriate extension)
        for name, source in functions_to_write:
            ext = ".sh" if language in ("bash", "zsh", "sh") else ".ps1"
            p = self._write_function_file(f"{name}{ext}", source)
            if p is not None:
                self._register_function_file(name, p)

        return results

    # ------------------------------------------------------------------ #
    #  Callable return + dependency injection                             #
    # ------------------------------------------------------------------ #

    def _get_function_data_by_name(self, *, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single compositional function record by name.

        Returns the full stored record (as a dict) or ``None`` if not found.
        """
        # Normalize to the Unify filter grammar (and avoid quote-escaping issues).
        try:
            normalized = normalize_filter_expr(f"name == {json.dumps(name)}")
        except Exception:
            normalized = f"name == {json.dumps(name)}"

        last_exc: Exception | None = None
        import time as _time

        # The backend can return 404 for missing contexts in fresh projects/tests.
        for delay in (0.0, 0.05, 0.15):
            if delay:
                _time.sleep(delay)
            try:
                logs = unify.get_logs(
                    context=self._compositional_ctx,
                    filter=normalized,
                    limit=1,
                    exclude_fields=list_private_fields(self._compositional_ctx),
                )
                if logs:
                    return logs[0].entries
                return None
            except _UnifyRequestError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 404:
                    last_exc = e
                    continue
                raise
            except Exception as e:
                last_exc = e
                break

        # Treat missing context as empty library.
        if isinstance(last_exc, _UnifyRequestError):
            status = getattr(getattr(last_exc, "response", None), "status_code", None)
            if status == 404:
                return None
        if last_exc is not None:
            raise last_exc
        return None

    def _create_in_process_callable(
        self,
        func_data: Dict[str, Any],
        *,
        namespace: Dict[str, Any],
    ) -> _InProcessFunctionProxy:
        """Create an in-process callable wrapped in a proxy with state mode support.

        The function is exec'd into the provided ``namespace``, placing the **raw
        function** there. The caller should NOT overwrite ``namespace[func_name]``
        with the returned proxy - this allows:

        - Inter-function calls to work naturally (``await b()`` calls raw ``b``)
        - ``typing.get_type_hints(fn_name)`` to resolve correctly via ``__wrapped__``
        - Custom decorators (``@my_decorator``) to work during exec()

        The returned proxy provides state mode control:
        ``.stateful()`` / ``.stateless()`` / ``.read_only()``.
        """
        func_name = func_data.get("name")
        if not isinstance(func_name, str) or not func_name:
            raise ValueError("func_data missing valid 'name'")

        implementation = func_data.get("implementation")
        if not isinstance(implementation, str) or not implementation.strip():
            raise ValueError(f"Function '{func_name}' has no implementation")

        implementation = _strip_custom_function_decorators(implementation)

        # Ensure user-defined annotation symbols don't cause NameErrors when callers
        # (e.g., CodeActActor) later resolve type hints via typing.get_type_hints().
        self._inject_forward_ref_annotation_placeholders(
            implementation,
            namespace=namespace,
        )

        exec(implementation, namespace)
        raw_fn = namespace.get(func_name)
        if not callable(raw_fn):
            raise ValueError(
                f"Function '{func_name}' not found after exec() into namespace",
            )

        # Wrap in proxy to provide state mode API (.stateless(), .read_only())
        return _InProcessFunctionProxy(
            function_manager=self,
            func_data=func_data,
            namespace=namespace,
            raw_callable=raw_fn,
        )

    @staticmethod
    def _inject_forward_ref_annotation_placeholders(
        implementation: str,
        *,
        namespace: Dict[str, Any],
    ) -> None:
        """
        Inject placeholder types for missing symbols referenced in annotations.

        Motivation:
        - `exec()` succeeds when annotations are strings, but later calls to
          `typing.get_type_hints(fn)` will evaluate forward-ref strings in
          `fn.__globals__` and can raise NameError if the referenced types are not
          present.
        - CodeActActor wants a callable that "just works" without manual seeding
          of domain-specific types into the namespace.

        This only attempts to satisfy *annotation resolution* (not runtime logic).
        If the function body actually uses a type (e.g. `Role.ADMIN`), the
        function must still import/define it itself.
        """
        try:
            tree = ast.parse(implementation)
        except Exception:
            return

        if not tree.body:
            return

        fn_node: Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]] = None
        if len(tree.body) == 1 and isinstance(
            tree.body[0],
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            fn_node = tree.body[0]
        else:
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fn_node = node
                    break
        if fn_node is None:
            return

        ann_exprs: List[ast.AST] = []
        args = fn_node.args
        for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
            if getattr(arg, "annotation", None) is not None:
                ann_exprs.append(arg.annotation)  # type: ignore[arg-type]
        if (
            args.vararg is not None
            and getattr(args.vararg, "annotation", None) is not None
        ):
            ann_exprs.append(args.vararg.annotation)  # type: ignore[arg-type]
        if (
            args.kwarg is not None
            and getattr(args.kwarg, "annotation", None) is not None
        ):
            ann_exprs.append(args.kwarg.annotation)  # type: ignore[arg-type]
        if getattr(fn_node, "returns", None) is not None:
            ann_exprs.append(fn_node.returns)  # type: ignore[arg-type]

        annotation_names: Set[str] = set()
        for expr in ann_exprs:
            # Forward-ref strings: parse the string itself as a Python expression.
            if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
                ref = expr.value.strip()
                if not ref:
                    continue
                try:
                    ref_tree = ast.parse(ref, mode="eval")
                except Exception:
                    continue
                for node in ast.walk(ref_tree):
                    if isinstance(node, ast.Name) and node.id:
                        annotation_names.add(node.id)
                continue

            for node in ast.walk(expr):
                if isinstance(node, ast.Name) and node.id:
                    annotation_names.add(node.id)

        if not annotation_names:
            return

        builtins_obj = namespace.get("__builtins__", {})
        if isinstance(builtins_obj, dict):
            builtin_names = set(builtins_obj.keys())
        else:
            builtin_names = set(dir(builtins_obj))

        typing_mod = namespace.get("typing")
        pydantic_mod = namespace.get("pydantic")

        for name in sorted(annotation_names):
            if name == "typing":
                continue
            if name in namespace:
                continue
            if name in builtin_names:
                continue

            # Common typing helpers can be recovered from typing when present.
            if typing_mod is not None and hasattr(typing_mod, name):
                try:
                    namespace[name] = getattr(typing_mod, name)
                    continue
                except Exception:
                    pass

            # Some common pydantic types may appear in annotations.
            if pydantic_mod is not None and hasattr(pydantic_mod, name):
                try:
                    namespace[name] = getattr(pydantic_mod, name)
                    continue
                except Exception:
                    pass

            # Fall back to a placeholder type to avoid NameError during hint resolution.
            try:
                namespace[name] = type(name, (), {})
            except Exception:
                # If something extremely unusual happens, just skip.
                continue

    def _create_venv_callable(
        self,
        func_data: Dict[str, Any],
        *,
        namespace: Dict[str, Any],
    ) -> Callable[..., Any]:
        """Create a proxy callable for a function that must run in an isolated venv."""
        return _VenvFunctionProxy(
            function_manager=self,
            func_data=func_data,
            namespace=namespace,
        )

    def _inject_dependencies(
        self,
        func_data: Dict[str, Any],
        *,
        namespace: Dict[str, Any],
        visited: Set[str],
    ) -> None:
        """Inject transitive dependencies into ``namespace`` (breadth-first).

        For in-process functions, the raw function (from exec) remains in the
        namespace for inter-function calls, decorators, and introspection.
        For venv functions, the proxy is placed in namespace (no raw function exists).
        """
        from collections import deque

        deps = func_data.get("depends_on") or []
        if not isinstance(deps, list):
            return

        q = deque([d for d in deps if isinstance(d, str) and d])
        while q:
            dep_name = q.popleft()
            if dep_name in visited:
                continue
            visited.add(dep_name)

            dep_data = self._get_function_data_by_name(name=dep_name)
            if not dep_data:
                logger.warning(
                    f"Dependency '{dep_name}' not found for '{func_data.get('name')}', skipping",
                )
                continue

            # Handle venv dependencies: proxy goes in namespace (only way to call them)
            if dep_data.get("venv_id") is not None:
                namespace[dep_name] = self._create_venv_callable(
                    dep_data,
                    namespace=namespace,
                )
                # Treat venv functions as atomic; do not recurse into their deps.
                continue

            # Handle in-process dependencies: exec puts raw function in namespace.
            # We call _create_in_process_callable to exec the function, but we
            # DON'T overwrite namespace with the proxy - the raw function stays
            # for inter-function calls, decorators, and introspection.
            self._create_in_process_callable(
                dep_data,
                namespace=namespace,
            )
            # Note: proxy is discarded; namespace[dep_name] remains the raw function

            nested = dep_data.get("depends_on") or []
            if isinstance(nested, list):
                for child in nested:
                    if isinstance(child, str) and child and child not in visited:
                        q.append(child)

    def _inject_callables_for_functions(
        self,
        func_rows: List[Dict[str, Any]],
        *,
        namespace: Dict[str, Any],
    ) -> List[Callable[..., Any]]:
        """Convert function records into callables and return proxies to caller.

        For in-process functions, the raw function (from exec) remains in the
        namespace for inter-function calls, decorators, and introspection.
        The returned proxies provide state mode control (.stateful/.stateless/.read_only).

        For venv functions, the proxy is placed in namespace (no raw function exists).
        """
        callables: List[Callable[..., Any]] = []
        visited: Set[str] = set()

        for func_data in func_rows:
            name = func_data.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("Function record missing valid 'name'")

            # Skip primitives (no stored implementation; names often contain dots).
            if func_data.get("is_primitive") is True:
                continue

            # Check if we've already processed this function (e.g., duplicate in results)
            if name in visited:
                # Already exec'd - just create a new proxy wrapping existing raw fn
                if func_data.get("venv_id") is not None:
                    fn = self._create_venv_callable(func_data, namespace=namespace)
                else:
                    raw_fn = namespace.get(name)
                    if callable(raw_fn):
                        fn = _InProcessFunctionProxy(
                            function_manager=self,
                            func_data=func_data,
                            namespace=namespace,
                            raw_callable=raw_fn,
                        )
                    else:
                        # Shouldn't happen, but fallback to full creation
                        fn = self._create_in_process_callable(
                            func_data,
                            namespace=namespace,
                        )
                callables.append(fn)
                continue

            visited.add(name)  # Prevent cycles from re-injecting the root function.
            self._inject_dependencies(func_data, namespace=namespace, visited=visited)

            # Create callable for the root function.
            if func_data.get("venv_id") is not None:
                # Venv: proxy goes in namespace (only way to call them)
                fn = self._create_venv_callable(func_data, namespace=namespace)
                namespace[name] = fn
            else:
                # In-process: exec puts raw function in namespace, return proxy to caller
                # DON'T overwrite namespace - raw function stays for internal use
                fn = self._create_in_process_callable(func_data, namespace=namespace)
                # Note: namespace[name] remains the raw function from exec()

            callables.append(fn)

        return callables

    # 2. Listing -------------------------------------------------------- #

    @functools.wraps(BaseFunctionManager.list_functions, updated=())
    def list_functions(
        self,
        *,
        include_implementations: bool = False,
        return_callable: bool = False,
        namespace: Optional[Dict[str, Any]] = None,
        also_return_metadata: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        if also_return_metadata and not return_callable:
            raise ValueError("also_return_metadata requires return_callable=True")

        if return_callable and namespace is None:
            raise ValueError("namespace required when return_callable=True")

        # Always build metadata in the existing shape for backwards-compat.
        metadata: Dict[str, Dict[str, Any]] = {}
        logs = unify.get_logs(
            context=self._compositional_ctx,
            exclude_fields=list_private_fields(self._compositional_ctx),
        )

        func_rows: List[Dict[str, Any]] = []
        for log in logs:
            ent = log.entries
            name = ent.get("name")
            if not isinstance(name, str):
                continue
            func_rows.append(ent)

            data: Dict[str, Any] = {
                "function_id": ent.get("function_id"),
                "language": ent.get(
                    "language",
                    "python",
                ),  # Default for backward compat
                "argspec": ent.get("argspec"),
                "docstring": ent.get("docstring", ""),
                "guidance_ids": ent.get("guidance_ids", []),
                "verify": ent.get("verify", True),
                "venv_id": ent.get("venv_id"),
            }
            if include_implementations:
                data["implementation"] = ent.get("implementation")
            metadata[name] = data

        if not return_callable:
            return metadata

        assert namespace is not None  # validated above
        callables_list = self._inject_callables_for_functions(
            func_rows,
            namespace=namespace,
        )
        callables_map = {
            row["name"]: cb
            for row, cb in zip(func_rows, callables_list)
            if isinstance(row.get("name"), str)
        }

        if also_return_metadata:
            return {"callables": callables_map, "metadata": metadata}  # type: ignore[return-value]

        return callables_map  # type: ignore[return-value]

    @functools.wraps(BaseFunctionManager.get_precondition, updated=())
    def get_precondition(self, *, function_name: str) -> Optional[Dict[str, Any]]:
        logs = unify.get_logs(
            context=self._compositional_ctx,
            filter=f"name == '{function_name}'",
            limit=1,
            exclude_fields=list_private_fields(self._compositional_ctx),
        )
        if not logs:
            return None

        return logs[0].entries.get("precondition")

    # 3. Deletion ------------------------------------------------------- #

    @functools.wraps(BaseFunctionManager.delete_function, updated=())
    def delete_function(
        self,
        *,
        function_id: Union[int, List[int]],
        delete_dependents: bool = True,
    ) -> Dict[str, str]:
        """
        Delete one or more functions and optionally their dependents in a single batch operation.

        Args:
            function_id: Function ID (int) or list of function IDs to delete.
            delete_dependents: If True, also delete all functions that depend on target(s).

        Returns:
            Dictionary mapping function names to "deleted" or "already_deleted".
        """
        # Normalize to list
        function_ids = [function_id] if isinstance(function_id, int) else function_id

        if not function_ids:
            return {}

        # Handle single function optimization
        if len(function_ids) == 1:
            log = self._get_log_by_function_id(
                function_id=function_ids[0],
                raise_if_missing=False,
            )
            if log is None:
                return {f"function_{function_ids[0]}": "already_deleted"}

            target_name = log.entries["name"]
            ids_to_delete = {function_ids[0]}
            log_ids_to_delete = [log.id]
            results = {target_name: "deleted"}
        else:
            # Multiple functions - build from all logs
            all_logs = unify.get_logs(
                context=self._compositional_ctx,
                exclude_fields=list_private_fields(self._compositional_ctx),
            )

            id_to_log = {lg.entries["function_id"]: lg for lg in all_logs}
            id_to_name = {
                lg.entries["function_id"]: lg.entries["name"] for lg in all_logs
            }

            ids_to_delete = set(function_ids)
            target_names = {
                id_to_name[fid] for fid in function_ids if fid in id_to_name
            }

            if not target_names:
                return {}

            log_ids_to_delete = [
                id_to_log[fid].id for fid in function_ids if fid in id_to_log
            ]
            results = {
                id_to_name[fid]: "deleted" for fid in function_ids if fid in id_to_name
            }

            function_deps = {
                lg.entries["function_id"]: set(lg.entries.get("depends_on", []))
                for lg in all_logs
            }

        if delete_dependents:
            # Get all logs if not already loaded
            if len(function_ids) == 1:
                all_logs = unify.get_logs(
                    context=self._compositional_ctx,
                    exclude_fields=list_private_fields(self._compositional_ctx),
                )
                id_to_log = {lg.entries["function_id"]: lg for lg in all_logs}
                id_to_name = {
                    lg.entries["function_id"]: lg.entries["name"] for lg in all_logs
                }
                function_deps = {
                    lg.entries["function_id"]: set(lg.entries.get("depends_on", []))
                    for lg in all_logs
                }
                target_names = {target_name}

            # BFS to find all transitive dependents
            to_process = set(target_names)
            processed = set()

            while to_process:
                current_name = to_process.pop()
                if current_name in processed:
                    continue
                processed.add(current_name)

                for fid, deps in function_deps.items():
                    if current_name in deps and fid not in ids_to_delete:
                        ids_to_delete.add(fid)
                        if fid in id_to_log:
                            log_ids_to_delete.append(id_to_log[fid].id)
                            dep_name = id_to_name[fid]
                            results[dep_name] = "deleted"
                            to_process.add(dep_name)

        # Batch delete all functions
        if log_ids_to_delete:
            unify.delete_logs(
                context=self._compositional_ctx,
                logs=log_ids_to_delete,
            )

        return results

    # 4. Filter --------------------------------------------------------- #

    @functools.wraps(BaseFunctionManager.filter_functions, updated=())
    def filter_functions(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        include_implementations: bool = True,
        return_callable: bool = False,
        namespace: Optional[Dict[str, Any]] = None,
        also_return_metadata: bool = False,
    ) -> List[Dict[str, Any]]:
        if also_return_metadata and not return_callable:
            raise ValueError("also_return_metadata requires return_callable=True")

        if return_callable and namespace is None:
            raise ValueError("namespace required when return_callable=True")

        normalized = normalize_filter_expr(filter)
        # The underlying Unify backend returns 404 when a context hasn't been created yet.
        # In tests and fresh projects, contexts are created lazily, so we retry briefly and
        # then treat missing context as "no functions" rather than crashing the Actor.
        import time as _time

        rows: List[Dict[str, Any]] = []
        last_exc: Exception | None = None
        for delay in (0.0, 0.05, 0.15):
            if delay:
                _time.sleep(delay)
            try:
                logs = unify.get_logs(
                    context=self._compositional_ctx,
                    filter=normalized,
                    offset=offset,
                    limit=limit,
                    exclude_fields=list_private_fields(self._compositional_ctx),
                )
                rows = [lg.entries for lg in logs]
                last_exc = None
                break
            except _UnifyRequestError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 404:
                    last_exc = e
                    continue
                raise
            except Exception as e:
                last_exc = e
                break

        # If we still see 404 after retries, treat as empty library.
        if isinstance(last_exc, _UnifyRequestError):
            status = getattr(getattr(last_exc, "response", None), "status_code", None)
            if status == 404:
                rows = []
        elif last_exc is not None:
            raise last_exc

        if not return_callable:
            # Strip implementations if not requested (reduces payload size)
            if not include_implementations:
                rows = [
                    {k: v for k, v in row.items() if k != "implementation"}
                    for row in rows
                ]
            return rows

        assert namespace is not None  # validated above
        # Note: rows must contain implementations for callable injection to work.
        callables_list = self._inject_callables_for_functions(rows, namespace=namespace)
        if also_return_metadata:
            # Strip implementations from metadata if not requested
            metadata_rows = rows
            if not include_implementations:
                metadata_rows = [
                    {k: v for k, v in row.items() if k != "implementation"}
                    for row in rows
                ]
            return {"callables": callables_list, "metadata": metadata_rows}  # type: ignore[return-value]
        return callables_list  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    #  Accessors and disk → context sync                                 #
    # ------------------------------------------------------------------ #

    def get_function_file_path(self, name: str) -> Optional[str]:
        p = self._function_path(name)
        return str(p) if p is not None else None

    def list_function_files(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        try:
            logs = unify.get_logs(
                context=self._compositional_ctx,
                exclude_fields=list_private_fields(self._compositional_ctx),
            )
            for lg in logs:
                nm = lg.entries.get("name")
                if not isinstance(nm, str):
                    continue
                p = self._function_path(nm)
                if p is not None:
                    out[nm] = str(p)
        except Exception:
            pass
        return out

    def sync_from_disk(self, *, prefer_file_when_newer: bool = True) -> List[str]:
        """
        Reconcile function files under functions/ with the context rows.

        Policy: if the on-disk file differs from the stored implementation, update
        the context to the file contents. Returns the list of function names updated.
        """
        updated: List[str] = []
        if self._functions_dir is None:
            return updated
        try:
            # Build a map of name→(log_id, impl)
            rows = unify.get_logs(
                context=self._compositional_ctx,
                exclude_fields=list_private_fields(self._compositional_ctx),
            )
            name_to_log: Dict[str, Tuple[int, str]] = {}
            for lg in rows:
                nm = lg.entries.get("name")
                if isinstance(nm, str):
                    name_to_log[nm] = (lg.id, lg.entries.get("implementation") or "")

            for name, (log_id, stored_impl) in name_to_log.items():
                p = self._function_path(name)
                if p is None or not p.exists():
                    continue
                try:
                    file_text = p.read_text(encoding="utf-8")
                except Exception:
                    continue
                if file_text.strip() == (stored_impl or "").strip():
                    # Ensure it's registered as protected
                    self._register_function_file(name, p)
                    continue

                # Parse and validate file to rebuild signature/docstring/depends_on
                try:
                    nm2, tree, node, _src = self._parse_implementation(file_text)
                    if nm2 != name:
                        # Skip mismatched names; keep 1:1 name↔file mapping
                        continue
                    namespace = create_base_globals()
                    exec(file_text, namespace)
                    fn_obj = namespace[name]
                    signature = str(inspect.signature(fn_obj))
                    docstring = inspect.getdoc(fn_obj) or ""
                    depends_on = list(self._collect_function_calls(node))
                    embedding_text = f"Function Name: {name}\nSignature: {signature}\nDocstring: {docstring}"
                    # Update unify row
                    unify.update_logs(
                        logs=[log_id],
                        context=self._compositional_ctx,
                        entries={
                            "argspec": signature,
                            "docstring": docstring,
                            "implementation": file_text,
                            "depends_on": depends_on,
                            "embedding_text": embedding_text,
                        },
                        overwrite=True,
                    )
                    # Ensure it's registered as protected
                    self._register_function_file(name, p)
                    updated.append(name)
                except Exception:
                    continue
        except Exception:
            return updated
        return updated

    # 5. Semantic Search ------------------------------------------------ #
    @functools.wraps(BaseFunctionManager.search_functions, updated=())
    def search_functions(
        self,
        *,
        query: str,
        n: int = 5,
        include_implementations: bool = True,
        return_callable: bool = False,
        namespace: Optional[Dict[str, Any]] = None,
        also_return_metadata: bool = False,
        include_primitives: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search for functions by semantic similarity to a natural-language query.

        Args:
            query: Natural-language text describing the desired function(s).
            n: Number of similar results to return.
            include_implementations: If True (default), include full source code.
            include_primitives: If True (default), sync and include primitives
                in the search results alongside user-defined functions.

        Returns:
            Up to n results ordered by similarity, including both user functions
            and primitives (if include_primitives=True).
        """
        if also_return_metadata and not return_callable:
            raise ValueError("also_return_metadata requires return_callable=True")

        if return_callable and namespace is None:
            raise ValueError("namespace required when return_callable=True")

        allowed_fields = list(Function.model_fields.keys())

        # Search user-defined functions in the Compositional context
        compositional_rows = table_search_top_k(
            context=self._compositional_ctx,
            references={"embedding_text": query},
            k=n,
            allowed_fields=allowed_fields,
            unique_id_field="function_id",
        )

        if not include_primitives:
            results = compositional_rows[:n]
        else:
            # Sync and search primitives
            self.sync_primitives()

            primitive_rows = table_search_top_k(
                context=self._primitives_ctx,
                references={"embedding_text": query},
                k=n,
                allowed_fields=allowed_fields,
                unique_id_field="function_id",
            )

            # Merge and sort by the private score column (lower distance = better match)
            all_rows = compositional_rows + primitive_rows
            sort_key: str | None = None
            for row in all_rows:
                for key in row.keys():
                    if key.startswith("_"):
                        sort_key = key
                        break
                if sort_key:
                    break
            if sort_key:
                all_rows.sort(key=lambda r: r.get(sort_key, float("inf")))
            results = all_rows[:n]

        if not return_callable:
            # Strip implementations if not requested (reduces payload size)
            if not include_implementations:
                results = [
                    {k: v for k, v in row.items() if k != "implementation"}
                    for row in results
                ]
            return results

        assert namespace is not None  # validated above

        # Only materialize non-primitive records as callables.
        # Note: exec_rows must contain implementations for callable injection to work.
        exec_rows = [
            r
            for r in results
            if isinstance(r, dict) and r.get("is_primitive") is not True
        ]
        callables_list = self._inject_callables_for_functions(
            exec_rows,
            namespace=namespace,
        )

        if also_return_metadata:
            # Strip implementations from metadata if not requested
            metadata_rows = exec_rows
            if not include_implementations:
                metadata_rows = [
                    {k: v for k, v in row.items() if k != "implementation"}
                    for row in exec_rows
                ]
            return {"callables": callables_list, "metadata": metadata_rows}  # type: ignore[return-value]

        return callables_list  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    #  Inverse linkage: Functions → Guidance                              #
    # ------------------------------------------------------------------ #

    def _guidance_context(self) -> str:
        ctxs = unify.get_active_context()
        read_ctx = ctxs.get("read")
        return f"{read_ctx}/Guidance" if read_ctx else "Guidance"

    def _get_guidance_ids_for_function(self, *, function_id: int) -> List[int]:
        # Prefer reading from the function row if present
        try:
            log = self._get_log_by_function_id(function_id=function_id)
            gids = log.entries.get("guidance_ids") or []
            if isinstance(gids, list) and gids:
                return [int(g) for g in gids]
        except Exception:
            pass

        # Fallback: scan Guidance rows that reference this function via function_ids
        gctx = self._guidance_context()
        try:
            rows = unify.get_logs(
                context=gctx,
                filter=f"{int(function_id)} in function_ids",
                exclude_fields=list_private_fields(gctx),
            )
            return [
                int(r.entries.get("guidance_id"))
                for r in rows
                if r.entries.get("guidance_id") is not None
            ]
        except Exception:
            return []

    def _get_guidance_for_function(
        self,
        *,
        function_id: int,
        include_images: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return guidance records linked to the function.

        Each dict includes: guidance_id, title, content, images (optional).
        """
        gids = self._get_guidance_ids_for_function(function_id=function_id)
        if not gids:
            return []
        if limit is not None:
            try:
                limit = int(limit)
            except Exception:
                limit = None
            if isinstance(limit, int) and limit >= 0:
                gids = gids[:limit]
        cond = " or ".join(f"guidance_id == {int(g)}" for g in gids)
        gctx = self._guidance_context()
        rows = unify.get_logs(
            context=gctx,
            filter=cond or "False",
            exclude_fields=list_private_fields(gctx),
        )
        out: List[Dict[str, Any]] = []
        for lg in rows:
            ent = lg.entries
            rec: Dict[str, Any] = {
                "guidance_id": ent.get("guidance_id"),
                "title": ent.get("title"),
                "content": ent.get("content"),
            }
            if include_images:
                rec["images"] = ent.get("images") or []
            out.append(rec)
        return out

    def _get_image_handles_for_function_guidance(
        self,
        *,
        function_id: int,
        limit: Optional[int] = None,
    ) -> List[ImageHandle]:
        """Return ImageHandle objects for images referenced by guidance linked to the function."""
        guids = self._get_guidance_for_function(
            function_id=function_id,
            include_images=True,
        )
        image_ids: List[int] = []
        for g in guids:
            imgs = g.get("images") or []
            # Support either raw list (ImageRefs) or a dict with root
            if isinstance(imgs, dict) and "root" in imgs:
                imgs = imgs.get("root") or []
            if not isinstance(imgs, list):
                continue
            for ref in imgs:
                try:
                    if isinstance(ref, dict):
                        # AnnotatedImageRef shape: {"raw_image_ref": {"image_id": X}, "annotation": ...}
                        if "raw_image_ref" in ref and isinstance(
                            ref["raw_image_ref"],
                            dict,
                        ):
                            iid = int(ref["raw_image_ref"].get("image_id"))
                            image_ids.append(iid)
                        elif "image_id" in ref:
                            image_ids.append(int(ref.get("image_id")))
                    else:
                        # If objects leaked through, try attribute access
                        iid = getattr(
                            getattr(ref, "raw_image_ref", ref),
                            "image_id",
                            None,
                        )
                        if iid is not None:
                            image_ids.append(int(iid))
                except Exception:
                    continue
        # Preserve order while de-duplicating
        image_ids = list(dict.fromkeys(image_ids))
        if limit is not None:
            try:
                limit = int(limit)
            except Exception:
                limit = None
            if isinstance(limit, int) and limit >= 0:
                image_ids = image_ids[:limit]

        im = ManagerRegistry.get_image_manager()
        return im.get_images(image_ids)

    def _attach_guidance_images_for_function_to_context(
        self,
        *,
        function_id: int,
        limit: Optional[int] = 3,
    ) -> Dict[str, Any]:
        """Attach images referenced by related guidance into the loop context.

        Returns a dict with keys:
            attached_count: int
            images: list of { meta: {...}, image: <base64> }
        """
        handles = self._get_image_handles_for_function_guidance(
            function_id=function_id,
            limit=limit,
        )
        images: List[Dict[str, Any]] = []
        for h in handles:
            try:
                raw_bytes = h.raw()
            except Exception:
                continue
            import base64

            b64 = base64.b64encode(raw_bytes).decode("utf-8")
            images.append(
                {
                    "meta": {
                        "image_id": int(h.image_id),
                        "caption": h.caption,
                        "timestamp": getattr(h.timestamp, "isoformat", lambda: "")(),
                    },
                    "image": b64,
                },
            )
        return {"attached_count": len(images), "images": images}

    # ------------------------------------------------------------------ #
    #  Virtual Environment Management                                    #
    # ------------------------------------------------------------------ #

    def _safe_get_venv_logs(
        self,
        *,
        filter: Optional[str] = None,
        limit: Optional[int] = None,
        exclude_fields: Optional[List[str]] = None,
        from_fields: Optional[List[str]] = None,
    ) -> List[unify.Log]:
        """Best-effort venv reads; treat missing contexts as empty."""
        import time as _time

        last_exc: Exception | None = None
        for delay in (0.0, 0.05, 0.15):
            if delay:
                _time.sleep(delay)
            try:
                return unify.get_logs(
                    context=self._venvs_ctx,
                    filter=filter,
                    limit=limit,
                    exclude_fields=exclude_fields,
                    from_fields=from_fields,
                )
            except _UnifyRequestError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 404:
                    last_exc = e
                    continue
                raise
            except Exception as e:
                last_exc = e
                break

        if isinstance(last_exc, _UnifyRequestError):
            status = getattr(getattr(last_exc, "response", None), "status_code", None)
            if status == 404:
                return []
        if last_exc is not None:
            raise last_exc
        return []

    def add_venv(self, *, venv: str) -> int:
        """
        Add a new virtual environment configuration.

        Args:
            venv: The pyproject.toml content as a string.

        Returns:
            The auto-assigned venv_id.
        """
        result = unity_create_logs(
            context=self._venvs_ctx,
            entries=[{"venv": venv}],
            add_to_all_context=self.include_in_multi_assistant_table,
        )
        # unity_create_logs can return either a dict or a list of Log objects
        if isinstance(result, list) and len(result) > 0:
            # List of Log objects - can extract venv_id directly from entries
            log = result[0]
            if hasattr(log, "entries"):
                venv_id = log.entries.get("venv_id")
                if venv_id is not None:
                    return venv_id
        elif isinstance(result, dict):
            log_ids = result.get("log_event_ids", [])
            if log_ids:
                logs = self._safe_get_venv_logs(
                    filter=f"id == {log_ids[0]}",
                    limit=1,
                )
                if logs and hasattr(logs[0], "entries"):
                    venv_id = logs[0].entries.get("venv_id")
                    if venv_id is not None:
                        return venv_id
        raise RuntimeError("Failed to retrieve venv_id after creation")

    def get_venv(self, *, venv_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a virtual environment by its ID.

        Args:
            venv_id: The unique identifier of the virtual environment.

        Returns:
            Dict with venv_id and venv content, or None if not found.
        """
        logs = self._safe_get_venv_logs(
            filter=f"venv_id == {venv_id}",
            limit=1,
            exclude_fields=list_private_fields(self._venvs_ctx),
        )
        if not logs:
            return None
        return logs[0].entries

    def list_venvs(self) -> List[Dict[str, Any]]:
        """
        List all virtual environments.

        Returns:
            List of dicts, each with venv_id and venv content.
        """
        logs = self._safe_get_venv_logs(
            exclude_fields=list_private_fields(self._venvs_ctx),
        )
        return [lg.entries for lg in logs]

    def delete_venv(self, *, venv_id: int) -> bool:
        """
        Delete a virtual environment by its ID.

        Functions referencing this venv will have their venv_id set to None
        (falling back to the default environment) via the foreign key cascade.

        Args:
            venv_id: The unique identifier of the virtual environment.

        Returns:
            True if deleted, False if not found.
        """
        logs = self._safe_get_venv_logs(
            filter=f"venv_id == {venv_id}",
            limit=1,
        )
        if not logs:
            return False
        unify.delete_logs(
            context=self._venvs_ctx,
            logs=[logs[0].id],
        )
        return True

    def update_venv(self, *, venv_id: int, venv: str) -> bool:
        """
        Update the content of an existing virtual environment.

        Args:
            venv_id: The unique identifier of the virtual environment.
            venv: The new pyproject.toml content.

        Returns:
            True if updated, False if not found.
        """
        logs = self._safe_get_venv_logs(
            filter=f"venv_id == {venv_id}",
            limit=1,
        )
        if not logs:
            return False
        unify.update_logs(
            context=self._venvs_ctx,
            logs=[logs[0].id],
            entries={"venv": venv},
            overwrite=True,
        )
        return True

    def set_function_venv(
        self,
        *,
        function_id: int,
        venv_id: Optional[int],
    ) -> bool:
        """
        Set the virtual environment for a function.

        Args:
            function_id: The function to update.
            venv_id: The venv_id to associate, or None for default environment.

        Returns:
            True if updated, False if function not found.
        """
        log = self._get_log_by_function_id(
            function_id=function_id,
            raise_if_missing=False,
        )
        if log is None:
            return False
        unify.update_logs(
            context=self._compositional_ctx,
            logs=[log.id],
            entries={"venv_id": venv_id},
            overwrite=True,
        )
        return True

    def get_function_venv(self, *, function_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the virtual environment associated with a function.

        Args:
            function_id: The function to query.

        Returns:
            The venv dict if the function has one, None if using default,
            or raises ValueError if function not found.
        """
        log = self._get_log_by_function_id(
            function_id=function_id,
            raise_if_missing=True,
        )
        venv_id = log.entries.get("venv_id")
        if venv_id is None:
            return None
        return self.get_venv(venv_id=venv_id)

    # ------------------------------------------------------------------ #
    #  Virtual Environment Execution Support                             #
    # ------------------------------------------------------------------ #

    def _get_venv_base_dir(self) -> Path:
        """Get the base directory for all custom venvs.

        The path includes the Unify context name to ensure isolation between
        different assistants/users and during parallel test runs.
        """
        # Get current context for isolation
        ctx = unify.get_active_context()
        ctx_name = ctx.get("read") or ctx.get("write") or "default"
        # Sanitize context name for filesystem use
        safe_ctx = ctx_name.replace("/", "_").replace("\\", "_")
        return Path.home() / ".unity" / "venvs" / safe_ctx

    def _get_venv_dir(self, venv_id: int) -> Path:
        """Get the directory for a specific venv."""
        return self._get_venv_base_dir() / str(venv_id)

    def _get_venv_python(self, venv_id: int) -> Path:
        """Get the path to the Python interpreter for a venv."""
        return self._get_venv_dir(venv_id) / ".venv" / "bin" / "python"

    def _get_venv_runner_path(self, venv_id: int) -> Path:
        """Get the path to the runner script for a venv."""
        return self._get_venv_dir(venv_id) / "venv_runner.py"

    def _get_runner_script_content(self) -> str:
        """Get the content of the standalone runner script."""
        runner_path = Path(__file__).parent / "venv_runner.py"
        return runner_path.read_text()

    def is_venv_ready(self, *, venv_id: int) -> bool:
        """
        Check if a virtual environment is ready for execution.

        Args:
            venv_id: The venv to check.

        Returns:
            True if the venv exists and is synced, False otherwise.
        """
        venv_data = self.get_venv(venv_id=venv_id)
        if venv_data is None:
            return False

        venv_dir = self._get_venv_dir(venv_id)
        pyproject_path = venv_dir / "pyproject.toml"
        python_path = self._get_venv_python(venv_id)
        runner_path = self._get_venv_runner_path(venv_id)

        # Check if all required files exist
        if not pyproject_path.exists() or not python_path.exists():
            return False

        # Check if pyproject.toml content matches (normalize line endings)
        stored_content = venv_data["venv"].strip()
        disk_content = pyproject_path.read_text().strip()
        if disk_content != stored_content:
            return False

        # Check if runner script exists
        if not runner_path.exists():
            return False

        return True

    async def prepare_venv(self, *, venv_id: int) -> Path:
        """
        Ensure a virtual environment is created and synced.

        This method is idempotent - if the venv already exists and is up-to-date,
        it returns immediately. Otherwise, it creates/updates the venv.

        Args:
            venv_id: The venv to prepare.

        Returns:
            Path to the Python interpreter in the venv.

        Raises:
            ValueError: If the venv_id does not exist.
            RuntimeError: If venv creation fails.
        """
        venv_data = self.get_venv(venv_id=venv_id)
        if venv_data is None:
            raise ValueError(f"VirtualEnv with ID {venv_id} not found")

        venv_content = venv_data["venv"]
        venv_dir = self._get_venv_dir(venv_id)
        pyproject_path = venv_dir / "pyproject.toml"
        python_path = self._get_venv_python(venv_id)
        runner_path = self._get_venv_runner_path(venv_id)

        # Check if already ready
        needs_sync = False
        if pyproject_path.exists():
            if pyproject_path.read_text().strip() != venv_content.strip():
                needs_sync = True
                logger.info(f"Venv {venv_id}: pyproject.toml changed, re-syncing")
        else:
            needs_sync = True
            logger.info(f"Venv {venv_id}: creating new venv")

        if needs_sync or not python_path.exists():
            # Create directory and write pyproject.toml
            venv_dir.mkdir(parents=True, exist_ok=True)
            pyproject_path.write_text(venv_content)

            # Run uv sync
            logger.info(f"Venv {venv_id}: running 'uv sync'...")
            import asyncio
            import shutil as _shutil
            import sys as _sys

            uv_bin = _shutil.which("uv")
            if uv_bin is None:
                try:
                    # NOTE: don't call `.resolve()` here. In venvs, `sys.executable` is
                    # often a symlink to the system Python, and resolving it would lose
                    # the venv bin directory (where `uv` is installed).
                    candidate = Path(_sys.executable).parent / "uv"
                    if candidate.exists():
                        uv_bin = str(candidate)
                except Exception:
                    uv_bin = None

            if uv_bin is None:
                raise RuntimeError(
                    "Failed to sync venv because the 'uv' executable was not found. "
                    "Install uv (recommended) or ensure it is available on PATH.",
                )

            process = await asyncio.create_subprocess_exec(
                uv_bin,
                "sync",
                cwd=str(venv_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else stdout.decode()
                raise RuntimeError(
                    f"Failed to sync venv {venv_id}: {error_msg}",
                )

            logger.info(f"Venv {venv_id}: sync complete")

        # Ensure runner script is present and up-to-date
        runner_content = self._get_runner_script_content()
        if not runner_path.exists() or runner_path.read_text() != runner_content:
            runner_path.write_text(runner_content)
            logger.info(f"Venv {venv_id}: runner script installed")

        return python_path

    async def _handle_rpc_call(
        self,
        path: str,
        kwargs: Dict[str, Any],
        primitives: Optional[Any] = None,
        computer_primitives: Optional[Any] = None,
    ) -> Any:
        """
        Handle an RPC call from a subprocess.

        Args:
            path: The RPC path (e.g., "contacts.ask", "computer.click")
            kwargs: The keyword arguments for the call
            primitives: The Primitives instance for state manager access
            computer_primitives: The ComputerPrimitives instance

        Returns:
            The result of the RPC call
        """
        parts = path.split(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid RPC path: {path}")

        manager_name, method_name = parts

        # Handle computer primitives
        if manager_name == "computer":
            if computer_primitives is None:
                raise RuntimeError("computer_primitives not available")
            method = getattr(computer_primitives, method_name, None)
            if method is None:
                raise AttributeError(
                    f"computer_primitives has no method '{method_name}'",
                )
            # ComputerPrimitives methods are sync, but we run in async context
            if asyncio.iscoroutinefunction(method):
                return await method(**kwargs)
            return method(**kwargs)

        # Handle state manager primitives
        if primitives is None:
            raise RuntimeError("primitives not available")

        manager = getattr(primitives, manager_name, None)
        if manager is None:
            raise AttributeError(f"primitives has no manager '{manager_name}'")

        method = getattr(manager, method_name, None)
        if method is None:
            raise AttributeError(
                f"primitives.{manager_name} has no method '{method_name}'",
            )

        if asyncio.iscoroutinefunction(method):
            return await method(**kwargs)
        return method(**kwargs)

    async def execute_in_venv(
        self,
        *,
        venv_id: int,
        implementation: str,
        call_kwargs: Optional[Dict[str, Any]] = None,
        is_async: bool = True,
        initial_state: Optional[Dict[str, Any]] = None,
        primitives: Optional[Any] = None,
        computer_primitives: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute a function implementation in a custom virtual environment.

        This method:
        1. Ensures the venv is prepared (lazy creation on first use)
        2. Spawns a subprocess with the venv's Python interpreter
        3. Handles bidirectional RPC for primitives and computer_primitives
        4. Returns the result from the subprocess

        Args:
            venv_id: The virtual environment to use.
            implementation: The function source code.
            call_kwargs: Keyword arguments to pass to the function.
            is_async: Whether the function is async (default True).
            initial_state: Optional serialized state to inject before execution.
                Used for read_only mode to inherit state from a persistent session.
            primitives: The Primitives instance for RPC access to state managers.
            computer_primitives: The ComputerPrimitives instance for RPC access.

        Returns:
            Dict with keys: result, error, stdout, stderr

        Raises:
            ValueError: If venv_id does not exist.
            RuntimeError: If execution fails.
        """
        call_kwargs = call_kwargs or {}

        # Ensure venv is ready
        python_path = await self.prepare_venv(venv_id=venv_id)
        runner_path = self._get_venv_runner_path(venv_id)

        # Prepare initial execution request
        execute_payload: Dict[str, Any] = {
            "type": "execute",
            "implementation": implementation,
            "call_kwargs": call_kwargs,
            "is_async": is_async,
        }
        if initial_state is not None:
            execute_payload["initial_state"] = initial_state

        execute_msg = json.dumps(execute_payload) + "\n"

        # Execute in subprocess with bidirectional communication
        # Use start_new_session=True to create a new process group, allowing
        # us to kill all child processes (including multiprocessing workers)
        # with a single os.killpg() call.
        # Note: start_new_session is not supported on Windows
        use_process_group = sys.platform != "win32"
        process = await asyncio.create_subprocess_exec(
            str(python_path),
            str(runner_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=use_process_group,
        )

        # Send initial execution request
        process.stdin.write(execute_msg.encode())
        await process.stdin.drain()

        # Handle bidirectional communication
        stderr_output = []

        async def read_stderr():
            """Read stderr in background."""
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
                stderr_output.append(line.decode())

        stderr_task = asyncio.create_task(read_stderr())

        try:
            while True:
                # Read next message from subprocess
                line = await process.stdout.readline()
                if not line:
                    # Process ended without sending complete message
                    await stderr_task
                    return {
                        "result": None,
                        "error": "Subprocess ended unexpectedly",
                        "stdout": "",
                        "stderr": "".join(stderr_output),
                    }

                try:
                    msg = json.loads(line.decode().strip())
                except json.JSONDecodeError as e:
                    continue  # Skip malformed lines

                msg_type = msg.get("type")

                if msg_type == "rpc_call":
                    # Handle RPC call from subprocess
                    request_id = msg.get("id")
                    path = msg.get("path", "")
                    rpc_kwargs = msg.get("kwargs", {})

                    try:
                        result = await self._handle_rpc_call(
                            path=path,
                            kwargs=rpc_kwargs,
                            primitives=primitives,
                            computer_primitives=computer_primitives,
                        )
                        response = (
                            json.dumps(
                                {
                                    "type": "rpc_result",
                                    "id": request_id,
                                    "result": self._make_json_serializable(result),
                                },
                            )
                            + "\n"
                        )
                    except Exception as e:
                        response = (
                            json.dumps(
                                {
                                    "type": "rpc_error",
                                    "id": request_id,
                                    "error": str(e),
                                },
                            )
                            + "\n"
                        )

                    process.stdin.write(response.encode())
                    await process.stdin.drain()

                elif msg_type == "complete":
                    # Subprocess finished
                    await stderr_task
                    return {
                        "result": msg.get("result"),
                        "error": msg.get("error"),
                        "stdout": msg.get("stdout", ""),
                        "stderr": msg.get("stderr", "") + "".join(stderr_output),
                    }

        except asyncio.CancelledError:
            # Task was cancelled (e.g., Actor.stop() was called)
            # Re-raise after cleanup in finally block
            raise
        except Exception as e:
            return {
                "result": None,
                "error": f"RPC error: {e}",
                "stdout": "",
                "stderr": "".join(stderr_output),
            }
        finally:
            # Cancel stderr reader task
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass

            # Ensure process and all its children are terminated
            if process.returncode is None:
                await self._terminate_process_group(process, use_process_group)

    async def _terminate_process_group(
        self,
        process: asyncio.subprocess.Process,
        use_process_group: bool,
    ) -> None:
        """
        Terminate a subprocess and all its children (process group).

        Sends SIGTERM first for graceful shutdown, then SIGKILL if the process
        doesn't terminate within the timeout.

        Args:
            process: The subprocess to terminate.
            use_process_group: Whether the process was started with start_new_session=True.
        """
        try:
            if use_process_group and process.pid is not None:
                # Kill the entire process group (subprocess + all its children)
                try:
                    pgid = os.getpgid(process.pid)
                    # Send SIGTERM for graceful shutdown
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    # Process already dead or no permission
                    pass
            else:
                # Fall back to terminating just the main process
                process.terminate()

            # Wait for process to terminate
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                # Process didn't terminate gracefully, force kill
                if use_process_group and process.pid is not None:
                    try:
                        pgid = os.getpgid(process.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
                else:
                    process.kill()
                # Wait for kill to complete
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
        except Exception:
            # Best effort cleanup - don't let cleanup errors propagate
            pass

    async def execute_function(
        self,
        *,
        function_name: str,
        call_kwargs: Optional[Dict[str, Any]] = None,
        target_venv_id: Optional[int] = ...,
        state_mode: Literal["stateful", "read_only", "stateless"] = "stateless",
        session_id: int = 0,
        venv_pool: Optional["VenvPool"] = None,
        shell_pool: Optional["ShellPool"] = None,
        primitives: Optional[Any] = None,
        computer_primitives: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute a stored function by name with optional state mode overrides.

        This method looks up a function by name from the function table and
        executes it. It automatically routes to the appropriate executor based
        on the function's language (Python vs shell).

        State modes:
        - "stateless" (default): Fresh subprocess with no inherited state. Pure
          function behavior. Backward compatible with previous behavior.
        - "stateful": Uses persistent pool connection. Variables from previous
          executions persist. Requires venv_pool (Python) or shell_pool (shell).
        - "read_only": Reads current state from pool but executes in ephemeral
          subprocess. Changes are NOT persisted. Useful for "what-if" exploration.

        Args:
            function_name: Name of the function to execute.
            call_kwargs: Keyword arguments to pass to the function.
            target_venv_id: Override the execution environment (Python only):
                - ... (Ellipsis): Use the function's stored venv_id (default)
                - None: Execute in the default Python environment
                - int: Execute in this specific venv_id
            state_mode: How to handle global state ("stateful", "read_only", "stateless").
            session_id: The session ID within the pool (default 0). Multiple sessions
                allow independent stateful execution contexts.
                Only applies to stateful/read_only modes.
            venv_pool: VenvPool for stateful/read_only modes with Python venv functions.
            shell_pool: ShellPool for stateful/read_only modes with shell functions.
            primitives: Primitives instance for RPC access to state managers.
            computer_primitives: ComputerPrimitives instance for browser/desktop RPC.

        Returns:
            Dict with keys: result, error, stdout, stderr

        Raises:
            ValueError: If the function doesn't exist or has no implementation.
            ValueError: If state_mode requires a pool but none is provided.
        """
        # Look up function by name
        func_data = self._get_function_data_by_name(name=function_name)
        if func_data is None:
            raise ValueError(f"Function '{function_name}' not found")

        implementation = func_data.get("implementation")
        if not isinstance(implementation, str) or not implementation.strip():
            raise ValueError(f"Function '{function_name}' has no implementation")

        # Check language and route appropriately
        language = func_data.get("language", "python")

        if language == "python":
            return await self._execute_python_function(
                func_data=func_data,
                implementation=implementation,
                call_kwargs=call_kwargs,
                target_venv_id=target_venv_id,
                state_mode=state_mode,
                session_id=session_id,
                venv_pool=venv_pool,
                primitives=primitives,
                computer_primitives=computer_primitives,
            )
        elif language in ("bash", "zsh", "sh", "powershell"):
            return await self._execute_shell_function(
                func_data=func_data,
                implementation=implementation,
                call_kwargs=call_kwargs,
                state_mode=state_mode,
                session_id=session_id,
                shell_pool=shell_pool,
                primitives=primitives,
                computer_primitives=computer_primitives,
            )
        else:
            raise ValueError(f"Unsupported function language: {language}")

    async def _execute_python_function(
        self,
        *,
        func_data: Dict[str, Any],
        implementation: str,
        call_kwargs: Optional[Dict[str, Any]],
        target_venv_id: Optional[int],
        state_mode: Literal["stateful", "read_only", "stateless"],
        session_id: int,
        venv_pool: Optional["VenvPool"],
        primitives: Optional[Any],
        computer_primitives: Optional[Any],
    ) -> Dict[str, Any]:
        """Execute a Python function with venv and state mode support."""
        # Strip @custom_function decorators (not available in subprocess runner)
        implementation = _strip_custom_function_decorators(implementation)

        # Determine execution target venv
        if target_venv_id is ...:
            # Use function's default venv_id
            exec_venv_id = func_data.get("venv_id")
        else:
            # User override
            exec_venv_id = target_venv_id

        # Determine if function is async
        is_async = "async def" in implementation

        call_kwargs = call_kwargs or {}

        # Handle execution based on venv and state_mode
        if exec_venv_id is None:
            # No venv - execute in default environment with state_mode support
            return await self._execute_in_default_env(
                implementation=implementation,
                call_kwargs=call_kwargs,
                is_async=is_async,
                state_mode=state_mode,
                session_id=session_id,
                primitives=primitives,
                computer_primitives=computer_primitives,
            )

        # Venv execution - state_mode matters
        venv_id = int(exec_venv_id)

        if state_mode == "stateful":
            # Use persistent connection via VenvPool
            if venv_pool is None:
                raise ValueError(
                    "state_mode='stateful' requires venv_pool for venv functions. "
                    "Either provide venv_pool or use state_mode='stateless'.",
                )
            return await venv_pool.execute_in_venv(
                venv_id=venv_id,
                implementation=implementation,
                call_kwargs=call_kwargs,
                is_async=is_async,
                session_id=session_id,
                primitives=primitives,
                computer_primitives=computer_primitives,
                function_manager=self,
            )

        elif state_mode == "read_only":
            # Get state from persistent connection, execute in ephemeral subprocess
            if venv_pool is None:
                raise ValueError(
                    "state_mode='read_only' requires venv_pool to read existing state. "
                    "Either provide venv_pool or use state_mode='stateless'.",
                )
            # Get current state from the persistent connection
            initial_state = await venv_pool.get_connection_state(
                venv_id=venv_id,
                function_manager=self,
                session_id=session_id,
            )
            # Execute in fresh subprocess with that state (not modifying persistent state)
            return await self.execute_in_venv(
                venv_id=venv_id,
                implementation=implementation,
                call_kwargs=call_kwargs,
                is_async=is_async,
                initial_state=initial_state,
                primitives=primitives,
                computer_primitives=computer_primitives,
            )

        else:  # state_mode == "stateless"
            # Fresh subprocess with no inherited state
            return await self.execute_in_venv(
                venv_id=venv_id,
                implementation=implementation,
                call_kwargs=call_kwargs,
                is_async=is_async,
                primitives=primitives,
                computer_primitives=computer_primitives,
            )

    async def _execute_shell_function(
        self,
        *,
        func_data: Dict[str, Any],
        implementation: str,
        call_kwargs: Optional[Dict[str, Any]],
        state_mode: Literal["stateful", "read_only", "stateless"],
        session_id: int,
        shell_pool: Optional["ShellPool"],
        primitives: Optional[Any],
        computer_primitives: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Execute a shell function with state mode support.

        For shell functions:
        - "stateless": Uses execute_shell_script (fresh subprocess each time)
        - "stateful": Uses ShellPool for persistent sessions
        - "read_only": Not yet implemented (requires state snapshot/restore)
        """
        from .shell_pool import ShellPool  # noqa: F811

        language = func_data.get("language", "bash")

        if state_mode == "stateless":
            # Use existing execute_shell_script (fresh subprocess each time)
            return await self.execute_shell_script(
                implementation=implementation,
                language=language,
                primitives=primitives,
                computer_primitives=computer_primitives,
            )

        elif state_mode == "stateful":
            if shell_pool is None:
                raise ValueError(
                    "state_mode='stateful' requires shell_pool for shell functions. "
                    "Either provide shell_pool or use state_mode='stateless'.",
                )

            # Execute in persistent session via ShellPool
            result = await shell_pool.execute(
                language=language,
                command=implementation,
                session_id=session_id,
            )

            return {
                "result": result.exit_code,  # For shell, "result" is exit code
                "error": result.error,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        elif state_mode == "read_only":
            # Get state from persistent session, execute in ephemeral session
            if shell_pool is None:
                raise ValueError(
                    "state_mode='read_only' requires shell_pool to read existing state. "
                    "Either provide shell_pool or use state_mode='stateless'.",
                )

            from .shell_session import ShellSession

            # Get current state from the persistent session
            session = await shell_pool.get_session(
                language=language,
                session_id=session_id,
            )
            state = await session.snapshot_state()

            # Execute in fresh ephemeral session with restored state
            ephemeral = ShellSession(language=language)
            try:
                await ephemeral.start()
                restore_result = await ephemeral.restore_state(state)
                if restore_result.error:
                    return {
                        "result": -1,
                        "error": f"Failed to restore state: {restore_result.error}",
                        "stdout": "",
                        "stderr": "",
                    }

                # Execute the command in ephemeral session
                result = await ephemeral.execute(implementation)

                return {
                    "result": result.exit_code,
                    "error": result.error,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            finally:
                await ephemeral.close()

    async def _execute_in_default_env(
        self,
        *,
        implementation: str,
        call_kwargs: Dict[str, Any],
        is_async: bool,
        state_mode: Literal["stateful", "read_only", "stateless"] = "stateless",
        session_id: int = 0,
        primitives: Optional[Any] = None,
        computer_primitives: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute a function in the default Python environment (no custom venv).

        This runs the function in-process using the project's Python environment.

        State modes:
        - stateless: Fresh globals each time (pure function behavior)
        - stateful: Persistent globals per session_id (Jupyter-notebook style)
        - read_only: Reads existing state but doesn't persist changes
        """
        from .execution_env import create_base_globals
        import io
        import traceback
        from contextlib import redirect_stdout, redirect_stderr

        # Determine which globals dict to use based on state_mode
        if state_mode == "stateful":
            # Use persistent session globals
            if session_id not in self._in_process_sessions:
                self._in_process_sessions[session_id] = create_base_globals()
            globals_dict = self._in_process_sessions[session_id]
        elif state_mode == "read_only":
            # Copy state from persistent session into fresh globals
            globals_dict = create_base_globals()
            if session_id in self._in_process_sessions:
                # Copy user-defined state (excluding base globals and dunder names)
                base_keys = set(create_base_globals().keys())
                for key, value in self._in_process_sessions[session_id].items():
                    if key not in base_keys and not key.startswith("_"):
                        # Shallow copy - sufficient for most state
                        globals_dict[key] = value
        else:  # stateless
            globals_dict = create_base_globals()

        # Inject primitives (always, since they may change between calls)
        if primitives is not None:
            globals_dict["primitives"] = primitives
        if computer_primitives is not None:
            globals_dict["computer_primitives"] = computer_primitives

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        result = None
        error = None

        try:
            # Extract function name from implementation
            tree = ast.parse(implementation)
            func_name = None
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_name = node.name
                    break

            if not func_name:
                raise ValueError("No function definition found in implementation")

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(implementation, globals_dict)
                fn = globals_dict.get(func_name)
                if fn is None:
                    raise ValueError(f"Function '{func_name}' not found after exec")

                if is_async:
                    result = await fn(**call_kwargs)
                else:
                    result = fn(**call_kwargs)

        except Exception:
            error = traceback.format_exc()

        return {
            "result": self._make_json_serializable(result),
            "error": error,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
        }

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert an object to a JSON-serializable form."""
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        # Handle Pydantic models
        try:
            from pydantic import BaseModel

            if isinstance(obj, BaseModel):
                return self._make_json_serializable(obj.model_dump())
        except ImportError:
            pass
        # For other types, convert to string representation
        return str(obj)

    # ────────────────────────────────────────────────────────────────────────────
    # Shell Script Execution with Primitives Bridge
    # ────────────────────────────────────────────────────────────────────────────

    def _get_shell_interpreter(self, language: str) -> List[str]:
        """
        Get the shell interpreter command for a given language.

        Args:
            language: One of "sh", "bash", "zsh", "powershell"

        Returns:
            List of command args to invoke the interpreter
        """
        interpreters = {
            "sh": ["/bin/sh"],
            "bash": ["/bin/bash"],
            "zsh": ["/bin/zsh"],
            "powershell": ["pwsh", "-NoProfile", "-NonInteractive", "-File"],
        }
        if language not in interpreters:
            raise ValueError(f"Unsupported shell language: {language}")
        return interpreters[language]

    def _get_primitives_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about available primitives for shell script introspection.

        Returns:
            Dict with structure:
            {
                "managers": {
                    "files": {
                        "description": "...",
                        "methods": {
                            "search_files": {"signature": "...", "docstring": "..."},
                            ...
                        }
                    },
                    ...
                }
            }
        """
        from .primitives import get_primitive_sources, MANAGER_METADATA

        result: Dict[str, Dict[str, Any]] = {"managers": {}}

        # Map class names to manager short names
        class_to_manager = {
            "ContactManager": "contacts",
            "TranscriptManager": "transcripts",
            "KnowledgeManager": "knowledge",
            "TaskScheduler": "tasks",
            "SecretManager": "secrets",
            "GuidanceManager": "guidance",
            "WebSearcher": "web",
            "FileManager": "files",
            "ComputerPrimitives": "computer",
        }

        # Collect methods from primitive sources
        for cls, method_names in get_primitive_sources():
            class_name = cls.__name__
            manager_name = class_to_manager.get(class_name)
            if not manager_name:
                continue

            # Get description from MANAGER_METADATA
            metadata = MANAGER_METADATA.get(manager_name, {})
            description = metadata.get("description", "")

            # Extract method signatures
            methods_info: Dict[str, Dict[str, str]] = {}
            for method_name in method_names:
                method = getattr(cls, method_name, None)
                if method is None:
                    continue

                # Unwrap functools.wraps
                fn = method
                while hasattr(fn, "__wrapped__"):
                    fn = fn.__wrapped__

                try:
                    sig = str(inspect.signature(fn))
                except (ValueError, TypeError):
                    sig = "(...)"

                docstring = inspect.getdoc(fn) or ""

                methods_info[method_name] = {
                    "signature": sig,
                    "docstring": docstring,
                }

            result["managers"][manager_name] = {
                "description": description,
                "methods": methods_info,
            }

        return result

    async def execute_shell_script(
        self,
        *,
        implementation: str,
        language: Literal["sh", "bash", "zsh", "powershell"] = "sh",
        call_args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        primitives: Optional[Any] = None,
        computer_primitives: Optional[Any] = None,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Execute a shell script with access to Unity primitives via RPC.

        This method runs a shell script in a subprocess while providing access
        to all Unity primitives (ContactManager, FileManager, etc.) via the
        `unity-primitive` CLI command.

        Shell scripts can call primitives like:
            result=$(unity-primitive files search_files --references '{"query": "budget"}')
            contacts=$(unity-primitive contacts ask --text "Find Alice")

        Args:
            implementation: The shell script source code.
            language: Shell interpreter to use ("sh", "bash", "zsh", "powershell").
            call_args: Optional list of positional arguments to pass to the script.
            env: Optional environment variables to add to the script's environment.
            cwd: Optional working directory for the script.
            primitives: The Primitives instance for RPC access to state managers.
            computer_primitives: The ComputerPrimitives instance for RPC access.
            timeout: Maximum execution time in seconds (default 5 minutes).

        Returns:
            Dict with keys:
            - result: The script's exit code (0 = success)
            - error: Error message if execution failed, None otherwise
            - stdout: Captured stdout from the script
            - stderr: Captured stderr from the script
        """
        call_args = call_args or []

        # Create temporary directory for script and socket
        with tempfile.TemporaryDirectory(prefix="unity_shell_") as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Write script to temporary file
            if language == "powershell":
                script_path = tmpdir_path / "script.ps1"
            else:
                script_path = tmpdir_path / "script.sh"

            script_path.write_text(implementation)
            script_path.chmod(0o755)

            # Create Unix domain socket for RPC
            socket_path = tmpdir_path / "rpc.sock"

            # Get the path to unity-primitive CLI
            shell_runner_path = Path(__file__).parent / "shell_runner.py"

            # Build environment for the subprocess
            script_env = os.environ.copy()
            script_env["UNITY_RPC_SOCKET"] = str(socket_path)
            # Add the shell_runner.py as unity-primitive command
            # We create a wrapper script that invokes python with shell_runner.py
            wrapper_path = tmpdir_path / "unity-primitive"
            python_path = sys.executable
            wrapper_path.write_text(
                f'#!/bin/sh\nexec "{python_path}" "{shell_runner_path}" "$@"\n',
            )
            wrapper_path.chmod(0o755)
            # Prepend tmpdir to PATH so unity-primitive is available
            script_env["PATH"] = f"{tmpdir}:{script_env.get('PATH', '')}"

            # Add user-provided environment variables
            if env:
                script_env.update(env)

            # Set up the RPC server (Unix domain socket)
            server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server_socket.bind(str(socket_path))
            server_socket.listen(5)
            server_socket.setblocking(False)

            # Start the shell script subprocess
            interpreter = self._get_shell_interpreter(language)
            cmd = interpreter + [str(script_path)] + call_args

            use_process_group = sys.platform != "win32"
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=script_env,
                cwd=cwd,
                start_new_session=use_process_group,
            )

            stdout_output: List[str] = []
            stderr_output: List[str] = []

            async def read_stdout():
                """Read stdout in background."""
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    stdout_output.append(line.decode())

            async def read_stderr():
                """Read stderr in background."""
                while True:
                    line = await process.stderr.readline()
                    if not line:
                        break
                    stderr_output.append(line.decode())

            async def handle_rpc_client(client_socket: socket.socket):
                """Handle a single RPC client connection."""
                loop = asyncio.get_event_loop()
                try:
                    # Read request
                    data = b""
                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                loop.sock_recv(client_socket, 4096),
                                timeout=1.0,
                            )
                            if not chunk:
                                break
                            data += chunk
                            if b"\n" in data:
                                break
                        except asyncio.TimeoutError:
                            if process.returncode is not None:
                                break
                            continue

                    if not data:
                        return

                    request = json.loads(data.decode("utf-8").strip())
                    request_id = request.get("id", "")
                    path = request.get("path", "")
                    kwargs = request.get("kwargs", {})

                    # Handle introspection requests
                    if path == "_introspect.list_primitives":
                        result = self._get_primitives_metadata()
                        response = {
                            "type": "rpc_result",
                            "id": request_id,
                            "result": result,
                        }
                    else:
                        # Handle regular RPC calls
                        try:
                            result = await self._handle_rpc_call(
                                path=path,
                                kwargs=kwargs,
                                primitives=primitives,
                                computer_primitives=computer_primitives,
                            )
                            result = self._make_json_serializable(result)
                            response = {
                                "type": "rpc_result",
                                "id": request_id,
                                "result": result,
                            }
                        except Exception as e:
                            logger.error(f"RPC error for {path}: {e}", exc_info=True)
                            response = {
                                "type": "rpc_error",
                                "id": request_id,
                                "error": str(e),
                            }

                    # Send response
                    response_data = (json.dumps(response) + "\n").encode("utf-8")
                    await loop.sock_sendall(client_socket, response_data)

                finally:
                    client_socket.close()

            async def accept_rpc_connections():
                """Accept and handle RPC connections from shell script."""
                loop = asyncio.get_event_loop()
                while process.returncode is None:
                    try:
                        client_socket, _ = await asyncio.wait_for(
                            loop.sock_accept(server_socket),
                            timeout=0.1,
                        )
                        # Handle client in background
                        asyncio.create_task(handle_rpc_client(client_socket))
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        if process.returncode is None:
                            logger.debug(f"RPC accept error: {e}")
                        break

            # Start all tasks
            stdout_task = asyncio.create_task(read_stdout())
            stderr_task = asyncio.create_task(read_stderr())
            rpc_task = asyncio.create_task(accept_rpc_connections())

            try:
                # Wait for process to complete with timeout
                try:
                    await asyncio.wait_for(process.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    # Process timed out
                    await self._terminate_process_group(process, use_process_group)
                    return {
                        "result": -1,
                        "error": f"Shell script timed out after {timeout}s",
                        "stdout": "".join(stdout_output),
                        "stderr": "".join(stderr_output),
                    }

                # Wait for stdout/stderr to be fully read
                await asyncio.gather(stdout_task, stderr_task)

                # Build result
                exit_code = process.returncode
                return {
                    "result": exit_code,
                    "error": (
                        None
                        if exit_code == 0
                        else f"Script exited with code {exit_code}"
                    ),
                    "stdout": "".join(stdout_output),
                    "stderr": "".join(stderr_output),
                }

            except asyncio.CancelledError:
                await self._terminate_process_group(process, use_process_group)
                raise

            except Exception as e:
                return {
                    "result": -1,
                    "error": str(e),
                    "stdout": "".join(stdout_output),
                    "stderr": "".join(stderr_output),
                }

            finally:
                # Clean up
                rpc_task.cancel()
                try:
                    await rpc_task
                except asyncio.CancelledError:
                    pass

                stdout_task.cancel()
                stderr_task.cancel()
                try:
                    await stdout_task
                except asyncio.CancelledError:
                    pass
                try:
                    await stderr_task
                except asyncio.CancelledError:
                    pass

                server_socket.close()

                # Ensure process is terminated
                if process.returncode is None:
                    await self._terminate_process_group(process, use_process_group)
