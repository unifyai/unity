#!/usr/bin/env python3
"""
Standalone runner script for executing functions in isolated virtual environments.

This script is designed to be copied into custom venvs and executed as a subprocess.
It has NO dependencies on the unity package - it's completely standalone.

Communication Protocol (Bidirectional JSON-RPC):

Initial request from main process:
    {"type": "execute", "implementation": str, "call_kwargs": dict, "is_async": bool}

During execution, subprocess can send RPC requests:
    {"type": "rpc_call", "id": str, "path": str, "kwargs": dict}

Main process responds with:
    {"type": "rpc_result", "id": str, "result": Any}
    {"type": "rpc_error", "id": str, "error": str}

Final response from subprocess:
    {"type": "complete", "result": Any, "error": str|null, "stdout": str, "stderr": str}

The script:
1. Reads initial execution request from stdin
2. Sets up proxy objects for primitives and computer_primitives
3. Executes the function, handling RPC calls as they occur
4. Returns the final result
"""

import asyncio
import io
import json
import sys
import threading
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from queue import Queue
from typing import Any, Dict


# ────────────────────────────────────────────────────────────────────────────
# RPC Communication Layer
# ────────────────────────────────────────────────────────────────────────────

# Global state for RPC communication
_rpc_responses: Dict[str, Queue] = {}
_rpc_lock = threading.Lock()
_stdin_lock = threading.Lock()
_stdout_lock = threading.Lock()


def send_message(msg: dict) -> None:
    """Send a JSON message to stdout (to main process)."""
    with _stdout_lock:
        # Write to the original stdout (before any capture)
        sys.__stdout__.write(json.dumps(msg) + "\n")
        sys.__stdout__.flush()


def read_message() -> dict:
    """Read a JSON message from stdin (from main process)."""
    with _stdin_lock:
        line = sys.__stdin__.readline()
        if not line:
            raise EOFError("stdin closed")
        return json.loads(line.strip())


def rpc_call_sync(path: str, kwargs: dict) -> Any:
    """Make a synchronous RPC call to the main process."""
    request_id = uuid.uuid4().hex

    # Create a queue for this request's response
    response_queue: Queue = Queue()
    with _rpc_lock:
        _rpc_responses[request_id] = response_queue

    try:
        # Send RPC request
        send_message(
            {
                "type": "rpc_call",
                "id": request_id,
                "path": path,
                "kwargs": kwargs,
            },
        )

        # Wait for response
        response = response_queue.get(timeout=300)  # 5 minute timeout

        if response.get("type") == "rpc_error":
            raise RuntimeError(f"RPC error: {response.get('error')}")

        return response.get("result")
    finally:
        with _rpc_lock:
            _rpc_responses.pop(request_id, None)


async def rpc_call_async(path: str, kwargs: dict) -> Any:
    """Make an async RPC call to the main process."""
    # Run sync RPC in executor to avoid blocking event loop
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, rpc_call_sync, path, kwargs)


def dispatch_rpc_response(msg: dict) -> None:
    """Dispatch an RPC response to the waiting caller."""
    request_id = msg.get("id")
    with _rpc_lock:
        if request_id in _rpc_responses:
            _rpc_responses[request_id].put(msg)


# ────────────────────────────────────────────────────────────────────────────
# Proxy Classes for Primitives
# ────────────────────────────────────────────────────────────────────────────


class ManagerProxy:
    """Proxy for a state manager (e.g., contacts, knowledge)."""

    def __init__(self, manager_name: str, is_async: bool = True):
        self._manager_name = manager_name
        self._is_async = is_async

    def _make_method(self, method_name: str):
        """Create a method that makes an RPC call."""
        path = f"{self._manager_name}.{method_name}"

        if self._is_async:

            async def async_method(**kwargs):
                return await rpc_call_async(path, kwargs)

            return async_method
        else:

            def sync_method(**kwargs):
                return rpc_call_sync(path, kwargs)

            return sync_method

    def __getattr__(self, name: str):
        # Return a callable for any method access
        return self._make_method(name)


class PrimitivesProxy:
    """
    Proxy for the primitives object.

    Provides access to all state manager methods via RPC.
    Usage: await primitives.contacts.ask(question="...")
    """

    def __init__(self, is_async: bool = True):
        self._is_async = is_async
        self._managers: Dict[str, ManagerProxy] = {}

    def __getattr__(self, name: str) -> ManagerProxy:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._managers:
            self._managers[name] = ManagerProxy(name, self._is_async)
        return self._managers[name]


class ComputerPrimitivesProxy:
    """
    Proxy for the computer_primitives object.

    Provides access to browser/desktop control methods via RPC.
    Usage: await computer_primitives.click(selector="...")
    """

    def __init__(self, is_async: bool = True):
        self._is_async = is_async

    def _make_method(self, method_name: str):
        """Create a method that makes an RPC call."""
        path = f"computer.{method_name}"

        if self._is_async:

            async def async_method(**kwargs):
                return await rpc_call_async(path, kwargs)

            return async_method
        else:

            def sync_method(**kwargs):
                return rpc_call_sync(path, kwargs)

            return sync_method

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._make_method(name)


# ────────────────────────────────────────────────────────────────────────────
# Execution Environment
# ────────────────────────────────────────────────────────────────────────────


def create_safe_globals(is_async: bool = True):
    """Create a sandboxed globals dict for function execution."""
    import collections
    import datetime
    import functools
    import re
    import statistics
    import typing

    # Safe builtins - excluding dangerous functions
    safe_builtins = {}
    builtins_dict = (
        __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    )
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
        "object",
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
        "__import__",
    ]:
        if k in builtins_dict:
            safe_builtins[k] = builtins_dict[k]

    globals_dict = {
        "__builtins__": safe_builtins,
        # Standard library modules
        "asyncio": asyncio,
        "re": re,
        "json": json,
        "datetime": datetime,
        "collections": collections,
        "statistics": statistics,
        "functools": functools,
        # Typing module and common types
        "typing": typing,
        "Any": typing.Any,
        "Callable": typing.Callable,
        "Dict": typing.Dict,
        "List": typing.List,
        "Optional": typing.Optional,
        "Tuple": typing.Tuple,
        "Set": typing.Set,
        "Union": typing.Union,
        "Literal": typing.Literal,
        # Primitives proxies
        "primitives": PrimitivesProxy(is_async=is_async),
        "computer_primitives": ComputerPrimitivesProxy(is_async=is_async),
    }

    # Try to add pydantic if available in this venv
    try:
        import pydantic
        from pydantic import BaseModel, Field

        globals_dict.update(
            {
                "pydantic": pydantic,
                "BaseModel": BaseModel,
                "Field": Field,
            },
        )
    except ImportError:
        pass

    return globals_dict


def execute_sync(implementation: str, call_kwargs: dict) -> dict:
    """Execute a synchronous function."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    result = None
    error = None

    try:
        globals_dict = create_safe_globals(is_async=False)

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(implementation, globals_dict)

            # Find the function that was defined
            func_name = None
            base_names = set(create_safe_globals(is_async=False).keys())
            for name, obj in globals_dict.items():
                if (
                    callable(obj)
                    and not name.startswith("_")
                    and name not in base_names
                ):
                    func_name = name
                    break

            if func_name is None:
                raise ValueError("No function found in implementation")

            fn = globals_dict[func_name]
            result = fn(**call_kwargs)

    except Exception:
        error = traceback.format_exc()

    return {
        "result": result,
        "error": error,
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue(),
    }


async def execute_async(implementation: str, call_kwargs: dict) -> dict:
    """Execute an asynchronous function."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    result = None
    error = None

    try:
        globals_dict = create_safe_globals(is_async=True)

        exec(implementation, globals_dict)

        # Find the async function that was defined
        func_name = None
        base_globals = set(create_safe_globals(is_async=True).keys())
        for name, obj in globals_dict.items():
            if (
                asyncio.iscoroutinefunction(obj)
                and not name.startswith("_")
                and name not in base_globals
            ):
                func_name = name
                break

        if func_name is None:
            raise ValueError("No async function found in implementation")

        fn = globals_dict[func_name]

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            result = await fn(**call_kwargs)

    except Exception:
        error = traceback.format_exc()

    return {
        "result": result,
        "error": error,
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue(),
    }


def make_json_serializable(obj):
    """Convert an object to a JSON-serializable form."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    # For other types, convert to string
    return str(obj)


# ────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ────────────────────────────────────────────────────────────────────────────


def run_with_rpc_loop(implementation: str, call_kwargs: dict, is_async: bool) -> dict:
    """
    Run a function with RPC support.

    This handles bidirectional communication:
    - Executes the function in a separate thread
    - Main thread handles RPC responses from stdin
    """
    result_queue: Queue = Queue()

    def execute_function():
        """Execute the function and put result in queue."""
        try:
            if is_async:
                res = asyncio.run(execute_async(implementation, call_kwargs))
            else:
                res = execute_sync(implementation, call_kwargs)
            result_queue.put(res)
        except Exception as e:
            result_queue.put(
                {
                    "result": None,
                    "error": traceback.format_exc(),
                    "stdout": "",
                    "stderr": "",
                },
            )

    # Start function execution in a thread
    exec_thread = threading.Thread(target=execute_function, daemon=True)
    exec_thread.start()

    # Main thread handles RPC responses
    while exec_thread.is_alive():
        # Check for RPC responses with a timeout so we can check if thread is done
        try:
            # Use select or polling to check stdin with timeout
            import select

            readable, _, _ = select.select([sys.__stdin__], [], [], 0.1)
            if readable:
                line = sys.__stdin__.readline()
                if line:
                    msg = json.loads(line.strip())
                    if msg.get("type") in ("rpc_result", "rpc_error"):
                        dispatch_rpc_response(msg)
        except Exception:
            # On Windows or if select fails, just do blocking read with thread check
            pass

    # Get the result
    return result_queue.get(timeout=1)


def main():
    """Main entry point for the runner."""
    # Read initial execution request from stdin
    try:
        line = sys.__stdin__.readline()
        if not line:
            send_message(
                {
                    "type": "complete",
                    "result": None,
                    "error": "No input received",
                    "stdout": "",
                    "stderr": "",
                },
            )
            sys.exit(1)

        input_data = json.loads(line.strip())
    except json.JSONDecodeError as e:
        send_message(
            {
                "type": "complete",
                "result": None,
                "error": f"Invalid JSON input: {e}",
                "stdout": "",
                "stderr": "",
            },
        )
        sys.exit(1)

    msg_type = input_data.get("type", "execute")
    if msg_type != "execute":
        send_message(
            {
                "type": "complete",
                "result": None,
                "error": f"Expected 'execute' message, got '{msg_type}'",
                "stdout": "",
                "stderr": "",
            },
        )
        sys.exit(1)

    implementation = input_data.get("implementation", "")
    call_kwargs = input_data.get("call_kwargs", {})
    is_async = input_data.get("is_async", False)

    # Execute with RPC support
    result = run_with_rpc_loop(implementation, call_kwargs, is_async)

    # Make result JSON-serializable
    result["result"] = make_json_serializable(result["result"])

    # Send completion message
    send_message(
        {
            "type": "complete",
            **result,
        },
    )


if __name__ == "__main__":
    main()
