#!/usr/bin/env python3
"""
CLI bridge for shell scripts to call primitives via RPC.

This script enables shell functions to invoke Unity primitives (ContactManager.ask,
FileManager.search_files, etc.) from within shell scripts. Communication happens
via a Unix domain socket to the parent Python process that's executing the shell script.

Usage from shell scripts:
    # Call a primitive method
    result=$(unity-primitive files search_files --references '{"query": "invoices"}' --k 5)

    # The result is JSON
    echo "$result" | jq '.[]'

    # List available managers
    unity-primitive --list-managers

    # List methods for a manager
    unity-primitive files --list-methods

Protocol:
    The CLI connects to a Unix domain socket (path in UNITY_RPC_SOCKET env var)
    and sends JSON-RPC requests. The parent process handles these requests and
    sends back JSON responses.

    Request: {"type": "rpc_call", "id": str, "path": str, "kwargs": dict}
    Response: {"type": "rpc_result", "id": str, "result": Any}
              {"type": "rpc_error", "id": str, "error": str}
"""

from __future__ import annotations

import json
import os
import socket
import sys
import uuid
from typing import Any, Dict, List


# ────────────────────────────────────────────────────────────────────────────
# RPC Communication
# ────────────────────────────────────────────────────────────────────────────


def send_rpc_request(socket_path: str, path: str, kwargs: Dict[str, Any]) -> Any:
    """
    Send an RPC request to the parent process and return the result.

    Args:
        socket_path: Path to the Unix domain socket
        path: RPC path (e.g., "files.search_files")
        kwargs: Keyword arguments for the method call

    Returns:
        The result from the RPC call

    Raises:
        RuntimeError: If the RPC call fails
        ConnectionError: If unable to connect to socket
    """
    request_id = uuid.uuid4().hex

    request = {
        "type": "rpc_call",
        "id": request_id,
        "path": path,
        "kwargs": kwargs,
    }

    # Connect to Unix domain socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(socket_path)

        # Send request as newline-delimited JSON
        sock.sendall((json.dumps(request) + "\n").encode("utf-8"))

        # Read response
        response_data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response_data += chunk
            # Check if we have a complete JSON message
            if b"\n" in response_data:
                break

        if not response_data:
            raise RuntimeError("No response from RPC server")

        response = json.loads(response_data.decode("utf-8").strip())

        if response.get("type") == "rpc_error":
            raise RuntimeError(f"RPC error: {response.get('error')}")

        return response.get("result")

    finally:
        sock.close()


def get_available_primitives(socket_path: str) -> Dict[str, Any]:
    """
    Get metadata about available primitives from the parent process.

    Returns:
        Dict with manager names and their available methods
    """
    return send_rpc_request(socket_path, "_introspect.list_primitives", {})


# ────────────────────────────────────────────────────────────────────────────
# Argument Parsing
# ────────────────────────────────────────────────────────────────────────────


def parse_value(value_str: str) -> Any:
    """
    Parse a command-line argument value to its appropriate Python type.

    Supports:
    - JSON objects/arrays: '{"key": "value"}', '[1, 2, 3]'
    - Numbers: 42, 3.14
    - Booleans: true, false
    - Null: null
    - Strings: everything else

    Args:
        value_str: The string value from command line

    Returns:
        Parsed value in appropriate Python type
    """
    # Try JSON first (handles objects, arrays, bools, null, numbers)
    try:
        return json.loads(value_str)
    except json.JSONDecodeError:
        pass

    # Return as string
    return value_str


def build_kwargs_from_args(args: List[str]) -> Dict[str, Any]:
    """
    Convert command-line arguments to kwargs dict.

    Supports:
    - --key value
    - --key=value
    - Positional args become _positional_0, _positional_1, etc.

    Args:
        args: List of command-line arguments after manager.method

    Returns:
        Dict of keyword arguments
    """
    kwargs: Dict[str, Any] = {}
    positional_idx = 0
    i = 0

    while i < len(args):
        arg = args[i]

        if arg.startswith("--"):
            # Named argument
            if "=" in arg:
                # --key=value format
                key, value = arg[2:].split("=", 1)
                kwargs[key] = parse_value(value)
            else:
                # --key value format
                key = arg[2:]
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    i += 1
                    kwargs[key] = parse_value(args[i])
                else:
                    # Flag without value (treat as True)
                    kwargs[key] = True
        else:
            # Positional argument
            kwargs[f"_positional_{positional_idx}"] = parse_value(arg)
            positional_idx += 1

        i += 1

    return kwargs


# ────────────────────────────────────────────────────────────────────────────
# CLI Commands
# ────────────────────────────────────────────────────────────────────────────


def cmd_list_managers(socket_path: str) -> int:
    """List available managers."""
    try:
        primitives = get_available_primitives(socket_path)
        managers = primitives.get("managers", {})

        print("Available managers:")
        for name, info in sorted(managers.items()):
            desc = info.get("description", "")
            print(f"  {name}: {desc}")

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_list_methods(socket_path: str, manager_name: str) -> int:
    """List methods for a specific manager."""
    try:
        primitives = get_available_primitives(socket_path)
        managers = primitives.get("managers", {})

        if manager_name not in managers:
            print(f"Error: Unknown manager '{manager_name}'", file=sys.stderr)
            print(
                f"Available managers: {', '.join(sorted(managers.keys()))}",
                file=sys.stderr,
            )
            return 1

        manager_info = managers[manager_name]
        methods = manager_info.get("methods", {})

        print(f"Methods for '{manager_name}':")
        for method_name, method_info in sorted(methods.items()):
            signature = method_info.get("signature", "()")
            docstring = method_info.get("docstring", "")
            # Truncate long docstrings
            if docstring:
                first_line = docstring.split("\n")[0][:80]
                print(f"  {method_name}{signature}")
                print(f"    {first_line}")
            else:
                print(f"  {method_name}{signature}")

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_call_method(
    socket_path: str,
    manager_name: str,
    method_name: str,
    args: List[str],
) -> int:
    """Call a primitive method."""
    try:
        kwargs = build_kwargs_from_args(args)
        path = f"{manager_name}.{method_name}"

        result = send_rpc_request(socket_path, path, kwargs)

        # Output result as JSON
        print(json.dumps(result, indent=2, default=str))
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# ────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ────────────────────────────────────────────────────────────────────────────


def main() -> int:
    """Main entry point for the CLI."""
    # Get socket path from environment
    socket_path = os.environ.get("UNITY_RPC_SOCKET")
    if not socket_path:
        print(
            "Error: UNITY_RPC_SOCKET environment variable not set.\n"
            "This command should be run from within a Unity shell function.",
            file=sys.stderr,
        )
        return 1

    # Parse arguments manually for flexibility
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(
            "Usage: unity-primitive [options] <manager> <method> [--arg value ...]\n"
            "\n"
            "Call Unity primitives from shell scripts.\n"
            "\n"
            "Options:\n"
            "  --list-managers     List available managers\n"
            "  -h, --help          Show this help message\n"
            "\n"
            "Manager options:\n"
            "  <manager> --list-methods   List methods for a manager\n"
            "\n"
            "Examples:\n"
            "  unity-primitive --list-managers\n"
            "  unity-primitive files --list-methods\n"
            '  unity-primitive files search_files --references \'{"query": "budget"}\' --k 5\n'
            "  unity-primitive knowledge ask --text 'What is our return policy?'\n",
        )
        return 0

    # Handle --list-managers
    if args[0] == "--list-managers":
        return cmd_list_managers(socket_path)

    # First non-option arg is manager name
    manager_name = args[0]
    remaining_args = args[1:]

    # Handle <manager> --list-methods
    if remaining_args and remaining_args[0] == "--list-methods":
        return cmd_list_methods(socket_path, manager_name)

    # Otherwise, next arg is method name
    if not remaining_args:
        print(
            f"Error: No method specified for manager '{manager_name}'",
            file=sys.stderr,
        )
        print(
            f"Use 'unity-primitive {manager_name} --list-methods' to see available methods",
            file=sys.stderr,
        )
        return 1

    method_name = remaining_args[0]
    method_args = remaining_args[1:]

    return cmd_call_method(socket_path, manager_name, method_name, method_args)


if __name__ == "__main__":
    sys.exit(main())
