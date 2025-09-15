"""
===================================================================
An interactive command-line sandbox for the FunctionManager.

This REPL provides a direct interface to the FunctionManager's core
features, allowing developers to add, list, search, and delete
reusable functions in the agent's "skill library."
"""

from __future__ import annotations

import ast
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List

# Ensure repository root is on the path for local execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()
from sandboxes.utils import (
    activate_project,
    build_cli_parser,
    configure_sandbox_logging,
)
from unity.function_manager.function_manager import FunctionManager

# Logger setup for the sandbox
LG = logging.getLogger("function_manager_sandbox")

# Help text displayed to the user in the REPL
_COMMANDS_HELP = """
FunctionManager Sandbox Commands
--------------------------------
Type a command and press Enter.

┌─────────────────── Commands ──────────────────────┐
│ add <path_to_file.py>   - Add function(s) from a Python file.     │
│ paste                   - Paste a function directly into the terminal.    │
│ list [mode] [n]         - List stored functions.                  │
│                           Modes: names, brief (default), full     │
│                           n: max number to show (default: all)    │
│ search <query> [n]      - Semantically search for functions.      │
│                           Optional: specify number of results (default: 5) │
│ delete <id>             - Delete a function by its ID.              │
│ help | h                - Show this help message.                 │
│ quit | exit             - Exit the sandbox.                       │
└───────────────────────────────────────────────────┘
"""


def extract_functions_from_source(source_code: str) -> List[str]:
    """
    Extract individual function definitions from source code.

    Args:
        source_code: Python source code containing one or more functions

    Returns:
        List of strings, each containing a single function definition
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        raise ValueError(f"Syntax error in source code: {e}")

    functions = []
    lines = source_code.splitlines()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Get the source lines for this function
            start_line = node.lineno - 1  # ast uses 1-based indexing
            # Find the end of the function
            end_line = start_line
            for child in ast.walk(node):
                if hasattr(child, "lineno"):
                    # Track the maximum line number within the function
                    if hasattr(child, "end_lineno"):
                        end_line = max(end_line, child.end_lineno - 1)
                    else:
                        end_line = max(end_line, child.lineno - 1)

            # Extract the function source
            function_lines = lines[start_line : end_line + 1]

            # Remove any leading empty lines but preserve indentation
            while function_lines and not function_lines[0].strip():
                function_lines.pop(0)

            # Ensure the function starts at column 0
            if function_lines:
                # Find the minimum indentation
                min_indent = float("inf")
                for line in function_lines:
                    if line.strip():  # Skip empty lines
                        indent = len(line) - len(line.lstrip())
                        min_indent = min(min_indent, indent)

                # Remove the common indentation
                if min_indent > 0 and min_indent != float("inf"):
                    function_lines = [
                        line[min_indent:] if len(line) > min_indent else line
                        for line in function_lines
                    ]

            function_source = "\n".join(function_lines)
            if function_source.strip():
                functions.append(function_source)

    return functions


async def _main_async() -> None:
    """Main asynchronous function to run the sandbox REPL."""
    # 1. Standard Sandbox Setup
    # Use shared utilities to parse args, activate Unify, and configure logs
    parser = build_cli_parser("FunctionManager Sandbox")
    args = parser.parse_args()
    os.environ["UNIFY_TRACED"] = "true" if args.traced else "false"

    activate_project(args.project_name, args.overwrite)
    configure_sandbox_logging(
        log_in_terminal=args.log_in_terminal,
        log_file=".logs_fm_sandbox.txt",
    )
    LG.setLevel(logging.INFO)

    # 2. Initialize the FunctionManager
    try:
        fm = FunctionManager()
        LG.info("FunctionManager initialized successfully.")
    except Exception as e:
        LG.error("Failed to initialize FunctionManager: %s", e, exc_info=True)
        print(
            f"❌ Critical Error: Could not start FunctionManager. Exiting. Error: {e}",
        )
        return

    # 3. Main REPL (Read-Eval-Print Loop)
    print(_COMMANDS_HELP)
    while True:
        try:
            raw_input = input("fm-sandbox> ").strip()
            if not raw_input:
                continue

            parts = raw_input.split(maxsplit=1)
            command = parts[0].lower()
            command_args = parts[1] if len(parts) > 1 else ""

            # --- Command Handling ---
            if command in {"quit", "exit"}:
                print("Exiting...")
                break

            elif command in {"help", "h", "?"}:
                print(_COMMANDS_HELP)

            elif command == "add":
                if not command_args:
                    print("Usage: add <path_to_file.py>")
                    continue
                try:
                    file_path = Path(command_args)
                    if not file_path.exists():
                        print(f"❌ Error: File not found at '{file_path}'")
                        continue
                    source_code = file_path.read_text()

                    # Extract individual functions from the file
                    functions = extract_functions_from_source(source_code)
                    if not functions:
                        print("❌ No functions found in the file.")
                        continue

                    print(f"Found {len(functions)} function(s) in the file.")

                    # Add each function individually
                    for func_source in functions:
                        # Show a preview of the function being added
                        func_lines = func_source.splitlines()
                        func_name_line = func_lines[0] if func_lines else ""
                        print(f"\nAdding: {func_name_line}")

                        results = fm.add_functions(implementations=[func_source])
                        for name, status in results.items():
                            if status == "added":
                                print(f"  ✅ Function '{name}' successfully added")
                            else:
                                print(f"  ❌ Function '{name}' failed: {status}")

                except (ValueError, SyntaxError) as e:
                    print(f"❌ Error adding functions: {e}")
                except Exception as e:
                    LG.error("Error during 'add' command: %s", e, exc_info=True)
                    print(f"❌ An unexpected error occurred: {e}")

            elif command == "paste":
                print(
                    "Paste your function code below. End with a blank line then CTRL+D (or CTRL+Z on Windows).",
                )
                lines = []
                while True:
                    try:
                        line = input()
                        lines.append(line)
                    except EOFError:
                        break
                source_code = "\n".join(lines)
                if not source_code.strip():
                    print("No code provided. Canceled.")
                    continue
                try:
                    results = fm.add_functions(implementations=[source_code])
                    for name, status in results.items():
                        print(f"✅ Function '{name}' status: {status}")
                except (ValueError, SyntaxError) as e:
                    print(f"❌ Error adding function: {e}")
                except Exception as e:
                    LG.error("Error during 'paste' command: %s", e, exc_info=True)
                    print(f"❌ An unexpected error occurred: {e}")

            elif command == "list":
                # Parse arguments for mode and limit
                mode = "brief"  # default mode
                limit = None  # default to show all

                if command_args:
                    args_parts = command_args.split()
                    for part in args_parts:
                        if part.lower() in ["names", "brief", "full"]:
                            mode = part.lower()
                        elif part.isdigit():
                            limit = int(part)

                # Get functions based on mode
                include_impl = mode == "full"
                functions = fm.list_functions(include_implementations=include_impl)

                if not functions:
                    print("No functions found in the library.")
                    continue

                # Apply limit if specified
                func_items = list(functions.items())
                if limit and limit < len(func_items):
                    func_items = func_items[:limit]
                    print(
                        f"\n--- Showing {limit} of {len(functions)} Functions (mode: {mode}) ---",
                    )
                else:
                    print(f"\n--- {len(functions)} Stored Functions (mode: {mode}) ---")

                for name, data in func_items:
                    func_id = data.get("function_id", "N/A")
                    if mode == "names":
                        # Show function names with IDs
                        print(f"• {name} (ID: {func_id})")
                    elif mode == "brief":
                        # Show name, ID, signature, and first line of docstring
                        print(f"\n• {name} (ID: {func_id})")
                        print(f"  Signature: {data['argspec']}")
                        if data["docstring"]:
                            # Show just the first line of docstring
                            first_line = data["docstring"].split("\n")[0]
                            if len(first_line) > 80:
                                first_line = first_line[:77] + "..."
                            print(f"  Description: {first_line}")
                    else:  # mode == "full"
                        # Show everything including implementation
                        print(f"\n#-- Function: {name} (ID: {func_id}) --#")
                        print(f"Signature: {data['argspec']}")
                        if data["docstring"]:
                            print(f"Docstring: {data['docstring']}")
                        print("--- Code ---")
                        print(data["implementation"])
                        print("#" + "-" * (14 + len(name)))

                if limit and limit < len(functions):
                    print(
                        f"\n(Showing {limit} of {len(functions)} functions. Use 'list {mode}' to see all.)",
                    )
                print("\n--- End of List ---")

            elif command == "search":
                if not command_args:
                    print("Usage: search <natural language query> [n]")
                    print("       n: optional number of results (default: 5)")
                    continue
                try:
                    # Parse the command arguments to extract query and optional n
                    args_parts = command_args.split()
                    n = 5  # default value
                    query = command_args

                    # Check if the last part is a number (n parameter)
                    if args_parts and args_parts[-1].isdigit():
                        n = int(args_parts[-1])
                        # Remove the number from the query
                        query = " ".join(args_parts[:-1])

                    # Validate we still have a query
                    if not query.strip():
                        print("Error: Please provide a search query.")
                        print("Usage: search <natural language query> [n]")
                        continue

                    print(
                        f"Searching for functions similar to: '{query}' (showing top {n} results)...",
                    )
                    results = fm.search_functions_by_similarity(query=query, n=n)
                    if not results:
                        print("No similar functions found.")
                        continue
                    print(f"\n--- Search Results (Top {min(len(results), n)}) ---")
                    for i, func in enumerate(results):
                        print(f"{i+1}. {func['name']} (ID: {func['function_id']})")
                        print(f"   Signature: {func['argspec']}")
                        print(f"   Docstring: {func.get('docstring', 'N/A')}")
                    print("\n--- End of Search ---")
                except ValueError as e:
                    print(f"❌ Invalid number format: {e}")
                except Exception as e:
                    LG.error("Error during 'search' command: %s", e, exc_info=True)
                    print(f"❌ An error occurred during search: {e}")

            elif command == "delete":
                if not command_args:
                    print("Usage: delete <function_id>")
                    continue
                try:
                    func_id = int(command_args)
                    # Note: delete_function does not currently support cascade, but we can add it later
                    result = fm.delete_function(function_id=func_id)
                    for name, status in result.items():
                        print(f"✅ Function '{name}' (ID: {func_id}) status: {status}")
                except ValueError:
                    print(
                        f"❌ Error: Invalid function ID '{command_args}'. Must be an integer.",
                    )
                except AssertionError as e:
                    print(f"❌ Error: {e}")
                except Exception as e:
                    LG.error("Error during 'delete' command: %s", e, exc_info=True)
                    print(f"❌ An unexpected error occurred: {e}")

            else:
                print(
                    f"Unknown command: '{command}'. Type 'help' for a list of commands.",
                )

        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        except Exception as e:
            # Catch-all for any other unexpected errors to keep the sandbox running
            LG.error(
                "An unexpected error occurred in the main loop: %s",
                e,
                exc_info=True,
            )
            print(f"\nAn unexpected error occurred: {e}\n")


def main() -> None:
    """Synchronous entry point for the sandbox."""
    try:
        asyncio.run(_main_async())
    except Exception as e:
        print(f"A critical error forced the sandbox to exit: {e}")
        LG.critical(
            "Sandbox forced to exit due to unhandled exception in main.",
            exc_info=True,
        )


if __name__ == "__main__":
    main()
