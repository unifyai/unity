Function Manager Sandbox
=========================

This folder contains an **interactive playground** for the `FunctionManager` component that lives in `unity/function_manager/function_manager.py`. The goal of this sandbox is to provide a direct, command-line interface for managing the agent's persistent "skill library" in isolation.

What is the `FunctionManager`?
------------------------------
`FunctionManager` is an abstraction that serves as the agent's persistent memory for reusable Python functions. It stores function implementations, signatures, docstrings, and dependency information, making them searchable and retrievable through both semantic similarity and structured queries. This allows actors like the `CodeActActor` to discover and incorporate existing code into their execution plans.

The manager provides comprehensive function lifecycle management with built-in security validation:

* **`add_functions(implementations)`** – Validates, compiles, and stores one or more Python functions with security checks
* **`list_functions()`** – Retrieves all stored functions with optional implementation details
* **`search_functions(query)`** – Finds functions using semantic search via embeddings
* **`search_functions(filter)`** – Performs structured filtering using Python expressions
* **`delete_function(function_id)`** – Removes functions with optional dependency cascade deletion
* **`get_precondition(function_name)`** – Retrieves stored preconditions for function execution

### Security & Validation Features

The `FunctionManager` includes robust security measures to ensure safe function execution:

- **Dangerous Built-ins Blocking**: Prevents use of potentially harmful functions like `eval`, `exec`, `open`, `input`, etc.
- **Function Isolation**: User-defined functions cannot directly call other user-defined functions to prevent complex dependencies
- **Method Call Allowance**: Permits safe method calls on objects (e.g., `computer_primitives.*`, `call_handle.*`)
- **AST-based Analysis**: Uses Abstract Syntax Tree parsing to analyze function calls and dependencies
- **Sandbox Execution**: Functions are compiled and validated in isolated namespaces

### Function Storage Schema

Each function is stored with comprehensive metadata:
- **function_id**: Unique identifier for the function
- **name**: Function name extracted from the definition
- **argspec**: Complete function signature (e.g., `(x: int, y: int) -> int`)
- **docstring**: Function documentation
- **implementation**: Full source code
- **calls**: List of functions/methods called within the function
- **embedding_text**: Combined text used for semantic search
- **precondition**: Optional state requirements for function execution

Running the sandbox
-------------------
The entry-point lives at `sandboxes/function_manager/sandbox.py` and can be executed directly or via Python's `-m` switch:

```bash
# Basic interactive session
python -m sandboxes.function_manager.sandbox

```

CLI flags
~~~~~~~~~
The sandbox uses the common helper in `sandboxes/utils.py`, so it shares a standard set of startup options:

```
--project_name / -p Name of the Unify **project/context** (default: "Sandbox")
--overwrite / -o    Delete any existing data for the chosen project before start
--project_version   Roll back to a specific project commit (int index)
--log_in_terminal   Stream logs to the terminal in addition to writing file logs
--log_tcp_port      Serve main logs over TCP on localhost:PORT (-1 auto-picks; 0 off)
--http_log_tcp_port Serve Unify Request logs over TCP on localhost:PORT (-1 auto when UNIFY_REQUESTS_DEBUG)
```

Interactive commands inside the REPL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once the sandbox starts, you can use the following commands at the prompt:

* **`add <path_to_file.py>`** – Add function(s) from a local Python file. Automatically extracts and adds each function individually with validation.
* **`paste`** – Enter a multi-line mode to paste function code directly. Finish with CTRL+D (Unix) or CTRL+Z+Enter (Windows).
* **`list [mode] [n]`** – Display stored functions with configurable verbosity:
  - Modes: `names` (compact list with IDs), `brief` (default - shows signatures), `full` (includes source code)
  - n: Optional limit on number of functions to show
  - Example: `list names 10` shows first 10 function names with IDs
* **`search <query> [n]`** – Perform a natural language search to find relevant functions using semantic similarity.
  - n: Optional number of results (default: 5)
  - Example: `search calculate statistics 3` finds top 3 matching functions
* **`delete <id>`** – Delete a function using its unique function_id (shown in list and search results). Cascades to dependent functions by default.
* **`help` / `h`** – Show the in-session command reference.
* **`quit` / `exit`** – Exit the sandbox.

### Example session
```text
$ python -m sandboxes.function_manager.sandbox
FunctionManager Sandbox Commands
--------------------------------
...

fm-sandbox> add ./my_functions/calculate.py
Found 2 function(s) in the file.

Adding: def calculate_sum(a: int, b: int) -> int:
  ✅ Function 'calculate_sum' successfully added

Adding: def calculate_product(a: int, b: int) -> int:
  ✅ Function 'calculate_product' successfully added

fm-sandbox> list names
--- 2 Stored Functions (mode: names) ---
• calculate_sum (ID: 1)
• calculate_product (ID: 2)
--- End of List ---

fm-sandbox> list brief
--- 2 Stored Functions (mode: brief) ---

• calculate_sum (ID: 1)
  Signature: (a: int, b: int) -> int
  Description: Calculates the sum of two integers.

• calculate_product (ID: 2)
  Signature: (a: int, b: int) -> int
  Description: Calculates the product of two integers.

--- End of List ---

fm-sandbox> search "add numbers" 1
Searching for functions similar to: 'add numbers' (showing top 1 results)...

--- Search Results (Top 1) ---
1. calculate_sum (ID: 1)
   Signature: (a: int, b: int) -> int
   Docstring: Calculates the sum of two integers.

--- End of Search ---

fm-sandbox> delete 1
✅ Function 'calculate_sum' (ID: 1) status: deleted
```

## Function Validation Examples

### ✅ Allowed Functions
```python
def process_data(data: list) -> dict:
    """Safe function that uses built-ins and method calls."""
    result = {}
    for item in data:
        # Built-in functions are allowed
        key = str(item.get('id', 0))
        # Method calls on objects are allowed
        result[key] = item.upper() if hasattr(item, 'upper') else item
    return result

def make_api_call(computer_primitives, endpoint: str) -> dict:
    """Function using action provider methods."""
    # Method calls on computer_primitives are explicitly allowed
    response = computer_primitives.get(endpoint)
    return response.json() if hasattr(response, 'json') else {}
```

### ❌ Blocked Functions
```python
# This would be rejected - calls dangerous built-in
def dangerous_function():
    eval("print('hello')")  # ❌ eval is blocked

# This would be rejected - calls another user function
def caller_function():
    return calculate_sum(1, 2)  # ❌ Cannot call user-defined functions
```

## Logging and debugging

- By default, logs are written to `.logs_fm_sandbox.txt` (overwritten each run). Pass `--log_in_terminal` to also stream logs to the terminal.
- Optional TCP streams:
  - Main logs: `--log_tcp_port -1` auto-picks an available port (or specify an explicit port). Connect with `nc 127.0.0.1 <PORT>`.
  - Unify Request logs only: `--http_log_tcp_port -1` auto-enables when `UNIFY_REQUESTS_DEBUG` is set; connect with `nc 127.0.0.1 <PORT>`.
- A dedicated Unify Request log file is also written to `.logs_unify_requests.txt`.

### Troubleshooting
* **Unify backend access** – The sandbox will attempt to create contexts and logs in your configured Unify project. If your credentials (`UNIFY_KEY`, `ORCHESTRA_URL`) are missing or invalid you may see HTTP errors.
* **Python file parsing** – When adding functions via `add <file>`, ensure the file contains valid Python syntax with functions starting at column 0 (no indentation).
* **Function validation errors** – Functions that call dangerous built-ins or other user-defined functions will be rejected with specific error messages.
* **Memory persistence** – Functions are stored persistently in the chosen project. Use `--overwrite` to start fresh if needed.

Happy experimenting! 🎉
