# FunctionManager

The `FunctionManager` maintains a catalogue of executable Python functions, split into two categories:

1. **Primitives** – System action methods (state manager APIs) exposed for direct invocation
2. **Compositional Functions** – User-specific functions stored with their source code

## Architecture

### Two Separate Contexts

Functions are stored in two dedicated Unify contexts to ensure stable IDs:

| Context | Purpose | ID Assignment |
|---------|---------|---------------|
| `Functions/Primitives` | System primitives | Explicit stable IDs (code-defined) |
| `Functions/Compositional` | User-specific functions | Auto-incrementing (backend-managed) |

This separation guarantees:
- Primitive IDs are consistent across all users
- Compositional function IDs are never affected by primitive changes
- No ID collisions between the two namespaces

### Primitive ID Stability

Primitives receive stable IDs derived from a hash of their fully-qualified name (e.g., "ContactManager.ask" → deterministic integer). This means:
- IDs are consistent across all deployments
- Adding/removing methods doesn't affect other primitives' IDs
- No manual ID management required

Primitive methods are auto-discovered from `@abstractmethod` definitions on base classes (e.g., `BaseContactManager`), minus an explicit exclusion list for non-primitive methods like `clear()`.

---

## The `primitives` Object

The `primitives` object provides lazy access to all state manager primitives. Imports and instantiations only happen when accessed:

```python
async def update_contacts_and_search():
    # Only ContactManager is imported/instantiated
    await primitives.contacts.update(text="Add Alice Smith, alice@example.com")

    # Only WebSearcher is imported/instantiated
    result = await primitives.web.ask(question="What is the weather in London?")
    return result
```

### Available Properties

| Property | Manager | Methods |
|----------|---------|---------|
| `primitives.contacts` | ContactManager | `ask`, `update` |
| `primitives.transcripts` | TranscriptManager | `ask` |
| `primitives.knowledge` | KnowledgeManager | `ask`, `update`, `refactor` |
| `primitives.tasks` | TaskScheduler | `ask`, `update`, `execute` |
| `primitives.secrets` | SecretManager | `ask`, `update` |
| `primitives.web` | WebSearcher | `ask` |
| `primitives.computer` | ComputerPrimitives | `navigate`, `act`, `observe`, `query`, `reason` |

---

## The `computer_primitives` Object

When executed via an Actor, `computer_primitives` provides web and desktop control:

```python
async def browse_and_extract():
    await computer_primitives.navigate("https://example.com")
    content = await computer_primitives.observe()
    answer = await computer_primitives.query("What is the main heading?")
    return answer
```

### Key Difference from `primitives.computer`

| Object | When Available | Use Case |
|--------|----------------|----------|
| `primitives.computer` | Always (sandbox) | Direct access, no caching/logging |
| `computer_primitives` | Actor execution only | Proxied with caching, logging, instrumentation |

In practice, use `computer_primitives` when your function runs via an Actor – it provides idempotency caching, action logging, and integration with the Actor's execution runtime.

---

## Primitive Synchronization

Primitives are lazily synchronized to the database:

1. On first access (e.g., `list_primitives()`, `search_functions()`), the manager calls `sync_primitives()`
2. A hash of all primitive signatures/docstrings is compared against the stored hash
3. If changed, all primitives are deleted and re-inserted with their stable IDs
4. The hash is stored in `Functions/Meta` for future comparisons

This ensures primitives stay in sync with the codebase while avoiding unnecessary database writes.

---

## Custom Functions & Venvs (Source-Defined)

The `custom/` folder enables **forward-deployed engineers** to add client-specific compositional functions and virtual environments directly in source code, which are automatically synchronized to the database.

### Why Use This?

When deploying client-specific branches:
- No need to inject function strings or venv configs via SQL or API calls
- Everything is version-controlled alongside the codebase
- Changes are automatically detected and synced via hash comparison
- Easy to audit, review, and distill back into main

### Folder Structure

```
unity/function_manager/custom/
├── __init__.py           # @custom_function decorator (don't modify)
├── functions/            # Custom compositional functions
│   ├── __init__.py
│   ├── example.py
│   └── acme_workflows.py
└── venvs/                # Custom virtual environments
    ├── __init__.py
    ├── example_minimal.toml
    └── acme_ml.toml
```

### Quick Start

**Step 1: Create a custom venv** (if needed)

```toml
# custom/venvs/acme_ml.toml
[project]
name = "acme-ml"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
]
```

**Step 2: Create custom functions**

```python
# custom/functions/acme_workflows.py
from unity.function_manager.custom import custom_function

@custom_function()
async def acme_data_export(format: str = "csv") -> str:
    """Export ACME's proprietary data."""
    data = await primitives.knowledge.ask(question="Get all ACME records")
    return f"Exported to /exports/acme.{format}"


@custom_function(venv_name="acme_ml", verify=False)
async def acme_ml_inference(input_data: dict) -> dict:
    """Run inference in ACME's ML environment."""
    import torch  # Available in acme_ml venv
    return {"prediction": "result"}


@custom_function(auto_sync=False)
async def draft_function():
    """Work-in-progress - NOT synced."""
    pass
```

**Step 3: Sync to database**

```python
fm = FunctionManager()
fm.sync_custom()  # Syncs venvs first, then functions
```

### Decorator Options

```python
@custom_function(
    venv_name="acme_ml",         # Reference to custom/venvs/<name>.toml
    venv_id=1,                   # Direct venv ID (prefer venv_name for custom venvs)
    verify=True,                 # Actor verifies execution result
    precondition={"url": "..."}, # Required state before execution
    auto_sync=False,             # Exclude from sync entirely
)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `venv_name` | `Optional[str]` | `None` | Name of custom venv (filename without .toml) |
| `venv_id` | `Optional[int]` | `None` | Direct venv ID (for non-custom venvs) |
| `verify` | `bool` | `True` | Whether Actor should verify function execution |
| `precondition` | `Optional[dict]` | `None` | Required state before function can run |
| `auto_sync` | `bool` | `True` | Set to `False` to exclude from auto-sync |

**Note:** When `venv_name` is set, it takes precedence over `venv_id`. The name is resolved to an actual `venv_id` during sync.

### Sync Behavior

| Entity | Scenario | Behavior |
|--------|----------|----------|
| **Venvs** | New `.toml` in source | Inserted with auto-assigned `venv_id` |
| **Venvs** | `.toml` changed | Updated in-place (preserves `venv_id`) |
| **Venvs** | `.toml` removed from source | Deleted from database |
| **Functions** | New function in source | Inserted with auto-assigned `function_id` |
| **Functions** | Function changed | Updated in-place (preserves `function_id`) |
| **Functions** | Function removed from source | Deleted from database |
| **Both** | User-added with same name | Overwritten by source version |
| **Both** | `auto_sync=False` / `_` prefix | Excluded from sync entirely |

### How Sync Works

```python
fm.sync_custom()  # Recommended: syncs both in correct order
# OR
fm.sync_custom_venvs()     # Step 1: returns {name: venv_id} mapping
fm.sync_custom_functions() # Step 2: uses mapping to resolve venv_name
```

1. **Venvs synced first** – so `venv_name` can be resolved to `venv_id`
2. **Aggregate hash compared** – if unchanged, sync is skipped (fast path)
3. **Per-item hashes compared** – only changed items are updated
4. **IDs preserved** – updates never change `function_id` or `venv_id`
5. **Deletions applied** – items removed from source are deleted from DB

### File Organization

| Location | Purpose | Naming |
|----------|---------|--------|
| `custom/functions/*.py` | Python files with `@custom_function` decorated functions | Files starting with `_` are ignored |
| `custom/venvs/*.toml` | pyproject.toml content for venvs | Filename (without `.toml`) becomes the venv name |

### Explicit Sync (No Auto-Sync)

Syncing is **explicit** – call `fm.sync_custom()` when you want to sync. This is deliberate:
- Engineers control when sync happens
- No surprises during development
- Can test changes before syncing

---

## Writing Compositional Functions

Compositional functions are stored with their full source code. They may be created by:
- **Source-defined** via the `custom/` folder (see above)
- The Actor generating and saving a function during execution
- Pre-provisioning functions for a specific user/client
- Direct API calls to `add_functions()`

### Format Requirements

Each implementation string must contain **exactly one function definition** starting at column 0:

```python
# ✓ Correct
async def my_function():
    result = await primitives.contacts.ask(question="Who is Alice?")
    return result

# ✗ Wrong - indented
    async def my_function():
        pass

# ✗ Wrong - multiple functions
def helper():
    pass

def main():
    helper()
```

### Imports: Optional (Pre-Injected Globals Available)

Functions are executed in a sandboxed environment with pre-injected globals. Common modules are already available, but **imports do work** if needed:

```python
# ✓ Works - using pre-injected globals (preferred for common modules)
async def example_no_import():
    data = json.dumps({"key": "value"})
    return data

# ✓ Also works - explicit import (useful for non-standard packages)
async def example_with_import():
    import numpy as np
    x = np.array([1, 2, 3])
    return x.sum()
```

The sandbox includes `__import__`, so any package installed in the environment can be imported.

### Best Practice: Domain Types & Type Hints (Avoid Surprise `NameError`s)

Compositional functions are often used by the Actor/CodeActActor by **retrieving a callable** (e.g. via `search_functions(..., return_callable=True)`) and executing it in a fresh sandbox namespace.

Here’s the simple rule:
- If you reference a symbol in the **function body**, **import/define it** in the function (don’t assume it exists in globals).
- If you reference a symbol only in **annotations** (including forward-ref strings), that’s usually fine.

```python
# ✅ OK: "User" only appears in the annotation (as a forward-ref string).
# The function body doesn't need the User symbol at runtime.
async def greet(user: "User") -> str:
    return f"Hello {user.name}"


# ⚠️ NOT OK: Role is used at runtime, so it MUST exist (import/define it).
async def is_admin(role: "Role") -> bool:
    return role == Role.ADMIN


# ✅ Preferred: import the runtime type inside the function.
async def is_admin(role: "Role") -> bool:
    from my_app.types import Role
    return role == Role.ADMIN
```

Note: if your code resolves type hints at runtime (e.g. `typing.get_type_hints(...)` or Pydantic model building),
then all referenced names must be resolvable. `FunctionManager` makes annotation-resolution more robust, but it cannot
guess the *real* domain objects for runtime logic—imports/definitions are the reliable solution.

#### Pre-Injected by Sandbox

These are always available (from `create_execution_globals()`):

| Category | Available Names |
|----------|-----------------|
| **Builtins** | `print`, `len`, `str`, `int`, `float`, `list`, `dict`, `set`, `range`, `isinstance`, `issubclass`, `hasattr`, `getattr`, `enumerate`, `zip`, `sorted`, `min`, `max`, `sum`, `any`, `all`, etc. |
| **Modules** | `asyncio`, `re`, `json`, `datetime`, `collections`, `statistics`, `functools` |
| **Typing** | `typing`, `Any`, `Callable`, `Dict`, `List`, `Optional`, `Tuple`, `Set`, `Union`, `Literal` |
| **Pydantic** | `pydantic`, `BaseModel`, `Field` |
| **Primitives** | `primitives` – lazy access to all state managers |
| **Steerable** | `SteerableToolHandle`, `start_async_tool_loop`, `new_llm_client` |

#### Injected by Actor at Runtime

When functions are executed via an Actor (`CodeActActor`, `SingleFunctionActor`), additional objects are injected:

| Name | Description |
|------|-------------|
| `computer_primitives` | Web/desktop control (navigate, act, observe, query, reason) |
| `request_clarification` | Ask the user for clarification during execution |

---

## Example: Complete Function

```python
async def research_contact(contact_name: str) -> str:
    """
    Research a contact by searching the web and updating their record.

    Args:
        contact_name: Name of the contact to research.

    Returns:
        Summary of what was found and updated.
    """
    # Query existing contact info
    contact_info = await primitives.contacts.ask(
        question=f"What do we know about {contact_name}?"
    )

    # Search the web for more info
    web_results = await primitives.web.ask(
        question=f"Find professional information about {contact_name}"
    )

    # Update the contact with new information
    await primitives.contacts.update(
        text=f"Update {contact_name} with: {web_results}"
    )

    return f"Updated {contact_name} with web research findings."
```

---

## Steerable Functions

Compositional functions can optionally return a **steerable handle** instead of a final result. This allows the calling layer (e.g., `SingleFunctionActor`) to forward steering operations (interject, pause, stop) into the running function.

### What is a Steerable Function?

A steerable function is one that:
1. Starts a background task (e.g., an async tool loop, an actor)
2. Returns a `SteerableToolHandle` immediately (before the task completes)
3. Allows the caller to interact with the running task via the handle

### Runtime Detection

Steerability is detected at **runtime** via `isinstance(result, SteerableToolHandle)`:

```python
from unity.common.async_tool_loop import SteerableToolHandle

result = await my_function()

if isinstance(result, SteerableToolHandle):
    # Function returned a steerable handle - can forward steering operations
    await result.interject("Please also check for errors")
    final_result = await result.result()
else:
    # Function returned a plain value - no steering possible
    final_result = result
```

### Writing a Steerable Function

Use the steerable infrastructure available in the execution globals:

```python
async def my_steerable_workflow(goal: str) -> SteerableToolHandle:
    """
    A steerable workflow that uses an async tool loop.

    The caller can interject, pause, or stop this workflow while it runs.
    """
    # Create an LLM client
    client = new_llm_client()
    client.set_system_message("You are a helpful assistant.")

    # Start an async tool loop - returns a handle immediately
    handle = start_async_tool_loop(
        client=client,
        message=goal,
        tools={},  # Add tools as needed
        loop_id="my-workflow",
    )

    # Return the handle - the loop continues running in the background
    return handle
```

### Available Infrastructure

These are injected into the execution globals by `create_execution_globals()`:

| Name | Purpose |
|------|---------|
| `SteerableToolHandle` | Base ABC for steerable handles and runtime `isinstance` checks |
| `start_async_tool_loop` | Factory function to create async tool loop handles |
| `new_llm_client` | Factory to create LLM clients for async tool loops |

### Handle Methods

Steerable handles provide these methods:

| Method | Description |
|--------|-------------|
| `await handle.result()` | Wait for and return the final result |
| `await handle.interject(message)` | Inject a message into the running task |
| `await handle.pause()` | Pause the task (in-flight operations continue) |
| `await handle.resume()` | Resume a paused task |
| `handle.stop(reason)` | Cancel the task immediately |
| `await handle.ask(question)` | Query the task's status without modifying it |

### Example: Steerable Research Workflow

```python
async def steerable_research(topic: str) -> SteerableToolHandle:
    """
    Research a topic with the ability to steer mid-flight.

    The caller can interject to refine the search, or stop early
    if enough information has been gathered.
    """
    client = new_llm_client()
    client.set_system_message(
        "You are a research assistant. Search the web and compile findings. "
        "Be thorough but respond to any user interjections to refine your approach."
    )

    # Define tools the LLM can use
    tools = {
        "web_search": primitives.web.ask,
        "save_finding": primitives.knowledge.update,
    }

    return start_async_tool_loop(
        client=client,
        message=f"Research the following topic thoroughly: {topic}",
        tools=tools,
        loop_id=f"research-{topic[:20]}",
        timeout=300,  # 5 minute timeout
    )
```

### Non-Steerable Functions

Regular functions that return plain values are **not** steerable:

```python
async def simple_lookup(name: str) -> str:
    """A simple function - returns a plain value, not steerable."""
    result = await primitives.contacts.ask(question=f"Who is {name}?")
    return result  # Plain string, not a handle
```

The execution layer will detect this via `isinstance` and handle it normally.

---

## API Summary

### Primitives

```python
fm = FunctionManager()

# Ensure primitives are synced
fm.sync_primitives()

# List all primitives
fm.list_primitives()

# Search includes primitives by default
fm.search_functions(query="navigate web", include_primitives=True)
```

### Compositional Functions

```python
# Add functions
fm.add_functions(implementations=["async def foo(): pass"])

# List functions
fm.list_functions(include_implementations=False)

# Search by similarity
fm.search_functions(query="contact management", n=5)

# Delete
fm.delete_function(function_id=1)
```
