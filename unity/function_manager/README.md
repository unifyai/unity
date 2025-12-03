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

Primitives receive IDs based on their position in `PRIMITIVE_SOURCES` (in `primitives.py`). This registry is **append-only** – new primitives are added at the end, existing entries are never reordered or removed. This ensures IDs remain stable across Unify upgrades.

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
| `primitives.guidance` | GuidanceManager | `ask`, `update` |
| `primitives.web` | WebSearcher | `ask` |
| `primitives.skills` | SkillManager | `ask` |
| `primitives.computer` | ComputerPrimitives | `navigate`, `act`, `observe`, `query`, `reason` |

---

## The `computer_primitives` Object

When executed via an Actor, `computer_primitives` provides browser and desktop control:

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

1. On first access (e.g., `list_primitives()`, `search_functions_by_similarity()`), the manager calls `sync_primitives()`
2. A hash of all primitive signatures/docstrings is compared against the stored hash
3. If changed, all primitives are deleted and re-inserted with their stable IDs
4. The hash is stored in `Functions/Meta` for future comparisons

This ensures primitives stay in sync with the codebase while avoiding unnecessary database writes.

---

## Custom Functions (Source-Defined)

The `custom/` folder enables **forward-deployed engineers** to add client-specific compositional functions directly in source code, which are automatically synchronized to `Functions/Compositional`.

### Why Use This?

When deploying client-specific branches:
- No need to inject function strings via SQL or API calls
- Functions are version-controlled alongside the codebase
- Changes are automatically detected and synced via hash comparison
- Easy to audit, review, and distill back into main

### Quick Start

1. **Create a Python file** in `unity/function_manager/custom/`
2. **Decorate functions** with `@custom_function()`
3. **Commit and push** to the client branch
4. Functions auto-sync on next `FunctionManager` initialization

```python
# unity/function_manager/custom/acme_workflows.py
from unity.function_manager.custom import custom_function

@custom_function()
async def acme_data_export(format: str = "csv") -> str:
    """
    Export ACME's proprietary data in the specified format.

    Args:
        format: Output format (csv, json, xlsx)

    Returns:
        Path to the exported file
    """
    # ACME-specific export logic
    data = await primitives.knowledge.ask(question="Get all ACME records")
    return f"Exported to /exports/acme.{format}"


@custom_function(venv_id=2, verify=False)
async def acme_ml_inference(input_data: dict) -> dict:
    """
    Run inference using ACME's ML model in their custom venv.
    """
    import torch  # Available in venv 2
    # ML inference logic...
    return {"prediction": "result"}


@custom_function(auto_sync=False)
async def draft_experimental_function():
    """
    Work-in-progress function - NOT synced to database.
    """
    pass
```

### Decorator Options

```python
@custom_function(
    venv_id=1,              # Run in custom virtual environment (default: None)
    verify=True,            # Actor verifies execution result (default: True)
    precondition={"url": "..."}, # Required state before execution (default: None)
    auto_sync=False,        # Exclude from sync entirely (default: True)
)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `venv_id` | `Optional[int]` | `None` | Virtual environment ID for isolated execution |
| `verify` | `bool` | `True` | Whether Actor should verify function execution |
| `precondition` | `Optional[dict]` | `None` | Required state before function can run |
| `auto_sync` | `bool` | `True` | Set to `False` to exclude from auto-sync |

### Sync Behavior

| Scenario | Behavior |
|----------|----------|
| **New function in source** | Inserted with auto-assigned `function_id` |
| **Function changed** | Updated in-place (preserves `function_id`) |
| **Function removed from source** | Deleted from database |
| **User-added function with same name** | Overwritten by source version |
| **`auto_sync=False`** | Excluded from sync entirely |

### How Sync Works

1. On `FunctionManager` init, `sync_custom_functions()` is available (not called by default)
2. An aggregate hash of all custom functions is compared against the stored hash
3. If unchanged, sync is skipped entirely (fast path)
4. If changed, per-function hashes are compared:
   - Matching hash → no update (preserves `function_id`)
   - Different hash → update in place
   - New function → insert
   - Removed from source → delete from database

### File Organization

```
unity/function_manager/custom/
├── __init__.py           # @custom_function decorator (don't modify)
├── acme_workflows.py     # Client-specific functions
├── data_processing.py    # Grouped by domain
└── _drafts.py            # Files starting with _ are ignored
```

- One or multiple functions per file
- Files starting with `_` are ignored (use for drafts)
- Functions without `@custom_function` decorator are ignored

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

#### Pre-Injected by Sandbox

These are always available (from `create_execution_globals()`):

| Category | Available Names |
|----------|-----------------|
| **Builtins** | `print`, `len`, `str`, `int`, `float`, `list`, `dict`, `set`, `range`, `isinstance`, `hasattr`, `getattr`, `enumerate`, `zip`, `sorted`, `min`, `max`, `sum`, `any`, `all`, etc. |
| **Modules** | `asyncio`, `re`, `json`, `datetime`, `collections`, `statistics`, `functools` |
| **Typing** | `typing`, `Any`, `Callable`, `Dict`, `List`, `Optional`, `Tuple`, `Set`, `Union`, `Literal` |
| **Pydantic** | `pydantic`, `BaseModel`, `Field` |
| **Primitives** | `primitives` – lazy access to all state managers |

#### Injected by Actor at Runtime

When functions are executed via an Actor (`HierarchicalActor`, `SingleFunctionActor`), additional objects are injected:

| Name | Description |
|------|-------------|
| `computer_primitives` | Browser/desktop control (navigate, act, observe, query, reason) |
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

## API Summary

### Primitives

```python
fm = FunctionManager()

# Ensure primitives are synced
fm.sync_primitives()

# List all primitives
fm.list_primitives()

# Search includes primitives by default
fm.search_functions_by_similarity(query="navigate browser", include_primitives=True)
```

### Compositional Functions

```python
# Add functions
fm.add_functions(implementations=["async def foo(): pass"])

# List functions
fm.list_functions(include_implementations=False)

# Search by similarity
fm.search_functions_by_similarity(query="contact management", n=5)

# Delete
fm.delete_function(function_id=1)
```
