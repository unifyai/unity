"""
Custom compositional functions and virtual environments for auto-sync.

This module provides the `@custom_function` decorator for marking functions
that should be automatically synchronized to `Functions/Compositional`.

## Folder Structure

```
custom/
├── __init__.py         # This file (decorator + exports)
├── functions/          # Custom compositional functions
│   ├── example.py
│   └── client_workflows.py
└── venvs/              # Custom virtual environments
    ├── ml_env.toml
    └── data_science.toml
```

## Usage

### Functions

Define functions with the @custom_function decorator:

```python
from unity.function_manager.custom import custom_function

@custom_function()
async def process_data(input: str) -> str:
    \"\"\"Process the input data.\"\"\"
    return input.upper()

@custom_function(venv_name="ml_env", verify=True)
async def ml_inference(data: dict) -> dict:
    \"\"\"Run ML inference in a custom venv.\"\"\"
    import torch
    # ...
```

### Virtual Environments

Create `.toml` files in `custom/venvs/` with pyproject.toml content:

```toml
# custom/venvs/ml_env.toml
[project]
name = "ml-env"
version = "0.1.0"
dependencies = ["torch>=2.0", "transformers>=4.30"]
```

## Decorator Options

- `venv_name: Optional[str]` - Name of custom venv (filename without .toml)
- `venv_id: Optional[int]` - Direct venv ID (prefer venv_name for custom venvs)
- `verify: bool = True` - Whether to verify function execution
- `precondition: Optional[dict]` - Required state before execution
- `auto_sync: bool = True` - Set to False to exclude from auto-sync
- `windows_os_required: bool = False` - Route execution to Windows VM when True

## Best Practice: Import Runtime Domain Types

Custom compositional functions are frequently executed by retrieving a callable from the
`FunctionManager` (e.g. semantic search) and running it in a fresh sandbox namespace.

Simple rule: **if you use a symbol in the function body, import/define it inside the function**.

Example:

```python
# ✅ OK: "User" is only an annotation (forward-ref string)
@custom_function()
async def greet(user: "User") -> str:
    return f"Hello {user.name}"

# ✅ Preferred: import the runtime type inside the function if you need it
@custom_function()
async def is_admin(role: "Role") -> bool:
    from my_app.types import Role
    return role == Role.ADMIN
```

## Sync Behavior

- Venvs are synced first, then functions
- `venv_name` is resolved to `venv_id` during function sync
- Hash-based change detection avoids unnecessary updates
- Removed from source = deleted from database
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class CustomFunctionMetadata:
    """Metadata attached to a custom function by the decorator."""

    venv_name: Optional[str] = None
    venv_id: Optional[int] = None
    verify: bool = True
    precondition: Optional[Dict[str, Any]] = None
    auto_sync: bool = True
    windows_os_required: bool = False


def custom_function(
    *,
    venv_name: Optional[str] = None,
    venv_id: Optional[int] = None,
    verify: bool = True,
    precondition: Optional[Dict[str, Any]] = None,
    auto_sync: bool = True,
    windows_os_required: bool = False,
) -> Callable:
    """
    Decorator to mark a function for auto-sync to Functions/Compositional.

    Args:
        venv_name: Name of a custom venv (from custom/venvs/<name>.toml).
                   Resolved to venv_id during sync. Preferred for custom venvs.
        venv_id: Direct virtual environment ID. Use for non-custom venvs or
                 when the ID is known. If both venv_name and venv_id are set,
                 venv_name takes precedence.
        verify: Whether the Actor should verify function execution (default True).
        precondition: Optional dict specifying required state before execution.
        auto_sync: If False, function is excluded from auto-sync (default True).
        windows_os_required: If True, function executes on remote Windows VM
                             when assistant has desktop_mode='windows' and
                             is_user_desktop=False. Use for Windows-only libraries
                             like xlwings or COM automation.

    Example:
        @custom_function(venv_name="ml_env")
        async def my_function(x: int) -> int:
            '''Run in the ml_env virtual environment.'''
            import torch
            return x * 2

        @custom_function(venv_name="excel_env", windows_os_required=True)
        async def process_excel(path: str) -> dict:
            '''Run on Windows VM with xlwings.'''
            import xlwings as xw
            return {"sheets": 1}
    """

    def decorator(func: Callable) -> Callable:
        # Attach metadata to the function
        func._custom_function_metadata = CustomFunctionMetadata(
            venv_name=venv_name,
            venv_id=venv_id,
            verify=verify,
            precondition=precondition,
            auto_sync=auto_sync,
            windows_os_required=windows_os_required,
        )
        return func

    return decorator


# Re-export for convenient imports
__all__ = ["custom_function", "CustomFunctionMetadata"]
