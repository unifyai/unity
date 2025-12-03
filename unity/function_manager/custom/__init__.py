"""
Custom compositional functions that are auto-synced to Functions/Compositional.

This folder contains Python files with function definitions that are automatically
synchronized to the database when the FunctionManager initializes. This enables
forward-deployed engineers to add client-specific functions directly in source code.

## Usage

Define functions with the @custom_function decorator:

```python
from unity.function_manager.custom import custom_function

@custom_function()
async def process_data(input: str) -> str:
    \"\"\"Process the input data.\"\"\"
    return input.upper()

@custom_function(venv_id=1, verify=True, precondition={"url": "https://..."})
async def ml_inference(data: dict) -> dict:
    \"\"\"Run ML inference in a custom venv.\"\"\"
    import torch
    # ...
```

## Decorator Options

- `venv_id: Optional[int]` - Run in a custom virtual environment
- `verify: bool = True` - Whether to verify function execution
- `precondition: Optional[dict]` - Required state before execution
- `auto_sync: bool = True` - Set to False to exclude from auto-sync

## Sync Behavior

- Functions are matched by name
- Hash-based change detection avoids unnecessary updates
- Local source always wins over user-added functions with same name
- Functions removed from source are deleted from the database
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class CustomFunctionMetadata:
    """Metadata attached to a custom function by the decorator."""

    venv_id: Optional[int] = None
    verify: bool = True
    precondition: Optional[Dict[str, Any]] = None
    auto_sync: bool = True


def custom_function(
    *,
    venv_id: Optional[int] = None,
    verify: bool = True,
    precondition: Optional[Dict[str, Any]] = None,
    auto_sync: bool = True,
) -> Callable:
    """
    Decorator to mark a function for auto-sync to Functions/Compositional.

    Args:
        venv_id: Optional virtual environment ID for isolated execution.
        verify: Whether the Actor should verify function execution (default True).
        precondition: Optional dict specifying required state before execution.
        auto_sync: If False, function is excluded from auto-sync (default True).

    Example:
        @custom_function(venv_id=1)
        async def my_function(x: int) -> int:
            '''My function docstring.'''
            return x * 2
    """

    def decorator(func: Callable) -> Callable:
        # Attach metadata to the function
        func._custom_function_metadata = CustomFunctionMetadata(
            venv_id=venv_id,
            verify=verify,
            precondition=precondition,
            auto_sync=auto_sync,
        )
        return func

    return decorator


# Re-export for convenient imports
__all__ = ["custom_function", "CustomFunctionMetadata"]
