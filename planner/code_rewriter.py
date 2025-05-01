from typing import List, Callable
import inspect
import textwrap
import uuid

from .context import context
from .unify_client import set_system_message, generate_prompt
from . import sandbox


def rewrite_function(fn: Callable) -> None:
    """
    Rewrites the given function by prompting the LLM with its current code,
    call stack context, and browser state. Updates the function in its module.

    Args:
        fn: The function object to rewrite

    Raises:
        RuntimeError: If the function cannot be rewritten
    """
    # Get the source code of the function
    try:
        source = inspect.getsource(fn)
    except (TypeError, OSError) as e:
        raise RuntimeError(f"Could not get source code for function: {e}")

    # Gather context for rewriting
    call_stack = context.get_call_stack()
    state_snapshot = context.last_state_snapshot()

    # Construct prompt for Unify
    prompt_parts: List[str] = [
        f"# Original function: {fn.__name__}",
        f"# Function source code:",
        source,
        "# Call stack context:",
        repr(call_stack),
        "# Browser state snapshot:",
        repr(state_snapshot),
        f"# Instruction: Revise the function '{fn.__name__}' while preserving its overall purpose.",
    ]
    prompt = "\n".join(prompt_parts)

    # Set system message and generate patched code
    set_system_message(
        "You are an expert Python developer. Rewrite the provided function to improve it based on the context."
    )
    new_src = generate_prompt(prompt)
    if not isinstance(new_src, str):
        raise RuntimeError("Unify did not return valid code string for rewriting.")

    # Dedent the source code
    new_src = textwrap.dedent(new_src)

    # Validate the new code in sandbox first
    try:
        sandbox.exec_plan(new_src)
    except Exception as e:
        raise RuntimeError(f"Generated code failed sandbox validation: {e}")

    # Get the module where the function is defined
    module = inspect.getmodule(fn)
    if module is None:
        raise RuntimeError(f"Could not determine module for function {fn.__name__}")

    # Create a copy of the module's globals for execution
    exec_globals = module.__dict__.copy()

    # Generate a unique module name to avoid conflicts
    temp_module_name = f"_temp_module_{uuid.uuid4().hex}"
    exec_globals["__name__"] = temp_module_name

    # Execute the new code in the module's context
    try:
        exec(new_src, exec_globals)
    except Exception as e:
        raise RuntimeError(f"Failed to execute rewritten function: {e}")

    # Get the patched function from the execution context
    if fn.__name__ not in exec_globals:
        raise RuntimeError(
            f"Rewritten function {fn.__name__} not found in execution context"
        )

    patched_fn = exec_globals[fn.__name__]

    # Replace the original function in its module
    setattr(module, fn.__name__, patched_fn)
