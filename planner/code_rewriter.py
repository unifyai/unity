from typing import List, Callable, Optional
import inspect
import textwrap
import uuid

from .context import context
from .unify_client import set_system_message, generate_prompt
from . import sandbox
from . import sys_msg


def rewrite_function(fn: Callable, new_src: Optional[str] = None) -> None:
    """
    Rewrites the given function by either using the provided source code or
    prompting the LLM with its current code, call stack context, and browser state.
    Updates the function in its module.

    Args:
        fn: The function object to rewrite
        new_src: Optional source code to use instead of generating via LLM

    Raises:
        RuntimeError: If the function cannot be rewritten
    """
    if new_src is None:
        # Get the source code of the function
        try:
            source = inspect.getsource(fn)
        except (TypeError, OSError) as e:
            raise RuntimeError(f"Could not get source code for function: {e}")

        # Gather context for rewriting
        call_stack = context.get_call_stack()
        state_snapshot = context.last_state_snapshot()

        # Format the prompt with function details
        prompt = sys_msg.REWRITE_FUNCTION_PROMPT.format(
            fn_name=fn.__name__,
            source=source,
            call_stack=repr(call_stack),
            state_snapshot=repr(state_snapshot),
        )

        # Set system message and generate patched code
        set_system_message(sys_msg.REWRITE_FUNCTION_SYS_MSG)
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
