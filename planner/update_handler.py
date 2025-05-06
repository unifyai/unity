import inspect
from . import primitives, zero_shot, sys_msg
from .unify_client import (
    generate_prompt,
    set_system_message,
    set_stateful as _set_stateful,
)
from .context import context
from .code_rewriter import rewrite_function
import logging
import difflib
import json

"""
Update handler module for the planner.

This module provides functions to handle user updates during plan execution.
After rewriting a target function on modify, we now generate and schedule a 
course_correction() state-sync helper instead of an ad-hoc bridge.
"""


def _build_course_payload():
    """
    Build a JSON payload containing browser state and call stack information
    for course correction.

    Returns:
        JSON string containing url, scroll position, and target entry function
    """
    # Get current browser state
    browser_state = context.last_state_snapshot()
    url = browser_state.get("url", "") if browser_state else ""
    scroll_y = browser_state.get("scroll_y", 0) if browser_state else 0

    # Get the current entry function from the call stack
    call_stack = context.get_call_stack()
    target_entry_fn = call_stack[-1] if call_stack else ""

    # Build the payload
    payload = {"url": url, "scroll_y": scroll_y, "target_entry_fn": target_entry_fn}

    return json.dumps(payload, indent=2)


def _schedule_and_resume(planner, fn_name):
    """
    Schedule a function to be executed via the call queue and resume the plan afterward.

    Args:
        planner: The planner instance
        fn_name: The name of the function to execute
    """

    def wrapper_function():
        try:
            # Get and execute the function
            fn = getattr(planner._plan_module, fn_name)
            fn()
        except Exception as e:
            logging.error(f"Error executing {fn_name} function: {e}")
        finally:
            # Clean up by removing the function from the module
            if hasattr(planner._plan_module, fn_name):
                delattr(planner._plan_module, fn_name)
            # Resume the plan
            planner._resume()

    # Schedule the wrapper function to be executed via the call queue
    planner._call_queue.put(wrapper_function)


def _build_diff_payload(fn_name, old_src, new_src):
    """
    Build a JSON payload containing diff information, call stack context, and browser state.

    Args:
        fn_name: Name of the function being modified
        old_src: Original source code of the function
        new_src: New source code of the function

    Returns:
        JSON string containing diff, call stack, and browser state information
    """
    # Generate unified diff
    diff = difflib.unified_diff(
        old_src.splitlines(keepends=True),
        new_src.splitlines(keepends=True),
        fromfile=f"a/{fn_name}",
        tofile=f"b/{fn_name}",
        n=3,
    )
    diff_text = "".join(diff)

    # Get call stack context (last 3 frames)
    call_stack = context.get_call_stack()
    stack_context = call_stack[-3:] if len(call_stack) >= 3 else call_stack

    # Get current browser state
    browser_state = context.last_state_snapshot()

    # Build the payload
    payload = {
        "diff": diff_text,
        "function_name": fn_name,
        "call_stack": stack_context,
        "browser_state": browser_state,
    }

    return json.dumps(payload, indent=2)


def _select_stack_function(update_text: str) -> str:
    """
    Select the most appropriate function from the call stack to modify based on the update text.

    Args:
        update_text: The user's update request text

    Returns:
        The name of the function to modify
    """
    # Get the current call stack
    call_stack = context.get_call_stack()

    if not call_stack:
        # If call stack is empty, return None to indicate no function was selected
        return None

    # Set system message and disable stateful context
    set_system_message(sys_msg.SELECT_STACK_FUNCTION_SYS_MSG)
    _set_stateful(False)

    # Format the call stack for the prompt
    call_stack_str = "Current call stack (most recent first):\n" + "\n".join(
        [f"- {func}" for func in reversed(call_stack)]
    )

    # Ask which function to patch
    prompt = sys_msg.SELECT_STACK_FUNCTION_PROMPT.format(
        update_text=update_text, call_stack_str=call_stack_str
    )

    selected_function = generate_prompt(prompt).strip()

    # Verify the selected function is actually in the call stack
    if selected_function in call_stack:
        return selected_function

    # If the selected function is not in the call stack, default to the top-most function
    return call_stack[-1] if call_stack else None


def _classify_update(update_text: str) -> str:
    """
    Classify the user update as 'exploratory' or 'modify' using the Unify agent.
    """
    # Set system message with explicit schema and disable stateful context
    set_system_message(sys_msg.CLASSIFY_UPDATE_SYS_MSG)
    _set_stateful(False)

    prompt = sys_msg.CLASSIFY_UPDATE_PROMPT.format(update_text=update_text)
    response = generate_prompt(prompt).strip().lower()

    # Strict matching for exact responses
    if response == "exploratory":
        return "exploratory"
    if response == "modify":
        return "modify"

    # Fallback pattern matching if exact match fails
    if "explor" in response:
        return "exploratory"
    if "modify" in response or "modif" in response:
        return "modify"

    # Default to modify if uncertain
    return "modify"


def _handle_exploration(planner, update_text: str) -> None:
    """
    Handle an exploratory user update by generating a Python function that reuses a single
    exploration tab (or creates one if needed), performs exploration, and signals completion.
    The function is executed via the call queue while the main plan is paused.
    """
    # Pause the current plan execution
    planner._pause()

    # Record the original URL from the current browser state
    browser_state = context.last_state_snapshot()
    original_tab_id = browser_state.get("active_tab", "") if browser_state else ""

    # Check if we already have an exploration tab
    exploration_tab_id = context.get_exploration_tab()

    # Build a zero-shot prompt to generate the exploration function
    set_system_message(sys_msg.EXPLORATION_FUNCTION_SYS_MSG)
    _set_stateful(False)

    # Create the prompt for generating the exploration function
    tab_action = "Uses" if exploration_tab_id else "Creates"
    exploration_tab_info = (
        f"Existing exploration tab ID: {exploration_tab_id}"
        if exploration_tab_id
        else "No existing exploration tab"
    )

    prompt = sys_msg.EXPLORATION_FUNCTION_PROMPT.format(
        tab_action=tab_action,
        update_text=update_text,
        original_tab_id=original_tab_id,
        exploration_tab_info=exploration_tab_info,
    )

    # Generate the exploration function code
    exploration_code = generate_prompt(prompt)

    try:
        # Execute the generated code in the planner module's namespace
        exec(exploration_code, planner._plan_module.__dict__)

        # Get the function from the module
        exploratory_task = getattr(planner._plan_module, "exploratory_task")

        # Create a wrapper function that manages the exploration context
        def wrapper_function():
            try:
                # Create a new tab if needed and enter exploration mode
                if not exploration_tab_id:
                    new_tab_id = primitives.new_tab()
                    context.enter_exploration(new_tab_id, original_tab_id)
                else:
                    # Reuse existing exploration tab
                    primitives.select_tab(exploration_tab_id)
                    # Increment exploration depth counter
                    context.enter_exploration(exploration_tab_id, original_tab_id)

                # Execute the exploration task
                exploratory_task()

            except Exception as e:
                logging.error(f"Error during exploration: {e}")
            finally:
                # Exit exploration mode and return to original tab
                context.exit_exploration()
                if original_tab_id:
                    primitives.select_tab(original_tab_id)

                # Resume the plan
                planner._resume()

        # Schedule the wrapper function to be executed via the call queue
        planner._call_queue.put(wrapper_function)

    except Exception as e:
        # Fallback if there's an error with the generated code
        logging.error(f"Error executing exploration code: {e}")

        # Define a simple fallback function with resume at the start
        def fallback_wrapper():
            try:
                # Create a new tab if needed and enter exploration mode
                if not exploration_tab_id:
                    new_tab_id = primitives.new_tab()
                    context.enter_exploration(new_tab_id)
                else:
                    # Reuse existing exploration tab
                    primitives.select_tab(exploration_tab_id)
                    # Increment exploration depth counter
                    context.enter_exploration(exploration_tab_id)

                # Simple exploration
                primitives.search(update_text)
                primitives.wait_for_user_signal()

            except Exception as e:
                logging.error(f"Error during fallback exploration: {e}")
            finally:
                # Exit exploration mode and return to original tab
                context.exit_exploration()
                if original_tab_id:
                    primitives.select_tab(original_tab_id)

                # Resume the plan
                planner._resume()

        # Schedule the fallback wrapper function
        planner._call_queue.put(fallback_wrapper)


def _handle_modification(planner, update_text: str) -> None:
    """
    Handle a modifying user update by identifying the target function,
    rewriting it, and scheduling a course correction function.
    """
    # Pause the plan execution
    planner._pause()

    # Use the new stack selection function to identify the target function
    target_function_name = _select_stack_function(update_text)

    # If no function was selected from the stack, fall back to the plan sketch approach
    if not target_function_name:
        # Create a plan sketch by listing all functions in the module
        plan_sketch = ""
        for name, obj in planner._plan_module.__dict__.items():
            if callable(obj) and not name.startswith("_") and inspect.isfunction(obj):
                plan_sketch += f"- {name}\n"

        # Set system message and disable stateful context
        set_system_message(sys_msg.SELECT_PLAN_FUNCTION_SYS_MSG)
        _set_stateful(False)

        # Ask which function to patch with improved context
        prompt_node = sys_msg.SELECT_PLAN_FUNCTION_PROMPT.format(
            update_text=update_text, plan_sketch=plan_sketch
        )
        target_function_name = generate_prompt(prompt_node).strip()

    # Get the target function from the module
    target_function = None
    try:
        target_function = getattr(planner._plan_module, target_function_name)
    except AttributeError:
        logging.warning(
            f"Warning: Function '{target_function_name}' not found in the current plan."
        )
        # If the function doesn't exist, we'll create a course correction function instead

    # Get the original source code if the target function exists
    old_src = ""
    if target_function and callable(target_function):
        try:
            old_src = inspect.getsource(target_function)
        except Exception as e:
            logging.error(f"Error getting source for function: {e}")

    # Generate revised code for the target function
    new_src = ""
    if target_function and callable(target_function):
        prompt_rewrite = sys_msg.REWRITE_FUNCTION_PROMPT.format(
            target_function_name=target_function_name,
            update_text=update_text,
            old_src=old_src,
        )
        new_src = generate_prompt(prompt_rewrite)

        try:
            # Rewrite the function using the code_rewriter
            rewrite_function(target_function, new_src)

            # Propagate docstring changes to parent function if available
            call_stack = context.get_call_stack()
            if len(call_stack) >= 2:
                parent_fn_name = call_stack[-2]
                try:
                    parent_fn = getattr(planner._plan_module, parent_fn_name)
                    if callable(parent_fn):
                        parent_src = inspect.getsource(parent_fn)
                        # Add minimal docstring update referencing the update text
                        updated_docstring = f'"""\n{parent_fn.__doc__ or ""}\n\nUpdated to handle: {update_text}\n"""'
                        rewrite_function(parent_fn, parent_src, updated_docstring)
                except Exception as e:
                    logging.error(f"Error updating parent function docstring: {e}")
        except Exception as e:
            logging.error(f"Error rewriting function: {e}")

    # Build the course payload with browser state and call stack information
    course_payload = _build_course_payload()

    # Generate a course correction function to handle the update
    set_system_message(sys_msg.COURSE_CORRECTION_SYS_MSG)

    prompt_cc = sys_msg.COURSE_CORRECTION_PROMPT.format(
        update_text=update_text, course_payload=course_payload
    )

    course_correction_code = generate_prompt(prompt_cc)

    try:
        # Execute the generated code in the planner module's namespace
        exec(course_correction_code, planner._plan_module.__dict__)

        # Check if we need to apply verification decorator
        if hasattr(planner._plan_module, "verify"):
            # Apply verification decorator if available
            verify = getattr(planner._plan_module, "verify")
            course_correction_fn = verify(
                getattr(planner._plan_module, "course_correction")
            )
            # Update the module with the decorated function
            setattr(planner._plan_module, "course_correction", course_correction_fn)

        # Schedule the course correction function and resume the plan
        _schedule_and_resume(planner, "course_correction")

    except Exception as e:
        # Fallback if there's an error with the generated code
        logging.error(f"Error executing course correction code: {e}")

        # Define a simple fallback function with resume at the start
        def fallback_wrapper():
            try:
                # Simple fallback behavior
                primitives.search(update_text)
            except Exception as fallback_error:
                logging.error(f"Error in fallback handler: {fallback_error}")
            finally:
                # Always resume the plan
                planner._resume()

        # Schedule the fallback wrapper function
        planner._call_queue.put(fallback_wrapper)


def handle_update(planner, update_text: str) -> None:
    """
    Entrypoint to process a user update and dispatch to exploration or modification.

    For exploratory updates:
    - Creates or reuses an exploration tab
    - Executes the exploration in a dedicated context
    - Returns to the original tab when done

    For modification updates:
    - Identifies and rewrites the target function
    - Generates a course_correction() function to sync browser state
    - Schedules the course_correction to run before resuming the plan

    Uses the pause/resume mechanism with callable scheduling to ensure
    clean transitions between plan states.

    Args:
        planner: The planner instance
        update_text: The user's update request text
    """
    kind = _classify_update(update_text)
    if kind == "exploratory":
        _handle_exploration(planner, update_text)
    else:
        _handle_modification(planner, update_text)
