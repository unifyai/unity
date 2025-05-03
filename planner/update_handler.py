import inspect
from . import primitives, zero_shot
from .unify_client import (
    generate_prompt,
    set_system_message,
    set_stateful as _set_stateful,
)
from .context import context
from .code_rewriter import rewrite_function
import logging
import difflib


# P4-BEGIN _select_stack_function
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
    system_message = "You are an assistant that identifies which function in a call stack needs to be modified based on a user update."
    set_system_message(system_message)
    _set_stateful(False)

    # Format the call stack for the prompt
    call_stack_str = "Current call stack (most recent first):\n" + "\n".join(
        [f"- {func}" for func in reversed(call_stack)]
    )

    # Ask which function to patch
    prompt = (
        f"Given the user update: {update_text}\n\n"
        f"{call_stack_str}\n\n"
        "Which function in the current call stack should be modified to address this update?\n"
        "Consider the semantic meaning of the function names and the user's request.\n"
        "Respond with just the exact function name from the stack, no additional text."
    )

    selected_function = generate_prompt(prompt).strip()

    # Verify the selected function is actually in the call stack
    if selected_function in call_stack:
        return selected_function

    # If the selected function is not in the call stack, default to the top-most function
    return call_stack[-1] if call_stack else None


# P4-END


def _classify_update(update_text: str) -> str:
    """
    Classify the user update as 'exploratory' or 'modify' using the Unify agent.
    """
    # Set system message with explicit schema and disable stateful context
    system_message = (
        "You are a classifier that determines if a user update is exploratory or modifying.\n"
        "An 'exploratory' update is when the user wants to investigate or explore something new.\n"
        "A 'modify' update is when the user wants to change or adjust the current plan."
    )
    set_system_message(system_message)
    _set_stateful(False)

    prompt = (
        f"Classify the following user update as either 'exploratory' or 'modify':\n\n"
        f"{update_text}\n\n"
        "An 'exploratory' update is when the user wants to investigate or explore something new.\n"
        "A 'modify' update is when the user wants to change or adjust the current plan.\n\n"
        "Respond with EXACTLY one word, either 'exploratory' or 'modify'. No other text."
    )
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


# P4-BEGIN _handle_exploration
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
    system_message = (
        "You are an expert Python programmer creating a browser automation function. "
        "The function should explore a topic in a dedicated tab and support nested explorations."
    )
    set_system_message(system_message)
    _set_stateful(False)

    # Create the prompt for generating the exploration function
    prompt = (
        f"Write a Python function named 'exploratory_task' that:\n"
        f"1. {'Uses' if exploration_tab_id else 'Creates'} a dedicated exploration tab\n"
        f"2. Searches for: '{update_text}'\n"
        f"3. Allows the user to explore and signals when exploration is complete\n"
        f"4. Returns to the original tab when done\n\n"
        f"Original tab ID: {original_tab_id}\n"
        f"{'Existing exploration tab ID: ' + exploration_tab_id if exploration_tab_id else 'No existing exploration tab'}\n\n"
        f"Use these primitives: new_tab(), search(), select_tab(), wait_for_user_signal()\n"
        f"The function should have no parameters and return None.\n"
        f"Provide ONLY the function code, no explanations."
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

    # P4-BEGIN modification_improvements
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
        system_message = "You are an assistant that identifies which function in a plan needs to be modified based on a user update."
        set_system_message(system_message)
        _set_stateful(False)

        # Ask which function to patch with improved context
        prompt_node = (
            f"Given the user update: {update_text}\n\n"
            f"Available functions:\n{plan_sketch}\n\n"
            "Which function in the current plan should be modified to address this update?\n"
            "Respond with just the exact function name, no additional text."
        )
        target_function_name = generate_prompt(prompt_node).strip()
    # P4-END modification_improvements

    # Get the target function from the module
    target_function = None
    try:
        target_function = getattr(planner._plan_module, target_function_name)
    except AttributeError:
        logging.warning(
            f"Warning: Function '{target_function_name}' not found in the current plan."
        )
        # If the function doesn't exist, we'll create a course correction function instead

    # P4-BEGIN diff_generation
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
        prompt_rewrite = (
            f"Rewrite the function '{target_function_name}' to satisfy: {update_text}.\n"
            "Use only the primitive helpers and preserve signature and docstring.\n"
            "Ensure the code is complete and syntactically correct.\n"
            f"Original code:\n{old_src}"
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

    # Generate a diff between old and new code
    diff_payload = ""
    if old_src and new_src:
        diff = difflib.unified_diff(
            old_src.splitlines(keepends=True),
            new_src.splitlines(keepends=True),
            fromfile=f"a/{target_function_name}",
            tofile=f"b/{target_function_name}",
            n=3,
        )
        diff_payload = "".join(diff)
    # P4-END diff_generation

    # Generate a course correction function to handle the update
    system_message = (
        "You are an expert Python programmer creating a browser automation function."
    )
    set_system_message(system_message)

    prompt_course_correction = (
        f"Write a Python function named 'course_correction' that implements this update: '{update_text}'.\n"
        f"The function should use browser primitives to make the necessary adjustments.\n"
        f"Use only these primitives: search(), click_button(), enter_text(), press_enter(), etc.\n"
        f"The function should have no parameters and return None.\n"
        f"Provide ONLY the function code, no explanations."
    )

    # Include diff information if available
    if diff_payload:
        prompt_course_correction += (
            f"\nThe following changes were made to the '{target_function_name}' function:\n"
            f"```\n{diff_payload}\n```\n"
            f"Your course_correction function should complement these changes.\n"
        )
    course_correction_code = generate_prompt(prompt_course_correction)

    try:
        # Execute the generated code in the planner module's namespace
        exec(course_correction_code, planner._plan_module.__dict__)

        # Get the function from the module
        course_correction = getattr(planner._plan_module, "course_correction")

        # Create a wrapper function that executes the course correction and resumes the plan
        def wrapper_function():
            planner._resume()
            course_correction()

        # Schedule the wrapper function to be executed via the call queue
        planner._call_queue.put(wrapper_function)

    except Exception as e:
        # Fallback if there's an error with the generated code
        logging.error(f"Error executing course correction code: {e}")

        # Define a simple fallback function with resume at the start
        def fallback_wrapper():
            planner._resume()
            primitives.search(update_text)

        # Schedule the fallback wrapper function
        planner._call_queue.put(fallback_wrapper)


def handle_update(planner, update_text: str) -> None:
    """
    Entrypoint to process a user update and dispatch to exploration or modification.
    Uses the new pause/resume mechanism with callable scheduling.
    """
    kind = _classify_update(update_text)
    if kind == "exploratory":
        _handle_exploration(planner, update_text)
    else:
        _handle_modification(planner, update_text)
