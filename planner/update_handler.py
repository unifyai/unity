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


def _handle_exploration(planner, update_text: str) -> None:
    """
    Handle an exploratory user update by generating a Python function that opens a new tab,
    performs exploration, and then closes the tab. The function is executed via the call queue
    while the main plan is paused.
    """
    # Pause the current plan execution
    planner._pause()

    # Record the original URL from the current browser state
    browser_state = context.last_state_snapshot()
    original_url = browser_state.get("url", "") if browser_state else ""
    original_tab_id = browser_state.get("active_tab", "") if browser_state else ""

    # Build a zero-shot prompt to generate the exploration function
    system_message = (
        "You are an expert Python programmer creating a browser automation function."
        "The function should explore a topic in a new tab and then return to the original state."
    )
    set_system_message(system_message)
    _set_stateful(False)

    # Create the prompt for generating the exploration function
    prompt = (
        f"Write a Python function named 'exploratory_task' that:\n"
        f"1. Opens a new browser tab\n"
        f"2. Searches for: '{update_text}'\n"
        f"3. Closes the tab when done\n"
        f"4. Returns to the original state\n\n"
        f"Original URL: {original_url}\n"
        f"Original tab ID: {original_tab_id}\n\n"
        f"Use only these primitives: new_tab(), search(), close_this_tab(), select_tab(), go_back()\n"
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

        # Create a wrapper function that executes the task and resumes the plan
        def wrapper_function():
            planner._resume()
            exploratory_task()

        # Schedule the wrapper function to be executed via the call queue
        planner._call_queue.put(wrapper_function)

    except Exception as e:
        # Fallback if there's an error with the generated code
        logging.error(f"Error executing exploration code: {e}")

        # Define a simple fallback function with resume at the start
        def fallback_wrapper():
            planner._resume()
            # Define the fallback exploratory task
            primitives.new_tab()
            primitives.search(update_text)
            primitives.close_this_tab()
            if original_tab_id:
                primitives.select_tab(original_tab_id)
            elif original_url:
                primitives.go_back()

        # Schedule the fallback wrapper function
        planner._call_queue.put(fallback_wrapper)


def _handle_modification(planner, update_text: str) -> None:
    """
    Handle a modifying user update by identifying the target function,
    rewriting it, and scheduling a course correction function.
    """
    # Pause the plan execution
    planner._pause()

    # Get the current call stack
    call_stack = context.get_call_stack()
    call_stack_str = (
        "Current call stack (most recent first):\n"
        + "\n".join([f"- {func}" for func in reversed(call_stack)])
        if call_stack
        else "Call stack is empty"
    )

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
        f"{call_stack_str}\n\n"
        f"Available functions:\n{plan_sketch}\n\n"
        "Which function in the current plan should be modified to address this update?\n"
        "Respond with just the exact function name, no additional text."
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

    if target_function and callable(target_function):
        # Generate revised code for the target function
        prompt_rewrite = (
            f"Rewrite the function '{target_function_name}' to satisfy: {update_text}.\n"
            "Use only the primitive helpers and preserve signature and docstring.\n"
            "Ensure the code is complete and syntactically correct."
        )
        new_code = generate_prompt(prompt_rewrite)

        try:
            # Rewrite the function using the code_rewriter
            rewrite_function(target_function, new_code)
        except Exception as e:
            logging.error(f"Error rewriting function: {e}")

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
