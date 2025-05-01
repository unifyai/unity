from .model import FunctionNode, Plan
from . import primitives, zero_shot
from .unify_client import (
    generate_prompt,
    set_system_message,
    client,
    set_stateful as _set_stateful,
)
from .context import context


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


def _find_node(root: FunctionNode, name: str) -> FunctionNode:
    """
    Recursively search for a FunctionNode with the given name.
    """
    if root.name == name:
        return root
    for child in root.body:
        if isinstance(child, FunctionNode):
            result = _find_node(child, name)
            if result:
                return result
    return None


def _handle_exploration(planner, update_text: str) -> None:
    """
    Handle an exploratory user update by opening a new tab, cloning state, and closing it.
    """
    # Pause the current plan
    planner.paused = True

    # Record the original URL from the current browser state
    browser_state = context.last_state_snapshot()
    if browser_state:
        original_url = browser_state.get("url")
        original_tab_id = browser_state.get("active_tab")

    # Build an exploration plan: open new tab, clone placeholder, close tab, return to original
    body = [
        primitives.new_tab(),
        # Use the update text as the search query for better exploration
        primitives.search(update_text),
    ]

    # Add navigation back to original state when closing
    if original_tab_id:
        body.append(primitives.close_this_tab())
        body.append(primitives.select_tab(original_tab_id))
    elif original_url:
        body.append(primitives.close_this_tab())
        body.append(primitives.go_back())
    else:
        body.append(primitives.close_this_tab())
    exploration_node = FunctionNode("exploration_temp", body)

    # Create exploration plan with same task completion queue
    original_plan = planner.current_plan
    exploration_plan = Plan(exploration_node, planner._task_completion_q)
    exploration_plan.task_completion_q = original_plan.task_completion_q

    # Push current plan and start exploration
    planner.plan_stack.append(original_plan)
    planner.current_plan = exploration_plan
    planner.paused = False


def _handle_modification(planner, update_text: str) -> None:
    """
    Handle a modifying user update by rewriting a function node and course-correcting.
    """
    # Pause execution
    planner.paused = True

    # Get the current call stack
    call_stack = context.get_call_stack()
    call_stack_str = (
        "Current call stack (most recent first):\n"
        + "\n".join([f"- {func}" for func in reversed(call_stack)])
        if call_stack
        else "Call stack is empty"
    )

    # Create a plan sketch for context
    plan_sketch = ""
    if planner.current_plan and planner.current_plan.root:
        plan_sketch = _create_plan_sketch(planner.current_plan.root)

    # Set system message and disable stateful context
    system_message = "You are an assistant that identifies which function in a plan needs to be modified based on a user update."
    set_system_message(system_message)
    _set_stateful(False)

    # Ask which function to patch with improved context
    prompt_node = (
        f"Given the user update: {update_text}\n\n"
        f"{call_stack_str}\n\n"
        f"Plan structure:\n{plan_sketch}\n\n"
        "Which function in the current plan should be patched to address this update?\n"
        "Respond with just the exact function name, no additional text."
    )
    node_name = generate_prompt(prompt_node).strip()

    # Locate the target FunctionNode
    root = planner.current_plan.root
    target_node = _find_node(root, node_name)

    # If node not found, default to root-level patching
    if not target_node:
        # Log the fallback
        print(
            f"Warning: Function '{node_name}' not found in the current plan. Falling back to root-level patching."
        )
        target_node = root

    # Generate revised code for that function via zero-shot
    prompt_rewrite = (
        f"Rewrite the function '{target_node.name}' to satisfy: {update_text}.\n"
        "Use only the primitive helpers and preserve signature and docstring.\n"
        "Ensure the code is complete and syntactically correct."
    )
    new_code = generate_prompt(prompt_rewrite)

    try:
        # Parse the new function into a FunctionNode
        new_plan = zero_shot._parse_generated_code(new_code)
        new_node = new_plan.root

        # Preserve parent link and replace in tree
        parent = target_node.parent
        new_node.parent = parent
        if parent:
            idx = parent.body.index(target_node)
            parent.body[idx] = new_node
        else:
            # Root-level replacement
            planner.current_plan.root = new_node
            planner.current_plan.current_node = new_node

        # Create a flat course-correction plan from the new node's primitives
        flat_node = FunctionNode("course_correction", list(new_node.body))
        course_plan = Plan(flat_node, planner._task_completion_q)

        # Store original plan state for restoration
        original_plan = planner.current_plan

        # Push the modified original plan and execute course correction
        planner.plan_stack.append(original_plan)
        planner.current_plan = course_plan

        # Ensure the course correction plan uses the same task completion queue
        course_plan.task_completion_q = original_plan.task_completion_q

    except Exception as e:
        # Handle parsing errors by falling back to simpler approach
        print(
            f"Error parsing generated code: {e}. Falling back to root-level patching."
        )

        # Create a simple correction with just a search primitive as fallback
        fallback_node = FunctionNode(
            "fallback_correction", [primitives.search(update_text)]
        )
        course_plan = Plan(fallback_node, planner._task_completion_q)
        planner.plan_stack.append(planner.current_plan)
        planner.current_plan = course_plan

    planner.paused = False


def _create_plan_sketch(node: FunctionNode, depth: int = 0) -> str:
    """
    Create a hierarchical sketch of the plan structure for better context.
    """
    indent = "  " * depth
    result = f"{indent}- {node.name}\n"

    for child in node.body:
        if isinstance(child, FunctionNode):
            result += _create_plan_sketch(child, depth + 1)

    return result


def handle_update(planner, update_text: str) -> None:
    """
    Entrypoint to process a user update and dispatch to exploration or modification.
    """
    kind = _classify_update(update_text)
    if kind == "exploratory":
        _handle_exploration(planner, update_text)
    else:
        _handle_modification(planner, update_text)

    # Ensure planner is unpaused after handling the update
    planner.paused = False
