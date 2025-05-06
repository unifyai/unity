"""
System messages for the planner module.
"""

# Zero-shot planning prompt
ZERO_SHOT_PROMPT = """
You are an expert Python programmer tasked with creating execution plans.
Your plans must be valid Python code that can be parsed and executed.
Always define a root function named 'root_plan' and include primitive helper calls.

You are given a task: {task_str}

Use ONLY the following primitive helper functions (no others):
{helper_list}

For each primitive, here's its purpose:
{helper_descriptions}

Define a root Python function named `root_plan`.
Include at least one call to a primitive helper and stub any other logic
with `raise NotImplementedError` statements.

Only the FIRST primitive call will be executed; subsequent helper calls **must** be stubbed with `raise NotImplementedError` and decorated with `@verify`.

Return ONLY valid Python code without any explanations or markdown formatting.
"""

#### Update Handler Prompts ####
# Function selection from call stack
SELECT_STACK_FUNCTION_SYS_MSG = "You are an assistant that identifies which function in a call stack needs to be modified based on a user update."
SELECT_STACK_FUNCTION_PROMPT = """Given the user update: {update_text}

{call_stack_str}

Which function in the current call stack should be modified to address this update?
Consider the semantic meaning of the function names and the user's request.
Respond with just the exact function name from the stack, no additional text."""

# Update classification
CLASSIFY_UPDATE_SYS_MSG = """You are a classifier that determines if a user update is exploratory or modifying.
An 'exploratory' update is when the user wants to investigate or explore something new.
A 'modify' update is when the user wants to change or adjust the current plan."""
CLASSIFY_UPDATE_PROMPT = """Classify the following user update as either 'exploratory' or 'modify':

{update_text}

An 'exploratory' update is when the user wants to investigate or explore something new.
A 'modify' update is when the user wants to change or adjust the current plan.

Respond with EXACTLY one word, either 'exploratory' or 'modify'. No other text."""

# Exploration function generation
EXPLORATION_FUNCTION_SYS_MSG = """You are an expert Python programmer creating a browser automation function. 
The function should explore a topic in a dedicated tab and support nested explorations."""
EXPLORATION_FUNCTION_PROMPT = """Write a Python function named 'exploratory_task' that:
1. {tab_action} a dedicated exploration tab
2. Searches for: '{update_text}'
3. Allows the user to explore and signals when exploration is complete
4. Returns to the original tab when done

Original tab ID: {original_tab_id}
{exploration_tab_info}

Use these primitives: new_tab(), search(), select_tab(), wait_for_user_signal()
The function should have no parameters and return None.
Provide ONLY the function code, no explanations."""

# Function selection from plan sketch
SELECT_PLAN_FUNCTION_SYS_MSG = "You are an assistant that identifies which function in a plan needs to be modified based on a user update."
SELECT_PLAN_FUNCTION_PROMPT = """Given the user update: {update_text}

Available functions:
{plan_sketch}

Which function in the current plan should be modified to address this update?
Respond with just the exact function name, no additional text."""

# Function rewriting
REWRITE_FUNCTION_SYS_MSG = "You are an expert Python developer. Rewrite the provided function to improve it based on the context."
REWRITE_FUNCTION_PROMPT = """Rewrite the function '{target_function_name}' to satisfy: {update_text}.
Use only the primitive helpers and preserve signature and docstring.
Ensure the code is complete and syntactically correct.
Original code:
{old_src}"""

# Course correction function generation
COURSE_CORRECTION_SYS_MSG = "You are an expert Python programmer creating a browser automation course correction function."
COURSE_CORRECTION_PROMPT = """Write a Python function named 'course_correction' that syncs browser state before resuming execution after this update: '{update_text}'.
The function should use browser primitives to navigate to the correct URL, set scroll position, and ensure the browser is in the right state.
Use only these primitives: open_url, scroll_down, click_button, enter_text, press_enter, select_tab, wait_for_user_signal.
The function should have no parameters and return None.
The course_correction function will be executed once and then automatically removed.
Provide ONLY the function code, no explanations.

Context information:
```
{course_payload}
```
"""

#### Verifier Prompts ####
VERIFY_PRIMITIVE_PROMPT = """
Verify if this browser automation primitive succeeded:

Primitive information:
{payload_json}

Respond with exactly one of: "ok", "reimplement", or "push_up_stack"
Explanation:
- "ok" if the primitive achieved its intent
- "reimplement" if the primitive failed and should be rewritten
- "push_up_stack" if the intent should be handled at a higher level
"""
