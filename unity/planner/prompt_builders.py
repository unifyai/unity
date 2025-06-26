from __future__ import annotations

import inspect
import textwrap
from typing import Callable, Dict


def _get_tools_doc(tools: Dict[str, Callable]) -> str:
    """Generates a markdown list of tools and their signatures."""
    doc_lines = []
    for name, func in tools.items():
        try:
            sig = str(inspect.signature(func))
            sig = sig.replace("(self, ", "(")
            sig = sig.replace("-> 'BrowserSessionHandle'", "")
            sig = sig.replace("-> 'PhoneCallHandle'", "")
            doc_lines.append(f"- `coms_manager.{name}{sig}`")
        except (ValueError, TypeError):
            doc_lines.append(f"- `coms_manager.{name}` (signature not available)")
    return "\n".join(doc_lines)


def build_initial_plan_prompt(
    goal: str,
    tools: Dict[str, Callable],
    existing_functions: dict,
    retry_msg: str,
    exploration_summary: str | None,
) -> str:
    """
    Builds the system prompt for generating an initial plan.

    Args:
        goal: The user's goal.
        tools: A dictionary of available tools from the ComsManager.
        existing_functions: A dictionary of reusable functions from the library.
        retry_msg: A message to include if a previous attempt failed.
        exploration_summary: Context from the exploration phase.

    Returns:
        The complete prompt string.
    """
    tools_doc = _get_tools_doc(tools)
    usage_examples = textwrap.dedent(
        """
        ### Tool Usage Examples

        **Browser Interaction:**
        ```python
        @verify
        async def check_unify_blog():
            # Each function needing browser access must start its own session.
            async with coms_manager.start_browser_session() as browser:
                # Use 'act' for all actions. Describe what you want to do.
                await browser.act("Navigate to unify.ai")
                await browser.act("Click the 'Blog' link in the main navigation")
                # Use 'observe' to get information from the page.
                blog_title = await browser.observe("What is the title of the first blog post?")
            return blog_title
        ```

        **Making a Call:**
        ```python
        @verify
        async def confirm_appointment():
            async with coms_manager.make_call(contact_id=123, purpose="Confirm appointment") as call:
                response = await call.ask("Are you still available for our 2pm meeting tomorrow?")
            return response
        ```
    """
    )
    existing_functions_doc = (
        "\n".join(
            f'- `{name}{data["argspec"]}`: {data["docstring"]}'
            for name, data in existing_functions.items()
        )
        or "None."
    )
    full_existing_code = "\n\n".join(
        data["implementation"] for data in existing_functions.values()
    )

    exploration_context = (
        f"**Context from Initial Exploration:**\n{exploration_summary}"
        if exploration_summary
        else ""
    )
    return textwrap.dedent(
        f"""
        You are an expert Python programmer tasked with generating a complete, single-file script to achieve a user's goal.

        **Primary Goal:** "{goal}"
        {exploration_context}
        {retry_msg}

        ---
        ### Core Instructions & Rules
        1.  **Single Code Block:** Your entire response MUST be a single, valid Python code block.
        2.  **Entry Point:** The script's main entry point MUST be `async def main_plan()`.
        3.  **Decomposition:** Break down complex problems into smaller, logical, self-contained `async def` helper functions.
        4.  **Decorators & Docstrings:** Every function you define MUST be decorated with `@verify` and include a concise one-line docstring.
        5.  **No Imports:** You MUST NOT use any `import` statements.

        ---
        ### Tool Usage Rules
        1.  **Strategy:** Your primary strategy MUST be to **Observe, then Act**. Before performing an action on an unfamiliar screen, use `browser.observe()` to understand the page structure and available elements. This will help you write more accurate `act()` commands.
        2.  **Granularity:** Break down complex UI interactions into a series of simple, single-purpose `act()` calls. Instead of a single command like `act("Select Urdu from the dropdown")`, you should generate:
            ```python
            await browser.act("Click the target language dropdown")
            await browser.act("Click the 'Urdu' option in the language list")
            ```
        3.  **Toolbox:** You have access to a global `coms_manager` object for all external interactions.
        4.  **Context Managers (`async with`):** Tools that create a session (`start_browser_session`, `Phone`) MUST be used within an `async with` block to ensure they are safely closed.
        5.  **Browser Handle API:** The `browser` handle returned by `start_browser_session` has **ONLY TWO** available methods:
            - `act(instruction: str)`: Give a natural language command for what to DO.
            - `observe(query: str)`: Ask a natural language question to GET information from the page.
            - **CRITICAL:** You MUST NOT attempt to call any other methods like `.click()`, `.navigate_to_url()`, or `.get_text()` on the `browser` handle. All actions are done through `act()`.
        6.  **Self-Contained Functions:** Each helper function should be independent. If a function needs to use the browser, it must create its own session using `async with coms_manager.start_browser_session() as browser:`. Do NOT pass browser handles as arguments to other functions.

        ---
        ### Tools Reference
        You have access to a global `coms_manager` object with the following methods:
        {tools_doc}

        ---
        ### Usage Examples
        {usage_examples}
        ---
        ### Existing Functions Library
        You may use these pre-existing functions if they are suitable.
        {existing_functions_doc}

        ---
        ### Code of Available Functions
        ```python
        {full_existing_code or "# No pre-existing functions were provided."}
        ```

        Begin your response now. Your response must start immediately with the code.
        """,
    )


def build_dynamic_implement_prompt(
    function_name: str,
    func_sig: inspect.Signature,
    goal: str,
    parent_code: str,
    browser_state: str,
    replan_context: str,
) -> str:
    """
    Builds the system prompt for dynamically implementing a function.

    Args:
        function_name: The name of the function to implement.
        func_sig: The signature of the function to implement.
        goal: The overall user goal.
        parent_code: The source code of the calling function.
        browser_state: A description of the current browser state.
        replan_context: A message providing context for a replan.

    Returns:
        The complete prompt string.
    """
    return textwrap.dedent(
        f"""
        You are an expert Python programmer. Your task is to write the implementation for the function `{function_name}`.

        **Overall Goal:** "{goal}"
        **Function to Implement:** `async def {function_name}{func_sig}`
        **Parent Function (for context):**
        ```python
        {parent_code}
        ```
        **Current Browser State:**
        {browser_state}

        {replan_context}

        ---
        ### Instructions & Rules
        1.  **Code Only:** Your output MUST be ONLY the Python code for the function, starting with the `@verify` decorator and the function definition. Do not include explanations or markdown.
        2.  **Strategy:** You are likely being asked to implement this because a previous attempt failed. Your first step should be to **re-assess the environment**. Use `browser.observe()` to confirm the current page is correct before attempting any actions. If the state is wrong, generate code to correct it first.
        3.  **Granularity:** Break down complex UI interactions into a series of simple, single-purpose `act()` calls.
        4.  **Add a Docstring:** The function implementation MUST include a concise one-line docstring explaining its purpose.
        5.  **Use Context Managers:** For tools that return a handle (`Phone`, `start_browser_session`), you MUST use an `async with` block.
        6.  **Tool Usage:** Use the `coms_manager` global object to interact with the environment. The `browser` handle from `start_browser_session` has ONLY two methods: `act(instruction: str)` and `observe(query: str)`. You MUST NOT call hallucinated methods like `.click()` or `.navigate()`.
        7.  **No Imports:** `import` statements are forbidden.

        Begin your response now.
        """,
    )


def build_verification_prompt(
    goal: str,
    function_name: str,
    function_docstring: str | None,
    interactions: list,
) -> str:
    """
    Builds the prompt for verifying a function's execution.

    Args:
        goal: The overall user goal.
        function_name: The name of the function being verified.
        function_docstring: The docstring of the function.
        interactions: A log of `act` and `observe` calls made.

    Returns:
        The complete prompt string for the verification LLM call.
    """
    interactions_log = (
        "\n".join(
            (
                f"- Action: `{act}`, Observation: `{obs or 'N/A'}`"
                if kind == "observe"
                else f"- Action: `{act}` with result `{obs}`"
            )
            for kind, act, obs in interactions
        )
        or "No browser actions were logged for this step."
    )

    return textwrap.dedent(
        f"""
        You are a meticulous verification agent. Your task is to assess if the executed actions successfully achieved the function's intended purpose, in the context of the overall goal.

        **Overall User Goal:** "{goal}"
        **Function Under Review:** `{function_name}`
        **Purpose of this function:** {function_docstring or 'No docstring provided.'}

        **Execution Log (Primitives Used):**
        {interactions_log}

        ---
        ### Assessment Task
        Based on the function's purpose and the execution log, provide your assessment as a single JSON object.
        - **Be pragmatic:** If the function's purpose is to gather data (like search results), and the log shows that the data was successfully retrieved, this should be considered a success (`ok`). The function does not need to perform extra analysis unless explicitly asked.
        - **Consider the overall goal:** If a function's individual purpose is unclear but its actions logically progress toward the overall user goal, you should also consider it a success (`ok`).
        - **Trust the code:** If the interaction log is empty, assess based on the function's likely outcome given its purpose and the overall goal. Assume it acted correctly unless there's a clear logical flaw.

        **Response Schema:**
        `{{"status": "...", "reason": "..."}}`

        **Valid Statuses:**
        - `ok`: The function's purpose was fully and correctly achieved.
        - `reimplement_local`: A tactical error occurred. The goal is correct, but the actions were wrong. The function needs to be re-written.
        - `replan_parent`: A strategic error occurred. The function itself is flawed or was called at the wrong time. The parent function needs to be replanned.
        - `fatal_error`: An unrecoverable error occurred that prevents any further progress.
        """,
    )


def build_plan_surgery_prompt(current_code: str, request: str) -> str:
    """
    Builds the prompt for modifying an existing plan script.

    Args:
        current_code: The current source code of the plan.
        request: The user's modification request.

    Returns:
        The complete prompt string.
    """
    return textwrap.dedent(
        f"""
        You are an expert Python programmer specializing in code modification. Your task is to rewrite an entire script to incorporate a user's change request.

        **Modification Request:**
        "{request}"

        ---
        ### Current Script
        ```python
        {current_code}
        ```

        ---
        ### Instructions & Rules
        1.  **Rewrite the Whole Script:** You must return a new, complete, and valid Python script that incorporates the requested change.
        2.  **Code Only:** Your response MUST be a single Python code block. Do NOT include any preamble, explanations, or markdown fences.
        3.  **Preserve Docstrings:** All functions should have a concise one-line docstring explaining their purpose.
        4.  **Preserve Decorators:** All `async def` functions MUST retain their `@verify` decorator.
        5.  **No Imports:** You MUST NOT use any `import` statements.
        6.  **Use Context Managers:** For tools that return a handle (`make_call`, `start_browser_session`), you MUST use an `async with` block.

        Begin your response now. Your response must start immediately with the code.
        """,
    )


def build_course_correction_prompt(
    old_code: str,
    new_code: str,
    current_state: str,
) -> str:
    """
    Builds the prompt to generate a course-correction script.

    Args:
        old_code: The previous version of the plan's code.
        new_code: The new version of the plan's code.
        current_state: A description of the current browser state.

    Returns:
        The complete prompt string.
    """
    return textwrap.dedent(
        f"""
        You are a state transition analyst. An agent's plan has been modified, and you must determine if its current state is compatible with the new plan. If not, you must generate a Python script to fix it.

        ---
        ### Context
        **Current Browser State:**
        {current_state}

        **Old Plan Snippet (for context):**
        ```python
        {old_code[:1000]}...
        ```

        **New Plan Code:**
        ```python
        {new_code}
        ```

        ---
        ### Task
        1.  Analyze if the **Current Browser State** is a suitable starting point for executing the **New Plan Code**.
        2.  If it is NOT suitable, write a script containing an `async def course_correction_main()` function. This script must use the `coms_manager` (via an `async with` block) to navigate to the correct starting state for the new plan. The script must be a single code block.
        3.  If the current state is already suitable, respond ONLY with the single word: `None`.
        4.  **CRITICAL**: The script MUST NOT use any `import` statements.

        Begin your response now.
        """,
    )


def build_exploration_prompt(goal: str, tools: Dict[str, Callable]) -> str:
    """
    Builds the system prompt for the pre-planning exploration phase.

    Args:
        goal: The high-level user goal.
        tools: A dictionary of available tools for exploration.

    Returns:
        The complete prompt string.
    """
    tool_doc = "\n".join(
        f"- `{name}{str(inspect.signature(func))}`" for name, func in tools.items()
    )
    return textwrap.dedent(
        f"""
        You are an intelligent Research Assistant. Your goal is to gather critical information to create a robust plan for the main objective.

        **Main Objective:** "{goal}"

        **Your Tools:**
        {tool_doc}

        **Your Task:**
        1. Think step-by-step to determine what information is needed.
        2. Use the `observe` tool to gather this information.
        3. If necessary, use `request_clarification` to ask for guidance.
        4. When you have gathered all necessary information, provide a final, concise summary of your findings. This summary will be used to generate the main plan. DO NOT say that you are ready; your final output MUST BE the summary itself.
        """,
    )


def build_ask_prompt(
    goal: str,
    state: str,
    call_stack: str,
    browser_context: str,
    context_log: str,
    question: str,
) -> str:
    """
    Builds the system prompt for answering questions about the plan's state.

    Args:
        goal: The overall goal of the plan.
        state: The current lifecycle state of the plan.
        call_stack: The current function call stack.
        browser_context: A summary of the current browser state.
        context_log: A log of recent actions.
        question: The user's question.

    Returns:
        The complete prompt string.
    """
    return textwrap.dedent(
        f"""
        You are an assistant analyzing an agent's state. Answer the user's question concisely based *only* on the provided context.

        **Goal:** {goal}
        **State:** {state}
        **Call Stack:** {call_stack}
        **Browser State:** {browser_context}
        **Recent Log:**
        {context_log}

        **Question:** "{question}"
        **Answer:**
    """,
    )
