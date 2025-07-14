from __future__ import annotations

import inspect
import textwrap
import json
from typing import Callable, Dict, Any, Optional, Type, List
from unity.common.llm_helpers import (
    class_api_overview,
    get_type_hints,
    SteerableToolHandle,
)


def _build_tool_signatures(tool_dict: Dict[str, Callable]) -> str:
    """
    Builds a JSON string of tool signatures.
    """
    tool_info = {}
    for name, fn in tool_dict.items():
        prefix = "async def " if inspect.iscoroutinefunction(fn) else "def "
        tool_info[name] = {
            "signature": f"{prefix}{name}{inspect.signature(fn)}",
            "docstring": inspect.getdoc(fn) or "No docstring available.",
        }
    return json.dumps(tool_info, indent=4)


def _build_handle_apis(tool_dict: Dict[str, Callable]) -> str:
    """
    Builds a consolidated block of API documentation for each unique handle
    type returned by the available tools.
    """
    handle_groups: Dict[Type[SteerableToolHandle], List[str]] = {}
    for name, func in tool_dict.items():
        try:
            hints = get_type_hints(func)
            return_type = hints.get("return")

            if (
                return_type
                and inspect.isclass(return_type)
                and issubclass(return_type, SteerableToolHandle)
            ):
                if return_type not in handle_groups:
                    handle_groups[return_type] = []
                handle_groups[return_type].append(f"`{name}`")
        except Exception:
            continue

    if not handle_groups:
        return "There are no special handle APIs for the available tools."

    final_docs = []
    for handle_class, tool_names in handle_groups.items():
        tools_list_str = ", ".join(sorted(tool_names))
        doc = f"**`{handle_class.__name__}` (returned by {tools_list_str})**\n"
        doc += "This handle represents an interactive session. Its available methods are:\n"
        doc += class_api_overview(handle_class)
        final_docs.append(doc)

    return "\n\n".join(final_docs)


def _build_rules_and_examples_prompt(
    tools: Dict[str, Callable],
    strategy_instruction: str,
    tool_usage_instruction: str,
) -> str:
    """Builds the reusable block of core rules and examples for code generation."""
    tool_reference = _build_tool_signatures(tools)
    handle_apis = _build_handle_apis(tools)

    strategy_instruction += textwrap.dedent(
        """\n
        ---
        ### Strategic Principles for Web Automation
        To create a robust plan, always consider these heuristics:
        1.  **Observe Before You Act**: Before attempting to click or type, use `browser_observe` to confirm the element is present and to get its precise text or description. You cannot act on what you cannot see.
        2.  **The Scroll Heuristic**: If an element is not immediately visible, it may be off-screen. The most common reason for this is needing to scroll. Your plan should include steps to scroll down the page to find elements.
        3.  **Specificity is Key**: When using `browser_act`, be as specific as possible. Instead of "click the button," prefer "click the 'Sign In' button with a blue background."
        ---

    """,
    )
    return textwrap.dedent(
        f"""
        ---
        ### Core Instructions & Rules
        1.  **Single Code Block:** Your entire response MUST be a single, valid Python code block, starting with `@verify`.
        2.  **Entry Point:** For a full plan, the main entry point MUST be `async def main_plan()`.
        3.  **Decomposition:** Break down complex problems into smaller, logical, self-contained `async def` helper functions.
        4.  **Decorators & Docstrings:** Every function you define MUST be decorated with `@verify` and include a concise one-line docstring.
        5.  **No Imports:** You MUST NOT use any `import` statements.
        6.  **Stubbing:** If you cannot implement a function immediately, stub it out with `raise NotImplementedError`.
        7.  **Await Keyword**: All `action_provider` methods that are asynchronous MUST be called with the `await` keyword.
        8.  **Structured Output**: For `observe` or `reason` calls that expect a structured answer (e.g., yes/no, a list of items), you MUST define a Pydantic `BaseModel` and pass it to the `response_format` argument to ensure reliable, parsable output.

        ---
        ### Strategy & Tool Usage
        {strategy_instruction}
        {tool_usage_instruction}

        ---
        ### Tools Reference
        You have access to a global `action_provider` object with the following methods. You must call them with the correct arguments as specified here.
        ```json
        {tool_reference}
        ```

        ---
        ### Handle APIs
        Some tools return a "handle" object for ongoing interaction. The available methods for these handles are listed below. You MUST only use the methods listed.

        {handle_apis}

        ---
        ### Usage Examples

        **Using a Handle-Based Tool (like sending a message):**
        ```python
        @verify
        async def send_confirmation_sms():
            # First, await the tool to get the interactive handle.
            sms_handle = await action_provider.send_sms_message("Text Jane Doe to confirm her 3pm appointment")

            # You can now interact with the handle if needed, or just get the final result.
            confirmation = await sms_handle.result()
            return confirmation
        ```

        **Simple Browser Interaction:**
        ```python
        @verify
        async def check_unify_blog():
            # The browser object can be used directly from the action_provider
            await action_provider.browser_act("Navigate to unify.ai")
            await action_provider.browser_act("Click the 'Blog' link in the main navigation")
            blog_title = await action_provider.browser_observe("What is the title of the first blog post?")
            return blog_title
        ```

        **Structured Outputs:**
        ```python
        # The strategy is to find a "Privacy Policy" link in a list of footer links and click it.
        # Step 1: Define a Pydantic model for structured observation.
        class LinkInfo(BaseModel):
            element_id: str = Field(description="The unique ID for the link element, like 'link_27'.")
            text: str = Field(description="The visible text of the link.")

        class FooterLinks(BaseModel):
            links: list[LinkInfo]

        @verify
        async def find_privacy_policy_link():
            # Step 2: OBSERVE to get structured data.
            footer_data = await action_provider.browser_observe(
                "List all links in the footer section with their text and element IDs.",
                response_format=FooterLinks
            )

            # Step 3: Perform LOGIC in Python to find the target.
            target_link_id = None
            for link in footer_data.links:
                if "privacy policy" in link.text.lower():
                    target_link_id = link.element_id
                    break

            # Step 4: ACT on the identified element.
            if target_link_id:
                await action_provider.browser_act(
                    f"Click the link with element ID 'target_link_id'",
                    "The page should navigate to the privacy policy."
                )
            else:
                # Handle the case where the link wasn't found.
                print("Could not find the 'Privacy Policy' link.")
        ```

        **Generic Reasoning:**
        ```python
        class Summary(BaseModel):
            one_sentence_summary: str = Field(description="A single sentence that captures the main point.")
            key_topics: list[str] = Field(description="A list of the main topics discussed.")

        @verify
        async def summarize_article(article_text: str):
            # Use the reason tool for analysis and structured extraction.
            result = await action_provider.reason(
                request="Summarize the provided article, extracting key topics.",
                context=article_text,
                response_format=Summary
            )
            print(f"Summary: result.one_sentence_summary")
            return result.key_topics
        ```
    """,
    )


def _format_existing_functions(existing_functions: Dict[str, Any]) -> str:
    """Formats the library of existing functions into clean code blocks."""
    if not existing_functions:
        return "None."

    unique_implementations = {
        textwrap.dedent(func_data.get("implementation", "")).strip()
        for func_data in existing_functions.values()
        if func_data.get("implementation")
    }

    if not unique_implementations:
        return "None."

    return "\n\n---\n\n".join(unique_implementations)


def build_initial_plan_prompt(
    goal: str,
    existing_functions: Dict[str, Any],
    retry_msg: str,
    exploration_summary: Optional[str] = None,
    *,
    tools: Dict[str, Callable],
) -> str:
    """
    Dynamically builds the system prompt for the Hierarchical Planner.
    """
    formatted_functions = _format_existing_functions(existing_functions)

    strategy_instruction = (
        "Decompose the problem into logical `async def` functions. Each function should represent a complete, "
        "meaningful sub-task from a user's perspective (e.g., 'search_for_product_and_navigate_to_images' is better than "
        "having separate functions for typing, pressing enter, and clicking the images tab)."
    )
    tool_usage_instruction = "Use the `action_provider` global object to interact with the environment. Available tools and their handle APIs have been described in the rules below."

    rules_and_examples = _build_rules_and_examples_prompt(
        tools,
        strategy_instruction,
        tool_usage_instruction,
    )

    return textwrap.dedent(
        f"""
        You are an expert Python programmer tasked with generating a complete, single-file script to achieve a user's goal.

        **Primary Goal:** "{goal}"
        {rules_and_examples}
        ---
        ### Existing Functions Library
        You may use these pre-existing functions if they are suitable.
        {formatted_functions}

        ---
        {retry_msg}

        Begin your response now. Your response must start immediately with the code.
    """,
    ).strip()


def build_dynamic_implement_prompt(
    function_name: str,
    func_sig: inspect.Signature,
    goal: str,
    parent_code: str,
    browser_state: str | None,
    replan_context: str,
    implementation_strategy: Optional[Any] = None,
    *,
    tools: Dict[str, Callable],
) -> str:
    """
    Builds the system prompt for dynamically implementing a function.

    This function is now context-aware. It will only include browser-specific
    instructions and state if the `browser_state` argument is provided.

    Args:
        function_name: The name of the function to implement.
        func_sig: The signature of the function to implement.
        goal: The overall user goal.
        parent_code: The source code of the calling function.
        browser_state: An optional description of the current browser state.
        replan_context: A message providing context for a replan.
        tools: The tools available to the function.

    Returns:
        The complete prompt string.
    """

    failure_analysis_section = ""
    if replan_context:
        failure_analysis_section = textwrap.dedent(
            f"""
            ---
            ### CRITICAL: Failure Analysis & Recovery Instructions
            You are being asked to implement this function because a previous attempt **failed**. You MUST analyze the following reason and write a new implementation that avoids this specific error.

            **Reason for Previous Failure:**
            {replan_context}
            ---
            """,
        )
    browser_context_section = ""
    if browser_state:
        browser_context_section = f"""**Current Browser State:**
        {browser_state}
        """

    strategy_section = ""
    if implementation_strategy:
        strategy_steps = "\n".join(implementation_strategy.steps)
        strategy_section = textwrap.dedent(
            f"""
            ---
            ### CRITICAL: New Implementation Strategy
            You have already analyzed the failure and created a new plan. You MUST write Python code that strictly follows these steps.

            **Rationale:** {implementation_strategy.rationale}
            **Steps to Follow:**
            {strategy_steps}
            ---
            """,
        )
    strategy_instruction = "Analyze the function's purpose and the available tools to decide on the best implementation strategy."
    tool_usage_instruction = "Use the `action_provider` global object to interact with the environment. Available tools and their handle APIs have been described in the rules below."
    rules_and_examples = _build_rules_and_examples_prompt(
        tools,
        strategy_instruction,
        tool_usage_instruction,
    )

    return textwrap.dedent(
        f"""
        You are an expert Python programmer. Your task is to **only**write the implementation for the function `{function_name}` and nothing else.

        **CRITICAL RULES:**
        1.  Your response MUST contain ONLY the Python code for the function `{function_name}`.
        2.  DO NOT write a `main_plan` or any other functions. Your response must begin immediately with the `@verify` decorator for the function you are implementing.

        **Overall Goal:** "{goal}"
        **Function to Implement:** `async def {function_name}{func_sig}`

        {failure_analysis_section}
        {strategy_section}

        ### Situation Analysis
        **Parent Function (for context):**
        ```python
        {parent_code}
        ```
        {browser_context_section}
        {rules_and_examples}

        Begin your response now. Your response must start immediately with the code.
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


def build_plan_surgery_prompt(
    current_code: str,
    request: str,
    *,
    tools: Dict[str, Callable],
) -> str:
    """
    Builds the prompt for modifying an existing plan script.

    Args:
        current_code: The current source code of the plan.
        request: The user's modification request.
        tools: The tools available to the function.
    Returns:
        The complete prompt string.
    """
    strategy_instruction = "Your task is to rewrite the script below to incorporate the user's change request."
    tool_usage_instruction = "Use the `action_provider` global object to interact with the environment. Available tools and their handle APIs have been described in the rules below."
    rules_and_examples = _build_rules_and_examples_prompt(
        tools,
        strategy_instruction,
        tool_usage_instruction,
    )

    return textwrap.dedent(
        f"""
        You are an expert Python programmer specializing in code modification.

        **Modification Request:**
        "{request}"

        ---
        ### Current Script
        ```python
        {current_code}
        ```
        {rules_and_examples}

        Begin your response now. Your response must start immediately with the code.
        """,
    )


def build_course_correction_prompt(
    old_code: str,
    new_code: str,
    current_state: str,
    *,
    tools: Dict[str, Callable],
) -> str:
    """
    Builds the prompt to generate a course-correction script.

    Args:
        old_code: The previous version of the plan's code.
        new_code: The new version of the plan's code.
        current_state: A description of the current browser state.
        tools: The tools available to the function.
    Returns:
        The complete prompt string.
    """
    strategy_instruction = """1.  Analyze if the **Current Browser State** is a suitable starting point for executing the **New Plan Code**.
2.  If it is NOT suitable, write a script containing an `async def course_correction_main()` function. This script must use the `action_provider` to navigate to the correct starting state for the new plan.
3.  If the current state is already suitable, respond ONLY with the single word: `None`."""
    tool_usage_instruction = "Use the `action_provider` global object as defined in the examples and reference to perform any needed actions."
    rules_and_examples = _build_rules_and_examples_prompt(
        tools,
        strategy_instruction,
        tool_usage_instruction,
    )

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
        {rules_and_examples}

        Begin your response now.
        """,
    )


def build_exploration_prompt(goal: str, *, tools: Dict[str, Callable]) -> str:
    """
    Builds the system prompt for the pre-planning exploration phase.

    Args:
        goal: The high-level user goal.
        tools: A dictionary of available tools for exploration.

    Returns:
        The complete prompt string.
    """
    tool_reference = _build_tool_signatures(tools)
    handle_apis = _build_handle_apis(tools)

    return textwrap.dedent(
        f"""
        You are an intelligent Research Assistant. Your goal is to gather critical information to create a robust plan for the main objective.

        **Main Objective:** "{goal}"

        **Your Task:**
        1. Think step-by-step to determine what information is needed.
        2. Use the `observe` tool to gather this information.
        3. If necessary, use `request_clarification` to ask for guidance.
        4. When you have gathered all necessary information, provide a final, concise summary of your findings. This summary will be used to generate the main plan. DO NOT say that you are ready; your final output MUST BE the summary itself.

         ---
        ### Tools Reference
        You have access to a global `action_provider` object with the following methods. You must call them with the correct arguments as specified here.
        ```json
        {tool_reference}
        ```

        ---
        ### Handle APIs
        Some tools return a "handle" object for ongoing interaction. The available methods for these handles are listed below. You MUST only use the methods listed.

        {handle_apis}

        Begin your response now. Your response must start immediately with the code.

        """,
    )


def build_should_explore_prompt(goal: str) -> str:
    """
    Builds the system prompt for determining if exploration is needed.
    """
    return textwrap.dedent(
        f"""
            You are a web browser agent assessing a task description from a user.
            The agent's goal is to generate a complete Python script to accomplish a task.
            The available tools are high-level: `act(instruction)` and `observe(query)`.
            The agent will be using the `act` tool to navigate the web and the `observe` tool to get information about the page.

            Analyze the following goal:
            **Goal:** "{goal}"

            Is the goal specific and actionable enough to directly write a Python script?
            Or is the goal ambiguous, broad, or lacking key details (like URLs, exact button text, or a clear workflow)
            that the agent would need to discover first using the `observe` and `request_clarification` tools?

            - If the goal is **clear and specific**, respond with the single word: **EXECUTE**.
            - If the goal is **ambiguous or requires information gathering**, respond with the single word: **EXPLORE**.
        """,
    )


def build_implementation_strategy_prompt(
    goal: str,
    function_name: str,
    function_docstring: str | None,
    failure_reason: str,
    browser_state: str | None,
    *,
    tools: Dict[str, Callable],
) -> str:
    """Builds a prompt to devise a new strategy for a failed function."""

    browser_context_section = (
        f"**Current Browser State:**\n{browser_state}" if browser_state else ""
    )
    tool_reference = _build_tool_signatures(tools)
    return textwrap.dedent(
        f"""
        You are a master strategist for a web automation agent. A function has failed to achieve its goal. Your task is to analyze the failure and devise a new, creative, step-by-step natural language plan to succeed.

        **Overall Goal:** "{goal}"
        **Function Under Review:** `{function_name}`
        **Purpose of this function:** {function_docstring or 'No docstring provided.'}

        **CRITICAL: Reason for Previous Failure:** "{failure_reason}"

        {browser_context_section}

        ---
        ### Available Tools
        {tool_reference}
        ---
        ### Your Task
        Based on the failure reason and current state, devise a new strategy. Think outside the box. If observing failed, consider acting (scrolling, clicking). If acting failed, consider observing first.
        Respond with ONLY the JSON object matching the requested schema.
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
