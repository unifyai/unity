from __future__ import annotations

import inspect
import textwrap
import json
from typing import Callable, Dict, Any, Optional
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
    handle_docs = []
    for name, func in tool_dict.items():
        try:
            hints = get_type_hints(func)
            return_type = hints.get("return")
            if (
                return_type
                and inspect.isclass(return_type)
                and issubclass(return_type, SteerableToolHandle)
            ):
                doc = f"**`{return_type.__name__}` (returned by `{name}`)**\n"
                doc += "This handle represents an interactive session. Its available methods are:\n"
                doc += class_api_overview(return_type)
                handle_docs.append(doc)
        except Exception:
            continue

    if not handle_docs:
        return "There are no special handle APIs for the available tools."

    return "\n\n".join(handle_docs)


def _build_rules_and_examples_prompt(
    tools: Dict[str, Callable],
    strategy_instruction: str,
    tool_usage_instruction: str,
    is_dynamic_implement: bool = False,
) -> str:
    """Builds the reusable block of core rules and examples for code generation."""
    tool_reference = _build_tool_signatures(tools)
    handle_apis = _build_handle_apis(tools)

    strategy_instruction += textwrap.dedent(
        """\n
        ---
        ### Strategic Principles for Web Automation
        To create a robust and efficient plan, follow these core principles:
        1.  **Trust the Agent's Autonomy**: The `browser_act` tool is autonomous. Give it high-level goals. Instead of writing separate steps for "click username field", "type username", "click password field", "type password", and "click login", you should create a single step: `await action_provider.browser_act("Log in with username 'test' and password 'pass123'")`. The agent will handle the intermediate steps.
        2.  **Combine Action and Verification**: Use the `expectation` parameter in `browser_act` to tell the agent what success looks like. This is more efficient than a separate `browser_observe` call. For example: `await action_provider.browser_act("Click the 'Add to Cart' button", expectation="The cart icon should show '1' item")`.
        3.  **Use `browser_observe` for Complex Data**: When you need to extract structured data (like a list of products, table contents, or form fields), use `browser_observe` with a Pydantic `response_format`. This is the best way to gather context before acting on complex pages.
        4.  **Describe Visually**: All browser tools operate on what is *visible*. Describe elements by their text, color, or relative position (e.g., "the blue 'Save' button at the bottom of the form"), not by HTML attributes.
        5.  **Use Fallback Capabilities**: If a website's interactive feature (e.g., a "Convert" button, a "Sort" dropdown) fails or doesn't meet your needs, don't give up. Instead, consider if you can achieve the goal using a more fundamental tool. For instance, if you can observe the raw data, you can often use `action_provider.reason` to perform the necessary calculation, transformation, or analysis yourself.
        ---
        """,
    )
    if is_dynamic_implement:
        instructions_and_rules = textwrap.dedent(
            """
            1.  **Single Code Block:** Your entire response MUST be a single, valid Python code block.
            2.  **No Imports:** You **MUST NOT** use any `import`/ `__import__` statements in your code. All standard library imports(eg: `asyncio`, `re`, `pydantic`) are already present within the execution environment so you can use them directly.
            3.  **Decorators & Docstrings:** Every **function** you define MUST include docstrings which include the function's purpose, its arguments, and its return value.
            4.  **Async All The Way**: All helper functions you define MUST be `async def`.
            5.  **Await Keyword**: All `action_provider` methods that are asynchronous MUST be called with the `await` keyword.
            6.  **Structured Output**: For `observe` or `reason` calls that expect a structured answer (e.g., yes/no, a list of items), you MUST define a Pydantic `BaseModel` and pass it to the `response_format` argument to ensure reliable, parsable output. **CRITICAL: Always define Pydantic models INSIDE the function where they are used, NOT at the module level, to avoid forward reference issues.**
            7. **Robust Error Handling**: Proactively use `try...except` blocks to handle potential **unexpected** failures (e.g., an element not being found) with informative error messages. However, **DO NOT** wrap calls to stubbed functions in a `try...except` block. Let `NotImplementedError` propagate. This is important because the agent will implement the stubbed function dynamically.
            """,
        )
    else:
        instructions_and_rules = textwrap.dedent(
            """
            1.  **Single Code Block:** Your entire response MUST be a single, valid Python code block.
            2.  **Entry Point:** For a full plan, the main entry point MUST be `async def main_plan()`.
            3.  **No Imports:** You **MUST NOT** use any `import`/ `__import__` statements in your code. All standard library imports(eg: `asyncio`, `re`, `pydantic`) are already present within the execution environment so you can use them directly.
            4.  **Decomposition:** Break down complex problems into smaller, logical, self-contained `async def` helper functions.
            5.  **Confidence-Based Stubbing**: Your primary goal is to create a robust plan.
                * **If a step is simple and you are highly confident** about how to perform it (e.g., `browser_navigate("https://google.com")`, `browser_act("Type 'reports' into the search bar")`), implement it directly.
                * **If a step is ambiguous or requires seeing the page first**, you **MUST** create a descriptive helper function stubbed with `raise NotImplementedError`. This is critical for actions like finding and applying non-standard filters, extracting data from a unique table layout, or navigating a custom multi-step form.
            6.  **Decorators & Docstrings:** Every **function** you define MUST include docstrings which include the function's purpose, its arguments, and its return value.
            7.  **Async All The Way**: All helper functions you define MUST be `async def`.
            8.  **Await Keyword**: All `action_provider` methods that are asynchronous MUST be called with the `await` keyword.
            9.  **Structured Output**: For `observe` or `reason` calls that expect a structured answer (e.g., yes/no, a list of items), you MUST define a Pydantic `BaseModel` and pass it to the `response_format` argument to ensure reliable, parsable output. **CRITICAL: Always define Pydantic models INSIDE the function where they are used, NOT at the module level, to avoid forward reference issues.**
            10. **Robust Error Handling**: Proactively use `try...except` blocks to handle potential **unexpected** failures (e.g., an element not being found) with informative error messages. However, **DO NOT** wrap calls to stubbed functions in a `try...except` block. Let `NotImplementedError` propagate. This is important because the agent will implement the stubbed function dynamically.
            """,
        )
    return textwrap.dedent(
        f"""
        ---
        ### Core Instructions & Rules
        {instructions_and_rules}
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

        **Using a Handle-Based Tool (like sending a message or making a call)**
        This example demonstrates how to use the `send_sms_message` tool to send a message.
        # Example 1: Sending a message
        ```python
        @verify
        async def send_confirmation_sms():
            # First, await the tool to get the interactive handle.
            sms_handle = await action_provider.send_sms_message("Text Jane Doe to confirm her 3pm appointment")

            # You can now interact with the handle if needed, or just get the final result.
            confirmation = await sms_handle.result()
            return confirmation
        ```

        # Example 2: Making an Interactive Phone Call
        This example demonstrates how to use the `start_call` tool to make an interactive phone call.
        ```python
        @verify
        async def make_appointment_followup_call():
            # Note: start_call is synchronous and returns a Call handle immediately
            call_handle = action_provider.start_call(
                phone_number="+1234567890",
                purpose="Follow up with patient about their upcoming appointment on Friday at 2 PM and confirm they received the pre-appointment instructions"
            )

            # The Call handle returns a SteerableToolHandle object.
            # You can use methods like ask(), interject(), or get the full result

            # Example of using ask() to get specific information during the call:
            ask_handle = await call_handle.ask("Do you have any allergies we should be aware of?")
            allergy_response = await ask_handle.result()

            # Or you can just wait for the full call result:
            call_result = await call_handle.result()

            # Extract key information from the call
            class CallOutcome(BaseModel):
                appointment_confirmed: bool = Field(description="Whether the patient confirmed the appointment")
                instructions_received: bool = Field(description="Whether they received the pre-appointment instructions")
                notes: str = Field(description="Any additional notes from the conversation")

            # Analyze the call transcript
            analysis = await action_provider.reason(
                request="Extract the key outcomes from this phone call transcript",
                context=call_result,
                response_format=CallOutcome
            )

            return analysis
        ```

        **Browser Automation Example**
        This example demonstrates how to combine navigation, observation with Pydantic models, and confidence-based stubbing to create a robust, multi-step web automation plan.

        ```python
        # This function is implemented directly because navigating and searching are simple, high-confidence actions.
        @verify
        async def search_for_product() -> str:
            \"\"\"Navigates to an e-commerce site and searches for a specific product.\"\"\"
            print("Navigating to store and searching for 'blue sneakers'.")
            await action_provider.browser_navigate("https://fakestore.example.com")
            await action_provider.browser_act(
                "Type 'blue sneakers' into the search bar and click the search button",
                expectation="The page should show a list of products related to 'blue sneakers'."
            )
            print("Search complete.")
            return "Successfully searched for products."

        # This function is a STUB. The layout of the search results page is unknown,
        # so we must wait until we can see it before we can reliably implement the extraction logic.
        # This is a perfect example of "Confidence-Based Stubbing".
        @verify
        async def find_and_select_top_rated_product() -> str:
            \"\"\"
            Analyzes the product list, finds the product with the highest rating, and navigates to its page.
            \"\"\"
            raise NotImplementedError("Implement logic to find the highest-rated product and get its URL.")

        # This is another STUB. The product details page layout is also unknown.
        @verify
        async def extract_product_price_and_reviews(product_url: str) -> dict:
            \"\"\"
            Given a product URL, this function navigates to the page and extracts the price and review count.
            \"\"\"
            # Note: A Pydantic model would be defined here during dynamic implementation, like this:
            # class ProductDetails(BaseModel):
            #     price: float
            #     review_count: int
            #
            # await action_provider.browser_navigate(product_url)
            # details = await action_provider.browser_observe(
            #     "Extract the price and number of reviews for this product.",
            #     response_format=ProductDetails
            # )
            # return details.dict()
            raise NotImplementedError("Implement logic to extract price and review count from the product page.")


        @verify
        async def main_plan():
            \"\"\"
            Main plan to find the price of the top-rated blue sneakers.
            \"\"\"
            # Step 1: Perform the search. This is a concrete, implemented step.
            await search_for_product()

            # Step 2: Find the specific product URL. This function is a stub and will be
            # implemented dynamically by the agent once it sees the search results page.
            # CRITICAL: Never wrap a stubbed function in a try-except block.
            # The NotImplementedError MUST be allowed to propagate to the agent.
            #
            # WRONG:
            # try:
            #     top_product_url = await find_and_select_top_rated_product()
            # except Exception:
            #     # This prevents the agent from implementing the stub. Do not do this.
            #     return "Failed"
            #
            # CORRECT:
            top_product_url = await find_and_select_top_rated_product()

            # Step 3: Extract details from that product's page. This is also a stub.
            product_info = await extract_product_price_and_reviews(top_product_url)

            print(f"Final Info Found: {{product_info}}")
            return f"The top-rated product costs {{product_info['price']}} and has {{product_info['review_count']}} reviews."

        ```

        **Fallback Strategy Example (using `reason` tool)**
        This example demonstrates how to create a robust function that first attempts to use a website's feature, but has a fallback plan to use the `reason` tool if the feature fails.
        ```python
        @verify
        async def get_price_in_euros(product_price_usd: float) -> float:
            \"\"\"
            Ensures the product price is available in Euros.

            This function demonstrates a fallback strategy. It first attempts to use
            the website's built-in currency converter. If that fails, it falls back
            to using the `reason` tool to perform the conversion manually.
            \"\"\"
            print(f"Attempting to convert price: ${{product_price_usd}}")

            # --- Primary Approach: Use the website's feature ---
            try:
                await action_provider.browser_act(
                    "Click the currency selector and choose 'EUR'",
                    expectation="The price should now be displayed in Euros (€)."
                )

                class PriceInfo(pydantic.BaseModel):
                    price_eur: float = pydantic.Field(description="The price in Euros.")

                observed_price = await action_provider.browser_observe(
                    "What is the product price in Euros?",
                    response_format=PriceInfo
                )
                print("Successfully converted price using the website's feature.")
                return observed_price.price_eur

            except Exception as e:
                print(f"Website's currency converter failed: ${{e}}. Attempting fallback.")

                # --- Fallback Approach: Use the `reason` tool ---
                try:
                    class ConversionResult(pydantic.BaseModel):
                        price_in_euros: float

                    # Assume a general exchange rate for the purpose of the task
                    conversion_request = (
                        f"Convert ${{product_price_usd}} USD to Euros. "
                        f"Assume an exchange rate of 1 USD = 0.92 EUR. "
                        f"Provide only the final numeric value."
                    )

                    result = await action_provider.reason(
                        request=conversion_request,
                        context=f"The price is ${{product_price_usd}} dollars.",
                        response_format=ConversionResult
                    )
                    print("Successfully converted price using the `reason` tool.")
                    return result.price_in_euros
                except Exception as reason_e:
                    raise ValueError(f"Both website interaction and manual reasoning failed. Error: ${{reason_e}}")
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
        You are an expert strategist. Your task is to generate a high-level Python script that outlines the **strategy** to achieve a user's goal.

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
    full_plan_source: str,
    call_stack: list[str],
    function_name: str,
    function_sig: inspect.Signature,
    function_docstring: str,
    parent_code: str,
    browser_state: str | None,
    has_browser_screenshot: bool,
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
        full_plan_source: The full source code of the plan.
        call_stack: The current function call stack.
        function_name: The name of the function to implement.
        function_sig: The signature of the function to implement.
        function_docstring: The docstring of the function to implement.
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
    if has_browser_screenshot:
        browser_context_section += """
        **Current Browser View (Screenshot):**
        An image of the current browser page has been provided. Analyze it carefully to inform your new implementation.
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
    call_stack_str = " -> ".join(call_stack)
    context_section = textwrap.dedent(
        f"""
    ---
    ### Full Plan Analysis
    You have access to the entire plan and the current call stack for complete strategic context.

    **Current Call Stack:**
    `{call_stack_str}`

    **Full Plan Source Code:**
    ```python
    {full_plan_source}
    ```
    ---
    """,
    )

    strategy_instruction = (
        "Your task is to analyze the situation and decide on the best course of action."
    )
    tool_usage_instruction = "Use the `action_provider` global object to interact with the environment. Available tools and their handle APIs have been described in the rules below."
    rules_and_examples = _build_rules_and_examples_prompt(
        tools,
        strategy_instruction,
        tool_usage_instruction,
        is_dynamic_implement=True,
    )

    return textwrap.dedent(
        f"""
        You are an expert Python programmer and a master strategist. Your task is to analyze the state of a running plan and decide the best course of action for the function `{function_name}`.

        **CRITICAL: You must choose one of three actions:**
        1.  **`implement_function`**: Write the Python code for `{function_name}`. Choose this if the function's goal is achievable from the current browser state.
        2.  **`skip_function`**: Bypass this function entirely. Choose this if you observe that the function's goal is **already completed** or is now **irrelevant**. For example, skip a "log in" function if you are already logged in.
        3.  **`replan_parent`**: Escalate the failure to the calling function. Choose this if the current function is **impossible to implement** because of a mistake made in a *previous* step. For example, if the goal is "apply filters" but the page has no filter controls, the error lies with the parent function that navigated to the wrong page or failed to get to the right state.

        {context_section}

        ### Situation Analysis
        **Function to Address:** `async def {function_name}{function_sig}`
        **Purpose of this Function:** "{function_docstring}"
        **Current Browser State:**
        {browser_state or "No browser state available."}
        A screenshot of the current browser page has been provided. **Use it as the primary source of truth.**

        {failure_analysis_section}
        {strategy_section}
        {rules_and_examples}

        Respond with ONLY the JSON object matching the `ImplementationDecision` schema.
        """,
    )


def build_verification_prompt(
    goal: str,
    function_name: str,
    function_docstring: str | None,
    function_source_code: str | None,
    interactions: list,
    has_browser_screenshot: bool,
    function_return_value: Any | None,
) -> str:
    """
    Builds the prompt for verifying a function's execution.

    Args:
        goal: The overall user goal.
        function_name: The name of the function being verified.
        function_docstring: The docstring of the function.
        function_source_code: The source code of the function.
        interactions: A log of tool interactions made.
        has_browser_screenshot: Whether a screenshot of the browser is provided.

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
    screenshot_context_section = ""
    if has_browser_screenshot:
        screenshot_context_section = textwrap.dedent(
            """
            ---
            ### CRITICAL: Visual Verification
            You have been provided a **screenshot** of the browser's final state after the function finished.
            - **Use this screenshot as the primary source of truth.**
            - If the interaction log claims success (e.g., "navigated to page X") but the screenshot clearly shows this did not happen, you MUST rule the function a failure (`reimplement_local`).
            - Use both the interaction log and the screenshot to make your assessment.
            """,
        )
    return_value_log = f"The function returned the following value:\n```\n{repr(function_return_value)}\n```"

    source_code_section = f"""
---
### Function Source Code
The full source code of the function that was just executed is provided below. Analyze it to understand its internal logic.
```python
{function_source_code or "Source code not available."}
```
"""

    return textwrap.dedent(
        f"""
        You are a meticulous verification agent. Your task is to assess if the executed actions successfully achieved the function's intended purpose and have made **meaningful and accurate progress** toward the **Overall User Goal**.

        **Overall User Goal:** "{goal}"
        **Function Under Review:** `{function_name}`
        **Purpose of this function (Intent):** {function_docstring or 'No docstring provided.'}

        {source_code_section}
        {screenshot_context_section}

        **Execution Log (Tool Interactions):**
        {interactions_log}

        **Function Return Value:**
        {return_value_log}

        ---
        ### Assessment Task
        Based on the function's purpose (intent), its source code (implementation), its return value, and the execution log, provide your assessment.
        - **Compare Intent vs. Implementation**: Does the source code correctly implement the logic described in the function's purpose?
        - **Trust the Return Value**: The `Function Return Value` is a critical piece of evidence. If the function was supposed to filter a list, check if the returned list is correctly filtered according to the source code.
        - **Be pragmatic:** If the function's purpose is to gather data (like search results), and the log shows that the data was successfully retrieved, this should be considered a success (`ok`). The function does not need to perform extra analysis unless explicitly asked.
        - **Compare the Result to the Goal**: Do not just check if the function *did something*. Check if the *outcome* of the function satisfies the requirements of the overall goal.

        **Valid Status Values:**
        - `ok`: The function's purpose was fully and correctly achieved.
        - `reimplement_local`: A tactical error occurred. The goal is correct, but the actions were wrong. The function needs to be re-written.
        - `replan_parent`: A strategic error occurred. The function itself is flawed or was called at the wrong time. The parent function needs to be replanned.
        - `fatal_error`: An unrecoverable error occurred that prevents any further progress.

        **Your Response:**
        - status: Choose one of the valid status values above
        - reason: Provide a clear, concise explanation for your assessment
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
        You are a Research Assistant for a web automation agent. Your mission is to gather facts to help write a successful script.

        **Main Objective:** "{goal}"

        **Your Task:**
        1.  **Think Step-by-Step**: What specific pieces of information are missing from the objective? (e.g., URLs, exact text on buttons, structure of a search results page).
        2.  **Use `browser_observe`**: Use this tool to gather the missing information. Ask targeted questions.
        3.  **Summarize Findings**: Once you have gathered the necessary details, your final output MUST be a concise summary of your findings. This summary will be fed into the next stage of planning. Do not just say you are ready.

        **Example Workflow:**
        - **Goal:** "Find the contact email on the UnifyAI website."
        - **Your Thought Process:** I need the URL for UnifyAI. Then I need to find a "Contact" or "About Us" link. Then I need to read that page to find the email.
        - **Your Final Summary:** "The website is unify.ai. The contact information is located on the '/contact' page, which is linked in the footer. The email address is not directly listed, but there is a contact form."

        ---
        ### Tools Reference
        {tool_reference}
        ---
        Begin. Your final output must be the summary.
        """,
    )


def build_should_explore_prompt(goal: str) -> str:
    """
    Builds the system prompt for determining if exploration is needed.
    """
    return textwrap.dedent(
        f"""
        You are a planning analyst for a web automation agent. Your job is to decide if a task description contains enough specific information to write a Python script directly, or if an initial information-gathering phase is required.

        **Analyze the following user goal:**
        "{goal}"

        **Decision Criteria:**
        - **EXECUTE directly if:** The goal contains specific URLs, precise names of buttons/links ("Images tab", "Sign In button"), and a clear, linear workflow.
        - **EXPLORE first if:** The goal is ambiguous. For example, it mentions a site but not a URL ("a popular news site"), asks to find something without specifying where ("find their contact email"), or implies a complex workflow that needs discovery ("find the cheapest can opener and add it to the cart"). The presence of a task like finding a specific item from a list of search results warrants exploration.

        - If the goal is **clear and specific**, respond with the single word: **EXECUTE**.
        - If the goal is **ambiguous or requires information gathering**, respond with the single word: **EXPLORE**.
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


def build_trace_summary_prompt(
    goal: str,
    action_log: str,
) -> str:
    """
    Builds the prompt for the Trace Summary LLM.

    Args:
        goal: The original high-level goal of the plan.
        action_log: The detailed execution trace from plan.action_log.

    Returns:
        The complete prompt string for the summarization call.
    """
    return textwrap.dedent(
        f"""
        You are an expert debugging analyst for an autonomous web agent.
        The following is a detailed action log from a failed plan execution. Your task is to read the entire trace and produce a concise, high-level summary of the strategic error.

        **Original Goal:** "{goal}"

        **Execution Trace / Action Log:**
        ```
        {action_log}
        ```

        **Your Analysis Task:**
        1.  Identify the root cause of the failure. Do not focus on the final error message, but on the sequence of events that led to it.
        2.  Explain the flaw in the plan's original strategy (e.g., "The plan incorrectly assumed X," or "The plan failed to perform step Y before Z").
        3.  Provide a clear, actionable recommendation for a new strategy that would avoid this failure.

        Respond with only the summary of your analysis. This summary will be used to rewrite the entire plan from scratch.
        """,
    )
