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


def _format_cache_summary(idempotency_cache: Dict[tuple, Any], last_n: int = 20) -> str:
    """
    Formats the last N cache entries, including tool call arguments, into a
    readable summary for the LLM.
    """
    if not idempotency_cache:
        return "### Cache Status\n- The cache is currently empty."

    summary_lines = [
        "### Cache Status (for Invalidation Planning)",
        "- The following functions have at least one cached action and are eligible for invalidation:",
    ]

    cacheable_functions = sorted(
        list(
            set(
                entry["meta"]["function"]
                for entry in idempotency_cache.values()
                if entry.get("meta") and entry["meta"].get("function")
            ),
        ),
    )
    summary_lines.append(f"  `{cacheable_functions}`")
    summary_lines.append(
        "- **Rule**: Only list functions from this list in `invalidate_functions`.",
    )
    summary_lines.append("\n### Recent Cached Actions:")

    recent_items = list(idempotency_cache.values())[-last_n:]

    for entry in recent_items:
        meta = entry.get("meta")
        interaction = entry.get("interaction_log")
        if not meta or not interaction:
            continue

        func = meta.get("function", "N/A")
        step = meta.get("step", "N/A")

        call_repr = interaction[1] if len(interaction) > 1 else "N/A"

        if len(call_repr) > 100:
            call_repr = call_repr[:97] + "..."

        summary_lines.append(
            f"- Func: `{func}`, Step: `{step}`, Call: `{call_repr}`",
        )

    return "\n".join(summary_lines)


def _build_shared_strategy_principles() -> str:
    """
    Builds the reusable block of strategic principles for automation prompts.
    This ensures consistency across initial planning, dynamic implementation, and interjections.
    """
    return textwrap.dedent(
        """
        ### 🧠 Strategic Principles for Automation
        To succeed, you must follow these core principles when writing or modifying code.

        1.  **Trust the Agent's Autonomy**: The `act` tool is autonomous. Give it high-level goals describing the desired outcome. Instead of "click username field," then "type username," then "click login," a single step is better: `await action_provider.act("Log in with username 'test' and password 'pass123'")`.
        2.  **Describe Visually and Functionally**: All browser tools operate on what is *visible*. Describe elements by their text, appearance, or relative position (e.g., "the blue 'Save' button at the bottom of the form"), not by HTML attributes which you cannot see.
        3.  **Use `observe` for Complex Data**: When you need to extract structured data (like a list of products, table contents, or form fields), use `observe` with a Pantic `response_format`. This is the most reliable way to gather information before acting.
        4.  **Isolate Core Logic**: When refactoring, identify the central, repeatable process. Omit one-time setup steps (like "open a new tab") from the generalized helper function. The goal is to create a function that represents a meaningful, reusable skill.
        5.  **Write General, Parameterized Functions**: Functions should be reusable tools, not single-use scripts. Pass specific values (like search terms, filenames, or credentials) as parameters. Function names should describe the *process*, not the data (e.g., `process_user(username: str)` is better than `process_user_smith()`).
        6.  **Use Fallbacks**: If a website's feature is unreliable (e.g., a buggy serving size calculator), create a fallback. First, try the website feature. If it fails, use the `reason` tool or pure Python to perform the calculation or transformation yourself. This makes your plan robust.
        """,
    )


def _build_code_act_rules_and_examples(action_provider) -> str:
    """Builds the reusable block of core rules and examples for CodeActActor."""
    all_tools = {}

    browser_tools = {
        "navigate": action_provider.navigate,
        "act": action_provider.act,
        "observe": action_provider.observe,
    }
    all_tools.update(browser_tools)

    comm_tools = {
        "send_sms_message": action_provider.send_sms_message,
        "send_email": action_provider.send_email,
        "send_whatsapp_message": action_provider.send_whatsapp_message,
        "start_call": action_provider.start_call,
        "join_meet": action_provider.join_meet,
    }
    all_tools.update(comm_tools)

    if hasattr(action_provider, "reason"):
        all_tools["reason"] = action_provider.reason

    tool_reference = _build_tool_signatures(all_tools)
    handle_apis = _build_handle_apis(all_tools)

    instructions_and_rules = textwrap.dedent(
        """
        ### 🎯 CRITICAL RULES FOR CODE EXECUTION


        1. **Stateful Execution**: Your code is executed in a persistent, stateful REPL-like environment. Variables, functions, and imports defined in one turn are available in all subsequent turns.

        2. **Use `await`**: The execution sandbox is asynchronous. You **MUST** use the `await` keyword for any action_provider operations:
           ```python
           # ✅ CORRECT: Using await
           await action_provider.navigate("https://example.com")
           result = await action_provider.observe("What is the heading?")

           # ❌ WRONG: Missing await
           action_provider.navigate("https://example.com")
           ```

        3. **Imports Inside Code**: All necessary imports must be included in the code you provide:
           ```python
           # ✅ CORRECT: Import inside the code execution
           from pydantic import BaseModel, Field
           from typing import Optional, List
           ```

        4. **Pydantic for Structured Observation**: When using `action_provider.observe` to extract structured data:
           ```python
           from pydantic import BaseModel, Field

           class PageInfo(BaseModel):
               title: str = Field(description="Page title")
               products: list[str] = Field(description="List of product names")

           # CRITICAL: Call model_rebuild() after defining nested models
           PageInfo.model_rebuild()

           result = await action_provider.observe(
               "Extract page information",
               response_format=PageInfo
           )
           ```

        5. **Error Handling**: If your code produces an error, the traceback will be returned. Read it carefully, correct your code, and try again.

        6. **Browser State Feedback**: After browser actions, you'll automatically receive:
           - The current URL
           - A screenshot of the page
           - Any output from your code

        7. **exit**: Your workflow should be:
           - Think about what you need to do
           - Write code to execute the action
           - Observe the results (output, screenshots, errors)
           - Continue with the next step or correct errors

        8. **Final Answer Rule**:
           - When the user's request has been fully addressed and you have the final answer, you **MUST** provide that answer directly as a tool-less assistant message.
           - Do not call a tool to print the final answer. Simply state the answer.

           # ✅ CORRECT:
           {
           "tool_calls": [],
           "messages": [
            {
                "role": "assistant",
                "content": "The final answer is: 42"
            }
           ]
           }

           # ❌ WRONG:
           {
           "tool_calls": [
            {
                "name": "execute_python_code",
                "arguments": {
                    "code": "print('The final answer is: 42')"
                }
            }
           ],
           "messages": []
        }
        """,
    )
    examples = textwrap.dedent(
        """
        ### 💡 Strategy & Examples

        Your primary workflow is an iterative loop: **Think → Code → Observe → Repeat**. You write a block of Python code, execute it, observe the output (including stdout, errors, and browser state), and then decide on the next block of code.

        ---

        **Example 1: Web Navigation and Structured Data Extraction**

        *User Request*: "What is the main heading and the text of the first paragraph on playwright.dev?"

        *Turn 1: Navigate to the website*
        * **Tool Call**:
            ```json
            {
              "tool_calls": [{
                "name": "execute_python_code",
                "arguments": {
                  "thought": "The first step is to navigate to the website specified in the user's request, which is playwright.dev.",
                  "code": "await action_provider.navigate('https://playwright.dev/')"
                }
              }]
            }
            ```
        * **Observation**:
            ```text
            --- BROWSER STATE ---
            URL: https://playwright.dev/
            [A screenshot of the Playwright homepage is available to you.]
            ```

        *Turn 2: Observe the content using a Pydantic model*
        * **Tool Call**:
            ```json
            {
              "tool_calls": [{
                "name": "execute_python_code",
                "arguments": {
                  "thought": "Great, I'm on the page. Now I'll extract the heading and paragraph text into a structured object for clarity. I'll define a Pydantic model right here in the sandbox.",
                  "code": "from pydantic import BaseModel, Field\n\nclass PageContent(BaseModel):\n    heading: str = Field(description=\"The main H1 heading of the page\")\n    first_paragraph: str = Field(description=\"The text of the first paragraph under the heading\")\n\nPageContent.model_rebuild()\n\npage_info = await action_provider.observe(\n    \"Extract the main heading and the first paragraph.\",\n    response_format=PageContent\n)\n\nprint(page_info.model_dump_json(indent=2))"
                }
              }]
            }
            ```
        * **Observation**:
            ```text
            --- STDOUT ---
            {
              "heading": "Playwright enables reliable end-to-end testing for modern web apps.",
              "first_paragraph": "Playwright is an open-source framework for web testing and automation. It allows testing Chromium, Firefox and WebKit with a single API."
            }
            --- BROWSER STATE ---
            URL: https://playwright.dev/
            [A screenshot of the Playwright homepage is available to you.]
            ```

        *Turn 3: Provide the final answer*
        * **Thought**: I have successfully extracted the information. I will now provide the final answer to the user without using any tools.
        * **Final Answer (tool-less response)**:
            The main heading on playwright.dev is 'Playwright enables reliable end-to-end testing for modern web apps.', and the first paragraph is 'Playwright is an open-source framework for web testing and automation. It allows testing Chromium, Firefox and WebKit with a single API.'

        ---
        **Example 2: Error Handling and Correction**

        *User Request*: "Get the title from example.com."

        *Turn 1: Navigate*
        * **Tool Call**:
            ```json
            {
              "tool_calls": [{
                "name": "execute_python_code",
                "arguments": {
                  "thought": "I need to navigate to example.com to get the title.",
                  "code": "await action_provider.navigate('https://example.com/')"
                }
              }]
            }
            ```
        * **Observation**: Success, browser is on example.com.

        *Turn 2: Attempt to extract data with a mistake*
        * **Tool Call**:
            ```json
            {
              "tool_calls": [{
                "name": "execute_python_code",
                "arguments": {
                  "thought": "I need to get the title. I will use Pydantic, but I'll deliberately forget to call `model_rebuild()` to demonstrate error handling.",
                  "code": "from pydantic import BaseModel\n\nclass PageTitle(BaseModel):\n    title: str\n\n# MISTAKE: I forgot to call PageTitle.model_rebuild()\n\ntitle_info = await action_provider.observe(\n    \"Extract the page title\",\n    response_format=PageTitle\n)\nprint(title_info)"
                }
              }]
            }
            ```
        * **Observation**:
            ```text
            --- ERROR ---
            Traceback (most recent call last):
              ...
            ValidationError: The response from the model did not conform to the expected Pydantic schema. [Reason: Could not find a registered Pydantic model named 'PageTitle'. Did you forget to call `PageTitle.model_rebuild()`?]
            ```

        *Turn 3: Correct the code based on the error*
        * **Tool Call**:
            ```json
            {
              "tool_calls": [{
                "name": "execute_python_code",
                "arguments": {
                  "thought": "The error message is very clear. I forgot to call `model_rebuild()` on my Pydantic class. I will add that line and retry.",
                  "code": "from pydantic import BaseModel\n\nclass PageTitle(BaseModel):\n    title: str\n\n# CORRECTED: I've added the required model_rebuild() call.\nPageTitle.model_rebuild()\n\ntitle_info = await action_provider.observe(\n    \"Extract the page title\",\n    response_format=PageTitle\n)\nprint(title_info.title)"
                }
              }]
            }
            ```
        * **Observation**:
            ```text
            --- STDOUT ---
            Example Domain
            ```
        * **Final Answer (tool-less)**: The title of the page is "Example Domain".

        ---

        **Example 3: Stateful Computation and Helper Functions**

        *User Request*: "On `fakestore.example.com`, find the average price of all products that cost less than $100."

        *Turn 1: Define a helper function and extract all product data*
        * **Tool Call**:
            ```json
            {
              "tool_calls": [{
                "name": "execute_python_code",
                "arguments": {
                  "thought": "This is a multi-step task. First, I'll extract all products. I know I'll need to parse prices that might be strings (e.g., '$25.99'), so I'll define a helper function to clean them. This function will persist in the sandbox for later.",
                  "code": "import re\nfrom pydantic import BaseModel, Field\nfrom typing import List\n\ndef parse_price(price_str: str) -> float:\n    nums = re.findall(r'[\\d.]+', price_str)\n    return float(nums[0]) if nums else 0.0\n\nclass Product(BaseModel):\n    name: str\n    price_text: str = Field(alias=\"price\")\n\nclass ProductList(BaseModel):\n    products: List[Product]\n\nProductList.model_rebuild()\n\nglobal all_products_data\nall_products_data = await action_provider.observe(\n    \"Extract all products with their name and price text\",\n    response_format=ProductList\n)\nprint(f\"Extracted {len(all_products_data.products)} products.\")"
                }
              }]
            }
            ```
        * **Observation**:
            ```text
            --- STDOUT ---
            Extracted 20 products.
            ```

        *Turn 2: Use the helper function and the stored variable to compute the average*
        * **Tool Call**:
            ```json
            {
              "tool_calls": [{
                "name": "execute_python_code",
                "arguments": {
                  "thought": "I have the product data in the `all_products_data` variable and my `parse_price` function is defined. Now I can perform the calculation in pure Python.",
                  "code": "prices_under_100 = []\nfor product in all_products_data.products:\n    price = parse_price(product.price_text)\n    if price < 100.0:\n        prices_under_100.append(price)\n\nif prices_under_100:\n    average = sum(prices_under_100) / len(prices_under_100)\n    result_text = f\"The average price of products under $100 is ${average:.2f}.\"\nelse:\n    result_text = \"No products found under $100.\"\n\nprint(result_text)"
                }
              }]
            }
            ```
        * **Observation**:
            ```text
            --- STDOUT ---
            The average price of products under $100 is $42.75.
            ```
        * **Final Answer (tool-less)**: The average price of products under $100 on the site is $42.75.

        ---

        **Example 4: Interactive Communication Workflow**

        *User Request*: "Text Jane Doe to confirm her appointment for tomorrow at 3 PM. Then, call her to ask if she has any dietary restrictions for the pre-appointment lunch."

        *Turn 1: Send the confirmation SMS*
        * **Tool Call**:
            ```json
            {
              "tool_calls": [{
                "name": "execute_python_code",
                "arguments": {
                  "thought": "I'll start by sending the SMS. The `send_sms_message` tool returns a handle, which I'll await to ensure the message is sent and get a result.",
                  "code": "sms_handle = await action_provider.send_sms_message(\n    description=\"Text Jane Doe to confirm her appointment for tomorrow at 3 PM.\"\n)\n\nsms_result = await sms_handle.result()\nprint(sms_result)"
                }
              }]
            }
            ```
        * **Observation**:
            ```text
            --- STDOUT ---
            Message successfully sent to Jane Doe (+1-555-123-4567): 'Hi Jane, this is a confirmation for your appointment tomorrow at 3 PM. Please reply to confirm.'
            ```

        *Turn 2: Initiate the interactive phone call*
        * **Tool Call**:
            ```json
            {
              "tool_calls": [{
                "name": "execute_python_code",
                "arguments": {
                  "thought": "The SMS is sent. Now I need to make the phone call. The `start_call` tool also returns a handle. I will store this handle in a global variable so I can interact with it in the next turn.",
                  "code": "global active_call_handle\nactive_call_handle = action_provider.start_call(\n    phone_number=\"Jane Doe\",\n    purpose=\"Ask about dietary restrictions for a lunch meeting.\"\n)\nprint(f\"Initiated call to Jane Doe. Handle ID: {active_call_handle._loop_id}\")"
                }
              }]
            }
            ```
        * **Observation**:
            ```text
            --- STDOUT ---
            Initiated call to Jane Doe. Handle ID: a4b1
            --- RESULT ---
            <AsyncToolUseLoopHandle object ...>
            ```

        *Turn 3: Interact with the live call using the handle*
        * **Tool Call**:
            ```json
            {
              "tool_calls": [{
                "name": "execute_python_code",
                "arguments": {
                  "thought": "The call is now active and the handle is stored in `active_call_handle`. I will use the handle's `.ask()` method to pose the question and get the answer.",
                  "code": "ask_handle = await active_call_handle.ask(\"Do you have any dietary restrictions for the lunch tomorrow?\")\n\ndietary_info = await ask_handle.result()\nprint(f\"Received dietary info: {dietary_info}\")\n\nawait active_call_handle.stop()\nprint(\"Call ended.\")"
                }
              }]
            }
            ```
        * **Observation**:
            ```text
            --- STDOUT ---
            Received dietary info: "Thanks for asking! I'm vegetarian."
            Call ended.
            ```
        * **Final Answer (tool-less)**: I've confirmed Jane Doe's appointment via SMS. I also called her and she mentioned her dietary restriction is vegetarian.
        """,
    )
    return f"""
{instructions_and_rules}

---
### Tools Reference
Within your code execution, you have access to a global `action_provider` object with these methods:
```json
{tool_reference}
```

---
### Handle APIs
Some tools return "handle" objects for ongoing interaction. Available methods:

{handle_apis}

---
{examples}
"""


def _build_initial_plan_rules_and_examples(
    tools: Dict[str, Callable],
    strategy_instruction: str,
    tool_usage_instruction: str,
) -> str:
    """Builds the reusable block of core rules and examples for initial planning."""
    tool_reference = _build_tool_signatures(tools)
    handle_apis = _build_handle_apis(tools)

    shared_principles = _build_shared_strategy_principles()
    strategy_instruction += textwrap.dedent(
        f"""\n
        ---
        {shared_principles}
        7. **Name for the Action, Not the Data**: Function names must describe the *process*, not the specific values being processed. Avoid hardcoding data like numbers or names into function names. This makes your plan robust and easy to modify later.
        8. **Handle Ambiguous or "Non-Goals"**: If the user's goal is not a specific, actionable task (e.g., it's vague, "I don't know," or an instruction like "I will guide you"), your responsibility is to generate a simple, empty plan that allows the user to provide the first real instruction via interjection.

        | ❌ Bad (Too Specific & Brittle)        | ✅ Good (Generic & Reusable)                    |
        | ------------------------------------ | --------------------------------------------- |
        | `async def process_user_smith()`       | `async def process_user(username: str)`       |
        | `async def get_report_for_q3()`        | `async def get_report(quarter: str)`          |
        | `async def extract_ten_items()`        | `async def extract_items(item_count: int)`    |

        | User Goal                                           | Correct Plan Output                                                                                                                              |
        | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
        | `"I\\'ll tell you what to do step-by-step."`            | `async def main_plan():\\n    "\"\"\"This is a teaching session. I will await user instructions.\"\"\"\\n    pass`                                            |
        | `"Ummm, I\\'m not sure yet."`                           | `async def main_plan():\\n    "\"\"\"The user\\'s goal is unclear. I will await instructions.\"\"\"\\n    pass`                                                |
        | `None` or Empty String                              | `async def main_plan():\\n    "\"\"\"No goal was provided. Awaiting user instructions.\"\"\"\\n    pass`                                                      |
        ---
        """,
    )

    instructions_and_rules = textwrap.dedent(
        """
        ### 🎯 CRITICAL RULES FOR INITIAL PLAN CREATION

        1.  **Single Code Block:** Your entire response MUST be a single, valid Python code block.
            ```python
            # ✅ CORRECT: Single code block response
            async def helper_function():
                ...

            async def main_plan():
                ...
            ```

        2.  **Entry Point:** For a full plan, the main entry point MUST be `async def main_plan()`.
            ```python
            # ✅ CORRECT: Always end with main_plan
            async def main_plan():
                \"\"\"Main entry point for the automation plan.\"\"\"
                await step_one()
                result = await step_two()
                return result
            ```

        3.  **Scope and Imports**: ALL imports must be placed **inside** functions, never at the top level.
            ```python
            # ❌ WRONG: Top-level import
            from typing import Optional

            # ✅ CORRECT: Import inside function
            async def my_function():
                from typing import Optional
                from pydantic import BaseModel
                ...
            ```

        4.  **Decomposition:** Break complex tasks into smaller, focused functions.
            ```python
            # ✅ GOOD: Each function has a single, clear purpose
            async def login_to_account(username: str, password: str):
                \"\"\"Logs into the user account.\"\"\"
                ...

            async def search_for_product(product_name: str):
                \"\"\"Searches for a specific product.\"\"\"
                ...
            ```

        5.  **Confidence-Based Stubbing**: The MOST IMPORTANT rule for robust planning.
            ```python
            # ✅ IMPLEMENT if confident (simple, predictable actions)
            async def navigate_to_shop():
                \"\"\"Navigate to the shop homepage.\"\"\"
                await action_provider.navigate("https://shop.example.com")

            # ✅ STUB if uncertain (complex extractions, unknown layouts)
            async def extract_shipping_options():
                \"\"\"Extract available shipping options and prices.\"\"\"
                # Need to see the page structure first
                raise NotImplementedError("Extract shipping options from checkout page")
            ```

            **CRITICAL**: **Purity of Stubs**: A stubbed function MUST contain ONLY a `raise NotImplementedError(...)` statement and its docstring. Do not include ANY `await action_provider` calls or other logic inside a stub.

            ```python
            # ❌ WRONG: Stub with a side-effect
            async def my_stub():
                "\"\"\"This is a bad stub.\"\"\""
                await action_provider.navigate("...")  # NEVER DO THIS
                raise NotImplementedError("Implement me")

            # ✅ CORRECT: A pure stub
            async def my_stub():
                "\"\"\"This is a perfect stub.\"\"\""
                raise NotImplementedError("Implement me")
            ```

        6.  **Decorators & Docstrings:** EVERY function needs proper documentation.
            ```python
            # ✅ CORRECT: Clear docstring with Args and Returns
            async def calculate_total_price(items: list[dict], tax_rate: float) -> float:
                \"\"\"Calculate the total price including tax.

                Args:
                    items: List of items with 'price' keys
                    tax_rate: Tax rate as a decimal (e.g., 0.08 for 8%)

                Returns:
                    float: Total price including tax
                \"\"\"
                ...
            ```

        7.  **Async All The Way**: ALL functions must be async.
            ```python
            # ❌ WRONG: Regular function
            def helper():
                pass

            # ✅ CORRECT: Async function
            async def helper():
                pass
            ```

        8.  **Await Keyword**: ALWAYS await async action_provider methods.
            ```python
            # ❌ WRONG: Missing await
            action_provider.navigate("https://example.com")

            # ✅ CORRECT: With await
            await action_provider.navigate("https://example.com")
            ```

        9.  **Structured Output with Pydantic - THE COMPLETE PATTERN:**
            ```python
            async def extract_structured_data():
                # Step 1: Import inside function
                from pydantic import BaseModel, Field
                from typing import Optional, List

                # Step 2: Define models inside function
                class Product(BaseModel):
                    name: str
                    price: float
                    # Step 3: Use Optional for potentially missing fields
                    rating: Optional[float] = Field(default=None)
                    in_stock: bool = Field(description="Availability status")

                class ProductList(BaseModel):
                    products: List[Product]
                    total: int

                # Step 4: CRITICAL - Call model_rebuild() on outermost model
                ProductList.model_rebuild()

                # Step 5: Use with response_format
                result = await action_provider.observe(
                    "Extract all products with details",
                    response_format=ProductList
                )

                # ❌ WRONG: Forgetting model_rebuild()
                # ❌ WRONG: Not using Optional for missing fields
                # ❌ WRONG: Defining models outside the function
            ```

        10. **Error Handling - NEVER SILENCE ERRORS:**
            ```python
            # ❌ WRONG: Silencing errors
            try:
                result = await risky_operation()
            except:
                return None  # Never do this!

            # ✅ CORRECT: Log and re-raise
            try:
                result = await risky_operation()
            except Exception as e:
                print(f"Operation failed: {e}")
                raise  # Always re-raise!

            # ⚠️ EXCEPTION: Never wrap stubbed functions
            # ❌ WRONG:
            try:
                await my_stubbed_function()  # Has NotImplementedError
            except:
                pass  # This breaks dynamic implementation!

            # ✅ CORRECT:
            await my_stubbed_function()  # Let NotImplementedError propagate
            ```

        11. **Action Provider Usage:**
            ```python
            # ❌ WRONG: Don't create or import ActionProvider
            from some_module import ActionProvider
            action_provider = ActionProvider()

            # ❌ WRONG: Don't type hint it
            def my_func(action_provider: ActionProvider):
                pass

            # ✅ CORRECT: Use it directly as a global
            async def my_func():
                await action_provider.navigate("...")
            ```

         12. **Requesting Clarification:**
            ```python
            # ✅ CORRECT: Call as a global function
            destination = await request_clarification("What is your destination city?")

            # ❌ WRONG: Do not call it on action_provider
            # destination = await action_provider.request_clarification(...)
            ```
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
        ### Usage Examples for Initial Plan Creation

        **COMPLETE EXAMPLE: E-commerce Automation Plan**
        This example demonstrates ALL the rules for creating a robust initial plan.
        ```python
        # RULE 6: Every function has proper docstrings with purpose, args, and returns
        @verify
        async def search_for_product(product_name: str) -> None:
            \"\"\"Searches for a specific product on the e-commerce site.

            Args:
                product_name: The name of the product to search for

            Returns:
                None
            \"\"\"
            # RULE 5: Implemented directly - high confidence action
            print(f"Searching for product: {{product_name}}")

            # RULE 8: Await all async action_provider methods
            await action_provider.navigate("https://shop.example.com")
            await action_provider.act(
                f"Type '{{product_name}}' in the search box and press Enter to load search results with products"
            )

        # RULE 5: STUB - Complex extraction requiring page analysis
        @verify
        async def extract_product_prices() -> list[dict]:
            \"\"\"Extracts all product information from search results.

            This needs to see the actual page structure to implement properly.

            Returns:
                list[dict]: List of products with name, price, rating
            \"\"\"
            # RULE 5: Use NotImplementedError for confidence-based stubbing
            raise NotImplementedError("Need to see search results page structure to extract products")

        # RULE 3: All imports inside functions
        @verify
        async def filter_by_price_range(min_price: float, max_price: float) -> None:
            \"\"\"Applies price filters to the search results.

            Args:
                min_price: Minimum price in dollars
                max_price: Maximum price in dollars

            Returns:
                None
            \"\"\"
            # RULE 3: Import inside the function
            from typing import Optional

            # RULE 10: Error handling with re-raise
            try:
                await action_provider.act(
                    f"Set price filter from ${{min_price}} to ${{max_price}} to filter products by price range"
                )
            except Exception as e:
                print(f"Failed to apply price filter: {{e}}")
                # RULE 10: MUST re-raise the exception
                raise

        # RULE 9: Pydantic models for structured data
        @verify
        async def verify_product_in_cart() -> dict:
            \"\"\"Verifies that the product was added to cart successfully.

            Returns:
                dict: Cart information including item count and total
            \"\"\"
            # RULE 3 & 9: Import Pydantic inside function
            from pydantic import BaseModel, Field
            from typing import Optional

            # RULE 9: Define model inside function
            class CartStatus(BaseModel):
                item_count: int = Field(description="Number of items in cart")
                total_price: float = Field(description="Total price of items")
                # RULE 9: Use Optional for fields that might be missing
                discount: Optional[float] = Field(default=None, description="Discount amount if any")

            # RULE 9: CRITICAL - Always rebuild the model
            CartStatus.model_rebuild()

            # RULE 9: Use response_format for structured output
            cart_info = await action_provider.observe(
                "What is the current cart status including item count and total price?",
                response_format=CartStatus
            )

            return {{
                "items": cart_info.item_count,
                "total": cart_info.total_price,
                "discount": cart_info.discount
            }}

        # RULE 4: All functions must be async
        @verify
        async def main_plan():
            \"\"\"Main plan to search for and purchase a product.

            This demonstrates a complete e-commerce automation flow.
            \"\"\"
            # Step 1: Search for the product
            await search_for_product("wireless headphones")

            # Step 2: Extract and analyze products (stubbed)
            # RULE 10: Don't wrap stubbed functions in try/except
            products = await extract_product_prices()

            # Step 3: Apply filters
            await filter_by_price_range(50.0, 150.0)

            # Step 4: Select and add to cart (would be implemented)
            # Step 5: Verify cart
            cart_status = await verify_product_in_cart()

            print(f"Cart has {{cart_status['items']}} items, total: ${{cart_status['total']}}")
            return cart_status
        ```

        **Example: Using Handle-Based Tools (SMS and Calls)**
        This demonstrates proper use of SteerableToolHandle for communication tools.
        ```python
        @verify
        async def send_appointment_reminders(appointments: list[dict]) -> list[str]:
            \"\"\"Sends SMS reminders for multiple appointments.

            Args:
                appointments: List of dicts with 'phone', 'time', 'doctor' keys

            Returns:
                list[str]: List of confirmation messages
            \"\"\"
            # RULE 3: Import inside function
            from typing import List

            confirmations = []

            # RULE 10: Proper error handling
            for appt in appointments:
                try:
                    # RULE 8: Await the async tool
                    sms_handle = await action_provider.send_sms_message(
                        f"Text {{appt['phone']}} about appointment at {{appt['time']}} with Dr. {{appt['doctor']}}"
                    )

                    # Handle returns allow interaction
                    result = await sms_handle.result()
                    confirmations.append(result)

                except Exception as e:
                    print(f"Failed to send SMS to {{appt['phone']}}: {{e}}")
                    # RULE 10: Re-raise to let actor handle
                    raise

            return confirmations

        @verify
        async def make_followup_call_with_questions(phone: str, questions: list[str]) -> dict:
            \"\"\"Makes an interactive phone call with specific questions.

            Args:
                phone: Phone number to call
                questions: List of questions to ask during call

            Returns:
                dict: Call summary with answers
            \"\"\"
            # RULE 3: All imports inside
            from pydantic import BaseModel, Field
            from typing import Optional, Dict

            # Note: start_call is synchronous
            call_handle = action_provider.start_call(
                phone_number=phone,
                purpose="Follow-up call to ask specific questions"
            )

            answers = {{}}

            # Use the handle's interactive methods
            for question in questions:
                ask_handle = await call_handle.ask(question)
                answer = await ask_handle.result()
                answers[question] = answer

            # Get full transcript
            full_result = await call_handle.result()

            # RULE 9: Structured analysis with Pydantic
            class CallAnalysis(BaseModel):
                all_questions_answered: bool
                followup_needed: bool
                satisfaction_level: Optional[str] = Field(default=None)

            CallAnalysis.model_rebuild()

            analysis = await action_provider.reason(
                request="Analyze if all questions were answered satisfactorily",
                context=f"Questions: {{questions}}\\nAnswers: {{answers}}\\nTranscript: {{full_result}}",
                response_format=CallAnalysis
            )

            return {{
                "answers": answers,
                "analysis": analysis.dict(),
                "transcript": full_result
            }}
        ```

        **Example: Complex Multi-Step Plan with Fallbacks**
        This shows advanced patterns including stubbing strategy and error recovery.
        ```python
        @verify
        async def process_customer_data() -> dict:
            \"\"\"Processes customer data with multiple fallback strategies.

            Demonstrates proper error handling and recovery patterns.

            Returns:
                dict: Processed customer information
            \"\"\"
            # RULE 3: Imports inside function
            from pydantic import BaseModel, Field
            from typing import Optional, List
            import json

            # Primary approach: Use the website's export feature
            try:
                await action_provider.act(
                    "Click on 'Export Data' button and select JSON format to start the download or display data"
                )

                # RULE 9: Structured extraction
                class ExportedData(BaseModel):
                    customers: List[dict]
                    export_date: str
                    total_count: int

                ExportedData.model_rebuild()

                data = await action_provider.observe(
                    "Extract the exported customer data",
                    response_format=ExportedData
                )

                return {{"source": "export", "data": data.dict()}}

            except Exception as e:
                print(f"Export feature failed: {{e}}")
                # RULE 10: Log but don't silence - try fallback

                # Fallback: Manually extract from table
                try:
                    class CustomerTable(BaseModel):
                        class Customer(BaseModel):
                            name: str
                            email: str
                            status: str
                            joined_date: Optional[str] = None

                        customers: List[Customer]

                    # RULE 9: Always rebuild outermost model
                    CustomerTable.model_rebuild()

                    table_data = await action_provider.observe(
                        "Extract all customer information from the visible table",
                        response_format=CustomerTable
                    )

                    return {{
                        "source": "table_extraction",
                        "data": {{
                            "customers": [c.dict() for c in table_data.customers],
                            "total_count": len(table_data.customers)
                        }}
                    }}

                except Exception as fallback_e:
                    print(f"Table extraction also failed: {{fallback_e}}")
                    # RULE 10: Must re-raise
                    raise ValueError(f"Both export ({{e}}) and table extraction ({{fallback_e}}) failed")

        # The main plan shows how everything comes together
        @verify
        async def main_plan():
            \"\"\"Main entry point demonstrating a complete workflow.

            RULE 2: This is the required entry point for the plan.
            \"\"\"
            print("Starting customer data processing workflow")

            # Navigate to the system
            await action_provider.navigate("https://crm.example.com")

            # Login (would typically be implemented or stubbed based on confidence)
            await login_to_system("admin", "password123")

            # Process the data with fallbacks
            customer_data = await process_customer_data()

            # Send notifications (demonstrates handle-based tools)
            if customer_data["source"] == "export":
                confirmations = await send_appointment_reminders([
                    {{"phone": "+1234567890", "time": "3pm", "doctor": "Smith"}}
                ])

            print(f"Workflow completed. Processed {{customer_data['data']['total_count']}} customers")
            return customer_data
        ```

        **Browser Automation with Strategic Stubbing**
        This example shows the recommended approach for web automation tasks.
        # This function is implemented directly because navigating and searching are simple, high-confidence actions.
        @verify
        async def search_for_product() -> str:
            \"\"\"Navigates to an e-commerce site and searches for a specific product.\"\"\"
            print("Navigating to store and searching for 'blue sneakers'.")
            await action_provider.navigate("https://fakestore.example.com")
            await action_provider.act(
                "Type 'blue sneakers' into the search bar and click the search button to show products related to 'blue sneakers'"
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
            # from pydantic import BaseModel, Field
            # class ProductDetails(BaseModel):
            #     price: float
            #     review_count: int
            #
            # await action_provider.navigate(product_url)
            # details = await action_provider.observe(
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
            from pydantic import BaseModel, Field
            print(f"Attempting to convert price: ${{product_price_usd}}")

            # --- Primary Approach: Use the website's feature ---
            try:
                await action_provider.act(
                    "Click the currency selector and choose 'EUR' to display the price in Euros (€)"
                )

                class PriceInfo(BaseModel):
                    price_eur: float = Field(description="The price in Euros.")

                observed_price = await action_provider.observe(
                    "What is the product price in Euros?",
                    response_format=PriceInfo
                )
                print("Successfully converted price using the website's feature.")
                return observed_price.price_eur

            except Exception as e:
                print(f"Website's currency converter failed: ${{e}}. Attempting fallback.")

                # --- Fallback Approach: Use the `reason` tool ---
                try:
                    class ConversionResult(BaseModel):
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

        **Example: Isolating Pure Logic for Efficiency**
        This example shows how to factor out a non-browser task into a separate, cacheable function.
        ```python
        @verify
        async def extract_sales_data_from_page() -> list[dict]:
            \"\"\\"Extracts raw sales data from a table on the current page.\"\"\\"
            from pydantic import BaseModel, Field
            from typing import List

            class SalesRecord(BaseModel):
                product_name: str
                quantity: int
                unit_price: float
                date: str

            class SalesData(BaseModel):
                records: List[SalesRecord]

            SalesData.model_rebuild()

            result = await action_provider.observe(
                "Extract all sales records from the table including product name, quantity, unit price, and date",
                response_format=SalesData
            )

            # Convert to list of dicts for easier processing
            return [record.dict() for record in result.records]

        @verify
        async def perform_complex_analysis(sales_records: list[dict]) -> dict:
            \"\"\\"
            Performs a time-consuming analysis on raw data.
            This function contains only pure Python logic and does not use the browser.
            \"\"\\"
            import asyncio
            from datetime import datetime

            print("Performing complex offline analysis...")

            # Simulate complex calculations
            total_sales = sum(r['quantity'] * r['unit_price'] for r in sales_records)
            average_sale = total_sales / len(sales_records) if sales_records else 0

            # Group by product (simulating complex logic)
            product_totals = {{}}
            for record in sales_records:
                product = record['product_name']
                amount = record['quantity'] * record['unit_price']
                product_totals[product] = product_totals.get(product, 0) + amount

            # Find best selling product
            best_product = max(product_totals.items(), key=lambda x: x[1]) if product_totals else (None, 0)

            # Simulate time-consuming computation
            await asyncio.sleep(5)  # Represents complex calculations

            print("Analysis complete.")
            return {{
                "total_sales": total_sales,
                "average_sale": average_sale,
                "best_product": best_product[0],
                "best_product_sales": best_product[1],
                "product_breakdown": product_totals
            }}

        @verify
        async def main_plan():
            \"\"\"
            Main plan to extract and analyze sales data.
            \"\"\"
            # Navigate to the sales report page
            await action_provider.navigate("https://example.com/sales-report")

            # The result of this step will be cached by the actor
            raw_data = await extract_sales_data_from_page()

            # If the plan is modified and restarts after this point,
            # this analysis function will NOT be re-run because its result
            # will be loaded from the cache, saving significant time.
            analysis_results = await perform_complex_analysis(raw_data)

            # Use the analysis results for further actions
            if analysis_results['best_product']:
                await action_provider.act(
                    f"Search for more information about {{analysis_results['best_product']}} to load the product detail page"
                )

            return analysis_results
        ```

        **Example: Proactive Clarification**
        This example shows how to generate a plan that asks for required information before acting.
        ```python
        # This plan is for a vague goal like "Book a hotel for me."
        # The LLM knows it's missing key details, so it uses the request_clarification primitive.
        @verify
        async def get_booking_details_from_user() -> dict:
            "\"\"\"Asks the user for all necessary details to book a hotel.\"\"\""
            city = await request_clarification("Sure, I can book a hotel. What city are you traveling to?")
            check_in = await request_clarification("What is your check-in date?")
            check_out = await request_clarification("And what is your check-out date?")
            return {{"city": city, "check_in": check_in, "check_out": check_out}}

        @verify
        async def main_plan():
            "\"\"\"Main plan to book a hotel after gathering user input.\"\"\""
            details = await get_booking_details_from_user()
            # ... now the plan would proceed to use these details for browser automation ...
            print(f"Searching for hotels in {{details['city']}} from {{details['check_in']}} to {{details['check_out']}}.")
            return f"Search initiated for {{details['city']}}."
        ```
    """,
    )


def _build_dynamic_implement_rules_and_examples(
    tools: Dict[str, Callable],
    strategy_instruction: str,
    tool_usage_instruction: str,
) -> str:
    """Builds the reusable block of core rules and examples for dynamic implementation."""
    tool_reference = _build_tool_signatures(tools)
    handle_apis = _build_handle_apis(tools)

    instructions_and_rules = textwrap.dedent(
        """
        ### 🎯 CRITICAL RULES FOR DYNAMIC FUNCTION IMPLEMENTATION

        1.  **Single Code Block:** Your entire response MUST be a single, valid Python code block.
            ```python
            # ✅ CORRECT: Just one function implementation
            async def extract_data():
                # Full implementation here
                pass

            # ❌ WRONG: Multiple functions or extra code
            def helper():
                pass
            async def extract_data():
                pass
            ```

        2.  **Scope and Imports:** ALL imports must be placed **inside** the function.
            ```python
            # ❌ WRONG: Top-level imports
            from pydantic import BaseModel
            from typing import Optional

            async def my_function():
                pass

            # ✅ CORRECT: All imports inside the function
            async def my_function():
                from pydantic import BaseModel
                from typing import Optional
                import json
                import re
                # Rest of implementation
            ```

        3.  **Decorators & Docstrings:** Include comprehensive docstrings, but NO decorators.
            ```python
            # ❌ WRONG: Using @verify decorator
            @verify
            async def process_data():
                pass

            # ✅ CORRECT: No decorators, clear docstring
            async def process_data(items: list[dict]) -> dict:
                \"\"\"Process and analyze item data.

                Args:
                    items: List of item dictionaries

                Returns:
                    dict: Processed results with statistics
                \"\"\"
            ```

        4.  **Async All The Way:** Function MUST be async.
            ```python
            # ❌ WRONG: Regular function
            def extract_info():
                return data

            # ✅ CORRECT: Async function
            async def extract_info():
                return data
            ```

        5.  **Await Keyword:** ALWAYS await async action_provider methods.
            ```python
            # ❌ WRONG: Missing await
            result = action_provider.observe("Get data")

            # ✅ CORRECT: With await
            result = await action_provider.observe("Get data")
            ```

        6.  **Structured Output with Pydantic - THE COMPLETE PATTERN:**
            ```python
            async def extract_structured_data():
                # Step 1: Import inside function
                from pydantic import BaseModel, Field
                from typing import Optional, List

                # Step 2: Define models inside function
                class Product(BaseModel):
                    name: str
                    price: float
                    # Step 3: Use Optional for potentially missing fields
                    rating: Optional[float] = Field(default=None)
                    in_stock: bool = Field(description="Availability status")

                class ProductList(BaseModel):
                    products: List[Product]
                    total: int

                # Step 4: CRITICAL - Call model_rebuild() on outermost model
                ProductList.model_rebuild()

                # Step 5: Use with response_format
                result = await action_provider.observe(
                    "Extract all products with details",
                    response_format=ProductList
                )

                # ❌ WRONG: Forgetting model_rebuild()
                # ❌ WRONG: Not using Optional for missing fields
                # ❌ WRONG: Defining models outside the function
            ```

        7.  **Robust Error Handling:** Log errors but ALWAYS re-raise.
            ```python
            # ❌ WRONG: Silencing errors
            try:
                result = await risky_operation()
            except Exception as e:
                print(f"Failed: {{e}}")
                return None  # Never do this!

            # ✅ CORRECT: Log and re-raise
            try:
                result = await risky_operation()
            except Exception as e:
                print(f"Operation failed: {{e}}")
                raise  # Always re-raise!

            # ✅ CORRECT: With fallback and re-raise
            try:
                # Primary approach
                result = await primary_method()
            except Exception as e:
                print(f"Primary failed: {{e}}")
                try:
                    # Fallback approach
                    result = await fallback_method()
                except Exception as fallback_e:
                    print(f"Fallback also failed: {{fallback_e}}")
                    raise ValueError(f"Both methods failed: {{e}}, {{fallback_e}}")
            ```

        8.  **Action Provider Usage:** Use directly as global, no imports or type hints.
            ```python
            # ❌ WRONG: Importing or typing ActionProvider
            from somewhere import ActionProvider
            def my_func(action_provider: ActionProvider):
                pass

            # ❌ WRONG: Creating ActionProvider instance
            action_provider = ActionProvider()

            # ✅ CORRECT: Use directly as if it exists globally
            async def my_func():
                result = await action_provider.navigate("https://example.com")
                data = await action_provider.observe("Get page title")
            ```

        9. **Requesting Clarification:**
            ```python
            # ✅ CORRECT: Call as a global function
            destination = await request_clarification("What is your destination city?")

            # ❌ WRONG: Do not call it on action_provider
            # destination = await action_provider.request_clarification(...)
            ```
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
        ### Usage Examples for Dynamic Function Implementation

        **CONTEXT:** You are implementing a SINGLE function that was previously stubbed. These examples show how to properly implement functions following all the rules above.

        **Example 1: Using Handle-Based Tools (SMS Message)**
        This shows how to use the `send_sms_message` tool which returns a SteerableToolHandle.
        ```python
        async def send_appointment_reminder(phone_number: str, appointment_details: str) -> str:
            \"\"\"Sends an SMS reminder about an appointment.

            Args:
                phone_number: The phone number to text
                appointment_details: Details about the appointment

            Returns:
                str: Confirmation message or delivery status
            \"\"\"
            print(f"Sending SMS to {{phone_number}}")

            # The send_sms_message tool returns a handle for ongoing interaction
            try:
                # Await the tool to get the interactive handle
                sms_handle = await action_provider.send_sms_message(
                    f"Text {{phone_number}} to remind them about their {{appointment_details}}"
                )

                # The handle allows ongoing interaction if needed
                # For simple cases, just get the final result
                result = await sms_handle.result()

                print(f"SMS sent successfully: {{result}}")
                return result

            except Exception as e:
                print(f"Failed to send SMS: {{e}}")
                raise
        ```

        **Example 2: Using Handle-Based Tools (Phone Call with Interaction)**
        This demonstrates the full capabilities of SteerableToolHandle with the start_call tool.
        ```python
        async def conduct_detailed_appointment_call(phone_number: str, appointment_info: dict) -> dict:
            \"\"\"Makes an interactive phone call to confirm appointment details.

            Args:
                phone_number: The phone number to call
                appointment_info: Dict with 'date', 'time', 'doctor' keys

            Returns:
                dict: Detailed call outcomes including confirmation status
            "\"\"\
            # Imports inside the function
            from pydantic import BaseModel, Field
            from typing import Optional

            print(f"Starting call to {{phone_number}}")

            # Note: start_call is synchronous and returns a Call handle immediately
            call_handle = action_provider.start_call(
                phone_number=phone_number,
                purpose=f"Confirm appointment on {{appointment_info['date']}} at {{appointment_info['time']}} with Dr. {{appointment_info['doctor']}}"
            )

            try:
                # The Call handle is a SteerableToolHandle with special methods
                # You can interact during the call with ask() or interject()

                # Example of asking a specific question during the call
                allergy_handle = await call_handle.ask("Do you have any medication allergies we should know about?")
                allergy_info = await allergy_handle.result()

                # You can also just wait for the full call to complete
                full_transcript = await call_handle.result()

                print("Call completed, analyzing results...")

                # Define models for structured analysis
                class CallSummary(BaseModel):
                    appointment_confirmed: bool = Field(description="Whether the appointment was confirmed")
                    has_allergies: bool = Field(description="Whether patient reported any allergies")
                    allergy_details: Optional[str] = Field(default=None, description="Specific allergy information if any")
                    needs_followup: bool = Field(description="Whether a follow-up call is needed")
                    additional_notes: Optional[str] = Field(default=None, description="Any other important information")

                # CRITICAL: Always rebuild Pydantic models
                CallSummary.model_rebuild()

                # Analyze the complete call
                analysis = await action_provider.reason(
                    request="Analyze this phone call transcript and extract key information about the appointment confirmation and any medical information discussed",
                    context=f"Full call transcript: {{full_transcript}}\n\nAllergy question response: {{allergy_info}}",
                    response_format=CallSummary
                )

                return {{
                    "confirmed": analysis.appointment_confirmed,
                    "allergies": {{
                        "has_allergies": analysis.has_allergies,
                        "details": analysis.allergy_details
                    }},
                    "needs_folloswup": analysis.needs_followup,
                    "notes": analysis.additional_notes
                }}

            except Exception as e:
                print(f"Call failed or was interrupted: {{e}}")
                raise

        ```

        **Example 3: Browser Data Extraction with Pydantic Models**
        This shows the proper pattern for extracting structured data from web pages.
        ```python
        async def extract_product_listings() -> list[dict]:
            \"\"\"Extracts all product information from a search results page.

            Returns:
                list[dict]: List of products with name, price, rating, and availability
            \"\"\"
            # All imports inside the function
            from pydantic import BaseModel, Field
            from typing import Optional, List

            print("Extracting product listings from current page...")

            # Define the data models inside the function
            class Product(BaseModel):
                name: str = Field(description="Product name or title")
                price: float = Field(description="Numeric price without currency symbol")
                currency: str = Field(description="Currency code or symbol")
                # Use Optional for fields that might not always be present
                rating: Optional[float] = Field(default=None, description="Average rating out of 5")
                review_count: Optional[int] = Field(default=None, description="Number of reviews")
                in_stock: bool = Field(description="Whether the item is available")
                image_url: Optional[str] = Field(default=None, description="Product image URL if visible")

            class ProductListings(BaseModel):
                products: List[Product] = Field(description="All products found on the page")
                total_results: Optional[int] = Field(default=None, description="Total number of results if shown")

            # CRITICAL: Always call model_rebuild() on the outermost model
            ProductListings.model_rebuild()

            try:
                # Use observe with structured output
                result = await action_provider.observe(
                    "Extract all products from this search results page. For each product, get the name, "
                    "numeric price (without currency), currency symbol/code, rating if shown, review count if shown, "
                    "stock availability, and image URL if visible. Also note the total number of results if displayed.",
                    response_format=ProductListings
                )

                # Process and return the data
                products_data = []
                for product in result.products:
                    products_data.append({{
                        "name": product.name,
                        "price": product.price,
                        "currency": product.currency,
                        "rating": product.rating,
                        "review_count": product.review_count,
                        "in_stock": product.in_stock,
                        "image_url": product.image_url
                    }})

                print(f"Successfully extracted {{len(products_data)}} products")
                return products_data

            except Exception as e:
                print(f"Failed to extract product listings: {{e}}")
                raise
        ```

        **Example 4: Complex Operation with Fallback Strategy**
        This demonstrates robust error handling with fallback approaches.
        ```python
        async def complete_checkout_process(payment_info: dict) -> dict:
            \"\"\"Completes the checkout process with payment information.

            Args:
                payment_info: Dict containing 'card_number', 'cvv', 'expiry', 'zip'

            Returns:
                dict: Order confirmation details
            \"\"\"
            from pydantic import BaseModel, Field
            from typing import Optional

            print("Starting checkout process...")

            # Define expected output structure
            class OrderConfirmation(BaseModel):
                order_number: str = Field(description="The order confirmation number")
                total_amount: float = Field(description="Total amount charged")
                delivery_date: Optional[str] = Field(default=None, description="Expected delivery date if shown")
                confirmation_email: Optional[str] = Field(default=None, description="Email where confirmation was sent")

            OrderConfirmation.model_rebuild()

            try:
                # Primary approach: Fill out the payment form
                await action_provider.act(
                    f"Fill out the payment form with card ending in {{payment_info['card_number'][-4:]}}, "
                    f"CVV {{payment_info['cvv']}}, expiry {{payment_info['expiry']}}, and billing zip {{payment_info['zip']}}. "
                    f"Then click the 'Place Order' or 'Complete Purchase' button to see the order confirmation page with order number"
                )

                # Extract confirmation details
                confirmation = await action_provider.observe(
                    "Extract the order confirmation number, total amount charged, expected delivery date, and confirmation email address",
                    response_format=OrderConfirmation
                )

                print(f"Order placed successfully: {{confirmation.order_number}}")
                return {{
                    "success": True,
                    "order_number": confirmation.order_number,
                    "total": confirmation.total_amount,
                    "delivery_date": confirmation.delivery_date,
                    "email": confirmation.confirmation_email
                }}

            except Exception as e:
                print(f"Primary checkout approach failed: {{e}}")

                # Fallback: Try alternative checkout flow
                try:
                    print("Attempting PayPal checkout as fallback...")

                    await action_provider.act(
                        "Click on 'PayPal' or 'Pay with PayPal' option to redirect to PayPal login or show PayPal frame"
                    )

                    # Note: In real scenario, would handle PayPal flow
                    # This is simplified for example
                    return {{
                        "success": True,
                        "order_number": "PAYPAL-PENDING",
                        "total": 0.0,
                        "delivery_date": None,
                        "email": None,
                        "payment_method": "paypal_redirect"
                    }}

                except Exception as fallback_e:
                    print(f"Fallback PayPal approach also failed: {{fallback_e}}")
                    # Re-raise with full context
                    raise ValueError(
                        f"Unable to complete checkout. "
                        f"Credit card error: {{e}}, "
                        f"PayPal error: {{fallback_e}}"
                    )
        ```

        **Example 5: Isolating Pure Logic for Efficiency**
        This demonstrates how to implement a function that separates browser interaction from complex data processing.
        The actor will cache results of each function, so separating pure logic ensures it won't be re-executed.
        ```python
        async def analyze_competitor_pricing(product_name: str) -> dict:
            \"\"\"Analyzes competitor pricing data for a specific product.

            This function demonstrates the pattern of isolating pure computation
            from browser interaction to leverage the actor's caching system.

            Args:
                product_name: The product to analyze pricing for

            Returns:
                dict: Analysis results including price statistics and recommendations
            \"\"\"
            from pydantic import BaseModel, Field
            from typing import List, Optional
            import asyncio

            print(f"Starting competitor pricing analysis for: {{product_name}}")

            # Define the data model for competitor prices
            class CompetitorPrice(BaseModel):
                store_name: str
                price: float
                shipping_cost: Optional[float] = Field(default=0.0)
                availability: str = Field(description="In stock, out of stock, limited stock")
                rating: Optional[float] = Field(default=None)

            class PricingData(BaseModel):
                product_name: str
                competitors: List[CompetitorPrice]
                search_timestamp: str

            PricingData.model_rebuild()

            # Step 1: Extract pricing data from the current page
            pricing_data = await action_provider.observe(
                f"Extract all competitor pricing information for {{product_name}}. "
                "Include store name, price, shipping cost if shown, availability status, and rating if available.",
                response_format=PricingData
            )

            # Step 2: Call a separate function for complex analysis
            # This is the key pattern - isolating the pure logic computation
            # If the plan restarts after this point, the analysis won't be re-run
            analysis_result = await _perform_pricing_analysis(pricing_data.dict())

            # Step 3: Use the analysis to make a decision
            if analysis_result['recommendation'] == 'match_lowest':
                await action_provider.act(
                    f"Update our price to {{analysis_result['suggested_price']}} and confirm the price update is successful"
                )

            return analysis_result

        # This helper function would be implemented separately in the plan
        # It contains pure logic with no browser interaction, making it cacheable
        async def _perform_pricing_analysis(pricing_data: dict) -> dict:
            \"\"\"Performs detailed statistical analysis on pricing data.

            This is a separate function containing only pure Python logic.
            The actor caches its result, so it won't re-execute if the plan restarts.
            \"\"\"
            import statistics
            import asyncio

            print("Performing complex pricing analysis...")

            competitors = pricing_data['competitors']

            # Extract total prices (price + shipping)
            total_prices = [
                c['price'] + c.get('shipping_cost', 0)
                for c in competitors
            ]

            # Simulate complex calculations that might take time
            await asyncio.sleep(2)  # Represents complex computation

            # Calculate statistics
            avg_price = statistics.mean(total_prices) if total_prices else 0
            median_price = statistics.median(total_prices) if total_prices else 0
            min_price = min(total_prices) if total_prices else 0
            max_price = max(total_prices) if total_prices else 0

            # Find best value (considering price and rating)
            best_value_store = None
            best_value_score = float('inf')

            for comp in competitors:
                total = comp['price'] + comp.get('shipping_cost', 0)
                # Lower score is better (price divided by rating)
                score = total / comp.get('rating', 3.0) if comp.get('rating') else total / 3.0
                if score < best_value_score and comp['availability'] != 'out of stock':
                    best_value_score = score
                    best_value_store = comp['store_name']

            # Determine pricing strategy
            our_target_margin = 0.95  # We want to be 5% below average
            suggested_price = round(avg_price * our_target_margin, 2)

            recommendation = 'match_lowest' if suggested_price < min_price else 'competitive'

            print("Analysis complete.")

            return {{
                "average_price": round(avg_price, 2),
                "median_price": round(median_price, 2),
                "min_price": round(min_price, 2),
                "max_price": round(max_price, 2),
                "best_value_store": best_value_store,
                "suggested_price": suggested_price,
                "recommendation": recommendation,
                "total_competitors": len(competitors),
                "analysis_timestamp": pricing_data['search_timestamp']
            }}
        ```

        **Example 6: Requesting Clarification During Implementation**
        This demonstrates how to use request_clarification when the implementation strategy is unclear.
        ```json
        {{
            "action": "request_clarification",
            "reason": "The page contains two visually identical 'Continue' buttons. I need to know which one to click to proceed with the payment.",
            "clarification_question": "I see two 'Continue' buttons on the page. Should I click the one in the 'Order Summary' section or the one at the very bottom of the page?"
        }}
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
    *,
    tools: Dict[str, Callable],
) -> str:
    """
    Dynamically builds the system prompt for the Hierarchical Actor.
    """
    formatted_functions = _format_existing_functions(existing_functions)

    tool_usage_instruction = "Use the `action_provider` global object to interact with the environment. Available tools and their handle APIs have been described in the rules below."

    rules_and_examples = _build_initial_plan_rules_and_examples(
        tools,
        "",
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
    goal: str,
    full_plan_source: str,
    call_stack: list[str],
    function_name: str,
    function_sig: inspect.Signature,
    function_docstring: str,
    parent_code: str,
    clarification_question: str | None,
    clarification_answer: str | None,
    has_browser_screenshot: bool,
    replan_context: str,
    *,
    tools: Dict[str, Callable],
    existing_code_for_modification: Optional[str] = None,
    recent_transcript: Optional[str] = None,
    parent_chat_context: Optional[list] = None,
    failed_interactions_trace: Optional[list] = None,
) -> str:
    """Builds the system prompt for dynamically implementing or modifying a function."""

    modification_instructions = ""
    if existing_code_for_modification:
        modification_instructions = textwrap.dedent(
            f"""
            ---
            ### 📌 CRITICAL INSTRUCTIONS: MODIFY EXISTING FUNCTION
            You MUST rewrite the entire `{function_name}` function to incorporate a new change. Analyze the user's request and the original code, then produce the complete, final version of the function.

            **Original Function Code:**
            ```python
            {existing_code_for_modification}
            ```

            **Modification Request:**
            {replan_context}

            **Your Task:**
            - **Rewrite the Entire Function:** Your output MUST be a single, complete `async def {function_name}` block.
            - **Integrate Changes:** Seamlessly blend the new logic with the old. If the user is adding a step, append it logically. If they are correcting a mistake, replace the faulty code.
            - **Update Docstrings:** Ensure the function's docstring accurately reflects the *complete* purpose of the newly modified function. If adding steps, you can use a numbered list.
            ---
            """,
        )
    else:
        modification_instructions = textwrap.dedent(
            f"""
            ---
            ### 📌 CRITICAL INSTRUCTIONS: IMPLEMENT STUB FUNCTION
            You are implementing this function for the first time. Its purpose was defined in the initial plan, but the implementation was deferred.

            **Reason for Implementation:**
            {replan_context}
            ---
            """,
        )

    debugging_trace_section = ""
    if failed_interactions_trace:
        formatted_trace = "\n".join(f"- {log}" for log in failed_interactions_trace)
        debugging_trace_section = textwrap.dedent(
            f"""
        ---
        ### 🕵️ Debugging Context: Agent Trace from Failed Attempt
        CRITICAL: The previous attempt to run this function failed. The following is the detailed trace from the browser agent during the failure. Analyze it carefully to understand the exact problem on the page and design a robust solution. This is your primary debugging tool.

        {formatted_trace}
        ---
        """,
        )
    clarification_section = ""
    if clarification_question and clarification_answer:
        clarification_section = textwrap.dedent(
            f"""
            ---
            ### User Clarification Provided
            CRITICAL: The plan was previously stuck, but the user has provided the following clarification. You MUST use this new information to fix the function.

            - **Your Question:** "{clarification_question}"
            - **User's Answer:** "{clarification_answer}"
            ---
            """,
        )

    transcript_section = ""
    if recent_transcript:
        transcript_section = textwrap.dedent(
            f"""
        ---
        ### Recent Conversation Transcript
        ```
        {recent_transcript}
        ```
        """,
        )

    chat_context_section = ""
    if parent_chat_context:
        chat_context_section = textwrap.dedent(
            f"""
        ---
        ### Full Parent Chat Context
        ```json
        {json.dumps(parent_chat_context, indent=2)}
        ```
        """,
        )

    browser_context_section = ""
    if has_browser_screenshot:
        browser_context_section = """
        **Current Browser View (Screenshot):**
        An image of the current browser page has been provided. Analyze it carefully to inform your new implementation.
        """

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

    strategy_instruction = _build_shared_strategy_principles()
    tool_usage_instruction = "Use the `action_provider` global object to interact with the environment. Available tools and their handle APIs have been described in the rules below."
    rules_and_examples = _build_dynamic_implement_rules_and_examples(
        tools,
        strategy_instruction,
        tool_usage_instruction,
    )

    return textwrap.dedent(
        f"""
        You are an expert Python programmer and a master strategist. Your task is to analyze the state of a running plan and decide the best course of action for the function `{function_name}`.

        ---
        ### Overall Goal (Source of Truth)
        Your implementation MUST satisfy all of the following requirements.

        {goal}
        ---

        **CRITICAL: You must choose one of four actions:**
        1.  **`implement_function`**: Write the Python code for `{function_name}`. Choose this if the function's goal is achievable from the current browser state. **Your code MUST be a single, self-contained `async def` function block. DO NOT include top-level imports or class definitions outside the function.** All necessary imports and helper classes MUST be defined *inside* the function.
        2.  **`skip_function`**: Bypass this function entirely. Choose this if you observe that the function's goal is **already completed** or is now **irrelevant**. For example, skip a "log in" function if you are already logged in.
        3.  **`replan_parent`**: Escalate the failure to the calling function. Choose this if the current function is **impossible to implement** because of a mistake made in a *previous* step. For example, if the goal is "apply filters" but the page has no filter controls, the error lies with the parent function that navigated to the wrong page or failed to get to the right state.
        4.  **`request_clarification`**: Ask the user for help. Choose this if you cannot devise a reliable strategy to fix the function from the available information. For example, if required UI elements are missing or behaving unexpectedly, or if there are multiple possible approaches and you're unsure which the user prefers. **You must provide a clear, specific `clarification_question`.**

        {modification_instructions}
        {debugging_trace_section}
        {clarification_section}
        {transcript_section}
        {chat_context_section}
        {context_section}

        ### Situation Analysis
        **Function to Address:** `async def {function_name}{function_sig}`
        **Purpose of this Function:** "{function_docstring}"
        {browser_context_section or "No browser state available."}
        A screenshot of the current browser page has been provided. **Use it as the primary source of truth.**

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
    clarification_question: Optional[str] = None,
    clarification_answer: Optional[str] = None,
    recent_transcript: Optional[str] = None,
    parent_chat_context: Optional[list] = None,
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
        clarification_question: An optional question that was previously asked.
        clarification_answer: An optional answer that was received.

    Returns:
        The complete prompt string for the verification LLM call.
    """
    formatted_interactions = []
    formatted_agent_traces = []
    for interaction in interactions:
        kind, act, obs, *logs = interaction
        logs = logs[0] if logs else []

        log_entry = ""
        if kind == "observe":
            log_entry = f"- Action: `{act}`, Observation: `{obs or 'N/A'}`"
        else:
            log_entry = f"- Action: `{act}` with result `{obs}`"

        if logs:
            log_details = "\n".join([f"    {line}" for line in logs])
            log_entry += f"\n  - Agent Logs:\n{log_details}"
            trace_log = "\n".join(f"  {line}" for line in logs)
            formatted_agent_traces.append(f"- For Action: `{act}`\n{trace_log}")

        formatted_interactions.append(log_entry)

    interactions_log = (
        "\n".join(formatted_interactions)
        or "No browser actions were logged for this step."
    )

    agent_trace_section = "No low-level agent trace was recorded for this step."
    if formatted_agent_traces:
        traces_joined = "\n".join(formatted_agent_traces)
        agent_trace_section = textwrap.dedent(
            f"""
        ---
        ### 🔬 Low-Level Agent Trace (Ground Truth)
        This is the detailed "thought process" from the underlying browser agent as it performed the actions. **This is your most important source of truth.** It reveals *why* an action was taken and what the agent observed at a micro-level. Analyze it carefully to understand the root cause of any success or failure.

        {traces_joined}
        ---
        """,
        )
    screenshot_context_section = ""
    if has_browser_screenshot:
        screenshot_context_section = textwrap.dedent(
            """
            ---
            ### 📸 Visual Evidence (Screenshot)
            You have been provided a **screenshot** of the browser's final state. Use this to visually confirm the outcome described in the agent trace.
            """,
        )
    return_value_log = f"```\n{repr(function_return_value)}\n```"

    source_code_section = f"""
---
### ⚙️ Function Implementation
```python
{function_source_code or "Source code not available."}
```
"""

    transcript_section = ""
    if recent_transcript:
        transcript_section = textwrap.dedent(
            f"""
        ---
        ### 📖 Recent Conversation Transcript
        ```
        {recent_transcript}
        ```
        """,
        )

    chat_context_section = ""
    if parent_chat_context:
        chat_context_section = textwrap.dedent(
            f"""
        ---
        ### 💬 Full Parent Chat Context
        ```json
        {json.dumps(parent_chat_context, indent=2)}
        ```
        """,
        )

    clarification_section = ""
    if clarification_question and clarification_answer:
        clarification_section = textwrap.dedent(
            f"""
        ---
        ### 💡 User Clarification Provided
        CRITICAL: You previously requested clarification because the outcome was ambiguous. The user has provided an answer. Use this new information as the deciding factor in your final assessment.

        - **Your Question:** "{clarification_question}"
        - **User's Answer:** "{clarification_answer}"
        ---
        """,
        )

    return textwrap.dedent(
        f"""
        You are a pragmatic and meticulous Quality Assurance expert for an autonomous agent. Your task is to assess if an executed function has made **meaningful and accurate progress** towards the **Overall User Goal**, even if the website behaves in unexpected ways.

        **🎯 Overall User Goal:** "{goal}"
        **🔍 Function Under Review:** `{function_name}`
        **Intent (Purpose of this function):** {function_docstring or 'No docstring provided.'}

        {source_code_section}
        {agent_trace_section}
        {screenshot_context_section}
        {clarification_section}
        {transcript_section}
        {chat_context_section}

        **📊 Execution Evidence**
        **Function Return Value:**
        {return_value_log}
        **High-Level Tool Interactions Log:**
        {interactions_log}

        ---
        ### 🧠 Your Decision-Making Framework
        You MUST follow this reasoning process to arrive at a decision.

        **Step 1: Scrutinize the Low-Level Agent Trace.**
        - This is your primary evidence. Read the agent's step-by-step reasoning. Does its logic hold up? Did it correctly identify elements? Did it notice any errors or unexpected UI changes?
        - The trace is the ground truth of what happened.

        **Step 2: Assess the Core Purpose vs. The Evidence.**
        - Look at the **Intent** and the **Overall Goal**. What was the most important outcome this function was supposed to achieve?
        - Compare this with the hard evidence: the agent's reasoning in the **Trace**, the visual **Screenshot**, and the final **Return Value**.

        **Step 3: Choose Your Action.**

        - **Is the outcome definitively correct and does it advance the goal?**
          - The **Agent Trace** shows sound reasoning, the **Screenshot** confirms the final state, and the **Return Value** is correct.
          - Choose **`ok`**.

        - **Is the outcome definitively wrong?**
          - The **Agent Trace** shows the agent made a mistake (e.g., clicked the wrong button, extracted wrong text). The **Screenshot** or **Return Value** confirms the error.
          - Choose **`reimplement_local`**.

        - **Is the function's entire premise flawed?**
          - The **Agent Trace** shows the agent correctly reasoning that it *cannot* perform the action (e.g., "I am looking for a 'shipping' button, but the page text says 'shipping is calculated at checkout'").
          - Choose **`replan_parent`**.

        - **Are you unsure, or did the agent's trace reveal confusion or ambiguity?**
          - The **Agent Trace** shows the agent struggling or making an assumption you cannot verify (e.g., "I clicked the button, but I am not sure if it worked.").
          - Choose **`request_clarification`**. This is your default for ambiguity.

        ---
        ### 💡 Examples of Verification Decisions


        **Example 1: A Clear Success (`ok`)**
        - **Goal**: "Find the cheapest flight to New York and select it."
        - **Function**: `select_cheapest_flight()`
        - **Agent Trace**:
        - `◆ [act] Find and select the cheapest flight option.`
        - `REASONING: I am scanning the flight results. I see three options: Delta for $250, United for $220, and Spirit for $195. The Spirit flight is the cheapest. I will click the 'Select' button for the Spirit flight.`
        - `⊙ click 'Select' button next to 'Spirit - $195'`
        - `✓ done`
        - **Return Value**: `{{'airline': 'Spirit', 'price': 195}}`
        - **Screenshot**: Shows the "Passenger Details" page, with a header that says "Your flight with Spirit to JFK".
        - **Correct Assessment**:
        ```json
        {{
            "status": "ok",
            "reason": "The function's intent was fully achieved. The Agent Trace clearly shows correct reasoning for identifying the cheapest flight. The Function Return Value confirms the correct data was extracted, and the Screenshot proves the action resulted in successful progress to the next step in the booking process."
        }}
        ```

        ---

        **Example 2: An Ambiguous Outcome (`request_clarification`)**
        - **Goal**: "Download the Q3 financial report."
        - **Function**: `download_q3_report()`
        - **Agent Trace**:
        - `◆ [act] Download the Q3 financial report.`
        - `REASONING: I have navigated to the reports page. I see two links: 'Q3 2024 Summary (PDF)' and 'Q3 2024 Raw Data (CSV)'. The user's request is ambiguous about the required format. I will assume the PDF summary is what they want and proceed with that.`
        - `⊙ click link with text 'Q3 2024 Summary (PDF)'`
        - `✓ done`
        - **Return Value**: `None`
        - **Screenshot**: Shows the browser's "File Download" dialog box for a file named `Q3_Report_Summary.pdf`.
        - **Correct Assessment**:
        ```json
        {{
            "status": "request_clarification",
            "reason": "The function's success is ambiguous. The Agent Trace reveals it encountered two valid options (PDF and CSV) and had to make an assumption. While it successfully initiated a download, it may not be the format the user needs. It is best to verify with the user before proceeding.",
            "clarification_question": "I found two versions of the Q3 report: a PDF summary and a CSV with raw data. I have started downloading the PDF. Is this the correct one, or do you need the CSV file instead?"
        }}
        ```

        ---

        **Example 3: A Clear Tactical Failure (`reimplement_local`)**
        - **Goal**: "Sign me up for the newsletter with 'test@example.com'."
        - **Function**: `submit_newsletter_signup(email='test@example.com')`
        - **Agent Trace**:
        - `◆ [act] Submit the newsletter signup form with the provided email.`
        - `REASONING: I see the email input field and the 'Subscribe' button. I will type the email address and then click the button.`
        - `⌨︎ type "test@example.com" into the email field`
        - `⊙ click 'Subscribe' button`
        - `✓ done`
        - **Return Value**: `None`
        - **Screenshot**: Shows the same signup form, but a new red error message is now visible below the input field, which reads: "Please provide a corporate email address."
        - **Correct Assessment**:
        ```json
        {{
            "status": "reimplement_local",
            "reason": "A tactical error occurred. The Agent Trace confirms the steps (typing, clicking) were executed as intended. However, the Screenshot provides definitive evidence of failure through the 'Please provide a corporate email address' error message. The function's logic needs to be re-run, likely after obtaining a valid email from the user."
        }}
        ---
        Now, provide your assessment based on all the evidence and the decision framework. Respond with ONLY the JSON object.
        """,
    )


def build_ask_prompt(
    goal: str,
    state: str,
    call_stack: str,
    context_log: str,
    question: str,
) -> str:
    """
    Builds the system prompt for answering questions about the plan's state.

    Args:
        goal: The overall goal of the plan.
        state: The current lifecycle state of the plan.
        call_stack: The current function call stack.
        context_log: A log of recent actions.
        question: The user's question.

    Returns:
        The complete prompt string.
    """
    return textwrap.dedent(
        f"""
        You are an AI assistant who is actively performing a web automation task. The user has paused to ask you a question. Your persona is that of the one performing the work. Speak in the first person ("I am doing...", "I just finished...").

        You have been provided with a complete picture of your current situation:
        1. **Current Goal:** This is your primary objective. It may have been updated by the user.
        2. **Full Action Log:** This is a chronological history of everything that has happened, including your actions, verifications, and any user interjections or clarifications. This is your memory.
        3. **Current Browser View:** A screenshot of what you see on the screen RIGHT NOW. This is your most important source of truth for visual questions.
        4. **Call Stack:** Shows which part of your plan you are currently executing.
        5. **Tools:** You have access to one tool: `query_browser`. This tool allows you to ask questions about the parent agent's memory, including its past actions and observations.

        First, carefully review the context of the parent agent provided below. Then, formulate a plan to answer the user's question. This may involve one or more calls to the `query_browser` tool. Once you have gathered enough information, provide a final, concise answer to the user.
        **Current Goal:** {goal}
        **Current State:** {state}
        **Current Call Stack:** {call_stack}

        --- FULL ACTION LOG ---
        {context_log}
        --- END LOG ---

        Based on all of this information, and paying close attention to the **Action Log** for recent user updates and the **Browser View** for the current visual state, answer the user's question.

        **User's Question:** "{question}"
        **Your Answer:**
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


def build_interjection_prompt(
    interjection: str,
    parent_chat_context: list[dict] | None,
    plan_source_code: str,
    call_stack: list[str],
    action_log: list[str],
    goal: str,
    idempotency_cache: Dict[tuple, Any],
    *,
    tools: Dict[str, Callable],
) -> str:
    """Builds the system prompt for the Interjection Handler LLM."""
    tool_reference = _build_tool_signatures(tools)
    handle_apis = _build_handle_apis(tools)
    strategy_principles = _build_shared_strategy_principles()

    cache_summary = _format_cache_summary(idempotency_cache)

    call_stack_str = (
        " -> ".join(call_stack) if call_stack else "Not inside any function."
    )
    recent_actions = "\n".join(f"- {log}" for log in action_log) or "No actions yet."
    chat_history = (
        json.dumps(parent_chat_context, indent=2)
        if parent_chat_context
        else "No prior conversation."
    )

    return textwrap.dedent(
        f"""
    You are an expert Python programmer and a master strategist responsible for steering a live-running automated plan. A user has interjected with a new instruction while the plan was executing.

    ### Full Situational Context
    - **User's Interjection:** "{interjection}"
    - **Current Goal (Source of Truth):** "{goal or 'None (This is a teaching session)'}"
    - **Full Conversation History:** {chat_history}
    - **Current Plan Source Code (`plan_source_code`):**
      ```python
      {plan_source_code}
      ```
    - **Current Execution Point (Call Stack):** `{call_stack_str}`
    - **Most Recent Plan Actions:**
      {recent_actions}

    {cache_summary}
    ---
    ### Cache Invalidation Rules (CRITICAL)
    1.  **No Phantom Invalidations**: Only list functions in `invalidate_functions` if they appear in the `Cache Status` list above.
    2.  **Surgical Invalidation**: Use `invalidate_functions` to clear the entire cache for a function, or `invalidate_steps` to clear only a portion of it. Be as minimal as possible to ensure an efficient replay.
    3.  **You may omit `cache`** if nothing needs invalidation.
    ---
    {strategy_principles}
    ---
    ### Your Task: Analyze, Decide, Patch, and Propose Cache Strategy

    **1. Analyze Intent and Choose an Action:** First, analyze the user's intent to choose the single best action from the Decision Tree below.

    **2. Perform Global Code Analysis:** Once you've chosen `modify_task` or `refactor_and_generalize`, you must act like an expert developer.
        - **Read the ENTIRE `plan_source_code`**.
        - **Identify ALL necessary changes.** A single user request might require changing a function's implementation, updating its call site in a parent function, and even modifying the docstrings.
        - **Generate Patches:** For every function that needs to be changed, create a `FunctionPatch` object containing its full, updated source code.

    **3. Devise an Optimal Cache Strategy (CRITICAL for `modify_task`):**

        **Golden Rule of Replay:** After you submit your patches, the plan **always restarts execution from the beginning of `main_plan`**. Your task is to craft a cache invalidation plan that makes this replay as fast as possible by preserving all valid caches.

        **Scenario 1: Invalidating Downstream Dependencies (`invalidate_functions`)**
        * **Situation:** The plan is `A_login() -> B_fetch_user_data("123") -> C_generate_report(...)`. The correctness of `C` depends on the data fetched in `B`. The user interjects: "Sorry, I meant user ID `'456'`."
        * **Analysis:** Changing the `user_id` in `B` will cause it to navigate to a new page and fetch different data. Because `C` relies on this data, its previous cached result is now invalid and must also be cleared. `A_login`, however, is unaffected.
        * **Correct `modify_task` Response:**
            ```json
            {{
                "action": "modify_task",
                "reason": "User changed the target user ID. This invalidates both the data fetching step (B) and the report generation step (C) which depends on it.",
                "patches": [
                    {{
                        "function_name": "main_plan",
                        "new_code": "async def main_plan():\\n    await A_login()\\n    user_data = await B_fetch_user_data(user_id='456')\\n    await C_generate_report(user_data)"
                    }}
                ],
                "cache": {{
                    "invalidate_functions": ["B_fetch_user_data", "C_generate_report"]
                }}
            }}
            ```
        * **Replay Analysis:**
            1.  Execution starts at `main_plan`.
            2.  `await A_login()` runs. **Result: CACHE HIT**.
            3.  `await B_fetch_user_data(user_id='456')` runs. **Result: CACHE MISS**. It executes for real.
            4.  `await C_generate_report(...)` runs. **Result: CACHE MISS**. It executes for real with the new data from `B`.

        ---

        **Scenario 2: Invalidating a Portion of a Function (`invalidate_steps`)**
        * **Situation:** Function `B` has 5 internal steps. After step 2 completes, the user interjects: "In step B, after step 2, you need to add a new action before continuing."
        * **Analysis:** Only the latter part of function `B` is affected. The initial steps (1 and 2) inside `B` are still valid and their caches should be preserved to save time.
        * **Correct `modify_task` Response:**
            ```json
            {{
                "action": "modify_task",
                "reason": "User added a new step in the middle of function B. Invalidating from step 3 onwards.",
                "patches": [
                    {{
                        "function_name": "B",
                        "new_code": "async def B(parameter: str):\\n    # step 1\\n    await action_provider.act('Step B1')\\n    # step 2\\n    await action_provider.act('Step B2')\\n    # new step 2.5\\n    await action_provider.act('Newly added Step B2.5')\\n    # step 3\\n    await action_provider.act('Step B3')\\n    # ..."
                    }}
                ],
                "cache": {{
                    "invalidate_steps": [
                        {{"function_name": "B", "from_step_inclusive": 3}}
                    ]
                }}
            }}
            ```
        * **Replay Analysis:**
            1.  Execution starts at `main_plan`.
            2.  `await A()` runs. **Result: CACHE HIT**.
            3.  `await B(...)` runs.
                * Internal Step 1: **CACHE HIT**.
                * Internal Step 2: **CACHE HIT**.
                * Internal Step 2.5 (new): **CACHE MISS**.
                * Internal Step 3 onwards: **CACHE MISS** (due to invalidation).
            4.  `await C()` runs. **Result: CACHE HIT**.

        ---

        **Scenario 3: No Invalidation Needed (Structural Change)**
        * **Situation:** The plan `A() -> B() -> C()` has fully completed `A` and `B`. The user interjects: "Actually, you don't need to do C. Just stop after B."
        * **Analysis:** The user is only changing the sequence of calls in `main_plan`. The internal logic and inputs for functions `A` and `B` have not changed, so their caches are perfectly valid.
        * **Correct `modify_task` Response:**
            ```json
            {{
                "action": "modify_task",
                "reason": "User requested to remove step C from the plan.",
                "patches": [
                    {{
                        "function_name": "main_plan",
                        "new_code": "async def main_plan():\\n    await A()\\n    await B()"
                    }}
                ]
            }}
            ```
        * **Replay Analysis:**
            1.  Execution starts at the *new* `main_plan`.
            2.  `await A()` runs. **Result: CACHE HIT**.
            3.  `await B()` runs. **Result: CACHE HIT**.
            4.  The plan finishes. The replay is extremely fast.

    ---
    #### 🧠 Distinguishing `modify_task` from `refactor_and_generalize`
    This is your most critical strategic decision.
    - **Choose `modify_task` to alter the BEHAVIOR of the current plan.** Use this when the user wants to add a step, correct a step, or change a parameter. The fundamental *structure* of the plan (which functions call which other functions) remains the same. Do not delete the existing steps and/or workflow unless the user specifically asks you to do so.
    - **Choose `refactor_and_generalize` to alter the STRUCTURE of the plan itself.** Use this when the user asks you to re-apply an entire taught sequence to a new target. This implies that a monolithic, step-by-step plan should be abstracted into a reusable, parameterized skill.

    ---
    ### Decision Tree & Action-Specific Examples
    You MUST respond with a JSON object that strictly adheres to the `InterjectionDecision` Pydantic model.

    #### 1. `modify_task` (Altering Plan Behavior)
    - **Context**: The plan has `main_plan()` which calls `search_products("laptops")`. The user says:
        > "Whoops, I meant to search for 'monitors', not laptops."
    - **Analysis**: The user's intent is to change a parameter. This modifies the plan's behavior but not its structure. A global analysis is needed to find all code that references "laptops". This requires patching both the `search_products` function (to change the default value) and the `main_plan` (to change the specific call).
    - **JSON Output**:
        ```json
        {{
            "action": "modify_task",
            "reason": "User wants to correct the search term from 'laptops' to 'monitors'.",
            "patches": [
                {{
                    "function_name": "main_plan",
                    "new_code": "async def main_plan():\\n    # ...\\n    await search_products(\\"monitors\\")\\n    # ..."
                }},
                {{
                    "function_name": "search_products",
                    "new_code": "async def search_products(product_type: str = \\"monitors\\") -> None:\\n    # ... function implementation ..."
                }}
            ]
        }}
        ```

    #### 2. `refactor_and_generalize` (Altering Plan Structure)
    - **Context**: The plan was taught step-by-step to find information on "Michael Smith". The `main_plan` is now a monolithic block of code with these steps. The user says:
        > "Awesome. Now do the same for 'Sam Parker'."
    - **Analysis**: The user is asking to re-apply the *entire taught process* to a new person. This is a structural change. The monolithic `main_plan` should be refactored into a reusable `enrich_lead(lead_name: str)` function.
    - **JSON Output**:
        ```json
        {{
            "action": "refactor_and_generalize",
            "reason": "User wants to repeat the taught lead enrichment process for a new person, 'Sam Parker'.",
            "generalization_context": "The user wants to apply the same process (search LinkedIn, GitHub, etc.) to the new lead 'Sam Parker'."
        }}
        ```

    #### 3. `replace_task` (Fundamental Goal Change)
    - **Context**: The current goal is to find a lasagna recipe. The user says:
        > "Actually, forget the recipe. Find me the cheapest flights from SFO to LAX for next weekend."
    - **Analysis**: This is a complete change of goal. The existing plan is irrelevant. The best action is to start over with a new goal.
    - **JSON Output**:
        ```json
        {{
            "action": "replace_task",
            "reason": "User has completely changed the goal from finding a recipe to booking a flight.",
            "new_goal": "Find the cheapest flights from SFO to LAX for next weekend."
        }}
        ```

    #### 4. `explore_detached` (Side Quest)
    - **Context**: The plan is in the middle of filling out a checkout form. The user says:
        > "Quick question - what's the weather like in New York right now?"
    - **Analysis**: This is a temporary, unrelated side-quest. It should be handled in a detached way (like a new tab) so it doesn't disrupt the main task's browser state.
    - **JSON Output**:
        ```json
        {{
            "action": "explore_detached",
            "reason": "User asked an unrelated question about the weather, which should be handled as a side-quest.",
            "new_goal": "Check the current weather in New York."
        }}
        ```

    #### 5. `clarify` (Ambiguous Instruction)
    - **Context**: An action just failed. The user says:
        > "No, that's wrong. Fix it."
    - **Analysis**: The instruction "Fix it" is ambiguous. It's impossible to generate a correct code patch without more specific information. The agent must ask for clarification.
    - **JSON Output**:
        ```json
        {{
            "action": "clarify",
            "reason": "The user's instruction 'Fix it' is too ambiguous. I need more specific details to make the correct change.",
            "clarification_question": "I understand the last step was incorrect. Could you please tell me more specifically what I should have done instead?"
        }}
        ```

    #### 6. `complete_task` (Task is Finished)
    - **Context**: The agent has just successfully provided the user with the requested information. The user says:
        > "Perfect, that's all I needed. Thanks!"
    - **Analysis**: The user is signaling that the task is complete and no further actions are required.
    - **JSON Output**:
        ```json
        {{
            "action": "complete_task",
            "reason": "User has confirmed the task is complete."
        }}
        ```
    ---
    ### Tools Reference
    Your generated code can use the global `action_provider` object with these methods:
    ```json
    {tool_reference}
    ```
    ### Handle APIs
    Some tools return "handle" objects for ongoing interaction:
    {handle_apis}
    ---
    Now, provide your decision. Your response must be ONLY the JSON object.
    """,
    ).strip()


def _build_simple_script_rules(tools: Dict[str, Callable]) -> str:
    """Builds a streamlined set of rules for simple, non-decomposed scripts."""
    tool_reference = _build_tool_signatures(tools)
    rules = textwrap.dedent(
        f"""
        ### 🎯 CRITICAL RULES FOR SCRIPTING
        1.  **Sequence of Calls**: Your code must be a simple sequence of `await` calls on the `action_provider`. Do not define new functions.
        2.  **Await Keyword**: You MUST `await` all `async` tool calls (like `navigate`, `act`, `observe`).
        3.  **Action Provider**: Use the `action_provider` object directly as if it's a global variable. Do not import or define it.

        ### Tools Reference
        You have access to a global `action_provider` object with the following methods.
        ```json
        {tool_reference}
        ```

        ### Examples of Correction Scripts

        # ---
        # Example 1: The agent navigated to the wrong page ('/settings') instead of the user's profile.
        # Goal: Get back to the correct user profile page.
            await action_provider.navigate("https://example.com/user/123/profile")

        # ---
        # Example 2: The agent opened an unwanted "Share" popup that is now obscuring the page content.
        # Goal: Close the popup to restore view of the underlying page.
        await action_provider.act("Click the 'X' or 'Close' button on the 'Share this article' popup")

        # ---
        # Example 3: The agent typed the wrong address into a form field.
        # Goal: Clear the incorrect text from the 'Street Address' field.
        # Note: The next implementation will handle typing the correct text. This script ONLY restores the state.
        await action_provider.act("Clear the text in the 'Street Address' field")

    """,
    )
    return rules


def build_course_correction_prompt(
    last_verified_function_name: str,
    last_verified_url: str,
    current_url: str,
    failed_function_name: str,
    failed_function_docstring: str,
    verification_reason: str,
    *,
    tools: Dict[str, Callable],
) -> str:
    """
    Builds the prompt for the course correction LLM.
    """
    scripting_rules = _build_simple_script_rules(tools)

    return textwrap.dedent(
        f"""
        You are a state recovery specialist for an autonomous web agent.

        A function just failed, and the browser may have been left in a corrupted state. Your task is to compare the state of the browser BEFORE the failure to its state AFTER the failure and decide if a course-correction script is needed.

        ---
        ### State Analysis

         **1. Reason for Failure (from Verification):** CRITICAL: The verification step determined the function failed for the following reason. Use this as your primary guide.
        "{verification_reason}"

        **2. The "Last Known Good" State (BEFORE the failure):**
        This is the state after the function `{last_verified_function_name}` completed successfully.
        - **URL:** `{last_verified_url}`
        - **Screenshot:** A screenshot of this state is provided. (1st image)

        **3. The "Current / Corrupted" State (AFTER the failure):**
        This is the state where the function `{failed_function_name}` (Purpose: "{failed_function_docstring}") failed.
        - **URL:** `{current_url}`
        - **Screenshot:** A screenshot of this current state is also provided. (2nd image)

        ---
        ### Your Task

        1.  **Analyze the Failure.** Did the failed function navigate away to a completely wrong page, or did it just fail an *interaction* on the correct page (e.g., couldn't click a button, a popup appeared)?
        2.  **Decide if Correction is Needed.**
            - If the browser is on a **completely irrelevant page**, set `correction_needed` to `true` and write code to navigate back to the "Last Known Good" state.
            - **IMPORTANT:** If the browser is still on the **correct page** and the failure was just a faulty interaction, the best course of action is often **no correction**. Set `correction_needed` to `false`. This allows the actor to immediately retry implementing the function on the correct page without wasting a navigation step.
            - If a popup or modal appeared that needs to be closed, set `correction_needed` to `true` and write a script to close it.
        3.  **If correction is needed, write `correction_code`.**
            - This must be a simple, self-contained Python script.
            - Use `action_provider.navigate` or `action_provider.act`.
            - **Goal:** Get from the "Current" state back to the "Last Known Good" state.
            - **Example:** If the agent is on the wrong page, the script might be `await action_provider.navigate('{last_verified_url}')`.
            - **Example:** If a popup is open, the script might be `await action_provider.act("Click the 'Close' button on the popup")`.
            - **Keep it simple!** Do not try to re-run the failed function. Only restore the state.

        ---
        ### Scripting Rules & Tool Reference
        You MUST follow these rules when writing the `correction_code`.
        {scripting_rules}
        ---

        Respond with ONLY the JSON object matching the `CourseCorrectionDecision` schema.
        """,
    )


def build_sandbox_merge_prompt(
    main_goal: str,
    main_plan_source: str,
    sandbox_goal: str,
    sandbox_result: str,
) -> str:
    """Builds the prompt for the sandbox merge decision LLM."""
    return textwrap.dedent(
        f"""
    You are a strategic assistant for an autonomous agent. A "sandbox" task was just completed, and you must decide if its findings should be used to modify the main plan.

    ### Main Plan Context
    - **Main Goal:** "{main_goal}"
    - **Main Plan Source Code:**
    ```python
    {main_plan_source}
    ```

    ### Sandbox Task Context
    - **Sandbox Goal:** "{sandbox_goal}"
    - **Sandbox Result:** "{sandbox_result}"

    ### Your Task
    1.  Analyze the sandbox result in the context of the main goal.
    2.  Does the sandbox result provide information or a completed sub-task that makes the main plan more efficient or more likely to succeed?
    3.  If yes, set `modification_needed` to `true` and formulate a `modification_request` that clearly instructs the actor on how to alter the main plan.
    4.  If no, set `modification_needed` to `false`.

    Respond ONLY with a JSON object matching the `SandboxMergeDecision` schema.
    """,
    )


def build_refactor_prompt(
    monolithic_code: str,
    generalization_request: str,
    action_log: str,
    current_url: str,
    *,
    tools: Dict[str, Callable],
) -> str:
    """
    Builds the prompt for refactoring a monolithic plan into modular functions,
    including intelligent state correction in the new main_plan.

    Args:
        monolithic_code: The source code of the current single-function plan.
        generalization_request: The user's request to generalize the logic.
        action_log: The full execution trace for deducing the start state.
        current_url: The browser's URL at the time of interjection.
        tools: The available tools for the actor.

    Returns:
        The complete prompt string for the refactoring LLM call.
    """
    strategy_instruction = "Your task is to rewrite the script below to incorporate the user's change request."
    tool_usage_instruction = "Use the `action_provider` global object to interact with the environment. Available tools and their handle APIs have been described in the rules below."
    rules_and_examples = _build_initial_plan_rules_and_examples(
        tools,
        strategy_instruction,
        tool_usage_instruction,
    )

    return textwrap.dedent(
        f"""
        You are an expert Python programmer who refactors monolithic scripts into modular, reusable code. You must be mindful of the agent's state when generating the new plan.

        ### Full Context
        - **User's Generalization Request:** "{generalization_request}"
        - **Browser's Current URL:** `{current_url}`
        - **Full Execution Action Log (for context):**
        ```
        {action_log}
        ```
        - **Current Monolithic Code to Refactor:**
        ```python
        {monolithic_code}
        ```

        ---
        ### Your Task: A Two-Part Refactoring Process

        **Part 1: Refactor the Logic into Reusable Helper Functions**
        - Analyze the monolithic code and identify the core, repeatable processes.
        - Group these steps into logical, well-documented helper functions with clear parameters. These functions are the "skills" the agent has learned (e.g., `login()`, `search_for_item(item_name: str)`, `add_to_cart()`).
        - Ensure these helper functions are generic and do not contain hardcoded values that should be parameters.

        **Part 2: Write an Intelligent `main_plan` Orchestrator**
        - Create a new `async def main_plan()` function.
        - Its purpose is to execute the user's immediate `generalization_request` by calling the helper functions you just created.
        - **CRITICAL STATE-AWARE LOGIC:**
            1.  **Analyze the Start State:** Look at the `action_log` to determine what the initial state of the *original* taught process was (e.g., it started on the homepage at "https://shop.example.com").
            2.  **Compare with Current State:** Compare that required start state with the `Browser's Current URL`. They will likely be different.
            3.  **Bridge the Gap:** Your `main_plan` must **bridge this state gap**. The very first step in your `main_plan` must be an `action_provider` call to get the browser from its current state to the necessary starting state for your helper functions. This is your "course correction" step.
            4.  **Execute the Goal:** After the state-setting step, `main_plan` should then call your helper functions in the correct order to fulfill the user's request.

        ---
        ### Example of the Expected Output

        **Scenario:**
        - **Taught Process:** The user guided the agent to go to an e-commerce site, search for "laptops", and add the first result to the cart. The plan ended on the product detail page for a specific laptop.
        - **Current URL:** `https://shop.example.com/products/laptop-xyz`
        - **Generalization Request:** "Great. Now do the same for 'keyboards'."

        **Your Correct Output (a single Python code block):**
        ```python
        # Part 1: The refactored helper functions (the "skills")
        @verify
        async def search_for_item(item_name: str):
            \"\"\"Searches for a given item on the site.\"\"\"
            # This skill assumes the browser is on the homepage to find the search bar.
            await action_provider.act(f"Type '{{item_name}}' into the search bar and press Enter")

        @verify
        async def add_first_item_to_cart():
            \"\"\"Clicks the 'Add to Cart' button for the first search result.\"\"\"
            await action_provider.act("Click the 'Add to Cart' button for the first item in the list")

        # Part 2: The intelligent `main_plan` orchestrator
        @verify
        async def main_plan():
            \"\"\"
            Orchestrates the process of searching for and adding 'keyboards' to the cart.
            It handles resetting the browser state as its first step.
            \"\"\"
            # CRITICAL: The agent is on a product page, but `search_for_item`
            # needs to be on the homepage. This is the state-bridging step.
            print("State correction: Navigating back to the homepage to start a new search.")
            await action_provider.navigate("https://shop.example.com/home")

            # Now, execute the generalized workflow.
            await search_for_item("keyboards")
            await add_first_item_to_cart()
            print("Successfully added keyboards to the cart.")

        ```

        {rules_and_examples}

        Begin your response now. Your response must start immediately with the JSON object.
        """,
    )


def build_precondition_prompt(
    function_source_code: str,
    interactions_log: str,
    has_entry_screenshot: bool,
) -> str:
    """
    Builds the prompt to determine the precondition for a function to run.

    Args:
        function_source_code: The source code of the function.
        interactions_log: A JSON string of the tool interactions during the function's run.
        has_entry_screenshot: Whether a screenshot of the browser is provided.
    """
    screenshot_section = ""
    if has_entry_screenshot:
        screenshot_section = textwrap.dedent(
            """
            ---
            ### CRITICAL: Visual Context (Entry Screenshot)
            You have been provided with a screenshot of the browser's state at the moment this function was called.
            - **Use this image as the primary source of truth** to determine the necessary starting conditions.
            - Analyze the image to describe the required visible elements (dialogs, buttons, forms, etc.).
            """,
        )

    return textwrap.dedent(
        f"""
        You are a state analysis expert for an autonomous web agent.
        A function that interacts with a web browser has just executed successfully. Your task is to describe the necessary **precondition** for this function to run correctly based on its first few actions and the visual state when it started.

        **Function Source Code:**
        ```python
        {function_source_code}
        ```

        **Execution Interaction Log:**
        ```json
        {interactions_log}
        ```

        {screenshot_section}

        **Your Task:**
        1.  Analyze the function's code, its interactions, and the entry screenshot.
        2.  If the first action is `navigate`, your primary goal is to populate the `precondition.url` field.
        3.  If the first action is `act` or `observe`, your primary goal is to populate the `precondition.description` field with a clear, verifiable description of the page state seen in the screenshot.
        4.  If the function does not interact with the browser at all, the status should be "not_applicable".

        Respond with ONLY the JSON object matching the `PreconditionDecision` schema.
        - `status`: "ok" if a precondition was identified, or "not_applicable" if none is needed.
        - `url`: If status is "ok", the URL of the page that must be present for the function to run. This is highly recommended.
        - `description`: If status is "ok", a description of the page state that must be present for the function to run. This is required if URL is not provided.
        """,
    )


def build_state_verification_prompt(
    precondition: Dict[str, Any],
) -> str:
    """
    Builds the prompt for an LLM to verify if the current browser state meets a required precondition.
    """
    precondition_str = json.dumps(precondition, indent=2)
    return textwrap.dedent(
        f"""
        You are a meticulous state verifier for an autonomous web agent. Your task is to determine if the current state of the web browser satisfies a function's required precondition.

        ---
        ### Analysis Task

        1.  **Required Precondition (The Goal State):**
            ```json
            {precondition_str}
            ```
        2.  **Visual Evidence:** A screenshot of the current browser page is provided. This is your primary source of truth.

        **Your Decision:**
        Compare the **Required Precondition** against the **Current Browser State**.
        - Does the current URL match the required URL (if specified)?
        - More importantly, does the visual content of the page (from the screenshot) match the required `description`?
        - For example, if the description is "The 'Admin Panel' dialog must be open", you must visually confirm that this dialog is present and visible in the screenshot.

        Respond with ONLY the JSON object matching the `StateVerificationDecision` schema.
        """,
    )


def build_proactive_correction_prompt(
    precondition: Dict[str, Any],
    current_url: str,
    *,
    tools: Dict[str, Any],
) -> str:
    """
    Builds the prompt for the LLM to generate a script to get from the current state to a target precondition state.
    """
    target_state_str = json.dumps(precondition, indent=2)
    scripting_rules = _build_simple_script_rules(tools)

    return textwrap.dedent(
        f"""
        You are a state recovery specialist for an autonomous web agent. The agent needs to achieve a specific browser state (a "precondition") before it can execute a function.

        Your task is to write a short Python script to bridge the gap between the current state and the target state.

        ---
        ### State Analysis

        **1. The "Current" State (Where you are now):**
        - **URL:** `{current_url}`
        - **Screenshot:** A screenshot of this current state is provided.

        **2. The "Target" Precondition (Where you need to be):**
        ```json
        {target_state_str}
        ```

        ---
        ### Your Task
        Write a `correction_code` snippet to get from the "Current" state to the "Target" state.
        - **Goal:** Your script should perform the necessary actions (e.g., clicking a button, filling a field, navigating) to satisfy the target precondition's `description`.
        - **Example:** If the current page is a dashboard and the target is "The 'Create New User' dialog must be open," your script should be `await action_provider.act("Click the 'Create New User' button to open the Create New User dialog")`.
        - **Keep it simple!** Only write the code needed to achieve the precondition.

        {scripting_rules}

        Respond with ONLY the JSON object matching the `CourseCorrectionDecision` schema. Set `correction_needed` to `true` if you write a script.
        """,
    )
