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


def _build_code_act_rules_and_examples(action_provider) -> str:
    """Builds the reusable block of core rules and examples for CodeActActor."""
    all_tools = {}

    browser_tools = {
        "browser_navigate": action_provider.browser_navigate,
        "browser_act": action_provider.browser_act,
        "browser_observe": action_provider.browser_observe,
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
           await action_provider.browser_navigate("[https://example.com](https://example.com)")
           result = await action_provider.browser_observe("What is the heading?")

           # ❌ WRONG: Missing await
           action_provider.browser_navigate("[https://example.com](https://example.com)")
           ```

        3. **Imports Inside Code**: All necessary imports must be included in the code you provide:
           ```python
           # ✅ CORRECT: Import inside the code execution
           from pydantic import BaseModel, Field
           from typing import Optional, List
           ```

        4. **Pydantic for Structured Observation**: When using `action_provider.browser_observe` to extract structured data:
           ```python
           from pydantic import BaseModel, Field

           class PageInfo(BaseModel):
               title: str = Field(description="Page title")
               products: list[str] = Field(description="List of product names")

           # CRITICAL: Call model_rebuild() after defining nested models
           PageInfo.model_rebuild()

           result = await action_provider.browser_observe(
               "Extract page information",
               response_format=PageInfo
           )
           ```

        5. **Error Handling**: If your code produces an error, the traceback will be returned. Read it carefully, correct your code, and try again.

        6. **Browser State Feedback**: After browser actions, you'll automatically receive:
           - The current URL
           - A screenshot of the page
           - Any output from your code

        7. **Communication Tools with Handles**: When using communication tools, they return handle objects for interaction:
           ```python
           # Send SMS and interact with handle
           sms_handle = await action_provider.send_sms_message(
               "Send John Doe a reminder about tomorrow's meeting"
           )

           # Check status
           status = await sms_handle.status()
           print(f"SMS Status: {status}")

           # Modify if needed
           await sms_handle.modify("Actually, make it for 3 PM instead")

           # Confirm sending
           await sms_handle.confirm()
           ```

        8. **Think → Code → Observe → Repeat**: Your workflow should be:
           - Think about what you need to do
           - Write code to execute the action
           - Observe the results (output, screenshots, errors)
           - Continue with the next step or correct errors

        9. **Final Answer Rule**:
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
                  "code": "await action_provider.browser_navigate('https://playwright.dev/')"
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
                  "code": "from pydantic import BaseModel, Field\n\nclass PageContent(BaseModel):\n    heading: str = Field(description=\"The main H1 heading of the page\")\n    first_paragraph: str = Field(description=\"The text of the first paragraph under the heading\")\n\nPageContent.model_rebuild()\n\npage_info = await action_provider.browser_observe(\n    \"Extract the main heading and the first paragraph.\",\n    response_format=PageContent\n)\n\nprint(page_info.model_dump_json(indent=2))"
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
                  "code": "await action_provider.browser_navigate('https://example.com/')"
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
                  "code": "from pydantic import BaseModel\n\nclass PageTitle(BaseModel):\n    title: str\n\n# MISTAKE: I forgot to call PageTitle.model_rebuild()\n\ntitle_info = await action_provider.browser_observe(\n    \"Extract the page title\",\n    response_format=PageTitle\n)\nprint(title_info)"
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
                  "code": "from pydantic import BaseModel\n\nclass PageTitle(BaseModel):\n    title: str\n\n# CORRECTED: I've added the required model_rebuild() call.\nPageTitle.model_rebuild()\n\ntitle_info = await action_provider.browser_observe(\n    \"Extract the page title\",\n    response_format=PageTitle\n)\nprint(title_info.title)"
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
                  "code": "import re\nfrom pydantic import BaseModel, Field\nfrom typing import List\n\ndef parse_price(price_str: str) -> float:\n    nums = re.findall(r'[\\d.]+', price_str)\n    return float(nums[0]) if nums else 0.0\n\nclass Product(BaseModel):\n    name: str\n    price_text: str = Field(alias=\"price\")\n\nclass ProductList(BaseModel):\n    products: List[Product]\n\nProductList.model_rebuild()\n\nglobal all_products_data\nall_products_data = await action_provider.browser_observe(\n    \"Extract all products with their name and price text\",\n    response_format=ProductList\n)\nprint(f\"Extracted {len(all_products_data.products)} products.\")"
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
        6.  **Isolate Pure Logic for Caching**: If your plan involves a complex calculation or a long data-processing loop that does not use the browser, factor it out into its own `async def` helper function. The actor automatically caches the results of successfully completed functions. By isolating this logic, you ensure it won't be re-executed if the plan restarts after a modification.
        7.  **Default Search Engine:** Prefer DuckDuckGo (https://duckduckgo.com) for searches unless the user specifies otherwise.
        8.  **Prefer .deb for Linux App Installations**: When installing apps, prefer .deb packages over other formats. `.deb` files are the most common and trusted format for Linux app installations. Then, use `dpkg` to install the app with full permissions.
        9.  **Desktop Mode**: You have control over a virtual desktop through the browser tools, and have access to apps like terminal, shell, browser, etc. You are also able to search for apps to be used in the desktop. Use screenshot to observe the desktop.
        ---
        """,
        # 10. **Browser Downloads**: The browser downloads directory is `/home/browser/Downloads`.
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
                await action_provider.browser_navigate("[https://shop.example.com](https://shop.example.com)")

            # ✅ STUB if uncertain (complex extractions, unknown layouts)
            async def extract_shipping_options():
                \"\"\"Extract available shipping options and prices.\"\"\"
                # Need to see the page structure first
                raise NotImplementedError("Extract shipping options from checkout page")
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
            action_provider.browser_navigate("[https://example.com](https://example.com)")

            # ✅ CORRECT: With await
            await action_provider.browser_navigate("[https://example.com](https://example.com)")
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
                result = await action_provider.browser_observe(
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
                await action_provider.browser_navigate("...")
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
            await action_provider.browser_navigate("[https://shop.example.com](https://shop.example.com)")
            await action_provider.browser_act(
                f"Type '{{product_name}}' in the search box and press Enter",
                expectation="Search results page should load with products"
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
                await action_provider.browser_act(
                    f"Set price filter from ${{min_price}} to ${{max_price}}",
                    expectation="Products should be filtered by price range"
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
            cart_info = await action_provider.browser_observe(
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
                await action_provider.browser_act(
                    "Click on 'Export Data' button and select JSON format",
                    expectation="Download should start or data should be displayed"
                )

                # RULE 9: Structured extraction
                class ExportedData(BaseModel):
                    customers: List[dict]
                    export_date: str
                    total_count: int

                ExportedData.model_rebuild()

                data = await action_provider.browser_observe(
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

                    table_data = await action_provider.browser_observe(
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
            await action_provider.browser_navigate("[https://crm.example.com](https://crm.example.com)")

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
            # from pydantic import BaseModel, Field
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
            from pydantic import BaseModel, Field
            print(f"Attempting to convert price: ${{product_price_usd}}")

            # --- Primary Approach: Use the website's feature ---
            try:
                await action_provider.browser_act(
                    "Click the currency selector and choose 'EUR'",
                    expectation="The price should now be displayed in Euros (€)."
                )

                class PriceInfo(BaseModel):
                    price_eur: float = Field(description="The price in Euros.")

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

            result = await action_provider.browser_observe(
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
            await action_provider.browser_navigate("https://example.com/sales-report")

            # The result of this step will be cached by the actor
            raw_data = await extract_sales_data_from_page()

            # If the plan is modified and restarts after this point,
            # this analysis function will NOT be re-run because its result
            # will be loaded from the cache, saving significant time.
            analysis_results = await perform_complex_analysis(raw_data)

            # Use the analysis results for further actions
            if analysis_results['best_product']:
                await action_provider.browser_act(
                    f"Search for more information about {{analysis_results['best_product']}}",
                    expectation="Product detail page should load"
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
        6.  **Isolate Pure Logic for Caching**: If your plan involves a complex calculation or a long data-processing loop that does not use the browser, factor it out into its own `async def` helper function. The actor automatically caches the results of successfully completed functions. By isolating this logic, you ensure it won't be re-executed if the plan restarts after a modification.
        ---
        """,
    )

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
            result = action_provider.browser_observe("Get data")

            # ✅ CORRECT: With await
            result = await action_provider.browser_observe("Get data")
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
                result = await action_provider.browser_observe(
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
                result = await action_provider.browser_navigate("https://example.com")
                data = await action_provider.browser_observe("Get page title")
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
            "\"\"\Sends an SMS reminder about an appointment.

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
            "\"\"\Makes an interactive phone call to confirm appointment details.

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
            "\"\"\Extracts all product information from a search results page.

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
                # Use browser_observe with structured output
                result = await action_provider.browser_observe(
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
            "\"\"\Completes the checkout process with payment information.

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
                await action_provider.browser_act(
                    f"Fill out the payment form with card ending in {{payment_info['card_number'][-4:]}}, "
                    f"CVV {{payment_info['cvv']}}, expiry {{payment_info['expiry']}}, and billing zip {{payment_info['zip']}}. "
                    f"Then click the 'Place Order' or 'Complete Purchase' button.",
                    expectation="Should see an order confirmation page with order number"
                )

                # Extract confirmation details
                confirmation = await action_provider.browser_observe(
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

                    await action_provider.browser_act(
                        "Click on 'PayPal' or 'Pay with PayPal' option",
                        expectation="Should redirect to PayPal login or show PayPal frame"
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
            "\"\"\Analyzes competitor pricing data for a specific product.

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
            pricing_data = await action_provider.browser_observe(
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
                await action_provider.browser_act(
                    f"Update our price to {{analysis_result['suggested_price']}}",
                    expectation="Price should be updated successfully"
                )

            return analysis_result

        # This helper function would be implemented separately in the plan
        # It contains pure logic with no browser interaction, making it cacheable
        async def _perform_pricing_analysis(pricing_data: dict) -> dict:
            "\"\"\Performs detailed statistical analysis on pricing data.

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

    strategy_instruction = (
        "Decompose the problem into logical `async def` functions. Each function should represent a complete, "
        "meaningful sub-task from a user's perspective (e.g., 'search_for_product_and_navigate_to_images' is better than "
        "having separate functions for typing, pressing enter, and clicking the images tab)."
    )
    tool_usage_instruction = "Use the `action_provider` global object to interact with the environment. Available tools and their handle APIs have been described in the rules below."

    rules_and_examples = _build_initial_plan_rules_and_examples(
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
    clarification_question: str | None,
    clarification_answer: str | None,
    has_browser_screenshot: bool,
    replan_context: str,
    *,
    tools: Dict[str, Callable],
    existing_code_for_modification: Optional[str] = None,
    recent_transcript: Optional[str] = None,
    parent_chat_context: Optional[list] = None,
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

    strategy_instruction = (
        "Your task is to analyze the situation and decide on the best course of action."
    )
    tool_usage_instruction = "Use the `action_provider` global object to interact with the environment. Available tools and their handle APIs have been described in the rules below."
    rules_and_examples = _build_dynamic_implement_rules_and_examples(
        tools,
        strategy_instruction,
        tool_usage_instruction,
    )

    return textwrap.dedent(
        f"""
        You are an expert Python programmer and a master strategist. Your task is to analyze the state of a running plan and decide the best course of action for the function `{function_name}`.

        **CRITICAL: You must choose one of three actions:**
        1.  **`implement_function`**: Write the Python code for `{function_name}`. Choose this if the function's goal is achievable from the current browser state. **Your code MUST be a single, self-contained `async def` function block. DO NOT include top-level imports or class definitions outside the function.** All necessary imports and helper classes MUST be defined *inside* the function.
        2.  **`skip_function`**: Bypass this function entirely. Choose this if you observe that the function's goal is **already completed** or is now **irrelevant**. For example, skip a "log in" function if you are already logged in.
        3.  **`replan_parent`**: Escalate the failure to the calling function. Choose this if the current function is **impossible to implement** because of a mistake made in a *previous* step. For example, if the goal is "apply filters" but the page has no filter controls, the error lies with the parent function that navigated to the wrong page or failed to get to the right state.
        4.  **`request_clarification`**: Ask the user for help. Choose this if you cannot devise a reliable strategy to fix the function from the available information. For example, if required UI elements are missing or behaving unexpectedly, or if there are multiple possible approaches and you're unsure which the user prefers. **You must provide a clear, specific `clarification_question`.**

        {modification_instructions}
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

    transcript_section = ""
    if recent_transcript:
        transcript_section = textwrap.dedent(
            f"""
        ---
        ### Recent Conversation Transcript
        The following is a summary of the most recent conversation turns, which may provide context for the function's execution.
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
        This is the broader conversation history that this plan is a part of.
        ```json
        {json.dumps(parent_chat_context, indent=2)}
        ```
        """,
        )

    return textwrap.dedent(
        f"""
        You are a meticulous verification agent. Your task is to assess if the executed actions successfully achieved the function's intended purpose and have made **meaningful and accurate progress** toward the **Overall User Goal**.

        **Overall User Goal:** "{goal}"
        **Function Under Review:** `{function_name}`
        **Purpose of this function (Intent):** {function_docstring or 'No docstring provided.'}

        {source_code_section}
        {screenshot_context_section}
        {transcript_section}
        {chat_context_section}

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
        - `request_clarification`: The function's goal is correct, but you need more information from the user to fix it. If you choose this, you MUST provide a clear, specific `clarification_question`.
        - `fatal_error`: An unrecoverable error occurred that prevents any further progress.

        **Your Response:**
        - status: Choose one of the valid status values above
        - reason: Provide a clear, concise explanation for your assessment
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
        You are an AI assistant in the middle of performing a task. The user has just asked a question.
        Based on the provided context, give a brief, natural, first-person response.
        Speak as if you are the one doing the work (e.g., "I'm currently looking for...").

        **Goal:** {goal}
        **State:** {state}
        **Call Stack:** {call_stack}
        **Current Browser View (Screenshot):**
        An image of the current browser page has been provided.
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


def build_interjection_prompt(
    interjection: str,
    parent_chat_context: list[dict] | None,
    plan_source_code: str,
    call_stack: list[str],
    action_log: list[str],
    is_teaching_session: bool,
) -> str:
    """
    Builds the system prompt for the Interjection Handler LLM.
    This prompt provides the LLM with full context of the plan and conversation,
    and instructs it to follow a specific decision tree to handle a user's
    interjection, outputting a structured JSON response.
    """
    call_stack_str = (
        " -> ".join(call_stack) if call_stack else "Not inside any function."
    )
    recent_actions = "\n".join(f"- {log}" for log in action_log) or "No actions yet."
    chat_history = (
        json.dumps(parent_chat_context, indent=2)
        if parent_chat_context
        else "No prior conversation."
    )

    teaching_session_rule = ""
    if is_teaching_session:
        teaching_session_rule = textwrap.dedent(
            """
        ---
        ### 📌 SPECIAL INSTRUCTIONS FOR THIS INTERJECTION

        **You are in a "Teaching Session".** The user is building a plan step-by-step.
        - If the user provides a new instruction to add to the plan (e.g., "now search for..."), you **MUST** choose the `modify_task` action.
        - If the user signals they are finished (e.g., "we're done", "that's all"), you **MUST** choose the `complete_task` action.
        ---
        """,
        )

    prompt = f"""
    You are an expert assistant responsible for steering a live-running automated plan.
    A user has interjected with a new instruction while a plan was executing.
    Your task is to analyze the user's intent and the current state of the plan, then decide on the correct course of action by following a strict decision tree.

    {teaching_session_rule}

    ### Full Situational Context

    **1. User's Interjection:**
    "{interjection}"

    **2. Full Conversation History (for semantic context):**
    ```json
    {chat_history}
    ```

    **3. Current Plan Source Code:**
    ```python
    {plan_source_code}
    ```

    **4. Current Execution Point (Call Stack):**
    `{call_stack_str}`

    **5. Most Recent Plan Actions:**
    {recent_actions}
    ---

    ### Your Task: Follow This Decision Tree Precisely

    **Question 1: Is the user signaling the end of the task or interactive session?**
    - Example: The user says, "That's it," "We're done," "End the session," or "Complete the task."
    - If YES, choose the `complete_task` action. This will allow the plan to execute its final version to completion.
    - If NO, proceed to Question 2.

    **Question 2: Is the user's request a fundamental change of goal that abandons or replaces the current task?**
    - Example: The plan is booking a flight, and the user says, "Actually, just look up recipes for me."
    - If YES, choose the `replace_task` action. For the `new_goal`, provide the user's new high-level objective.
    - If NO, proceed to Question 3.

    **Question 3: Is the user asking to generalize, repeat, or refactor the previously taught steps for a new subject or context?**
    - Example: After teaching a multi-step process for one item, the user says, "Now do the same for 'Sam Parker'," or "Great, now apply that to all other files in the folder."
    - If YES, choose the `refactor_and_generalize` action. For `generalization_context`, provide the new context (e.g., "Sam Parker", "all other files in the folder").
    - If NO, proceed to Question 4.

    **Question 4: Is the request a direct modification, a new step, or a correction for the *current* task?**
    - Example: The plan is researching on a website, and the user says, "No, use my LinkedIn profile for this research," or "Now, put those findings into a presentation."
    - If YES, choose the `modify_task` action. For `modification_request`, rephrase the user's instruction as a clear, actionable request for the actor to implement. For `target_function`, identify the most relevant function from the call stack to modify.
    - If NO, proceed to Question 5.

    **Question 5: Is the request a temporary, exploratory side-quest that doesn't alter the main goal?**
    - Example: The plan is creating a presentation, and the user says, "Quickly run the slide show so I can see how it looks."
    - If YES, choose the `explore_detached` action. For `new_goal`, provide the specific, temporary goal of the side-quest.
    - If NO, proceed to Question 6.

    **Question 6: Is the user's intent unclear, or does it require more information to proceed confidently?**
    - Example: The user says, "Make it better."
    - If YES, choose the `clarify` action. For `clarification_question`, formulate a concise, multiple-choice or open-ended question to ask the user.

    ---
    ### Output Format

    You MUST respond with a JSON object that strictly adheres to the `InterjectionDecision` Pydantic model. Do not add any other text or explanation.

    **Example `modify_task` response:**
    ```json
    {{
        "action": "modify_task",
        "reason": "The user wants to add a new step to the existing plan: creating a presentation from the research findings.",
        "modification_request": "After extracting the research findings, create a new Google Slides presentation and summarize the key points on the first slide.",
        "target_function": "main_plan"
    }}
    ```

    **Example `complete_task` response:**
    ```json
    {{
        "action": "complete_task",
        "reason": "The user has indicated that the teaching session is finished and the plan should now execute to completion."
    }}
    ```
    """
    return textwrap.dedent(prompt).strip()


def _build_simple_script_rules(tools: Dict[str, Callable]) -> str:
    """Builds a streamlined set of rules for simple, non-decomposed scripts."""
    tool_reference = _build_tool_signatures(tools)
    rules = textwrap.dedent(
        f"""
        ### 🎯 CRITICAL RULES FOR SCRIPTING
        1.  **Sequence of Calls**: Your code must be a simple sequence of `await` calls on the `action_provider`. Do not define new functions.
        2.  **Await Keyword**: You MUST `await` all `async` tool calls (like `browser_navigate`, `browser_act`, `browser_observe`).
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
        await action_provider.browser_navigate("https://example.com/user/123/profile")

        # ---
        # Example 2: The agent opened an unwanted "Share" popup that is now obscuring the page content.
        # Goal: Close the popup to restore view of the underlying page.
        await action_provider.browser_act("Click the 'X' or 'Close' button on the 'Share this article' popup")

        # ---
        # Example 3: The agent typed the wrong address into a form field.
        # Goal: Clear the incorrect text from the 'Street Address' field.
        # Note: The next implementation will handle typing the correct text. This script ONLY restores the state.
        await action_provider.browser_act("Clear the text in the 'Street Address' field")

    """,
    )
    return rules


def build_course_correction_prompt(
    last_verified_function_name: str,
    last_verified_url: str,
    current_url: str,
    failed_function_name: str,
    failed_function_docstring: str,
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

        **1. The "Last Known Good" State (BEFORE the failure):**
        This is the state after the function `{last_verified_function_name}` completed successfully.
        - **URL:** `{last_verified_url}`
        - **Screenshot:** A screenshot of this state is provided. (1st image)

        **2. The "Current / Corrupted" State (AFTER the failure):**
        This is the state where the function `{failed_function_name}` (Purpose: "{failed_function_docstring}") failed.
        - **URL:** `{current_url}`
        - **Screenshot:** A screenshot of this current state is also provided. (2nd image)

        ---
        ### Your Task

        1.  **Compare the two states.** Did the failed function navigate away from the correct page, open an unexpected modal, or otherwise alter the page structure in a way that prevents the *next* attempt from succeeding?
        2.  **Decide if correction is needed.**
            - If the states are the same or the changes are irrelevant, set `correction_needed` to `false`.
            - If the states are different and the browser needs to be returned to the "Last Known Good" state, set `correction_needed` to `true`.
        3.  **If correction is needed, write `correction_code`.**
            - This must be a simple, self-contained Python script.
            - Use `action_provider.browser_navigate` or `action_provider.browser_act`.
            - **Goal:** Get from the "Current" state back to the "Last Known Good" state.
            - **Example:** If the agent is on the wrong page, the script might be `await action_provider.browser_navigate('{last_verified_url}')`.
            - **Example:** If a popup is open, the script might be `await action_provider.browser_act("Click the 'Close' button on the popup")`.
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
    *,
    tools: Dict[str, Callable],
) -> str:
    """
    Builds the prompt for refactoring a monolithic plan into modular functions.

    Args:
        monolithic_code: The source code of the current single-function plan.
        generalization_request: The user's request to generalize the logic.
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
        You are an expert Python programmer specializing in code refactoring and generalization.
        Your task is to refactor the provided monolithic Python function into a set of smaller, logical, and reusable `async def` helper functions.

        **User's Generalization Request:**
        "{generalization_request}"

        **Current Monolithic Code to Refactor:**
        ```python
        {monolithic_code}
        ```

        **Your Task & Instructions:**
        1.  **Identify the Core Logic:** Analyze the user's request and the existing code to identify the central, repeated sequence of actions (e.g., the steps to process one item).
        2.  **Create a Parameterized Function:** Encapsulate this core logic within a new, parameterized helper function. For example, `async def process_item(item_name: str)`.
        3.  **Rewrite `main_plan`:** Rewrite the `main_plan` to be a clean coordinator. It should preserve the logic for the original subject that was taught but should now call your new helper functions, incorporating the user's generalization request.
        4.  **Follow All Rules:** Your final output must adhere to all the established rules for plan creation, including docstrings, async usage, and placing imports inside functions.

        {rules_and_examples}

        Begin your response now. Your response must be a single, complete Python code block containing the fully refactored script.
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
        2.  If the first action is `browser_navigate`, your primary goal is to populate the `precondition.url` field.
        3.  If the first action is `browser_act` or `browser_observe`, your primary goal is to populate the `precondition.description` field with a clear, verifiable description of the page state seen in the screenshot.
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
        - **Example:** If the current page is a dashboard and the target is "The 'Create New User' dialog must be open," your script should be `await action_provider.browser_act("Click the 'Create New User' button")`.
        - **Keep it simple!** Only write the code needed to achieve the precondition.

        {scripting_rules}

        Respond with ONLY the JSON object matching the `CourseCorrectionDecision` schema. Set `correction_needed` to `true` if you write a script.
        """,
    )
