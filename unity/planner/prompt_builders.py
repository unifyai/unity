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
                await action_provider.browser_navigate("https://shop.example.com")

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
            action_provider.browser_navigate("https://example.com")

            # ✅ CORRECT: With await
            await action_provider.browser_navigate("https://example.com")
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
            await action_provider.browser_navigate("https://shop.example.com")
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
                    # RULE 10: Re-raise to let planner handle
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
            await action_provider.browser_navigate("https://crm.example.com")

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
    Dynamically builds the system prompt for the Hierarchical Planner.
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
    rules_and_examples = _build_initial_plan_rules_and_examples(
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


def build_interjection_prompt(
    interjection: str,
    parent_chat_context: list[dict] | None,
    plan_source_code: str,
    call_stack: list[str],
    action_log: list[str],
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

    prompt = f"""
    You are an expert assistant responsible for steering a live-running automated plan.
    A user has interjected with a new instruction while a plan was executing.
    Your task is to analyze the user's intent and the current state of the plan, then decide on the correct course of action by following a strict decision tree.

    ---
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

    **Question 1: Is the user's request a fundamental change of goal that abandons or replaces the current task?**
    - Example: The plan is booking a flight, and the user says, "Actually, just look up recipes for me."
    - If YES, choose the `replace_task` action. For the `new_goal`, provide the user's new high-level objective.
    - If NO, proceed to Question 2.

    **Question 2: Is the request a direct modification, a new step, or a correction for the *current* task?**
    - Example: The plan is researching on a website, and the user says, "No, use my LinkedIn profile for this research," or "Now, put those findings into a presentation."
    - If YES, choose the `modify_task` action. For `modification_request`, rephrase the user's instruction as a clear, actionable request for the planner to implement. For `target_function`, identify the most relevant function from the call stack to modify.
    - If NO, proceed to Question 3.

    **Question 3: Is the request a temporary, exploratory side-quest that doesn't alter the main goal?**
    - Example: The plan is creating a presentation, and the user says, "Quickly run the slide show so I can see how it looks."
    - If YES, choose the `explore_detached` action. For `new_goal`, provide the specific, temporary goal of the side-quest.
    - If NO, proceed to Question 4.

    **Question 4: Is the user's intent unclear, or does it require more information to proceed confidently?**
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
    last_verified_page_analysis,
    has_last_verified_screenshot: bool,
    current_url: str,
    current_page_analysis,
    has_current_screenshot: bool,
    failed_function_name: str,
    failed_function_docstring: str,
    *,
    tools: Dict[str, Callable],
) -> str:
    """
    Builds the prompt for the course correction LLM.
    """
    last_page_analysis_str = last_verified_page_analysis.model_dump_json(indent=2)
    current_page_analysis_str = current_page_analysis.model_dump_json(indent=2)

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
        - **Page Analysis:**
          ```json
          {last_page_analysis_str}
          ```
        - **Screenshot:** A screenshot of this state is provided.

        **2. The "Current / Corrupted" State (AFTER the failure):**
        This is the state where the function `{failed_function_name}` (Purpose: "{failed_function_docstring}") failed.
        - **URL:** `{current_url}`
        - **Page Analysis:**
          ```json
          {current_page_analysis_str}
          ```
        - **Screenshot:** A screenshot of this current state is also provided.

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
