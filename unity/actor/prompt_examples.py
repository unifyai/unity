"""
Centralized example library for Actor prompt builders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Example:
    """Container for a single example with metadata."""

    title: str
    code: str
    description: str
    environment_tags: List[str]  # e.g., ["computer"], ["primitives"], ["mixed"]


# ---------------------------------------------------------------------------
# 2. Core Patterns (Environment-Agnostic)
# ---------------------------------------------------------------------------


def get_confidence_based_stubbing_example() -> str:
    """Example: stub uncertain steps with clear TODO markers."""

    return """
# Example: Confidence-based stubbing
async def main_plan():
    # High confidence: direct implementation (tool-domain agnostic)
    records = await fetch_records()

    # Low confidence: stub with clear marker
    def send_email_to_contacts(_records: list) -> str:
        # TODO (LOW confidence): decide email service + template + recipients policy.
        raise NotImplementedError("Stub: email sending not yet implemented")

    # Continue with high-confidence steps
    result = send_email_to_contacts(records)
    return result
"""


def get_structured_output_example() -> str:
    """Example: use Pydantic for structured extraction."""

    return """
# Example: Structured output with Pydantic
from pydantic import BaseModel
from typing import Optional

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None

async def extract_structured(query: str) -> ContactInfo:
    # In practice: pass response_format=ContactInfo to a tool call and return the model.
    return ContactInfo(name="Alice Smith", email="alice.smith@example.com", phone="1112223333")
"""


def get_error_handling_example() -> str:
    """Example: graceful error handling with fallbacks."""

    return """
# Example: Error handling with fallbacks
async def find_with_fallback(query: str) -> str:
    try:
        # Primary: strict query
        out = await strict_lookup(query)
        if "not found" in str(out).lower():
            raise ValueError("Primary lookup failed")
        return str(out)
    except Exception as e:
        # Fallback: broader query
        out2 = await broad_lookup(query)
        return f"Fallback: {out2}"
"""


def get_handle_steering_example() -> str:
    """Example: steering in-flight handles."""

    return """
# Example: Handle steering (pause/resume/interject)
async def run_with_steering() -> str:
    # Start a long-running tool loop (returns steerable handle)
    handle = await start_long_running_work()

    # Pause execution to review progress
    await handle.pause()

    # Query current status
    status_handle = await handle.ask("What is the current progress?")
    status = await status_handle.result()

    # Inject guidance based on status
    await handle.interject("Prioritize data validation before proceeding.")

    # Resume execution
    await handle.resume()

    # Wait for completion
    result = await handle.result()
    return result
"""


def get_clarification_example() -> str:
    """Example: handling clarification requests from tools."""

    return """
# Example: Clarification requests
async def tool_call_that_may_clarify() -> str:
    # Some tool calls may request clarification mid-flight; the Actor/pane can:
    # - answer immediately if low-risk and known
    # - pause and ask the user for a disambiguating choice
    handle = await start_mutation_that_may_clarify()
    return await handle.result()
"""


def get_library_function_reuse_example() -> str:
    """Example: recognizing when to call an existing library function directly.

    Shows the pattern of checking library skills and calling them when they match the goal.
    """

    return """
# Example: Reusing an existing library function
async def main_plan():
    '''User goal: Schedule a weekly team meeting every Monday at 9am.

    ANALYSIS: The library has a skill 'schedule_recurring_task' that handles
    recurring task creation with specific timing. This matches our need exactly.
    '''

    # ✅ CORRECT: Call the existing skill directly
    result = await schedule_recurring_task(
        name="Team Meeting",
        description="Weekly team sync",
        frequency="weekly",
        weekday="MO",
        time="09:00"
    )
    return result

    # ❌ WRONG: Don't reimplement what already exists
    # task_handle = await primitives.tasks.update("Create weekly task...")
    # This would duplicate the skill's logic unnecessarily
"""


def get_library_function_composition_example() -> str:
    """Example: composing multiple library functions for a complex goal.

    Shows how to orchestrate existing skills when the goal requires multiple steps.
    """

    return """
# Example: Composing library functions for a complex workflow
async def main_plan():
    '''User goal: Find all contacts in New York and send them a meeting invite.

    ANALYSIS: The library has:
    - 'filter_contacts_by_location' for geographic filtering
    - 'send_bulk_notification' for batch messaging
    These can be composed to achieve the goal.
    '''

    # Step 1: Use existing skill to filter contacts
    ny_contacts = await filter_contacts_by_location(location="New York")

    # Step 2: Use existing skill to send notifications
    result = await send_bulk_notification(
        contact_list=ny_contacts,
        message="Team meeting tomorrow at 2pm",
        medium="email"
    )

    return f"Sent invites to {result['count']} contacts in New York"
"""


def get_library_function_adaptation_example() -> str:
    """Example: adapting an existing skill for a slightly different use case.

    Shows when to call a skill with different parameters vs reimplementing.
    """

    return """
# Example: Adapting a library function with parameters
async def main_plan():
    '''User goal: Get the phone number for the contact who works at Tesla.

    ANALYSIS: The library has 'find_contact_by_attribute' which can find contacts
    by any attribute. We can use it with employer="Tesla" to match our need.
    '''

    # ✅ CORRECT: Adapt the existing skill with appropriate parameters
    from pydantic import BaseModel

    class ContactResult(BaseModel):
        name: str
        phone: str | None
        employer: str | None

    ContactResult.model_rebuild()

    # The existing skill accepts flexible attribute filtering
    tesla_contact = await find_contact_by_attribute(
        attribute="employer",
        value="Tesla",
        response_format=ContactResult
    )

    return f"{tesla_contact.name}'s phone: {tesla_contact.phone or 'Not available'}"
"""


# ---------------------------------------------------------------------------
# 3. Computer Examples
# ---------------------------------------------------------------------------


def get_computer_navigation_example() -> str:
    """Example: navigate and extract data from a webpage."""

    return '''
# Example: Computer navigation and extraction
async def fetch_product_price(product_url: str) -> float:
    """Navigate to product page and extract price."""
    await computer_primitives.navigate(product_url)

    # Extract structured data using observe
    from pydantic import BaseModel

    class ProductInfo(BaseModel):
        name: str
        price: float
        in_stock: bool

    info = await computer_primitives.observe(
        "Extract product name, price, and stock status",
        response_format=ProductInfo
    )

    return info.price
'''


def get_computer_multistep_example() -> str:
    """Example: multi-step computer workflow with verification."""

    return '''
# Example: Multi-step computer workflow
async def complete_checkout(cart_items: list) -> str:
    """Complete e-commerce checkout flow."""
    # Navigate to checkout
    await computer_primitives.navigate("https://shop.example.com/checkout")

    # Fill shipping info
    await computer_primitives.act("Fill shipping address: 123 Main St, City, 12345")

    # Verify shipping info was entered correctly
    verification = await computer_primitives.observe(
        "Is the shipping address '123 Main St, City, 12345' displayed?"
    )
    if "no" in verification.lower():
        raise ValueError("Shipping address verification failed")

    # Complete payment
    await computer_primitives.act("Click 'Complete Order' button")

    # Confirm order placed
    confirmation = await computer_primitives.observe("Extract order confirmation number")
    return f"Order placed: {confirmation}"
'''


def get_computer_screenshot_driven_example() -> str:
    """Example: screenshot-driven implementation (within a computer loop).

    UI actions are guided by the current page state (captured as screenshots/evidence).
    """

    return """
# Example: Screenshot-driven implementation
async def proceed_using_screenshot() -> str:
    await computer_primitives.navigate("https://example.com/setup")

    # After computer actions, the loop surfaces a screenshot as an image block.
    # Prefer acting directly from that context, and only use observe for structured extraction
    # or when a precise, machine-checkable answer is required.

    await computer_primitives.act("Using the screenshot, click the 'Continue' button.")
    return await computer_primitives.observe("From the new screenshot, confirm we reached the next step.")
"""


def get_computer_session_execution_example() -> str:
    """Example: web navigation and structured data extraction using session-based execution."""

    return """
**Example: Web Navigation and Structured Data Extraction**

*User Request*: "What is the main heading and the text of the first paragraph on playwright.dev?"

*Turn 1: Navigate to the website*
* **Tool Call**:
    ```json
    {
      "tool_calls": [{
        "name": "execute_code",
        "arguments": {
          "thought": "The first step is to navigate to the website specified in the user\'s request, which is playwright.dev.",
          "code": "await computer_primitives.navigate(\'https://playwright.dev/\')",
          "language": "python",
          "state_mode": "stateful",
          "session_id": 0
        }
      }]
    }
    ```
* **Observation**:
    ```text
    --- COMPUTER STATE ---
    URL: https://playwright.dev/
    [A screenshot is available to you as an image block.]
    ```

*Turn 2: Observe the content using a Pydantic model*
* **Tool Call**:
    ```json
    {
      "tool_calls": [{
        "name": "execute_code",
        "arguments": {
          "thought": "Great, I\'m on the page. Now I\'ll extract the heading and paragraph text into a structured object for clarity. I\'ll define a Pydantic model right here in the sandbox.",
          "code": "from pydantic import BaseModel, Field\\n\\nclass PageContent(BaseModel):\\n    heading: str = Field(description=\\"The main H1 heading of the page\\")\\n    first_paragraph: str = Field(description=\\"The text of the first paragraph under the heading\\")\\n\\nPageContent.model_rebuild()\\n\\npage_info = await computer_primitives.observe(\\n    \\"Extract the main heading and the first paragraph.\\",\\n    response_format=PageContent\\n)\\n\\nprint(page_info.model_dump_json(indent=2))",
          "language": "python",
          "state_mode": "stateful",
          "session_id": 0
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
    --- COMPUTER STATE ---
    URL: https://playwright.dev/
    [A screenshot is available to you as an image block.]
    ```

*Turn 3: Provide the final answer*
* **Thought**: I have successfully extracted the information. I will now provide the final answer to the user without using any tools.
* **Final Answer (tool-less response)**:
    The main heading on playwright.dev is \'Playwright enables reliable end-to-end testing for modern web apps.\', and the first paragraph is \'Playwright is an open-source framework for web testing and automation. It allows testing Chromium, Firefox and WebKit with a single API.\'
"""


def get_computer_pydantic_example() -> str:
    """Example: using Pydantic models for structured observation in computer automation."""

    return """
**Example: Pydantic for Structured Observation (Computer)**

When using `computer_primitives.observe` to extract structured data, define a Pydantic model
inside the code, call `model_rebuild()`, and pass `response_format`:

```json
{
  "tool_calls": [{
    "name": "execute_code",
    "arguments": {
      "thought": "Extract structured page info using a Pydantic model.",
      "code": "from pydantic import BaseModel, Field\\n\\nclass PageContent(BaseModel):\\n    heading: str = Field(description=\\"The main H1 heading of the page\\")\\n    first_paragraph: str = Field(description=\\"The text of the first paragraph under the heading\\")\\n\\nPageContent.model_rebuild()\\n\\npage_info = await computer_primitives.observe(\\n    \\"Extract the main heading and the first paragraph.\\",\\n    response_format=PageContent\\n)\\n\\nprint(page_info.model_dump_json(indent=2))",
      "language": "python",
      "state_mode": "stateful",
      "session_id": 0
    }
  }]
}
```

**Key points:**
- Define the model inside the code block (imports must be inline)
- **CRITICAL**: Call `model_rebuild()` after defining nested models
- Pass `response_format=YourModel` to `observe()` for structured extraction
"""


def get_computer_error_correction_example() -> str:
    """Example: error handling and correction workflow in computer automation."""

    return """
**Example: Error Handling and Correction**

*User Request*: "Get the title from example.com."

*Turn 1: Navigate*
* **Tool Call**:
    ```json
    {
      "tool_calls": [{
        "name": "execute_code",
        "arguments": {
          "thought": "I need to navigate to example.com to get the title.",
          "code": "await computer_primitives.navigate(\'https://example.com/\')",
          "language": "python",
          "state_mode": "stateful",
          "session_id": 0
        }
      }]
    }
    ```
* **Observation**: Success, the computer environment is on example.com.

*Turn 2: Attempt to extract data with a mistake*
* **Tool Call**:
    ```json
    {
      "tool_calls": [{
        "name": "execute_code",
        "arguments": {
          "thought": "I need to get the title. I will use Pydantic, but I\'ll deliberately forget to call `model_rebuild()` to demonstrate error handling.",
          "code": "from pydantic import BaseModel\\n\\nclass PageTitle(BaseModel):\\n    title: str\\n\\n# MISTAKE: I forgot to call PageTitle.model_rebuild()\\n\\ntitle_info = await computer_primitives.observe(\\n    \\"Extract the page title\\",\\n    response_format=PageTitle\\n)\\nprint(title_info)",
          "language": "python",
          "state_mode": "stateful",
          "session_id": 0
        }
      }]
    }
    ```
* **Observation**:
    ```text
    --- ERROR ---
    Traceback (most recent call last):
      ...
    ValidationError: The response from the model did not conform to the expected Pydantic schema. [Reason: Could not find a registered Pydantic model named \'PageTitle\'. Did you forget to call `PageTitle.model_rebuild()`?]
    ```

*Turn 3: Correct the code based on the error*
* **Tool Call**:
    ```json
    {
      "tool_calls": [{
        "name": "execute_code",
        "arguments": {
          "thought": "The error message is very clear. I forgot to call `model_rebuild()` on my Pydantic class. I will add that line and retry.",
          "code": "from pydantic import BaseModel\\n\\nclass PageTitle(BaseModel):\\n    title: str\\n\\n# CORRECTED: I\'ve added the required model_rebuild() call.\\nPageTitle.model_rebuild()\\n\\ntitle_info = await computer_primitives.observe(\\n    \\"Extract the page title\\",\\n    response_format=PageTitle\\n)\\nprint(title_info.title)",
          "language": "python",
          "state_mode": "stateful",
          "session_id": 0
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
"""


def get_computer_stateful_workflow_example() -> str:
    """Example: stateful computation with helper functions persisting across turns."""

    return """
**Example: Stateful Computation and Helper Functions**

*User Request*: "On `fakestore.example.com`, find the average price of all products that cost less than $100."

*Turn 1: Define a helper function and extract all product data*
* **Tool Call**:
    ```json
    {
      "tool_calls": [{
        "name": "execute_code",
        "arguments": {
          "thought": "This is a multi-step task. First, I\'ll extract all products. I know I\'ll need to parse prices that might be strings (e.g., \'$25.99\'), so I\'ll define a helper function to clean them. This function will persist in the sandbox for later.",
          "code": "import re\\nfrom pydantic import BaseModel, Field\\nfrom typing import List\\n\\ndef parse_price(price_str: str) -> float:\\n    nums = re.findall(r\'[\\\\d.]+\', price_str)\\n    return float(nums[0]) if nums else 0.0\\n\\nclass Product(BaseModel):\\n    name: str\\n    price_text: str = Field(alias=\\"price\\")\\n\\nclass ProductList(BaseModel):\\n    products: List[Product]\\n\\nProductList.model_rebuild()\\n\\nglobal all_products_data\\nall_products_data = await computer_primitives.observe(\\n    \\"Extract all products with their name and price text\\",\\n    response_format=ProductList\\n)\\nprint(f\\"Extracted {len(all_products_data.products)} products.\\")",
          "language": "python",
          "state_mode": "stateful",
          "session_id": 0
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
        "name": "execute_code",
        "arguments": {
          "thought": "I have the product data in the `all_products_data` variable and my `parse_price` function is defined. Now I can perform the calculation in pure Python.",
          "code": "prices_under_100 = []\\nfor product in all_products_data.products:\\n    price = parse_price(product.price_text)\\n    if price < 100.0:\\n        prices_under_100.append(price)\\n\\nif prices_under_100:\\n    average = sum(prices_under_100) / len(prices_under_100)\\n    result_text = f\\"The average price of products under $100 is ${average:.2f}.\\"\\nelse:\\n    result_text = \\"No products found under $100.\\"\\n\\nprint(result_text)",
          "language": "python",
          "state_mode": "stateful",
          "session_id": 0
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
"""


def get_computer_interactive_workflow_example() -> str:
    """Example: interactive communication workflow with handle-based interaction."""

    return """
**Example: Interactive Communication Workflow**

*User Request*: "Text Jane Doe to confirm her appointment for tomorrow at 3 PM. Then, call her to ask if she has any dietary restrictions for the pre-appointment lunch."

*Turn 1: Send the confirmation SMS*
* **Tool Call**:
    ```json
    {
      "tool_calls": [{
        "name": "execute_code",
        "arguments": {
          "thought": "I\'ll start by sending the SMS. The `send_sms_message` tool returns a handle, which I\'ll await to ensure the message is sent and get a result.",
          "code": "sms_handle = await computer_primitives.send_sms_message(\\n    description=\\"Text Jane Doe to confirm her appointment for tomorrow at 3 PM.\\"\\n)\\n\\nsms_result = await sms_handle.result()\\nprint(sms_result)",
          "language": "python",
          "state_mode": "stateful",
          "session_id": 0
        }
      }]
    }
    ```
* **Observation**:
    ```text
    --- STDOUT ---
    Message successfully sent to Jane Doe (+1-555-123-4567): \'Hi Jane, this is a confirmation for your appointment tomorrow at 3 PM. Please reply to confirm.\'
    ```

*Turn 2: Initiate the interactive phone call*
* **Tool Call**:
    ```json
    {
      "tool_calls": [{
        "name": "execute_code",
        "arguments": {
          "thought": "The SMS is sent. Now I need to make the phone call. The `start_call` tool also returns a handle. I will store this handle in a global variable so I can interact with it in the next turn.",
          "code": "global active_call_handle\\nactive_call_handle = computer_primitives.start_call(\\n    phone_number=\\"Jane Doe\\",\\n    purpose=\\"Ask about dietary restrictions for a lunch meeting.\\"\\n)\\nprint(f\\"Initiated call to Jane Doe. Handle ID: {active_call_handle._loop_id}\\")",
          "language": "python",
          "state_mode": "stateful",
          "session_id": 0
        }
      }]
    }
    ```
* **Observation**:
    ```text
    --- STDOUT ---
    Initiated call to Jane Doe. Handle ID: a4b1
    --- RESULT ---
    <AsyncToolLoopHandle object ...>
    ```

*Turn 3: Interact with the live call using the handle*
* **Tool Call**:
    ```json
    {
      "tool_calls": [{
        "name": "execute_code",
        "arguments": {
          "thought": "The call is now active and the handle is stored in `active_call_handle`. I will use the handle\'s `.ask()` method to pose the question and get the answer.",
          "code": "ask_handle = await active_call_handle.ask(\\"Do you have any dietary restrictions for the lunch tomorrow?\\")\\n\\ndietary_info = await ask_handle.result()\\nprint(f\\"Received dietary info: {dietary_info}\\")\\n\\nawait active_call_handle.stop()\\nprint(\\"Call ended.\\")",
          "language": "python",
          "state_mode": "stateful",
          "session_id": 0
        }
      }]
    }
    ```
* **Observation**:
    ```text
    --- STDOUT ---
    Received dietary info: "Thanks for asking! I\'m vegetarian."
    Call ended.
    ```
* **Final Answer (tool-less)**: I\'ve confirmed Jane Doe\'s appointment via SMS. I also called her and she mentioned her dietary restriction is vegetarian.
"""


# ---------------------------------------------------------------------------
# 4. Primitives Examples (State Managers)
# ---------------------------------------------------------------------------


def get_primitives_contact_ask_example() -> str:
    """Example: read-only contact query."""

    return '''
# Example: Read-only contact query
async def find_contact_email(name: str) -> str:
    """Find a contact's email address by name."""
    # ContactManager.ask is read-only and returns a steerable handle
    handle = await primitives.contacts.ask(f"What is {name}'s email address?")

    # Wait for result
    answer = await handle.result()

    # Extract email from natural language answer
    # (In practice, use response_format for structured output)
    return answer
'''


def get_primitives_contact_update_example() -> str:
    """Example: contact mutation."""

    return '''
# Example: Contact mutation
async def update_contact_phone(email: str, phone: str) -> str:
    """Update a contact's phone number."""
    # ContactManager.update is a mutation and returns a steerable handle
    instruction = f"Update the contact with email {email}: set phone to {phone}"
    handle = await primitives.contacts.update(instruction)

    # Wait for completion
    result = await handle.result()
    return result
'''


def get_primitives_cross_manager_example() -> str:
    """Example: cross-manager workflow (KnowledgeManager → ContactManager)."""

    return '''
# Example: Cross-manager workflow
async def find_employee_count_for_contact(contact_name: str) -> int:
    """Find how many employees work at a contact's company.

    This demonstrates cross-manager integration:
    1. KnowledgeManager.ask internally calls ContactManager.ask to find employer
    2. KnowledgeManager.ask then queries its own tables for employee count
    """
    # Single high-level query; KM handles cross-manager coordination internally
    handle = await primitives.knowledge.ask(
        f"How many employees are at the company {contact_name} works at?"
    )

    answer = await handle.result()

    # Extract count from natural language answer
    # (In practice, use response_format for structured output)
    import re
    match = re.search(r'(\\d+)', answer)
    return int(match.group(1)) if match else 0
'''


def get_primitives_task_execute_example() -> str:
    """Example: task execution with steering."""

    return '''
# Example: Task execution with steering
async def execute_task_by_description_with_guidance(description: str) -> str:
    """Find and execute a task by description, with mid-flight guidance."""
    from pydantic import BaseModel

    class TaskIdResult(BaseModel):
        task_id: int
        task_name: str

    TaskIdResult.model_rebuild()

    # Step 1: Find the task_id using structured output from `ask(...)`.
    # (TaskScheduler.execute requires an integer task_id.)
    lookup_handle = await primitives.tasks.ask(
        f"Find the task that best matches: {description}. Return the task_id and name.",
        response_format=TaskIdResult,
    )
    task_info = await lookup_handle.result()

    # Step 2: Execute using the task_id (returns a steerable ActiveQueue handle).
    handle = await primitives.tasks.execute(task_id=task_info.task_id)

    # Inject guidance early in execution
    await handle.interject("Provide a progress update after each major step.")

    # Query status mid-execution
    status_handle = await handle.ask("What is the current status?")
    status = await status_handle.result()

    # Stop early if needed
    if "error" in status.lower():
        handle.stop(reason="Detected error in status")

    # Wait for completion
    result = await handle.result()
    return result
'''


def get_primitives_task_lookup_and_execute_example() -> str:
    """Example: Task lookup via ask(response_format=...) then execute(task_id=...).

    TaskScheduler.execute requires task_id: int.
    """

    return '''
# Example: Task lookup and execution pattern
from pydantic import BaseModel

class TaskIdResult(BaseModel):
    task_id: int
    task_name: str
    task_description: str

TaskIdResult.model_rebuild()

async def find_and_execute_task(search_query: str) -> str:
    """Locate a task by description and execute it."""
    # Step 1: Use ask with response_format to get structured task info
    lookup_handle = await primitives.tasks.ask(
        f"Find the task matching: {search_query}",
        response_format=TaskIdResult,
    )
    task = await lookup_handle.result()

    # Step 2: Execute the task using its task_id
    exec_handle = await primitives.tasks.execute(task_id=task.task_id)

    # Wait for completion
    result = await exec_handle.result()
    return f"Executed task '{task.task_name}': {result}"
'''


def get_primitives_dynamic_methods_example() -> str:
    """Example: using dynamic handle methods."""

    return '''
# Example: Dynamic handle methods (append_to_queue)
async def execute_task_and_append(task_a_id: int, task_b_id: int) -> str:
    """Execute a task and append another to its queue.

    TaskScheduler.execute returns an ActiveQueue handle that exposes
    a dynamic method: append_to_queue_<tool>_<id>(task_id=...)
    """
    # Start task A
    handle = await primitives.tasks.execute(task_id=task_a_id)

    # The handle exposes a dynamic append method
    # (exact name depends on handle instance; use introspection or ask)
    # For this example, assume we know the method name pattern

    # Append task B to the queue while A is running
    # Note: In practice, the LLM discovers this method via tool introspection
    await handle.append_to_queue(task_id=task_b_id)

    # Wait for completion (both tasks will execute in order)
    result = await handle.result()
    return result
'''


def get_primitives_files_describe_example() -> str:
    """Example: discovering file storage layout via `primitives.files.describe(...)`."""

    return '''
# Example: Discover file storage layout
async def discover_file_structure(file_path: str) -> dict:
    """Discover what tables/contexts are available in a file."""
    storage = await primitives.files.describe(file_path=file_path)

    # storage contains:
    # - indexed_exists: bool (whether file has been indexed)
    # - has_tables: bool (whether tables were extracted)
    # - tables: list of table info with context_path, name, description

    if storage.has_tables:
        for table in storage.tables:
            print(f"Table: {table.name}")
            print(f"  Context: {table.context_path}")
            print(f"  Description: {table.description}")
    return storage
'''


def get_primitives_files_reduce_example() -> str:
    """Example: aggregating data via `primitives.files.reduce(...)`."""

    return '''
# Example: Aggregate data from a file table
async def count_records_by_category(table_context: str) -> dict:
    """Count records grouped by a category column."""
    result = await primitives.files.reduce(
        context=table_context,
        metric="count",
        columns="id",
        group_by="category",
        filter="status == 'active'",
    )
    return result
'''


def get_primitives_files_filter_example() -> str:
    """Example: filtering rows via `primitives.files.filter_files(...)`."""

    return '''
# Example: Filter rows from a file table
async def get_recent_records(table_context: str) -> list:
    """Get recent records matching a filter."""
    rows = await primitives.files.filter_files(
        context=table_context,
        filter="created_date > '2024-01-01'",
        columns=["id", "name", "created_date", "status"],
        limit=50,
    )
    return rows
'''


def get_primitives_files_search_example() -> str:
    """Example: semantic search via `primitives.files.search_files(...)`."""

    return '''
# Example: Semantic search over file data
async def search_for_topic(table_context: str, query: str) -> list:
    """Search for records semantically matching a query."""
    hits = await primitives.files.search_files(
        context=table_context,
        query=query,
        columns=["description"],  # embedded column(s) to search
        limit=10,
    )
    return hits
'''


def get_primitives_files_visualize_example() -> str:
    """Example: generating charts via `primitives.files.visualize(...)`."""

    return '''
# Example: Generate a chart from file data
async def plot_distribution(table_context: str) -> str:
    """Generate a bar chart showing distribution by category."""
    result = await primitives.files.visualize(
        tables=table_context,
        plot_type="bar",
        x_axis="category",
        y_axis="amount",
        metric="sum",
        title="Amount by Category",
    )
    # result contains the plot URL
    return result.get("url") if isinstance(result, dict) else result
'''


def get_primitives_guidance_ask_example() -> str:
    """Example: read-only guidance query."""

    return '''
# Example: Read-only guidance query
async def get_incident_response_guidance() -> str:
    """Ask for incident response guidance."""
    handle = await primitives.guidance.ask("What guidance do you have for incident response?")
    return await handle.result()
'''


def get_primitives_guidance_update_example() -> str:
    """Example: guidance mutation via `primitives.guidance.update(...)`."""

    return '''
# Example: Guidance update (create/edit)
async def create_runbook_entry() -> str:
    """Create a new guidance entry (runbook)."""
    instruction = (
        "Create a new guidance entry titled 'Runbook: DB Failover'. "
        "Include step-by-step failover procedure, validation checks, and rollback steps."
    )
    handle = await primitives.guidance.update(instruction)
    return await handle.result()
'''


def get_primitives_web_ask_example() -> str:
    """Example: web research query via `primitives.web.ask(...)`."""

    return '''
# Example: Web research query
async def research_latest_news() -> str:
    """Ask the WebSearcher for time-sensitive info."""
    handle = await primitives.web.ask("What are the major world news headlines this week?")
    return await handle.result()
'''


# ---------------------------------------------------------------------------
# 5. Mixed Examples (Computer + Primitives)
# ---------------------------------------------------------------------------


def get_mixed_browse_persist_example() -> str:
    """Example: browse for data and persist via state managers."""

    return '''
# Example: Browse and persist workflow
async def scrape_and_save_contact(linkedin_url: str) -> str:
    """Scrape contact info from LinkedIn and save to ContactManager."""
    # Browse to profile
    await computer_primitives.navigate(linkedin_url)

    # Extract structured data
    from pydantic import BaseModel
    from typing import Optional

    class LinkedInProfile(BaseModel):
        name: str
        email: Optional[str] = None
        company: str

    profile = await computer_primitives.observe(
        "Extract name, email, and current company from profile",
        response_format=LinkedInProfile
    )

    # Persist to ContactManager
    instruction = f"Create contact: {profile.name}, email {profile.email}, employer {profile.company}"
    handle = await primitives.contacts.update(instruction)
    result = await handle.result()

    return f"Saved contact: {result}"
'''


def get_mixed_concurrent_example() -> str:
    """Example: concurrent computer and state manager operations."""

    return '''
# Example: Concurrent computer + state manager operations
import asyncio

async def gather_contact_info_concurrently(name: str, company_url: str) -> dict:
    """Gather contact info from multiple sources concurrently."""
    # Start both operations concurrently
    contact_handle = primitives.contacts.ask(f"Find {name}'s email and phone")

    # Navigate and extract company info in parallel
    async def fetch_company_info():
        await computer_primitives.navigate(company_url)
        return await computer_primitives.observe("Extract company size and industry")

    # Wait for both to complete
    contact_result, company_info = await asyncio.gather(
        contact_handle.result(),
        fetch_company_info()
    )

    return {
        "contact": contact_result,
        "company": company_info
    }
'''


def get_mixed_interjection_routing_example() -> str:
    """Example: routing interjections to in-flight handles."""

    return """
# Example: Interjection routing to in-flight handles
async def search_multiple_sources_with_correction(query: str) -> dict:
    # Start multiple concurrent searches
    contact_handle = primitives.contacts.ask(query)
    transcript_handle = primitives.transcripts.ask(query)

    # If user interjects with clarification (e.g., "I meant David Smith"),
    # the Actor's pane can broadcast the interjection to all in-flight handles
    # via: pane.broadcast_interject("User clarified: David Smith", filter=...)

    # Wait for results (interjections handled by pane)
    contact_result = await contact_handle.result()
    transcript_result = await transcript_handle.result()

    return {
        "contacts": contact_result,
        "transcripts": transcript_result
    }
"""


# ---------------------------------------------------------------------------
# 5b. Interjection Examples (Routing vs Patching)
# ---------------------------------------------------------------------------


def get_interjection_routing_only_examples() -> str:
    """Examples: routing-only interjections (no patches / no cache invalidation).

    These examples are intentionally written with *natural user language* (no "broadcast",
    no internal references like "main_plan"), and demonstrate the preferred output shape
    when the interjection can be satisfied by steering *already in-flight* handles.
    """

    return """
### Interjection Examples: Routing-Only (NO PATCHES)

**Rule of thumb:** If the user’s interjection can be satisfied by *steering what’s already running*
(tone/formatting/scope preferences for current outputs), prefer **routing-only**:
- Do **NOT** emit `patches`
- Do **NOT** emit `cache`
- Use `routing_action="broadcast_filtered"` (or `"targeted"` if the user singled out a specific handle)

#### Example A — Targeted tone adjustment (only the relevant handle)
- **Context**: There is an in-flight `primitives.contacts.ask(...)` that is just listing contacts, and an in-flight
  `primitives.transcripts.ask(...)` that is generating outreach tone/greeting guidance.
- **User**: “Actually—let’s make the outreach much more casual and friendly. Use ‘Hey <FirstName>,’ and don’t use last names or ‘Dear’.”
- **Correct JSON** (route only to the *relevant* in-flight handle):
```json
{
  "action": "modify_task",
  "reason": "This is a preference update for what is currently being produced by an in-flight handle. Routing is sufficient; patching would be unnecessarily disruptive.",
  "routing_action": "targeted",
  "target_handle_ids": ["<transcripts_handle_id_from_pane_snapshot>"],
  "routed_message": "Make the outreach much more casual and friendly. Use greeting 'Hey <FirstName>,' (first name only). Avoid last names and do not use 'Dear'."
}
```

#### Example D — Targeted refinement of an in-flight lookup (ranking / tie-breaker)
- **Context**: There is an in-flight `primitives.tasks.update(...)` creating a task, and an in-flight
  `primitives.contacts.ask(...)` identifying an assignee. The user adds a *selection preference* for the in-flight lookup.
- **User**: “For the Berlin contact lookup, prioritize anyone who has ‘manager’ or ‘lead’ in their role.”
- **Correct JSON** (route only; do NOT patch/restart):
```json
{
  "action": "modify_task",
  "reason": "This is a refinement for how the in-flight contact lookup should choose/format its answer. Steering the running contacts handle is sufficient; patching would unnecessarily cancel/restart execution.",
  "routing_action": "targeted",
  "target_handle_ids": ["<contacts_handle_id_from_pane_snapshot>"],
  "routed_message": "For the Berlin contact lookup: prioritize anyone whose role/title contains 'manager' or 'lead'. If multiple match, pick the most senior."
}
```

#### Example E — Upstream scope correction BEFORE downstream steps consume the result (staged workflow)
- **Context**: There is an in-flight `primitives.contacts.ask(...)` producing a contact set. The plan has *not yet awaited* that handle’s `.result()`,
  and downstream steps (e.g., transcripts/knowledge) will be spawned *based on* the returned contacts.
- **User**: “Actually, for this quarterly sync, let’s focus only on the Berlin office. When you list the active contacts, include only Berlin contacts.”
- **Correct JSON** (route only to the upstream handle; downstream will naturally use the corrected `contacts_result`):
```json
{
  "action": "modify_task",
  "reason": "The correction applies to an already in-flight upstream lookup whose result will feed downstream steps. Steering the in-flight contacts handle is sufficient; patching/restarting is unnecessarily disruptive.",
  "routing_action": "targeted",
  "target_handle_ids": ["<contacts_handle_id_from_pane_snapshot>"],
  "routed_message": "Scope update for the contact list currently in progress: include ONLY Berlin office contacts (names + office + email)."
}
```

#### Example B — Scope filter (“focus only on Q4”)
- **User**: “One change: for the summaries you’re producing right now, focus only on Q4 2024 and ignore Q3.”
- **Correct JSON**:
```json
{
  "action": "modify_task",
  "reason": "This is a scope constraint for the current in-flight summaries; route it to the running handles instead of patching plan code.",
  "routing_action": "broadcast_filtered",
  "broadcast_filter": {
    "capabilities": ["interjectable"],
    "origin_tool_prefixes": ["primitives.contacts", "primitives.transcripts", "primitives.knowledge"]
  },
  "routed_message": "For the summaries currently in progress: focus ONLY on Q4 2024 (Oct/Nov/Dec). Ignore Q3 entirely and do not mention it."
}
```

#### Example C — Conciseness (“keep it short”)
- **User**: “Keep everything you’re working on right now concise. Only the essentials—no quotes, no extra detail.”
- **Correct JSON**:
```json
{
  "action": "modify_task",
  "reason": "This is a formatting preference for outputs of in-flight handles; routing-only is the correct and least disruptive approach.",
  "routing_action": "broadcast_filtered",
  "broadcast_filter": { "capabilities": ["interjectable"] },
  "routed_message": "Be concise and to the point. Only include essential info. No quotes. No extra detail."
}
```

#### Anti-pattern (avoid circular “patch so replay is consistent”)
- **User**: “Keep the summaries you’re producing right now concise.”
- **Wrong reasoning**: “I should patch the plan prompts so that if we replay, it stays concise.”
- **Why wrong**: Adding `patches` triggers cancellation/restart and cache invalidation. If routing is sufficient,
  don’t force a restart “for replay consistency”.
""".strip()


# ---------------------------------------------------------------------------
# 6. Verification Examples (Prompt Builders)
# ---------------------------------------------------------------------------


def get_verification_ok_example() -> str:
    """Verification example: clear success (environment-agnostic).

    Shows how to map trace + evidence → decision.
    """
    return """
### Verification Example: ok (core)
- **Goal**: "Add Alice to my contacts."
- **Intent**: `create_contact(name='Alice', email='alice@corp.com')`
- **Agent Trace**:
  - `REASONING: I will create the contact and confirm the result.`
  - `tool_call: primitives.contacts.update(...)`
  - `✓ done`
- **Evidence**: Return value indicates success.
- **Decision**:
```json
{"status": "ok", "reason": "Trace shows correct action; return value confirms contact was created, so this step meaningfully advances the goal."}
```
""".strip()


def get_verification_ambiguous_example() -> str:
    """Verification example: ambiguity → request clarification (environment-agnostic)."""
    return """
### Verification Example: request_clarification (core)
- **Goal**: "Invite Bob to the calendar event."
- **Intent**: `invite_attendee(name='Bob')`
- **Agent Trace**:
  - `REASONING: I found two Bobs; I chose Bob S. and proceeded.`
  - `tool_call: primitives.contacts.ask(...)`
- **Evidence**: The trace reveals an unverified assumption (which Bob).
- **Decision**:
```json
{"status": "request_clarification", "reason": "The trace shows an ambiguous choice between two valid contacts; outcome cannot be verified as correct.", "clarification_question": "I found two contacts named Bob. Which one should I invite (e.g. Bob Smith or Bob Stone)?"}
```
""".strip()


def get_verification_tactical_failure_example() -> str:
    """Verification example: tactical failure → reimplement locally (environment-agnostic)."""
    return """
### Verification Example: reimplement_local (core)
- **Goal**: "Extract the support email from the page."
- **Intent**: `extract_support_email()`
- **Agent Trace**:
  - `REASONING: I see an email-like string; I will return it.`
  - `returned: 'support@example.com'`
- **Evidence**: Screenshot/visible text actually shows `help@company.com` as the support email.
- **Decision**:
```json
{"status": "reimplement_local", "reason": "The returned value does not match evidence. Root cause: extraction logic picked the wrong email string. Fix strategy: re-implement to search for the labeled 'Support' section and extract the email adjacent to it."}
```
""".strip()


def get_verification_strategic_failure_example() -> str:
    """Verification example: strategic failure → replan parent (environment-agnostic)."""
    return """
### Verification Example: replan_parent (core)
- **Goal**: "Apply a price filter under $500."
- **Intent**: `apply_price_filter(max_price=500)`
- **Agent Trace**:
  - `REASONING: I cannot find any filter controls on this page.`
  - `✓ done (no action possible)`
- **Evidence**: Screenshot shows a landing page without product list or filters.
- **Decision**:
```json
{"status": "replan_parent", "reason": "The function is impossible because the parent failed to navigate to a product listing page with filters. Fix strategy: update the parent to navigate into the catalog/results page before calling this function."}
```
""".strip()


def get_computer_verification_extraction_example() -> str:
    """Verification example: extraction (computer environment)."""
    return """
### Computer Verification Example: extraction
- **Goal**: "Find the product price."
- **Intent**: `extract_price()`
- **Agent Trace**:
  - `computer_primitives.observe('Extract price')`
  - `returned: '$199'`
- **Evidence**: Screenshot shows the price as `$299` (the `$199` is the crossed-out old price).
- **Decision**:
```json
{"status": "reimplement_local", "reason": "Tactical mismatch: trace executed, but evidence contradicts return value. Fix strategy: re-extract the *current* price (non-struck, highlighted) and verify against the screenshot."}
```
""".strip()


def get_computer_verification_multistep_example() -> str:
    """Verification example: multi-step success (computer environment)."""
    return """
### Computer Verification Example: multistep ok
- **Goal**: "Submit the signup form."
- **Intent**: `submit_signup(email='a@corp.com')`
- **Agent Trace**:
  - `computer_primitives.act('Type email…')`
  - `computer_primitives.act('Click Sign up')`
  - `✓ done`
- **Evidence**: Screenshot shows a confirmation banner 'Thanks for signing up'.
- **Decision**:
```json
{"status": "ok", "reason": "Trace shows correct actions and screenshot confirms successful submission, so the function advanced the overall goal."}
```
""".strip()


def get_primitives_verification_contact_update_example() -> str:
    """Verification example: primitives update success (primitives environment)."""
    return """
### Primitives Verification Example: contact update ok
- **Goal**: "Update Carol's phone number."
- **Intent**: `update_phone(email='carol@corp.com', phone='555-0101')`
- **Agent Trace**:
  - `primitives.contacts.update(...)`
  - `✓ done`
- **Evidence**: Return value indicates the contact was updated successfully.
- **Decision**:
```json
{"status": "ok", "reason": "The update call completed and the return value indicates success; this is sufficient evidence for a state-manager mutation."}
```
""".strip()


def get_primitives_verification_cross_manager_example() -> str:
    """Verification example: primitives cross-manager strategic failure (primitives environment)."""
    return """
### Primitives Verification Example: cross-manager replan_parent
- **Goal**: "Add the CEO of Acme to contacts."
- **Intent**: `create_ceo_contact(company='Acme')`
- **Agent Trace**:
  - `REASONING: I queried the company but found no CEO field.`
  - `primitives.knowledge.ask(...)`
  - `✓ done (insufficient data)`
- **Evidence**: Return value indicates missing CEO data; nothing to create.
- **Decision**:
```json
{"status": "replan_parent", "reason": "Strategic failure: the parent plan assumed CEO data existed. Fix strategy: replan to first find a reliable source for the CEO (or ask user) before calling contact creation."}
```
""".strip()


def get_mixed_verification_browse_persist_example() -> str:
    """Verification example: mixed browse + persist success (computer + primitives)."""
    return """
### Mixed Verification Example: browse + persist ok
- **Goal**: "Find support email on the site and save it to Knowledge."
- **Intent**: `scrape_support_email_and_save()`
- **Agent Trace**:
  - `computer_primitives.observe('Extract support email') -> 'help@company.com'`
  - `primitives.knowledge.update('Save support_email=help@company.com')`
  - `✓ done`
- **Evidence**: Screenshot shows the extracted email; update return value indicates persistence succeeded.
- **Decision**:
```json
{"status": "ok", "reason": "The computer evidence matches the extracted email, and the state-manager update confirms it was saved."}
```
""".strip()


def get_verification_examples_for_environments(
    has_computer: bool,
    has_primitives: bool,
) -> str:
    """Return verification examples appropriate for the given environment combination."""
    sections: list[str] = []

    # Core verification examples are always useful (environment-agnostic).
    sections.append(
        "### Core Verification Examples (Environment-Agnostic)\n"
        + "\n\n".join(
            [
                get_verification_ok_example().strip(),
                get_verification_ambiguous_example().strip(),
            ],
        ),
    )

    if has_computer:
        sections.append(
            "### Computer Verification Examples\n"
            + "\n\n".join(
                [
                    get_computer_verification_extraction_example().strip(),
                    get_computer_verification_multistep_example().strip(),
                ],
            ),
        )

    if has_primitives:
        sections.append(
            "### Primitives Verification Examples\n"
            + "\n\n".join(
                [
                    get_primitives_verification_contact_update_example().strip(),
                    get_primitives_verification_cross_manager_example().strip(),
                ],
            ),
        )

    if has_computer and has_primitives:
        sections.append(
            "### Mixed Verification Examples\n"
            + get_mixed_verification_browse_persist_example().strip(),
        )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# 7. Example Function Map (for ToolSurfaceRegistry)
# ---------------------------------------------------------------------------


def get_example_function_map() -> dict[str, callable]:
    """Get a mapping of example function names to their callables.

    This is used by ToolSurfaceRegistry.prompt_examples() to generate
    examples for exposed managers.
    """
    return {
        # Contacts
        "get_primitives_contact_ask_example": get_primitives_contact_ask_example,
        "get_primitives_contact_update_example": get_primitives_contact_update_example,
        # Tasks
        "get_primitives_task_execute_example": get_primitives_task_execute_example,
        "get_primitives_task_lookup_and_execute_example": get_primitives_task_lookup_and_execute_example,
        "get_primitives_dynamic_methods_example": get_primitives_dynamic_methods_example,
        # Knowledge
        "get_primitives_knowledge_ask_example": get_primitives_cross_manager_example,
        "get_primitives_knowledge_update_example": lambda: "",  # placeholder
        # Transcripts
        "get_primitives_transcript_ask_example": lambda: "",  # placeholder
        # Files (using real FileManager primitives)
        "get_primitives_files_describe_example": get_primitives_files_describe_example,
        "get_primitives_files_reduce_example": get_primitives_files_reduce_example,
        "get_primitives_files_filter_example": get_primitives_files_filter_example,
        "get_primitives_files_search_example": get_primitives_files_search_example,
        "get_primitives_files_visualize_example": get_primitives_files_visualize_example,
        # Guidance
        "get_primitives_guidance_ask_example": get_primitives_guidance_ask_example,
        "get_primitives_guidance_update_example": get_primitives_guidance_update_example,
        # Web
        "get_primitives_web_ask_example": get_primitives_web_ask_example,
        # Secrets
        "get_primitives_secrets_ask_example": lambda: "",  # placeholder
        "get_primitives_secrets_update_example": lambda: "",  # placeholder
        # Data
        "get_primitives_data_filter_example": lambda: "",  # placeholder
        "get_primitives_data_reduce_example": lambda: "",  # placeholder
    }


# ---------------------------------------------------------------------------
# 8. Function-First Pattern Examples (for CodeActActor)
# ---------------------------------------------------------------------------


def get_function_first_pattern_example() -> str:
    """Example: prioritizing pre-saved functions via FunctionManager tools (CodeAct style)."""

    return r"""
# ✅ PATTERN: Function-First Workflow (CodeActActor)
# If FunctionManager tools are available, ALWAYS search for an existing function
# BEFORE writing custom logic with raw primitives.
#
# Step 1 (JSON TOOL CALL): search for an existing function
# {
#   "name": "FunctionManager_search_functions",
#   "arguments": {"query": "contacts prefer phone", "n": 5}
# }
#
# Step 2 (JSON TOOL CALL): execute with state_mode="stateful" (REQUIRED!)
# {
#   "name": "execute_code",
#   "arguments": {
#     "language": "python",
#     "state_mode": "stateful",
#     "code": "result = await ask_contacts_question('Which of our contacts prefers phone contact?')\nprint(result)"
#   }
# }
#
# IMPORTANT: You MUST use state_mode="stateful" because functions are injected into Session 0.
# Using stateless creates a fresh session where the function is NOT available!
#
# If no function exists, THEN fall back to composing with primitives directly in Python.
"""


def get_function_first_anti_pattern_example() -> str:
    """Anti-pattern: skipping FunctionManager search when it exists (CodeAct style)."""

    return r"""
# ❌ ANTI-PATTERN #1: Skipping FunctionManager when it's available
#
# DON'T do this:
#   - immediately call raw primitives
#   - re-implement logic that likely exists as a stored function
#
# Example (bad):
#   handle = await primitives.contacts.ask("Which contacts prefer phone?")
#   result = await handle.result()
#
# ❌ ANTI-PATTERN #2: Using stateless mode after FunctionManager search
#
# DON'T do this:
# {
#   "name": "execute_code",
#   "arguments": {
#     "language": "python",
#     "state_mode": "stateless",
#     "code": "result = await ask_contacts_question(...)"
#   }
# }
# ERROR: NameError - function not available in fresh session!
#
# ✅ CORRECT:
#   1) Call FunctionManager_search_functions(...) as a JSON tool call
#   2) Call execute_code with state_mode="stateful" and invoke the injected function
"""


def get_function_parameter_exploration_example() -> str:
    """Example: reading function metadata before calling (CodeAct style)."""

    return r"""
# ✅ PATTERN: Read function signatures before calling
#
# FunctionManager tool results include:
# - name
# - argspec
# - docstring
#
# Use that to pick the right parameters before calling the injected function in Python.
#
# Example:
# 1) JSON tool call:
#    FunctionManager_search_functions(query="update guidance runbook", n=5)
#
# 2) Inspect returned `argspec`/docstring (mentally), then in Python:
#    result = await update_guidance("Create a runbook titled 'Runbook: DB Failover' ...")
#    print(result)
"""


def get_function_manager_stateful_requirement_example() -> str:
    """Example: FunctionManager functions require stateful sessions (CRITICAL)."""

    return r"""
# 🚨 CRITICAL PATTERN: FunctionManager + Stateful Sessions
#
# Functions from FunctionManager are injected into Session 0's namespace.
# You MUST use state_mode="stateful" in execute_code to access them.
#
# WHY: stateless mode creates a FRESH session each time, so injected functions
# are NOT available in that new session → NameError.
#
# ✅ CORRECT WORKFLOW:
#
# Step 1 (JSON TOOL CALL): Search for function (injects into Session 0)
# {
#   "name": "FunctionManager_search_functions",
#   "arguments": {"query": "store knowledge", "n": 5}
# }
# Returns: [{"name": "store_knowledge", "argspec": "(fact: str) -> str", ...}]
#
# Step 2 (JSON TOOL CALL): Execute with state_mode="stateful" (REQUIRED!)
# {
#   "name": "execute_code",
#   "arguments": {
#     "language": "python",
#     "state_mode": "stateful",
#     "code": "result = await store_knowledge('Office hours are 9-5 PT')\nprint(result)"
#   }
# }
# ✅ Works! Function is available in Session 0.
#
# ❌ ANTI-PATTERN (causes NameError):
#
# {
#   "name": "execute_code",
#   "arguments": {
#     "language": "python",
#     "state_mode": "stateless",
#     "code": "result = await store_knowledge('Office hours are 9-5 PT')"
#   }
# }
# ❌ ERROR: NameError: name 'store_knowledge' is not defined
# WHY: stateless creates fresh session where function was NOT injected!
#
# Mental Model:
#
#   FunctionManager_search_functions(...)
#            ↓
#      Injects into Session 0 namespace
#            ↓
#   execute_code(state_mode="stateful", ...)
#            ↓
#      ✅ Function available!
#
#   execute_code(state_mode="stateless", ...)
#            ↓
#      Creates NEW session (not Session 0)
#            ↓
#      ❌ Function NOT available! (NameError)
"""


def get_code_act_function_first_examples() -> str:
    """Get function-first examples for CodeActActor."""
    examples = [
        # Put the critical stateful requirement example FIRST for maximum visibility
        get_function_manager_stateful_requirement_example().strip(),
        get_function_first_pattern_example().strip(),
        get_function_first_anti_pattern_example().strip(),
        get_function_parameter_exploration_example().strip(),
    ]
    return "\n\n".join(examples)


# ---------------------------------------------------------------------------
# 8b. Multi-language + multi-session execution examples (for CodeActActor)
# ---------------------------------------------------------------------------


def get_code_act_session_examples() -> str:
    """Examples: using execute_code with sessions across Python + shell.

    These examples are written in JSON tool-call form (not Python), because
    `execute_code` is itself a tool call.
    """

    return r"""
### Multi-Language + Multi-Session Execution (CodeActActor)

**Key idea:** Use `execute_code` for *everything* (Python + shell), and use sessions
to preserve state across multiple tool calls.

**⚠️ CRITICAL: FunctionManager Functions Require Stateful Sessions**

When using FunctionManager tools, functions are injected into Session 0. You MUST use
`state_mode="stateful"` to access them:

```
FunctionManager_search_functions(...)
         ↓
   Injects into Session 0 namespace
         ↓
execute_code(state_mode="stateful", ...)   ←── ✅ Uses Session 0, function available!

execute_code(state_mode="stateless", ...)  ←── ❌ Creates NEW session, NameError!
```

#### Example A — Stateful shell session for repo navigation
```json
{
  "tool_calls": [{
    "name": "execute_code",
    "arguments": {
      "thought": "Create or reuse a persistent bash session for navigating the repo.",
      "language": "bash",
      "state_mode": "stateful",
      "session_name": "repo_nav",
      "code": "pwd && ls"
    }
  }]
}
```

#### Example B — Continue the same shell session (cd persists)
```json
{
  "tool_calls": [{
    "name": "execute_code",
    "arguments": {
      "thought": "Continue in the same session so cwd is preserved.",
      "language": "bash",
      "state_mode": "stateful",
      "session_name": "repo_nav",
      "code": "cd unity && ls && git status -sb"
    }
  }]
}
```

#### Example C — Stateful Python session for iterative work
```json
{
  "tool_calls": [{
    "name": "execute_code",
    "arguments": {
      "thought": "Use a stateful Python session so variables persist between calls.",
      "language": "python",
      "state_mode": "stateful",
      "session_name": "analysis",
      "code": "x = 41\nx += 1\nprint(x)"
    }
  }]
}
```

#### Example D — Session discovery and inspection
```json
{
  "tool_calls": [
    { "name": "list_sessions", "arguments": {} },
    {
      "name": "inspect_state",
      "arguments": {
        "detail": "names",
        "session_name": "analysis",
        "language": "python"
      }
    }
  ]
}
```

#### Example E — Stateless execution (no persistence)
```json
{
  "tool_calls": [{
    "name": "execute_code",
    "arguments": {
      "thought": "Run a one-off command with no session/persistence.",
      "language": "bash",
      "state_mode": "stateless",
      "code": "echo hello"
    }
  }]
}
```
""".strip()


# ---------------------------------------------------------------------------
# 9. Helper Functions for Prompt Composition
# ---------------------------------------------------------------------------


def get_core_pattern_examples() -> str:
    """Get all core pattern examples (environment-agnostic)."""

    examples = [
        get_library_function_reuse_example().strip(),
        get_library_function_composition_example().strip(),
        get_library_function_adaptation_example().strip(),
        get_confidence_based_stubbing_example().strip(),
        get_structured_output_example().strip(),
        get_error_handling_example().strip(),
        get_handle_steering_example().strip(),
        get_clarification_example().strip(),
    ]
    return "\n\n".join(examples)


def get_code_act_pattern_examples() -> str:
    """Get core pattern examples relevant to CodeActActor.

    Includes error handling, clarification patterns, and function-first workflow
    that complement the primitives examples.
    """

    examples = [
        get_error_handling_example().strip(),
        get_clarification_example().strip(),
    ]
    return "\n\n".join(examples)


def get_computer_examples() -> str:
    """Get all computer-specific examples."""

    examples = [
        get_computer_navigation_example().strip(),
        get_computer_multistep_example().strip(),
        get_computer_screenshot_driven_example().strip(),
        get_computer_session_execution_example().strip(),
        get_computer_pydantic_example().strip(),
        get_computer_error_correction_example().strip(),
        get_computer_stateful_workflow_example().strip(),
        get_computer_interactive_workflow_example().strip(),
    ]
    return "\n\n".join(examples)


def get_primitives_examples(*, managers: set[str] | None = None) -> str:
    """Get state manager examples, optionally filtered by manager.

    DEPRECATED: Use ToolSurfaceRegistry.prompt_examples(scope) instead.

    Args:
        managers: If provided, only include examples for these managers.
                  If None, include all examples.

    Returns:
        Formatted string with relevant examples.
    """
    from unity.function_manager.primitives import (
        PrimitiveScope,
        VALID_MANAGER_ALIASES,
        get_registry,
    )

    if managers is None:
        scope = PrimitiveScope.all_managers()
    else:
        # Filter to valid aliases only
        valid = managers & VALID_MANAGER_ALIASES
        if not valid:
            return ""
        scope = PrimitiveScope(scoped_managers=frozenset(valid))

    return get_registry().prompt_examples(scope)


def get_mixed_examples() -> str:
    """Get all mixed-mode examples."""

    examples = [
        get_mixed_browse_persist_example().strip(),
        get_mixed_concurrent_example().strip(),
        get_mixed_interjection_routing_example().strip(),
    ]
    return "\n\n".join(examples)


def get_examples_for_environments(
    has_computer: bool,
    has_primitives: bool,
    include_core: bool = True,
    *,
    managers: set[str] | None = None,
) -> str:
    """Get examples appropriate for the given environment combination.

    Args:
        has_computer: Whether computer_primitives environment is active
        has_primitives: Whether primitives environment is active
        include_core: Whether to include core patterns (default: True)
        managers: If provided, only include examples for these managers

    Returns:
        Formatted string with relevant examples
    """

    sections: list[str] = []

    if include_core:
        sections.append(
            "### Core Patterns (Environment-Agnostic)\n" + get_core_pattern_examples(),
        )

    if has_computer:
        sections.append("### Computer Examples\n" + get_computer_examples())

    if has_primitives:
        sections.append(
            "### State Manager Examples\n" + get_primitives_examples(managers=managers),
        )

    if has_computer and has_primitives:
        sections.append("### Mixed-Mode Examples\n" + get_mixed_examples())

    return "\n\n".join(sections)
