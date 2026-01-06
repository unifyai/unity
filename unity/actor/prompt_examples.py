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
    environment_tags: List[str]  # e.g., ["browser"], ["primitives"], ["mixed"]


# ---------------------------------------------------------------------------
# 2. Core Patterns (Environment-Agnostic)
# ---------------------------------------------------------------------------


def get_confidence_based_stubbing_example() -> str:
    """Example: stub uncertain steps with clear TODO markers.

    Source: `unity/actor/prompt_builders.py` (planning rules/examples),
    plus the same stub→JIT-implement pattern used in
    `tests/test_conductor/test_real/test_actor.py`.
    """

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
    """Example: use Pydantic for structured extraction.

    Source: `tests/test_contact_manager/test_ask.py` (ask scenarios + structured judgement)
    """

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
    """Example: graceful error handling with fallbacks.

    Source: `tests/test_contact_manager/test_ask.py` (semantic retrieval + negative cases)
    """

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
    """Example: steering in-flight handles.

    Source: `tests/test_task_scheduler/test_execute.py` (execute handle forwards ask/interject/pause/resume)
    """

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
    """Example: handling clarification requests from tools.

    Source: `tests/test_contact_manager/test_ask.py` (clarification queues),
    and `tests/test_task_scheduler/test_execute.py` (steering surface on returned handles).
    """

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
# 3. Browser Examples
# ---------------------------------------------------------------------------


def get_browser_navigation_example() -> str:
    """Example: navigate and extract data from a webpage.

    Source: `tests/test_conductor/test_real/test_actor.py` (real actor uses navigate/observe/act)
    """

    return '''
# Example: Browser navigation and extraction
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


def get_browser_multistep_example() -> str:
    """Example: multi-step browser workflow with verification.

    Source: `tests/test_conductor/test_real/test_actor.py` (navigate/act/observe + verification loop)
    """

    return '''
# Example: Multi-step browser workflow
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


def get_browser_screenshot_driven_example() -> str:
    """Example: screenshot-driven implementation (within a browser loop).

    Source: `tests/test_conductor/test_real/test_actor.py` (real actor uses navigate/act/observe),
    where UI actions are guided by the current page state (captured as screenshots/evidence).
    """

    return """
# Example: Screenshot-driven implementation
async def proceed_using_screenshot() -> str:
    await computer_primitives.navigate("https://example.com/setup")

    # The Actor captures the page screenshot as evidence; use observe to read UI state from it.
    visible = await computer_primitives.observe("From the screenshot, is there a 'Continue' button visible?")
    if "no" in str(visible).lower():
        raise ValueError("Expected a 'Continue' button in the screenshot")

    await computer_primitives.act("Using the screenshot, click the 'Continue' button.")
    return await computer_primitives.observe("From the new screenshot, confirm we reached the next step.")
"""


# ---------------------------------------------------------------------------
# 4. Primitives Examples (State Managers)
# ---------------------------------------------------------------------------


def get_primitives_contact_ask_example() -> str:
    """Example: read-only contact query.

    Source: `tests/test_conductor/test_real/test_contacts.py::test_ask_calls_manager`,
    `tests/test_contact_manager/test_ask.py` (semantic ask cases).
    """

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
    """Example: contact mutation.

    Source: `tests/test_conductor/test_real/test_contacts.py::test_update_calls_manager`
    """

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
    """Example: cross-manager workflow (KnowledgeManager → ContactManager).

    Source: `tests/test_knowledge_manager/test_cross_manager_integration.py::test_ask_joins_contact_and_company`
    """

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
    """Example: task execution with steering.

    Source: `tests/test_task_scheduler/test_execute.py` (execute + handle forwarding)
    """

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

    Source: Correct pattern for TaskScheduler.execute which requires task_id: int
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
    """Example: using dynamic handle methods.

    Source: `tests/test_task_scheduler/test_execute.py` (introspection + append_to_queue exposure)
    """

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


def get_primitives_files_ask_example() -> str:
    """Example: read-only file inventory query.

    Shows how to use the File manager via `primitives.files.ask(...)`.
    """

    return '''
# Example: Read-only files query (inventory)
async def list_available_filesystems() -> str:
    """List available filesystems/roots and summarize what is available."""
    handle = await primitives.files.ask(
        "List available filesystems/roots and provide a brief inventory overview."
    )
    return await handle.result()
'''


def get_primitives_files_organize_example() -> str:
    """Example: file organization (rename/move/delete) via `primitives.files.organize(...)`."""

    return '''
# Example: File organization (rename/move)
async def organize_project_files() -> str:
    """Rename/move files using the File manager."""
    # NOTE: File-manager paths are typically root-relative to the active filesystem adapter.
    # Avoid leading "/" unless you truly intend an absolute host path.
    instruction = "Rename docs/notes.txt to docs/notes-2024.txt and move reports/q1.pdf to archive/q1.pdf."
    handle = await primitives.files.organize(instruction)
    return await handle.result()
'''


def get_primitives_files_get_tools_example() -> str:
    """Example: passing FileManager tools to functions that accept a tools parameter."""

    return '''
# primitives.files provides TWO interfaces:
#
# 1. DIRECT METHOD CALLS (use for your own data operations):
#    result = await primitives.files.reduce(table=..., metric="count", ...)
#    tables = await primitives.files.tables_overview()
#
# 2. GET_TOOLS (use ONLY when passing to functions that accept `tools: FileTools`):
#    tools = primitives.files.get_tools()
#    result = await some_function(tools, other_args...)

async def call_function_with_tools(target_fn, **kwargs):
    """When a function signature shows `tools: FileTools`, pass get_tools()."""
    tools = primitives.files.get_tools()
    return await target_fn(tools, **kwargs)
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
# 5. Mixed Examples (Browser + Primitives)
# ---------------------------------------------------------------------------


def get_mixed_browse_persist_example() -> str:
    """Example: browse for data and persist via state managers.

    Source: browser interaction patterns in `tests/test_conductor/test_real/test_actor.py`,
    and contact mutation patterns in `tests/test_conductor/test_real/test_contacts.py`.
    """

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
    """Example: concurrent browser and state manager operations.

    Source: concurrency patterns used throughout eval tests, and conceptually aligned with
    multi-handle scenarios in `tests/test_contact_manager/test_ask.py` (interject) and
    `tests/test_task_scheduler/test_execute.py` (handle lifecycle).
    """

    return '''
# Example: Concurrent browser + state manager operations
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
    """Example: routing interjections to in-flight handles.

    Source: `tests/test_contact_manager/test_ask.py::test_ask_interject` (interject mid-flight),
    and `tests/test_task_scheduler/test_execute.py` (steerable handles support interject/ask/result).
    """

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


def get_browser_verification_extraction_example() -> str:
    """Verification example: browser extraction (browser environment)."""
    return """
### Browser Verification Example: extraction
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


def get_browser_verification_multistep_example() -> str:
    """Verification example: browser multi-step success (browser environment)."""
    return """
### Browser Verification Example: multistep ok
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
    """Verification example: mixed browse + persist success (browser + primitives)."""
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
{"status": "ok", "reason": "The browser evidence matches the extracted email, and the state-manager update confirms it was saved."}
```
""".strip()


def get_verification_examples_for_environments(
    has_browser: bool,
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

    if has_browser:
        sections.append(
            "### Browser Verification Examples\n"
            + "\n\n".join(
                [
                    get_browser_verification_extraction_example().strip(),
                    get_browser_verification_multistep_example().strip(),
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

    if has_browser and has_primitives:
        sections.append(
            "### Mixed Verification Examples\n"
            + get_mixed_verification_browse_persist_example().strip(),
        )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# 7. Helper Functions for Prompt Composition
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

    Excludes library function and stubbing patterns (not applicable to CodeAct).
    Includes patterns for error handling and clarification that complement
    the primitives examples already provided via _build_state_manager_rules_and_examples().
    """

    examples = [
        get_error_handling_example().strip(),
        get_clarification_example().strip(),
    ]
    return "\n\n".join(examples)


def get_browser_examples() -> str:
    """Get all browser-specific examples."""

    examples = [
        get_browser_navigation_example().strip(),
        get_browser_multistep_example().strip(),
        get_browser_screenshot_driven_example().strip(),
    ]
    return "\n\n".join(examples)


def get_primitives_examples() -> str:
    """Get all state manager examples."""

    examples = [
        get_primitives_contact_ask_example().strip(),
        get_primitives_contact_update_example().strip(),
        get_primitives_cross_manager_example().strip(),
        get_primitives_task_lookup_and_execute_example().strip(),
        get_primitives_task_execute_example().strip(),
        get_primitives_dynamic_methods_example().strip(),
        get_primitives_files_ask_example().strip(),
        get_primitives_files_organize_example().strip(),
        get_primitives_files_get_tools_example().strip(),
        get_primitives_guidance_ask_example().strip(),
        get_primitives_guidance_update_example().strip(),
        get_primitives_web_ask_example().strip(),
    ]
    return "\n\n".join(examples)


def get_mixed_examples() -> str:
    """Get all mixed-mode examples."""

    examples = [
        get_mixed_browse_persist_example().strip(),
        get_mixed_concurrent_example().strip(),
        get_mixed_interjection_routing_example().strip(),
    ]
    return "\n\n".join(examples)


def get_examples_for_environments(
    has_browser: bool,
    has_primitives: bool,
    include_core: bool = True,
) -> str:
    """Get examples appropriate for the given environment combination.

    Args:
        has_browser: Whether computer_primitives environment is active
        has_primitives: Whether primitives environment is active
        include_core: Whether to include core patterns (default: True)

    Returns:
        Formatted string with relevant examples
    """

    sections: list[str] = []

    if include_core:
        sections.append(
            "### Core Patterns (Environment-Agnostic)\n" + get_core_pattern_examples(),
        )

    if has_browser:
        sections.append("### Browser Examples\n" + get_browser_examples())

    if has_primitives:
        sections.append("### State Manager Examples\n" + get_primitives_examples())

    if has_browser and has_primitives:
        sections.append("### Mixed-Mode Examples\n" + get_mixed_examples())

    return "\n\n".join(sections)
