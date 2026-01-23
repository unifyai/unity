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
# 3. Browser Examples
# ---------------------------------------------------------------------------


def get_browser_navigation_example() -> str:
    """Example: navigate and extract data from a webpage."""

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
    """Example: multi-step browser workflow with verification."""

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

    UI actions are guided by the current page state (captured as screenshots/evidence).
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
    # File-manager paths are typically root-relative to the active filesystem adapter.
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
#    storage = await primitives.files.describe(file_path="path/to/file.xlsx")
#    result = await primitives.files.reduce(context=storage.tables[0].context_path, ...)
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
    """Example: concurrent browser and state manager operations."""

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
# 7. Example Tags (for filtering by manager/capability)
# ---------------------------------------------------------------------------

# Maps manager names to their associated example function names
EXAMPLE_TAGS: dict[str, list[str]] = {
    "contacts": [
        "get_primitives_contact_ask_example",
        "get_primitives_contact_update_example",
    ],
    "tasks": [
        "get_primitives_task_execute_example",
        "get_primitives_task_lookup_and_execute_example",
        "get_primitives_dynamic_methods_example",
    ],
    "knowledge": [
        "get_primitives_cross_manager_example",
    ],
    "files": [
        "get_primitives_files_ask_example",
        "get_primitives_files_organize_example",
        "get_primitives_files_get_tools_example",
    ],
    "guidance": [
        "get_primitives_guidance_ask_example",
        "get_primitives_guidance_update_example",
    ],
    "web": [
        "get_primitives_web_ask_example",
    ],
    "core": [
        "get_library_function_reuse_example",
        "get_library_function_composition_example",
        "get_library_function_adaptation_example",
        "get_confidence_based_stubbing_example",
        "get_structured_output_example",
        "get_error_handling_example",
        "get_handle_steering_example",
        "get_clarification_example",
    ],
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


def get_browser_examples() -> str:
    """Get all browser-specific examples."""

    examples = [
        get_browser_navigation_example().strip(),
        get_browser_multistep_example().strip(),
        get_browser_screenshot_driven_example().strip(),
    ]
    return "\n\n".join(examples)


def get_primitives_examples(*, managers: set[str] | None = None) -> str:
    """Get state manager examples, optionally filtered by manager.

    Args:
        managers: If provided, only include examples for these managers.
                  If None, include all examples.

    Returns:
        Formatted string with relevant examples.
    """
    # Map function names to their callables
    all_fns: dict[str, callable] = {
        "get_primitives_contact_ask_example": get_primitives_contact_ask_example,
        "get_primitives_contact_update_example": get_primitives_contact_update_example,
        "get_primitives_cross_manager_example": get_primitives_cross_manager_example,
        "get_primitives_task_lookup_and_execute_example": get_primitives_task_lookup_and_execute_example,
        "get_primitives_task_execute_example": get_primitives_task_execute_example,
        "get_primitives_dynamic_methods_example": get_primitives_dynamic_methods_example,
        "get_primitives_files_ask_example": get_primitives_files_ask_example,
        "get_primitives_files_organize_example": get_primitives_files_organize_example,
        "get_primitives_files_get_tools_example": get_primitives_files_get_tools_example,
        "get_primitives_guidance_ask_example": get_primitives_guidance_ask_example,
        "get_primitives_guidance_update_example": get_primitives_guidance_update_example,
        "get_primitives_web_ask_example": get_primitives_web_ask_example,
    }

    if managers is None:
        # Return all examples
        return "\n\n".join(fn().strip() for fn in all_fns.values())

    # Filter by manager tags
    included_fn_names: set[str] = set()
    for mgr in managers:
        included_fn_names.update(EXAMPLE_TAGS.get(mgr, []))

    examples = []
    for fn_name, fn in all_fns.items():
        if fn_name in included_fn_names:
            examples.append(fn().strip())

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
    *,
    managers: set[str] | None = None,
) -> str:
    """Get examples appropriate for the given environment combination.

    Args:
        has_browser: Whether computer_primitives environment is active
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

    if has_browser:
        sections.append("### Browser Examples\n" + get_browser_examples())

    if has_primitives:
        sections.append(
            "### State Manager Examples\n" + get_primitives_examples(managers=managers),
        )

    if has_browser and has_primitives:
        sections.append("### Mixed-Mode Examples\n" + get_mixed_examples())

    return "\n\n".join(sections)
