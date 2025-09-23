"""
Generate initial HierarchicalActor plans for WebVoyager benchmark tasks.

This script:
1. Monkey patches HP to return initial plan code without execution
2. Modifies prompts to encourage full implementations (no stubs)
3. Supports single task or batch processing from JSONL
4. Writes each plan to a separate Python file
"""

import asyncio
import json
import os
import sys
import textwrap
import argparse
import re
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

os.environ["UNIFY_MODEL"] = "o4-mini@openai"
from dotenv import load_dotenv

load_dotenv()

import unify

unify.activate("web_voyager_eval", overwrite=True)

from unity.actor import HierarchicalActor
from unity.actor.hierarchical_actor import HierarchicalPlan, _HierarchicalPlanState
import unity.actor.prompt_builders as prompt_builders
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evals/plan_generation.log"),
    ],
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("urllib3").propagate = False
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ========== MONKEY PATCHES ==========


def patched_build_initial_plan_prompt(
    goal: str,
    tools: dict,
    existing_functions: dict = None,
    retry_msg: str = "",
    exploration_summary: Optional[str] = None,
) -> str:
    """Modified prompt that strongly encourages full implementations."""
    # Build a completely new prompt without any stubbing instructions
    tool_reference = prompt_builders._build_tool_signatures(tools)
    handle_apis = prompt_builders._build_handle_apis(tools)
    formatted_functions = prompt_builders._format_existing_functions(
        existing_functions or {},
    )

    prompt = textwrap.dedent(
        f"""
        You are an expert strategist and implementer. Your task is to generate a COMPLETE, FULLY IMPLEMENTED Python script to achieve a user's goal.

        **Primary Goal:** "{goal}"

        ---
        ### CRITICAL IMPLEMENTATION REQUIREMENTS

        1. **COMPLETE IMPLEMENTATIONS ONLY**: Every function you write MUST be fully implemented with working code.
        2. **NO STUBS OR PLACEHOLDERS**: NEVER use 'raise NotImplementedError' or any other placeholder.
        3. **NO DECORATORS**: DO NOT use @verify or any other decorators - just define clean async functions.
        4. **Entry Point**: The main entry point MUST be `async def main_plan()`.
        5. **No Imports**: You MUST NOT use any `import` statements. All needed modules are pre-imported.
        6. **Async Functions**: All functions MUST be `async def`.
        7. **Await Calls**: All `action_provider` methods that are async MUST be called with `await`.
        8. **Structured Output**: Use Pydantic BaseModel for structured data extraction.
        9. **Error Handling**: Include proper try/except blocks for error handling.

        ---
        ### IMPLEMENTATION STRATEGY

        When implementing browser automation:
        1. **Observe Before Acting**: Always use `browser_observe` to confirm elements exist before interacting.
        2. **Be Specific**: Use precise descriptions like "Click the blue 'Submit' button" not just "Click button".
        3. **Handle Dynamic Content**: Include logic for scrolling, waiting, or retrying if elements aren't immediately visible.
        4. **Implement Search Logic**: When searching for items:
           - Navigate to the page
           - Find and interact with search elements
           - Scroll through results if needed
           - Click on specific items based on their text/attributes
        5. **Extract Data Properly**: When extracting prices or text:
           - Observe the page structure
           - Look for specific patterns (e.g., currency symbols, product names)
           - Return structured data using Pydantic models
        6. **Website Navigation**: For tasks that require navigating to a specific website:
           - Start with `browser_act("Navigate to [URL]")`
           - Wait for page load if needed
           - Use `browser_observe` to understand the page structure
        7. **Search and Filter**: When finding specific items:
           - Look for search boxes, filter options, or navigation menus
           - Use descriptive actions like "Type 'vegetarian lasagna' in the search box"
           - Apply filters step by step

        ---
        ### Tools Reference
        You have access to a global `action_provider` object with these methods:
        ```json
        {tool_reference}
        ```

        ---
        ### Handle APIs
        {handle_apis}

        ---
        ### Usage Examples

        **Using a Handle-Based Tool (like sending a message or making a call):**

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

        **Multiple Steps with Stubs & Dynamic Implementation:**
        This example shows the correct way to structure a plan that defers a complex step.

        ```python
        @verify
        async def login_to_portal():
            # This part is simple and can be implemented directly.
            await action_provider.browser_act("Navigate to [https://portal.example.com/login](https://portal.example.com/login)")
            await action_provider.browser_act("Enter 'user@example.com' into the email field")
            await action_provider.browser_act("Click the 'Next' button")

        @verify
        async def scrape_user_dashboard():
            # This is a complex step that requires seeing the dashboard page first.
            # Therefore, we correctly stub it out.
            raise NotImplementedError("Implement logic to find and extract data from the user dashboard.")

        @verify
        async def main_plan():
            # In the main plan, we call the functions in order.
            # Notice there is NO try...except block here.
            # The actor is designed to automatically catch the NotImplementedError from
            # scrape_user_dashboard, implement that function, and then resume the plan.
            await login_to_portal()
            dashboard_data = await scrape_user_dashboard()
            return dashboard_data
        ```

        **Using Structured Outputs:**
        ```python
        # Example 1: Extract product information from a search results page
        class ProductInfo(BaseModel):
            name: str = Field(description="The product name as displayed")
            price: str = Field(description="The price shown, including currency symbol")
            in_stock: bool = Field(description="Whether the item shows as available")

        class SearchResults(BaseModel):
            products: list[ProductInfo] = Field(description="List of visible products")
            total_count: str = Field(description="Total number of results shown on page")

        @verify
        async def extract_search_results():
            # Observe the page to extract structured product data
            results = await action_provider.browser_observe(
                "List all visible products on this search results page with their prices and availability status. Also note the total result count.",
                response_format=SearchResults
            )

            # Now we can process the data programmatically
            affordable_products = []
            for p in results.products:
                try:
                    # Remove $ and commas, then convert to float
                    price_value = float(p.price.replace("$", "").replace(",", "").strip())
                    if price_value < 50:
                        affordable_products.append(p)
                except ValueError:
                    # Skip products with unparseable prices
                    pass
            return results

        # Example 2: Navigate through a multi-step form by reading visible labels
        class FormField(BaseModel):
            label: str = Field(description="The visible label text for this form field")
            field_type: str = Field(description="Type of input: 'text', 'dropdown', 'checkbox', etc.")
            is_required: bool = Field(description="Whether the field shows a required indicator like * or 'required'")

        class FormAnalysis(BaseModel):
            page_title: str = Field(description="The form's title or heading")
            fields: list[FormField] = Field(description="All visible form fields")
            submit_button_text: str = Field(description="Text on the submit button")

        @verify
        async def fill_checkout_form():
            # First, analyze what's on the form
            form_info = await action_provider.browser_observe(
                "Analyze this form page. What is the title, what fields are visible, and what does the submit button say?",
                response_format=FormAnalysis
            )

            # Use the structured data to interact with specific fields
            for field in form_info.fields:
                if field.is_required and field.field_type == "text":
                    if "email" in field.label.lower():
                        await action_provider.browser_act(
                            f"Click on the text field labeled '{{field.label}}' and type 'user@example.com'",
                            "The email field should now contain 'user@example.com'"
                        )
                    elif "name" in field.label.lower():
                        await action_provider.browser_act(
                            f"Click on the text field labeled '{{field.label}}' and type 'John Doe'",
                            "The name field should now contain 'John Doe'"
                        )

            # Submit using the exact button text we observed
            await action_provider.browser_act(
                f"Click the '{{form_info.submit_button_text}}' button",
                "The form should be submitted and we should see a confirmation page"
            )
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
        ---
        ### Existing Functions Library
        You may use these pre-existing functions if they are suitable:
        {formatted_functions}

        ---
        {retry_msg}

        Remember: Every function must be FULLY IMPLEMENTED. NO STUBS. NO NotImplementedError.
        Your response must start immediately with the code.
    """,
    ).strip()

    return prompt


# Apply the monkey patch
prompt_builders.build_initial_plan_prompt = patched_build_initial_plan_prompt


def post_process_generated_code(code: str) -> str:
    """
    Post-process the generated code to:
    1. Remove @verify decorators
    2. Ensure required imports are at the top
    3. Ensure action_provider initialization is present after imports
    """
    # Remove @verify decorators
    code = re.sub(r"@verify\s*\n", "", code)

    # Split the code into lines
    lines = code.split("\n")

    # Collect existing imports and non-import code separately
    import_lines = []
    code_lines = []
    in_imports_section = True

    for line in lines:
        stripped = line.strip()
        if in_imports_section:
            if stripped.startswith(("import ", "from ")) or not stripped:
                import_lines.append(line)
            else:
                # We've reached the first non-import line
                in_imports_section = False
                code_lines.append(line)
        else:
            code_lines.append(line)

    # Define required imports
    required_imports = [
        "import asyncio",
        "import re",
        "from pydantic import BaseModel, Field",
        "from typing import List, Optional",
        "from unity.actor.action_provider import ActionProvider",
    ]

    # Check which imports are missing
    existing_imports_text = "\n".join(import_lines)
    imports_to_add = []

    for imp in required_imports:
        # Check if this import (or a variation) already exists
        if imp.startswith("import "):
            module = imp.split()[1]
            if f"import {module}" not in existing_imports_text:
                imports_to_add.append(imp)
        elif imp.startswith("from "):
            # For 'from X import Y' statements, check more carefully
            parts = imp.split()
            module = parts[1]
            if imp == "from pydantic import BaseModel, Field":
                # Check if pydantic imports exist in any form
                if "from pydantic import" not in existing_imports_text:
                    imports_to_add.append(imp)
            elif imp == "from typing import List, Optional":
                # Check if typing imports exist
                if "from typing import" not in existing_imports_text:
                    imports_to_add.append(imp)
            elif imp == "from unity.actor.action_provider import ActionProvider":
                # Check if ActionProvider import exists
                if (
                    "from unity.actor.action_provider import ActionProvider"
                    not in existing_imports_text
                ):
                    imports_to_add.append(imp)

    # Build the final code
    final_lines = []

    # Add all imports (existing + new)
    final_lines.extend(imports_to_add)
    final_lines.extend(import_lines)

    # Remove any empty lines at the end of imports
    while final_lines and not final_lines[-1].strip():
        final_lines.pop()

    # Add blank line after imports
    if final_lines:
        final_lines.append("")

    # Add action_provider initialization
    final_lines.append("action_provider = ActionProvider()")
    final_lines.append("")

    # Check if action_provider initialization already exists in code_lines
    has_action_provider = any(
        "action_provider = ActionProvider()" in line for line in code_lines
    )

    # Add the rest of the code, skipping any existing action_provider initialization
    for line in code_lines:
        if "action_provider = ActionProvider()" not in line or not has_action_provider:
            final_lines.append(line)
        if "action_provider = ActionProvider()" in line:
            has_action_provider = True  # Mark that we've seen it

    # Join and clean up
    result = "\n".join(final_lines)

    # Clean up multiple blank lines
    result = re.sub(r"\n\n\n+", "\n\n", result)

    # Clean up any pydantic.BaseModel or pydantic.Field references
    result = re.sub(r"pydantic\.BaseModel", "BaseModel", result)
    result = re.sub(r"pydantic\.Field", "Field", result)

    # Ensure there's no leading/trailing whitespace
    result = result.strip() + "\n"

    return result


# 2. Mock HierarchicalPlan to return initial code instead of executing
class MockHierarchicalPlan(HierarchicalPlan):
    """Mock plan that just returns the generated code instead of executing it."""

    async def _initialize_and_run(self):
        """Override to just generate and return the plan code."""
        self.action_log.append("Generating initial plan code...")
        try:
            # Skip exploration for initial plan generation
            self._state = _HierarchicalPlanState.RUNNING

            # Generate the initial plan
            self.plan_source_code = await self.actor._generate_initial_plan(
                self.goal,
                self.exploration_summary,
            )
            self.action_log.append("Initial plan generated successfully.")

            # Post-process the generated code
            self.plan_source_code = post_process_generated_code(self.plan_source_code)

            # Set the plan code as the final result and mark as completed
            self._state = _HierarchicalPlanState.COMPLETED
            self._set_final_result(self.plan_source_code)

        except Exception as e:
            self._state = _HierarchicalPlanState.ERROR
            self._set_final_result(f"ERROR: Plan generation failed: {e}")


# 3. Patch the actor to use our mock plan
original_execute = HierarchicalActor._execute_and_return_handle


async def patched_execute(self, task_description: str, **kwargs):
    """Return our mock plan instead of the real one."""
    return MockHierarchicalPlan(
        actor=self,
        goal=task_description,
        parent_chat_context=kwargs.get("parent_chat_context"),
        clarification_up_q=kwargs.get("clarification_up_q"),
        clarification_down_q=kwargs.get("clarification_down_q"),
        max_escalations=self.max_escalations,
        max_local_retries=self.max_local_retries,
    )


HierarchicalActor._execute_and_return_handle = patched_execute

# 4. Also patch the execute method to bypass the active task check for batch processing
original_execute = HierarchicalActor.execute


async def patched_execute(self, task_description: str, **kwargs):
    """Override execute to allow multiple sequential tasks in batch mode."""
    # Always clear any existing active task before starting a new one
    self._active_task = None

    # Call the original execute method
    return await original_execute(self, task_description, **kwargs)


HierarchicalActor.act = patched_execute


# ========== MAIN FUNCTIONALITY ==========


async def generate_plan_for_task(
    task: str,
    task_id: str,
    output_dir: Path,
    actor: HierarchicalActor,
) -> Dict[str, Any]:
    """
    Generate a plan for a single task and save it to a file.

    Args:
        task: The task description
        task_id: Unique identifier for the task
        output_dir: Directory to save the generated plan
        actor: The HierarchicalActor instance

    Returns:
        Dict with task_id, success status, and any error message
    """
    logger.info(f"Processing task {task_id}: {task[:100]}...")

    result = {
        "task_id": task_id,
        "task": task,
        "success": False,
        "error": None,
        "has_stubs": False,
        "output_file": None,
    }

    try:
        # Execute the task (which will just generate the plan)
        active_task = await actor.act(task)
        plan_code = await active_task.result()

        # Check if the plan contains NotImplementedError
        if "NotImplementedError" in plan_code:
            result["has_stubs"] = True
            logger.warning(f"Task {task_id}: Plan contains NotImplementedError stubs!")

        # Save the plan to a file
        output_file = output_dir / f"{task_id}.py"
        with open(output_file, "w") as f:
            # Add header comment
            f.write(
                f'"""\nGenerated plan for task: {task}\nTask ID: {task_id}\nGenerated at: {datetime.now().isoformat()}\n"""\n\n',
            )
            f.write(plan_code)

        result["success"] = True
        result["output_file"] = str(output_file)
        logger.info(f"Task {task_id}: Successfully generated plan → {output_file}")

    except Exception as e:
        error_msg = str(e)
        result["error"] = error_msg
        logger.error(f"Task {task_id}: Failed to generate plan - {error_msg}")

    return result


async def process_jsonl_file(
    jsonl_path: Path,
    output_dir: Path,
    limit: Optional[int] = None,
) -> None:
    """
    Process a JSONL file containing WebVoyager tasks.

    Expected JSONL format:
    {"task_id": "123", "task": "Find a recipe..."}

    Args:
        jsonl_path: Path to the JSONL file
        output_dir: Directory to save generated plans
        limit: Optional limit on number of tasks to process
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a summary file
    summary_file = output_dir / "generation_summary.json"
    results = []

    # Read tasks from JSONL
    tasks = []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                data = json.loads(line.strip())

                # Handle different possible formats
                # Format 1: {"task_id": "123", "task": "..."}
                # Format 2: {"id": "123", "query": "...", ...}
                # Format 3: {"id": "123", "ques": "...", "web": "...", ...}

                # Extract task_id
                task_id = data.get("task_id") or data.get("id") or f"task_{i}"

                # Extract task description
                task = (
                    data.get("task")
                    or data.get("query")
                    or data.get("ques")
                    or data.get("description", "")
                )

                # For patchedTasks.jsonl format, prepend the website context if available
                if "web" in data and "ques" in data:
                    # This is the patchedTasks format
                    web_url = data["web"]
                    web_name = data.get("web_name", "")
                    # Enhance the task with website context
                    task = (
                        f"Navigate to {web_url} ({web_name}) and {task}"
                        if web_name
                        else f"Navigate to {web_url} and {task}"
                    )

                if task:
                    tasks.append((task_id, task))
                    logger.info(f"Loaded task {task_id}: {task[:100]}...")
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON at line {i+1}")

    logger.info(f"Loaded {len(tasks)} tasks from {jsonl_path}")

    # Initialize actor once for all tasks
    actor = HierarchicalActor(
        headless=True,
        max_local_retries=2,
        max_escalations=1,
        timeout=1000,
    )

    try:
        # Process each task
        for i, (task_id, task) in enumerate(tasks):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing task {i+1}/{len(tasks)}")
            logger.info(f"{'='*60}")

            result = await generate_plan_for_task(task, task_id, output_dir, actor)
            results.append(result)

            # Clear the active task after each plan completes
            actor.clear_active_task()

            # Save intermediate results
            with open(summary_file, "w") as f:
                json.dump(
                    {
                        "total_tasks": len(tasks),
                        "processed": i + 1,
                        "results": results,
                    },
                    f,
                    indent=2,
                )

            # Small delay between tasks to avoid rate limiting
            await asyncio.sleep(1)

    finally:
        await actor.close()
        await asyncio.sleep(0.5)

    # Final summary
    successful = sum(1 for r in results if r["success"])
    with_stubs = sum(1 for r in results if r["has_stubs"])

    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL SUMMARY:")
    logger.info(f"Total tasks: {len(tasks)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {len(tasks) - successful}")
    logger.info(f"Plans with stubs: {with_stubs}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info(f"{'='*60}")


async def main():
    parser = argparse.ArgumentParser(
        description="Generate initial HierarchicalActor plans. Supports multiple JSONL formats including WebVoyager tasks.",
    )
    parser.add_argument(
        "input",
        help="Task string or path to JSONL file (supports multiple formats)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="evals/generated_plans",
        help="Output directory for generated plans (default: evals/generated_plans)",
    )
    parser.add_argument("-i", "--task-id", help="Task ID (for single task mode)")
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        help="Limit number of tasks to process (for JSONL mode)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)

    # Check if input is a file or a task string
    input_path = Path(args.input)
    if input_path.exists() and input_path.suffix == ".jsonl":
        # Batch mode
        await process_jsonl_file(input_path, output_dir, args.limit)
    else:
        # Single task mode
        output_dir.mkdir(parents=True, exist_ok=True)

        task = args.input
        task_id = args.task_id or f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        actor = HierarchicalActor(
            headless=True,
            max_local_retries=2,
            max_escalations=1,
            timeout=1000,
        )

        try:
            result = await generate_plan_for_task(task, task_id, output_dir, actor)

            if result["success"]:
                print(f"\n✅ Successfully generated plan → {result['output_file']}")
                if result["has_stubs"]:
                    print("⚠️  Warning: Plan contains NotImplementedError stubs")
            else:
                print(f"\n❌ Failed to generate plan: {result['error']}")

        finally:
            await actor.close()
            await asyncio.sleep(0.5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise
