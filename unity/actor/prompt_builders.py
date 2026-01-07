from __future__ import annotations

import inspect
import textwrap
import json
from typing import Callable, Dict, Any, Optional, Mapping, TYPE_CHECKING
from unity.actor.prompt_examples import get_verification_examples_for_environments
from unity.common.llm_helpers import (
    class_api_overview,
    get_type_hints,
)
from unity.common.async_tool_loop import SteerableToolHandle

if TYPE_CHECKING:
    from unity.actor.environments.base import BaseEnvironment


def _build_verification_static_prefix(
    environments: Mapping[str, "BaseEnvironment"] | None = None,
) -> str:
    """
    Builds the static, cacheable prefix for verification prompts.
    Returns:
        Static prefix string for verification prompts (≥2,048 tokens)
    """
    has_browser = environments is None or "computer_primitives" in environments
    has_primitives = environments is not None and "primitives" in environments
    examples_block = get_verification_examples_for_environments(
        has_browser=has_browser,
        has_primitives=has_primitives,
    )
    return textwrap.dedent(
        f"""
        You are a pragmatic and meticulous QA reviewer for an autonomous agent. Your job is to assess whether an executed function made **meaningful, accurate progress** toward the **Overall User Goal** using the best available evidence (trace, environment evidence, return value).

        ---
        ### Decision-Making Framework
        Follow this process to arrive at a decision:

        **Step 1: Scrutinize the Low-Level Agent Trace (primary evidence).**
        - Check whether the reasoning is sound and the described actions match the task.
        - Look for confusion, unverified assumptions, or missing preconditions.

        **Step 2: Compare intent vs. evidence.**
        - Use the function's intent and goal as the source of truth.
        - Compare against the trace, environment evidence (e.g., screenshot/URL), and the return value.

        **Step 3: Choose one action.**

        - If the outcome is definitively correct and advances the goal: choose **`ok`**.

        - If the outcome is definitively wrong (trace or evidence contradicts success): choose **`reimplement_local`**.

        - If the function is impossible due to missing preconditions or a parent mistake: choose **`replan_parent`**.

        - If success is ambiguous or the trace shows an unverified assumption: choose **`request_clarification`** (default for ambiguity).

        ### Critical for failures (`reimplement_local`, `replan_parent`)
        Your `reason` MUST include:
        1. **Detailed root cause** (cite specific trace/evidence/return value).
        2. **Actionable fix strategy** (how to re-implement locally or how to replan the parent).

        ---
        ### Examples
        {examples_block}

        ### Function Execution to Verify

        """,
    ).strip()


def _build_dynamic_implement_static_prefix(
    tools: Dict[str, Callable],
    environments: Mapping[str, "BaseEnvironment"] | None = None,
) -> str:
    """
    Builds the static, cacheable prefix for dynamic implementation prompts.
    Args:
        tools: Dictionary of available tool functions

    Returns:
        Static prefix string for implementation prompts (≥2,048 tokens)
    """
    if environments:
        namespaces = ", ".join(f"`{ns}`" for ns in environments.keys())
        tool_usage_instruction = (
            "Use the injected global objects (namespaces) to interact with the environment. "
            f"Available namespaces: {namespaces}."
        )
    else:
        tool_usage_instruction = "Use the injected global objects (namespaces) to interact with the environment."
    rules_and_examples = _build_dynamic_implement_rules_and_examples(
        tools,
        tool_usage_instruction,
        environments,
    )
    env_context_str = _format_environment_contexts(environments)
    env_section = f"\n\n---\n\n{env_context_str}" if env_context_str else ""

    return textwrap.dedent(
        f"""
        You are an expert Python programmer and a master strategist. Your task is to analyze the state of a running plan and decide the best course of action for a function.

        ---
        **CRITICAL: You must choose one of four actions:**
        1.  **`implement_function`**: Write the Python code for the function. Choose this if the function's goal is achievable from the current browser state. **Your code MUST be a single, self-contained `async def` function block. DO NOT include top-level imports or class definitions outside the function.** All necessary imports and helper classes MUST be defined *inside* the function.
        2.  **`skip_function`**: Bypass this function entirely. Choose this if you observe that the function's goal is **already completed** or is now **irrelevant**. For example, skip a "log in" function if you are already logged in.
        3.  **`replan_parent`**: Escalate the failure to the calling function. Choose this if the current function is **impossible to implement** because of a mistake made in a *previous* step. For example, if the goal is "apply filters" but the page has no filter controls, the error lies with the parent function that navigated to the wrong page or failed to get to the right state.
        4.  **`request_clarification`**: Ask the user for help. Choose this if you cannot devise a reliable strategy to fix the function from the available information. For example, if required UI elements are missing or behaving unexpectedly, or if there are multiple possible approaches and you're unsure which the user prefers. **You must provide a clear, specific `clarification_question`.**

        {rules_and_examples}
        {env_section}

        ---

        ### Current Implementation Task

        """,
    ).strip()


def _build_core_implementation_rules() -> str:
    """
    Builds environment-agnostic core implementation rules.

    Returns:
        Formatted string with 9 critical implementation rules
    """
    return textwrap.dedent(
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

        5.  **Await Keyword**: ALWAYS use the `await` keyword when calling ANY `async def` function. This includes all environment methods AND any helper functions or skills you call.
            ```python
            # ✅ CORRECT: Awaiting environment methods (browser or primitives)
            await computer_primitives.navigate("https://example.com")
            contact = await primitives.contacts.ask("Find John Doe")

            # ✅ CORRECT: Awaiting a helper function/skill from the library
            result = await helper_func_1("arg1", "arg2")

            # ❌ WRONG: Forgetting to await
            result = helper_func_1("arg1", "arg2")
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

                # Step 5: Use with response_format (works with any environment)
                result = await some_tool.observe(
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

        8.  **Environment Globals Usage:** Use environment globals directly, no imports or type hints.
            ```python
            # ❌ WRONG: Importing or typing environments
            from somewhere import ComputerPrimitives
            def my_func(computer_primitives: ComputerPrimitives):
                pass

            # ❌ WRONG: Creating environment instances
            computer_primitives = ComputerPrimitives()

            # ✅ CORRECT: Use injected globals directly
            async def my_func():
                # Browser environment
                result = await computer_primitives.navigate("https://example.com")
                data = await computer_primitives.observe("Get page title")

                # State manager environment
                contact = await primitives.contacts.ask("Find John Doe")
            ```

        9. **Requesting Clarification:**
            ```python
            # ✅ CORRECT: Call as a global function
            destination = await request_clarification("What is your destination city?")

            # ❌ WRONG: Do not call it on an environment
            # destination = await computer_primitives.request_clarification(...)
            ```
        """,
    ).strip()


def _build_browser_implementation_examples() -> str:
    """
    Builds browser-specific implementation examples from the centralized library.

    Returns:
        Formatted string with browser navigation, multi-step, and screenshot-driven examples
    """
    from unity.actor.prompt_examples import (
        get_browser_navigation_example,
        get_browser_multistep_example,
        get_browser_screenshot_driven_example,
    )

    return textwrap.dedent(
        f"""
        ### Browser Implementation Examples

        When implementing functions that use `computer_primitives`, follow these patterns:

        {get_browser_navigation_example().strip()}

        {get_browser_multistep_example().strip()}

        {get_browser_screenshot_driven_example().strip()}
        """,
    ).strip()


def _build_primitives_implementation_examples() -> str:
    """
    Builds state manager implementation examples from the centralized library.

    Returns:
        Formatted string with contact, cross-manager, and task execution examples
    """
    from unity.actor.prompt_examples import (
        get_primitives_contact_ask_example,
        get_primitives_contact_update_example,
        get_primitives_cross_manager_example,
        get_primitives_files_ask_example,
        get_primitives_files_get_tools_example,
        get_primitives_files_organize_example,
        get_primitives_guidance_ask_example,
        get_primitives_guidance_update_example,
        get_primitives_task_execute_example,
        get_primitives_web_ask_example,
    )

    return textwrap.dedent(
        f"""
        ### State Manager Implementation Examples

        When implementing functions that use `primitives`, follow these patterns:

        {get_primitives_contact_ask_example().strip()}

        {get_primitives_contact_update_example().strip()}

        {get_primitives_cross_manager_example().strip()}

        {get_primitives_files_ask_example().strip()}

        {get_primitives_files_organize_example().strip()}

        {get_primitives_files_get_tools_example().strip()}

        {get_primitives_guidance_ask_example().strip()}

        {get_primitives_guidance_update_example().strip()}

        {get_primitives_task_execute_example().strip()}

        {get_primitives_web_ask_example().strip()}
        """,
    ).strip()


def _build_mixed_implementation_examples() -> str:
    """
    Builds mixed-mode implementation examples from the centralized library.

    Returns:
        Formatted string with browse-persist and concurrent examples
    """
    from unity.actor.prompt_examples import (
        get_mixed_browse_persist_example,
        get_mixed_concurrent_example,
    )

    return textwrap.dedent(
        f"""
        ### Mixed-Mode Implementation Examples

        When implementing functions that use both `computer_primitives` and `primitives`, follow these patterns:

        {get_mixed_browse_persist_example().strip()}

        {get_mixed_concurrent_example().strip()}
        """,
    ).strip()


def _build_interjection_static_prefix(
    tools: Dict[str, Callable],
    environments: Mapping[str, "BaseEnvironment"] | None = None,
) -> str:
    """
    Builds the static, cacheable prefix for interjection prompts.

    Includes environment-agnostic cache invalidation rules, decision tree,
    routing guidance for in-flight handles, and examples for browser,
    primitives, and mixed-mode workflows.

    Args:
        tools: Dictionary of available tool functions
        environments: Active environments for conditional sections

    Returns:
        Static prefix string for interjection prompts (≥2,048 tokens)
    """
    from unity.actor.prompt_examples import get_interjection_routing_only_examples

    tool_reference = _build_tool_reference_by_namespace(tools, environments)
    handle_apis = _build_handle_apis(tools)
    env_context_str = _format_environment_contexts(environments)
    env_section = f"\n\n---\n\n{env_context_str}" if env_context_str else ""
    routing_only_examples = get_interjection_routing_only_examples()

    return textwrap.dedent(
        f"""
        You are an expert Python programmer and a master strategist responsible for steering a live-running automated plan. A user has interjected with a new instruction while the plan was executing.

        ---
        ### Cache Invalidation Rules (CRITICAL)
        1.  **No Phantom Invalidations**: Only list functions in `invalidate_functions` if they appear in the `Cache Status` list above.
        2.  **Surgical Invalidation**: Use `invalidate_functions` to clear the entire cache for a function, or `invalidate_steps` to clear only a portion of it. Be as minimal as possible to ensure an efficient replay.
        3.  **You may omit `cache`** if nothing needs invalidation.
        ---
        ### Your Task: Analyze, Decide (Routing vs Patching), and Only Patch When Needed

        **1. Analyze Intent:** Choose the best action from the Decision Tree below.

        **2. Decide: Routing-only vs Plan Patching (CRITICAL)**
            - **Routing-only is the default** when the user’s interjection is a *preference update* for what is already running
              (tone, conciseness, formatting, minor scope constraints like “focus only on Q4” for the summaries currently being produced).
            - If the user changes scope/selection for an **already in-flight** tool call and the plan has **not yet consumed the result**,
              prefer **routing-only to that in-flight handle**. Downstream work will
              naturally use the corrected result; do **not** patch/restart “for consistency”.
            - Users will **not** say “broadcast this” or “don’t modify main_plan”. You must infer intent from natural language.
            - **Patches are expensive and disruptive**: they cancel/restart execution and may invalidate caches. Do not patch unless strictly required.
            - **Never create patches just to restate/echo the user’s preference into tool prompts** when routing to in-flight handles can apply it immediately.
            - **Do NOT use circular reasoning** like “patch so replay won’t revert to the old instruction.” Adding `patches` is what *causes* a replay/restart.
              If routing-only is sufficient, avoid patching and avoid the restart entirely.
            - Prefer **targeted routing** when the interjection clearly applies to only one handle (e.g., “make the outreach email tone more casual”
              should route to the in-flight `primitives.transcripts.*` handle generating that content, not to an unrelated contact-list handle).

        **3. Generate Patches (ONLY if modifying code is required):**
            - Read entire `plan_source_code`; identify all changes (implementation, call sites, docstrings).
            - Create `FunctionPatch` for each modified function.
            - Omit `patches` for routing-only interjections (set `routing_action` instead).
            - If you include `patches`, your `reason` must state *why routing-only is insufficient* (e.g., future tool calls must change, plan logic must change, goal changed).

        **4. Devise Cache Strategy (CRITICAL for `modify_task` WITH PATCHES):**

            **Replay Rule:** The plan restarts from `main_plan` after patches. Invalidate only what's affected to preserve valid caches.

            **Scenario 1: Downstream Dependencies (`invalidate_functions`)**
            * Plan: `fetch_records(filter="active") -> process_records() -> generate_report()`. User: "Use filter='archived'."
            * Invalidate: `["fetch_records", "process_records", "generate_report"]` (filter change affects all downstream).

            **Scenario 2: Partial Invalidation (`invalidate_steps`)**
            * Plan: `find_contact(name="Alice") -> update_contact_title()`. User: "I meant Alice Smith."
            * Invalidate: `[{{"function_name": "find_contact", "from_step_inclusive": 1}}]` (preserve earlier steps, clear from step 1).

            **Scenario 3: No Invalidation (Structural Change)**
            * Plan: `prepare_data() -> transform_data() -> save_results()`. User: "Skip save."
            * Omit `cache` field (only removing a step; existing function logic unchanged).

            **Scenario 4: Routing-Only (No Code/Cache)**
            * Plan has concurrent ops. User: "Be concise."
            * Use `routing_action: "broadcast_filtered"` with no `patches` or `cache`.

        ---
        {routing_only_examples}

        ---
        #### 🧠 `modify_task` vs `refactor_and_generalize`
        - **`modify_task`**: Alter behavior (add/correct step, change parameter). Structure stays the same.
        - **`refactor_and_generalize`**: Alter structure (re-apply taught sequence to new target; abstract into reusable skill).

        ---
        ### Decision Tree & Action-Specific Examples
        You MUST respond with a JSON object that strictly adheres to the `InterjectionDecision` Pydantic model.

        #### 1. `modify_task` (Alter Behavior)
        - User: "Search 'monitors' not 'laptops'."
        - Patch both `main_plan` call and `search_products` default. Invalidate affected functions.

        #### 2. `refactor_and_generalize` (Alter Structure)
        - User: "Now do the same for 'Sam Parker'." (after teaching process for "Michael Smith")
        - Refactor monolithic plan into reusable parameterized function.

        #### 3. `replace_task` (Goal Change)
        - User: "Forget the recipe. Find flights SFO to LAX."
        - Complete goal change; start over with new goal.

        #### 4. `request_clarification` (Ambiguous)
        - User: "Make it cheaper." (while searching for laptops)
        - Unclear intent (price filter? different product? coupon?). Ask for clarification.

        #### 5. Primitives-Only Interjection (Correcting Contact Name)
        - **Context**: Plan is executing `await primitives.contacts.ask("Find Alice's email")`
        - **User interjects**: "I meant Alice Smith, not Alice Wonder"
        - **Analysis**: Clarification should be routed to in-flight handle, not patched into code
        - **JSON Output**:
            ```json
            {{
                "action": "modify_task",
                "reason": "User clarified contact name; route to in-flight handle.",
                "routing_action": "targeted",
                "target_handle_ids": ["<handle_id_from_pane_snapshot>"],
                "routed_message": "User clarified: Alice Smith"
            }}
            ```

        #### 6. Mixed-Mode Interjection (Browser + Primitives)
        - **Context**: Plan is browsing LinkedIn and saving contacts concurrently
        - **User interjects**: "Prioritize data validation before saving"
        - **Analysis**: Guidance applies to both browser extraction and contact updates
        - **JSON Output**:
            ```json
            {{
                "action": "modify_task",
                "reason": "User wants validation priority; broadcast to all in-flight operations.",
                "routing_action": "broadcast_filtered",
                "broadcast_filter": {{"capabilities": ["interjectable"]}},
                "routed_message": "Prioritize data validation before proceeding."
            }}
            ```

        ---
        ### Routing Interjections to In-Flight Handles (SteerableToolPane)

        If the dynamic prompt includes an **In-Flight Handles (Steerable)** section, there are active in-flight
        handles that can receive interjections. You must decide whether to route the user's interjection to those
        handles, in addition to choosing your primary `action`.

        Use these optional routing fields in your JSON:
        - `routing_action`: one of `"none"`, `"targeted"`, `"broadcast_filtered"`
        - `target_handle_ids`: list of handle ids (required when `routing_action="targeted"`)
        - `broadcast_filter`: filter dict (used when `routing_action="broadcast_filtered"`)
        - `routed_message`: optional rewritten message for the routed handles (defaults to the user's interjection)

        Routing guidelines:
        1. If the user referenced one or more specific handle ids, use `routing_action="targeted"` and copy those ids.
        2. If the instruction applies broadly (e.g., "be concise", "stop asking clarifying questions"), use
           `routing_action="broadcast_filtered"` with a filter like `{{"capabilities":["interjectable"]}}` and optionally
           constrain by `origin_tool_prefixes` (e.g., `["primitives.contacts"]`).
        3. If there are no active handles, or the interjection is only about plan-level code changes, use `routing_action="none"`.
        4. Routing complements the primary `action` (you may both patch the plan and route to in-flight handles).

        **Routing-only example (NO PATCHES):**
            - User: "Be concise and don't ask clarifying questions — apply this to whatever is currently running."
            - Correct JSON:
            ```json
            {{
              "action": "modify_task",
              "reason": "No plan change needed; route the preference to in-flight handles.",
              "routing_action": "broadcast_filtered",
              "broadcast_filter": {{"capabilities": ["interjectable"]}},
              "routed_message": "Please be concise and avoid asking clarifying questions."
            }}
            ```

        ---
        ### Tool Reference

        **Available Tools:**
        ```json
        {tool_reference}
        ```

        **Handle APIs:**
        {handle_apis}

        {env_section}

        ---

        ### Current Situation

        """,
    ).strip()


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


def _format_environment_contexts(
    environments: Mapping[str, "BaseEnvironment"] | None,
) -> str:
    """Collects environment-provided prompt contexts into a single markdown block."""
    if not environments:
        return ""

    env_contexts: list[str] = []
    for namespace, env in environments.items():
        try:
            ctx = env.get_prompt_context()
        except Exception:
            ctx = ""
        if not ctx or not ctx.strip():
            continue
        env_contexts.append(f"### {namespace} Environment\n{ctx.strip()}")

    if not env_contexts:
        return ""

    return "\n\n---\n\n".join(env_contexts).strip()


def _group_tools_by_namespace(
    tools: Dict[str, Callable],
    environments: Mapping[str, "BaseEnvironment"] | None,
) -> dict[str, Dict[str, Callable]]:
    """
    Groups fully-qualified tool names into their owning namespaces.

    Tool names produced by the actor are expected to be fully-qualified
    (e.g. \"computer_primitives.navigate\", \"primitives.contacts.ask\").
    """
    groups: dict[str, Dict[str, Callable]] = {}
    known_namespaces = set(environments.keys()) if environments else set()

    for tool_name, fn in tools.items():
        if "." in tool_name:
            ns = tool_name.split(".", 1)[0]
        else:
            ns = "global"
        if known_namespaces and ns not in known_namespaces and ns != "global":
            ns = "other"
        groups.setdefault(ns, {})[tool_name] = fn

    # Stable ordering: known namespaces first, then global/other.
    if not known_namespaces:
        return groups

    ordered: dict[str, Dict[str, Callable]] = {}
    for ns in environments.keys():
        if ns in groups:
            ordered[ns] = groups[ns]
    for ns in ("global", "other"):
        if ns in groups:
            ordered[ns] = groups[ns]
    # Include any remaining namespaces deterministically.
    for ns in sorted(set(groups.keys()) - set(ordered.keys())):
        ordered[ns] = groups[ns]
    return ordered


def _build_tool_reference_by_namespace(
    tools: Dict[str, Callable],
    environments: Mapping[str, "BaseEnvironment"] | None,
) -> str:
    """Build a nested tool reference JSON grouped by environment namespace."""
    grouped = _group_tools_by_namespace(tools, environments)
    tool_info: dict[str, Any] = {}
    for ns, ns_tools in grouped.items():
        ns_info: dict[str, Any] = {}
        for fq_name, fn in ns_tools.items():
            # Inner key is the tool name without the leading namespace prefix.
            inner_name = (
                fq_name[len(ns) + 1 :] if fq_name.startswith(f"{ns}.") else fq_name
            )
            prefix = "async def " if inspect.iscoroutinefunction(fn) else "def "
            ns_info[inner_name] = {
                "signature": f"{prefix}{fq_name}{inspect.signature(fn)}",
                "docstring": inspect.getdoc(fn) or "No docstring available.",
            }
        tool_info[ns] = ns_info
    return json.dumps(tool_info, indent=4)


def _build_handle_apis(tool_dict: Dict[str, Callable]) -> str:
    # Deduplicate by return type class to avoid showing the same handle API multiple times
    seen_handle_types = {}  # Maps return_type class to list of tool names

    for name, func in tool_dict.items():
        try:
            hints = get_type_hints(func)
            return_type = hints.get("return")
            if (
                return_type
                and inspect.isclass(return_type)
                and issubclass(return_type, SteerableToolHandle)
            ):
                if return_type not in seen_handle_types:
                    seen_handle_types[return_type] = []
                seen_handle_types[return_type].append(name)
        except Exception:
            continue

    if not seen_handle_types:
        return "There are no special handle APIs for the available tools."

    handle_docs = []
    for return_type, tool_names in seen_handle_types.items():

        examples = ", ".join(f"`{name}`" for name in tool_names)
        doc = f"**`{return_type.__name__}` (returned by {examples})**\n"
        doc += "This handle represents an interactive session. Its available methods are:\n"
        doc += class_api_overview(return_type)
        handle_docs.append(doc)

    return "\n\n".join(handle_docs)


def _format_existing_functions(existing_functions: Dict[str, Any]) -> str:
    """Formats the library of existing functions into clean code blocks."""
    if not existing_functions:
        return "None."

    formatted_summaries = []
    for name, func_data in existing_functions.items():
        signature = func_data.get("argspec", "()")
        docstring = func_data.get("docstring", "No docstring available.")
        implementation = func_data.get("implementation", "")
        if implementation is None:
            implementation = ""

        # Show implementation if it's not too large (< 1000 lines)
        impl_lines = implementation.count("\n") + 1 if implementation else 0
        if impl_lines > 1000:
            # Too large - hide it
            prefix = (
                "async def"
                if isinstance(implementation, str)
                and "async def" in implementation.lstrip()[:10]
                else "def"
            )
            summary = (
                f"{prefix} {name}{signature}:\n"
                f'    """\n'
                f"    {textwrap.indent(docstring, '    ').strip()}\n"
                f'    """\n'
                f"    # ... (implementation is hidden)\n"
            )
        else:
            # Show the full implementation
            summary = implementation.strip()

        formatted_summaries.append(summary)

    if not formatted_summaries:
        return "None."

    return "\n---\n".join(formatted_summaries)


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


def _format_images_for_prompt(images: Optional[dict[str, Any]]) -> str:
    """Creates a markdown section for images if they are provided."""
    if not images:
        return ""

    image_lines = [
        "The user has provided the following images for additional context. ",
        "Refer to these images to better understand the user's intent and any relevant visual details.",
    ]
    for key, handle in images.items():
        try:
            caption = getattr(handle, "caption", "No caption provided.")
            image_lines.append(f"- Image `{key}`: {caption}")
        except Exception:
            continue

    return "\n\n" + "\n".join(image_lines)


def _build_core_planning_rules() -> str:
    """
    Environment-agnostic planning rules (condensed from rules 1-13).
    Removes verbose inline code examples for better token efficiency.
    """
    return textwrap.dedent(
        """
        ### 🎯 CRITICAL RULES FOR INITIAL PLAN CREATION

        1.  **Single Code Block:** Your entire response MUST be a single, valid Python code block. No explanatory text before or after.

        2.  **Entry Point:** For a full plan, the main entry point MUST be `async def main_plan()`.

        3.  **Scoped Imports**: ALL imports must be placed **inside** functions, never at the top level.

        4.  **Decomposition:** Break complex tasks into smaller, focused functions. Each function should have a single, clear purpose.

        5.  **Library Function Reuse** (CRITICAL):
            - **ALWAYS check the Existing Functions Library section below** before writing new code.
            - If a library function matches your goal, call it directly instead of reimplementing its logic.
            - If multiple library functions together achieve the goal, compose them in your plan.
            - Only write new functions when no library function is semantically related to the task.

        6.  **Prefer Direct Tool Calls for Simple Goals** (ALSO CRITICAL):
            - If the user's goal can be completed with **a single state-manager call** (e.g., `primitives.tasks.ask`, `primitives.knowledge.ask`, `primitives.contacts.update`, etc.), write a **minimal plan**:
              - call the one correct tool
              - `await handle.result()`
              - return the answer
            - Avoid helper functions, avoid stubs, and avoid calling multiple different managers unless the request truly spans domains.
            - Use stubs (`raise NotImplementedError(...)`) only for genuinely uncertain, non-trivial logic—not for straightforward manager/tool calls.

        7.  **Async/Await**: ALL functions must be `async def`. ALWAYS use `await` when calling any async function (tools, state managers, helper functions).

        8.  **Structured Output with Pydantic (Only When Needed)**:
            - Import Pydantic inside the function.
            - Define models inside the function.
            - Use `Optional` for fields that might be missing.
            - **CRITICAL**: Call `model_rebuild()` on the outermost model before use.
            - Pass the model to `response_format` parameter.
            - Prefer plain strings / simple return values unless the user explicitly requests structured output or it's necessary to safely parse tool results.

        9.  **Error Handling - NEVER SILENCE ERRORS**:
            - Log exceptions and re-raise them. Never catch and ignore.
            - **EXCEPTION**: Never wrap stubbed functions in try/except (this breaks dynamic implementation).

        10. **Tool Provider Usage**:
            - Use injected global objects directly (provided as globals in your environment).
            - Never import or instantiate tool providers yourself.
            - Never type hint the injected globals.

        11. **Requesting Clarification:**
            - Call `request_clarification("...")` as a global function when you need user input.
            - Do NOT call it as a method on tool providers.

        12. **Return the Final Value**: If the last step returns a value, your `main_plan` MUST capture and return it.

        13. **Function Naming**: Name functions for the *action/process*, not the specific data. Use parameters for specific values.
            - ✅ Good: `async def process_user(username: str)`
            - ❌ Bad: `async def process_user_smith()`

        14. **Handle Ambiguous Goals**: If the user's goal is vague, empty, or "I'll guide you step-by-step", generate a simple plan with just `pass`:
            ```python
            async def main_plan():
                \"\"\"Awaiting user instructions.\"\"\"
                pass
            ```
        """,
    ).strip()


def _build_browser_planning_examples() -> str:
    """Browser-specific planning examples using the centralized library."""
    from unity.actor.prompt_examples import get_browser_examples

    return textwrap.dedent(
        f"""
        ---
        ### Browser Automation Examples

        {get_browser_examples()}
        """,
    ).strip()


def _build_primitives_planning_examples(
    *,
    managers: set[str] | None = None,
) -> str:
    """State manager planning examples using the centralized library.

    Args:
        managers: If provided, only include examples for these managers.
    """
    from unity.actor.prompt_examples import get_primitives_examples

    return textwrap.dedent(
        f"""
        ---
        ### State Manager Examples (`primitives`)

        {get_primitives_examples(managers=managers)}
        """,
    ).strip()


def _build_mixed_planning_examples() -> str:
    """Mixed-mode planning examples using the centralized library."""
    from unity.actor.prompt_examples import get_mixed_examples

    return textwrap.dedent(
        f"""
        ---
        ### Mixed-Mode Examples (Browser + State Managers)

        {get_mixed_examples()}
        """,
    ).strip()


def _build_browser_rules_and_examples(computer_primitives) -> str:
    """Builds the browser-centric rules/examples block (legacy CodeAct content)."""
    all_tools = {}

    browser_tools = {
        "navigate": computer_primitives.navigate,
        "act": computer_primitives.act,
        "observe": computer_primitives.observe,
    }
    all_tools.update(browser_tools)

    if hasattr(computer_primitives, "reason"):
        all_tools["reason"] = computer_primitives.reason

    tool_reference = _build_tool_signatures(all_tools)
    handle_apis = _build_handle_apis(all_tools)

    instructions_and_rules = textwrap.dedent(
        """
        ### 🎯 CRITICAL RULES FOR CODE EXECUTION


        1. **Stateful Execution**: Your code is executed in a persistent, stateful REPL-like environment. Variables, functions, and imports defined in one turn are available in all subsequent turns.

        2. **Use `await`**: The execution sandbox is asynchronous. You **MUST** use the `await` keyword for any computer_primitives operations:
           ```python
           # ✅ CORRECT: Using await
           await computer_primitives.navigate("https://example.com")
           result = await computer_primitives.observe("What is the heading?")

           # ❌ WRONG: Missing await
           computer_primitives.navigate("https://example.com")
           ```

        3. **Imports Inside Code**: All necessary imports must be included in the code you provide:
           ```python
           # ✅ CORRECT: Import inside the code execution
           from pydantic import BaseModel, Field
           from typing import Optional, List
           ```

        4. **Pydantic for Structured Observation**: When using `computer_primitives.observe` to extract structured data:
           ```python
           from pydantic import BaseModel, Field

           class PageInfo(BaseModel):
               title: str = Field(description="Page title")
               products: list[str] = Field(description="List of product names")

           # CRITICAL: Call model_rebuild() after defining nested models
           PageInfo.model_rebuild()

           result = await computer_primitives.observe(
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
                  "code": "await computer_primitives.navigate('https://playwright.dev/')"
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
                  "code": "from pydantic import BaseModel, Field\n\nclass PageContent(BaseModel):\n    heading: str = Field(description=\"The main H1 heading of the page\")\n    first_paragraph: str = Field(description=\"The text of the first paragraph under the heading\")\n\nPageContent.model_rebuild()\n\npage_info = await computer_primitives.observe(\n    \"Extract the main heading and the first paragraph.\",\n    response_format=PageContent\n)\n\nprint(page_info.model_dump_json(indent=2))"
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
                  "code": "await computer_primitives.navigate('https://example.com/')"
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
                  "code": "from pydantic import BaseModel\n\nclass PageTitle(BaseModel):\n    title: str\n\n# MISTAKE: I forgot to call PageTitle.model_rebuild()\n\ntitle_info = await computer_primitives.observe(\n    \"Extract the page title\",\n    response_format=PageTitle\n)\nprint(title_info)"
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
                  "code": "from pydantic import BaseModel\n\nclass PageTitle(BaseModel):\n    title: str\n\n# CORRECTED: I've added the required model_rebuild() call.\nPageTitle.model_rebuild()\n\ntitle_info = await computer_primitives.observe(\n    \"Extract the page title\",\n    response_format=PageTitle\n)\nprint(title_info.title)"
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
                  "code": "import re\nfrom pydantic import BaseModel, Field\nfrom typing import List\n\ndef parse_price(price_str: str) -> float:\n    nums = re.findall(r'[\\d.]+', price_str)\n    return float(nums[0]) if nums else 0.0\n\nclass Product(BaseModel):\n    name: str\n    price_text: str = Field(alias=\"price\")\n\nclass ProductList(BaseModel):\n    products: List[Product]\n\nProductList.model_rebuild()\n\nglobal all_products_data\nall_products_data = await computer_primitives.observe(\n    \"Extract all products with their name and price text\",\n    response_format=ProductList\n)\nprint(f\"Extracted {len(all_products_data.products)} products.\")"
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
                  "code": "sms_handle = await computer_primitives.send_sms_message(\n    description=\"Text Jane Doe to confirm her appointment for tomorrow at 3 PM.\"\n)\n\nsms_result = await sms_handle.result()\nprint(sms_result)"
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
                  "code": "global active_call_handle\nactive_call_handle = computer_primitives.start_call(\n    phone_number=\"Jane Doe\",\n    purpose=\"Ask about dietary restrictions for a lunch meeting.\"\n)\nprint(f\"Initiated call to Jane Doe. Handle ID: {active_call_handle._loop_id}\")"
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
Within your code execution, you have access to a global `computer_primitives` object with these methods:
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


def _build_generic_execution_rules() -> str:
    """Domain-agnostic execution rules for code-first actors."""
    return textwrap.dedent(
        """
        ### 🎯 CRITICAL RULES FOR CODE EXECUTION

        1. **Stateful Execution**: Your code runs in a persistent, stateful REPL-like environment. Variables, functions, and imports defined in one turn are available in subsequent turns.

        2. **Use `await`**: The execution sandbox is asynchronous. You **MUST** use `await` for any async calls.

        3. **Imports Inside Code**: All necessary imports must be included in the code you provide.

        4. **Pydantic for Structured Data (When Supported)**: If a tool supports structured outputs via a `response_format` or schema, define Pydantic models inside the code and call `model_rebuild()` on the outermost model.

        5. **Error Handling**: If your code produces an error, the traceback will be returned. Read it carefully, correct your code, and try again.

        6. **Final Answer Rule**:
           - When the user's request has been fully addressed, you **MUST** provide the final answer directly as a tool-less assistant message.
           - Do not call a tool to print the final answer.
        """,
    ).strip()


def _build_state_manager_rules_and_examples(
    *,
    managers: set[str] | None = None,
    include_examples: bool = True,
) -> str:
    """Rules (and optionally examples) for the `primitives` state manager environment.

    Args:
        managers: If provided, only include examples for these managers.
                  If None, include all managers.
        include_examples: If False, only return rules without examples.
                          Use this when examples are provided elsewhere to avoid duplication.

    Note: Routing guidance (manager descriptions) is now provided by
    StateManagerEnvironment.get_prompt_context() which dynamically generates
    from MANAGER_METADATA. This avoids duplication.
    """
    rules = textwrap.dedent(
        """
        ### 🧩 State Manager Rules

        - **Read vs write**:
          - `await primitives.<manager>.ask(...)` is typically **pure** (read-only).
          - `await primitives.<manager>.update(...)`, `.execute(...)`, `.refactor(...)` are **impure** (they mutate state or start work).

        - **Prefer return values as evidence**: treat return values from state managers as the primary ground truth.

        - **Steerable handles**: some calls return handles. Capture them and await their results:
          ```python
          handle = await primitives.tasks.execute(task_id=123)
          result = await handle.result()
          ```
        """,
    ).strip()

    if not include_examples:
        return rules

    from unity.actor.prompt_examples import get_primitives_examples

    examples = get_primitives_examples(managers=managers)

    return f"{rules}\n\n### Implementation Examples\n\n{examples}"


def _build_code_act_rules_and_examples(
    computer_primitives=None,
    *,
    environments: Mapping[str, "BaseEnvironment"] | None = None,
) -> str:
    """
    Builds the reusable rules/examples block for CodeAct-style execution.

    Backward compatibility:
    - If called with a single positional argument, treat it as legacy `computer_primitives`.
    - New preferred usage passes `environments=...` for environment-aware composition.
    """
    if environments is None:
        # Legacy: browser-only content.
        return _build_browser_rules_and_examples(computer_primitives)

    parts: list[str] = []

    # Always include domain-agnostic execution rules first.
    parts.append(_build_generic_execution_rules())

    # Add core patterns (error handling, clarification) - these complement the
    # primitives examples and are useful regardless of which environments are active.
    from unity.actor.prompt_examples import (
        get_code_act_pattern_examples,
        get_code_act_function_first_examples,
    )

    core_patterns = get_code_act_pattern_examples()
    if core_patterns:
        parts.append(f"### Core Patterns\n\n{core_patterns}")

    # Add function-first guidance when FunctionManager is likely available
    function_first = get_code_act_function_first_examples()
    if function_first:
        parts.append(f"### Function-First Workflow (CRITICAL)\n\n{function_first}")

    cp = None
    if "computer_primitives" in environments:
        try:
            cp = environments["computer_primitives"].get_instance()
        except Exception:
            cp = None
    if cp is not None:
        parts.append(_build_browser_rules_and_examples(cp))

    if "primitives" in environments:
        # Get exposed managers from the environment if available
        env = environments["primitives"]
        managers = getattr(env, "_exposed_managers", None)
        parts.append(_build_state_manager_rules_and_examples(managers=managers))

    return "\n\n---\n\n".join(p for p in parts if p and p.strip()).strip()


def _build_initial_plan_rules_and_examples(
    tools: Dict[str, Callable],
    strategy_instruction: str,
    tool_usage_instruction: str,
    environments: Mapping[str, "BaseEnvironment"] | None = None,
) -> str:
    """Builds the reusable block of core rules and examples for initial planning."""
    # Tool metadata
    tool_reference = _build_tool_reference_by_namespace(tools, environments)
    handle_apis = _build_handle_apis(tools)

    # Core rules (environment-agnostic)
    core_rules = _build_core_planning_rules()

    # Detect active environments
    if environments is None:
        # Legacy mode: infer from tool namespaces to avoid browser assumptions
        # for primitives-only callers
        tool_names = list(tools.keys())
        has_browser = any(name.startswith("computer_primitives") for name in tool_names)
        has_primitives = any(name.startswith("primitives") for name in tool_names)
    else:
        # Modern mode: use explicit environment map
        has_browser = "computer_primitives" in environments
        has_primitives = "primitives" in environments

    routing_instruction_str = ""
    if has_primitives:
        routing_instruction_str = textwrap.dedent(
            """
            **Routing & Tool Choice (IMPORTANT)**
            - When the user's goal is primarily an information request (a question), prefer calling the appropriate state manager via `primitives.*` rather than generating the answer purely from scratch.
            - **External / general knowledge** (public info not stored in our state): default to `await primitives.web.ask(...)` even for stable concepts.
            - Do NOT use `computer_primitives.reason(...)` as a substitute for `primitives.web.ask(...)`. Use `reason` only to structure/summarize after you've gathered evidence (e.g., from web search).
            """,
        ).strip()

    # Get exposed managers from environment for filtering
    managers_filter = None
    if environments and "primitives" in environments:
        env = environments["primitives"]
        managers_filter = getattr(env, "_exposed_managers", None)

    # Compose examples based on active environments
    example_sections = []

    # Always include core pattern examples (environment-agnostic)
    from unity.actor.prompt_examples import get_core_pattern_examples

    core_examples = get_core_pattern_examples()
    if core_examples:
        example_sections.append(
            f"### Core Patterns (Environment-Agnostic)\n\n{core_examples}",
        )

    if has_browser:
        example_sections.append(_build_browser_planning_examples())

    if has_primitives:
        example_sections.append(
            _build_primitives_planning_examples(managers=managers_filter),
        )

    if has_browser and has_primitives:
        example_sections.append(_build_mixed_planning_examples())

    examples_str = "\n\n".join(example_sections) if example_sections else ""

    # Note: primitives_guidance_str is now minimal - detailed manager descriptions
    # are provided by StateManagerEnvironment.get_prompt_context() via env_section.
    # Examples are provided separately via examples_str, so we only include rules here.
    primitives_guidance_str = ""
    if has_primitives:
        primitives_guidance_str = textwrap.dedent(
            f"""
            ---
            ### State Manager Guidance (`primitives`)
            {_build_state_manager_rules_and_examples(managers=managers_filter, include_examples=False)}
            """,
        ).strip()

    # Compose final prompt
    return textwrap.dedent(
        f"""
        ---
        ### Core Instructions & Rules
        {core_rules}

        ---
        ### Strategy & Tool Usage
        {strategy_instruction}
        {tool_usage_instruction}
        {routing_instruction_str}

        ---
        ### Tools Reference
        You have access to global objects (namespaces) for your active environments.
        ```json
        {tool_reference}
        ```

        ---
        ### Handle APIs
        Some tools return a "handle" object for ongoing interaction.
        {handle_apis}

        {primitives_guidance_str}

        ---
        ### Usage Examples
        {examples_str}
        """,
    ).strip()


def _build_dynamic_implement_rules_and_examples(
    tools: Dict[str, Callable],
    tool_usage_instruction: str,
    environments: Mapping[str, "BaseEnvironment"] | None = None,
) -> str:
    """Builds the reusable block of core rules and examples for dynamic implementation."""
    tool_reference = _build_tool_reference_by_namespace(tools, environments)
    handle_apis = _build_handle_apis(tools)

    # Detect active environments
    if environments is None:
        # Legacy mode: infer from tool namespaces
        tool_names = list(tools.keys())
        has_browser = any(
            name.startswith("computer_primitives.") for name in tool_names
        )
        has_primitives = any(name.startswith("primitives.") for name in tool_names)
    else:
        # Modern mode: use explicit environment map
        has_browser = "computer_primitives" in environments
        has_primitives = "primitives" in environments

    # Primitives-only mode: simplified rules without browser references
    if (not has_browser) and has_primitives:
        simplified_rules = textwrap.dedent(
            """
            ### 🎯 CRITICAL RULES FOR DYNAMIC FUNCTION IMPLEMENTATION

            1.  **Single Code Block:** Respond with a single `async def ...` function implementation only (no extra code blocks).

            2.  **Scope and Imports:** ALL imports must be inside the function.

            3.  **No Decorators:** Do not include `@verify` or other decorators.

            4.  **Async All The Way:** The function MUST be async.

            5.  **Await Keyword:** ALWAYS `await` any async call, including state manager calls (e.g. `await primitives.contacts.ask(...)`).

            6.  **Robust Error Handling:** Log errors but ALWAYS re-raise.

            7.  **Use injected globals directly:** Use `primitives` as a global. Do not import or instantiate it, and do not type-hint it.

            8.  **Requesting Clarification:** Call `request_clarification(\"...\")` as a global function when needed.
            """,
        ).strip()

        return textwrap.dedent(
            f"""
            ---
            ### Core Instructions & Rules
            {simplified_rules}
            ---
            ### Tool Usage
            {tool_usage_instruction}

            ---
            ### Tools Reference
            Tools are grouped by namespace below; you MUST call them with the correct arguments as specified here.
            ```json
            {tool_reference}
            ```

            ---
            ### Handle APIs
            Some tools return a \"handle\" object for ongoing interaction. The available methods for these handles are listed below. You MUST only use the methods listed.

            {handle_apis}

            ---
            ### State Manager Guidance (`primitives`)
            {_build_state_manager_rules_and_examples(include_examples=False)}

            {_build_primitives_implementation_examples()}
            """,
        ).strip()

    # Browser or mixed mode: full rules with environment-specific examples
    sections = []

    # Core rules (always included)
    sections.append(
        f"### Core Instructions & Rules\n{_build_core_implementation_rules()}",
    )

    # Strategy and tool usage
    sections.append(
        f"### Tool Usage\n{tool_usage_instruction}",
    )

    # Tool reference and handle APIs
    sections.append(
        f"### Tools Reference\nTools are grouped by namespace below; you MUST call them with the correct arguments as specified here.\n```json\n{tool_reference}\n```",
    )
    sections.append(
        f'### Handle APIs\nSome tools return a "handle" object for ongoing interaction. The available methods for these handles are listed below. You MUST only use the methods listed.\n\n{handle_apis}',
    )

    # Environment-specific examples
    if has_browser:
        sections.append(_build_browser_implementation_examples())

    if has_primitives:
        # Add state manager guidance section (examples provided separately below)
        sections.append(
            f"### State Manager Guidance (`primitives`)\n{_build_state_manager_rules_and_examples(include_examples=False)}",
        )
        sections.append(_build_primitives_implementation_examples())

    if has_browser and has_primitives:
        sections.append(_build_mixed_implementation_examples())

    return "\n\n---\n\n".join(sections)


def build_initial_plan_prompt(
    goal: str,
    existing_functions: Dict[str, Any],
    retry_msg: str,
    *,
    tools: Dict[str, Callable],
    environments: Mapping[str, "BaseEnvironment"] | None = None,
    images: Optional[dict[str, Any]] = None,
) -> str:
    """
    Dynamically builds the system prompt for the Hierarchical Actor.
    """
    formatted_functions = _format_existing_functions(existing_functions)
    image_context_str = _format_images_for_prompt(images)

    env_context_str = _format_environment_contexts(environments)
    env_section = f"\n\n---\n\n{env_context_str}" if env_context_str else ""
    if environments:
        namespaces = ", ".join(f"`{ns}`" for ns in environments.keys())
        tool_usage_instruction = (
            "Use the injected global objects (namespaces) to interact with the environment. "
            f"Available namespaces: {namespaces}."
        )
    else:
        tool_usage_instruction = "Use the injected global objects (namespaces) to interact with the environment."

    rules_and_examples = _build_initial_plan_rules_and_examples(
        tools,
        "",
        tool_usage_instruction,
        environments,
    )

    library_instruction = textwrap.dedent(
        f"""
        ### YOUR AVAILABLE FUNCTIONS (Already Loaded & Callable)

        The following functions are **ALREADY DEFINED and LOADED** in your execution environment.
        They are DIRECTLY CALLABLE in your code. DO NOT redefine them. DO NOT reimplement their logic.

        **HOW TO USE THESE FUNCTIONS:**

        ✅ CORRECT - Direct call in your plan:
        ```python
        async def main_plan():
            # Just call the existing function directly!
            result = await ask_tasks("Which tasks are due today?", response_format=MyModel)
            return result
        ```

        ❌ INCORRECT - Creating a wrapper that reimplements the same logic:
        ```python
        async def query_tasks_due_today():  # ❌ WRONG! ask_tasks() already exists!
            # ❌ Don't call primitives.tasks.ask directly when ask_tasks() exists!
            handle = await primitives.tasks.ask(...)
            return await handle.result()

        async def main_plan():
            result = await query_tasks_due_today()  # ❌ Just call ask_tasks() instead!
            return result
        ```

        **CRITICAL RULES - READ CAREFULLY:**
        1. **CALL, DON'T REDEFINE**: These functions ALREADY EXIST in the runtime. Call them directly by name.
        2. **CHECK THE LIBRARY FIRST**: Before writing ANY new function, scan the available functions below. If one matches your goal, USE IT.
        3. **DIRECT PRIMITIVE CALLS ARE USUALLY WRONG**: If you find yourself writing `await primitives.tasks.ask(...)` or `await primitives.web.ask(...)`, check if a library function already wraps it (like `ask_tasks`, `ask_web`, etc). Use the library function instead.
        4. **COMPOSE IF NEEDED**: If your goal requires multiple steps, orchestrate the existing functions in main_plan().
        5. **ONLY CREATE NEW FUNCTIONS WHEN**: No existing function is semantically related to your goal, OR you need helper logic that doesn't exist in the library.

        **When to use existing functions vs write new code:**
        - ✅ **USE existing function**: The function's purpose matches your goal → Call it directly
        - ✅ **USE existing function**: You can achieve the goal by calling 2-3 existing functions → Compose them
        - ❌ **WRITE new code**: No existing function is semantically related to the goal
        - ❌ **WRITE new code**: Existing functions would require complex workarounds

        **Your Available Functions:**
        ```python
        {formatted_functions}
        ```

        Remember: These are NOT examples to copy - they are CALLABLE FUNCTIONS you can use directly!
        """,
    )

    return textwrap.dedent(
        f"""
        You are an expert strategist. Your task is to generate a high-level Python script that outlines the **strategy** to achieve a user's goal.

        **Primary Goal:** "{goal}"
        {rules_and_examples}
        {env_section}
        ---
        {library_instruction}
        ---
        {retry_msg}

        {image_context_str}
        Begin your response now. Your response must start immediately with the code.
    """,
    ).strip()


def build_dynamic_implement_prompt(
    goal: str,
    scoped_context: str,
    call_stack: list[str],
    function_name: str,
    function_sig: inspect.Signature | str,
    function_docstring: str,
    clarification_question: str | None,
    clarification_answer: str | None,
    replan_context: str,
    has_browser_screenshot: bool = True,
    *,
    tools: Dict[str, Callable],
    existing_functions: Dict[str, Any],
    environments: Mapping[str, "BaseEnvironment"] | None = None,
    recent_transcript: Optional[str] = None,
    parent_chat_context: Optional[list] = None,
    images: Optional[dict[str, Any]] = None,
) -> tuple[str, str]:
    """
    Builds the system prompt for dynamically implementing or modifying a function.

    Returns a tuple of (static_prefix, dynamic_content) for prompt caching.
    The static prefix contains rules, strategy, and examples (~3,500 tokens),
    while the dynamic content contains goal, context, and function-specific data.
    """

    formatted_functions = _format_existing_functions(existing_functions)
    library_instruction = textwrap.dedent(
        f"""
        ### Existing Functions Library (Your Skills)
        **Check this library first** before writing new code. If a function accomplishes the goal (or most of it), call it directly.

        **CRITICAL RULES:**
        1. **Prefer Reuse**: If a library function handles the task, call it instead of reimplementing.
        2. **Only Write the Call**: Write ONLY the call (e.g., `await your_skill_name()`). **DO NOT include its source code**. The framework injects it automatically.
        3. **Compose When Needed**: Orchestrate multiple library functions if the goal requires multiple steps.
        4. **Adapt with Parameters**: Try different parameter values before reimplementing from scratch.

        ```python
        {formatted_functions}
        ```
        """,
    )
    modification_instructions = ""
    image_context_str = _format_images_for_prompt(images)

    if "implement from stub" in replan_context.lower():
        modification_instructions = textwrap.dedent(
            f"""
            ---
            ### 📌 CRITICAL INSTRUCTIONS: IMPLEMENT STUB FUNCTION `{function_name}`
            You are implementing this function for the first time. Its purpose was defined in the initial plan, but the implementation was deferred.

            **Reason for Implementation:**
            {replan_context}
            ---
            """,
        )
    elif (
        "modify" in replan_context.lower()
        or "fix" in replan_context.lower()
        or "crashed" in replan_context.lower()
    ):
        modification_instructions = textwrap.dedent(
            f"""
            ---
            ### 📌 CRITICAL INSTRUCTIONS: MODIFY EXISTING FUNCTION `{function_name}`
            You MUST rewrite the entire `{function_name}` function based on the provided `replan_context`. Analyze the reason, the suggested fix strategy (if provided in the reason), and the original code within the `Scoped Plan Analysis` section below, then produce the complete, final version of the function.

            **Modification Context & Reason:**
            {replan_context}

            **Your Task:**
            - **Focus on 'Current Function Source':** Locate the code for `{function_name}` within the `Scoped Plan Analysis` block below.
            - **Rewrite the Entire Function:** Your output MUST be a single, complete `async def {function_name}` block containing the corrected code.
            - **Integrate Changes:** Apply the necessary fixes or modifications based on the `replan_context`.
            - **Update Docstrings:** Ensure the function's docstring accurately reflects its purpose *after* your modifications.
            ---
            """,
        )
    else:
        modification_instructions = textwrap.dedent(
            f"""
            ---
            ### 📌 CRITICAL INSTRUCTIONS: ADDRESS FUNCTION `{function_name}`
            Analyze the situation described in `replan_context` and the code provided in `Scoped Plan Analysis` below. Decide the best course of action (implement, modify, skip, etc.) for function `{function_name}`.

            **Context:**
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
    has_browser_env = environments is None or "computer_primitives" in environments
    if has_browser_env and has_browser_screenshot:
        browser_context_section = textwrap.dedent(
            """
            **Current Browser View (Screenshot):**
            An image of the current browser page has been provided. Analyze it carefully to inform your implementation or modification. Use it as the primary source of truth for the visual state.
            """,
        )

    call_stack_str = (
        " -> ".join(call_stack)
        if call_stack
        else "N/A (Snapshot Unavailable or Top Level)"
    )
    context_section = textwrap.dedent(
        f"""
    ---
    ### Scoped Plan Analysis & Call Stack (Snapshot)
    This is a snapshot of the plan's source code and call stack captured at the point of failure or when the stub was encountered. Use this historical context to implement or modify `{function_name}` accurately.

    **Call Stack Snapshot:**
    `{call_stack_str}`

    {scoped_context}
    ---
    """,
    )

    static_prefix = _build_dynamic_implement_static_prefix(tools, environments)

    dynamic_content = textwrap.dedent(
        f"""
        ---
        ### Overall Goal (Source of Truth)
        Your implementation MUST satisfy all of the following requirements.

        {goal}
        ---

        {library_instruction}

        {modification_instructions}

        ---
        ### 🕵️ Diagnosis & Fix Strategy (Primary Context)
        CRITICAL: The `replan_context` below contains the expert diagnosis from the verification step, explaining *why* the previous attempt failed or why implementation is needed, along with a **suggested strategy** for the fix or implementation. Use this as your primary guide.

        **Context (`replan_context`):**
        {replan_context}
        ---

        {clarification_section}
        {transcript_section}
        {chat_context_section}
        {context_section}

        ### Situation Analysis
        **Function to Address:** `async def {function_name}{function_sig}`
        **Purpose of this Function:** "{function_docstring}"
        {browser_context_section or "No browser state available."}

        {image_context_str}

        Respond with ONLY the JSON object matching the `ImplementationDecision` schema.
        """,
    ).strip()

    return (static_prefix, dynamic_content)


def build_verification_prompt(
    goal: str,
    function_name: str,
    function_docstring: str | None,
    scoped_context: str,
    interactions: list,
    evidence: Dict[str, Any],
    function_return_value: Any | None,
    clarification_question: Optional[str] = None,
    clarification_answer: Optional[str] = None,
    recent_transcript: Optional[str] = None,
    parent_chat_context: Optional[list] = None,
    environments: Mapping[str, "BaseEnvironment"] | None = None,
) -> tuple[str, str]:
    """
    Builds the prompt for verifying a function's execution.

    Returns a tuple of (static_prefix, dynamic_content) for prompt caching.
    The static prefix contains the decision framework and examples (~4,500 tokens),
    while the dynamic content contains execution-specific data.

    Args:
        goal: The overall user goal.
        function_name: The name of the function being verified.
        function_docstring: The docstring of the function.
        scoped_context: The scoped context of the function (parent, current, and child function source code).
        interactions: A log of tool interactions made.
        evidence: Dictionary of evidence from all active environments.
            Dynamically builds evidence sections based on available evidence types:
            - Visual evidence (browser screenshots)
            - Return value evidence (state manager operations)
            - Mixed evidence (both): if both visual evidence and state-manager return-value evidence are present,
              a dedicated mixed-evidence section is added instructing cross-checking and discrepancy resolution.
        clarification_question: An optional question that was previously asked.
        clarification_answer: An optional answer that was received.
        environments: Optional mapping of active environment namespaces to environment objects.
            Used to select environment-appropriate verification examples in the static prefix.

    Returns:
        Tuple of (static_prefix, dynamic_content) for prompt caching.
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
        or "No tool actions were logged for this step."
    )

    agent_trace_section = "No low-level agent trace was recorded for this step."
    if formatted_agent_traces:
        traces_joined = "\n".join(formatted_agent_traces)
        agent_trace_section = textwrap.dedent(
            f"""
        ---
        ### 🔬 Low-Level Agent Trace (Ground Truth)
        This is the detailed low-level tool trace captured during execution. **This is your most important source of truth.** It reveals *why* actions were taken and what the agent observed at a micro-level across whatever tool domains were used (browser, state managers, etc.). Analyze it carefully to understand the root cause of any success or failure.

        {traces_joined}
        ---
        """,
        )

    # Build evidence sections dynamically based on what's available.
    evidence_sections: list[str] = []

    # Visual evidence (browser).
    browser_evidence = evidence.get("computer_primitives")
    has_browser_screenshot_evidence = False
    if isinstance(browser_evidence, dict):
        if "screenshot" in browser_evidence and "error" not in browser_evidence:
            has_browser_screenshot_evidence = True
            url = browser_evidence.get("url", "N/A")
            evidence_sections.append(
                textwrap.dedent(
                    f"""
                    ---
                    ### 📸 Visual Evidence (Browser)
                    You have been provided a **screenshot** of the browser's final state.
                    - **URL**: {url}
                    - Use the screenshot to visually confirm the outcome described in the agent trace.
                    """,
                ),
            )
        elif "error" in browser_evidence:
            evidence_sections.append(
                textwrap.dedent(
                    f"""
                    ---
                    ### ⚠️ Browser Evidence Unavailable
                    Could not capture browser state: {browser_evidence.get('error')}
                    """,
                ),
            )

    # State manager evidence (return values).
    primitives_evidence = evidence.get("primitives")
    has_primitives_return_value_evidence = False
    if (
        isinstance(primitives_evidence, dict)
        and primitives_evidence.get("type") == "return_value"
    ):
        has_primitives_return_value_evidence = True
        evidence_sections.append(
            textwrap.dedent(
                """
                ---
                ### 📊 System State Evidence (Return Values)
                For state manager operations, the **return value** is the primary evidence.
                - If the function returned a success message or data, that confirms the operation succeeded.
                - If the function returned an error or unexpected value, that indicates failure.
                - Analyze the return value in the "Function Return Value" section below.
                """,
            ),
        )

    # Mixed evidence (browser + primitives): add dedicated instructions when both are present.
    if has_browser_screenshot_evidence and has_primitives_return_value_evidence:
        evidence_sections.append(
            textwrap.dedent(
                """
                ---
                ### 🔀 Mixed Evidence (Browser + Return Value)
                Both a browser screenshot and a state-manager return value are available.
                - **Cross-check the screenshot against the returned value, resolve discrepancies, and prefer the ground-truth source.**
                - Treat the **screenshot** as ground truth for the browser/UI final state and treat the **return value** as ground truth for state-manager mutation outcomes.
                - If they disagree, explain why (cite the trace/evidence) and choose the most conservative status (often `reimplement_local` or `request_clarification`).
                """,
            ),
        )

    evidence_context_section = (
        "\n".join(evidence_sections).strip() if evidence_sections else ""
    )
    return_value_log = f"```\n{repr(function_return_value)}\n```"

    source_code_section = f"""
---
### ⚙️ Scoped Source Code Context
This provides the parent, current, and child function source code to give you complete context
for understanding the current execution context.

{scoped_context}
---
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

    static_prefix = _build_verification_static_prefix(environments=environments)

    dynamic_content = textwrap.dedent(
        f"""
        **🎯 Overall User Goal:** "{goal}"
        **🔍 Function Under Review:** `{function_name}`
        **Intent (Purpose of this function):** {function_docstring or 'No docstring provided.'}

        {source_code_section}
        {agent_trace_section}
        {evidence_context_section}
        {clarification_section}
        {transcript_section}
        {chat_context_section}

        **📊 Execution Evidence**
        **Function Return Value:**
        {return_value_log}
        **High-Level Tool Interactions Log:**
        {interactions_log}

        ---
        Now, provide your assessment based on all the evidence and the decision framework. Respond with ONLY the JSON object.
        """,
    ).strip()

    return (static_prefix, dynamic_content)


def build_interjection_prompt(
    interjection: str,
    parent_chat_context: list[dict] | None,
    scoped_context: str,
    call_stack: list[str],
    action_log: list[str],
    goal: str,
    idempotency_cache: Dict[tuple, Any],
    *,
    tools: Dict[str, Callable],
    environments: Mapping[str, "BaseEnvironment"] | None = None,
    images: Optional[dict[str, Any]] = None,
    pane_snapshot: Optional[dict[str, Any]] = None,
) -> tuple[str, str]:
    """
    Builds the system prompt for the Interjection Handler LLM.

    Returns a tuple of (static_prefix, dynamic_content) for prompt caching.
    The static prefix contains strategy, cache rules, and decision tree (~4,000 tokens),
    while the dynamic content contains the interjection and current execution state.
    """
    cache_summary = _format_cache_summary(idempotency_cache)
    image_context_str = _format_images_for_prompt(images)
    pane_context_str = ""
    if pane_snapshot and pane_snapshot.get("active_handles"):
        handles = pane_snapshot.get("active_handles") or []
        if handles:
            pane_context_str = "\n### In-Flight Handles (Steerable)\n"
            pane_context_str += "The following handles are currently active and can receive interjections:\n"
            for h in handles:
                pane_context_str += (
                    f"- **handle_id**: `{h.get('handle_id')}` | "
                    f"**origin_tool**: `{h.get('origin_tool')}` | "
                    f"**status**: `{h.get('status')}` | "
                    f"**capabilities**: {h.get('capabilities')}\n"
                )
            pane_context_str += (
                "\nYou can route this interjection to relevant handles using the `routing_action` field. "
                "Use 'targeted' to route to specific handle_ids (especially if the user referenced handle ids), "
                "or 'broadcast_filtered' to route to all handles matching criteria (e.g., all `primitives.contacts` handles).\n"
            )

    call_stack_str = (
        " -> ".join(call_stack) if call_stack else "Not inside any function."
    )
    recent_actions = "\n".join(f"- {log}" for log in action_log) or "No actions yet."
    chat_history = (
        json.dumps(parent_chat_context, indent=2)
        if parent_chat_context
        else "No prior conversation."
    )

    static_prefix = _build_interjection_static_prefix(tools, environments)

    dynamic_content = textwrap.dedent(
        f"""
    ### Full Situational Context
    - **User's Interjection:** "{interjection}"
    - **Current Goal (Source of Truth):** "{goal or 'None (This is a teaching session)'}"
    - **Full Conversation History:** {chat_history}
    - **Scoped Source Code Context:**
      {scoped_context}
    - **Current Execution Point (Call Stack):** `{call_stack_str}`
    - **Most Recent Plan Actions:**
      {recent_actions}
    {image_context_str}
    {pane_context_str}

    {cache_summary}
    ---
    Now, provide your decision. Your response must be ONLY the JSON object.
    """,
    ).strip()

    return (static_prefix, dynamic_content)


def build_ask_prompt(
    goal: str,
    state: str,
    call_stack: str,
    context_log: str,
    question: str,
    environments: Mapping[str, "BaseEnvironment"] | None = None,
    evidence: Dict[str, Any] | None = None,
) -> str:
    """
    Builds the system prompt for answering questions about the plan's state.

    Args:
        goal: The overall goal of the plan.
        state: The current lifecycle state of the plan.
        call_stack: The current function call stack.
        context_log: A log of recent actions.
        question: The user's question.
        environments: The active environments for conditional prompt sections.
        evidence: Evidence dict from all environments (screenshot, return values, etc.).
            Keys are environment namespaces; values are evidence dicts from env.capture_state().

    Returns:
        The complete prompt string.
    """
    # Determine available evidence types
    has_browser = "computer_primitives" in (environments or {})
    has_primitives = "primitives" in (environments or {})

    # Check for visual evidence (screenshot from any environment)
    has_visual_evidence = False
    if evidence:
        for env_namespace, env_evidence in evidence.items():
            if env_evidence.get("type") == "screenshot" or "screenshot" in env_evidence:
                has_visual_evidence = True
                break

    # Check for primitives evidence (return values)
    has_primitives_evidence = False
    if evidence:
        for env_namespace, env_evidence in evidence.items():
            if env_evidence.get("type") == "return_value":
                has_primitives_evidence = True
                break

    # Build context items dynamically
    context_items = []
    context_items.append("1. **Goal:** Primary objective")
    context_items.append("2. **Action Log:** Chronological history")

    if has_visual_evidence:
        context_items.append("3. **Visual Evidence:** Screenshots")

    if has_primitives_evidence:
        context_items.append(
            f"{len(context_items) + 1}. **State Evidence:** Return values from primitives",
        )

    context_items.append(
        f"{len(context_items) + 1}. **Call Stack:** Current execution point",
    )
    context_items.append(
        f"{len(context_items) + 1}. **Tools:** `query` for memory/history",
    )

    context_items_str = "\n        ".join(context_items)

    # Determine task type
    if has_browser and has_visual_evidence:
        task_type = "web automation task"
    elif has_primitives:
        task_type = "state management task"
    else:
        task_type = "task"

    # Add guidance with examples
    guidance = ""
    if has_primitives_evidence:
        guidance = (
            "\n\n**State Evidence:**\n"
            "- Use recent `primitives.*` return values for data questions\n"
            "- Check log for mutations (update/execute) to see changes\n"
            '- Ex: "How many tasks?" → check `primitives.tasks.ask`; "Did update succeed?" → check log'
        )

    if has_visual_evidence and has_primitives_evidence:
        guidance += '\n\n**Mixed-Mode:** Combine screenshot + primitives state for complete answers (e.g., "What\'s our progress?").'

    # Update context note
    context_note = "reviewing **Action Log**"
    if has_visual_evidence:
        context_note += " + **Visual Evidence**"
    if has_primitives_evidence:
        context_note += " + **State Evidence**"

    return textwrap.dedent(
        f"""
        You are an AI assistant performing a {task_type}. User paused to ask:

        **Available Context:**
        {context_items_str}
        {guidance}

        **Goal:** {goal}
        **State:** {state}
        **Call Stack:** {call_stack}

        --- ACTION LOG ---
        {context_log}
        --- END LOG ---

        Answer by {context_note}.

        **Question:** "{question}"
        **Answer:**
    """,
    ).strip()


def build_sandbox_merge_prompt(
    main_goal: str,
    main_plan_source: str,
    sandbox_goal: str,
    sandbox_result: str,
) -> str:
    """
    Builds the prompt for the sandbox merge decision LLM.

    NOTE: This prompt is already domain-agnostic and works for browser,
    state manager, and mixed-modality workflows without modification.
    """
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
    current_url: str | None,
    *,
    tools: Dict[str, Callable],
    environments: Mapping[str, "BaseEnvironment"] | None = None,
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
    tool_usage_instruction = (
        "Use the injected global objects (namespaces) to interact with the environment. "
        "Available tools and their handle APIs have been described in the rules below."
    )
    rules_and_examples = _build_initial_plan_rules_and_examples(
        tools,
        strategy_instruction,
        tool_usage_instruction,
        environments=environments,
    )
    browser_context = f"- **Current URL:** `{current_url}`\n" if current_url else ""

    # Build example section - use browser example if URL is available, otherwise generic state-manager example
    if current_url:
        example_section = textwrap.dedent(
            """
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
                await computer_primitives.act(f"Type '{item_name}' into the search bar and press Enter")

            @verify
            async def add_first_item_to_cart():
                \"\"\"Clicks the 'Add to Cart' button for the first search result.\"\"\"
                await computer_primitives.act("Click the 'Add to Cart' button for the first item in the list")

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
                await computer_primitives.navigate("https://shop.example.com/home")

                # Now, execute the generalized workflow.
                await search_for_item("keyboards")
                await add_first_item_to_cart()
                print("Successfully added keyboards to the cart.")

            ```
            """,
        ).strip()
    else:
        example_section = textwrap.dedent(
            """
            ---
            ### Example of the Expected Output

            **Scenario:**
            - **Taught Process:** The user guided the agent to query contact "Alice Smith" and update her job title.
            - **Generalization Request:** "Great. Now do the same for 'Bob Johnson' with title 'Manager'."

            **Your Correct Output (a single Python code block):**
            ```python
            # Part 1: The refactored helper functions (the "skills")
            @verify
            async def update_contact_title(name: str, title: str):
                \"\"\"Queries a contact and updates their job title.\"\"\"
                contact_info = await primitives.contacts.ask(f"Find contact named {name}")
                await primitives.contacts.update(f"Update {name}'s job title to {title}")
                return contact_info

            # Part 2: The intelligent `main_plan` orchestrator
            @verify
            async def main_plan():
                \"\"\"
                Orchestrates the process of updating Bob Johnson's title to Manager.
                \"\"\"
                result = await update_contact_title("Bob Johnson", "Manager")
                print(f"Successfully updated contact: {result}")

            ```
            """,
        ).strip()

    return textwrap.dedent(
        f"""
        You are an expert Python programmer who refactors monolithic scripts into modular, reusable code. You must be mindful of the agent's state when generating the new plan.

        ### Full Context
        - **User's Generalization Request:** "{generalization_request}"
        {browser_context.strip() if browser_context else ""}
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
            2.  **Compare with Current State:** Compare that required start state with the current system state available to you (e.g., a URL if present, a task context, or other environment state).
            3.  **Bridge the Gap:** Your `main_plan` must **bridge this state gap**. The very first step in your `main_plan` must use the appropriate environment namespace/tool(s) to get from the current state to the necessary starting state for your helper functions. This is your "course correction" step.
            4.  **Execute the Goal:** After the state-setting step, `main_plan` should then call your helper functions in the correct order to fulfill the user's request.

        {example_section}

        {rules_and_examples}

        Begin your response now. Your response must start immediately with the JSON object.
        """,
    )


def build_precondition_prompt(
    function_source_code: str,
    interactions_log: str,
    has_entry_screenshot: bool,
    environments: Mapping[str, "BaseEnvironment"] | None = None,
) -> str:
    """
    Builds the prompt to determine the precondition for a function to run.

    Args:
        function_source_code: The source code of the function.
        interactions_log: A JSON string of the tool interactions during the function's run.
        has_entry_screenshot: Whether a screenshot or visual evidence is provided.
        environments: The active environments for conditional examples and language.
    """
    has_browser = "computer_primitives" in (environments or {})

    screenshot_section = ""
    if has_entry_screenshot:
        evidence_type = (
            "the execution environment" if not has_browser else "the browser"
        )
        screenshot_section = textwrap.dedent(
            f"""
            ---
            ### CRITICAL: Visual Context (Entry Screenshot)
            You have been provided with a screenshot of {evidence_type}'s state at the moment this function was called.
            - **Use this image as the primary source of truth** to determine the necessary starting conditions.
            - Analyze the image to describe the required visible elements.
            """,
        )

    # Conditional examples and language based on environment
    if has_browser:
        agent_description = "an autonomous web agent"
        function_description = "A function that interacts with a web browser"
        state_basis = "Based on the function's first few actions and the visual screenshot, what *must* be true about the page for this function to succeed?"
        url_guidance = """3.  **Prioritize Description Over Specific URLs.**
            * If the function's purpose is generic (like extracting search results or items from a list), the **URL is incidental**. The important precondition is the *type* of page. In this case, provide a `description` like "A search results page must be visible" or "A product listing page must be displayed." **Do not include a specific URL.**
            * If the function's purpose is tied to a **specific, static page** (like a homepage, a login page, or a settings dashboard), then the **URL is essential**. Provide the `url` in the precondition. A `description` can still be added for clarity (e.g., "The main login form must be visible")."""
        examples = """**Examples:**
        * **Good (Generic):** `{ "status": "ok", "description": "A list of search results for a recipe is displayed." }`
        * **Good (Specific):** `{ "status": "ok", "url": "https://example.com/login", "description": "The main login form is visible." }`
        * **Bad (Too Rigid):** `{ "status": "ok", "url": "https://example.com/search?q=laptops" }` (This is not reusable for other searches)."""
    else:
        agent_description = "an autonomous agent"
        function_description = "A function"
        state_basis = "Based on the function's first few actions and any provided evidence, what *must* be true about the system state for this function to succeed?"
        url_guidance = """3.  **Prioritize Description Over Specific Identifiers.**
            * If the function's purpose is generic, describe the *type* of state required.
            * If the function's purpose is tied to a specific resource, include identifiers when necessary."""
        examples = """**Examples:**
        * **Good:** `{ "status": "ok", "description": "Contact 'Alice Smith' exists in the system." }`
        * **Good:** `{ "status": "ok", "description": "Task queue contains at least one pending task." }`
        * **Bad:** `{ "status": "ok", "description": "The system is running." }` (Too vague, not verifiable)."""

    return textwrap.dedent(
        f"""
        You are a state analysis expert for {agent_description}.
        {function_description} has just executed successfully. Your task is to describe the necessary **precondition** for this function to run correctly. A good precondition is general enough to be reusable but specific enough to ensure the function works.

        **Function Source Code:**
        ```python
        {function_source_code}
        ```

        **Execution Interaction Log:**
        ```json
        {interactions_log}
        ```

        {screenshot_section}

        **Your Task & Reasoning Framework:**
        1.  **Analyze the function's purpose.** What is the core task?
        2.  **Determine the required state.** {state_basis}
        {url_guidance}
        4.  **Be concise and verifiable.** The description should be a simple statement about the required state.

        {examples}

        Respond with ONLY the JSON object matching the `PreconditionDecision` schema.
        """,
    )


#   1.  **Dual Environments**: You can see and interact with two main components:
#             - A **Chromium web browser** for all internet-related tasks.
#             - An **`xterm` terminal** for all command-line operations.

#         2.  **Workflow Integration**: The most powerful solutions often involve using both environments together. A common workflow is to use the browser to find and download a file, then use the terminal to install or process that file.

#         3.  **OS-Awareness**: The terminal is a standard Debian Linux environment.
#             - Use `apt-get` for package management (e.g., `apt-get update && apt-get install -y <package>`).
#             - Use `dpkg -i <file.deb>` to install downloaded Debian packages.
#             - The default download directory for the browser is `/tmp/unify/assistant/browser/install`. You must use this full path when accessing downloaded files from the terminal.

#         4.  **Command Chaining**: For multi-step terminal operations, chain commands with `&&` within a single `act` call to ensure they execute in the correct sequence and context (e.g., `cd /tmp/downloads && ./install.sh`).
