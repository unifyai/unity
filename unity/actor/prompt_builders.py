from __future__ import annotations

import inspect
import textwrap
import json
from typing import Callable, Dict, Optional, Mapping, TYPE_CHECKING

from unity.common.llm_helpers import (
    class_api_overview,
    get_type_hints,
)
from unity.common.async_tool_loop import SteerableToolHandle

if TYPE_CHECKING:
    from unity.actor.environments.base import BaseEnvironment
from unity.actor.prompt_examples import (
    get_code_act_pattern_examples,
    get_code_act_function_first_examples,
    get_code_act_session_examples,
    get_computer_examples,
)


def build_code_act_prompt(
    *,
    environments: Mapping[str, "BaseEnvironment"],
    tools: Optional[Dict[str, Callable]] = None,
) -> str:
    """
    Build the rich system prompt for the CodeActActor.

    Notes
    -----
    This is intentionally a pure prompt builder (no side effects).
    """
    from unity.common.prompt_helpers import render_tools_block

    has_execute_code = bool(tools and "execute_code" in tools)

    rules_and_examples = _build_code_act_rules_and_examples(
        environments=environments,
        has_execute_code=has_execute_code,
    )

    has_computer_env = "computer_primitives" in environments
    # Detect FunctionManager tools early for critical rules section
    has_fm_tools = tools and any(
        str(k).startswith("FunctionManager_") for k in tools.keys()
    )

    if has_execute_code:
        primary_names = [
            "execute_code",
            "list_sessions",
            "inspect_state",
            "close_session",
            "close_all_sessions",
        ]
        primary_tools = {k: tools[k] for k in primary_names if k in tools}
        primary_tool_reference = (
            _build_tool_signatures(primary_tools) if primary_tools else ""
        )

        role_line = (
            "You are an expert agent that solves tasks by writing and executing code."
        )
        capabilities_line = (
            "Your primary tool is a multi-language, multi-session execution environment where you can run Python and shell code, "
            "and (when enabled) control computer interfaces and other tool domains."
            if has_computer_env
            else "Your primary tool is a multi-language, multi-session execution environment where you can use whatever tool "
            "domains are available via injected environment globals (e.g. state managers, and optionally computer/desktop)."
        )

        critical_rules = _build_critical_rules_section(
            has_function_manager=has_fm_tools,
        )

        prompt = f"""
### Your Role: Code-First Automation Agent
{role_line} {capabilities_line}

{critical_rules}

### Primary Execution & Session Tools
These tools are called via **structured JSON tool calls**, NOT inside Python code.
They are the only supported way to run Python/shell code and manage sessions.

```json
{primary_tool_reference or "{{}}"}
```

{rules_and_examples}
"""
    else:
        # can_compose=False mode: no code sandbox, only stored function execution.
        role_line = (
            "You are an expert agent that solves tasks by discovering and executing "
            "pre-stored functions from a function library."
        )
        capabilities_line = (
            "You do NOT write or execute arbitrary code. Instead, you use the "
            "FunctionManager discovery tools to find relevant stored functions, "
            "then invoke them via `execute_function`."
        )

        prompt = f"""
### Your Role: Function Execution Agent
{role_line} {capabilities_line}

### Workflow
1. **Discover** stored functions using `FunctionManager_search_functions`,
   `FunctionManager_filter_functions`, or `FunctionManager_list_functions`.
2. **Pick** the best match by name from the search results.
3. **Execute** it via `execute_function(function_name=..., call_kwargs=...)`.
4. If no matching function exists, report that clearly — do NOT attempt to
   write or compose code yourself.

{rules_and_examples}
"""

    if tools:
        # Filter out primary execution/session tools since they have a dedicated section above.
        additional_tools = {
            k: v
            for k, v in tools.items()
            if k
            not in {
                "execute_code",
                "list_sessions",
                "inspect_state",
                "close_session",
                "close_all_sessions",
            }
        }
        if additional_tools:
            prompt += (
                f"\n### Additional Tools (JSON Tool Calls)\n"
                f"These tools are called via **structured JSON tool calls**, NOT inside Python code.\n\n"
                f"{render_tools_block(additional_tools)}\n"
            )

        # FunctionManager guidance block: only include execute_code interaction
        # guidance when execute_code is actually available in the tool set.
        if has_fm_tools and has_execute_code:
            prompt += """
### Function Library (DETAILED GUIDANCE)

You have access to a catalogue of **pre-stored reusable functions** via the FunctionManager tools listed above.

⚠️ **CRITICAL REMINDER**: Functions are injected into **Session 0's namespace**.
You **MUST** use `state_mode="stateful"` when calling `execute_code` to access injected functions.
Using `state_mode="stateless"` creates a fresh session → functions NOT available → **NameError**.

**🎯 FUNCTION-FIRST WORKFLOW:**

1. **ALWAYS search first** using FunctionManager tools (structured JSON tool calls, NOT Python code):
   - `FunctionManager_search_functions` - semantic search for functions
   - `FunctionManager_filter_functions` - filter-based search
   - `FunctionManager_list_functions` - list all available functions

   **Important**: Do this **before** you call `execute_code` for a new user request.
   Even if you *think* the correct answer is a direct `primitives.*` call, still search first —
   if a relevant function exists, you must use it.

2. **Functions are automatically injected into Session 0** after searching.
   - The function name(s) returned by the tool become available in Session 0's namespace.
   - Dependencies are injected automatically (including nested helper functions).
   - Venv-backed functions work transparently (subprocess RPC hidden behind an awaitable callable).
   - **You MUST use `state_mode="stateful"` to access them** (stateless creates a new session!).

3. **If found → USE IT**: Pre-saved functions are tested, optimized, and handle edge cases.
   Don't re-explore tables/schemas when a function already exists.

4. **Read signatures carefully**: Check `argspec` in the search results for parameter options
   like `group_by`, `include_plots`, date filters, etc.

5. **Execute found functions** in your Python code with `state_mode="stateful"`.

**✅ CORRECT Example workflow:**
```
# Step 1 (JSON TOOL CALL): Search for function
{
  "name": "FunctionManager_search_functions",
  "arguments": {"query": "contacts prefer phone", "n": 5}
}
# Returns: [{"name": "ask_contacts_question", "argspec": "(question: str) -> str", ...}]

# Step 2 (JSON TOOL CALL): Execute with state_mode="stateful" (REQUIRED!)
{
  "name": "execute_code",
  "arguments": {
    "language": "python",
    "state_mode": "stateful",
    "code": "result = await ask_contacts_question('Which contacts prefer phone?')\\nprint(result)"
  }
}
```

**❌ WRONG Example (causes NameError):**
```
# Step 1: Search (correct)
FunctionManager_search_functions(query="contacts prefer phone", n=5)

# Step 2: Execute with WRONG state_mode
{
  "name": "execute_code",
  "arguments": {
    "language": "python",
    "state_mode": "stateless",
    "code": "result = await ask_contacts_question(...)"
  }
}
# ERROR: NameError: 'ask_contacts_question' is not defined
# WHY: stateless creates fresh session, function NOT available!
```

**❌ ANTI-PATTERN (AVOID THIS):**
```python
# DON'T explore tables when a function already exists!
storage = await primitives.files.describe(file_path="...")  # Unnecessary!
columns = await primitives.files.list_columns(context="...")  # Unnecessary!
```

**When passing tools to functions:**
- Functions accepting `tools: FileTools` need: `tools = primitives.files.get_tools()`
- For direct data operations, use: `await primitives.files.reduce(...)`

### Two Types of "State" (Important Distinction)

There are two independent "state" concepts in this system:

| Concept | What It Controls | When to Use |
|---------|------------------|-------------|
| **CodeAct Session State** (`execute_code` `state_mode` parameter) | Whether variables/imports persist between `execute_code` calls | Use `stateful` for multi-step work AND when using FunctionManager functions |
| **Function Execution Mode** (`.stateless()` / `.read_only()` methods) | Whether a FunctionManager function's internal state persists | Use `.stateless()` for pure functions, default for iterative work |

**Key insight**: These are independent! You can call a stateless function in a stateful session.

**For FunctionManager functions**: You **MUST** use `state_mode="stateful"` in `execute_code` to access
injected functions, regardless of which execution mode you use for the function itself.

### Function Execution Modes (for the function itself)

Functions support three **execution modes** for fine-grained control over the **function's own** state:

| Mode | Syntax | State Behavior |
|------|--------|----------------|
| **stateful** (default) | `await func(...)` | Function's internal state persists across calls |
| **stateless** | `await func.stateless(...)` | Fresh environment for function, no inherited state |
| **read_only** | `await func.read_only(...)` | Function sees current state, but changes are discarded |

**When to use each mode:**

- **stateful** (default): Use for iterative workflows where you build up state incrementally.
  Example: load data once, then run multiple analyses that reference the loaded data.

- **stateless**: Use for pure functions that should produce identical results regardless of
  execution history. Guarantees reproducibility and prevents accidental state pollution.

- **read_only**: Use for "what-if" exploration without side effects. Inspect or transform
  current state without committing changes.

**Example usage:**
```python
# Stateful (default) - function's state persists
await load_dataset(path="data.csv")  # First call: loads 'df' into function's context
await analyze_dataset()               # Second call: can access 'df'

# Stateless - isolated execution for the function
result = await compute_score.stateless(values=[1, 2, 3])

# Read-only - see state without modifying it
preview = await transform_data.read_only(sample_size=100)
```

### Inspecting Code Sessions (execute_code)

Before deciding how to call a function, you can inspect what state currently exists:

```
list_sessions()
```

Then inspect a specific session:
```
inspect_state(detail="summary", session_name="repo_nav")
```

**When to inspect state:**
- Before calling a function that might depend on prior state
- When debugging unexpected behavior (is the state what you expect?)
- When deciding whether to use stateless (isolation) vs stateful (extend existing state)

**Example workflow:**
1. Use `list_sessions()` to see what **CodeAct sessions** exist (Python + shell).
2. Use `inspect_state(...)` to inspect a specific session (e.g., to check cwd / variable names).
3. Run `execute_code(..., state_mode="stateful")` to use FunctionManager-injected functions.
4. Choose the function execution mode (`await fn(...)` / `.stateless()` / `.read_only()`)
   based on whether you want that **skill's internal state** to persist, be isolated, or be discarded.
"""
    return prompt


def _build_critical_rules_section(has_function_manager: bool) -> str:
    """Build the critical rules section that appears first in the prompt.

    This section contains the most important rules that models frequently miss,
    formatted prominently for maximum visibility.
    """
    if not has_function_manager:
        return ""

    return textwrap.dedent(
        """
        ### 🚨 CRITICAL RULES (READ FIRST)

        #### 1. FunctionManager + Stateful Sessions (MOST IMPORTANT)

        When using FunctionManager tools, you **MUST** use `state_mode="stateful"` in `execute_code`:

        | Step | What Happens |
        |------|--------------|
        | `FunctionManager_search_functions(...)` | Function is **injected into Session 0's namespace** |
        | `execute_code(state_mode="stateful", ...)` | ✅ Uses Session 0 → function **available** |
        | `execute_code(state_mode="stateless", ...)` | ❌ Creates NEW session → function **NOT available** → `NameError` |

        **✅ CORRECT workflow:**
        ```
        Step 1: FunctionManager_search_functions(query="...", n=5)  # JSON tool call
        Step 2: execute_code(language="python", state_mode="stateful", code="result = await found_function(...)")
        ```

        **❌ WRONG workflow (causes NameError):**
        ```
        Step 1: FunctionManager_search_functions(query="...", n=5)  # JSON tool call
        Step 2: execute_code(language="python", state_mode="stateless", code="result = await found_function(...)")
        #        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ BUG: stateless creates fresh session, function NOT available!
        ```

        #### 2. Function-First Workflow

        If FunctionManager tools are available, **ALWAYS search BEFORE calling `execute_code`** for a new request:
        1. Search with `FunctionManager_search_functions` (even for "simple" requests)
        2. If a function exists → use it with `state_mode="stateful"`
        3. Only fall back to raw `primitives.*` if no relevant function exists

        #### 3. Default is Stateless (But FunctionManager Requires Stateful)

        - `execute_code` defaults to `state_mode="stateless"` (fresh, isolated execution)
        - **EXCEPTION**: After FunctionManager search, you MUST use `state_mode="stateful"`
        - For multi-step workflows building state, also use `state_mode="stateful"`
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


def _build_generic_execution_rules() -> str:
    """Domain-agnostic execution rules for code-first actors."""
    return textwrap.dedent(
        """
        ### 🎯 CRITICAL RULES FOR CODE EXECUTION

        1. **Session-Based Execution**:
           - All code execution happens via the `execute_code` tool (JSON tool call).
           - **Default is `state_mode="stateless"`** (fresh run; no persistence).
           - Choose `state_mode="stateful"` when you need persistent state across multiple calls.
           - Choose `state_mode="read_only"` when you need to use an existing session's state without persisting changes.
           - **⚠️ EXCEPTION: When using FunctionManager functions, you MUST use `state_mode="stateful"`**
             because functions are injected into Session 0's namespace. Stateless mode creates a fresh session
             where the functions are NOT available (causes NameError).
           - Use `list_sessions()` / `inspect_state()` to discover and understand active sessions.

        2. **Use `await`**: The execution sandbox is asynchronous. You **MUST** use `await` for any async calls.

        3. **Imports Inside Code**: All necessary imports must be included in the code you provide.

        4. **Pydantic for Structured Data (When Supported)**: If a tool supports structured outputs via a `response_format` or schema, define Pydantic models inside the code and call `model_rebuild()` on the outermost model.

        5. **Function-First (When Available)**:
           - If any tool names start with `FunctionManager_`, you **MUST** perform a FunctionManager search **before** you call `execute_code` for a new user request.
           - This rule applies even if the request seems "simple" (including direct state manager calls like `primitives.contacts.ask`, `primitives.tasks.update`, `primitives.guidance.update`, etc.). If a relevant function exists, you must use it.
           - **CRITICAL**: After searching, use `state_mode="stateful"` in `execute_code` to access injected functions.
           - Workflow:
             1) Make a FunctionManager tool call (structured JSON tool call) to search.
             2) If a relevant function exists, call `execute_code` with `state_mode="stateful"` and invoke the function in Python.
             3) Only fall back to calling `primitives.*`/`computer_primitives.*` directly if no relevant function exists.
           - You may skip re-searching only when you already searched in this session and you are confident the needed callable is already injected.

        6. **Error Handling**: If your code produces an error, the traceback will be returned. Read it carefully, correct your code, and try again.

        7. **Final Answer Rule**:
           - When the user's request has been fully addressed, you **MUST** provide the final answer directly as a tool-less assistant message.
           - Do not call a tool to print the final answer.
        """,
    ).strip()


def _build_computer_execution_rules() -> str:
    """Computer-specific execution rules (composition guidance)."""
    return textwrap.dedent(
        """
        ### 🎯 Computer Execution Rules

        1. **Session-Based Execution**:
           - You execute code by calling the `execute_code` tool (JSON tool call), specifying:
             - `language` ("python" or a shell)
             - `state_mode` ("stateless" | "stateful" | "read_only")
             - optionally `session_id` / `session_name`
           - **Default behavior is stateless** unless you explicitly choose `state_mode="stateful"` or `state_mode="read_only"`.
           - Use **stateful sessions** when you need a "tab"/"notebook" that persists across multiple steps (e.g., navigate then observe).
           - Use **stateless** for one-off checks (most reliable / least surprising).
           - Use **read_only** to inspect an existing session without persisting changes.

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

        6. **Computer State Feedback**: After computer actions, you'll automatically receive:
           - The current computer state metadata (e.g., URL when available)
           - A screenshot (as an image block) when available
           - Any output from your code

        7. **Final Answer Rule**:
           - When the user's request has been fully addressed and you have the final answer, you **MUST** provide that answer directly as a tool-less assistant message.
           - Do not call a tool to print the final answer. Simply state the answer.
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
    from unity.function_manager.primitives.ToolSurfaceRegistry. This avoids duplication.
    """
    rules = textwrap.dedent(
        """
        ### 🧩 State Manager Rules

        - **Do not answer from scratch when `primitives` is available**:
          - If the user asks an information question, prefer calling the relevant state manager via `await primitives.<manager>.ask(...)`
            instead of answering purely from memory.
          - This applies even when you think you “already know” the answer — use the manager as evidence/ground truth.

        - **External / public info** (definitions, general concepts, news, weather, “today/latest/now”):
          - Default to `await primitives.web.ask(...)` when available (even for stable concepts/definitions).
          - Do not answer from memory without consulting `primitives.web.ask(...)` unless the user explicitly requests no tool use.

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
    *,
    environments: Mapping[str, "BaseEnvironment"],
    has_execute_code: bool = True,
) -> str:
    """
    Builds the reusable rules/examples block for CodeAct-style execution.

    Composes environment-aware prompt content from execution rules, registry-based
    method documentation, and examples.
    """
    parts: list[str] = []

    # execute_code-specific rules and examples are only relevant when the tool
    # is available. When can_compose=False the tool is masked and the LLM
    # should not receive any references to it.
    if has_execute_code:
        # Domain-agnostic execution rules (session state, imports, etc.)
        parts.append(_build_generic_execution_rules())

        core_patterns = get_code_act_pattern_examples()
        if core_patterns:
            parts.append(f"### Core Patterns\n\n{core_patterns}")

        function_first = get_code_act_function_first_examples()
        if function_first:
            parts.append(
                f"### Function-First Workflow (CRITICAL)\n\n{function_first}",
            )

        session_examples = get_code_act_session_examples()
        if session_examples:
            parts.append(
                f"### Sessions & Multi-Language Execution (CRITICAL)\n\n{session_examples}",
            )

    if "computer_primitives" in environments:
        # Add execution rules (composition guidance)
        parts.append(_build_computer_execution_rules())

        # Add method documentation (from registry)
        env = environments["computer_primitives"]
        env_ctx = env.get_prompt_context()
        if env_ctx:
            parts.append(env_ctx)
        computer_examples = get_computer_examples()
        if computer_examples:
            parts.append(
                f"### Computer Examples\n\n{computer_examples}",
            )

    if "primitives" in environments:
        # Get scope from the environment
        env = environments["primitives"]
        scope = getattr(env, "primitive_scope", None)
        managers = set(scope.scoped_managers) if scope else None
        parts.append(_build_state_manager_rules_and_examples(managers=managers))
        env_ctx = env.get_prompt_context()
        parts.append(env_ctx)

    return "\n\n---\n\n".join(p for p in parts if p and p.strip()).strip()
