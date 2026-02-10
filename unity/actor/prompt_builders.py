from __future__ import annotations

import inspect
import textwrap
import json
from typing import Callable, Dict, Optional, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from unity.actor.environments.base import BaseEnvironment
from unity.actor.prompt_examples import (
    get_code_act_pattern_examples,
    get_code_act_function_first_examples,
    get_code_act_session_examples,
    get_computer_examples,
)

# ---------------------------------------------------------------------------
# Static prompt content (inlined rather than wrapped in trivial functions)
# ---------------------------------------------------------------------------

_CRITICAL_RULES_FM = textwrap.dedent("""
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
""").strip()

_EXECUTION_RULES = textwrap.dedent("""
    ### Code Execution Rules

    1. **Session-Based Execution**:
       - All code execution happens via the `execute_code` tool (JSON tool call).
       - **Default is `state_mode="stateless"`** (fresh run; no persistence).
       - Choose `state_mode="stateful"` when you need persistent state across multiple calls (including FunctionManager — see Critical Rules above).
       - Choose `state_mode="read_only"` when you need to use an existing session's state without persisting changes.
       - Use `list_sessions()` / `inspect_state()` to discover and understand active sessions.

    2. **Use `await`**: The execution sandbox is asynchronous. You **MUST** use `await` for any async calls.

    3. **Imports Inside Code**: All necessary imports must be included in the code you provide.

    4. **Pydantic for Structured Data (When Supported)**: If a tool supports structured outputs via a `response_format` or schema, define Pydantic models inside the code and call `model_rebuild()` on the outermost model.

    5. **Sandbox Helpers**: The following helpers are available in `execute_code` Python sessions:
       - `notify(payload)` — Send a progress notification (dict) to the outer handle without blocking. Use for long-running tasks to report intermediate status.
       - `display(obj)` — Emit rich output (text or PIL images) to stdout. Images are auto-resized and base64-encoded. Use instead of `print()` when outputting images.

    6. **Error Handling**: If your code produces an error, the traceback will be returned. Read it carefully, correct your code, and try again.

    7. **Final Answer Rule**:
       - When the user's request has been fully addressed, you **MUST** provide the final answer directly as a tool-less assistant message.
       - Do not call a tool to print the final answer.
""").strip()

_COMPUTER_STATE_FEEDBACK = textwrap.dedent("""
    ### Computer State Feedback

    After computer actions (`computer_primitives.act`, `.navigate`, `.observe`), you automatically receive:
    - The current computer state metadata (e.g., URL when available)
    - A screenshot (as an image block) when available
    - Any output from your code

    Use **stateful sessions** for multi-step computer workflows (e.g., navigate then observe).
""").strip()


# ---------------------------------------------------------------------------
# Private helpers with real logic
# ---------------------------------------------------------------------------


def _build_tool_signatures(tool_dict: Dict[str, Callable]) -> str:
    """Builds a JSON string of tool signatures via introspection."""
    tool_info = {}
    for name, fn in tool_dict.items():
        prefix = "async def " if inspect.iscoroutinefunction(fn) else "def "
        tool_info[name] = {
            "signature": f"{prefix}{name}{inspect.signature(fn)}",
            "docstring": inspect.getdoc(fn) or "No docstring available.",
        }
    return json.dumps(tool_info, indent=4)


def _build_additional_tools_block(
    *,
    tools: Optional[Dict[str, Callable]],
    has_fm_tools: bool,
    has_execute_code: bool,
    render_tools_block: Callable,
) -> str:
    """Build the additional tools + FunctionManager guidance block.

    This appears right after the primary execution tools and before
    the rules/examples, so FunctionManager tool signatures are close
    to the Critical Rules that reference them.
    """
    parts: list[str] = []

    if tools:
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
            parts.append(
                f"### Additional Tools (JSON Tool Calls)\n"
                f"These tools are called via **structured JSON tool calls**, NOT inside Python code.\n\n"
                f"{render_tools_block(additional_tools)}",
            )

        if has_fm_tools and has_execute_code:
            parts.append(
                """\
### Function Execution Modes & State Concepts

**When passing tools to functions:**
- Functions accepting `tools: FileTools` need: `tools = primitives.files.get_tools()`
- For direct data operations, use: `await primitives.files.reduce(...)`

#### Two Types of "State" (Important Distinction)

There are two independent "state" concepts in this system:

| Concept | What It Controls | When to Use |
|---------|------------------|-------------|
| **CodeAct Session State** (`execute_code` `state_mode` parameter) | Whether variables/imports persist between `execute_code` calls | Use `stateful` for multi-step work AND when using FunctionManager functions |
| **Function Execution Mode** (`.stateless()` / `.read_only()` methods) | Whether a FunctionManager function's internal state persists | Use `.stateless()` for pure functions, default for iterative work |

**Key insight**: These are independent! You can call a stateless function in a stateful session.

#### Function Execution Modes (for the function itself)

| Mode | Syntax | State Behavior |
|------|--------|----------------|
| **stateful** (default) | `await func(...)` | Function's internal state persists across calls |
| **stateless** | `await func.stateless(...)` | Fresh environment for function, no inherited state |
| **read_only** | `await func.read_only(...)` | Function sees current state, but changes are discarded |

**Example:**
```python
# Stateful (default) - function's state persists
await load_dataset(path="data.csv")
await analyze_dataset()  # can access data loaded above

# Stateless - isolated execution
result = await compute_score.stateless(values=[1, 2, 3])

# Read-only - see state without modifying it
preview = await transform_data.read_only(sample_size=100)
```""",
            )

    return "\n\n".join(parts)


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
    rules = textwrap.dedent("""
        ### 🧩 State Manager Rules

        - **Do not answer from scratch when `primitives` is available**:
          - If the user asks an information question, prefer calling the relevant state manager via `await primitives.<manager>.ask(...)`
            instead of answering purely from memory.
          - This applies even when you think you "already know" the answer — use the manager as evidence/ground truth.

        - **Read vs write**:
          - `await primitives.<manager>.ask(...)` is typically **pure** (read-only).
          - `await primitives.<manager>.update(...)`, `.execute(...)`, `.refactor(...)` are **impure** (they mutate state or start work).

        - **Prefer return values as evidence**: treat return values from state managers as the primary ground truth.

        - **Steerable handles**: Manager calls return `SteerableToolHandle` objects for in-flight control.
          You can either **await the result** for immediate use, or **return the handle as the last expression**
          of `execute_code` to hand steering control back to the outer loop (see `execute_code` docstring).

          **SteerableToolHandle API:**

          | Method | Returns | Purpose |
          |--------|---------|---------|
          | `await handle.result()` | `str` | Wait for the final result |
          | `await handle.ask(question)` | `SteerableToolHandle` | Query status without modifying execution |
          | `await handle.interject(message)` | `None` | Inject corrections or context mid-flight |
          | `await handle.pause()` | `str | None` | Pause at the next safe point |
          | `await handle.resume()` | `str | None` | Resume a paused operation |
          | `await handle.stop(reason=None)` | `None` | Terminate immediately |
          | `handle.done()` | `bool` | Check if execution has completed |

          ```python
          handle = await primitives.tasks.execute(task_id=123)
          result = await handle.result()  # wait for completion

          # Mid-flight steering (while handle is running):
          await handle.interject("Also include the Q2 numbers")
          await handle.pause()   # pause if needed
          await handle.resume()  # continue later
          await handle.stop()    # cancel if no longer needed
          ```
    """).strip()

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
        parts.append(_EXECUTION_RULES)

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
        parts.append(_COMPUTER_STATE_FEEDBACK)

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
        env = environments["primitives"]
        scope = getattr(env, "primitive_scope", None)
        managers = set(scope.scoped_managers) if scope else None
        parts.append(_build_state_manager_rules_and_examples(managers=managers))
        env_ctx = env.get_prompt_context()
        parts.append(env_ctx)

    return "\n\n---\n\n".join(p for p in parts if p and p.strip()).strip()


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

        critical_rules = _CRITICAL_RULES_FM if has_fm_tools else ""

        additional_tools_block = _build_additional_tools_block(
            tools=tools,
            has_fm_tools=bool(has_fm_tools),
            has_execute_code=True,
            render_tools_block=render_tools_block,
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

{additional_tools_block}

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

        additional_tools_block = _build_additional_tools_block(
            tools=tools,
            has_fm_tools=bool(
                tools
                and any(str(k).startswith("FunctionManager_") for k in tools.keys()),
            ),
            has_execute_code=False,
            render_tools_block=render_tools_block,
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

{additional_tools_block}

{rules_and_examples}
"""

    return prompt
