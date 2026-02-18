from __future__ import annotations

import inspect
import textwrap
import json
from typing import Callable, Dict, Optional, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from unity.actor.environments.base import BaseEnvironment
from unity.actor.prompt_examples import (
    get_code_act_pattern_examples,
    get_code_act_discovery_first_examples,
    get_code_act_session_examples,
)

# ---------------------------------------------------------------------------
# Static prompt content (inlined rather than wrapped in trivial functions)
# ---------------------------------------------------------------------------

_FUNCTION_AND_GUIDANCE_LIBRARY = textwrap.dedent("""
    ### Function & Guidance Library

    You have access to two complementary discovery systems:

    * **FunctionManager** — stores concrete, reusable function implementations
      (the building blocks). Search results include a `guidance_ids` field
      linking to related guidance entries.
    * **GuidanceManager** — stores high-level guidance on composing functions
      together to accomplish broader tasks (the recipes / playbooks). Search
      results include `function_ids` pointing back to concrete implementations.

    Always search **both** before writing new code with raw `primitives.*`
    calls:

    1. `FunctionManager_search_functions` — find existing implementations
    2. `GuidanceManager_search_guidance` — find compositional guidance and
       workflows
    3. If a function exists, call it in `execute_code`; if guidance exists,
       follow its workflow
    4. Only fall back to raw `primitives.*` if neither library has relevant
       entries

    #### Function Execution Modes

    Functions support execution mode overrides independent of the session's
    `state_mode`:

    | Mode | Syntax | Behavior |
    |------|--------|----------|
    | **stateful** (default) | `await func(...)` | Function's internal state persists across calls |
    | **stateless** | `await func.stateless(...)` | Fresh environment, no inherited state |
    | **read_only** | `await func.read_only(...)` | Sees current state, changes discarded |
""").strip()

_DISCOVERY_FIRST_POLICY = textwrap.dedent("""
    ### Discovery-First Policy (Active)

    A tool policy is enforced that **requires** you to call both
    `FunctionManager_search_functions` and `GuidanceManager_search_guidance`
    before any other tools become available. Until both have been called at
    least once, only the FunctionManager and GuidanceManager discovery tools
    are visible to you.

    **Call both on your first turn** — they are independent and can be issued
    as parallel tool calls in a single assistant message. Once both discovery
    calls complete, the full tool set (execute_code, primitives, etc.) unlocks
    automatically.

    This policy exists to ensure you always check the existing function and
    guidance libraries before attempting to solve a task from scratch.
""").strip()

_EXECUTION_RULES = textwrap.dedent("""
    ### Execution Rules

    1. **Session-Based Execution**:
       - All code execution happens via the `execute_code` tool (JSON tool call).
       - **Default is `state_mode="stateless"`** (fresh run; no persistence).
       - Choose `state_mode="stateful"` when you need intermediate variables to persist across multiple calls.
       - Choose `state_mode="read_only"` when you need to use an existing session's state without persisting changes.
       - Use `list_sessions()` / `inspect_state()` to discover and understand active sessions.

    2. **Use `await`**: The execution sandbox is asynchronous. You **MUST** use `await` for any async calls.

    3. **Imports Inside Code**: All necessary imports must be included in the code you provide.

    4. **Pydantic for Structured Data (When Supported)**: If a tool supports structured outputs via a `response_format` or schema, define Pydantic models inside the code and call `model_rebuild()` on the outermost model.

    5. **Sandbox Helpers**: The following helpers are available in `execute_code` Python sessions:

       **Progress Notifications (`notify`)**
       - `notify(payload)` sends a non-blocking progress event (dict) to the outer handle.
       - Treat `primitives.*` calls as long-running by default because they run nested tool loops.
       - For each `primitives.*` call, emit at least one kickoff notification before the call.
       - If you await a primitive result and continue with more work, emit a completion notification with concrete outcome data.
       - If you return a primitive handle directly as the last expression, still emit one kickoff notification before returning the handle.
       - Use notifications at meaningful milestones only (start of a major step, completion of a major step, and measurable progress points).

       **What makes a strong notification**
       - Concrete: include useful details like counts, batch indexes, item names, or completed step names.
       - Specific: report what changed since the last update, not generic activity.
       - Informative: help the user understand remaining work and current status.
       - User-facing: explain progress in plain language the end user can understand.
       - High-level: summarize outcomes and next steps, not internal implementation details.

       **Anti-patterns to avoid**
       - Generic filler text with no signal (for example: "working on it", "still processing", "please wait").
       - Repeating the same update without new information.
       - Over-notifying for trivial operations that complete almost immediately.
       - Skipping notifications around `primitives.*` calls unless there is a clear reason.
       - Dumping low-level internals (stack traces, call IDs, schema/debug metadata) into user progress updates.

       **Example payloads**
       - Progress: `{"type": "progress", "message": "...", "step": 2, "total": 5}`
       - Step completion: `{"type": "step_complete", "step_name": "...", "result_summary": "..."}`
       - Custom: any dict schema that communicates real progress clearly.

       **Display Helper (`display`)**
       - `display(obj)` emits rich output (text or PIL images) to stdout.
       - Images are auto-resized and base64-encoded.
       - Use `display(...)` instead of `print(...)` for image output.

    6. **Error Handling**: If your code produces an error, the traceback will be returned. Read it carefully, correct your code, and try again.

    7. **Final Answer Rule**:
       - When the user's request has been fully addressed, you **MUST** provide the final answer directly as a tool-less assistant message.
       - Do not call a tool to print the final answer.
""").strip()

_STORAGE_DEFERRED_NOTICE = textwrap.dedent("""
    ### Skill Storage Is Handled Separately

    A dedicated skill-consolidation process will run automatically after you
    return your result. It will review your full execution trajectory and
    decide which functions, compositional guidance, or workflows are worth
    storing for future reuse — both as concrete function implementations
    and as high-level guidance on how to compose them.

    This means:
    - **Ignore** any language in the request about "remembering",
      "storing", or "saving" skills, workflows, or functions. That concern
      is fully covered by the follow-up process.
    - **Focus entirely on producing the best possible result.** Do not
      spend any effort on persistence, storage, or skill management.
    - You do not have storage tools available, and you do not need them.
""").strip()


def _build_filesystem_context() -> str:
    from pathlib import Path, PurePosixPath

    from unity.file_manager.settings import get_local_root

    resolved = get_local_root()
    # Display as ~/... when the path is inside the user's home directory.
    # This keeps the prompt stable across environments (critical for LLM
    # response caching) while still being accurate.
    try:
        relative = PurePosixPath(resolved).relative_to(Path.home())
        local_root = f"~/{relative}"
    except ValueError:
        local_root = resolved
    return textwrap.dedent(f"""
        ### Filesystem Context

        Your working directory is `{local_root}`.  This directory **persists
        across every interaction** with the user — files you create today will
        still be here weeks or months from now.  All relative paths resolve
        from this directory.

        | Location | Purpose |
        |----------|---------|
        | `Downloads/` | **Inbound** — user-sent attachments are auto-saved here. |
        | `Outputs/` | **Outbound** — save generated files here (reports, CSVs, images, etc.) so the caller can attach and send them to the user. May be auto-cleared between sessions. |
        | `Screenshots/User/` | Auto-captured frames from the user's screen share. Read-only, cleared between sessions. |
        | `Screenshots/Assistant/` | Auto-captured frames from the assistant's desktop. Read-only, cleared between sessions. |
        | `.env` | Environment secrets managed by SecretManager. |
        | Everything else | Your own persistent workspace — organize however makes sense for the work. |

        **File conventions:**
        - **Inbound**: Attachments arrive at `Downloads/<filename>`.  Reference
          them with relative paths (e.g. `Downloads/report.pdf`).
        - **Outbound**: Save files for the user to `Outputs/` and include the
          relative path in your final answer (e.g. `Outputs/summary.csv`).
        - **Screenshots**: Timestamped JPEGs auto-saved during screen sharing.
          Reference them for programmatic access (image analysis, OCR,
          comparison, etc.) using relative paths
          (e.g. `Screenshots/Assistant/2026-02-16T14-30-45.123456.jpg`).
        - **Stay inside the workspace**: Always use relative paths. Do not
          reference absolute paths outside `{local_root}` (e.g. `/tmp`,
          `/var`).  Everything you need is inside this workspace.

        **When to use the filesystem vs. primitives:**
        Most tasks will not require reading or writing local files.  The
        state manager primitives are the primary way to persist information:
        contacts, knowledge, tasks, skills, guidance, and so on — each with
        purpose-built storage, retrieval, and search.  Do not duplicate what
        primitives already handle (e.g. saving contact details to a .txt
        file, or writing Python functions to local scripts).  The local
        filesystem is better suited for working artifacts: data files being
        processed, intermediate results, or anything that benefits from
        conventional file-based organization.  When you do use it for
        longer-lived material, keep it organized — this workspace will
        accumulate across many interactions.
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
    render_tools_block: Callable,
) -> str:
    """Render signatures for non-primary tools (FM discovery, install, etc.)."""
    if not tools:
        return ""

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
    if not additional_tools:
        return ""

    return (
        f"#### Additional Tools\n"
        f"These tools are called via **structured JSON tool calls**, NOT inside Python code.\n\n"
        f"{render_tools_block(additional_tools)}"
    )


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

        discovery_first = get_code_act_discovery_first_examples()
        if discovery_first:
            parts.append(
                f"### Discovery-First Workflow\n\n{discovery_first}",
            )

        session_examples = get_code_act_session_examples()
        if session_examples:
            parts.append(
                f"### Sessions & Multi-Language Execution\n\n{session_examples}",
            )

    # Each environment provides its own rules, docs, and examples.
    for _ns, env in environments.items():
        env_ctx = env.get_prompt_context()
        if env_ctx and env_ctx.strip():
            parts.append(env_ctx)

    # Cross-environment (mixed) examples when computer tools are available.
    env = environments.get("primitives")
    if env is not None:
        _has_computer = any(
            k.startswith("primitives.computer.") for k in env.get_tools()
        )
        _has_state = any(
            k.startswith("primitives.")
            and not k.startswith("primitives.computer.")
            and not k.startswith("primitives.actor.")
            for k in env.get_tools()
        )
        if _has_computer and _has_state:
            from unity.actor.prompt_examples import get_mixed_examples

            mixed = get_mixed_examples()
            if mixed and mixed.strip():
                parts.append(f"### Mixed-Mode Examples\n\n{mixed}")

    return "\n\n---\n\n".join(p for p in parts if p and p.strip()).strip()


def build_code_act_prompt(
    *,
    environments: Mapping[str, "BaseEnvironment"],
    tools: Optional[Dict[str, Callable]] = None,
    can_store: bool = False,
    guidelines: Optional[str] = None,
    discovery_first_policy: bool = False,
) -> str:
    """Build the system prompt for the CodeActActor.

    Assembles prompt sections in a fixed order, skipping sections that
    don't apply to the current configuration. This is intentionally a
    pure prompt builder (no side effects).

    Parameters
    ----------
    discovery_first_policy:
        When ``True``, appends guidance explaining the discovery-first tool
        policy (both FM and GM must be called before other tools unlock).
    """
    from unity.common.prompt_helpers import render_tools_block

    has_execute_code = bool(tools and "execute_code" in tools)
    has_fm_tools = bool(
        tools and any(str(k).startswith("FunctionManager_") for k in tools.keys()),
    )
    has_gm_tools = bool(
        tools and any(str(k).startswith("GuidanceManager_") for k in tools.keys()),
    )

    additional_tools_block = _build_additional_tools_block(
        tools=tools,
        render_tools_block=render_tools_block,
    )

    rules_and_examples = _build_code_act_rules_and_examples(
        environments=environments,
        has_execute_code=has_execute_code,
    )

    parts: list[str] = []

    if has_execute_code:
        parts.append(
            "### Role\n\n"
            "You are an expert agent that solves tasks by writing and executing code. "
            "Your primary tool is a multi-language, multi-session execution environment "
            "for running Python and shell code with access to injected tool domains.",
        )

        if guidelines:
            parts.append(
                f"### Guidelines\n\n"
                f"Follow these guidelines throughout this session:\n\n"
                f"{guidelines}",
            )

        parts.append(_build_filesystem_context())

        primary_names = [
            "execute_code",
            "list_sessions",
            "inspect_state",
            "close_session",
            "close_all_sessions",
        ]
        primary_tools = {k: tools[k] for k in primary_names if k in tools}
        primary_sigs = _build_tool_signatures(primary_tools) if primary_tools else "{}"

        tools_section = (
            "### Tools\n\n"
            "#### Execution & Session Tools\n"
            "These tools are called via **structured JSON tool calls**, NOT inside Python code.\n\n"
            f"```json\n{primary_sigs}\n```"
        )
        if additional_tools_block:
            tools_section += f"\n\n{additional_tools_block}"
        parts.append(tools_section)

        parts.append(_EXECUTION_RULES)

        if has_fm_tools or has_gm_tools:
            parts.append(_FUNCTION_AND_GUIDANCE_LIBRARY)
            if discovery_first_policy:
                parts.append(_DISCOVERY_FIRST_POLICY)

        if can_store:
            parts.append(_STORAGE_DEFERRED_NOTICE)

        if rules_and_examples:
            parts.append(rules_and_examples)

    else:
        parts.append(
            "### Role\n\n"
            "You are an expert agent that solves tasks by discovering and executing "
            "pre-stored functions from a function library. "
            "You do NOT write or execute arbitrary code. Instead, you use the "
            "FunctionManager discovery tools to find relevant stored functions, "
            "then invoke them via `execute_function`.",
        )

        if guidelines:
            parts.append(
                f"### Guidelines\n\n"
                f"Follow these guidelines throughout this session:\n\n"
                f"{guidelines}",
            )

        workflow = (
            "### Workflow\n\n"
            "1. **Discover** stored functions using `FunctionManager_search_functions`,\n"
            "   `FunctionManager_filter_functions`, or `FunctionManager_list_functions`.\n"
            "2. **Pick** the best match by name from the search results.\n"
            "3. **Execute** it via `execute_function(function_name=..., call_kwargs=...)`.\n"
            "4. If no matching function exists, report that clearly — do NOT attempt to\n"
            "   write or compose code yourself."
        )
        if additional_tools_block:
            workflow += f"\n\n{additional_tools_block}"
        parts.append(workflow)

        if rules_and_examples:
            parts.append(rules_and_examples)

    return "\n\n".join(p for p in parts if p and p.strip())
