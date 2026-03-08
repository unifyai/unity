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

    You have access to two complementary systems:

    * **FunctionManager** (read-only) — stores concrete, reusable function
      implementations (the building blocks). Search results include a
      `guidance_ids` field linking to related guidance entries.
    * **GuidanceManager** (read + write) — stores procedural how-to
      information: step-by-step instructions, standard operating procedures,
      software usage walkthroughs, and strategies for composing functions
      together. Search results include `function_ids` pointing back to
      concrete implementations.

    Always search **both** before writing new code with raw `primitives.*`
    calls:

    1. `FunctionManager_search_functions` — find existing implementations
    2. `GuidanceManager_search` — find procedural instructions and
       compositional strategies
    3. If a function exists, call it via `execute_function`; if guidance exists,
       follow its procedure
    4. Only fall back to raw `primitives.*` if neither library has relevant
       entries

    #### Writing Guidance

    When the user provides procedural instructions, operating procedures,
    or step-by-step walkthroughs that should be remembered for future use,
    store them directly via `GuidanceManager_add_guidance`. This is
    appropriate when the *act of persisting the guidance is the task itself*
    (e.g. "remember how to log into X", "here are the steps for Y").

    Function storage is handled separately by a post-completion review
    process — do not attempt to store functions during this loop.

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
    `FunctionManager_search_functions` and `GuidanceManager_search`
    before any other tools become available. Until both have been called at
    least once, only the FunctionManager and GuidanceManager read-only
    discovery tools are visible to you.

    **Call both on your first turn** — they are independent and can be issued
    as parallel tool calls in a single assistant message. Once both discovery
    calls complete, the full tool set unlocks automatically — including
    `execute_code`, primitives, and GuidanceManager write tools
    (`GuidanceManager_add_guidance`, `GuidanceManager_update_guidance`,
    `GuidanceManager_delete_guidance`).

    This policy exists to ensure you always check the existing function and
    guidance libraries before attempting to solve a task from scratch.
""").strip()

_EXECUTION_RULES = textwrap.dedent("""
    ### Tool Selection: `execute_function` vs `execute_code`

    **This is the most important decision you make on every turn.**

    | Scenario | Tool |
    |----------|------|
    | Single primitive call (e.g. `primitives.contacts.ask`, `primitives.web.ask`, `primitives.knowledge.update`) | **`execute_function`** |
    | Single stored function call (discovered via FunctionManager) | **`execute_function`** |
    | Multi-step composition, conditional logic, loops, or combining multiple calls with intermediate results | **`execute_code`** |
    | Shell commands (`bash`, `zsh`, `sh`, `powershell`) | **`execute_code`** |

    **Why this matters:** `execute_function` structurally guarantees that
    the returned handle is exposed to the outer loop for steering (ask,
    stop, pause, resume). With `execute_code`, the handle is only adopted
    if it is the last expression — which is easy to break by adding
    prints, notifications, or error handling around the call.

    **Rule of thumb:** If you can express the task as
    `execute_function(function_name="...", call_kwargs={...})`, always
    do so. Only reach for `execute_code` when you genuinely need to
    compose multiple steps or write conditional/iterative logic.

    ### Execution Rules

    1. **Session-Based Execution**:
       - **Default is `state_mode="stateless"`** (fresh run; no persistence).
       - Choose `state_mode="stateful"` when you need intermediate variables to persist across multiple calls.
       - Choose `state_mode="read_only"` when you need to use an existing session's state without persisting changes.
       - Use `list_sessions()` / `inspect_state()` to discover and understand active sessions.

    2. **Use `await`**: The execution sandbox is asynchronous. You **MUST** use `await` for any async calls.

    3. **Imports Inside Code**: All necessary imports must be included in the code you provide.

    4. **Pydantic for Structured Data (When Supported)**: If a tool supports structured outputs via a `response_format` or schema, define Pydantic models inside the code and call `model_rebuild()` on the outermost model.

    5. **Sandbox Helpers** (available only inside `execute_code` Python sessions):

       **Progress Notifications (`notify`)**
       - `notify(payload)` sends a non-blocking progress event (dict) to the outer handle.
       - Notifications are only relevant inside **multi-step `execute_code` blocks**.
         For single-primitive calls, use `execute_function` — the outer loop
         handles progress automatically via the adopted handle.
       - When composing multiple primitives in `execute_code`, emit notifications
         at meaningful milestones (start of a major step, completion, measurable progress).

       **What makes a strong notification**
       - Concrete: include useful details like counts, batch indexes, item names, or completed step names.
       - Specific: report what changed since the last update, not generic activity.
       - Informative: help the user understand remaining work and current status.
       - User-facing: explain progress in plain language the end user can understand.
       - High-level: summarize outcomes and next steps, not internal implementation details.

       **Anti-patterns to avoid**
       - Wrapping a single primitive call in `execute_code` just to add `notify()` around it — use `execute_function` instead.
       - Generic filler text with no signal (for example: "working on it", "still processing", "please wait").
       - Repeating the same update without new information.
       - Over-notifying for trivial operations that complete almost immediately.
       - Dumping low-level internals (stack traces, call IDs, schema/debug metadata) into user progress updates.

       **Example payloads**
       - Progress: `{"type": "progress", "message": "...", "step": 2, "total": 5}`
       - Step completion: `{"type": "step_complete", "step_name": "...", "result_summary": "..."}`
       - Custom: any dict schema that communicates real progress clearly.

       **Display Helper (`display`)**
       - `display(obj)` emits rich output (text or PIL images) to stdout.
       - Images are base64-encoded.
       - Use `display(...)` instead of `print(...)` for image output.

    6. **Error Handling**: If your code produces an error, the traceback will be returned. Read it carefully, correct your code, and try again.

    7. **Final Answer Rule**:
       - When the user's request has been fully addressed, you **MUST** provide the final answer directly as a tool-less assistant message.
       - Do not call a tool to print the final answer.

    8. **Surface Uncertainties in Your Response**:
       - When you encounter ambiguity during execution — mapping
         approximate labels to schema fields, choosing between plausible
         interpretations of source data, making assumptions where
         information was unclear — include a brief **Uncertainties**
         section at the end of your final answer listing the judgment
         calls you were least confident about.
       - This complements (not replaces) clarification requests. If
         ambiguity is a genuine blocker, request clarification as
         normal. But for the many smaller judgment calls you make while
         proceeding, surface them in the response so the user can
         verify and correct if needed.
       - Focus on decisions that could materially affect the output.
         Do not list trivial or obvious choices.
""").strip()

_INCREMENTAL_EXECUTION = textwrap.dedent("""
    ### Incremental Execution

    The right granularity depends on how predictable each step is.

    **Deterministic work** — pure computation, data transforms, file I/O
    with known schemas — can and should run in a single `execute_code`
    block.  Don't fragment code that you are confident will run correctly
    from start to finish.

    **Uncertain interactions** — browser automation, UI clicks, unfamiliar
    APIs, coordinate-based actions, web scraping — should be broken into
    small steps with verification between each.  The more unpredictable
    the outcome, the more incremental you should be.

    Guidelines for uncertain / interactive work:

    1. **One step per call**: Execute one meaningful action, then review
       the output before deciding the next step.

    2. **Stateful sessions**: Use `state_mode="stateful"` so variables,
       session handles, and intermediate results persist across calls.

    3. **Verify before scaling**: Before writing a loop or repeating a
       pattern, execute the body once and confirm the result.  Only
       generalize to iteration after the single case works correctly.

    4. **Read-only for exploration**: Use `state_mode="read_only"` to
       branch off a known-good intermediate state and try alternative
       approaches without risking that state.

    5. **Inspect results**: After each uncertain step, print or display
       key outputs — don't assume success.
""").strip()

_STORAGE_DEFERRED_NOTICE = textwrap.dedent("""
    ### Skill Storage

    You can proactively store reusable skills at any point during execution
    using the `store_skills` tool. This is useful when you have just
    completed a complex subtask and recognize a pattern worth preserving.

    A dedicated skill-consolidation process will also run automatically
    after you return your result, reviewing your full execution trajectory.
    So you are not obligated to call `store_skills` — use it when you judge
    it valuable, not as a routine step.

    When to use `store_skills`:
    - After completing a complex workflow that discovered non-obvious
      configuration or composition strategies.
    - When the user explicitly asks you to remember or store a skill.
    - Before transitioning to a different phase, to capture learnings
      from the current phase.

    When NOT needed:
    - Trivial operations unlikely to be reused.
    - Every single code execution — the automatic post-completion review
      is comprehensive.

    **Guidance is separate**: if the user explicitly asks you to remember
    procedures, instructions, or how-to information, store it directly via
    `GuidanceManager_add_guidance` as part of the current task. `store_skills`
    is for extracting reusable function implementations and compositional
    strategies from the execution trajectory.

    **Before compression**: when the context window is approaching capacity,
    `store_skills` and `compress_context` will be the only tools available.
    If the current trajectory contains unstored skills worth preserving,
    call `store_skills` first (with a specific request describing what to
    store), then `compress_context`. If nothing new is worth storing — or
    you have already called `store_skills` for the valuable parts — go
    straight to `compress_context`.
""").strip()


_EXTERNAL_APP_INTEGRATION = textwrap.dedent("""
    ### External App Integration

    When integrating with external services (cloud storage, communication
    platforms, project management tools, CRMs, accounting software, etc.),
    follow this pattern:

    1. **Check for credentials**: Use `primitives.secrets.ask(...)` to check
       if API credentials, tokens, or keys for the service are already stored.
       If not, inform the caller and explain they can add them via the
       console's Secrets page (Resources → Secrets).

    2. **Install the SDK**: Use `install_python_packages` to install the
       service's official Python SDK (e.g., `google-cloud-storage` for Google
       Cloud, `slack-sdk` for Slack, `boto3` for AWS, `stripe` for Stripe).

    3. **Integrate**: Write Python code that uses the SDK with the stored
       credentials to interact with the service. Credentials are synced to
       environment variables via the `.env` file managed by SecretManager —
       use `os.environ` to access them after confirming their names via
       `primitives.secrets.ask(...)`.

    4. **Store for reuse**: After a successful integration, store reusable
       functions via `store_skills` and document the setup via
       `GuidanceManager_add_guidance` so future interactions can reuse the
       integration without rediscovery.

    **Prefer Python SDKs over CLI tools.** Python packages benefit from full
    environment management (isolated venvs, dependency resolution via
    `install_python_packages`). Shell CLI tools have no equivalent dependency
    management. Most services offer Python SDKs that are more reliable and
    composable for programmatic use.
""").strip()


_FAST_PATH_AWARENESS = textwrap.dedent("""
    ### Fast-Path Awareness

    During interactive screen-share sessions, the outer process may handle
    simple computer actions (browser navigation, clicks, scrolls) via fast
    paths instead of routing them through you.  You will see these as
    interjection messages tagged `[Fast-path request]` and
    `[Fast-path result]`.

    **Your role:** Monitor these interjections and intervene when the fast
    path is out of its depth.  Specifically, escalate via `notify()` when:

    - The fast-path result indicates failure or confusion (e.g. it tried to
      "navigate to Secret Manager" instead of using `primitives.secrets`)
    - The task falls within guidance you have loaded (e.g. a login procedure
      with specific credential handling steps)
    - The task requires capabilities the fast path lacks: stored credentials
      (`${SECRET_NAME}` injection via `type_text`), multi-step workflows,
      or data extraction with structured schemas

    **How to escalate:**

    ```python
    notify({"type": "escalation", "message": "The fast path attempted X "
        "but I have loaded guidance for this — I should handle it directly "
        "using primitives.secrets and the stored login procedure."})
    ```

    After escalating, **proceed with execution** — do not wait for
    permission.  The outer process will see your notification and coordinate
    accordingly.

    **When NOT to intervene:** Simple atomic actions (click, scroll,
    navigate to URL, basic web search) that complete successfully are
    working as intended.  Only escalate when the fast path is clearly
    failing or attempting work beyond its scope.
""").strip()


def _build_filesystem_context() -> str:
    from unity.file_manager.settings import get_local_root

    resolved = get_local_root()
    return textwrap.dedent(f"""
        ### Filesystem Context

        Your working directory is `{resolved}`.  This directory **persists
        across every interaction** with the user — files you create today will
        still be here weeks or months from now.  **Always use full absolute
        paths** (starting with `{resolved}/`) when referencing any file or
        directory.  Never use relative paths.

        | Location | Purpose |
        |----------|---------|
        | `{resolved}/Downloads/` | **Inbound** — user-sent attachments are auto-saved here. |
        | `{resolved}/Outputs/` | **Outbound** — save generated files here (reports, CSVs, images, etc.) so the caller can attach and send them to the user. May be auto-cleared between sessions. |
        | `{resolved}/Screenshots/User/` | Auto-captured frames from the user's screen share. Read-only, cleared between sessions. |
        | `{resolved}/Screenshots/Assistant/` | Auto-captured frames from the assistant's desktop. Read-only, cleared between sessions. |
        | `{resolved}/Screenshots/Webcam/` | Auto-captured frames from the user's webcam. Read-only, cleared between sessions. |
        | `{resolved}/.env` | Environment secrets managed by SecretManager. |
        | Everything else | Your own persistent workspace — organize however makes sense for the work. |

        **File conventions:**
        - **Inbound**: Attachments arrive at `{resolved}/Downloads/<filename>`.
          Reference them with full paths (e.g. `{resolved}/Downloads/report.pdf`).
        - **Outbound**: Save files for the user to `{resolved}/Outputs/` and
          include the full path in your final answer
          (e.g. `{resolved}/Outputs/summary.csv`).
        - **Screenshots**: Timestamped JPEGs auto-saved during screen sharing.
          Reference them for programmatic access (image analysis, OCR,
          comparison, etc.) using full paths
          (e.g. `{resolved}/Screenshots/Assistant/2026-02-16T14-30-45.123456.jpg`).
        - **Stay inside the workspace**: Always use full absolute paths
          rooted under `{resolved}/`.  Do not reference paths outside this
          workspace (e.g. `/tmp`, `/var`).  Everything you need is inside
          this workspace.

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
            "execute_function",
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

        if _has_computer:
            parts.append(_FAST_PATH_AWARENESS)

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
            "execute_function",
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
        parts.append(_INCREMENTAL_EXECUTION)
        parts.append(_EXTERNAL_APP_INTEGRATION)

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
