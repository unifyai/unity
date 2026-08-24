"""
Tests for CodeActActor prompt builder quality.

These tests are intentionally "high-signal string assertions" rather than
snapshot tests. They verify that:
- The prompt teaches the JSON-call convention and defers each tool's
  contract to its schema (the loop renders every callable's docstring and
  signature into the tool list riding each request) — no second rendering.
- The prompt contains diverse examples: sessions, computer, primitives, mixed.
- The prompt contains no legacy `execute_python_code` references.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from unify.actor.code_act_actor import CodeActActor
from unify.actor.prompt_builders import build_code_act_prompt
from unify.session_details import SESSION_DETAILS, AssistantDetails


@pytest.fixture()
def desktop_entitled_assistant():
    """A session whose assistant holds the managed Computer Use add-on.

    The computer prompt sections are only rendered for an entitled
    assistant; without this, ambient session details default to
    unentitled and the environment context is the unavailability stub.
    """
    original = SESSION_DETAILS.assistant
    SESSION_DETAILS.assistant = AssistantDetails(
        desktop_mode="ubuntu",
        managed_desktop_status="active",
    )
    yield SESSION_DETAILS.assistant
    SESSION_DETAILS.assistant = original


class _DummyEnv:
    """Minimal environment stub for build_code_act_prompt (prompt-context only)."""

    def __init__(self, prompt_context: str):
        self._prompt_context = prompt_context

    def get_prompt_context(self) -> str:
        return self._prompt_context

    def get_tools(self) -> dict:
        return {}


class _DummyToolEnv(_DummyEnv):
    """Minimal environment stub with configurable tool names."""

    def __init__(self, prompt_context: str, tools: dict[str, Any]):
        super().__init__(prompt_context)
        self._tools = tools

    def get_tools(self) -> dict:
        return self._tools


def _real_envs_mixed() -> Mapping[str, Any]:
    """Real environments that produce self-contained prompt context."""
    from unify.function_manager.primitives import ComputerPrimitives
    from unify.actor.environments.computer import ComputerEnvironment
    from unify.actor.environments.state_managers import StateManagerEnvironment
    from unify.actor.environments.base import _CompositeEnvironment

    cp = ComputerPrimitives(computer_mode="mock")
    composite = _CompositeEnvironment(
        [
            ComputerEnvironment(cp),
            StateManagerEnvironment(),
        ],
    )
    return {"primitives": composite}


@pytest.mark.timeout(30)
def test_code_act_prompt_defers_tool_contracts_to_schemas_and_no_legacy_name():
    """The prompt teaches the JSON-call convention only; each tool's contract
    lives in its schema (docstring → description via the loop), never in a
    second in-prompt rendering of signatures/docstrings."""
    actor = CodeActActor()
    tools = dict(actor.get_tools("act"))
    prompt = build_code_act_prompt(
        environments=_real_envs_mixed(),
        tools=tools,
    )

    assert "execute_python_code" not in prompt
    assert "### Tools" in prompt
    assert "structured JSON tool calls" in prompt
    assert "is its schema" in prompt

    # No second rendering of the contracts the tool list already carries.
    assert '"signature":' not in prompt
    assert "#### Execution & Session Tools" not in prompt
    assert "#### Additional Tools" not in prompt
    assert "Tools (name → argspec):" not in prompt

    # The contracts still reach the model: the loop converts each callable's
    # docstring into its schema description on every request.
    import inspect as _inspect

    from unify.common.prompt_helpers import unwrap_tool_callable

    for name in (
        "execute_function",
        "execute_code",
        "list_sessions",
        "inspect_state",
        "close_session",
        "close_all_sessions",
    ):
        assert name in tools
        assert _inspect.getdoc(unwrap_tool_callable(tools[name]))
    ec_doc = _inspect.getdoc(unwrap_tool_callable(tools["execute_code"])) or ""
    assert "Execute arbitrary code in a specified language and state mode." in ec_doc
    assert "multi-step composition" in ec_doc.lower()

    # Selection policy (not contract) stays inline in the prompt.
    assert "multi-step composition" in prompt.lower()


@pytest.mark.timeout(30)
def test_code_act_prompt_includes_computer_primitives_and_selection_contracts(
    desktop_entitled_assistant,
):
    actor = CodeActActor()
    try:
        prompt = build_code_act_prompt(
            environments=_real_envs_mixed(),
            tools=dict(actor.get_tools("act")),
        )
    finally:
        pass

    assert "list_sessions" in prompt

    assert "Viewing Computer State" in prompt
    # Computer method documentation (from environment's get_prompt_context)
    assert "primitives.computer" in prompt.lower()
    assert "navigate" in prompt
    assert "act" in prompt
    assert "observe" in prompt

    # State-manager guidance (primitives)
    assert "### State Manager Rules" in prompt
    assert "returning the handle as the last expression" in prompt
    assert "immediate in-code composition" in prompt
    assert "neutral or uncertain" in prompt.lower()
    assert "default to returning the handle" in prompt.lower()

    # Handle-adoption contract stays inline. (call_kwargs exact typing is a
    # mechanism fact asserted on the execute_function docstring instead.)
    assert "### Tool Selection: `execute_function` vs `execute_code`" in prompt
    assert "last expression" in prompt
    assert "never consume a handle" in prompt


@pytest.mark.timeout(30)
def test_code_act_prompt_includes_task_guidance_only_with_task_primitives():
    prompt_with_tasks = build_code_act_prompt(
        environments={
            "primitives": _DummyToolEnv(
                "Task primitives are available.",
                {"primitives.tasks.update": object()},
            ),
        },
        tools={},
    )
    prompt_without_tasks = build_code_act_prompt(
        environments={
            "primitives": _DummyToolEnv(
                "Only contact primitives are available.",
                {"primitives.contacts.ask": object()},
            ),
        },
        tools={},
    )

    # The section is a compact contract stub; depth (task types, entrypoint
    # conventions, binding resolution) lives in the platform/durable-tasks
    # guidance entry behind the consult pointer.
    assert "Durable Scheduled And Triggered Tasks" in prompt_with_tasks
    assert "primitives.tasks.update" in prompt_with_tasks
    assert "live (armed)" in prompt_with_tasks
    assert "status armed AND trigger bindings" in prompt_with_tasks
    # When the primitive confirmed the fields, the actor must report them —
    # never claim failure or uncertainty.
    assert "claim failure or uncertainty" in prompt_with_tasks
    # Consult pointer phrased to hit the platform/durable-tasks title.
    assert "creating verifying arming and running" in prompt_with_tasks
    assert "Durable Scheduled And Triggered Tasks" not in prompt_without_tasks
    # The prompt must not carry incident-specific workaround phrasing.
    assert "loop to re-discover the" not in prompt_with_tasks
    assert "person-like token" not in prompt_with_tasks
    assert "Never fake" not in prompt_with_tasks


@pytest.mark.timeout(30)
def test_code_act_prompt_teaches_refresh_token_oauth_helper():
    actor = CodeActActor()
    prompt = build_code_act_prompt(
        environments={},
        tools=dict(actor.get_tools("act")),
    )

    assert "def query_llm(" in prompt
    assert "def list_llms(provider: 'str | None' = None) -> 'list[str]'" in prompt
    assert (
        "def get_oauth_access_token(provider: str, *, "
        "min_ttl_seconds: int = 300) -> str"
    ) in prompt
    # The section is a contract stub — the handle semantics stay inline,
    # the full procedure defers to help(get_oauth_access_token).
    assert "capability handle" in prompt
    assert "Do not print, log, return, or store this handle." in prompt
    assert "MICROSOFT_GRAPH_BASE" in prompt
    assert "help(get_oauth_access_token)" in prompt

    # ...while the docstring behind help() carries the full procedure the
    # stub defers to.
    import inspect as _inspect

    from unify.common.runtime_oauth import get_oauth_access_token

    doc = _inspect.getdoc(get_oauth_access_token) or ""
    assert 'get_oauth_access_token("microsoft")' in doc
    assert 'get_oauth_access_token("google")' in doc
    assert "MICROSOFT_GRAPH_BASE" in doc
    assert "GOOGLE_GRANTED_SCOPES" in doc


@pytest.mark.timeout(30)
def test_code_act_prompt_routes_comms_and_defers_method_docs_to_search():
    """The comms manager surfaces through the routing overview and the
    discovery/introspection pointer; per-method docs are not inlined."""
    from unify.actor.environments.state_managers import StateManagerEnvironment
    from unify.function_manager.primitives import PrimitiveScope, Primitives

    actor = CodeActActor()
    env = StateManagerEnvironment(
        Primitives(primitive_scope=PrimitiveScope.single("comms")),
    )
    prompt = build_code_act_prompt(
        environments={"primitives": env},
        tools=dict(actor.get_tools("act")),
    )

    # Routing overview names the manager and teaches when to reach for it.
    # (The overview is description + use-when only; per-manager examples do
    # not render, with canvas as the one exception.)
    assert "primitives.comms" in prompt
    assert "Assistant-Owned Communication" in prompt
    assert "proactively contact people" in prompt
    assert "Text Alice that the meeting moved" not in prompt

    # Method docs live behind search + introspection, not inline; the
    # discovery/introspection bridge lives in the static Sandbox Environment
    # table, not in a per-environment Method Discovery section.
    assert "### Method Discovery & Introspection" not in prompt
    assert "### Sandbox Environment" in prompt
    assert "### Method Reference (core)" not in prompt
    assert "FunctionManager_search_functions" in prompt
    assert "inspect.signature" in prompt
    assert "do not guess signatures" in prompt
    assert "help(primitives.<manager>.<method>)" in prompt
    # Primitives are searchable — the prompt must never claim they are
    # excluded from FunctionManager search.
    assert "excluded from FunctionManager" not in prompt
    assert ".send_whatsapp" not in prompt
    assert ".send_teams_message" not in prompt


@pytest.mark.timeout(30)
def test_incremental_execution_present_and_execution_rules_not_duplicated():
    """Incremental Execution section is present; _EXECUTION_RULES appears exactly once."""
    actor = CodeActActor()
    prompt = build_code_act_prompt(
        environments=_real_envs_mixed(),
        tools=dict(actor.get_tools("act")),
    )

    assert "### Incremental Execution" in prompt
    assert "Verify before scaling" in prompt
    assert "Read-only for exploration" in prompt

    exec_rules_marker = "### Tool Selection: `execute_function` vs `execute_code`"
    assert (
        prompt.count(exec_rules_marker) == 1
    ), f"Expected _EXECUTION_RULES exactly once, found {prompt.count(exec_rules_marker)}"


@pytest.mark.timeout(30)
def test_code_act_prompt_platform_capabilities_index(desktop_entitled_assistant):
    """Platform self-knowledge depth lives in builtin guidance; the prompt
    carries a static capabilities index with one consult-path line per
    guidance-backed or gated domain, rendered only when discovery tools
    exist."""
    actor = CodeActActor()
    prompt = build_code_act_prompt(
        environments=_real_envs_mixed(),
        tools=dict(actor.get_tools("act")),
    )

    assert "### Platform Self-Knowledge" not in prompt
    assert "### Platform Capabilities Index" in prompt
    # One consult-path line per guidance-backed or gated domain.
    assert "answering questions about the Unify platform" in prompt
    assert "integrating external apps credentials OAuth envelopes" in prompt
    assert "help(get_oauth_access_token)" in prompt
    assert "help(query_llm)" in prompt
    assert "choosing between overlapping state managers" in prompt
    assert "driving desktops and reading screens" in prompt
    assert "unlock macOS user desktop" in prompt
    # Durable-tasks depth lives in guidance — the index names the path.
    assert "creating verifying arming and running" in prompt
    assert "help(primitives.ingestion.submit)" in prompt
    # Confidentiality one-liner stays inline.
    assert "never reproduce their contents" in prompt

    # Without discovery tools, the index must not point at tools the
    # assistant cannot call.
    all_tools = dict(actor.get_tools("act"))
    no_discovery = {
        k: v
        for k, v in all_tools.items()
        if not str(k).startswith(
            ("FunctionManager_", "GuidanceManager_", "KnowledgeManager_"),
        )
    }
    prompt_no_discovery = build_code_act_prompt(
        environments={},
        tools=no_discovery,
    )
    assert "### Platform Capabilities Index" not in prompt_no_discovery


@pytest.mark.timeout(30)
def test_external_and_oauth_sections_gate_independently():
    """The two config gates (integration packages; workspace OAuth) are
    independent, default to today's always-on behavior, and a gated-off
    section keeps its capabilities-index line."""
    actor = CodeActActor()
    tools = dict(actor.get_tools("act"))

    default_prompt = build_code_act_prompt(environments={}, tools=tools)
    assert "### External App Integration" in default_prompt
    assert "### OAuth Access Token Helper" in default_prompt

    no_integrations = build_code_act_prompt(
        environments={},
        tools=tools,
        include_external_app_integration=False,
    )
    assert "### External App Integration" not in no_integrations
    assert "### OAuth Access Token Helper" in no_integrations
    assert "integrating external apps credentials OAuth envelopes" in no_integrations

    no_oauth = build_code_act_prompt(
        environments={},
        tools=tools,
        include_oauth_helper=False,
    )
    assert "### OAuth Access Token Helper" not in no_oauth
    assert "### External App Integration" in no_oauth
    assert "help(get_oauth_access_token)" in no_oauth


@pytest.mark.timeout(30)
def test_policy_contracts_have_a_guarded_destination(desktop_entitled_assistant):
    """Every guarded contract has an explicit destination. The ten policy
    contracts stay inline in the prompt (retrieval may miss and gates may
    not fire); the two mechanism contracts are asserted at their docstring
    destinations — `call_kwargs` exact typing in the `execute_function`
    docstring (which feeds the tool's JSON schema on every request), verbatim
    identifier copying in the `TaskScheduler.ask`/`.update` docstrings
    behind help()/FM search."""
    actor = CodeActActor()
    prompt = build_code_act_prompt(
        environments=_real_envs_mixed(),
        tools=dict(actor.get_tools("act")),
        discovery_first_policy=True,
        include_external_app_integration=False,
        include_oauth_helper=False,
    )

    # 1. Confirmation-before-external-send / consequential-action contract.
    assert "confirmation for write, destructive, bulk-export, or sensitive" in prompt

    # 2. User-desktop file-harvesting prohibition (2-line contract).
    assert "primitives.computer.user_desktop.files" in prompt
    assert "Never retrieve their file content" in prompt

    # 3. Notification contract.
    assert "send_notification" in prompt
    assert "verifiably finished" in prompt

    # 4. Tool-less final answer + Uncertainties section.
    assert "tool-less assistant message" in prompt
    assert "Uncertainties" in prompt

    # 5. Data provenance — no model knowledge formatted as sourced data.
    assert "never present model knowledge as sourced" in prompt
    assert "never formatted as sourced records" in prompt

    # 6. Verify-against-evidence.
    assert "a step that ran is not a step that" in prompt

    # 7. Proactive clarification.
    assert "prefer asking over" in prompt

    # 8. Handle-adoption / steering contract.
    assert "structurally guarantees" in prompt
    assert "last expression" in prompt
    assert "never consume a handle" in prompt

    # 9. Browser-session reuse (visible browser reattach, ~4 lines).
    assert "reattach by" in prompt
    assert "get_session" in prompt

    # 10. Discovery-first CORRECT procedure.
    assert "The CORRECT procedure:" in prompt
    assert "first tool-calling assistant message" in prompt

    # Mechanism destination A: call_kwargs exact typing lives in the
    # execute_function docstring (identifier-literalism defense).
    import inspect as _inspect

    from unify.common.prompt_helpers import unwrap_tool_callable

    execute_function = unwrap_tool_callable(
        dict(actor.get_tools("act"))["execute_function"],
    )
    ef_doc = _inspect.getdoc(execute_function) or ""
    assert "Values keep the callee's own" in ef_doc
    assert '``{"max_results": 5}``' in ef_doc
    assert '``{"max_results": "5"}``' in ef_doc
    # The docstring becomes the tool's schema description on every request,
    # so the contract still reaches the model inline — without a second
    # rendering in the prompt itself.
    assert "Values keep the callee's own" not in prompt

    # Mechanism destination B: verbatim identifier copying lives in the
    # TaskScheduler docstrings surfaced by help() and FM search.
    from unify.task_scheduler.task_scheduler import TaskScheduler

    for method in (TaskScheduler.update, TaskScheduler.ask):
        doc = _inspect.getdoc(method) or ""
        assert "copy them verbatim into ``text``" in doc
        assert "literal string" in doc


@pytest.mark.timeout(30)
def test_code_act_prompt_teaches_execution_surfaces_with_user_desktop_caution():
    """The surface section documents local/assistant/user desktops and cautions
    that the user desktop is a personal machine requiring consent."""
    actor = CodeActActor()
    prompt = build_code_act_prompt(
        environments=_real_envs_mixed(),
        tools=dict(actor.get_tools("act")),
    )

    assert "### Execution Surface" in prompt
    assert 'surface="local"' in prompt
    assert 'surface="assistant_desktop"' in prompt
    assert 'surface="user_desktop"' in prompt

    # User-desktop caution / consent posture.
    assert "personal machine" in prompt
    assert "confirm with the user" in prompt
    assert "Console consent" in prompt
    assert "can be revoked mid-run" in prompt

    # Remote surfaces remain stateless one-shots.
    assert "stateless one-shots" in prompt


@pytest.mark.timeout(30)
def test_python_first_principle_present():
    """The Python-first principle is included in the execution rules."""
    actor = CodeActActor()
    prompt = build_code_act_prompt(
        environments=_real_envs_mixed(),
        tools=dict(actor.get_tools("act")),
    )

    assert "Python-first principle" in prompt
    assert "install_python_packages" in prompt
    assert "install_shell_packages" in prompt


@pytest.mark.timeout(30)
def test_code_act_prompt_includes_reasoning_helper_decision_guidance():
    actor = CodeActActor()
    prompt = build_code_act_prompt(
        environments=_real_envs_mixed(),
        tools=dict(actor.get_tools("act")),
    )

    # The full query_llm teaching (good uses, anti-patterns, model choice)
    # lives in query_llm.__doc__ behind help(); the prompt keeps a compact
    # when-to-use block, folded into the Sandbox Environment section.
    assert "### Deterministic Code With LLM-Native Semantic Processing" not in prompt
    assert "### Sandbox Environment" in prompt
    assert "async def query_llm(" in prompt
    assert "def list_llms(provider: 'str | None' = None) -> 'list[str]'" in prompt
    assert "unstructured -> structured" in prompt
    assert "unstructured -> unstructured" in prompt
    assert "draft, respond, rewrite" in prompt
    assert "keep it deterministic" in prompt
    assert "Semantic downgrades are bugs" in prompt
    assert "templates pretending" in prompt
    assert "keyword ladders" in prompt
    assert "keep the query_llm(...) call" in prompt
    assert "inside the stored function" in prompt
    # Statelessness and the code -> query_llm -> sub-agent dial are prompt-level
    # doctrine; the full frame lives in the storage-review prompt.
    assert "stateless — a memoryless" in prompt
    assert "never in the model" in prompt
    assert "a dial, not a mode switch" in prompt
    assert "lowest notch that preserves the judgment" in prompt
    assert "whose plan must be discovered at runtime" in prompt
    # The consult path replaces the inline model-selection tables.
    assert "help(query_llm)" in prompt
    assert "Choosing A Model For `query_llm(...)`" not in prompt
    assert "Artificial Analysis (https://artificialanalysis.ai/)" not in prompt
    assert "LLM Query Helpers: `query_llm(...)` And `list_llms(...)`" not in prompt

    # ...while help(query_llm) still carries the full contract.
    import inspect as _inspect

    from unify.common.reasoning import query_llm

    doc = _inspect.getdoc(query_llm) or ""
    assert "Choosing A Model For `query_llm(...)`" in doc
    assert "Artificial Analysis (https://artificialanalysis.ai/)" in doc
    assert "ARC Prize leaderboard: https://arcprize.org/leaderboard" in doc


@pytest.mark.timeout(30)
def test_code_act_prompt_includes_compressed_reasoning_contracts():
    """The semantic-vs-deterministic teaching renders as compact prose."""
    actor = CodeActActor()
    prompt = build_code_act_prompt(
        environments=_real_envs_mixed(),
        tools=dict(actor.get_tools("act")),
    )

    assert "count unread emails from Alice" in prompt
    assert "do not call query_llm(...)" in prompt
    assert "deterministic pre-filter" in prompt


@pytest.mark.timeout(30)
def test_code_act_prompt_does_not_make_reason_mandatory_for_every_loop():
    actor = CodeActActor()
    prompt = build_code_act_prompt(
        environments=_real_envs_mixed(),
        tools=dict(actor.get_tools("act")),
    )

    assert "use" in prompt
    assert (
        "``query_llm(...)`` only where meaning-based judgment is doing real work"
        in prompt
    )
    assert "exact logic is enough" in prompt
    assert "freely mix deterministic substeps and semantic substeps" in prompt
    assert "do not call query_llm(...)" in prompt
    assert "Use query_llm(...) for every loop" not in prompt
    assert "Always call query_llm(...)" not in prompt


@pytest.mark.timeout(30)
def test_discovery_first_guidance_separates_search_from_execution_choice():
    """Discovery-first should not imply that a missing library hit means execute_code."""
    actor = CodeActActor()
    prompt = build_code_act_prompt(
        environments=_real_envs_mixed(),
        tools=dict(actor.get_tools("act")),
        discovery_first_policy=True,
    )

    assert "Discovery index scope" in prompt
    # Primitives are searchable now; only prompt-documented callables
    # (computer methods, prompt-injected functions/guidance) stay out.
    assert "the built-in `primitives.*` catalogue" in prompt
    assert "they never appear in search" in prompt
    assert "Search is a discovery step" in prompt
    assert "not an execution decision." in prompt
    assert (
        "if the request or discovery step already identifies one exact function"
        in prompt
    )


@pytest.mark.timeout(30)
def test_discovery_first_examples_no_longer_model_execute_code_as_default_fallback():
    """Discovery-first examples should not teach no-hit => write custom code."""
    actor = CodeActActor()
    prompt = build_code_act_prompt(
        environments=_real_envs_mixed(),
        tools=dict(actor.get_tools("act")),
        discovery_first_policy=True,
    )

    assert (
        "If no function exists, THEN fall back to composing with primitives directly in Python."
        not in prompt
    )
    assert (
        "FunctionManager-discovered functions are available in all execute_code calls"
        not in prompt
    )
    assert "Use `execute_code` for *everything* (Python + shell)" not in prompt
    assert (
        "if one exact function or primitive call is enough, use execute_function"
        in prompt
    )


@pytest.mark.timeout(30)
def test_python_first_principle_absent_without_execute_code():
    """The principle is absent when execute_code is not available."""
    actor = CodeActActor()
    all_tools = dict(actor.get_tools("act"))
    tools = {k: v for k, v in all_tools.items() if k != "execute_code"}
    prompt = build_code_act_prompt(
        environments={},
        tools=tools,
    )

    assert "Python-first principle" not in prompt


@pytest.mark.timeout(30)
def test_custom_environment_prompt_context_included():
    """Custom environments (not computer_primitives/primitives) should have their
    prompt context included in the generated prompt."""
    actor = CodeActActor()

    custom_marker = "### Custom Widget Tools\n- `widget.create(name)` — create a widget"
    envs: Mapping[str, Any] = {
        "primitives": _DummyEnv(
            "### State manager tools\n- `await primitives.contacts.ask(...)`",
        ),
        "widget_tools": _DummyEnv(custom_marker),
    }

    prompt = build_code_act_prompt(
        environments=envs,
        tools=dict(actor.get_tools("act")),
    )

    assert custom_marker in prompt


@pytest.mark.timeout(30)
def test_multiple_custom_environments_all_included():
    """Multiple custom environments should each have their prompt context included."""
    actor = CodeActActor()

    marker_a = "### Alpha Environment\nAlpha-specific guidance for the LLM."
    marker_b = "### Beta Environment\nBeta-specific guidance for the LLM."
    envs: Mapping[str, Any] = {
        "alpha": _DummyEnv(marker_a),
        "beta": _DummyEnv(marker_b),
    }

    prompt = build_code_act_prompt(
        environments=envs,
        tools=dict(actor.get_tools("act")),
    )

    assert marker_a in prompt
    assert marker_b in prompt


@pytest.mark.timeout(30)
def test_custom_environment_empty_prompt_context_excluded():
    """Custom environments returning empty prompt context should not inject noise."""
    actor = CodeActActor()

    envs: Mapping[str, Any] = {
        "empty_env": _DummyEnv(""),
        "whitespace_env": _DummyEnv("   \n  "),
    }

    prompt = build_code_act_prompt(
        environments=envs,
        tools=dict(actor.get_tools("act")),
    )

    # The prompt should still be valid (no crash) and not contain stray whitespace blocks.
    assert "empty_env" not in prompt
    assert "whitespace_env" not in prompt


@pytest.mark.timeout(30)
def test_computer_environment_prompt_context_from_registry(
    desktop_entitled_assistant,
):
    """ComputerEnvironment should derive prompt context from registry."""
    from unify.function_manager.primitives import ComputerPrimitives
    from unify.actor.environments.computer import ComputerEnvironment

    cp = ComputerPrimitives(computer_mode="mock")
    env = ComputerEnvironment(cp)
    context = env.get_prompt_context()

    assert context  # Non-empty
    assert "primitives.computer" in context.lower()
    # All dynamic methods appear in the name index
    assert "navigate" in context
    assert "act" in context
    assert "observe" in context
    assert "query" in context
    assert "get_links" in context
    assert "get_content" in context
    # The name index carries signatures + one-line summaries and points at
    # help()/inspect.signature — full docstring dumps are gone.
    assert "### Computer Method Reference (name index)" in context
    assert "help(primitives.computer.desktop.<method>)" in context
    assert "\nParameters\n" not in context and "  Parameters" not in context
    # Managed desktop layout anchors (full contract in dedicated test)
    assert "HOME=/Unity" in context
    assert "/Unity/Downloads" in context


@pytest.mark.timeout(30)
def test_computer_environment_managed_desktop_filesystem_paths(
    desktop_entitled_assistant,
):
    """Computer prompt teaches VM home/Downloads layout, not /home/unityuser."""
    from unify.function_manager.primitives import ComputerPrimitives
    from unify.actor.environments.computer import ComputerEnvironment

    context = ComputerEnvironment(
        ComputerPrimitives(computer_mode="mock"),
    ).get_prompt_context()

    assert "#### Managed desktop filesystem" in context
    assert "HOME=/Unity" in context
    assert "/Unity/Downloads" in context
    assert "/Unity/Local" in context
    assert "Unity > Downloads" in context
    assert "not `/home/unityuser`" in context
    assert "/home/unityuser/Unity/Downloads" in context
    assert "on this VM desktop only" in context
    # Section sits before the your-vs-user desktop split
    assert context.index("#### Managed desktop filesystem") < context.index(
        "### Your Desktop vs. a User's Desktop",
    )


@pytest.mark.timeout(30)
def test_filesystem_context_distinguishes_pod_workspace_from_managed_desktop():
    """Filesystem Context is the pod Local workspace, distinct from /Unity."""
    from unify.actor.prompt_builders import _build_filesystem_context
    from unify.file_manager.settings import get_local_root

    section = _build_filesystem_context()
    local_root = get_local_root()

    assert "### Filesystem Context" in section
    assert "local (pod) workspace" in section
    assert local_root in section
    assert "/Unity/Downloads" in section
    assert "/Unity/Local" in section
    assert "Managed desktop filesystem" in section
    assert "do not treat them as this pod workspace" in section
    assert "unityuser" in section
    assert "/home/unityuser" in section
    assert "not the desktop home" in section


# ────────────────────────────────────────────────────────────────────────────
# External app integration section
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.timeout(30)
def test_external_app_integration_present():
    """The external app integration section is included when execute_code is available."""
    actor = CodeActActor()
    prompt = build_code_act_prompt(
        environments=_real_envs_mixed(),
        tools=dict(actor.get_tools("act")),
    )

    # The section is a compact contract stub — envelope shape and scope
    # check stay operative inline; depth defers to the
    # platform/integrations-envelopes guidance entry.
    assert "### External App Integration" in prompt
    assert "primitives.secrets.ask" in prompt
    assert "install_python_packages" in prompt
    assert "the **Integrations** tab in the console" in prompt
    # Result-envelope contract.
    assert "`connect_required`" in prompt
    assert "`missing_scope`" in prompt
    assert "no custom retry/sleep" in prompt
    # Scope-check operative contract.
    assert "OAuth scope check" in prompt
    assert "GOOGLE_GRANTED_SCOPES" in prompt
    assert "do not attempt the call" in prompt
    # Consult pointer to the enriched entry.
    assert "integrating external apps credentials OAuth envelopes" in prompt


@pytest.mark.timeout(30)
def test_external_app_integration_absent_without_execute_code():
    """The section is absent when execute_code is not available (discovery-only mode)."""
    actor = CodeActActor()
    all_tools = dict(actor.get_tools("act"))
    tools = {k: v for k, v in all_tools.items() if k != "execute_code"}
    prompt = build_code_act_prompt(
        environments={},
        tools=tools,
    )

    assert "### External App Integration" not in prompt
    assert "Discovery index scope" in prompt
    assert "prompt-documented callable by exact name" in prompt


# ────────────────────────────────────────────────────────────────────────────
# Guidelines composition (constructor baseline + per-invocation overlay)
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.timeout(30)
def test_guidelines_neither_specified():
    """No guidelines at all -> no ### Guidelines section in the prompt."""
    actor = CodeActActor()
    prompt = build_code_act_prompt(
        environments={},
        tools=dict(actor.get_tools("act")),
        guidelines=None,
    )
    assert "### Guidelines" not in prompt


@pytest.mark.timeout(30)
def test_guidelines_constructor_only():
    """Constructor-level guidelines appear in a single ### Guidelines section."""
    actor = CodeActActor(guidelines="Always respond in formal English.")
    base = actor._base_guidelines
    effective = "\n\n".join(filter(None, [base, None])) or None

    prompt = build_code_act_prompt(
        environments={},
        tools=dict(actor.get_tools("act")),
        guidelines=effective,
    )
    assert prompt.count("### Guidelines") == 1
    assert "Always respond in formal English." in prompt


@pytest.mark.timeout(30)
def test_guidelines_per_invocation_only():
    """Per-invocation guidelines appear in a single ### Guidelines section."""
    actor = CodeActActor()
    per_invocation = "Check all contact fields."
    effective = (
        "\n\n".join(filter(None, [actor._base_guidelines, per_invocation])) or None
    )

    prompt = build_code_act_prompt(
        environments={},
        tools=dict(actor.get_tools("act")),
        guidelines=effective,
    )
    assert prompt.count("### Guidelines") == 1
    assert "Check all contact fields." in prompt


@pytest.mark.timeout(30)
def test_guidelines_both_compose():
    """Constructor + per-invocation guidelines compose into one ### Guidelines section."""
    actor = CodeActActor(guidelines="Always respond in formal English.")
    per_invocation = "Check all contact fields."
    effective = (
        "\n\n".join(
            filter(None, [actor._base_guidelines, per_invocation]),
        )
        or None
    )

    prompt = build_code_act_prompt(
        environments={},
        tools=dict(actor.get_tools("act")),
        guidelines=effective,
    )
    assert prompt.count("### Guidelines") == 1
    assert "Always respond in formal English." in prompt
    assert "Check all contact fields." in prompt
    # Constructor guidelines come first
    idx_base = prompt.index("Always respond in formal English.")
    idx_overlay = prompt.index("Check all contact fields.")
    assert idx_base < idx_overlay


@pytest.mark.timeout(30)
def test_storage_notice_matches_session_mode():
    """The skill-storage notice must describe the schedule the run actually
    gets: one-shot acts consolidate after the final result; persistent
    sessions consolidate after each completed turn and surface background
    notes the model should act on for repeat deliverables."""
    actor = CodeActActor()
    tools = dict(actor.get_tools("act"))

    one_shot = build_code_act_prompt(
        environments=_real_envs_mixed(),
        tools=tools,
        can_store=True,
    )
    assert "after you return your result" in one_shot
    assert "after each completed turn" not in one_shot

    session = build_code_act_prompt(
        environments=_real_envs_mixed(),
        tools=tools,
        can_store=True,
        persist=True,
    )
    assert "after each completed turn" in session
    assert "after you return your result" not in session
    # The convergence contract: repeat requests are one execution and a
    # report, and amendments edit the stored function in place rather than
    # triggering a fresh replan.
    assert "do not re-derive the procedure inline" in session
    assert "one execution and a report" in session
    assert "`overwrite=True` edit" in session

    without_store = build_code_act_prompt(
        environments=_real_envs_mixed(),
        tools=tools,
        can_store=False,
        persist=True,
    )
    assert "Skill Storage" not in without_store
