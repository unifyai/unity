"""
Comprehensive tests for Actor prompt builders.

High-level intent:
- **Environment-awareness**: prompts adapt to active namespaces (browser vs primitives vs mixed),
  avoid browser-only assumptions when no browser env is present, and preserve backward
  compatibility when `environments=None`.
- **Dynamic context capture**: prompts correctly include (or omit) execution-time context such
  as call stacks, scoped source context, interactions (with optional agent logs), polymorphic
  evidence, idempotency cache summaries, and steerable pane snapshots.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping
from unittest.mock import AsyncMock

import pytest

from unity.actor.environments.base import BaseEnvironment, ToolMetadata
from unity.actor.hierarchical_actor import (
    HierarchicalActorHandle,
    _HierarchicalHandleState,
)
from unity.common.async_tool_loop import SteerableToolHandle
from unity.actor.prompt_builders import (
    build_ask_prompt,
    build_dynamic_implement_prompt,
    build_initial_plan_prompt,
    build_interjection_prompt,
    build_precondition_prompt,
    build_refactor_prompt,
    build_verification_prompt,
)


# ============================================================================
# Shared test stubs / helpers
# ============================================================================


class _DummyEnvironment:
    """Minimal environment stub for prompt builder tests (prompt-context only)."""

    def __init__(self, prompt_context: str = ""):
        self._prompt_context = prompt_context

    def get_prompt_context(self) -> str:
        return self._prompt_context


class _MockEnvironment(BaseEnvironment):
    """Minimal BaseEnvironment implementation for prompt builder tests."""

    def __init__(self, name: str):
        self._name = name

    @property
    def namespace(self) -> str:
        return self._name

    def get_instance(self) -> Any:
        return object()

    def get_tools(self) -> Dict[str, ToolMetadata]:
        return {}

    def get_prompt_context(self) -> str:
        return ""

    async def capture_state(self) -> Dict[str, Any]:
        return {}


async def _dummy_contacts_ask(query: str) -> str:
    return f"asked: {query}"


async def _dummy_contacts_update(instruction: str) -> str:
    return f"updated: {instruction}"


async def _dummy_navigate(url: str) -> None:
    return None


async def _dummy_act(instruction: str) -> None:
    return None


async def _dummy_observe(question: str, response_format: Any = None) -> Any:
    return None


def _tools_mixed() -> Dict[str, Callable[..., Any]]:
    """Return a representative mixed tool surface (browser + primitives)."""

    return {
        "computer_primitives.navigate": _dummy_navigate,
        "computer_primitives.act": _dummy_act,
        "computer_primitives.observe": _dummy_observe,
        "primitives.contacts.ask": _dummy_contacts_ask,
        "primitives.contacts.update": _dummy_contacts_update,
    }


def _environments_mixed() -> Mapping[str, Any]:
    """Return representative mixed environments (browser + primitives) with prompt context."""

    return {
        "computer_primitives": _DummyEnvironment(
            "### Computer tools (`computer_primitives`)",
        ),
        "primitives": _DummyEnvironment(
            "### State manager tools (`primitives`)\n- `await primitives.contacts.ask(...)`\n",
        ),
    }


def _existing_functions_library() -> Dict[str, Any]:
    """Return a representative existing-functions library in the shape expected by prompts."""

    return {
        "trusted_skill": {
            "argspec": "(name: str) -> dict",
            "docstring": "Look up a user by name and return a dict.",
            "implementation": "async def trusted_skill(name: str) -> dict:\n    return {'name': name}\n",
        },
        "other_skill": {
            "argspec": "() -> None",
            "docstring": "Do something useful.",
            "implementation": "def other_skill() -> None:\n    return None\n",
        },
    }


def _scoped_context_str(
    *,
    include_parent: bool = True,
    include_children: bool = True,
) -> str:
    """Return a realistic `scoped_context` string as inserted into prompts."""

    parts: list[str] = []
    if include_parent:
        parts.append(
            "### Parent Function Source\n"
            "```python\n"
            "async def parent_fn():\n"
            '    """Parent function."""\n'
            "    return 'ok'\n"
            "```",
        )
    parts.append(
        "### Current Function Source\n"
        "```python\n"
        "async def current_fn():\n"
        '    """Current function."""\n'
        "    return 'ok'\n"
        "```",
    )
    if include_children:
        parts.append(
            "### Children Source (Functions it may call)\n"
            "```python\n"
            "# Child Function: child_fn\n"
            "async def child_fn(x: int) -> int:\n"
            "    return x + 1\n"
            "```",
        )
    return "\n\n---\n\n".join(parts)


def _images() -> list[Any]:
    """Return AnnotatedImageRef-like objects for test image context."""
    from unity.image_manager.types import AnnotatedImageRef, RawImageRef

    return [
        AnnotatedImageRef(
            raw_image_ref=RawImageRef(image_id=1),
            annotation="Login screen",
        ),
        AnnotatedImageRef(
            raw_image_ref=RawImageRef(image_id=2),
            annotation="Confirmation modal",
        ),
    ]


def _interactions(
    *,
    with_agent_logs: bool,
    include_observe: bool = True,
) -> list[tuple]:
    """Return interactions in the runtime tuple format used by verification prompts."""

    out: list[tuple] = []
    if include_observe:
        if with_agent_logs:
            out.append(
                (
                    "observe",
                    "computer_primitives.observe((('What is on screen?',), {}))",
                    "{'title': 'Home'}",
                    [
                        "Step 1: Capture screenshot",
                        "Step 2: OCR/parse DOM",
                        "Step 3: Return structured observation",
                    ],
                ),
            )
        else:
            out.append(
                (
                    "observe",
                    "computer_primitives.observe((('What is on screen?',), {}))",
                    "{'title': 'Home'}",
                ),
            )
    if with_agent_logs:
        out.append(
            (
                "tool_call",
                "primitives.contacts.update((('Add Alice',), {}))",
                "Updated contact",
                [
                    "Calling ContactManager.update",
                    "Validated schema",
                    "Wrote to DB",
                ],
            ),
        )
    else:
        out.append(
            (
                "tool_call",
                "primitives.contacts.update((('Add Alice',), {}))",
                "Updated contact",
            ),
        )
    return out


def _evidence(
    *,
    browser: bool,
    primitives: bool,
    browser_error: bool = False,
) -> dict[str, Any]:
    """Return a polymorphic evidence dict shaped like env.capture_state() outputs."""

    evidence: dict[str, Any] = {}
    if browser:
        if browser_error:
            evidence["computer_primitives"] = {"error": "Could not capture screenshot"}
        else:
            evidence["computer_primitives"] = {
                "screenshot": "base64:abc123",
                "url": "https://example.com/path",
            }
    if primitives:
        evidence["primitives"] = {
            "type": "return_value",
            "note": "Return values are the primary evidence for state manager operations.",
        }
    return evidence


def _idempotency_cache(n: int) -> Dict[tuple, Any]:
    """Return idempotency cache values with the runtime-stored `meta` + `interaction_log` shape."""

    cache: Dict[tuple, Any] = {}
    for i in range(n):
        cache[
            (
                ("main_plan",),
                (),
                (),
                i,
                "primitives.contacts.update",
                f"(({i!r},), {{}})",
            )
        ] = {
            "result": f"result-{i}",
            "interaction_log": (
                "tool_call",
                f"primitives.contacts.update((('Add {i}',), {{}}))",
                f"Updated contact {i}",
            ),
            "meta": {
                "function": "main_plan",
                "step": i,
                "tool": "primitives.contacts.update",
            },
        }
    return cache


def _pane_snapshot(n_handles: int) -> dict[str, Any]:
    """Return a representative steerable-tool-pane snapshot with many in-flight handles."""

    handles: list[dict[str, Any]] = []
    for i in range(n_handles):
        handles.append(
            {
                "handle_id": f"h_{i}",
                "origin_tool": "primitives.contacts.update",
                "status": "running" if i % 2 == 0 else "paused",
                "capabilities": ["interjectable", "pausable"],
            },
        )
    return {"active_handles": handles, "pending_clarifications_count": 0}


# ============================================================================
# Dynamic context capture / formatting tests
# ============================================================================


def test_initial_plan_includes_existing_functions_library_and_retry_and_images() -> (
    None
):
    """`build_initial_plan_prompt` includes library, retry, and images sections when provided."""

    tools = _tools_mixed()
    envs = _environments_mixed()
    existing = _existing_functions_library()
    retry_msg = "RETRY: previous attempt failed due to missing auth."

    prompt = build_initial_plan_prompt(
        goal="Find Alice and update her contact.",
        existing_functions=existing,
        retry_msg=retry_msg,
        tools=tools,
        environments=envs,
        images=_images(),
    )

    assert "### YOUR AVAILABLE FUNCTIONS (Already Loaded & Callable)" in prompt
    assert "trusted_skill" in prompt
    assert "other_skill" in prompt
    assert retry_msg in prompt
    assert "The user has provided the following images" in prompt
    assert "Image 0: Login screen" in prompt
    assert "Image 1: Confirmation modal" in prompt


def test_all_codegen_prompts_include_simplicity_first_principles() -> None:
    """All codegen prompt types should include the shared simplicity-first guidance."""

    marker = "### Simplicity-First Planning (CRITICAL)"

    prompt = build_initial_plan_prompt(
        goal="Find Alice and update her contact.",
        existing_functions=_existing_functions_library(),
        retry_msg="",
        tools=_tools_mixed(),
        environments=_environments_mixed(),
        images=None,
    )
    assert marker in prompt

    static_prefix, _dynamic = build_dynamic_implement_prompt(
        goal="Keep contacts up to date.",
        scoped_context=_scoped_context_str(include_parent=True, include_children=True),
        call_stack=["main_plan", "parent_fn", "current_fn"],
        function_name="current_fn",
        function_sig="() -> str",
        function_docstring="Current function.",
        clarification_question=None,
        clarification_answer=None,
        replan_context="Implement from stub.",
        has_browser_screenshot=False,
        tools=_tools_mixed(),
        existing_functions=_existing_functions_library(),
        environments=_environments_mixed(),
        recent_transcript=None,
        parent_chat_context=None,
        images=None,
    )
    assert marker in static_prefix

    static_prefix, _dynamic = build_interjection_prompt(
        interjection="Be concise.",
        parent_chat_context=None,
        scoped_context=_scoped_context_str(include_parent=True, include_children=True),
        call_stack=["main_plan"],
        action_log=["step: started"],
        goal="Goal",
        idempotency_cache=None,
        tools=_tools_mixed(),
        environments=_environments_mixed(),
        images=None,
        pane_snapshot=None,
    )
    assert marker in static_prefix


def test_initial_plan_with_empty_state_is_safe() -> None:
    """`build_initial_plan_prompt` is safe with empty library + no retry + no images."""

    prompt = build_initial_plan_prompt(
        goal="Do something.",
        existing_functions={},
        retry_msg="",
        tools=_tools_mixed(),
        environments=_environments_mixed(),
        images=None,
    )
    assert "### YOUR AVAILABLE FUNCTIONS (Already Loaded & Callable)" in prompt
    assert "None." in prompt


def test_dynamic_implement_includes_scoped_context_and_call_stack_formatting() -> None:
    """`build_dynamic_implement_prompt` includes scoped context and formats call stack consistently."""

    static_prefix, dynamic = build_dynamic_implement_prompt(
        goal="Keep contacts up to date.",
        scoped_context=_scoped_context_str(include_parent=True, include_children=True),
        call_stack=["main_plan", "parent_fn", "current_fn"],
        function_name="current_fn",
        function_sig="() -> str",
        function_docstring="Current function.",
        clarification_question=None,
        clarification_answer=None,
        replan_context="Implement from stub.",
        has_browser_screenshot=False,
        tools=_tools_mixed(),
        existing_functions=_existing_functions_library(),
        environments=_environments_mixed(),
        recent_transcript=None,
        parent_chat_context=None,
        images=None,
    )

    assert "### Scoped Plan Analysis & Call Stack (Snapshot)" in dynamic
    assert "`main_plan -> parent_fn -> current_fn`" in dynamic
    assert "### Parent Function Source" in dynamic
    assert "### Current Function Source" in dynamic
    assert "### Children Source (Functions it may call)" in dynamic
    assert "Current Browser View (Screenshot)" not in dynamic
    assert "No browser state available." in dynamic
    assert "Tools are grouped by namespace" in static_prefix
    assert (
        '"computer_primitives"' in static_prefix
        or "`computer_primitives`" in static_prefix
    )
    assert '"primitives"' in static_prefix or "`primitives`" in static_prefix


def test_dynamic_implement_includes_optional_sections_only_when_provided() -> None:
    """`build_dynamic_implement_prompt` includes clarification/transcript/chat context only when provided."""

    parent_chat_context = [{"role": "user", "content": "hello"}]
    recent_transcript = "user: hello\nassistant: hi"

    _static_prefix, dynamic = build_dynamic_implement_prompt(
        goal="Goal",
        scoped_context=_scoped_context_str(
            include_parent=False,
            include_children=False,
        ),
        call_stack=[],
        function_name="f",
        function_sig="(x: int) -> int",
        function_docstring="doc",
        clarification_question="What is X?",
        clarification_answer="X is 123.",
        replan_context="Fix: crashed due to KeyError.",
        has_browser_screenshot=True,
        tools=_tools_mixed(),
        existing_functions={},
        environments=_environments_mixed(),
        recent_transcript=recent_transcript,
        parent_chat_context=parent_chat_context,
        images=_images(),
    )

    assert "### User Clarification Provided" in dynamic
    assert "What is X?" in dynamic
    assert "X is 123." in dynamic
    assert "### Recent Conversation Transcript" in dynamic
    assert recent_transcript in dynamic
    assert "### Full Parent Chat Context" in dynamic
    assert json.dumps(parent_chat_context, indent=2) in dynamic
    assert "### 📌 CRITICAL INSTRUCTIONS: MODIFY EXISTING FUNCTION `f`" in dynamic
    assert "Current Browser View (Screenshot)" in dynamic
    assert "The user has provided the following images" in dynamic
    assert "Image 0: Login screen" in dynamic


def test_verification_includes_agent_trace_when_present() -> None:
    """`build_verification_prompt` renders a low-level agent trace when agent logs are present."""

    _static_prefix, dynamic = build_verification_prompt(
        goal="Goal",
        function_name="current_fn",
        function_docstring="doc",
        scoped_context=_scoped_context_str(include_parent=True, include_children=True),
        interactions=_interactions(with_agent_logs=True),
        evidence=_evidence(browser=True, primitives=False),
        function_return_value={"ok": True},
        clarification_question=None,
        clarification_answer=None,
        recent_transcript=None,
        parent_chat_context=None,
        environments=_environments_mixed(),
    )

    assert "### 🔬 Low-Level Agent Trace (Ground Truth)" in dynamic
    assert "- For Action:" in dynamic
    assert "Step 1: Capture screenshot" in dynamic
    assert "\n  Step 2: OCR/parse DOM\n" in dynamic


def test_verification_omits_agent_trace_when_absent() -> None:
    """`build_verification_prompt` uses a fallback when no agent logs exist in interactions."""

    _static_prefix, dynamic = build_verification_prompt(
        goal="Goal",
        function_name="f",
        function_docstring=None,
        scoped_context=_scoped_context_str(
            include_parent=False,
            include_children=False,
        ),
        interactions=_interactions(with_agent_logs=False),
        evidence={},
        function_return_value=None,
        environments=_environments_mixed(),
    )

    assert "No low-level agent trace was recorded for this step." in dynamic
    assert "### 🔬 Low-Level Agent Trace" not in dynamic


def test_verification_evidence_sections_and_mixed_evidence() -> None:
    """`build_verification_prompt` includes evidence sections and mixed-evidence instructions when appropriate."""

    _static_prefix, dynamic = build_verification_prompt(
        goal="Goal",
        function_name="f",
        function_docstring="doc",
        scoped_context=_scoped_context_str(
            include_parent=False,
            include_children=False,
        ),
        interactions=[],
        evidence=_evidence(browser=True, primitives=True),
        function_return_value="ok",
        environments=_environments_mixed(),
    )

    assert "No tool actions were logged for this step." in dynamic
    assert "### 📸 Visual Evidence (Browser)" in dynamic
    assert "https://example.com/path" in dynamic
    assert "### 📊 System State Evidence (Return Values)" in dynamic
    assert "### 🔀 Mixed Evidence (Browser + Return Value)" in dynamic
    assert "Function Return Value:" in dynamic
    assert "```" in dynamic
    assert repr("ok") in dynamic


def test_verification_browser_evidence_unavailable_section() -> None:
    """`build_verification_prompt` includes a browser-evidence-unavailable section on capture errors."""

    _static_prefix, dynamic = build_verification_prompt(
        goal="Goal",
        function_name="f",
        function_docstring="doc",
        scoped_context=_scoped_context_str(
            include_parent=False,
            include_children=False,
        ),
        interactions=[],
        evidence=_evidence(browser=True, primitives=False, browser_error=True),
        function_return_value="ok",
        environments=_environments_mixed(),
    )
    assert "### ⚠️ Browser Evidence Unavailable" in dynamic
    assert "Could not capture browser state" in dynamic


def test_verification_includes_clarification_transcript_and_parent_chat_context_sections() -> (
    None
):
    """`build_verification_prompt` includes clarification/transcript/chat context sections when provided."""

    parent_chat_context = [{"role": "user", "content": "Earlier context"}]
    recent_transcript = "user: earlier\nassistant: noted"

    _static_prefix, dynamic = build_verification_prompt(
        goal="Goal",
        function_name="current_fn",
        function_docstring="doc",
        scoped_context=_scoped_context_str(include_parent=True, include_children=True),
        interactions=_interactions(with_agent_logs=True),
        evidence=_evidence(browser=False, primitives=True),
        function_return_value={"ok": True},
        clarification_question="Which Alice do you mean?",
        clarification_answer="Alice Smith in Sales.",
        recent_transcript=recent_transcript,
        parent_chat_context=parent_chat_context,
        environments=_environments_mixed(),
    )

    assert "### 💡 User Clarification Provided" in dynamic
    assert "Which Alice do you mean?" in dynamic
    assert "Alice Smith in Sales." in dynamic
    assert "### 📖 Recent Conversation Transcript" in dynamic
    assert recent_transcript in dynamic
    assert "### 💬 Full Parent Chat Context" in dynamic
    assert json.dumps(parent_chat_context, indent=2) in dynamic


def test_interjection_includes_scoped_context_call_stack_actions_cache_and_pane_snapshot() -> (
    None
):
    """`build_interjection_prompt` includes scoped context, recent actions, cache summary, and pane snapshot."""

    static_prefix, dynamic = build_interjection_prompt(
        interjection="Actually use Bob instead of Alice.",
        parent_chat_context=[{"role": "user", "content": "prev"}],
        scoped_context=_scoped_context_str(include_parent=True, include_children=True),
        call_stack=["main_plan", "current_fn"],
        action_log=["Did A", "Did B"],
        goal="Goal",
        idempotency_cache=_idempotency_cache(5),
        tools=_tools_mixed(),
        environments=_environments_mixed(),
        images=_images(),
        pane_snapshot=_pane_snapshot(12),
    )

    assert "### Full Situational Context" in dynamic
    assert "User's Interjection" in dynamic
    assert "Current Execution Point (Call Stack)" in dynamic
    assert "`main_plan -> current_fn`" in dynamic
    assert "Most Recent Plan Actions" in dynamic
    assert "- Did A" in dynamic
    assert "- Did B" in dynamic
    assert "Scoped Source Code Context" in dynamic
    assert "### Parent Function Source" in dynamic
    assert "### Cache Status" in dynamic
    assert "### Recent Cached Actions:" in dynamic
    assert "Func: `main_plan`" in dynamic
    assert "### In-Flight Handles (Steerable)" in dynamic
    assert "**handle_id**: `h_0`" in dynamic
    assert "**origin_tool**: `primitives.contacts.update`" in dynamic
    assert "**status**:" in dynamic
    assert "**capabilities**:" in dynamic
    assert "**handle_id**: `h_11`" in dynamic
    assert "### Tool Reference" in static_prefix


def test_interjection_with_empty_state_is_safe() -> None:
    """`build_interjection_prompt` is safe when call stack/actions/cache/pane snapshot are empty."""

    _static_prefix, dynamic = build_interjection_prompt(
        interjection="Stop.",
        parent_chat_context=None,
        scoped_context="",
        call_stack=[],
        action_log=[],
        goal="",
        idempotency_cache={},
        tools=_tools_mixed(),
        environments=_environments_mixed(),
        images=None,
        pane_snapshot=None,
    )
    assert "Not inside any function." in dynamic
    assert "No actions yet." in dynamic
    assert "The cache is currently empty." in dynamic


def test_interjection_prompt_with_annotated_images() -> None:
    """`build_interjection_prompt` includes annotated images in the correct format."""
    from unity.image_manager.types import AnnotatedImageRef, RawImageRef

    images = [
        AnnotatedImageRef(
            raw_image_ref=RawImageRef(image_id=42),
            annotation="Screenshot of login page with username field highlighted",
        ),
        AnnotatedImageRef(
            raw_image_ref=RawImageRef(image_id=43),
            annotation="Error message popup showing 'Invalid credentials'",
        ),
        RawImageRef(image_id=44),  # Raw image without annotation
    ]

    _static_prefix, dynamic = build_interjection_prompt(
        interjection="The login isn't working - see the attached screenshots.",
        parent_chat_context=None,
        scoped_context="",
        call_stack=["main_plan", "login_flow"],
        action_log=["Navigated to login page", "Entered username"],
        goal="Login to the application",
        idempotency_cache={},
        tools=_tools_mixed(),
        environments=_environments_mixed(),
        images=images,
        pane_snapshot=None,
    )

    # Verify interjection message is included
    assert "The login isn't working" in dynamic

    # Verify images section header is present
    assert "The user has provided the following images" in dynamic

    # Verify annotated images show their annotations
    assert (
        "Image 0: Screenshot of login page with username field highlighted" in dynamic
    )
    assert "Image 1: Error message popup showing 'Invalid credentials'" in dynamic

    # Verify raw image without annotation shows appropriate fallback
    assert "Image 2:" in dynamic
    assert "raw image, no annotation" in dynamic


def test_interjection_prompt_with_mixed_image_refs() -> None:
    """`build_interjection_prompt` handles ImageRefs (RootModel) correctly."""
    from unity.image_manager.types import AnnotatedImageRef, ImageRefs, RawImageRef

    # Create ImageRefs wrapper (RootModel with .root attribute)
    images = ImageRefs(
        root=[
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=100),
                annotation="Dashboard view",
            ),
        ],
    )

    _static_prefix, dynamic = build_interjection_prompt(
        interjection="Update the dashboard",
        parent_chat_context=None,
        scoped_context="",
        call_stack=[],
        action_log=[],
        goal="Modify dashboard",
        idempotency_cache={},
        tools=_tools_mixed(),
        environments=_environments_mixed(),
        images=images,
        pane_snapshot=None,
    )

    assert "The user has provided the following images" in dynamic
    assert "Image 0: Dashboard view" in dynamic


# ============================================================================
# Domain-agnostic environment-awareness tests (merged from the old module)
# ============================================================================


def test_initial_plan_prompt_primitives_only_has_no_browser_namespace() -> None:
    """`build_initial_plan_prompt` should not mention `computer_primitives` in primitives-only mode."""

    tools: Dict[str, Callable[..., Any]] = {
        "primitives.contacts.ask": _dummy_contacts_ask,
        "primitives.contacts.update": _dummy_contacts_update,
    }
    environments = {
        "primitives": _DummyEnvironment(
            "### State manager tools (`primitives`)\n- `await primitives.contacts.ask(...)`\n",
        ),
    }

    prompt = build_initial_plan_prompt(
        goal="Find John Doe and add him as a contact.",
        existing_functions={},
        retry_msg="",
        tools=tools,
        environments=environments,
        images=None,
    )

    assert "computer_primitives" not in prompt
    assert "`primitives`" in prompt or '"primitives"' in prompt
    assert "primitives.contacts.ask" in prompt


def test_dynamic_implement_prompt_primitives_only_has_no_browser_namespace() -> None:
    """`build_dynamic_implement_prompt` should not mention `computer_primitives` in primitives-only mode."""

    tools: Dict[str, Callable[..., Any]] = {
        "primitives.contacts.ask": _dummy_contacts_ask,
        "primitives.contacts.update": _dummy_contacts_update,
    }
    environments = {
        "primitives": _DummyEnvironment(
            "### State manager tools (`primitives`)\n- `await primitives.contacts.ask(...)`\n",
        ),
    }

    static_prefix, dynamic_content = build_dynamic_implement_prompt(
        goal="Keep contacts up to date.",
        scoped_context="",
        call_stack=[],
        function_name="ensure_contact_exists",
        function_sig="(name: str) -> dict",
        function_docstring="Ensure a contact exists and return it.",
        clarification_question=None,
        clarification_answer=None,
        replan_context="Implement from stub.",
        has_browser_screenshot=True,
        tools=tools,
        existing_functions={},
        environments=environments,
        recent_transcript=None,
        parent_chat_context=None,
        images=None,
    )

    assert "computer_primitives" not in static_prefix
    assert "computer_primitives" not in dynamic_content
    assert '"primitives"' in static_prefix or "`primitives`" in static_prefix


def test_prompts_include_primitives_guidance_when_browser_env_is_present() -> None:
    """In mixed mode, plan/implement prompts should still include primitives guidance sections."""

    tools: Dict[str, Callable[..., Any]] = {
        "computer_primitives.navigate": _dummy_navigate,
        "computer_primitives.act": _dummy_act,
        "computer_primitives.observe": _dummy_observe,
        "primitives.contacts.ask": _dummy_contacts_ask,
        "primitives.contacts.update": _dummy_contacts_update,
    }
    environments = {
        "computer_primitives": _DummyEnvironment(
            "### Computer tools (`computer_primitives`)",
        ),
        "primitives": _DummyEnvironment(
            "### State manager tools (`primitives`)\n- `await primitives.contacts.ask(...)`\n",
        ),
    }

    plan_prompt = build_initial_plan_prompt(
        goal="Research John Doe and update his contact.",
        existing_functions={},
        retry_msg="",
        tools=tools,
        environments=environments,
        images=None,
    )
    assert "State Manager Examples" in plan_prompt or "primitives" in plan_prompt

    static_prefix, _dynamic_content = build_dynamic_implement_prompt(
        goal="Keep contacts up to date.",
        scoped_context="",
        call_stack=[],
        function_name="enrich_contact",
        function_sig="(name: str) -> dict",
        function_docstring="Enrich a contact.",
        clarification_question=None,
        clarification_answer=None,
        replan_context="Implement from stub.",
        has_browser_screenshot=False,
        tools=tools,
        existing_functions={},
        environments=environments,
        recent_transcript=None,
        parent_chat_context=None,
        images=None,
    )
    assert "State Manager Guidance (`primitives`)" in static_prefix


def test_interjection_prompt_primitives_only_is_environment_aware() -> None:
    """`build_interjection_prompt` should be namespace-grouped and avoid browser assumptions in primitives-only mode."""

    tools: Dict[str, Callable[..., Any]] = {
        "primitives.contacts.ask": _dummy_contacts_ask,
        "primitives.contacts.update": _dummy_contacts_update,
    }
    environments = {
        "primitives": _DummyEnvironment(
            "### State manager tools (`primitives`)\n- `await primitives.contacts.ask(...)`\n",
        ),
    }

    static_prefix, dynamic_content = build_interjection_prompt(
        interjection="Actually, use Bob instead of Alice.",
        parent_chat_context=None,
        scoped_context="",
        call_stack=[],
        action_log=[],
        goal="Create a contact.",
        idempotency_cache={},
        tools=tools,
        environments=environments,
        images=None,
    )

    assert "computer_primitives" not in static_prefix
    assert '"primitives"' in static_prefix
    assert "primitives.contacts.ask" in static_prefix
    assert "State manager tools" in static_prefix
    assert "Actually, use Bob instead of Alice" in dynamic_content


def test_interjection_prompt_mixed_mode_routing() -> None:
    """Interjection prompt should include mixed-mode routing examples."""

    tools = {
        "computer_primitives.navigate": _dummy_navigate,
        "primitives.contacts.ask": _dummy_contacts_ask,
    }
    environments = {
        "computer_primitives": _DummyEnvironment("### Browser"),
        "primitives": _DummyEnvironment("### State managers"),
    }

    static_prefix, _ = build_interjection_prompt(
        interjection="Be concise",
        parent_chat_context=None,
        scoped_context="",
        call_stack=[],
        action_log=[],
        goal="Browse and save",
        idempotency_cache={},
        tools=tools,
        environments=environments,
        images=None,
    )

    assert "broadcast" in static_prefix.lower()
    assert "routing" in static_prefix.lower()


def test_refactor_prompt_allows_missing_url_and_is_not_browser_assuming() -> None:
    """`build_refactor_prompt` tolerates `current_url=None` and avoids browser-only instructions."""

    tools: Dict[str, Callable[..., Any]] = {
        "primitives.contacts.ask": _dummy_contacts_ask,
        "primitives.contacts.update": _dummy_contacts_update,
    }
    environments = {
        "primitives": _DummyEnvironment(
            "### State manager tools (`primitives`)\n- `await primitives.contacts.ask(...)`\n",
        ),
    }

    prompt = build_refactor_prompt(
        monolithic_code="async def main_plan():\n    return 1\n",
        generalization_request="Now do the same for Bob.",
        action_log="",
        current_url=None,
        tools=tools,
        environments=environments,
    )

    assert "Browser's Current URL" not in prompt
    assert "Current URL" not in prompt
    assert "Use the `computer_primitives` global object" not in prompt
    assert "namespaces" in prompt


@pytest.mark.asyncio
async def test_clear_browser_queue_is_noop_without_browser_env() -> None:
    """`HierarchicalActorHandle._clear_browser_queue_for_run` is a no-op with no browser env configured."""

    handle = HierarchicalActorHandle.__new__(HierarchicalActorHandle)
    handle.actor = SimpleNamespace(environments={})
    handle.action_log = []
    await handle._clear_browser_queue_for_run(run_id_to_clear=123)


@pytest.mark.asyncio
async def test_interject_does_not_require_browser_env_for_interrupt() -> None:
    """Interjection handling should not touch browser primitives when no browser env exists."""

    handle = HierarchicalActorHandle.__new__(HierarchicalActorHandle)

    handle.actor = SimpleNamespace(
        environments={},
        tools={},
        _get_scoped_context_from_plan_state=lambda _self: (_ for _ in ()).throw(
            RuntimeError("stop here"),
        ),
        _format_scoped_context_for_prompt=lambda _ctx: "",
    )

    handle._get_computer_primitives = lambda: (_ for _ in ()).throw(
        AssertionError("should not access computer_primitives"),
    )

    handle._state = _HierarchicalHandleState.RUNNING
    handle.action_log = []
    handle.parent_chat_context = None
    handle.call_stack = []
    handle.goal = ""
    handle.idempotency_cache = {}
    handle._is_valid_method = lambda _name: True
    handle._interject_lock = asyncio.Lock()
    handle._cancel_all_background_tasks = AsyncMock(return_value=None)
    handle.pause = AsyncMock(return_value=None)

    result = await handle.interject("hello")
    assert "Error processing interjection" in result
    assert "stop here" in result


@pytest.mark.asyncio
async def test_build_ask_prompt_with_browser_environment_and_visual_evidence() -> None:
    """`build_ask_prompt` includes browser-specific sections only when visual evidence is available."""

    environments = {"computer_primitives": _MockEnvironment("computer_primitives")}
    evidence = {"computer_primitives": {"type": "screenshot", "data": "base64_data"}}

    prompt = build_ask_prompt(
        goal="Search for recipes",
        state="RUNNING",
        call_stack="main_plan -> search_recipes",
        context_log="- Navigated to site\n- Clicked search",
        question="What's the current page?",
        environments=environments,
        evidence=evidence,
    )

    assert "web automation task" in prompt
    assert "Visual Evidence" in prompt or "screenshot" in prompt.lower()


@pytest.mark.asyncio
async def test_build_ask_prompt_with_browser_environment_no_visual_evidence() -> None:
    """`build_ask_prompt` is generic when browser exists but no visual evidence is attached."""

    environments = {"computer_primitives": _MockEnvironment("computer_primitives")}
    evidence = None

    prompt = build_ask_prompt(
        goal="Search for recipes",
        state="RUNNING",
        call_stack="main_plan -> search_recipes",
        context_log="- Navigated to site\n- Clicked search",
        question="What's the current page?",
        environments=environments,
        evidence=evidence,
    )

    assert "web automation task" not in prompt
    assert "Visual Evidence" not in prompt
    assert "task" in prompt.lower()


@pytest.mark.asyncio
async def test_build_ask_prompt_with_mixed_environments_and_visual_evidence() -> None:
    """With mixed environments, `build_ask_prompt` includes browser view only when visual evidence is available."""

    environments = {
        "computer_primitives": _MockEnvironment("computer_primitives"),
        "primitives": _MockEnvironment("primitives"),
    }
    evidence = {"computer_primitives": {"type": "screenshot", "data": "base64"}}

    prompt = build_ask_prompt(
        goal="Browse and save contacts",
        state="RUNNING",
        call_stack="main_plan",
        context_log="- Browsed LinkedIn\n- Saved contact",
        question="What's our progress?",
        environments=environments,
        evidence=evidence,
    )

    assert "web automation" in prompt
    assert "Visual Evidence" in prompt or "screenshot" in prompt.lower()


@pytest.mark.asyncio
async def test_build_ask_prompt_with_mixed_environments_no_visual_evidence() -> None:
    """With mixed environments, `build_ask_prompt` omits browser view when no visual evidence is attached."""

    environments = {
        "computer_primitives": _MockEnvironment("computer_primitives"),
        "primitives": _MockEnvironment("primitives"),
    }

    prompt = build_ask_prompt(
        goal="Browse and save contacts",
        state="RUNNING",
        call_stack="main_plan",
        context_log="- Browsed LinkedIn\n- Saved contact",
        question="What's our progress?",
        environments=environments,
        evidence=None,
    )

    assert "web automation" not in prompt
    assert "Visual Evidence" not in prompt


@pytest.mark.asyncio
async def test_build_ask_prompt_with_evidence_dict_mixed() -> None:
    """`build_ask_prompt` handles evidence from multiple environments."""

    environments = {
        "computer_primitives": _MockEnvironment("computer_primitives"),
        "primitives": _MockEnvironment("primitives"),
    }
    evidence = {
        "computer_primitives": {"type": "screenshot", "url": "https://example.com"},
        "primitives": {"type": "return_value", "value": "Contact found"},
    }

    prompt = build_ask_prompt(
        goal="Browse and save",
        state="RUNNING",
        call_stack="main_plan",
        context_log="- Browsed site\n- Saved contact",
        question="What's our progress?",
        environments=environments,
        evidence=evidence,
    )

    assert "Visual Evidence" in prompt or "screenshot" in prompt.lower()
    assert "State Manager Evidence" in prompt or "return value" in prompt.lower()


@pytest.mark.asyncio
async def test_build_ask_prompt_with_primitives_evidence_only() -> None:
    """`build_ask_prompt` handles primitives evidence without browser."""

    environments = {"primitives": _MockEnvironment("primitives")}
    evidence = {"primitives": {"type": "return_value", "value": "3 contacts found"}}

    prompt = build_ask_prompt(
        goal="Query contacts",
        state="RUNNING",
        call_stack="main_plan",
        context_log="- Queried contacts",
        question="How many contacts?",
        environments=environments,
        evidence=evidence,
    )

    assert "State Manager Evidence" in prompt or "return value" in prompt.lower()
    assert "Visual Evidence" not in prompt
    assert "Browser" not in prompt


@pytest.mark.asyncio
async def test_build_ask_prompt_backward_compatibility() -> None:
    """`build_ask_prompt` works when environments=None and evidence=None (backward compatibility)."""

    prompt = build_ask_prompt(
        goal="Do something",
        state="RUNNING",
        call_stack="main_plan",
        context_log="- Action 1",
        question="What happened?",
        environments=None,
        evidence=None,
    )

    assert prompt is not None
    assert len(prompt) > 0


@pytest.mark.asyncio
async def test_build_precondition_prompt_with_browser_environment() -> None:
    """`build_precondition_prompt` includes URL/page-based examples when browser env is active."""

    environments = {"computer_primitives": _MockEnvironment("computer_primitives")}

    prompt = build_precondition_prompt(
        function_source_code="async def search(): pass",
        interactions_log='[{"action": "click"}]',
        has_entry_screenshot=True,
        environments=environments,
    )

    assert "url" in prompt.lower()
    assert "page" in prompt.lower()


@pytest.mark.asyncio
async def test_build_precondition_prompt_visual_context_generic() -> None:
    """When screenshot exists but no browser env, visual-context language should be environment-generic."""

    environments = {"primitives": _MockEnvironment("primitives")}

    prompt = build_precondition_prompt(
        function_source_code="async def task(): pass",
        interactions_log="[]",
        has_entry_screenshot=True,
        environments=environments,
    )

    assert "execution environment" in prompt.lower() or "environment" in prompt.lower()
    assert "browser's state" not in prompt.lower()


@pytest.mark.asyncio
async def test_build_precondition_prompt_backward_compatibility() -> None:
    """`build_precondition_prompt` works when environments=None (backward compatibility)."""

    prompt = build_precondition_prompt(
        function_source_code="async def task(): pass",
        interactions_log="[]",
        has_entry_screenshot=False,
        environments=None,
    )

    assert prompt is not None
    assert len(prompt) > 0


# ============================================================================
# Actor integration tests: confirm capture + forwarding into prompt builders
# ============================================================================


class _SentinelPromptBuilderCalled(RuntimeError):
    """Raised by patched prompt builders to stop execution after capturing arguments."""


class _HistoryHandle(SteerableToolHandle):
    """Minimal SteerableToolHandle implementation that exposes `get_history()` for proxy capture."""

    def __init__(
        self,
        *,
        final_result: str = "done",
        history: list[str] | None = None,
    ) -> None:
        self._stopped = False
        self._final_result = final_result
        self._history = history or ["loop:step1", "loop:step2", "loop:step3"]

    async def ask(self, question: str, *, parent_chat_context_cont=None, images=None):
        return self

    def interject(self, message: str, *, parent_chat_context_cont=None, images=None):
        return None

    def stop(self, reason: str | None = None, *, parent_chat_context_cont=None):
        self._stopped = True
        return None

    async def pause(self):
        return None

    async def resume(self):
        return None

    def done(self):
        return self._stopped

    async def result(self):
        return self._final_result

    async def next_clarification(self) -> dict:
        return {}

    async def next_notification(self) -> dict:
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        return None

    def get_history(self) -> list[str]:
        return list(self._history)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_actor_captures_and_passes_call_stack_and_scoped_context_to_dynamic_implement_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actor execution should capture call stack + source context and forward them into `build_dynamic_implement_prompt`."""

    from unity.actor.hierarchical_actor import HierarchicalActor
    import unity.actor.hierarchical_actor as hierarchical_actor_mod
    import unity.actor.prompt_builders as prompt_builders

    # Create actor with mock browser (no external services needed).
    actor = HierarchicalActor(headless=True, computer_mode="mock", connect_now=False)

    # We'll patch the prompt builder to record the actor-supplied snapshots, and patch
    # `llm_call` to stop execution after the prompt is built (so we don't do any real LLM work).
    seen_calls: list[dict[str, Any]] = []

    original_builder = prompt_builders.build_dynamic_implement_prompt

    def _spy_build_dynamic_implement_prompt(*args: Any, **kwargs: Any):
        seen_calls.append(
            {
                "function_name": kwargs.get("function_name"),
                "call_stack": kwargs.get("call_stack"),
                "scoped_context": kwargs.get("scoped_context"),
            },
        )
        return original_builder(*args, **kwargs)

    async def _spy_llm_call(*args: Any, **kwargs: Any):
        raise _SentinelPromptBuilderCalled("stop before LLM call")

    monkeypatch.setattr(
        prompt_builders,
        "build_dynamic_implement_prompt",
        _spy_build_dynamic_implement_prompt,
    )
    monkeypatch.setattr(hierarchical_actor_mod, "llm_call", _spy_llm_call)

    canned_plan = """
async def stubbed_step():
    \"\"\"A stub that must trigger dynamic implementation.\"\"\"
    raise NotImplementedError("stub: implement this later")

async def main_plan():
    \"\"\"Main entrypoint.\"\"\"
    await stubbed_step()
"""

    active_task: HierarchicalActorHandle | None = None
    try:
        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Test dynamic implement capture",
            persist=False,
        )

        # Cancel auto-start task before injecting plan source.
        if active_task._execution_task:
            active_task._execution_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await active_task._execution_task

        active_task.plan_source_code = actor._sanitize_code(canned_plan, active_task)
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        # Wait until the dynamic-implement prompt gets built at least once.
        async def _wait_for_prompt_calls() -> None:
            deadline = asyncio.get_event_loop().time() + 60
            while asyncio.get_event_loop().time() < deadline:
                if seen_calls:
                    return
                await asyncio.sleep(0.05)
            raise AssertionError(
                "Timed out waiting for build_dynamic_implement_prompt to be called.",
            )

        await _wait_for_prompt_calls()

        # We specifically care that the stubbed function's dynamic-implement prompt was built.
        stub_calls = [c for c in seen_calls if c.get("function_name") == "stubbed_step"]
        assert (
            stub_calls
        ), f"Expected a dynamic-implement prompt for 'stubbed_step', got: {[c.get('function_name') for c in seen_calls]}"

        captured = stub_calls[0]
        call_stack = captured.get("call_stack") or []
        scoped_context = captured.get("scoped_context") or ""

        assert isinstance(call_stack, list)
        assert "main_plan" in call_stack
        assert "stubbed_step" in call_stack
        assert call_stack[-1] == "stubbed_step"

        # The scoped context string should include current + parent source extracted by the Actor.
        assert "async def stubbed_step" in scoped_context
        assert "async def main_plan" in scoped_context

    finally:
        if active_task and not active_task.done():
            with contextlib.suppress(Exception):
                await active_task.stop()
        with contextlib.suppress(Exception):
            await actor.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_actor_captures_tool_calls_in_idempotency_cache_and_passes_cache_to_interjection_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After executing tools, Actor should populate idempotency cache and pass it into `build_interjection_prompt`."""

    from unity.actor.hierarchical_actor import HierarchicalActor
    import unity.actor.prompt_builders as prompt_builders

    actor = HierarchicalActor(headless=True, computer_mode="mock", connect_now=False)
    actor.computer_primitives.act = AsyncMock(return_value=None)

    called = asyncio.Event()
    captured: dict[str, Any] = {}

    def _spy_build_interjection_prompt(*args: Any, **kwargs: Any):
        captured["idempotency_cache"] = kwargs.get("idempotency_cache")
        called.set()
        raise _SentinelPromptBuilderCalled("interjection prompt captured")

    monkeypatch.setattr(
        prompt_builders,
        "build_interjection_prompt",
        _spy_build_interjection_prompt,
    )

    canned_plan = """
async def main_plan():
    \"\"\"Run exactly one tool call so the proxy writes a cache entry.\"\"\"
    await computer_primitives.act("do something")
    return "done"
"""

    active_task: HierarchicalActorHandle | None = None
    try:
        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Test interjection cache capture",
            persist=True,
        )
        if active_task._execution_task:
            active_task._execution_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await active_task._execution_task

        active_task.plan_source_code = actor._sanitize_code(canned_plan, active_task)
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        # Wait until main plan finishes and handle pauses for interjection (persist=True).
        async def _wait_for_paused_for_interjection():
            deadline = asyncio.get_event_loop().time() + 60
            while asyncio.get_event_loop().time() < deadline:
                if (
                    active_task._state
                    == _HierarchicalHandleState.PAUSED_FOR_INTERJECTION
                ):
                    return
                await asyncio.sleep(0.05)
            raise AssertionError("Timed out waiting for PAUSED_FOR_INTERJECTION state.")

        await _wait_for_paused_for_interjection()

        assert (
            active_task.idempotency_cache
        ), "Expected actor to populate idempotency_cache after tool call."

        # Trigger interjection; patched prompt builder will stop execution after capturing args.
        status = await active_task.interject("Be concise")
        assert "Error processing interjection" in status or "captured" in status.lower()

        await asyncio.wait_for(called.wait(), timeout=30)

        cache = captured.get("idempotency_cache")
        assert isinstance(cache, dict)
        assert (
            cache
        ), "Expected non-empty idempotency_cache passed into interjection prompt."

        # Cache entries should include meta.tool for the executed tool call.
        meta_tools = [
            (v.get("meta") or {}).get("tool")
            for v in cache.values()
            if isinstance(v, dict)
        ]
        assert any(
            isinstance(t, str) and t.endswith("computer_primitives.act")
            for t in meta_tools
        )

    finally:
        if active_task and not active_task.done():
            with contextlib.suppress(Exception):
                await active_task.stop()
        with contextlib.suppress(Exception):
            await actor.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_actor_passes_primitives_handle_history_into_verification_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actor should capture primitives handle sub-loop history and pass it into `build_verification_prompt` interactions."""

    from unity.actor.hierarchical_actor import HierarchicalActor, VerificationAssessment
    import unity.actor.hierarchical_actor as hierarchical_actor_mod
    import unity.actor.prompt_builders as prompt_builders

    actor = HierarchicalActor(headless=True, computer_mode="mock", connect_now=False)

    # Patch the underlying primitives manager to return a handle that exposes `get_history()`.
    primitives_env = actor.environments.get("primitives")
    assert (
        primitives_env is not None
    ), "Expected default 'primitives' environment to be present."
    primitives_obj = primitives_env.get_instance()
    primitives_obj.tasks.execute = AsyncMock(
        return_value=_HistoryHandle(final_result="ok-from-handle"),
    )

    # Spy on verification prompt builder but allow it to run.
    seen: list[dict[str, Any]] = []
    original_verification_builder = prompt_builders.build_verification_prompt

    def _spy_build_verification_prompt(*args: Any, **kwargs: Any):
        seen.append(
            {
                "function_name": kwargs.get("function_name"),
                "scoped_context": kwargs.get("scoped_context"),
                "interactions": kwargs.get("interactions"),
                "evidence": kwargs.get("evidence"),
            },
        )
        return original_verification_builder(*args, **kwargs)

    monkeypatch.setattr(
        prompt_builders,
        "build_verification_prompt",
        _spy_build_verification_prompt,
    )

    # Avoid real LLM calls in verification. We only care that prompt building happened
    # with the actor-captured interactions/evidence.
    async def _mock_llm_call(*args: Any, **kwargs: Any) -> str:
        return VerificationAssessment(status="ok", reason="mock").model_dump_json()

    monkeypatch.setattr(hierarchical_actor_mod, "llm_call", _mock_llm_call)

    canned_plan = """
async def main_plan():
    \"\"\"Execute a primitives tool that returns a handle, then await its .result().\"\"\"
    h = await primitives.tasks.execute("run something")
    out = await h.result()
    return out
"""

    active_task: HierarchicalActorHandle | None = None
    try:
        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Test verification prompt capture",
            persist=False,
        )
        if active_task._execution_task:
            active_task._execution_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await active_task._execution_task

        active_task.plan_source_code = actor._sanitize_code(canned_plan, active_task)
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        # Wait until verification prompt builder has been invoked.
        async def _wait_for_seen() -> None:
            deadline = asyncio.get_event_loop().time() + 60
            while asyncio.get_event_loop().time() < deadline:
                if seen:
                    return
                await asyncio.sleep(0.05)
            raise AssertionError(
                "Timed out waiting for build_verification_prompt to be called.",
            )

        await _wait_for_seen()

        # Find the verification item for main_plan (this is what we executed).
        main_items = [x for x in seen if x.get("function_name") == "main_plan"]
        assert (
            main_items
        ), f"Expected verification prompt for main_plan; saw: {[x.get('function_name') for x in seen]}"

        item = main_items[0]
        interactions = item.get("interactions") or []
        assert isinstance(interactions, list)

        # Expect at least one interaction to include history as the 4th tuple element,
        # produced by HistoryCapturingHandleProxy when awaiting `.result()`.
        history_interactions = [
            t
            for t in interactions
            if isinstance(t, tuple) and len(t) >= 4 and isinstance(t[3], list) and t[3]
        ]
        assert (
            history_interactions
        ), f"Expected an interaction with captured history; got: {interactions!r}"

        # Ensure it's specifically a handle method `.result()` entry.
        # (The action string is the handle proxy call_repr, e.g. `... .result(((), {}))`.)
        assert any(
            ".result(" in str(t[1]) for t in history_interactions
        ), f"Expected a '.result()' interaction; got: {history_interactions!r}"

        # Evidence should include environment keys (best-effort).
        evidence = item.get("evidence") or {}
        assert isinstance(evidence, dict)
        assert "primitives" in evidence

        # Scoped context is best-effort and may be empty (call stack can be empty by verification time).
        scoped = item.get("scoped_context")
        assert scoped is not None
        assert isinstance(scoped, str)

    finally:
        if active_task and not active_task.done():
            with contextlib.suppress(Exception):
                await active_task.stop()
        with contextlib.suppress(Exception):
            await actor.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_actor_passes_primitives_cache_and_history_into_interjection_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actor interjection prompt should receive cache entries created by primitives tool calls (including handle history)."""

    from unity.actor.hierarchical_actor import HierarchicalActor, VerificationAssessment
    import unity.actor.hierarchical_actor as hierarchical_actor_mod
    import unity.actor.prompt_builders as prompt_builders

    actor = HierarchicalActor(headless=True, computer_mode="mock", connect_now=False)

    primitives_env = actor.environments.get("primitives")
    assert (
        primitives_env is not None
    ), "Expected default 'primitives' environment to be present."
    primitives_obj = primitives_env.get_instance()
    primitives_obj.tasks.execute = AsyncMock(
        return_value=_HistoryHandle(final_result="ok-from-handle"),
    )

    captured: dict[str, Any] = {}
    called = asyncio.Event()

    def _spy_build_interjection_prompt(*args: Any, **kwargs: Any):
        captured["idempotency_cache"] = kwargs.get("idempotency_cache")
        called.set()
        raise _SentinelPromptBuilderCalled("interjection prompt captured")

    monkeypatch.setattr(
        prompt_builders,
        "build_interjection_prompt",
        _spy_build_interjection_prompt,
    )

    # Avoid real LLM calls in background verification that may run post-completion with persist=True.
    async def _mock_llm_call(*args: Any, **kwargs: Any) -> str:
        return VerificationAssessment(status="ok", reason="mock").model_dump_json()

    monkeypatch.setattr(hierarchical_actor_mod, "llm_call", _mock_llm_call)

    canned_plan = """
async def main_plan():
    \"\"\"Execute primitives tool returning a handle, then await result so history is cached.\"\"\"
    h = await primitives.tasks.execute("run something")
    out = await h.result()
    return out
"""

    active_task: HierarchicalActorHandle | None = None
    try:
        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Test primitives interjection cache",
            persist=True,
        )
        if active_task._execution_task:
            active_task._execution_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await active_task._execution_task

        active_task.plan_source_code = actor._sanitize_code(canned_plan, active_task)
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        # Wait until the plan pauses for interjection.
        async def _wait_for_paused_for_interjection():
            deadline = asyncio.get_event_loop().time() + 60
            while asyncio.get_event_loop().time() < deadline:
                if (
                    active_task._state
                    == _HierarchicalHandleState.PAUSED_FOR_INTERJECTION
                ):
                    return
                await asyncio.sleep(0.05)
            raise AssertionError("Timed out waiting for PAUSED_FOR_INTERJECTION state.")

        await _wait_for_paused_for_interjection()

        assert (
            active_task.idempotency_cache
        ), "Expected idempotency_cache populated by primitives tool call."

        # Trigger interjection to force prompt build; error is expected due to sentinel.
        _status = await active_task.interject("Be concise")
        await asyncio.wait_for(called.wait(), timeout=30)

        cache = captured.get("idempotency_cache")
        assert isinstance(cache, dict)
        assert cache

        # Ensure at least one cache entry was created for a primitives tool call.
        meta_tools = [
            (v.get("meta") or {}).get("tool")
            for v in cache.values()
            if isinstance(v, dict)
        ]
        assert any(
            isinstance(t, str) and t.startswith("primitives.") for t in meta_tools
        ), meta_tools

        # Ensure at least one cached interaction log contains history (4th element list).
        interaction_logs = [
            v.get("interaction_log") for v in cache.values() if isinstance(v, dict)
        ]
        assert any(
            isinstance(t, tuple) and len(t) >= 4 and isinstance(t[3], list) and t[3]
            for t in interaction_logs
        ), interaction_logs

    finally:
        if active_task and not active_task.done():
            with contextlib.suppress(Exception):
                await active_task.stop()
        with contextlib.suppress(Exception):
            await actor.close()
