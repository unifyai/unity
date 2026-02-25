"""
Tests for CodeActActor prompt builder quality.

These tests are intentionally "high-signal string assertions" rather than
snapshot tests. They verify that:
- The prompt exposes the correct primary tools (`execute_code` + session tools)
  using introspected signatures/docstrings (not hardcoded copies).
- The prompt contains diverse examples: sessions, computer, primitives, mixed.
- The prompt contains no legacy `execute_python_code` references.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from unity.actor.code_act_actor import CodeActActor
from unity.actor.prompt_builders import build_code_act_prompt


class _DummyEnv:
    """Minimal environment stub for build_code_act_prompt (prompt-context only)."""

    def __init__(self, prompt_context: str):
        self._prompt_context = prompt_context

    def get_prompt_context(self) -> str:
        return self._prompt_context

    def get_tools(self) -> dict:
        return {}


def _real_envs_mixed() -> Mapping[str, Any]:
    """Real environments that produce self-contained prompt context."""
    from unity.function_manager.primitives import ComputerPrimitives
    from unity.actor.environments.computer import ComputerEnvironment
    from unity.actor.environments.state_managers import StateManagerEnvironment
    from unity.actor.environments.base import _CompositeEnvironment

    cp = ComputerPrimitives(computer_mode="mock")
    composite = _CompositeEnvironment(
        [
            ComputerEnvironment(cp),
            StateManagerEnvironment(),
        ],
    )
    return {"primitives": composite}


@pytest.mark.timeout(30)
def test_code_act_prompt_has_primary_execute_code_and_session_tools_and_no_legacy_name():
    actor = CodeActActor()
    try:
        prompt = build_code_act_prompt(
            environments=_real_envs_mixed(),
            tools=dict(actor.get_tools("act")),
        )
    finally:
        pass

    assert "execute_python_code" not in prompt
    assert "execute_code" in prompt
    assert "list_sessions" in prompt
    assert "inspect_state" in prompt
    assert "close_session" in prompt
    assert "close_all_sessions" in prompt

    # Introspection-based docstring snippet from the actual tool implementation.
    assert "multi-step composition" in prompt.lower()
    assert (
        "multi-language + multi-session" in prompt.lower()
        or "multi-session" in prompt.lower()
    )


@pytest.mark.timeout(30)
def test_code_act_prompt_includes_diverse_examples_sessions_computer_primitives_and_mixed():
    actor = CodeActActor()
    try:
        prompt = build_code_act_prompt(
            environments=_real_envs_mixed(),
            tools=dict(actor.get_tools("act")),
        )
    finally:
        pass

    # Sessions examples (execute_code JSON blocks)
    assert "Sessions & Multi-Language Execution" in prompt
    assert '"language": "bash"' in prompt
    assert '"language": "python"' in prompt
    assert '"name": "list_sessions"' in prompt or "list_sessions" in prompt

    assert "Viewing Computer State" in prompt
    # Computer method documentation (from environment's get_prompt_context)
    assert "primitives.computer" in prompt.lower()
    assert "navigate" in prompt
    assert "act" in prompt
    assert "observe" in prompt

    # State-manager guidance + examples (primitives)
    assert "### State Manager Rules" in prompt
    assert "### Implementation Examples" in prompt
    assert "return the handle as the last expression" in prompt
    assert "immediate in-code composition" in prompt
    assert "neutral or uncertain" in prompt.lower()
    assert "default to returning the handle" in prompt.lower()
    assert "execute_function vs execute_code decision" in prompt


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
def test_computer_environment_prompt_context_from_registry():
    """ComputerEnvironment should derive prompt context from registry."""
    from unity.function_manager.primitives import ComputerPrimitives
    from unity.actor.environments.computer import ComputerEnvironment

    cp = ComputerPrimitives(computer_mode="mock")
    env = ComputerEnvironment(cp)
    context = env.get_prompt_context()

    assert context  # Non-empty
    assert "primitives.computer" in context.lower()
    # All dynamic methods should be documented
    assert "navigate" in context
    assert "act" in context
    assert "observe" in context
    assert "query" in context
    assert "get_links" in context
    assert "get_content" in context
    # Docstrings should include parameter documentation
    assert "Parameters" in context
