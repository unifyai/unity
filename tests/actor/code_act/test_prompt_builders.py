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

    def __init__(self, prompt_context: str, instance: Any | None = None):
        self._prompt_context = prompt_context
        self._instance = instance

    def get_prompt_context(self) -> str:
        return self._prompt_context

    def get_instance(self) -> Any:
        if self._instance is None:
            raise RuntimeError("No instance configured")
        return self._instance


class _ComputerPrimitivesStub:
    async def navigate(self, url: str) -> None:
        _ = url
        return None

    async def act(self, instruction: str) -> str:
        _ = instruction
        return "ok"

    async def observe(self, question: str, response_format: Any = None) -> Any:
        _ = (question, response_format)
        return None


def _envs_mixed() -> Mapping[str, Any]:
    return {
        "computer_primitives": _DummyEnv(
            "### Computer tools (`computer_primitives`)\n- `navigate`, `act`, `observe`",
            instance=_ComputerPrimitivesStub(),
        ),
        "primitives": _DummyEnv(
            "### State manager tools (`primitives`)\n- `await primitives.contacts.ask(...)`\n",
            instance=object(),
        ),
    }


@pytest.mark.timeout(30)
def test_code_act_prompt_has_primary_execute_code_and_session_tools_and_no_legacy_name():
    actor = CodeActActor(headless=True, computer_mode="mock")
    try:
        prompt = build_code_act_prompt(
            environments=_envs_mixed(),
            tools=dict(actor.get_tools("act")),
        )
    finally:
        # build_code_act_prompt is sync; actor.close is async and not required for this unit test
        pass

    assert "execute_python_code" not in prompt
    assert "execute_code" in prompt
    assert "list_sessions" in prompt
    assert "inspect_state" in prompt
    assert "close_session" in prompt
    assert "close_all_sessions" in prompt

    # Introspection-based docstring snippet from the actual tool implementation.
    assert "brain execution" in prompt.lower()
    assert (
        "multi-language + multi-session" in prompt.lower()
        or "multi-session" in prompt.lower()
    )


@pytest.mark.timeout(30)
def test_code_act_prompt_includes_diverse_examples_sessions_computer_primitives_and_mixed():
    actor = CodeActActor(headless=True, computer_mode="mock")
    try:
        prompt = build_code_act_prompt(
            environments=_envs_mixed(),
            tools=dict(actor.get_tools("act")),
        )
    finally:
        pass

    # Sessions examples (execute_code JSON blocks)
    assert "Sessions & Multi-Language Execution" in prompt
    assert '"language": "bash"' in prompt
    assert '"language": "python"' in prompt
    assert '"name": "list_sessions"' in prompt or "list_sessions" in prompt

    assert "Computer State Feedback" in prompt
    # Computer method documentation (from environment's get_prompt_context)
    assert "computer_primitives" in prompt.lower()
    assert "navigate" in prompt
    assert "act" in prompt
    assert "observe" in prompt

    # State-manager guidance + examples (primitives)
    assert "### 🧩 State Manager Rules" in prompt
    assert "### Implementation Examples" in prompt


@pytest.mark.timeout(30)
def test_computer_environment_prompt_context_from_registry():
    """ComputerEnvironment should derive prompt context from registry."""
    from unity.function_manager.primitives import ComputerPrimitives
    from unity.actor.environments.computer import ComputerEnvironment

    cp = ComputerPrimitives(computer_mode="mock")
    env = ComputerEnvironment(cp)
    context = env.get_prompt_context()

    assert context  # Non-empty
    assert "computer_primitives" in context.lower()
    # All dynamic methods should be documented
    assert "navigate" in context
    assert "act" in context
    assert "observe" in context
    assert "query" in context
    assert "get_links" in context
    assert "get_content" in context
    # Static methods too
    assert "reason" in context
    # Docstrings should include parameter documentation
    assert "Parameters" in context
