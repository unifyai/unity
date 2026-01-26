from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, Field
from unittest.mock import AsyncMock

from unity.actor.code_act_actor import CodeActActor

pytestmark = pytest.mark.eval


class RepoNavSummary(BaseModel):
    bash_count: int = Field(description="Count derived from bash output.")
    python_count: int = Field(description="Count derived in Python from bash output.")
    used_bash: bool
    used_python: bool


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_repo_navigation_and_analysis_via_shell_then_python():
    """
    Real-world scenario: use bash to inspect the repo, then Python to summarize.

    We intentionally keep the task tiny/fast while still exercising:
    - multi-language switching (bash -> python)
    - tool-result handoff (Python consumes bash stdout from the chat)
    """
    RepoNavSummary.model_rebuild()

    actor = CodeActActor(headless=True, computer_mode="mock", timeout=60)
    actor._computer_primitives.navigate = AsyncMock(return_value=None)
    actor._computer_primitives.act = AsyncMock(return_value="Action completed")
    actor._computer_primitives.observe = AsyncMock(return_value="Page content observed")

    handle = await actor.act(
        "Do the following steps using ONLY JSON tool calls (no prose until the final answer):\n"
        "1) Call execute_code(language='bash', state_mode='stateless') to run:\n"
        "   `find tests/actor/test_code_act -maxdepth 1 -name 'test_*.py' | wc -l`\n"
        "2) Then call execute_code(language='python', state_mode='stateless') to parse the previous bash stdout,\n"
        "   extract the integer count, and return it.\n"
        "3) Finally, return a JSON object with keys: bash_count, python_count, used_bash, used_python.\n",
        response_format=RepoNavSummary,
        persist=False,
        clarification_enabled=False,
    )
    try:
        res = await asyncio.wait_for(handle.result(), timeout=170)
        assert isinstance(res, RepoNavSummary)
        assert res.bash_count == res.python_count
        assert res.bash_count > 0
        assert res.used_bash is True
        assert res.used_python is True
    finally:
        await actor.close()


class ValidationRecoveryResult(BaseModel):
    saw_validation_error: bool
    final_stdout: str


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_validation_error_self_correction():
    """
    Real-world scenario: Actor makes an invalid call, sees structured validation error,
    then self-corrects and succeeds.
    """
    ValidationRecoveryResult.model_rebuild()

    actor = CodeActActor(headless=True, computer_mode="mock", timeout=60)
    actor._computer_primitives.navigate = AsyncMock(return_value=None)
    actor._computer_primitives.act = AsyncMock(return_value="Action completed")
    actor._computer_primitives.observe = AsyncMock(return_value="Page content observed")

    handle = await actor.act(
        "You MUST do these steps in order:\n"
        "1) Intentionally make an INVALID execute_code call: language='bash', state_mode='stateless', session_name='oops', code='echo hi'.\n"
        "   Confirm you saw a structured validation error dict (error_type='validation').\n"
        "2) Self-correct by calling execute_code again, but WITHOUT a session (still stateless), same code.\n"
        "3) Return JSON with keys: saw_validation_error (bool), final_stdout (string).\n"
        "Do not invent outputs; only use what you observe from tool results.\n",
        response_format=ValidationRecoveryResult,
        persist=False,
        clarification_enabled=False,
    )
    try:
        res = await asyncio.wait_for(handle.result(), timeout=170)
        assert isinstance(res, ValidationRecoveryResult)
        assert res.saw_validation_error is True
        assert "hi" in (res.final_stdout or "")
    finally:
        await actor.close()
