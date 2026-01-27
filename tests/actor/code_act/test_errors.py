import asyncio

import pytest
from unittest.mock import AsyncMock

from unity.actor.code_act_actor import CodeActActor


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_execute_code_can_run_without_bound_sandbox():
    """execute_code should work even if the sandbox ContextVar is missing."""
    actor = CodeActActor(headless=True, computer_mode="mock")
    actor._computer_primitives.navigate = AsyncMock(return_value=None)
    actor._computer_primitives.act = AsyncMock(return_value="Action completed")
    actor._computer_primitives.observe = AsyncMock(return_value="Page content observed")

    tools = actor.get_tools("act")
    execute_code = tools["execute_code"]

    out = await execute_code(
        "run",
        "print('x')",
        language="python",
        state_mode="stateful",
        session_id=0,
        venv_id=None,
    )
    assert isinstance(out, dict)
    assert out.get("error") is None
    assert "x" in str(out.get("stdout") or "")

    await actor.close()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_act_capacity_timeout_is_fast_when_configured():
    """act() should raise a clear error when semaphore acquisition times out."""
    actor = CodeActActor(headless=True, computer_mode="mock")
    actor._computer_primitives.navigate = AsyncMock(return_value=None)
    actor._computer_primitives.act = AsyncMock(return_value="Action completed")
    actor._computer_primitives.observe = AsyncMock(return_value="Page content observed")

    # Exhaust capacity and ensure timeout is fast for the test.
    actor._act_semaphore = asyncio.Semaphore(0)
    actor._act_semaphore_timeout_s = 0.01

    with pytest.raises(RuntimeError) as e:
        await actor.act("hello")

    assert "at capacity" in str(e.value).lower()

    await actor.close()
