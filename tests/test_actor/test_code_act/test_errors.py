import asyncio

import pytest
from unittest.mock import AsyncMock

from unity.actor.code_act_actor import CodeActActor


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_execute_python_code_returns_error_when_sandbox_not_bound():
    """execute_python_code should fail gracefully if the sandbox ContextVar is missing."""
    actor = CodeActActor(headless=True, computer_mode="mock")
    actor._computer_primitives.navigate = AsyncMock(return_value=None)
    actor._computer_primitives.act = AsyncMock(return_value="Action completed")
    actor._computer_primitives.observe = AsyncMock(return_value="Page content observed")

    tools = actor.get_tools("act")
    execute_python_code = tools["execute_python_code"]

    out = await execute_python_code("run", "print('x')")
    assert isinstance(out, dict)
    assert out.get("error") is not None
    assert "not bound" in str(out.get("error")).lower()

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
