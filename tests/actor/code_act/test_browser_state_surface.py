import pytest

from unity.actor.code_act_actor import CodeActActor, ExecutionResult
from unity.common._async_tool.formatting import serialize_tool_content


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_execute_code_surfaces_browser_state_when_browser_used():
    """
    When sandbox execution invokes browser primitives, the tool result should
    include the current browser state so the model can reason about the page
    without requiring an additional perception step.
    """
    actor = CodeActActor(headless=True, computer_mode="mock", timeout=30)
    try:
        tools = actor._build_tools()
        execute_code = tools["execute_code"]

        res = await execute_code(
            thought="Navigate so browser tools are exercised",
            language="python",
            state_mode="stateful",
            code="await computer_primitives.navigate('https://example.com')",
        )

        assert isinstance(res, ExecutionResult)
        assert res.browser_used is True

        llm_content = serialize_tool_content(
            tool_name="execute_code",
            payload=res,
            is_final=True,
        )
        rendered = str(llm_content)

        assert "browser_state" in rendered
        assert "screenshot" in rendered
        assert "url" in rendered
    finally:
        try:
            await actor.close()
        except Exception:
            pass
