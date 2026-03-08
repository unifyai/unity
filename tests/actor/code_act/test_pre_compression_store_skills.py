"""Extra compression tools: store_skills alongside compress_context at 70%.

When the context window hits the 70% threshold and ``extra_compression_tools``
is configured, the loop exposes those tools alongside ``compress_context``
(with ``tool_choice="required"``).  The prompt guides the LLM to call
``store_skills`` first **if** the trajectory contains unstored skills worth
preserving, then ``compress_context``.  The LLM may skip ``store_skills``
if it judges there is nothing new to store.

This test monkeypatches ``context_over_threshold`` to simulate reaching the
70% threshold after the LLM has executed non-trivial code, and verifies
that the infrastructure works and that ordering is correct when both tools
are called.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unity.actor.code_act_actor import CodeActActor
from unity.common.async_tool_loop import AsyncToolLoopHandle
from unity.function_manager.function_manager import FunctionManager


class _StubGuidanceManager:
    """Minimal GuidanceManager stand-in with the methods the actor registers."""

    def search(self, references=None, k=10):
        return []

    def filter(self, filter=None, offset=0, limit=100):
        return []

    def add_guidance(self, *, title, content, function_ids=None):
        return {"details": {"guidance_id": 1}}

    def update_guidance(
        self,
        *,
        guidance_id,
        title=None,
        content=None,
        function_ids=None,
    ):
        return {"details": {"guidance_id": guidance_id}}

    def delete_guidance(self, *, guidance_id):
        return {"deleted": True}


def _make_delayed_threshold(trigger_after: int = 2):
    """Build a ``context_over_threshold`` replacement that triggers the 70%
    threshold only after *trigger_after* checks (giving the LLM enough
    turns to build a meaningful trajectory)."""
    _check_count = [0]

    def _fake(n_tokens: int, threshold: float, max_input_tokens: int) -> bool:
        if threshold >= 0.7:
            _check_count[0] += 1
            return _check_count[0] > trigger_after
        return False

    return _fake


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_compress_threshold_exposes_extra_tools_and_compress_is_called():
    """When the 70% threshold fires with ``extra_compression_tools``, the
    loop exposes ``store_skills`` + ``compress_context``, and the LLM
    eventually calls ``compress_context``.

    If the LLM also calls ``store_skills``, it must precede
    ``compress_context``."""

    fm = FunctionManager(include_primitives=False)
    gm = _StubGuidanceManager()

    actor = CodeActActor(
        function_manager=fm,
        guidance_manager=gm,
        timeout=120,
    )

    mock_proactive_handle = MagicMock()

    async def _quick_storage_result():
        return "Stored 1 function: test_helper"

    mock_proactive_handle.result = _quick_storage_result
    mock_proactive_handle.done = MagicMock(return_value=True)

    async def _mock_restart(self):
        async def _done():
            return "Context compressed. Continuing from where you left off."

        self._task = asyncio.create_task(_done())

    with (
        patch(
            "unity.common._async_tool.loop.context_over_threshold",
            _make_delayed_threshold(trigger_after=2),
        ),
        patch(
            "unity.actor.code_act_actor._start_proactive_storage_loop",
            return_value=mock_proactive_handle,
        ),
        patch(
            "unity.actor.code_act_actor._start_storage_check_loop",
            return_value=None,
        ),
        patch.object(
            AsyncToolLoopHandle,
            "_restart_with_compressed_context",
            _mock_restart,
        ),
        patch(
            "unity.actor.code_act_actor.publish_manager_method_event",
            new_callable=AsyncMock,
        ),
    ):
        try:
            handle = await actor.act(
                "Write a Python function that computes the Fibonacci sequence "
                "iteratively. Execute it with n=10 and show me the result.",
                can_store=True,
                persist=False,
                clarification_enabled=False,
            )
            result = await asyncio.wait_for(handle.result(), timeout=120)

            history = handle.get_history()
            tool_names: list[str] = []
            for msg in history:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        tool_names.append(tc["function"]["name"])

            assert (
                "compress_context" in tool_names
            ), f"Expected compress_context to be called, got: {tool_names}"

            if "store_skills" in tool_names:
                ss_idx = tool_names.index("store_skills")
                cc_idx = tool_names.index("compress_context")
                assert ss_idx < cc_idx, (
                    f"store_skills (index {ss_idx}) must precede "
                    f"compress_context (index {cc_idx}) when both are "
                    f"called. Full order: {tool_names}"
                )
        finally:
            try:
                await actor.close()
            except Exception:
                pass
