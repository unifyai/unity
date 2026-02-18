"""Discovery-first policy: CodeActActor requires FM + GM discovery before free rein.

Verifies that when the CodeActActor has both FunctionManager and GuidanceManager
tools, the default tool policy gates on both being called at least once.  The
prompt advises calling both on the first turn as parallel tool calls.
"""

import asyncio

import pytest

from tests.helpers import _handle_project
from unity.actor.code_act_actor import CodeActActor
from unity.function_manager.function_manager import FunctionManager

pytestmark = pytest.mark.eval


class _StubGuidanceManager:
    """Minimal GuidanceManager stand-in with the method signatures needed to
    create the tool surface.  All read operations return empty results."""

    def search(self, references=None, k=10):
        """Search for guidance entries by semantic similarity to reference content."""
        return []

    def filter(self, filter=None, offset=0, limit=100):
        """Filter guidance entries using a Python filter expression."""
        return []

    def add_guidance(self, *, title, content, function_ids=None):
        """Add a guidance entry describing a compositional workflow or playbook."""
        return {"details": {"guidance_id": 1}}

    def update_guidance(
        self,
        *,
        guidance_id,
        title=None,
        content=None,
        function_ids=None,
    ):
        """Update an existing guidance entry."""
        return {"details": {"guidance_id": guidance_id}}

    def delete_guidance(self, *, guidance_id):
        """Delete a guidance entry by ID."""
        return {"deleted": True}


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_discovery_first_parallel_fm_and_gm():
    """Both FM and GM discovery calls should appear on the first assistant turn.

    The discovery-first policy restricts tool visibility until both have been
    called.  The prompt explicitly advises issuing them as parallel tool calls
    in a single message.  We verify:

    1. The first assistant message with tool_calls contains at least one
       FunctionManager call AND at least one GuidanceManager call.
    2. The actor eventually produces a final result (the full tool set
       unlocked after discovery).
    """
    fm = FunctionManager(include_primitives=False)
    gm = _StubGuidanceManager()

    actor = CodeActActor(
        function_manager=fm,
        guidance_manager=gm,
        timeout=120,
    )

    try:
        handle = await actor.act(
            "What is 2 + 2?",
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=120)
        assert result is not None

        history = handle.get_history()
        first_assistant_with_tools = next(
            (
                m
                for m in history
                if m.get("role") == "assistant" and m.get("tool_calls")
            ),
            None,
        )
        assert (
            first_assistant_with_tools is not None
        ), "Expected at least one assistant message with tool_calls"

        tool_names = [
            tc["function"]["name"] for tc in first_assistant_with_tools["tool_calls"]
        ]
        has_fm = any(n.startswith("FunctionManager_") for n in tool_names)
        has_gm = any(n.startswith("GuidanceManager_") for n in tool_names)

        assert has_fm and has_gm, (
            f"First assistant turn should contain both a FunctionManager and a "
            f"GuidanceManager discovery call (issued in parallel).  "
            f"Got tool calls: {tool_names}"
        )
    finally:
        try:
            if not handle.done():
                await handle.stop("test cleanup")
        except Exception:
            pass
        try:
            await actor.close()
        except Exception:
            pass
