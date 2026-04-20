"""Discovery-first policy: CodeActActor requires FM + GM discovery before free rein.

Verifies that when the CodeActActor has both FunctionManager and GuidanceManager
tools, the default tool policy gates on both being called at least once.  The
prompt advises calling both on the first turn as parallel tool calls.
"""

import asyncio

import pytest

from tests.async_helpers import _wait_for_condition
from tests.helpers import _handle_project
from unity.actor.code_act_actor import CodeActActor
from unity.actor.environments import StateManagerEnvironment
from unity.function_manager.function_manager import FunctionManager
from unity.function_manager.primitives import Primitives, PrimitiveScope
from unity.guidance_manager.guidance_manager import GuidanceManager
from unity.manager_registry import ManagerRegistry

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


def _force_simulated_managers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch manager implementations to simulated mode for a deterministic eval."""
    from unity.settings import SETTINGS

    for name in (
        "CONTACT",
        "TASK",
        "TRANSCRIPT",
        "KNOWLEDGE",
        "GUIDANCE",
        "SECRET",
        "WEB",
        "FILE",
        "DATA",
    ):
        monkeypatch.setenv(f"UNITY_{name}_IMPL", "simulated")
        attr = name.lower()
        if hasattr(SETTINGS, attr):
            monkeypatch.setattr(
                getattr(SETTINGS, attr),
                "IMPL",
                "simulated",
                raising=False,
            )

    ManagerRegistry.clear()


def _assistant_tool_names(history: list[dict]) -> list[str]:
    """Flatten assistant tool-call names from a handle transcript."""
    tool_names: list[str] = []
    for msg in history:
        if msg.get("role") != "assistant":
            continue
        for tool_call in msg.get("tool_calls") or []:
            tool_names.append(tool_call["function"]["name"])
    return tool_names


async def _wait_for_tool_result_in_history(
    handle,
    tool_name: str,
    *,
    timeout: float = 120.0,
) -> None:
    """Wait until a tool result with *tool_name* appears in the transcript."""

    async def _predicate():
        return any(
            msg.get("role") == "tool" and msg.get("name") == tool_name
            for msg in handle.get_history()
        )

    await _wait_for_condition(_predicate, poll=0.1, timeout=timeout)


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
    gm = GuidanceManager()

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


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_discovery_first_prefers_minimal_exact_call_over_execute_code(
    monkeypatch: pytest.MonkeyPatch,
):
    """An exact single-call instruction should stay on the execute_function path."""
    _force_simulated_managers(monkeypatch)

    scope = PrimitiveScope(scoped_managers=frozenset({"contacts"}))
    primitives = Primitives(primitive_scope=scope)
    env = StateManagerEnvironment(primitives)
    fm = FunctionManager()
    gm = GuidanceManager()

    actor = CodeActActor(
        environments=[env],
        function_manager=fm,
        guidance_manager=gm,
        timeout=200,
    )
    handle = None

    try:
        handle = await actor.act(
            "Use the required discovery-first workflow: search both "
            "FunctionManager and GuidanceManager first. After discovery, choose "
            "the minimal correct execution path. One exact primitive call is "
            "sufficient here: primitives.contacts.ask(text='Find all contacts "
            "located in Berlin'). Do not write custom code or compose multiple "
            "steps. The exact primitive call is still valid even if search "
            "results look sparse.",
            can_compose=True,
            clarification_enabled=False,
        )

        await _wait_for_tool_result_in_history(
            handle,
            "execute_function",
            timeout=120,
        )
        result = await asyncio.wait_for(handle.result(), timeout=120)
        assert result is not None

        history = handle.get_history()
        tool_names = _assistant_tool_names(history)

        assert any(name.startswith("FunctionManager_") for name in tool_names), (
            "Expected at least one FunctionManager discovery call before the "
            f"execution step. Got tool calls: {tool_names}"
        )
        assert any(name.startswith("GuidanceManager_") for name in tool_names), (
            "Expected at least one GuidanceManager discovery call before the "
            f"execution step. Got tool calls: {tool_names}"
        )
        assert "execute_function" in tool_names, (
            "Expected the actor to use execute_function once the exact primitive "
            f"call was identified. Got tool calls: {tool_names}"
        )
        assert "execute_code" not in tool_names, (
            "The actor should not escalate to execute_code when one exact "
            f"primitive call is sufficient. Got tool calls: {tool_names}"
        )
    finally:
        try:
            if handle is not None and not handle.done():
                await handle.stop("test cleanup")
        except Exception:
            pass
        try:
            await actor.close()
        except Exception:
            pass
