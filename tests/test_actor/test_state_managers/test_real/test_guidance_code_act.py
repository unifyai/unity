"""Real GuidanceManager routing tests for CodeActActor.

These mirror `test_guidance.py` but use CodeActActor (code-first tool loop).
"""

import pytest

from tests.helpers import _handle_project
from tests.test_actor.test_state_managers.utils import (
    assert_code_act_function_manager_used,
    extract_code_act_execute_python_code_snippets,
    make_code_act_actor,
)
from unity.function_manager.function_manager import FunctionManager
from unity.manager_registry import ManagerRegistry


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager():
    """CodeAct routes read-only guidance question → primitives.guidance.ask."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):
        gm = ManagerRegistry.get_guidance_manager()
        gm._add_guidance(
            title="Onboarding Overview",
            content="We walk through onboarding steps for new users.",
        )

        handle = await actor.act(
            "What does the Guidance entry titled 'Onboarding Overview' say?",
            clarification_enabled=False,
        )
        result = await handle.result()

        assert "onboarding" in str(result).lower()
        assert "primitives.guidance.ask" in calls
        assert all(c.startswith("primitives.guidance.") for c in calls)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager_memoized():
    """CodeAct uses FunctionManager (when available) for guidance queries."""
    fm = FunctionManager()
    implementation = """
async def ask_guidance_question(question: str, response_format=None) -> str:
    \"\"\"Query internal guidance via the guidance manager (read-only).\"\"\"
    handle = await primitives.guidance.ask(question, response_format=response_format)
    return await handle.result()
"""
    fm.add_functions(implementations=implementation, overwrite=True)

    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
        function_manager=fm,
    ) as (actor, _primitives, calls):
        gm = ManagerRegistry.get_guidance_manager()
        gm._add_guidance(
            title="Onboarding Overview",
            content="We walk through onboarding steps for new users.",
        )

        handle = await actor.act(
            "What does the Guidance entry titled 'Onboarding Overview' say?",
            clarification_enabled=False,
        )
        result = await handle.result()

        assert "onboarding" in str(result).lower()
        assert_code_act_function_manager_used(handle)
        snippets = "\n\n".join(extract_code_act_execute_python_code_snippets(handle))
        assert "ask_guidance_question" in snippets

        assert "primitives.guidance.ask" in calls
        assert all(c.startswith("primitives.guidance.") for c in calls)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_update_calls_manager():
    """CodeAct routes guidance mutation → primitives.guidance.update."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):
        gm = ManagerRegistry.get_guidance_manager()

        handle = await actor.act(
            "Create a new guidance entry titled 'Incident Response' with the content "
            "'Escalate sev-1 to on-call within 5 minutes.'",
            clarification_enabled=False,
        )
        await handle.result()

        assert "primitives.guidance.update" in calls
        assert "primitives.guidance.ask" not in calls

        rows = gm._filter(filter="title == 'Incident Response'")
        assert len(rows) > 0, "Expected 'Incident Response' entry to be created"
        content = rows[0].content.lower()
        assert "sev-1" in content or "on-call" in content


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_update_calls_manager_memoized():
    """CodeAct uses FunctionManager (when available) for guidance mutations."""
    fm = FunctionManager()
    implementation = """
async def create_guidance_entry(title: str, content: str) -> str:
    \"\"\"Create a guidance entry via the guidance manager.\"\"\"
    handle = await primitives.guidance.update(
        f"Create a new guidance entry titled '{title}' with the content '{content}'."
    )
    return await handle.result()
"""
    fm.add_functions(implementations=implementation, overwrite=True)

    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
        function_manager=fm,
    ) as (actor, _primitives, calls):
        gm = ManagerRegistry.get_guidance_manager()

        handle = await actor.act(
            "Create a new guidance entry titled 'Incident Response' with the content "
            "'Escalate sev-1 to on-call within 5 minutes.'",
            clarification_enabled=False,
        )
        await handle.result()

        assert_code_act_function_manager_used(handle)
        snippets = "\n\n".join(extract_code_act_execute_python_code_snippets(handle))
        assert "create_guidance_entry" in snippets

        assert "primitives.guidance.update" in calls
        assert "primitives.guidance.ask" not in calls

        rows = gm._filter(filter="title == 'Incident Response'")
        assert len(rows) > 0, "Expected 'Incident Response' entry to be created"
        content = rows[0].content.lower()
        assert "sev-1" in content or "on-call" in content
