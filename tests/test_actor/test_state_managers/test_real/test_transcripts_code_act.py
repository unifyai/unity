"""Real TranscriptManager routing tests for CodeActActor.

These mirror `test_transcripts.py` but use CodeActActor (code-first tool loop).
"""

from datetime import datetime, timezone

import pytest

from tests.helpers import _handle_project
from tests.test_actor.test_state_managers.utils import (
    assert_code_act_function_manager_used,
    extract_code_act_execute_python_code_snippets,
    make_code_act_actor,
)
from unity.contact_manager.types.contact import Contact
from unity.function_manager.function_manager import FunctionManager
from unity.manager_registry import ManagerRegistry


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager():
    """CodeAct routes transcript question → primitives.transcripts.ask."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):
        tm = ManagerRegistry.get_transcript_manager()

        alice = Contact(
            first_name="Alice",
            surname="Smith",
            email_address="alice.smith@example.com",
        )
        bob = Contact(
            first_name="Bob",
            surname="Jones",
            email_address="bob.jones@example.com",
        )

        tm.log_first_message_in_new_exchange(
            {
                "medium": "email",
                "sender_id": alice,
                "receiver_ids": [bob],
                "timestamp": datetime.now(timezone.utc),
                "content": "Subject: Q3 Budget\nBody: Final numbers are ready for review.",
            },
        )

        handle = await actor.act(
            "Show the most recent message that mentions the budget.",
            clarification_enabled=False,
        )
        result = await handle.result()

        assert result and len(str(result)) > 0
        assert "primitives.transcripts.ask" in calls
        assert all(c.startswith("primitives.transcripts.") for c in calls)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager_memoized():
    """CodeAct uses FunctionManager (when available) for transcript queries."""
    fm = FunctionManager()
    implementation = """
async def ask_transcripts_question(question: str, response_format=None) -> str:
    \"\"\"Query transcripts/messages via the transcripts manager (read-only).\"\"\"
    handle = await primitives.transcripts.ask(question, response_format=response_format)
    return await handle.result()
"""
    fm.add_functions(implementations=implementation, overwrite=True)

    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
        function_manager=fm,
    ) as (actor, _primitives, calls):
        tm = ManagerRegistry.get_transcript_manager()

        alice = Contact(
            first_name="Alice",
            surname="Smith",
            email_address="alice.smith@example.com",
        )
        bob = Contact(
            first_name="Bob",
            surname="Jones",
            email_address="bob.jones@example.com",
        )

        tm.log_first_message_in_new_exchange(
            {
                "medium": "email",
                "sender_id": alice,
                "receiver_ids": [bob],
                "timestamp": datetime.now(timezone.utc),
                "content": "Subject: Q3 Budget\nBody: Final numbers are ready for review.",
            },
        )

        handle = await actor.act(
            "Show the most recent message that mentions the budget.",
            clarification_enabled=False,
        )
        result = await handle.result()

        assert result and len(str(result)) > 0
        assert_code_act_function_manager_used(handle)
        snippets = "\n\n".join(extract_code_act_execute_python_code_snippets(handle))
        assert "ask_transcripts_question" in snippets

        assert "primitives.transcripts.ask" in calls
        assert all(c.startswith("primitives.transcripts.") for c in calls)
