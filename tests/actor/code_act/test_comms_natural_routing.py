"""Verify natural-language CodeAct requests route to `primitives.comms`.

This test intentionally avoids naming the primitive in the user request.
It checks whether the actor can infer the correct assistant-owned comms
surface from prompt context alone when given a natural outbound request.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.actor.state_managers.utils import (
    extract_code_act_execute_code_snippets,
    extract_code_act_execute_function_names,
    make_code_act_actor,
    wait_for_recorded_primitives_call,
)
from unity.manager_registry import ManagerRegistry

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


def _force_simulated_contacts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch ContactManager to simulated mode for deterministic comms routing."""
    from unity.settings import SETTINGS

    monkeypatch.setenv("UNITY_CONTACT_IMPL", "simulated")
    monkeypatch.setattr(SETTINGS.contact, "IMPL", "simulated", raising=False)
    ManagerRegistry.clear()


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_natural_unify_message_request_routes_to_comms_primitive(
    monkeypatch: pytest.MonkeyPatch,
):
    """A natural outbound request should independently reach for comms primitives.

    The request names the recipient and medium in plain English, but never
    mentions `primitives.comms` or any concrete method name. The actor should
    still infer that the correct assistant-owned tool is
    `primitives.comms.send_unify_message`.
    """

    _force_simulated_contacts(monkeypatch)

    contact_manager = ManagerRegistry.get_contact_manager()
    contact_manager.update_contact(
        contact_id=2,
        first_name="Alice",
        should_respond=True,
    )

    mock_broker = MagicMock()
    mock_broker.publish = AsyncMock(return_value=0)

    handle = None
    with (
        patch(
            "unity.comms.primitives.get_event_broker",
            return_value=mock_broker,
        ),
        patch(
            "unity.comms.primitives.comms_utils.send_unify_message",
            new=AsyncMock(return_value={"success": True}),
        ),
    ):
        async with make_code_act_actor(
            impl="simulated",
            exposed_managers={"comms"},
        ) as (actor, _, calls):
            try:
                handle = await actor.act(
                    "Please send contact 2 a quick Unify message saying the design "
                    "review moved to 3pm tomorrow. Keep it short and do not ask "
                    "clarifying questions.",
                    clarification_enabled=False,
                )

                await wait_for_recorded_primitives_call(
                    calls,
                    "primitives.comms.send_unify_message",
                    timeout=120,
                )
                result = await asyncio.wait_for(handle.result(), timeout=120)
                assert result is not None

                unexpected_comms_calls = [
                    call
                    for call in calls
                    if call.startswith("primitives.comms.")
                    and call != "primitives.comms.send_unify_message"
                ]
                assert not unexpected_comms_calls, (
                    "Expected the actor to settle on the Unify outbound primitive "
                    f"for this request, but saw extra comms calls: {unexpected_comms_calls}. "
                    f"Full call sequence: {calls}"
                )

                execute_function_names = extract_code_act_execute_function_names(handle)
                execute_code_snippets = extract_code_act_execute_code_snippets(handle)
                assert any(
                    name == "primitives.comms.send_unify_message"
                    for name in execute_function_names
                ) or any(
                    "primitives.comms.send_unify_message" in snippet
                    for snippet in execute_code_snippets
                ), (
                    "Expected the actor transcript to show an explicit routing step "
                    "to primitives.comms.send_unify_message, but it did not.\n"
                    f"execute_function targets: {execute_function_names}\n"
                    f"execute_code snippets:\n{chr(10).join(execute_code_snippets)}"
                )
            finally:
                try:
                    if handle is not None and not handle.done():
                        await handle.stop("test cleanup")
                except Exception:
                    pass
