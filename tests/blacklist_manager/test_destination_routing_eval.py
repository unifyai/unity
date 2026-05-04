from __future__ import annotations

import pytest

from tests.destination_routing_helpers import (
    FAMILY_SPACE_DESTINATION,
    PERSONAL_DESTINATIONS,
    DestinationRoutingDecision,
    assert_personal_or_clarification,
    routing_decision_prompt,
    tool_name,
)
from unity.common.reasoning import reason

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


@pytest.mark.asyncio
async def test_household_blacklist_routes_to_family_space():
    decision = await reason(
        routing_decision_prompt(
            "Block 0800 999 0001 on every adult's phone. It has been calling "
            "the kids after school and the household agreed nobody should answer.",
        ),
        response_format=DestinationRoutingDecision,
    )

    assert decision.manager.lower().replace("_", "") == "blacklistmanager"
    assert tool_name(decision) == "create_blacklist_entry"
    assert decision.destination == FAMILY_SPACE_DESTINATION
    assert decision.clarification_requested is False


@pytest.mark.asyncio
async def test_personal_blacklist_stays_personal():
    decision = await reason(
        routing_decision_prompt(
            "Block 07700 900111 for me. It is an ex I do not want to hear from, "
            "but this is only about my own calls and messages.",
        ),
        response_format=DestinationRoutingDecision,
    )

    assert decision.manager.lower().replace("_", "") == "blacklistmanager"
    assert tool_name(decision) == "create_blacklist_entry"
    assert decision.destination in PERSONAL_DESTINATIONS
    assert decision.clarification_requested is False


@pytest.mark.asyncio
async def test_ambiguous_blacklist_does_not_guess_wider():
    decision = await reason(
        routing_decision_prompt("Block this number: 07700 900222."),
        response_format=DestinationRoutingDecision,
    )

    assert tool_name(decision) in {"create_blacklist_entry", "request_clarification"}
    assert_personal_or_clarification(decision)
