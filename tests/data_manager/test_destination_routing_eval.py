from __future__ import annotations

import pytest

from tests.destination_routing_helpers import (
    PATCH_SPACE_DESTINATION,
    DestinationRoutingDecision,
    routing_decision_prompt,
    tool_name,
)
from unity.common.reasoning import reason

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


@pytest.mark.asyncio
async def test_team_kpi_data_routes_to_patch_space():
    decision = await reason(
        routing_decision_prompt(
            "Insert today's Patch-1 open-work-order count into our KPI table: "
            "47 open jobs, 12 overdue, and 6 waiting on parts. The whole repairs "
            "patch should be able to query it during tomorrow morning's standup.",
        ),
        response_format=DestinationRoutingDecision,
    )

    assert decision.manager.lower().replace("_", "") == "datamanager"
    assert tool_name(decision) == "insert_rows"
    assert decision.destination == PATCH_SPACE_DESTINATION
    assert decision.clarification_requested is False
