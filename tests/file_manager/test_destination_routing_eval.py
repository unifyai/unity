from __future__ import annotations

import pytest

from tests.destination_routing_helpers import (
    PATCH_SPACE_DESTINATION,
    PERSONAL_DESTINATIONS,
    DestinationRoutingDecision,
    routing_decision_prompt,
    tool_name,
)
from unity.common.reasoning import reason

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


@pytest.mark.asyncio
async def test_team_reference_file_routes_to_patch_space():
    decision = await reason(
        routing_decision_prompt(
            "Save the new Patch-1 customer-handling SOP so every operative can "
            "look it up before visiting tenants. The file is attached as "
            "patch-1-customer-handling-sop.pdf.",
        ),
        response_format=DestinationRoutingDecision,
    )

    assert decision.manager.lower().replace("_", "") == "filemanager"
    assert tool_name(decision) in {"ingest_files", "save_attachment"}
    assert decision.destination == PATCH_SPACE_DESTINATION
    assert decision.clarification_requested is False


@pytest.mark.asyncio
async def test_personal_screenshot_stays_personal():
    decision = await reason(
        routing_decision_prompt(
            "Save this browser screenshot for me so I can refer back to it later "
            "while I finish my own notes.",
        ),
        response_format=DestinationRoutingDecision,
    )

    assert decision.manager.lower().replace("_", "") == "filemanager"
    assert tool_name(decision) in {"ingest_files", "save_attachment"}
    assert decision.destination in PERSONAL_DESTINATIONS
    assert decision.clarification_requested is False
