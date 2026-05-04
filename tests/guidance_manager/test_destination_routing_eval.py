from __future__ import annotations

import uuid

import pytest

from tests.destination_routing_helpers import (
    RoutingScenario,
    assert_personal_tool_destination,
    assert_tool_destination,
    llm_config as llm_config,  # noqa: F401
    rows_containing,
    run_direct_routing_loop,
)
from unity.common.llm_helpers import methods_to_tool_dict
from unity.guidance_manager.guidance_manager import GuidanceManager

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


@pytest.mark.asyncio
async def test_private_preference_stays_in_personal_guidance(llm_config):
    """A user-specific working preference does not leak into either space."""

    scenario = RoutingScenario("guidance_personal")
    scenario.setup()
    manager = GuidanceManager()
    sentinel = f"PRIVATE-CADENCE-{uuid.uuid4().hex[:10]}"

    try:
        messages = await run_direct_routing_loop(
            llm_config=llm_config,
            tools=methods_to_tool_dict(manager.add_guidance, include_class_name=True),
            accessible_spaces=scenario.space_summaries,
            loop_id="destination-routing-guidance-personal",
            message=(
                f"Remember for how you work with me: {sentinel}. When drafting my "
                "end-of-day notes, keep the tone compact and include only the two "
                "decisions I personally need to make tomorrow."
            ),
        )

        assert_personal_tool_destination(messages, "add_guidance")
        assert rows_containing(manager._ctx, sentinel)
        assert not rows_containing(
            f"Spaces/{scenario.patch_space_id}/Guidance",
            sentinel,
        )
        assert not rows_containing(
            f"Spaces/{scenario.research_space_id}/Guidance",
            sentinel,
        )
    finally:
        scenario.teardown()


@pytest.mark.asyncio
async def test_team_guidance_routes_to_the_matching_shared_space(llm_config):
    """A team SOP lands in the shared Guidance root for the relevant domain."""

    scenario = RoutingScenario("guidance_shared")
    scenario.setup()
    manager = GuidanceManager()
    sentinel = f"FIELD-SOP-{uuid.uuid4().hex[:10]}"

    try:
        messages = await run_direct_routing_loop(
            llm_config=llm_config,
            tools=methods_to_tool_dict(manager.add_guidance, include_class_name=True),
            accessible_spaces=scenario.space_summaries,
            loop_id="destination-routing-guidance-shared",
            message=(
                f"Store a guidance rule titled '{sentinel}': for recurring patch "
                "outages, coordinator replies should lead with current customer "
                "impact, contractor ETA, and next escalation owner before any "
                "background explanation."
            ),
        )

        assert_tool_destination(messages, "add_guidance", scenario.patch_destination)
        assert rows_containing(f"Spaces/{scenario.patch_space_id}/Guidance", sentinel)
        assert not rows_containing(manager._ctx, sentinel)
        assert not rows_containing(
            f"Spaces/{scenario.research_space_id}/Guidance",
            sentinel,
        )
    finally:
        scenario.teardown()
