from __future__ import annotations

import uuid

import pytest
import unify

from tests.destination_routing_helpers import (
    RoutingScenario,
    assert_tool_destination,
    llm_config as llm_config,  # noqa: F401
    rows_containing,
    run_direct_routing_loop,
)
from unity.common.llm_helpers import methods_to_tool_dict
from unity.function_manager.function_manager import FunctionManager

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


@pytest.mark.asyncio
async def test_team_function_routes_to_the_matching_shared_space(llm_config):
    """A team automation helper lands in shared Functions for that team."""

    scenario = RoutingScenario("function_shared")
    scenario.setup()
    manager = FunctionManager(include_primitives=False)
    function_name = f"normalize_patch_ticket_{uuid.uuid4().hex[:8]}"

    try:
        messages = await run_direct_routing_loop(
            llm_config=llm_config,
            tools=methods_to_tool_dict(manager.add_functions, include_class_name=True),
            accessible_spaces=scenario.space_summaries,
            loop_id="destination-routing-functions-shared",
            message=(
                "Create the small Python helper our field dispatch coordinators can "
                f"reuse named {function_name}. It should accept a ticket reference "
                "string and return only the uppercase alphanumeric characters and "
                "hyphens, with surrounding whitespace removed."
            ),
        )

        assert_tool_destination(messages, "add_functions", scenario.patch_destination)
        assert [
            row
            for row in unify.get_logs(
                context=f"Spaces/{scenario.patch_space_id}/Functions/Compositional",
                filter=f"name == '{function_name}'",
                limit=10,
            )
        ]
        assert (
            unify.get_logs(
                context=manager._compositional_ctx,
                filter=f"name == '{function_name}'",
                limit=10,
            )
            == []
        )
        assert (
            rows_containing(
                f"Spaces/{scenario.research_space_id}/Functions/Compositional",
                function_name,
            )
            == []
        )
    finally:
        scenario.teardown()
