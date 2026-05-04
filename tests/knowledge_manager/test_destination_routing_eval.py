from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import pytest

from tests.destination_routing_helpers import (
    RoutingScenario,
    assert_tool_destination,
    llm_config as llm_config,  # noqa: F401
    rows_containing,
)
from unity.common.async_tool_loop import start_async_tool_loop
from unity.common.llm_client import new_llm_client
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.knowledge_manager.prompt_builders import build_update_prompt

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


def _knowledge_table_rows_containing(
    root: str,
    table: str,
    sentinel: str,
) -> list[dict[str, Any]]:
    return rows_containing(f"{root}/Knowledge/{table}", sentinel)


async def _run_knowledge_update(
    *,
    manager: KnowledgeManager,
    llm_config: dict[str, str],
    message: str,
) -> list[dict[str, Any]]:
    tools = dict(manager.get_tools("update"))
    client = new_llm_client(**llm_config)
    client.set_system_message(
        build_update_prompt(
            tools=tools,
            table_schemas_json=json.dumps(
                manager._tables_overview(),
                indent=4,
                sort_keys=True,
            ),
            include_activity=True,
        ).to_list(),
    )
    handle = start_async_tool_loop(
        client,
        message=message,
        tools=tools,
        loop_id="destination-routing-knowledge-update",
        tool_policy=manager._default_update_tool_policy,
    )
    try:
        await asyncio.wait_for(handle.result(), timeout=240)
        return handle.get_history()
    finally:
        if not handle.done():
            await handle.stop("test cleanup")


@pytest.mark.asyncio
async def test_team_knowledge_routes_to_the_matching_shared_space(llm_config):
    """A team operational note lands in the relevant shared Knowledge root."""

    scenario = RoutingScenario("knowledge_shared")
    scenario.setup()
    manager = KnowledgeManager(include_contacts=False)
    sentinel = f"BLUEJAY-{uuid.uuid4().hex[:10]}"

    try:
        manager._create_table(
            name="OperationalNotes",
            columns={"note": "str", "source": "str"},
        )
        manager._create_table(
            name="OperationalNotes",
            columns={"note": "str", "source": "str"},
            destination=scenario.patch_destination,
        )
        manager._create_table(
            name="OperationalNotes",
            columns={"note": "str", "source": "str"},
            destination=scenario.research_destination,
        )

        messages = await _run_knowledge_update(
            manager=manager,
            llm_config=llm_config,
            message=(
                "Capture this as durable operational knowledge: when two overnight "
                f"compressor callbacks mention {sentinel}, the dispatch lead should "
                "escalate the incident to the Bluejay rota before the morning handoff."
            ),
        )

        assert_tool_destination(messages, "add_rows", scenario.patch_destination)
        assert _knowledge_table_rows_containing(
            f"Spaces/{scenario.patch_space_id}",
            "OperationalNotes",
            sentinel,
        )
        assert not _knowledge_table_rows_containing(
            scenario.context,
            "OperationalNotes",
            sentinel,
        )
        assert not _knowledge_table_rows_containing(
            f"Spaces/{scenario.research_space_id}",
            "OperationalNotes",
            sentinel,
        )
    finally:
        scenario.teardown()
