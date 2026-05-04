"""CodeActActor evals for dashboard shared-space routing."""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager

import pytest

from tests.actor.state_managers.utils import make_code_act_actor
from unity.common.accessible_spaces_block import build_accessible_spaces_block
from unity.common.context_registry import ContextRegistry
from unity.manager_registry import ManagerRegistry
from unity.memory_manager import broader_context
from unity.session_details import SESSION_DETAILS, SpaceSummary

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]

PATCH_SPACE_ID = 7107
EXECUTIVE_SPACE_ID = 9109
EVAL_TIMEOUT_SECONDS = 300.0


def _set_dashboard_spaces(
    *,
    space_ids: list[int],
    space_summaries: list[SpaceSummary],
) -> None:
    SESSION_DETAILS.space_ids = list(space_ids)
    SESSION_DETAILS.space_summaries = list(space_summaries)
    ManagerRegistry.clear()
    ContextRegistry.clear()
    broader_context.reset()


@contextmanager
def _dashboard_spaces():
    original_space_ids = list(SESSION_DETAILS.space_ids)
    original_space_summaries = list(SESSION_DETAILS.space_summaries)
    patch_summary = SpaceSummary(
        space_id=PATCH_SPACE_ID,
        name="Patch-1 Operations",
        description=(
            "South-East repairs patch daily operations for supervisors and "
            "operatives. Carries patch-level work-order data, open repair "
            "counts, and shared dashboard pools the whole patch reads."
        ),
    )
    executive_summary = SpaceSummary(
        space_id=EXECUTIVE_SPACE_ID,
        name="Executive Dashboards",
        description=(
            "Executive team board-deck dashboards, cross-patch KPI reviews, "
            "and monthly leadership reporting."
        ),
    )
    _set_dashboard_spaces(
        space_ids=[PATCH_SPACE_ID, EXECUTIVE_SPACE_ID],
        space_summaries=[patch_summary, executive_summary],
    )
    try:
        yield
    finally:
        _set_dashboard_spaces(
            space_ids=original_space_ids,
            space_summaries=original_space_summaries,
        )


async def _await_eval_result(handle):
    return await asyncio.wait_for(handle.result(), timeout=EVAL_TIMEOUT_SECONDS)


def _record_tokens(records) -> set[str]:
    return {record.token for record in records}


def _live_tiles(tiles):
    return [tile for tile in tiles if tile.has_data_bindings]


def _binding_contexts(tile) -> set[str]:
    assert tile.data_bindings_json is not None
    contexts: set[str] = set()
    for binding in json.loads(tile.data_bindings_json):
        if "context" in binding:
            contexts.add(binding["context"])
        contexts.update(binding.get("tables", []))
    assert contexts
    return contexts


def _show_only_space(space_id: int) -> None:
    SESSION_DETAILS.space_ids = [space_id]
    ContextRegistry.clear()


def _show_no_spaces() -> None:
    SESSION_DETAILS.space_ids = []
    ContextRegistry.clear()


def _routing_guidelines() -> str:
    return build_accessible_spaces_block(SESSION_DETAILS.space_summaries)


@pytest.mark.asyncio
@pytest.mark.timeout(EVAL_TIMEOUT_SECONDS)
async def test_team_dashboard_routes_to_patch_space():
    """A team-facing dashboard request lands in the named shared space."""
    with _dashboard_spaces():
        async with make_code_act_actor(
            impl="simulated",
            exposed_managers={"dashboards"},
        ) as (actor, primitives, calls):
            handle = await actor.act(
                (
                    "Create a Patch-1 operations dashboard that the whole repairs "
                    "patch can open for daily work-order monitoring."
                ),
                guidelines=_routing_guidelines(),
                clarification_enabled=False,
                can_store=False,
            )
            result = await _await_eval_result(handle)
            assert result is not None

            _show_only_space(PATCH_SPACE_ID)
            patch_dashboards = await primitives.dashboards.list_dashboards()
            assert (
                patch_dashboards
            ), f"Expected a Patch-1 dashboard to be created. Calls: {calls}"

            _show_only_space(EXECUTIVE_SPACE_ID)
            executive_dashboards = await primitives.dashboards.list_dashboards()
            assert executive_dashboards == []

            _show_no_spaces()
            hidden_dashboards = await primitives.dashboards.list_dashboards()
            assert hidden_dashboards == []


@pytest.mark.asyncio
@pytest.mark.timeout(EVAL_TIMEOUT_SECONDS)
async def test_private_dashboard_stays_personal():
    """A private dashboard request honors the personal privacy floor."""
    with _dashboard_spaces():
        async with make_code_act_actor(
            impl="simulated",
            exposed_managers={"dashboards"},
        ) as (actor, primitives, calls):
            handle = await actor.act(
                (
                    "Create my private productivity dashboard for tracking my own "
                    "focus time and follow-ups. This is just for me."
                ),
                guidelines=_routing_guidelines(),
                clarification_enabled=False,
                can_store=False,
            )
            result = await _await_eval_result(handle)
            assert result is not None

            _show_no_spaces()
            personal_dashboards = await primitives.dashboards.list_dashboards()
            personal_tokens = _record_tokens(personal_dashboards)
            assert (
                personal_tokens
            ), f"Expected a personal dashboard to remain visible. Calls: {calls}"

            _show_only_space(PATCH_SPACE_ID)
            patch_visible_tokens = _record_tokens(
                await primitives.dashboards.list_dashboards(),
            )
            assert patch_visible_tokens == personal_tokens

            _show_only_space(EXECUTIVE_SPACE_ID)
            executive_visible_tokens = _record_tokens(
                await primitives.dashboards.list_dashboards(),
            )
            assert executive_visible_tokens == personal_tokens


@pytest.mark.asyncio
@pytest.mark.timeout(EVAL_TIMEOUT_SECONDS)
async def test_team_live_tile_routes_to_patch_space_and_inherits_dashboard_scope():
    """A team dashboard tile lands in Patch-1 and inherits its data root."""
    with _dashboard_spaces():
        async with make_code_act_actor(
            impl="simulated",
            exposed_managers={"dashboards"},
        ) as (actor, primitives, calls):
            handle = await actor.act(
                (
                    "Add a live tile to the Patch-1 operations dashboard showing "
                    "today's open work-order count from WorkOrders data, so the "
                    "whole repairs patch can use it during standup."
                ),
                guidelines=_routing_guidelines(),
                clarification_enabled=False,
                can_store=False,
            )
            result = await _await_eval_result(handle)
            assert result is not None

            _show_only_space(PATCH_SPACE_ID)
            patch_live_tiles = _live_tiles(await primitives.dashboards.list_tiles())
            assert (
                len(patch_live_tiles) == 1
            ), f"Expected one Patch-1 live tile. Calls: {calls}"
            tile = patch_live_tiles[0]
            assert tile.data_scope == "dashboard"
            assert any(
                context.startswith(f"Spaces/{PATCH_SPACE_ID}/")
                for context in _binding_contexts(tile)
            )

            _show_only_space(EXECUTIVE_SPACE_ID)
            assert _live_tiles(await primitives.dashboards.list_tiles()) == []

            _show_no_spaces()
            assert _live_tiles(await primitives.dashboards.list_tiles()) == []


@pytest.mark.asyncio
@pytest.mark.timeout(EVAL_TIMEOUT_SECONDS)
async def test_private_dashboard_tile_can_bind_to_team_data():
    """A private watch tile stays personal while binding to Patch-1 data."""
    with _dashboard_spaces():
        async with make_code_act_actor(
            impl="simulated",
            exposed_managers={"dashboards"},
        ) as (actor, primitives, calls):
            handle = await actor.act(
                (
                    "On my private dashboard, add a live tile showing the Patch-1 "
                    "open work-order count from the WorkOrders data. I want to "
                    "watch the team metric personally without publishing this tile "
                    "to the whole patch."
                ),
                guidelines=_routing_guidelines(),
                clarification_enabled=False,
                can_store=False,
            )
            result = await _await_eval_result(handle)
            assert result is not None

            live_tiles = _live_tiles(await primitives.dashboards.list_tiles())
            assert len(live_tiles) == 1, f"Expected one live-data tile. Calls: {calls}"
            tile = live_tiles[0]
            assert tile.data_scope == f"space:{PATCH_SPACE_ID}"
            assert any(
                context.startswith(f"Spaces/{PATCH_SPACE_ID}/")
                for context in _binding_contexts(tile)
            )

            _show_no_spaces()
            personal_tiles = await primitives.dashboards.list_tiles()
            personal_live_tokens = {
                item.token for item in personal_tiles if item.has_data_bindings
            }
            assert personal_live_tokens == {tile.token}


@pytest.mark.asyncio
@pytest.mark.timeout(EVAL_TIMEOUT_SECONDS)
async def test_executive_overview_tile_routes_to_executive_space():
    """An executive board-deck tile lands in the executive shared space."""
    with _dashboard_spaces():
        async with make_code_act_actor(
            impl="simulated",
            exposed_managers={"dashboards"},
        ) as (actor, primitives, calls):
            handle = await actor.act(
                (
                    "Add a live board-deck KPI tile to the executive dashboard "
                    "showing this month's cross-patch closure rate from WorkOrders "
                    "data for the monthly leadership review."
                ),
                guidelines=_routing_guidelines(),
                clarification_enabled=False,
                can_store=False,
            )
            result = await _await_eval_result(handle)
            assert result is not None

            _show_only_space(EXECUTIVE_SPACE_ID)
            executive_live_tiles = _live_tiles(await primitives.dashboards.list_tiles())
            assert (
                len(executive_live_tiles) == 1
            ), f"Expected one executive live tile. Calls: {calls}"
            tile = executive_live_tiles[0]
            assert tile.data_scope == "dashboard"
            assert any(
                context.startswith(f"Spaces/{EXECUTIVE_SPACE_ID}/")
                for context in _binding_contexts(tile)
            )

            _show_only_space(PATCH_SPACE_ID)
            assert _live_tiles(await primitives.dashboards.list_tiles()) == []

            _show_no_spaces()
            assert _live_tiles(await primitives.dashboards.list_tiles()) == []


@pytest.mark.asyncio
@pytest.mark.timeout(EVAL_TIMEOUT_SECONDS)
async def test_ambiguous_dashboard_request_does_not_publish_to_a_space():
    """An underspecified dashboard request clarifies or stays personal."""
    with _dashboard_spaces():
        async with make_code_act_actor(
            impl="simulated",
            exposed_managers={"dashboards"},
        ) as (actor, primitives, calls):
            handle = await actor.act(
                "Make a dashboard for tracking late deliveries.",
                guidelines=_routing_guidelines(),
                clarification_enabled=False,
                can_store=False,
            )
            result = await _await_eval_result(handle)
            assert result is not None

            _show_no_spaces()
            personal_dashboards = await primitives.dashboards.list_dashboards()
            personal_tokens = _record_tokens(personal_dashboards)

            _show_only_space(PATCH_SPACE_ID)
            patch_visible_tokens = _record_tokens(
                await primitives.dashboards.list_dashboards(),
            )
            assert patch_visible_tokens == personal_tokens

            _show_only_space(EXECUTIVE_SPACE_ID)
            executive_visible_tokens = _record_tokens(
                await primitives.dashboards.list_dashboards(),
            )
            assert executive_visible_tokens == personal_tokens
            assert not patch_visible_tokens - personal_tokens
            assert not executive_visible_tokens - personal_tokens
