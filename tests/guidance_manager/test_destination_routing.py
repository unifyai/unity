from __future__ import annotations

import unify

from tests.destination_routing_helpers import (
    manager_routing_context as manager_routing_context,  # noqa: F401
)
from unity.guidance_manager.guidance_manager import GuidanceManager


def test_guidance_writes_route_to_destination_and_reads_merge_roots(
    manager_routing_context,
):
    """Guidance mutations target one root while filters see personal and shared rows."""

    _, space_id = manager_routing_context
    manager = GuidanceManager()

    manager.add_guidance(title="Private rule", content="For my private drafts.")
    manager.add_guidance(
        title="Team rule",
        content="For shared operations.",
        destination=f"space:{space_id}",
    )

    assert [row.entries["title"] for row in unify.get_logs(context=manager._ctx)] == [
        "Private rule",
    ]
    assert [
        row.entries["title"]
        for row in unify.get_logs(context=f"Spaces/{space_id}/Guidance")
    ] == ["Team rule"]
    assert {row.title for row in manager.filter()} == {"Private rule", "Team rule"}

    outcome = manager.delete_guidance(guidance_id=1, destination="space:404404")
    assert outcome["error_kind"] == "invalid_destination"
