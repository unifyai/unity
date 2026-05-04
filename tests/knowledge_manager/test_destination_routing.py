from __future__ import annotations

import unify

from tests.destination_routing_helpers import (
    manager_routing_context as manager_routing_context,  # noqa: F401
)
from unity.knowledge_manager.knowledge_manager import KnowledgeManager


def test_knowledge_writes_route_to_destination_and_reads_merge_roots(
    manager_routing_context,
):
    """Knowledge tables write to one root while filter reads all reachable roots."""

    _, space_id = manager_routing_context
    manager = KnowledgeManager()

    manager._create_table(
        name="Runbooks",
        columns={"title": "str", "audience": "str"},
    )
    manager._create_table(
        name="Runbooks",
        columns={"title": "str", "audience": "str"},
        destination=f"space:{space_id}",
    )
    manager._add_rows(
        table="Runbooks",
        rows=[{"title": "Private escalation", "audience": "personal"}],
    )
    manager._add_rows(
        table="Runbooks",
        rows=[{"title": "Team escalation", "audience": "shared"}],
        destination=f"space:{space_id}",
    )
    manager._create_table(
        name="PersonalOnly",
        columns={"title": "str"},
    )
    manager._add_rows(
        table="PersonalOnly",
        rows=[{"title": "Only personal root"}],
    )

    personal_rows = unify.get_logs(context=f"{manager._ctx}/Runbooks")
    shared_rows = unify.get_logs(context=f"Spaces/{space_id}/Knowledge/Runbooks")

    assert [row.entries["audience"] for row in personal_rows] == ["personal"]
    assert [row.entries["audience"] for row in shared_rows] == ["shared"]
    assert {
        row["title"] for row in manager._filter(tables=["Runbooks"])["Runbooks"]
    } == {"Private escalation", "Team escalation"}
    assert manager._filter(tables=["PersonalOnly"])["PersonalOnly"] == [
        {"title": "Only personal root", "row_id": 0},
    ]

    outcome = manager._add_rows(
        table="Runbooks",
        rows=[{"title": "Invisible", "audience": "shared"}],
        destination="space:404404",
    )
    assert outcome["error_kind"] == "invalid_destination"
