"""Two bodies syncing their primitives converge on a Hive-wide union.

Each body writes its own ``PrimitiveScope`` into the shared
``Hives/{hive_id}/Functions/Primitives`` catalog. Bodies with narrower
scopes do not remove extras contributed by other bodies, so the catalog
holds the union of every body's registered managers — the contract
documented on :meth:`FunctionManager.sync_primitives`.

The test hits the live backend because the convergence claim is
ultimately a statement about what the shared table holds after two
independent writers have committed. A mock would erase the contract.
"""

from __future__ import annotations

import pytest
import unify

from tests.helpers import bind_body, cleanup_hive_context, unique_hive_id
from unity.function_manager.function_manager import FunctionManager
from unity.function_manager.primitives.scope import PrimitiveScope
from unity.session_details import SESSION_DETAILS


def _body_with_scope(hive_id: int, agent_id: int, scope: PrimitiveScope) -> FunctionManager:
    bind_body(hive_id=hive_id, agent_id=agent_id)
    return FunctionManager(primitive_scope=scope, daemon=False)


@pytest.mark.requires_real_unify
def test_two_bodies_sync_primitives_into_hive_wide_union():
    hive_id = unique_hive_id()
    primitives_ctx = f"Hives/{hive_id}/Functions/Primitives"

    scope_a = PrimitiveScope.single("contacts")
    scope_b = PrimitiveScope.single("knowledge")

    try:
        body_a = _body_with_scope(hive_id, agent_id=1001, scope=scope_a)
        body_a.sync_primitives()

        body_b = _body_with_scope(hive_id, agent_id=2002, scope=scope_b)
        body_b.sync_primitives()

        rows = unify.get_logs(
            context=primitives_ctx,
            filter="is_primitive == True",
            limit=1000,
        )

        authors = {row.entries.get("authoring_assistant_id") for row in rows}
        assert {1001, 2002}.issubset(authors), (
            f"expected both bodies to have stamped rows, got authors={authors}"
        )

        names = {row.entries.get("name") for row in rows if row.entries.get("name")}
        contact_hits = [n for n in names if n.startswith("primitives.contacts.")]
        knowledge_hits = [n for n in names if n.startswith("primitives.knowledge.")]
        assert contact_hits, "Body A's contacts primitives missing from shared catalog"
        assert knowledge_hits, "Body B's knowledge primitives missing from shared catalog"

        body_a_authored = {
            row.entries.get("name")
            for row in rows
            if row.entries.get("authoring_assistant_id") == 1001
        }
        body_b_authored = {
            row.entries.get("name")
            for row in rows
            if row.entries.get("authoring_assistant_id") == 2002
        }
        assert all(n.startswith("primitives.contacts.") for n in body_a_authored)
        assert all(n.startswith("primitives.knowledge.") for n in body_b_authored)
    finally:
        SESSION_DETAILS.reset()
        cleanup_hive_context(hive_id)
