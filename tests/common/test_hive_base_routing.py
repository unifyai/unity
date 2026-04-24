"""Cross-manager routing contract for Hive-shared tables.

Exercises the three pieces of machinery that have to agree for the
manager cluster's Hive-shared storage to be safe:

1. :meth:`ContextRegistry.base_for` routes a table to the Hive root or
   the per-body root based on the active ``SESSION_DETAILS.hive_id``.
2. Dynamic sub-tables (``Data/<dataset>``, ``Knowledge/<table>``) inherit
   the Hive root through the prefix match in
   :meth:`ContextRegistry._is_hive_scoped`, so callers using the
   ``<root>/<child>`` short-form do not have to enumerate children in
   the scoped-table inventory.
3. Aggregation shells (``{user}/All/...``, ``All/...``) are *not*
   provisioned for Hive paths, because ``Hives/{hive_id}/...`` is already
   the cross-body shared surface and aggregation siblings would just
   bloat the tree.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from unity.common.context_registry import (
    HIVE_CONTEXT_PREFIX,
    ContextRegistry,
    TableContext,
    _HIVE_SCOPED_TABLES,
)
from unity.session_details import SESSION_DETAILS


pytestmark = pytest.mark.usefixtures("pinned_hive_body")


@pytest.mark.parametrize("table_name", sorted(_HIVE_SCOPED_TABLES))
def test_every_hive_scoped_table_routes_to_hive_root_in_hive_mode(table_name):
    SESSION_DETAILS.hive_id = 42
    assert ContextRegistry.base_for(table_name) == f"{HIVE_CONTEXT_PREFIX}42"


@pytest.mark.parametrize("table_name", sorted(_HIVE_SCOPED_TABLES))
def test_every_hive_scoped_table_routes_to_per_body_root_in_solo_mode(
    table_name, pinned_hive_body,
):
    assert SESSION_DETAILS.hive_id is None
    assert ContextRegistry.base_for(table_name) == pinned_hive_body


@pytest.mark.parametrize(
    "child_table",
    [
        "Data/sales_pipeline",
        "Data/analytics/daily",
        "Knowledge/Products",
        "Knowledge/company/policies",
    ],
)
def test_dynamic_child_of_hive_scoped_root_inherits_hive_routing(child_table):
    SESSION_DETAILS.hive_id = 42
    assert ContextRegistry.base_for(child_table) == f"{HIVE_CONTEXT_PREFIX}42"


def test_per_body_overlay_children_stay_per_body(pinned_hive_body):
    """Tables whose root is not Hive-shared never leak onto the Hive root."""
    SESSION_DETAILS.hive_id = 42
    assert ContextRegistry.base_for("Events") == pinned_hive_body
    assert ContextRegistry.base_for("Events/call_summary") == pinned_hive_body
    assert ContextRegistry.base_for("ContactMembership") == pinned_hive_body


def test_hive_paths_skip_aggregation_shell_provisioning():
    """``_ensure_all_contexts`` is a no-op for any path under ``Hives/``.

    Minting ``Hives/{hive}/All/...`` shells would bloat the tree with
    unused siblings and force Hive cascade deletes to walk them, so the
    context registry short-circuits there.
    """
    table = TableContext(name="Knowledge", description="stub")

    with patch(
        "unity.common.context_registry._create_context_with_retry",
    ) as create_ctx:
        ContextRegistry._ensure_all_contexts("Hives/42/Knowledge", table)

    create_ctx.assert_not_called()


def test_per_body_paths_still_get_aggregation_shells():
    """Per-body roots keep their ``{user}/All/...`` and ``All/...`` shells."""
    table = TableContext(name="Knowledge", description="stub")

    with patch(
        "unity.common.context_registry._create_context_with_retry",
    ) as create_ctx:
        ContextRegistry._ensure_all_contexts("u7/7/Knowledge", table)

    created_paths = [call.args[0] for call in create_ctx.call_args_list]
    assert "u7/All/Knowledge" in created_paths
    assert "All/Knowledge" in created_paths
