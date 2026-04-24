"""DashboardManager Hive-base contract.

Both ``"Dashboards/Tiles"`` and ``"Dashboards/Layouts"`` are in
``_HIVE_SCOPED_TABLES`` so a body inside a Hive resolves them under
``Hives/{hive_id}/...``; a solo body keeps its per-body
``{user}/{assistant}/...`` root. Tile data bindings pointing at
``Data/<dataset>`` resolve against the active Hive base automatically
because ``Data`` is also Hive-shared. Tile and Dashboard record rows
carry ``authoring_assistant_id`` so Hive-shared rows can be attributed
back to the authoring body.
"""

from __future__ import annotations

import pytest

from unity.common.context_registry import ContextRegistry
from unity.dashboard_manager.types.dashboard import DashboardRecordRow
from unity.dashboard_manager.types.tile import TileRecordRow
from unity.session_details import SESSION_DETAILS

pytestmark = pytest.mark.usefixtures("pinned_hive_body")

_HIVE_SHARED_SUBTABLES = ("Dashboards/Tiles", "Dashboards/Layouts")


@pytest.mark.parametrize("table", _HIVE_SHARED_SUBTABLES)
def test_hive_member_resolves_dashboard_subtable_to_hive_root(table):
    SESSION_DETAILS.hive_id = 42
    assert ContextRegistry.base_for(table) == "Hives/42"


@pytest.mark.parametrize("table", _HIVE_SHARED_SUBTABLES)
def test_solo_body_resolves_dashboard_subtable_to_per_body_root(table, pinned_hive_body):
    assert SESSION_DETAILS.hive_id is None
    assert ContextRegistry.base_for(table) == pinned_hive_body


def test_tile_data_binding_resolves_against_hive_base_for_hive_member():
    """``Data/<dataset>`` tile bindings inherit the Hive base transparently."""
    SESSION_DETAILS.hive_id = 42
    assert ContextRegistry.base_for("Data/sales_pipeline") == "Hives/42"


def test_tile_data_binding_resolves_against_per_body_base_for_solo(pinned_hive_body):
    assert SESSION_DETAILS.hive_id is None
    assert ContextRegistry.base_for("Data/sales_pipeline") == pinned_hive_body


@pytest.mark.parametrize("model", [TileRecordRow, DashboardRecordRow])
def test_shared_record_models_declare_authoring_assistant_id(model):
    field = model.model_fields.get("authoring_assistant_id")
    assert field is not None
    assert field.default is None
