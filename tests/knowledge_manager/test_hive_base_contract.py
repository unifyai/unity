"""KnowledgeManager Hive-base contract.

Knowledge tables are user-defined: their schema is not fixed by a pydantic
model. The Hive contract lives at two layers:

1. ``"Knowledge"`` is in ``_HIVE_SCOPED_TABLES`` so a body inside a Hive
   reads and writes it under ``Hives/{hive_id}/Knowledge/...``; a solo body
   keeps its per-body ``{user}/{assistant}/Knowledge/...`` root.
2. Row inserts stamp ``authoring_assistant_id`` on every row at write time
   and the update path strips the column so the stamp is immutable after
   creation.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from unity.common.context_registry import ContextRegistry
from unity.knowledge_manager.ops import AUTHORING_COLUMN, add_rows, update_rows
from unity.session_details import SESSION_DETAILS

pytestmark = pytest.mark.usefixtures("pinned_hive_body")


def test_hive_member_resolves_knowledge_to_hive_root():
    SESSION_DETAILS.hive_id = 42
    assert ContextRegistry.base_for("Knowledge") == "Hives/42"


def test_solo_body_resolves_knowledge_to_per_body_root(pinned_hive_body):
    assert SESSION_DETAILS.hive_id is None
    assert ContextRegistry.base_for("Knowledge") == pinned_hive_body


def _fake_knowledge_manager(dm: MagicMock) -> SimpleNamespace:
    return SimpleNamespace(
        _data_manager=dm,
        _ctx="u7/7/Knowledge",
        _include_contacts=False,
        _contacts_ctx=None,
        include_in_multi_assistant_table=False,
    )


def test_add_rows_stamps_authoring_assistant_id_on_every_row():
    dm = MagicMock()
    dm.insert_rows.return_value = [1, 2]
    km = _fake_knowledge_manager(dm)

    add_rows(km, table="Products", rows=[{"name": "a"}, {"name": "b"}])

    call_rows = dm.insert_rows.call_args.kwargs["rows"]
    assert all(row[AUTHORING_COLUMN] == 7 for row in call_rows)


def test_add_rows_preserves_caller_supplied_authoring_assistant_id():
    dm = MagicMock()
    dm.insert_rows.return_value = [1]
    km = _fake_knowledge_manager(dm)

    add_rows(
        km,
        table="Products",
        rows=[{"name": "a", AUTHORING_COLUMN: 99}],
    )

    assert dm.insert_rows.call_args.kwargs["rows"][0][AUTHORING_COLUMN] == 99


def test_update_rows_strips_authoring_assistant_id(monkeypatch):
    dm = MagicMock()
    dm.get_table.return_value = {"unique_keys": ["row_id"]}
    dm.filter.return_value = [{"row_id": 1, "id": 500}]
    km = _fake_knowledge_manager(dm)

    captured: dict = {}

    def _fake_update_logs(**kwargs):
        captured.update(kwargs)
        return {}

    import unity.knowledge_manager.ops as ops_module

    monkeypatch.setattr(ops_module.unify, "update_logs", _fake_update_logs)

    update_rows(
        km,
        table="Products",
        updates={1: {"name": "new", AUTHORING_COLUMN: 999}},
    )

    assert captured["entries"] == [{"name": "new"}]
