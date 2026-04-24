"""DataManager Hive-base contract.

``"Data"`` is in ``_HIVE_SCOPED_TABLES``, so a body inside a Hive reads and
writes every ``Data/...`` row under ``Hives/{hive_id}/Data/...``; a solo
body keeps its per-body ``{user}/{assistant}/Data/...`` root.

``insert_rows_impl`` stamps ``authoring_assistant_id`` on every inserted
row so shared rows can be attributed back to the authoring body;
``update_rows_impl`` strips the column from caller-supplied updates so the
stamp is immutable once written.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from unity.common.context_registry import ContextRegistry
from unity.data_manager.ops.mutation_ops import (
    AUTHORING_COLUMN,
    insert_rows_impl,
    update_rows_impl,
)
from unity.session_details import SESSION_DETAILS

pytestmark = pytest.mark.usefixtures("pinned_hive_body")


def test_hive_member_resolves_data_root_to_hive_base():
    SESSION_DETAILS.hive_id = 42
    assert ContextRegistry.base_for("Data") == "Hives/42"


def test_solo_body_resolves_data_root_to_per_body_base(pinned_hive_body):
    assert SESSION_DETAILS.hive_id is None
    assert ContextRegistry.base_for("Data") == pinned_hive_body


def test_hive_member_resolves_data_subtable_to_hive_base():
    """``Data/<dataset>`` inherits the Hive root through the prefix match."""
    SESSION_DETAILS.hive_id = 42
    assert ContextRegistry.base_for("Data/sales_pipeline") == "Hives/42"


def test_insert_rows_stamps_authoring_assistant_id():
    with patch(
        "unity.data_manager.ops.mutation_ops.unify_create_logs",
    ) as create_logs:
        create_logs.return_value = []
        insert_rows_impl(
            "Hives/42/Data/sales_pipeline",
            rows=[{"stage": "qualified"}, {"stage": "won"}],
        )

    stamped = create_logs.call_args.kwargs["entries"]
    assert all(row[AUTHORING_COLUMN] == 7 for row in stamped)


def test_insert_rows_preserves_caller_stamp():
    with patch(
        "unity.data_manager.ops.mutation_ops.unify_create_logs",
    ) as create_logs:
        create_logs.return_value = []
        insert_rows_impl(
            "Hives/42/Data/sales_pipeline",
            rows=[{"stage": "won", AUTHORING_COLUMN: 99}],
        )

    assert create_logs.call_args.kwargs["entries"][0][AUTHORING_COLUMN] == 99


def test_update_rows_strips_authoring_assistant_id():
    fake_log = type(
        "_Log",
        (),
        {"id": 500, "entries": {"stage": "qualified", AUTHORING_COLUMN: 7}},
    )()
    captured: dict = {}

    with patch("unity.data_manager.ops.mutation_ops.unify") as unify_mod:
        unify_mod.get_logs.return_value = [fake_log]

        def _fake_log(**kwargs):
            captured.update(kwargs)

        with patch(
            "unity.data_manager.ops.mutation_ops.unify_log",
            side_effect=_fake_log,
        ):
            update_rows_impl(
                "Hives/42/Data/sales_pipeline",
                updates={"stage": "won", AUTHORING_COLUMN: 999},
                filter="stage == 'qualified'",
            )

    assert captured[AUTHORING_COLUMN] == 7
    assert captured["stage"] == "won"
