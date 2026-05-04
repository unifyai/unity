from __future__ import annotations

import uuid

import pytest
import unify

from tests.helpers import _handle_project
from unity.common.context_registry import ContextRegistry
from unity.data_manager.data_manager import DataManager
from unity.manager_registry import ManagerRegistry
from unity.session_details import SESSION_DETAILS


def _fresh_manager() -> DataManager:
    ManagerRegistry.clear()
    ContextRegistry.clear()
    return DataManager()


def _configure_spaces() -> tuple[int, int]:
    base_space_id = 20_000_000 + uuid.uuid4().int % 1_000_000_000
    space_ids = (base_space_id, base_space_id + 1)
    SESSION_DETAILS.space_ids = list(space_ids)
    SESSION_DETAILS.space_summaries = [
        {
            "space_id": space_ids[0],
            "name": "Revenue Ops",
            "description": "Shared workspace for revenue operations data.",
        },
        {
            "space_id": space_ids[1],
            "name": "Support Ops",
            "description": "Shared workspace for support operations data.",
        },
    ]
    return space_ids


def _reset_spaces(space_ids: tuple[int, int], suffix: str) -> None:
    for space_id in space_ids:
        try:
            unify.delete_context(f"Spaces/{space_id}/Data/{suffix}")
        except Exception:
            pass
    SESSION_DETAILS.space_ids = []
    SESSION_DETAILS.space_summaries = []
    ContextRegistry.clear()


@_handle_project
def test_data_writes_route_to_destination_and_reads_merge_roots():
    space_ids = _configure_spaces()
    table_suffix = f"destination_routing/{uuid.uuid4().hex}"
    manager = _fresh_manager()

    try:
        personal_path = manager.create_table(
            table_suffix,
            fields={"label": "str", "amount": "int"},
        )
        shared_path = manager.create_table(
            table_suffix,
            fields={"label": "str", "amount": "int"},
            destination=f"space:{space_ids[0]}",
        )

        manager.insert_rows(personal_path, [{"label": "personal", "amount": 1}])
        manager.insert_rows(
            table_suffix,
            [{"label": "shared", "amount": 2}],
            destination=f"space:{space_ids[0]}",
        )

        assert shared_path == f"Spaces/{space_ids[0]}/Data/{table_suffix}"
        shared_meta = manager.get_table(shared_path)
        assert "row_id" in shared_meta.get("unique_keys", [])
        assert {row["label"] for row in manager.filter(shared_path)} == {"shared"}
        assert {row["label"] for row in manager.filter(personal_path)} == {"personal"}

        merged = manager.filter(table_suffix, columns=["label", "amount"])
        assert {row["label"] for row in merged} == {"personal", "shared"}
        prefixed_merged = manager.filter(
            f"Data/{table_suffix}",
            columns=["label", "amount"],
        )
        assert {row["label"] for row in prefixed_merged} == {"personal", "shared"}
        assert manager.reduce(table_suffix, metric="sum", columns="amount") == 3

        prefixed_path = manager.create_table(
            f"Data/{table_suffix}/prefixed_default",
            fields={"label": "str"},
        )
        assert prefixed_path.endswith(f"/Data/{table_suffix}/prefixed_default")
        assert not prefixed_path.startswith("Data/")
    finally:
        _reset_spaces(space_ids, table_suffix)


@_handle_project
def test_data_prefixed_paths_default_to_personal_without_spaces():
    SESSION_DETAILS.space_ids = []
    SESSION_DETAILS.space_summaries = []
    ContextRegistry.clear()
    table_suffix = f"prefixed_personal/{uuid.uuid4().hex}"
    manager = _fresh_manager()

    personal_path = manager.create_table(
        f"Data/{table_suffix}",
        fields={"label": "str"},
    )
    manager.insert_rows(
        f"Data/{table_suffix}",
        [{"label": "personal"}],
    )

    assert personal_path.endswith(f"/Data/{table_suffix}")
    assert manager.filter(f"Data/{table_suffix}", columns=["label"]) == [
        {"label": "personal"},
    ]


@_handle_project
def test_data_shared_only_metadata_and_join_reads_use_visible_roots():
    space_ids = _configure_spaces()
    table_suffix = f"destination_routing_metadata/{uuid.uuid4().hex}"
    left_suffix = f"{table_suffix}/left"
    right_suffix = f"{table_suffix}/right"
    manager = _fresh_manager()

    try:
        shared_left = manager.create_table(
            left_suffix,
            fields={"label": "str", "join_key": "int"},
            destination=f"space:{space_ids[0]}",
        )
        shared_right = manager.create_table(
            right_suffix,
            fields={"join_key": "int", "amount": "int"},
            destination=f"space:{space_ids[0]}",
        )
        manager.insert_rows(
            left_suffix,
            [{"label": "shared", "join_key": 7}],
            destination=f"space:{space_ids[0]}",
        )
        manager.insert_rows(
            right_suffix,
            [{"join_key": 7, "amount": 42}],
            destination=f"space:{space_ids[0]}",
        )

        assert manager.describe_table(left_suffix).context == shared_left
        assert "label" in manager.get_columns(left_suffix)
        assert "row_id" in manager.get_table(left_suffix).get("unique_keys", [])
        assert shared_left in manager.list_tables(
            prefix=left_suffix,
            include_column_info=False,
        )

        joined = manager.filter_join(
            tables=[left_suffix, right_suffix],
            join_expr=f"{left_suffix}.join_key == {right_suffix}.join_key",
            select={
                f"{left_suffix}.label": "label",
                f"{right_suffix}.amount": "amount",
            },
        )

        assert joined == [{"label": "shared", "amount": 42}]
        assert (
            manager.reduce_join(
                tables=[left_suffix, right_suffix],
                join_expr=f"{left_suffix}.join_key == {right_suffix}.join_key",
                select={
                    f"{left_suffix}.label": "label",
                    f"{right_suffix}.amount": "amount",
                },
                metric="sum",
                columns="amount",
            )
            == 42
        )
        assert manager.reduce_join(
            tables=[left_suffix, right_suffix],
            join_expr=f"{left_suffix}.join_key == {right_suffix}.join_key",
            select={
                f"{left_suffix}.label": "label",
                f"{right_suffix}.amount": "amount",
            },
            metric="sum",
            columns="amount",
            group_by="label",
        ) == {"shared": 42}
        assert shared_right == f"Spaces/{space_ids[0]}/Data/{right_suffix}"
    finally:
        _reset_spaces(space_ids, table_suffix)


@_handle_project
def test_data_invalid_destination_returns_tool_error():
    _configure_spaces()
    manager = _fresh_manager()

    try:
        outcome = manager.create_table(
            f"bad_destination/{uuid.uuid4().hex}",
            destination="space:99999999",
        )
    finally:
        SESSION_DETAILS.space_ids = []
        SESSION_DETAILS.space_summaries = []
        ContextRegistry.clear()

    assert outcome["error_kind"] == "invalid_destination"
    assert outcome["details"]["destination"] == "space:99999999"


@pytest.mark.parametrize(
    "call",
    [
        lambda manager, context: manager.create_table(
            context,
            destination="space:99999999",
        ),
        lambda manager, context: manager.delete_table(
            context,
            dangerous_ok=True,
            destination="space:99999999",
        ),
        lambda manager, context: manager.rename_table(
            context,
            f"{context}_renamed",
            destination="space:99999999",
        ),
        lambda manager, context: manager.create_column(
            context,
            column_name="label",
            column_type="str",
            destination="space:99999999",
        ),
        lambda manager, context: manager.delete_column(
            context,
            column_name="label",
            destination="space:99999999",
        ),
        lambda manager, context: manager.rename_column(
            context,
            old_name="label",
            new_name="name",
            destination="space:99999999",
        ),
        lambda manager, context: manager.create_derived_column(
            context,
            column_name="total",
            equation="amount * 2",
            destination="space:99999999",
        ),
        lambda manager, context: manager.join_tables(
            left_table=context,
            right_table=f"{context}_right",
            join_expr=f"{context}.id == {context}_right.id",
            dest_table=f"{context}_joined",
            select={f"{context}.id": "id"},
            destination="space:99999999",
        ),
        lambda manager, context: manager.insert_rows(
            context,
            [{"label": "x"}],
            destination="space:99999999",
        ),
        lambda manager, context: manager.update_rows(
            context,
            updates={"label": "y"},
            filter="label == 'x'",
            destination="space:99999999",
        ),
        lambda manager, context: manager.delete_rows(
            context,
            filter="label == 'x'",
            destination="space:99999999",
        ),
        lambda manager, context: manager.ingest(
            context,
            rows=[{"label": "x"}],
            destination="space:99999999",
        ),
        lambda manager, context: manager.ensure_vector_column(
            context,
            source_column="label",
            destination="space:99999999",
        ),
        lambda manager, context: manager.vectorize_rows(
            context,
            source_column="label",
            destination="space:99999999",
        ),
    ],
)
@_handle_project
def test_data_write_tools_return_tool_error_for_invalid_destination(call):
    _configure_spaces()
    manager = _fresh_manager()

    try:
        outcome = call(manager, f"bad_destination/{uuid.uuid4().hex}")
    finally:
        SESSION_DETAILS.space_ids = []
        SESSION_DETAILS.space_summaries = []
        ContextRegistry.clear()

    assert outcome["error_kind"] == "invalid_destination"
    assert outcome["details"]["destination"] == "space:99999999"
