"""Tests for DashboardManager Pydantic types and tile_ops helpers."""

import json
from unittest.mock import patch

import pytest
from pydantic import TypeAdapter

from unity.dashboard_manager.ops.tile_ops import (
    _match_context,
    ensure_binding_aliases,
    resolve_binding_contexts,
    serialize_bindings,
    validate_on_data,
)
from unity.dashboard_manager.types.tile import (
    DataBinding,
    FilterBinding,
    JoinBinding,
    JoinReduceBinding,
    ReduceBinding,
    TileRecord,
    TileRecordRow,
    TileResult,
)
from unity.dashboard_manager.types.dashboard import (
    DashboardRecordRow,
    DashboardResult,
    TilePosition,
)


class TestTileTypes:
    def test_tile_record_row_minimal(self):
        row = TileRecordRow(
            token="abc123abc123",
            title="Test Tile",
            html_content="<h1>Hello</h1>",
        )
        assert row.token == "abc123abc123"
        assert row.has_data_bindings is False
        assert row.data_binding_contexts is None

    def test_tile_record_with_bindings(self):
        row = TileRecordRow(
            token="abc123abc123",
            title="Live Tile",
            html_content="<div></div>",
            has_data_bindings=True,
            data_binding_contexts="Data/Sales,Data/Orders",
        )
        assert row.has_data_bindings is True
        assert "Data/Sales" in row.data_binding_contexts

    def test_tile_record_includes_id(self):
        record = TileRecord(
            tile_id=42,
            token="abc123abc123",
            title="Test",
            html_content="<p>content</p>",
        )
        assert record.tile_id == 42

    def test_tile_result_succeeded(self):
        result = TileResult(
            url="https://example.com/tile/view/abc123",
            token="abc123",
            title="Test",
        )
        assert result.succeeded is True

    def test_tile_result_failed(self):
        result = TileResult(error="Something went wrong")
        assert result.succeeded is False


class TestFilterBinding:
    def test_minimal(self):
        b = FilterBinding(context="Data/X")
        assert b.operation == "filter"
        assert b.context == "Data/X"
        assert b.alias is None
        assert b.filter is None
        assert b.columns is None
        assert b.exclude_columns is None
        assert b.order_by is None
        assert b.descending is False
        assert b.limit is None
        assert b.offset is None
        assert b.group_by is None

    def test_with_query_params(self):
        b = FilterBinding(
            context="Data/Sales/Monthly",
            alias="sales",
            filter="year == 2025",
            columns=["month", "revenue"],
            exclude_columns=["internal_id"],
            order_by="month",
            descending=True,
            limit=500,
            offset=10,
            group_by=["region"],
        )
        assert b.filter == "year == 2025"
        assert b.columns == ["month", "revenue"]
        assert b.limit == 500
        assert b.group_by == ["region"]

    def test_default_operation_field(self):
        b = FilterBinding(context="Data/X")
        assert b.operation == "filter"


class TestReduceBinding:
    def test_minimal(self):
        b = ReduceBinding(context="Data/Sales", metric="count", columns="id")
        assert b.operation == "reduce"
        assert b.metric == "count"
        assert b.columns == "id"

    def test_with_group_by(self):
        b = ReduceBinding(
            context="Data/Sales",
            metric="sum",
            columns=["revenue", "cost"],
            filter="year == 2025",
            group_by=["region"],
            result_where="revenue > 1000",
        )
        assert b.group_by == ["region"]
        assert b.result_where == "revenue > 1000"


class TestJoinBinding:
    def test_minimal(self):
        b = JoinBinding(
            tables=["Data/Orders", "Data/Customers"],
            join_expr="Data/Orders.cust_id == Data/Customers.id",
            select={"Data/Orders.amount": "amount", "Data/Customers.name": "name"},
        )
        assert b.operation == "join"
        assert len(b.tables) == 2
        assert b.mode == "inner"
        assert b.result_limit == 100
        assert b.result_offset == 0

    def test_full_params(self):
        b = JoinBinding(
            tables=["Data/A", "Data/B"],
            join_expr="Data/A.id == Data/B.fk",
            select={"Data/A.val": "val"},
            mode="left",
            left_where="active == True",
            right_where="created > '2025-01-01'",
            result_where="val > 10",
            result_limit=50,
            result_offset=25,
        )
        assert b.mode == "left"
        assert b.left_where == "active == True"
        assert b.result_limit == 50


class TestJoinReduceBinding:
    def test_minimal(self):
        b = JoinReduceBinding(
            tables=["Data/Orders", "Data/Products"],
            join_expr="Data/Orders.pid == Data/Products.id",
            select={"Data/Orders.amount": "amount", "Data/Products.cat": "cat"},
            metric="sum",
            columns="amount",
        )
        assert b.operation == "join_reduce"
        assert b.metric == "sum"

    def test_with_group_by(self):
        b = JoinReduceBinding(
            tables=["Data/A", "Data/B"],
            join_expr="Data/A.id == Data/B.fk",
            select={"Data/A.val": "val", "Data/B.cat": "cat"},
            metric="avg",
            columns=["val"],
            group_by=["cat"],
            result_where="val > 5",
        )
        assert b.group_by == ["cat"]


class TestDataBindingDiscriminator:
    """Verify the discriminated union dispatches correctly from dicts."""

    adapter = TypeAdapter(DataBinding)

    def test_filter_from_dict(self):
        b = self.adapter.validate_python(
            {"operation": "filter", "context": "Data/X", "columns": ["a"]},
        )
        assert isinstance(b, FilterBinding)

    def test_filter_default_operation(self):
        b = self.adapter.validate_python(
            {"operation": "filter", "context": "Data/X"},
        )
        assert isinstance(b, FilterBinding)

    def test_reduce_from_dict(self):
        b = self.adapter.validate_python(
            {
                "operation": "reduce",
                "context": "Data/X",
                "metric": "count",
                "columns": "id",
            },
        )
        assert isinstance(b, ReduceBinding)

    def test_join_from_dict(self):
        b = self.adapter.validate_python(
            {
                "operation": "join",
                "tables": ["Data/A", "Data/B"],
                "join_expr": "Data/A.id == Data/B.fk",
                "select": {"Data/A.x": "x"},
            },
        )
        assert isinstance(b, JoinBinding)

    def test_join_reduce_from_dict(self):
        b = self.adapter.validate_python(
            {
                "operation": "join_reduce",
                "tables": ["Data/A", "Data/B"],
                "join_expr": "Data/A.id == Data/B.fk",
                "select": {"Data/A.x": "x"},
                "metric": "sum",
                "columns": "x",
            },
        )
        assert isinstance(b, JoinReduceBinding)

    def test_missing_operation_raises(self):
        with pytest.raises(Exception):
            self.adapter.validate_python({"context": "Data/X"})


class TestTileRecordRowNewFields:
    def test_on_data_script_default_none(self):
        row = TileRecordRow(
            token="abc123abc123",
            title="Test",
            html_content="<h1>Hello</h1>",
        )
        assert row.on_data_script is None
        assert row.data_bindings_json is None

    def test_on_data_script_set(self):
        row = TileRecordRow(
            token="abc123abc123",
            title="Test",
            html_content="<div></div>",
            on_data_script="const rows = data.sales;",
            data_bindings_json='[{"operation":"filter","context":"Data/X","alias":"sales"}]',
        )
        assert row.on_data_script == "const rows = data.sales;"
        assert "filter" in row.data_bindings_json

    def test_round_trip_model_dump(self):
        row = TileRecordRow(
            token="abc123abc123",
            title="Test",
            html_content="<div></div>",
            on_data_script="console.log(data);",
            data_bindings_json='[{"operation":"filter","context":"X","alias":"x"}]',
        )
        d = row.model_dump()
        restored = TileRecordRow(**d)
        assert restored.on_data_script == row.on_data_script
        assert restored.data_bindings_json == row.data_bindings_json


class TestSerializeBindings:
    def test_serialize_filter(self):
        bindings = [FilterBinding(context="Data/X", alias="x")]
        result = serialize_bindings(bindings)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["operation"] == "filter"
        assert parsed[0]["alias"] == "x"

    def test_serialize_mixed(self):
        bindings = [
            FilterBinding(context="Data/A", alias="a"),
            ReduceBinding(
                context="Data/B",
                alias="b",
                metric="sum",
                columns="val",
            ),
        ]
        result = serialize_bindings(bindings)
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["operation"] == "filter"
        assert parsed[1]["operation"] == "reduce"

    def test_round_trip(self):
        original = FilterBinding(
            context="Data/Sales",
            alias="sales",
            columns=["month", "revenue"],
            order_by="month",
        )
        serialized = serialize_bindings([original])
        parsed = json.loads(serialized)
        restored = FilterBinding(**parsed[0])
        assert restored.context == original.context
        assert restored.alias == original.alias
        assert restored.columns == original.columns


class TestEnsureBindingAliases:
    def test_auto_generates_from_context(self):
        bindings = [FilterBinding(context="Data/Sales/Monthly")]
        result = ensure_binding_aliases(bindings)
        assert result[0].alias == "monthly"

    def test_preserves_existing_alias(self):
        bindings = [FilterBinding(context="Data/X", alias="my_data")]
        result = ensure_binding_aliases(bindings)
        assert result[0].alias == "my_data"

    def test_auto_generates_for_join(self):
        bindings = [
            JoinBinding(
                tables=["Data/A", "Data/B"],
                join_expr="Data/A.id == Data/B.fk",
                select={"Data/A.x": "x"},
            ),
        ]
        result = ensure_binding_aliases(bindings)
        assert result[0].alias == "binding_0"

    def test_raises_on_duplicate_aliases(self):
        bindings = [
            FilterBinding(context="Data/X", alias="dup"),
            FilterBinding(context="Data/Y", alias="dup"),
        ]
        with pytest.raises(ValueError, match="duplicate alias 'dup'"):
            ensure_binding_aliases(bindings)

    def test_raises_on_invalid_identifier(self):
        bindings = [FilterBinding(context="Data/X", alias="my-data")]
        with pytest.raises(ValueError, match="not a valid JS identifier"):
            ensure_binding_aliases(bindings)

    def test_raises_on_leading_digit(self):
        bindings = [FilterBinding(context="Data/X", alias="123sales")]
        with pytest.raises(ValueError, match="not a valid JS identifier"):
            ensure_binding_aliases(bindings)

    def test_sanitises_non_identifier_context(self):
        bindings = [FilterBinding(context="Data/My Sales-2025")]
        result = ensure_binding_aliases(bindings)
        assert (
            result[0].alias.isidentifier()
            or result[0].alias.replace("$", "").isidentifier()
        )


class TestValidateOnData:
    def test_none_on_data_passes(self):
        validate_on_data(None, None)

    def test_on_data_without_bindings_raises(self):
        with pytest.raises(ValueError, match="on_data requires data_bindings"):
            validate_on_data("console.log(data);", None)

    def test_whitespace_only_raises(self):
        bindings = [FilterBinding(context="Data/X")]
        with pytest.raises(ValueError, match="non-empty JS code"):
            validate_on_data("   ", bindings)

    def test_valid_on_data_passes(self):
        bindings = [FilterBinding(context="Data/X")]
        validate_on_data("console.log(data);", bindings)


class TestDashboardTypes:
    def test_tile_position_defaults(self):
        pos = TilePosition(tile_token="tok123")
        assert pos.x == 0
        assert pos.y == 0
        assert pos.w == 6
        assert pos.h == 4

    def test_tile_position_custom(self):
        pos = TilePosition(tile_token="tok123", x=4, y=2, w=8, h=6)
        assert pos.x == 4
        assert pos.w == 8

    def test_dashboard_record_row(self):
        layout = json.dumps([{"tile_token": "t1", "x": 0, "y": 0, "w": 6, "h": 4}])
        row = DashboardRecordRow(
            token="dash123dash1",
            title="My Dashboard",
            layout=layout,
            tile_count=1,
        )
        assert row.tile_count == 1

    def test_dashboard_result_succeeded(self):
        result = DashboardResult(
            url="https://example.com/dashboard/view/abc123",
            token="abc123",
            title="Dashboard",
            tiles=[TilePosition(tile_token="t1")],
        )
        assert result.succeeded is True

    def test_dashboard_result_failed(self):
        result = DashboardResult(error="Failed")
        assert result.succeeded is False


# ---------------------------------------------------------------------------
# Context resolution tests
# ---------------------------------------------------------------------------

BASE = "uid-abc/42"

KNOWN_CONTEXTS = {
    f"{BASE}/Data",
    f"{BASE}/Data/MidlandHeart/Repairs/Facts/WorkOrders",
    f"{BASE}/Data/MidlandHeart/Repairs/Dims/Contractors",
    f"{BASE}/Data/Sales/Monthly",
    f"{BASE}/Contacts",
    f"{BASE}/Tasks",
    f"{BASE}/Knowledge/FAQ",
}


class TestMatchContext:
    """Unit tests for the _match_context resolution cascade."""

    def test_exact_match_fully_qualified(self):
        result = _match_context(
            f"{BASE}/Data/Sales/Monthly",
            BASE,
            KNOWN_CONTEXTS,
        )
        assert result == f"{BASE}/Data/Sales/Monthly"

    def test_base_prefixed_root_relative_data(self):
        result = _match_context("Data/Sales/Monthly", BASE, KNOWN_CONTEXTS)
        assert result == f"{BASE}/Data/Sales/Monthly"

    def test_base_prefixed_contacts(self):
        result = _match_context("Contacts", BASE, KNOWN_CONTEXTS)
        assert result == f"{BASE}/Contacts"

    def test_base_prefixed_tasks(self):
        result = _match_context("Tasks", BASE, KNOWN_CONTEXTS)
        assert result == f"{BASE}/Tasks"

    def test_base_prefixed_knowledge(self):
        result = _match_context("Knowledge/FAQ", BASE, KNOWN_CONTEXTS)
        assert result == f"{BASE}/Knowledge/FAQ"

    def test_suffix_match_relative(self):
        result = _match_context(
            "MidlandHeart/Repairs/Facts/WorkOrders",
            BASE,
            KNOWN_CONTEXTS,
        )
        assert result == f"{BASE}/Data/MidlandHeart/Repairs/Facts/WorkOrders"

    def test_suffix_match_deeper_relative(self):
        result = _match_context(
            "Repairs/Dims/Contractors",
            BASE,
            KNOWN_CONTEXTS,
        )
        assert result == f"{BASE}/Data/MidlandHeart/Repairs/Dims/Contractors"

    def test_no_match_raises(self):
        with pytest.raises(ValueError, match="No context found"):
            _match_context("Nonexistent/Table", BASE, KNOWN_CONTEXTS)

    def test_ambiguous_match_raises(self):
        ambiguous_known = {
            f"{BASE}/Data/ProjectA/Orders",
            f"{BASE}/Data/ProjectB/Orders",
        }
        with pytest.raises(ValueError, match="Ambiguous context"):
            _match_context("Orders", BASE, ambiguous_known)

    def test_empty_path_raises(self):
        with pytest.raises(ValueError, match="Empty context path"):
            _match_context("", BASE, KNOWN_CONTEXTS)

    def test_whitespace_stripped(self):
        result = _match_context("  Contacts  ", BASE, KNOWN_CONTEXTS)
        assert result == f"{BASE}/Contacts"

    def test_leading_slash_stripped(self):
        result = _match_context("/Data/Sales/Monthly", BASE, KNOWN_CONTEXTS)
        assert result == f"{BASE}/Data/Sales/Monthly"


class TestResolveBindingContexts:
    """Tests for resolve_binding_contexts with mocked unify API."""

    def _patch(self):
        """Return a context manager that patches ContextRegistry and unify."""
        patches = {}
        patches["registry"] = patch(
            "unity.dashboard_manager.ops.tile_ops.ContextRegistry",
        )
        patches["unify"] = patch(
            "unity.dashboard_manager.ops.tile_ops.unify",
        )
        return patches

    def test_filter_binding_resolved(self):
        with (
            patch(
                "unity.dashboard_manager.ops.tile_ops.ContextRegistry",
            ) as mock_reg,
            patch(
                "unity.dashboard_manager.ops.tile_ops.unify",
            ) as mock_unify,
        ):
            mock_reg._base_context = BASE
            mock_unify.get_contexts.return_value = {k: "" for k in KNOWN_CONTEXTS}

            bindings = [FilterBinding(context="Data/Sales/Monthly", alias="sales")]
            result = resolve_binding_contexts(bindings)

            assert len(result) == 1
            assert result[0].context == f"{BASE}/Data/Sales/Monthly"
            assert result[0].alias == "sales"

    def test_reduce_binding_resolved(self):
        with (
            patch(
                "unity.dashboard_manager.ops.tile_ops.ContextRegistry",
            ) as mock_reg,
            patch(
                "unity.dashboard_manager.ops.tile_ops.unify",
            ) as mock_unify,
        ):
            mock_reg._base_context = BASE
            mock_unify.get_contexts.return_value = {k: "" for k in KNOWN_CONTEXTS}

            bindings = [
                ReduceBinding(context="Contacts", metric="count", columns="name"),
            ]
            result = resolve_binding_contexts(bindings)

            assert result[0].context == f"{BASE}/Contacts"

    def test_join_binding_tables_and_expr_resolved(self):
        with (
            patch(
                "unity.dashboard_manager.ops.tile_ops.ContextRegistry",
            ) as mock_reg,
            patch(
                "unity.dashboard_manager.ops.tile_ops.unify",
            ) as mock_unify,
        ):
            mock_reg._base_context = BASE
            mock_unify.get_contexts.return_value = {k: "" for k in KNOWN_CONTEXTS}

            bindings = [
                JoinBinding(
                    tables=[
                        "Data/MidlandHeart/Repairs/Facts/WorkOrders",
                        "Data/MidlandHeart/Repairs/Dims/Contractors",
                    ],
                    join_expr=(
                        "Data/MidlandHeart/Repairs/Facts/WorkOrders.contractor_id"
                        " == "
                        "Data/MidlandHeart/Repairs/Dims/Contractors.id"
                    ),
                    select={
                        "Data/MidlandHeart/Repairs/Facts/WorkOrders.amount": "amount",
                        "Data/MidlandHeart/Repairs/Dims/Contractors.name": "name",
                    },
                ),
            ]
            result = resolve_binding_contexts(bindings)
            b = result[0]

            wo = f"{BASE}/Data/MidlandHeart/Repairs/Facts/WorkOrders"
            co = f"{BASE}/Data/MidlandHeart/Repairs/Dims/Contractors"
            assert b.tables == [wo, co]
            assert wo in b.join_expr
            assert co in b.join_expr
            assert any(wo in k for k in b.select)
            assert any(co in k for k in b.select)

    def test_join_reduce_binding_resolved(self):
        with (
            patch(
                "unity.dashboard_manager.ops.tile_ops.ContextRegistry",
            ) as mock_reg,
            patch(
                "unity.dashboard_manager.ops.tile_ops.unify",
            ) as mock_unify,
        ):
            mock_reg._base_context = BASE
            mock_unify.get_contexts.return_value = {k: "" for k in KNOWN_CONTEXTS}

            bindings = [
                JoinReduceBinding(
                    tables=["Data/Sales/Monthly", "Contacts"],
                    join_expr="Data/Sales/Monthly.rep_id == Contacts.id",
                    select={
                        "Data/Sales/Monthly.revenue": "revenue",
                        "Contacts.name": "rep",
                    },
                    metric="sum",
                    columns="revenue",
                    group_by=["rep"],
                ),
            ]
            result = resolve_binding_contexts(bindings)
            b = result[0]

            sm = f"{BASE}/Data/Sales/Monthly"
            ct = f"{BASE}/Contacts"
            assert b.tables == [sm, ct]
            assert sm in b.join_expr
            assert ct in b.join_expr
            assert any(sm in k for k in b.select)
            assert any(ct in k for k in b.select)

    def test_no_base_context_returns_unchanged(self):
        with (
            patch(
                "unity.dashboard_manager.ops.tile_ops.ContextRegistry",
            ) as mock_reg,
            patch(
                "unity.dashboard_manager.ops.tile_ops.unify",
            ) as mock_unify,
        ):
            mock_reg._base_context = None
            mock_unify.get_active_context.return_value = {"read": ""}

            bindings = [FilterBinding(context="Data/X", alias="x")]
            result = resolve_binding_contexts(bindings)

            assert result[0].context == "Data/X"

    def test_no_known_contexts_returns_unchanged(self):
        with (
            patch(
                "unity.dashboard_manager.ops.tile_ops.ContextRegistry",
            ) as mock_reg,
            patch(
                "unity.dashboard_manager.ops.tile_ops.unify",
            ) as mock_unify,
        ):
            mock_reg._base_context = BASE
            mock_unify.get_contexts.return_value = {}

            bindings = [FilterBinding(context="Data/X", alias="x")]
            result = resolve_binding_contexts(bindings)

            assert result[0].context == "Data/X"

    def test_already_qualified_passes_through(self):
        with (
            patch(
                "unity.dashboard_manager.ops.tile_ops.ContextRegistry",
            ) as mock_reg,
            patch(
                "unity.dashboard_manager.ops.tile_ops.unify",
            ) as mock_unify,
        ):
            mock_reg._base_context = BASE
            mock_unify.get_contexts.return_value = {k: "" for k in KNOWN_CONTEXTS}

            fq = f"{BASE}/Data/Sales/Monthly"
            bindings = [FilterBinding(context=fq, alias="sales")]
            result = resolve_binding_contexts(bindings)

            assert result[0].context == fq

    def test_suffix_match_relative_path(self):
        with (
            patch(
                "unity.dashboard_manager.ops.tile_ops.ContextRegistry",
            ) as mock_reg,
            patch(
                "unity.dashboard_manager.ops.tile_ops.unify",
            ) as mock_unify,
        ):
            mock_reg._base_context = BASE
            mock_unify.get_contexts.return_value = {k: "" for k in KNOWN_CONTEXTS}

            bindings = [
                FilterBinding(
                    context="MidlandHeart/Repairs/Facts/WorkOrders",
                    alias="wo",
                ),
            ]
            result = resolve_binding_contexts(bindings)

            assert result[0].context == (
                f"{BASE}/Data/MidlandHeart/Repairs/Facts/WorkOrders"
            )
