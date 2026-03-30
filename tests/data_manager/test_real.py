"""
Integration tests for the real DataManager against the Unify backend.

These tests exercise the full DataManager public API (table CRUD, column ops,
query/mutation/embedding/visualization/join operations) using the real backend,
following the same ``@_handle_project`` pattern as ContactManager's test_basic.py.

Each test gets a fresh, isolated Unify context that is cleaned up after the test.
"""

from __future__ import annotations

import pytest

from unity.data_manager.data_manager import DataManager
from unity.data_manager.types import TableDescription, PlotResult
from unity.manager_registry import ManagerRegistry
from tests.helpers import _handle_project

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _fresh_dm() -> DataManager:
    """Create a fresh DataManager instance (clears registry singleton)."""
    ManagerRegistry.clear()
    return DataManager()


# ────────────────────────────────────────────────────────────────────────────
# Table Management
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_create_table_basic():
    """create_table should create a table and return resolved path."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/products",
        description="Product catalog",
        fields={"item_id": "int", "name": "str", "price": "float"},
    )

    assert "test_real/products" in path
    # Under _handle_project the path is fully qualified with the project prefix,
    # but the "Data" segment should always be present.
    assert "Data" in path


@_handle_project
def test_create_table_with_unique_keys_and_auto_counting():
    """create_table should accept unique_keys and auto_counting."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/orders",
        description="Order records",
        fields={"order_id": "int", "product": "str", "quantity": "int"},
        unique_keys={"order_id": "int"},
        auto_counting={"order_id": None},
    )

    assert "test_real/orders" in path

    # Verify metadata via get_table
    ctx_info = dm.get_table(path)
    assert ctx_info.get("description") == "Order records"


@_handle_project
def test_describe_table():
    """describe_table should return structured TableDescription."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/desc",
        description="For describe test",
        fields={"item_id": "int", "name": "str", "active": "bool"},
    )

    desc = dm.describe_table(path)

    assert isinstance(desc, TableDescription)
    assert desc.context == path
    assert desc.description == "For describe test"

    col_names = [c.name for c in desc.table_schema.columns]
    assert "item_id" in col_names
    assert "name" in col_names
    assert "active" in col_names


@_handle_project
def test_get_columns():
    """get_columns should return column type info including private columns."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/cols",
        fields={"item_id": "int", "value": "str"},
    )

    columns = dm.get_columns(path)

    assert isinstance(columns, dict)
    assert "item_id" in columns
    assert "value" in columns


@_handle_project
def test_get_table():
    """get_table should return lightweight context metadata."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/meta",
        description="Metadata test",
        fields={"item_id": "int"},
        unique_keys={"item_id": "int"},
    )

    ctx_info = dm.get_table(path)

    assert isinstance(ctx_info, dict)
    assert ctx_info.get("description") == "Metadata test"


@_handle_project
def test_list_tables():
    """list_tables should discover tables by prefix."""
    dm = _fresh_dm()

    path_a = dm.create_table("test_real/list_a")
    path_b = dm.create_table("test_real/list_b")

    # Derive a common prefix from the actual returned path (which is fully
    # qualified under _handle_project) rather than hard-coding "Data/...".
    common_prefix = path_a.rsplit("/list_a", 1)[0] + "/list"

    # With column info (default)
    tables = dm.list_tables(prefix=common_prefix)
    assert isinstance(tables, dict)
    assert path_a in tables
    assert path_b in tables

    # Without column info
    names = dm.list_tables(prefix=common_prefix, include_column_info=False)
    assert isinstance(names, list)
    assert path_a in names
    assert path_b in names


@_handle_project
def test_delete_table():
    """delete_table should remove the table."""
    dm = _fresh_dm()

    path = dm.create_table("test_real/to_delete")

    names_before = dm.list_tables(prefix=path, include_column_info=False)
    assert path in names_before

    dm.delete_table(path, dangerous_ok=True)

    names_after = dm.list_tables(prefix=path, include_column_info=False)
    assert path not in names_after


@_handle_project
def test_delete_table_requires_dangerous_ok():
    """delete_table should raise without dangerous_ok."""
    dm = _fresh_dm()

    path = dm.create_table("test_real/protected")

    with pytest.raises((ValueError, Exception)):
        dm.delete_table(path)


@_handle_project
def test_rename_table():
    """rename_table should move data to new context path."""
    dm = _fresh_dm()

    old_path = dm.create_table("test_real/old_name")
    dm.insert_rows(old_path, [{"x": 1}])

    new_path = old_path.replace("old_name", "new_name")
    dm.rename_table(old_path, new_path)

    # Old path should be gone
    old_tables = dm.list_tables(prefix=old_path, include_column_info=False)
    assert old_path not in old_tables

    # New path should exist with data
    rows = dm.filter(new_path)
    assert len(rows) == 1
    assert rows[0]["x"] == 1


# ────────────────────────────────────────────────────────────────────────────
# Column Operations
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_create_column():
    """create_column should add a new column to the table."""
    dm = _fresh_dm()

    path = dm.create_table("test_real/col_ops", fields={"item_id": "int"})

    dm.create_column(path, column_name="score", column_type="float")

    columns = dm.get_columns(path)
    assert "score" in columns


@_handle_project
def test_delete_column():
    """delete_column should remove a column."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/del_col",
        fields={"item_id": "int", "temp": "str"},
    )

    dm.delete_column(path, column_name="temp")

    columns = dm.get_columns(path)
    assert "temp" not in columns
    assert "item_id" in columns


@_handle_project
def test_rename_column():
    """rename_column should change the column name."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/ren_col",
        fields={"item_id": "int", "old_name": "str"},
    )
    # The Unify backend requires the table to have data before renaming columns.
    dm.insert_rows(path, [{"item_id": 1, "old_name": "placeholder"}])

    dm.rename_column(path, old_name="old_name", new_name="new_name")

    columns = dm.get_columns(path)
    assert "new_name" in columns
    assert "old_name" not in columns


@_handle_project
def test_create_derived_column():
    """create_derived_column should compute values from existing columns."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/derived",
        fields={"quantity": "int", "unit_price": "float"},
    )
    dm.insert_rows(path, [{"quantity": 5, "unit_price": 10.0}])

    dm.create_derived_column(
        path,
        column_name="total",
        equation="{quantity} * {unit_price}",
    )

    rows = dm.filter(path, columns=["quantity", "unit_price", "total"])
    assert len(rows) == 1
    assert rows[0]["total"] == 50.0


# ────────────────────────────────────────────────────────────────────────────
# Insert, Filter, Update, Delete
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_insert_and_filter_rows():
    """insert_rows + filter should round-trip data correctly."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/crud",
        fields={"item_id": "int", "name": "str", "price": "float", "category": "str"},
    )

    ids = dm.insert_rows(
        path,
        [
            {"item_id": 1, "name": "Widget A", "price": 10.0, "category": "widgets"},
            {"item_id": 2, "name": "Widget B", "price": 20.0, "category": "widgets"},
            {"item_id": 3, "name": "Gadget X", "price": 50.0, "category": "gadgets"},
        ],
    )

    assert len(ids) == 3

    # Filter all
    rows = dm.filter(path)
    assert len(rows) == 3

    # Filter with expression
    widgets = dm.filter(path, filter="category == 'widgets'")
    assert len(widgets) == 2
    assert all(r["category"] == "widgets" for r in widgets)


@_handle_project
def test_filter_with_column_selection():
    """filter should respect columns and exclude_columns."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/colsel",
        fields={"item_id": "int", "name": "str", "secret": "str"},
    )
    dm.insert_rows(path, [{"item_id": 1, "name": "Alice", "secret": "hidden"}])

    # Include only specified columns
    rows = dm.filter(path, columns=["item_id", "name"])
    assert len(rows) == 1
    assert "item_id" in rows[0]
    assert "name" in rows[0]
    assert "secret" not in rows[0]

    # Exclude specific columns
    rows = dm.filter(path, exclude_columns=["secret"])
    assert len(rows) == 1
    assert "secret" not in rows[0]
    assert "item_id" in rows[0]


@_handle_project
def test_filter_pagination():
    """filter should support limit and offset for pagination."""
    dm = _fresh_dm()

    path = dm.create_table("test_real/page", fields={"seq": "int"})
    dm.insert_rows(path, [{"seq": i} for i in range(10)])

    page1 = dm.filter(path, limit=3, offset=0)
    page2 = dm.filter(path, limit=3, offset=3)

    assert len(page1) == 3
    assert len(page2) == 3

    # No overlap
    ids1 = {r["seq"] for r in page1}
    ids2 = {r["seq"] for r in page2}
    assert ids1.isdisjoint(ids2)


@_handle_project
def test_filter_return_ids_only():
    """filter with return_ids_only=True should return log IDs."""
    dm = _fresh_dm()

    path = dm.create_table("test_real/ids", fields={"item_id": "int"})
    dm.insert_rows(path, [{"item_id": 1}, {"item_id": 2}, {"item_id": 3}])

    ids = dm.filter(path, return_ids_only=True)

    assert isinstance(ids, list)
    assert len(ids) == 3
    assert all(isinstance(i, int) for i in ids)


@_handle_project
def test_insert_with_unique_keys():
    """insert_rows into a table with unique_keys should upsert server-side."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/dedupe",
        fields={"sku": "str", "price": "float"},
        unique_keys={"sku": "str"},
    )

    dm.insert_rows(path, [{"sku": "A1", "price": 10.0}])
    dm.insert_rows(path, [{"sku": "A1", "price": 15.0}])

    rows = dm.filter(path)
    assert len(rows) == 1
    assert rows[0]["price"] == 15.0


@_handle_project
def test_update_rows():
    """update_rows should modify matching rows."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/upd",
        fields={"item_id": "int", "status": "str"},
    )
    dm.insert_rows(
        path,
        [
            {"item_id": 1, "status": "pending"},
            {"item_id": 2, "status": "pending"},
            {"item_id": 3, "status": "done"},
        ],
    )

    updated = dm.update_rows(
        path,
        updates={"status": "approved"},
        filter="status == 'pending'",
    )

    assert updated == 2

    approved = dm.filter(path, filter="status == 'approved'")
    assert len(approved) == 2


@_handle_project
def test_delete_rows_by_filter():
    """delete_rows with filter should remove matching rows."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/del",
        fields={"item_id": "int", "temp": "bool"},
    )
    dm.insert_rows(
        path,
        [
            {"item_id": 1, "temp": True},
            {"item_id": 2, "temp": False},
            {"item_id": 3, "temp": True},
        ],
    )

    deleted = dm.delete_rows(path, filter="temp == True", dangerous_ok=True)
    assert deleted == 2

    remaining = dm.filter(path)
    assert len(remaining) == 1
    assert remaining[0]["item_id"] == 2


@_handle_project
def test_delete_rows_by_log_ids():
    """delete_rows with log_ids should remove specific rows."""
    dm = _fresh_dm()

    path = dm.create_table("test_real/del_ids", fields={"item_id": "int"})
    dm.insert_rows(path, [{"item_id": 1}, {"item_id": 2}, {"item_id": 3}])

    # Get log IDs for item_id == 1
    ids_to_delete = dm.filter(path, filter="item_id == 1", return_ids_only=True)
    assert len(ids_to_delete) == 1

    deleted = dm.delete_rows(path, log_ids=ids_to_delete, dangerous_ok=True)
    assert deleted == 1

    remaining = dm.filter(path)
    assert len(remaining) == 2


@_handle_project
def test_delete_rows_requires_filter_or_log_ids():
    """delete_rows should raise when neither filter nor log_ids is provided."""
    dm = _fresh_dm()

    path = dm.create_table("test_real/del_safe", fields={"item_id": "int"})
    dm.insert_rows(path, [{"item_id": 1}])

    with pytest.raises(ValueError, match="filter or log_ids"):
        dm.delete_rows(path, dangerous_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# Reduce (aggregation)
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_reduce_count():
    """reduce with count metric should return row count."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/red",
        fields={"item_id": "int", "price": "float", "category": "str"},
    )
    dm.insert_rows(
        path,
        [
            {"item_id": 1, "price": 10.0, "category": "A"},
            {"item_id": 2, "price": 20.0, "category": "A"},
            {"item_id": 3, "price": 50.0, "category": "B"},
        ],
    )

    count = dm.reduce(path, metric="count", columns="item_id")
    assert count == 3


@_handle_project
def test_reduce_sum_and_avg():
    """reduce with sum and avg metrics."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/red_agg",
        fields={"item_id": "int", "value": "float"},
    )
    dm.insert_rows(
        path,
        [
            {"item_id": 1, "value": 10.0},
            {"item_id": 2, "value": 20.0},
            {"item_id": 3, "value": 30.0},
        ],
    )

    total = dm.reduce(path, metric="sum", columns="value")
    assert total == 60.0

    avg = dm.reduce(path, metric="mean", columns="value")
    assert avg == 20.0


@_handle_project
def test_reduce_with_filter():
    """reduce should respect filter expressions."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/red_filt",
        fields={"item_id": "int", "category": "str"},
    )
    dm.insert_rows(
        path,
        [
            {"item_id": 1, "category": "A"},
            {"item_id": 2, "category": "A"},
            {"item_id": 3, "category": "B"},
        ],
    )

    count_a = dm.reduce(
        path,
        metric="count",
        columns="item_id",
        filter="category == 'A'",
    )
    assert count_a == 2


@_handle_project
def test_reduce_with_group_by():
    """reduce with group_by should return per-group results."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/red_grp",
        fields={"item_id": "int", "category": "str", "amount": "float"},
    )
    dm.insert_rows(
        path,
        [
            {"item_id": 1, "category": "A", "amount": 10.0},
            {"item_id": 2, "category": "A", "amount": 20.0},
            {"item_id": 3, "category": "B", "amount": 50.0},
        ],
    )

    results = dm.reduce(path, metric="sum", columns="amount", group_by="category")

    # reduce with group_by returns a dict keyed by group value, e.g.
    # {"A": {"sum": 30.0, ...}, "B": {"sum": 50.0, ...}}
    assert isinstance(results, dict)
    assert "A" in results
    assert "B" in results

    # The backend may return "sum" or "shared_value" depending on group size;
    # extract whichever is non-None.
    def _get_metric(group_result: dict) -> float:
        val = group_result.get("sum")
        if val is None:
            val = group_result.get("shared_value")
        assert val is not None
        return val

    assert _get_metric(results["A"]) == 30.0
    assert _get_metric(results["B"]) == 50.0


# ────────────────────────────────────────────────────────────────────────────
# Embedding operations
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_ensure_vector_column_and_vectorize():
    """ensure_vector_column + vectorize_rows should prepare table for search."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/emb",
        fields={"item_id": "int", "text": "str"},
    )
    dm.insert_rows(
        path,
        [
            {"item_id": 1, "text": "Machine learning for image classification"},
            {"item_id": 2, "text": "Natural language processing with transformers"},
            {"item_id": 3, "text": "Database indexing and query optimization"},
        ],
    )

    emb_col = dm.ensure_vector_column(path, source_column="text")
    assert emb_col == "_text_emb"

    count = dm.vectorize_rows(path, source_column="text")
    assert isinstance(count, int)

    # Verify embeddings exist
    desc = dm.describe_table(path)
    assert desc.has_embeddings
    assert "_text_emb" in desc.embedding_columns


# ────────────────────────────────────────────────────────────────────────────
# Search (semantic)
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_search_semantic():
    """search should return semantically relevant results."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/srch",
        fields={"item_id": "int", "text": "str"},
    )
    dm.insert_rows(
        path,
        [
            {"item_id": 1, "text": "Machine learning for image classification"},
            {"item_id": 2, "text": "Natural language processing with transformers"},
            {"item_id": 3, "text": "Database indexing and query optimization"},
            {"item_id": 4, "text": "Deep neural networks and computer vision"},
        ],
    )

    # Ensure embeddings
    dm.ensure_vector_column(path, source_column="text")
    dm.vectorize_rows(path, source_column="text")

    results = dm.search(
        path,
        references={"text": "neural network image recognition"},
        k=2,
    )

    assert len(results) <= 2
    # Top results should relate to ML/vision, not databases
    top_ids = {r["item_id"] for r in results}
    assert top_ids.issubset({1, 2, 4})


# ────────────────────────────────────────────────────────────────────────────
# Join operations
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_filter_join():
    """filter_join should combine rows from two tables."""
    dm = _fresh_dm()

    products_path = dm.create_table(
        "test_real/join_products",
        fields={"product_id": "int", "name": "str", "price": "float"},
    )
    orders_path = dm.create_table(
        "test_real/join_orders",
        fields={"order_id": "int", "product_ref": "int", "quantity": "int"},
    )

    dm.insert_rows(
        products_path,
        [
            {"product_id": 1, "name": "Widget", "price": 10.0},
            {"product_id": 2, "name": "Gadget", "price": 25.0},
        ],
    )
    dm.insert_rows(
        orders_path,
        [
            {"order_id": 101, "product_ref": 1, "quantity": 5},
            {"order_id": 102, "product_ref": 2, "quantity": 3},
            {"order_id": 103, "product_ref": 1, "quantity": 2},
        ],
    )

    results = dm.filter_join(
        tables=[orders_path, products_path],
        join_expr=f"{orders_path}.product_ref == {products_path}.product_id",
        select={
            f"{orders_path}.order_id": "order_id",
            f"{orders_path}.quantity": "qty",
            f"{products_path}.name": "product_name",
            f"{products_path}.price": "price",
        },
    )

    assert len(results) == 3

    for r in results:
        assert "order_id" in r
        assert "product_name" in r
        assert "price" in r


@_handle_project
def test_filter_join_with_result_where():
    """filter_join should support post-join filtering."""
    dm = _fresh_dm()

    left = dm.create_table(
        "test_real/jl",
        fields={"item_id": "int", "val": "float"},
    )
    right = dm.create_table(
        "test_real/jr",
        fields={"item_id": "int", "label": "str"},
    )

    dm.insert_rows(left, [{"item_id": 1, "val": 10.0}, {"item_id": 2, "val": 100.0}])
    dm.insert_rows(
        right,
        [{"item_id": 1, "label": "low"}, {"item_id": 2, "label": "high"}],
    )

    results = dm.filter_join(
        tables=[left, right],
        join_expr=f"{left}.item_id == {right}.item_id",
        select={
            f"{left}.item_id": "row_id",
            f"{left}.val": "value",
            f"{right}.label": "label",
        },
        result_where="value > 50",
    )

    assert len(results) == 1
    assert results[0]["label"] == "high"


@_handle_project
def test_join_tables_materialized():
    """join_tables should create a materialized destination table."""
    dm = _fresh_dm()

    t1 = dm.create_table("test_real/jt1", fields={"item_id": "int", "x": "str"})
    t2 = dm.create_table("test_real/jt2", fields={"item_id": "int", "y": "str"})

    dm.insert_rows(t1, [{"item_id": 1, "x": "a"}])
    dm.insert_rows(t2, [{"item_id": 1, "y": "b"}])

    # Use a relative dest path — _resolve_context will qualify it.
    result_path = dm.join_tables(
        left_table=t1,
        right_table=t2,
        join_expr=f"{t1}.item_id == {t2}.item_id",
        dest_table="test_real/jt_dest",
        select={f"{t1}.x": "x_val", f"{t2}.y": "y_val"},
    )

    rows = dm.filter(result_path)
    assert len(rows) == 1
    assert rows[0]["x_val"] == "a"
    assert rows[0]["y_val"] == "b"

    # Cleanup
    dm.delete_table(result_path, dangerous_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# Visualization
#
# Note: Plot tests are marked xfail because the Console Plot API currently
# returns HTTP 400 "Missing projectConfig.projectName" due to a snake_case vs
# camelCase key mismatch in plot_ops._build_project_config_dict.
# The tests below verify correct *DataManager* usage regardless.
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.xfail(
    reason="Console Plot API snake_case/camelCase key mismatch",
    strict=False,
)
@_handle_project
def test_plot_bar_chart():
    """plot should return a PlotResult with a URL."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/plot",
        fields={"category": "str", "revenue": "float"},
    )
    dm.insert_rows(
        path,
        [
            {"category": "East", "revenue": 100.0},
            {"category": "West", "revenue": 200.0},
            {"category": "North", "revenue": 150.0},
        ],
    )

    result = dm.plot(
        path,
        plot_type="bar",
        x="category",
        y="revenue",
        aggregate="sum",
        title="Revenue by Region",
    )

    assert isinstance(result, PlotResult)
    assert result.succeeded
    assert result.url is not None


@pytest.mark.xfail(
    reason="Console Plot API snake_case/camelCase key mismatch",
    strict=False,
)
@_handle_project
def test_plot_batch():
    """plot_batch should produce one PlotResult per context."""
    dm = _fresh_dm()

    p1 = dm.create_table("test_real/pb1", fields={"seq": "int", "val": "float"})
    p2 = dm.create_table("test_real/pb2", fields={"seq": "int", "val": "float"})

    dm.insert_rows(p1, [{"seq": 1, "val": 10.0}, {"seq": 2, "val": 20.0}])
    dm.insert_rows(p2, [{"seq": 1, "val": 30.0}, {"seq": 2, "val": 40.0}])

    results = dm.plot_batch(
        [p1, p2],
        plot_type="bar",
        x="seq",
        y="val",
    )

    assert len(results) == 2
    assert all(isinstance(r, PlotResult) for r in results)
    assert all(r.succeeded for r in results)


@pytest.mark.xfail(
    reason="Console Plot API snake_case/camelCase key mismatch",
    strict=False,
)
@_handle_project
def test_plot_histogram():
    """plot histogram should work with x-only config."""
    dm = _fresh_dm()

    path = dm.create_table(
        "test_real/hist",
        fields={"price": "float"},
    )
    dm.insert_rows(
        path,
        [{"price": float(i)} for i in range(20)],
    )

    result = dm.plot(
        path,
        plot_type="histogram",
        x="price",
        bin_count=5,
        title="Price Distribution",
    )

    assert isinstance(result, PlotResult)
    assert result.succeeded


# ────────────────────────────────────────────────────────────────────────────
# End-to-end: full CRUD lifecycle
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_full_lifecycle():
    """End-to-end: create → insert → query → update → delete → drop."""
    dm = _fresh_dm()

    # Create
    path = dm.create_table(
        "test_real/lifecycle",
        description="Lifecycle test",
        fields={"item_id": "int", "name": "str", "score": "float"},
    )

    # Insert
    dm.insert_rows(
        path,
        [
            {"item_id": 1, "name": "Alpha", "score": 80.0},
            {"item_id": 2, "name": "Beta", "score": 90.0},
            {"item_id": 3, "name": "Gamma", "score": 70.0},
        ],
    )

    # Verify
    assert dm.reduce(path, metric="count", columns="item_id") == 3
    assert dm.reduce(path, metric="sum", columns="score") == 240.0

    # Filter
    high = dm.filter(path, filter="score >= 80")
    assert len(high) == 2

    # Update
    dm.update_rows(path, updates={"score": 95.0}, filter="name == 'Gamma'")
    gamma = dm.filter(path, filter="name == 'Gamma'")
    assert gamma[0]["score"] == 95.0

    # Delete a row
    dm.delete_rows(path, filter="name == 'Alpha'", dangerous_ok=True)
    assert dm.reduce(path, metric="count", columns="item_id") == 2

    # Add a column
    dm.create_column(path, column_name="grade", column_type="str")

    # Describe to confirm
    desc = dm.describe_table(path)
    col_names = [c.name for c in desc.table_schema.columns]
    assert "grade" in col_names

    # Drop the table
    dm.delete_table(path, dangerous_ok=True)

    # Confirm it's gone
    remaining = dm.list_tables(prefix=path, include_column_info=False)
    assert path not in remaining
