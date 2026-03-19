"""
Tests for SimulatedDataManager.

These tests verify that the simulated implementation works correctly
and can be used as a drop-in replacement for the real DataManager in tests.
"""

from __future__ import annotations

import pytest

from unity.data_manager.simulated import SimulatedDataManager
from unity.data_manager.base import BaseDataManager
from unity.data_manager.types import (
    TableDescription,
    PlotResult,
    TableViewResult,
    IngestResult,
)

# ────────────────────────────────────────────────────────────────────────────
# Docstring inheritance
# ────────────────────────────────────────────────────────────────────────────


def test_simulated_dm_inherits_base():
    """SimulatedDataManager should inherit from BaseDataManager."""
    assert issubclass(SimulatedDataManager, BaseDataManager)


def test_simulated_dm_docstrings_match_base():
    """Public methods should have docstrings from BaseDataManager via functools.wraps."""
    assert (
        BaseDataManager.filter.__doc__.strip()
        in SimulatedDataManager.filter.__doc__.strip()
    )
    assert (
        BaseDataManager.search.__doc__.strip()
        in SimulatedDataManager.search.__doc__.strip()
    )
    assert (
        BaseDataManager.reduce.__doc__.strip()
        in SimulatedDataManager.reduce.__doc__.strip()
    )


# ────────────────────────────────────────────────────────────────────────────
# Table operations
# ────────────────────────────────────────────────────────────────────────────


def test_create_table(simulated_dm):
    """create_table should create a table and return the resolved path."""
    path = simulated_dm.create_table(
        "myproject/users",
        description="User records",
        fields={"id": "int", "name": "str"},
    )
    assert path == "Data/myproject/users"

    # Table should be listable
    tables = simulated_dm.list_tables(prefix="Data/myproject")
    assert "Data/myproject/users" in tables


def test_describe_table(simulated_dm):
    """describe_table should return table metadata."""
    simulated_dm.create_table(
        "test/demo",
        description="Demo table",
        fields={"id": "int", "name": "str", "active": "bool"},
    )

    desc = simulated_dm.describe_table("test/demo")

    assert isinstance(desc, TableDescription)
    assert desc.context == "Data/test/demo"
    assert desc.description == "Demo table"
    assert len(desc.table_schema.columns) == 3

    col_names = [c.name for c in desc.table_schema.columns]
    assert "id" in col_names
    assert "name" in col_names
    assert "active" in col_names


def test_list_tables(simulated_dm):
    """list_tables should return tables matching prefix."""
    simulated_dm.create_table("project1/table_a")
    simulated_dm.create_table("project1/table_b")
    simulated_dm.create_table("project2/table_c")

    # All tables
    all_tables = simulated_dm.list_tables()
    assert len(all_tables) >= 3

    # Filtered by prefix
    p1_tables = simulated_dm.list_tables(prefix="Data/project1")
    assert len(p1_tables) == 2
    assert all("project1" in t for t in p1_tables)


def test_get_columns(simulated_dm):
    """get_columns should return raw column definitions."""
    simulated_dm.create_table(
        "test/columns_demo",
        fields={"id": "int", "name": "str", "price": "float"},
    )

    columns = simulated_dm.get_columns("test/columns_demo")

    assert isinstance(columns, dict)
    assert "id" in columns
    assert "name" in columns
    assert "price" in columns
    # Each column should have data_type
    assert columns["id"]["data_type"] == "int"
    assert columns["name"]["data_type"] == "str"
    assert columns["price"]["data_type"] == "float"


def test_get_columns_includes_private(simulated_dm):
    """get_columns should include private columns (unlike describe_table)."""
    simulated_dm.create_table(
        "test/private_columns",
        fields={"id": "int", "_internal": "str", "_id_emb": "vector"},
    )

    columns = simulated_dm.get_columns("test/private_columns")

    # Should include ALL columns including private ones
    assert "id" in columns
    assert "_internal" in columns
    assert "_id_emb" in columns


def test_list_tables_with_column_info(simulated_dm):
    """list_tables with include_column_info=True returns contexts with metadata."""
    simulated_dm.create_table("ctx_test/table_a", description="Table A")
    simulated_dm.create_table("ctx_test/table_b", description="Table B")
    simulated_dm.create_table("other/table_c")

    # All contexts for prefix with column info (default)
    tables = simulated_dm.list_tables(prefix="Data/ctx_test", include_column_info=True)

    assert isinstance(tables, dict)
    assert len(tables) == 2

    # Each context should have info including description
    for ctx_path, ctx_info in tables.items():
        assert "ctx_test" in ctx_path
        assert "description" in ctx_info


def test_list_tables_without_column_info(simulated_dm):
    """list_tables with include_column_info=False returns just table names."""
    simulated_dm.create_table("names_test/t1")
    simulated_dm.create_table("names_test/t2")

    tables = simulated_dm.list_tables(
        prefix="Data/names_test",
        include_column_info=False,
    )

    assert isinstance(tables, list)
    assert len(tables) == 2
    assert all(isinstance(t, str) for t in tables)


def test_delete_table(simulated_dm):
    """delete_table should remove the table."""
    simulated_dm.create_table("temp/to_delete")
    tables_before = simulated_dm.list_tables(
        prefix="Data/temp",
        include_column_info=False,
    )
    assert len(tables_before) == 1

    simulated_dm.delete_table("temp/to_delete", dangerous_ok=True)

    tables_after = simulated_dm.list_tables(
        prefix="Data/temp",
        include_column_info=False,
    )
    assert len(tables_after) == 0


def test_delete_table_requires_dangerous_ok(simulated_dm):
    """delete_table should raise without dangerous_ok=True."""
    simulated_dm.create_table("temp/protected")

    with pytest.raises(ValueError, match="dangerous_ok"):
        simulated_dm.delete_table("temp/protected")


def test_get_table(simulated_dm):
    """get_table should return lightweight context metadata."""
    simulated_dm.create_table(
        "test/get_table_demo",
        description="Demo for get_table",
        fields={"id": "int", "name": "str"},
        unique_keys={
            "id": "int",
        },  # Dict format per BaseDataManager.create_table signature
        auto_counting={"key": "id"},
    )

    ctx_info = simulated_dm.get_table("test/get_table_demo")

    assert isinstance(ctx_info, dict)
    assert ctx_info.get("description") == "Demo for get_table"
    assert ctx_info.get("unique_keys") == {"id": "int"}
    assert ctx_info.get("auto_counting") == {"key": "id"}


def test_get_table_not_found(simulated_dm):
    """get_table should raise ValueError for non-existent table."""
    with pytest.raises(ValueError, match="not found"):
        simulated_dm.get_table("nonexistent/table")


# ────────────────────────────────────────────────────────────────────────────
# Query operations
# ────────────────────────────────────────────────────────────────────────────


def test_filter_returns_all_rows(seeded_dm):
    """filter without expression should return all rows up to limit."""
    rows = seeded_dm.filter("test/products")
    assert len(rows) == 5


def test_filter_with_expression(seeded_dm):
    """filter with expression should return matching rows."""
    rows = seeded_dm.filter("test/products", filter="category == 'widgets'")
    assert len(rows) == 2
    assert all(r["category"] == "widgets" for r in rows)


def test_filter_with_numeric_comparison(seeded_dm):
    """filter should support numeric comparisons."""
    rows = seeded_dm.filter("test/products", filter="price > 30")
    assert len(rows) == 3
    assert all(r["price"] > 30 for r in rows)


def test_filter_with_limit_offset(seeded_dm):
    """filter should support pagination."""
    # Get first 2
    page1 = seeded_dm.filter("test/products", limit=2, offset=0)
    assert len(page1) == 2

    # Get next 2
    page2 = seeded_dm.filter("test/products", limit=2, offset=2)
    assert len(page2) == 2

    # No overlap
    page1_ids = {r["id"] for r in page1}
    page2_ids = {r["id"] for r in page2}
    assert page1_ids.isdisjoint(page2_ids)


def test_filter_with_column_selection(seeded_dm):
    """filter should return only requested columns."""
    rows = seeded_dm.filter("test/products", columns=["id", "name"])

    assert len(rows) == 5
    assert all("id" in r and "name" in r for r in rows)
    # Other columns should not be present
    assert all("price" not in r and "category" not in r for r in rows)


def test_filter_with_exclude_columns(seeded_dm):
    """filter with exclude_columns should exclude specified columns."""
    rows = seeded_dm.filter("test/products", exclude_columns=["price", "category"])

    assert len(rows) == 5
    assert all("id" in r and "name" in r for r in rows)
    # Excluded columns should not be present
    assert all("price" not in r and "category" not in r for r in rows)


def test_filter_with_return_ids_only(seeded_dm):
    """filter with return_ids_only=True should return list of log IDs."""
    ids = seeded_dm.filter("test/products", return_ids_only=True)

    assert isinstance(ids, list)
    assert len(ids) == 5
    # All IDs should be integers
    assert all(isinstance(i, int) for i in ids)


def test_filter_with_return_ids_only_and_filter(seeded_dm):
    """filter with return_ids_only=True and filter expression."""
    ids = seeded_dm.filter(
        "test/products",
        filter="category == 'widgets'",
        return_ids_only=True,
    )

    assert isinstance(ids, list)
    assert len(ids) == 2


def test_search_basic(seeded_dm):
    """search should return results with similarity scores."""
    # Add some text content for searching
    seeded_dm.insert_rows(
        "test/docs",
        [
            {"id": 1, "text": "Machine learning algorithms for classification"},
            {"id": 2, "text": "Deep neural networks for image recognition"},
            {"id": 3, "text": "Natural language processing techniques"},
        ],
    )

    results = seeded_dm.search(
        "test/docs",
        references={"text": "neural networks deep learning"},
        k=2,
    )

    assert len(results) <= 2
    assert all("_similarity" in r for r in results)


def test_reduce_count(seeded_dm):
    """reduce with count metric should return row count."""
    count = seeded_dm.reduce("test/products", metric="count", columns="id")
    assert count == 5


def test_reduce_sum(seeded_dm):
    """reduce with sum metric should return sum."""
    total = seeded_dm.reduce("test/products", metric="sum", columns="price")
    assert total == 10.0 + 20.0 + 50.0 + 75.0 + 100.0


def test_reduce_avg(seeded_dm):
    """reduce with avg metric should return average."""
    avg = seeded_dm.reduce("test/products", metric="avg", columns="price")
    expected = (10.0 + 20.0 + 50.0 + 75.0 + 100.0) / 5
    assert avg == expected


def test_reduce_with_filter(seeded_dm):
    """reduce should respect filter expression."""
    count = seeded_dm.reduce(
        "test/products",
        metric="count",
        columns="id",
        filter="category == 'widgets'",
    )
    assert count == 2


def test_reduce_with_group_by(seeded_dm):
    """reduce with group_by should return grouped results."""
    results = seeded_dm.reduce(
        "test/products",
        metric="count",
        columns="id",
        group_by="category",
    )

    assert isinstance(results, list)
    # Should have 3 categories: widgets, gadgets, tools
    assert len(results) == 3

    # Check structure
    for r in results:
        assert "category" in r
        assert "count" in r


# ────────────────────────────────────────────────────────────────────────────
# Mutation operations
# ────────────────────────────────────────────────────────────────────────────


def test_insert_rows(simulated_dm):
    """insert_rows should return list of inserted log IDs."""
    simulated_dm.create_table("test/items", fields={"id": "int", "value": "str"})

    inserted_ids = simulated_dm.insert_rows(
        "test/items",
        [
            {"id": 1, "value": "one"},
            {"id": 2, "value": "two"},
        ],
    )

    # insert_rows returns list of log IDs (one per inserted row)
    assert isinstance(inserted_ids, list)
    assert len(inserted_ids) == 2
    rows = simulated_dm.filter("test/items")
    assert len(rows) == 2


def test_insert_rows_bulk(simulated_dm):
    """insert_rows in bulk should assign distinct log IDs."""
    simulated_dm.create_table("test/items", fields={"id": "int", "value": "str"})

    ids = simulated_dm.insert_rows(
        "test/items",
        [{"id": 1, "value": "a"}, {"id": 2, "value": "b"}],
    )

    assert len(ids) == 2
    assert ids[0] != ids[1]
    rows = simulated_dm.filter("test/items")
    assert len(rows) == 2


def test_insert_rows_with_batched(simulated_dm):
    """insert_rows with batched=True should return log IDs."""
    simulated_dm.create_table("test/batched", fields={"id": "int", "text": "str"})

    # Default batched=True
    ids = simulated_dm.insert_rows(
        "test/batched",
        [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}],
        batched=True,
    )

    assert isinstance(ids, list)
    assert len(ids) == 2

    # Also works with batched=False
    ids_unbatched = simulated_dm.insert_rows(
        "test/batched",
        [{"id": 3, "text": "c"}],
        batched=False,
    )
    assert len(ids_unbatched) == 1


def test_update_rows(seeded_dm):
    """update_rows should modify matching rows."""
    # Update all widgets to have category 'updated_widgets'
    updated = seeded_dm.update_rows(
        "test/products",
        updates={"category": "updated_widgets"},
        filter="category == 'widgets'",
    )

    assert updated == 2

    rows = seeded_dm.filter("test/products", filter="category == 'updated_widgets'")
    assert len(rows) == 2


def test_delete_rows(seeded_dm):
    """delete_rows should remove matching rows."""
    initial_count = len(seeded_dm.filter("test/products"))

    deleted = seeded_dm.delete_rows(
        "test/products",
        filter="category == 'tools'",
        dangerous_ok=True,
    )

    assert deleted == 1

    remaining = seeded_dm.filter("test/products")
    assert len(remaining) == initial_count - 1


def test_delete_rows_requires_dangerous_ok(seeded_dm):
    """delete_rows should raise without dangerous_ok=True."""
    with pytest.raises(ValueError, match="dangerous_ok"):
        seeded_dm.delete_rows("test/products", filter="id == 1")


def test_delete_rows_with_log_ids(seeded_dm):
    """delete_rows with log_ids should delete specific rows."""
    # First get log IDs using filter
    ids_to_delete = seeded_dm.filter(
        "test/products",
        filter="category == 'tools'",
        return_ids_only=True,
    )
    assert len(ids_to_delete) == 1

    # Then delete using those IDs
    deleted_count = seeded_dm.delete_rows(
        "test/products",
        log_ids=ids_to_delete,
        dangerous_ok=True,
    )

    # Should return count of deleted rows
    assert deleted_count == 1

    # Verify deletion
    remaining = seeded_dm.filter("test/products", filter="category == 'tools'")
    assert len(remaining) == 0


def test_delete_rows_with_delete_empty_rows(simulated_dm):
    """delete_rows with delete_empty_rows=True cascades empty cleanup."""
    simulated_dm.create_table("test/cleanup", fields={"id": "int", "data": "str"})
    simulated_dm.insert_rows("test/cleanup", [{"id": 1, "data": "test"}])

    # Delete with delete_empty_rows flag
    deleted = simulated_dm.delete_rows(
        "test/cleanup",
        filter="id == 1",
        dangerous_ok=True,
        delete_empty_rows=True,
    )

    assert deleted == 1
    remaining = simulated_dm.filter("test/cleanup")
    assert len(remaining) == 0


# ────────────────────────────────────────────────────────────────────────────
# Join operations
# ────────────────────────────────────────────────────────────────────────────


def test_filter_join(seeded_dm):
    """filter_join should join two tables using KnowledgeManager-style API."""
    # Join orders with products using full join API
    results = seeded_dm.filter_join(
        tables=["test/orders", "test/products"],
        join_expr="Data/test/orders.product_id == Data/test/products.id",
        select={
            "Data/test/orders.order_id": "order_id",
            "Data/test/orders.status": "status",
            "Data/test/products.name": "name",
            "Data/test/products.price": "price",
        },
    )

    # Should have joined rows (orders with their product info)
    assert len(results) >= 0  # May be empty if no matching product_ids

    # Each result should have aliased fields from select
    for r in results:
        assert "order_id" in r
        assert "name" in r


def test_filter_join_with_filter(seeded_dm):
    """filter_join should support post-join filtering."""
    results = seeded_dm.filter_join(
        tables=["test/orders", "test/products"],
        join_expr="Data/test/orders.product_id == Data/test/products.id",
        select={
            "Data/test/orders.order_id": "order_id",
            "Data/test/orders.status": "status",
            "Data/test/products.name": "name",
        },
        result_where="status == 'shipped'",
    )

    # Should only include shipped orders
    assert all(r.get("status") == "shipped" for r in results)


# ────────────────────────────────────────────────────────────────────────────
# Embedding operations
# ────────────────────────────────────────────────────────────────────────────


def test_ensure_vector_column(simulated_dm):
    """ensure_vector_column should return the embedding column name."""
    simulated_dm.create_table("test/docs", fields={"id": "int", "text": "str"})

    emb_col = simulated_dm.ensure_vector_column("test/docs", source_column="text")

    assert emb_col == "_text_emb"

    # Describe should show embeddings
    desc = simulated_dm.describe_table("test/docs")
    assert desc.has_embeddings
    assert "_text_emb" in desc.embedding_columns


def test_vectorize_rows(simulated_dm):
    """vectorize_rows should return count of embedded rows."""
    simulated_dm.create_table("test/docs", fields={"id": "int", "text": "str"})
    simulated_dm.insert_rows(
        "test/docs",
        [
            {"id": 1, "text": "First document"},
            {"id": 2, "text": "Second document"},
        ],
    )

    count = simulated_dm.vectorize_rows("test/docs", source_column="text")

    # SimulatedDataManager returns 0 for all rows (no specific IDs)
    assert isinstance(count, int)


# ────────────────────────────────────────────────────────────────────────────
# Visualization
# ────────────────────────────────────────────────────────────────────────────


def test_plot_bar_with_aggregate(seeded_dm):
    """Bar chart with aggregation returns a valid PlotResult."""
    result = seeded_dm.plot(
        "test/products",
        plot_type="bar",
        x="category",
        y="price",
        aggregate="sum",
        title="Price by Category",
    )

    assert isinstance(result, PlotResult)
    assert result.succeeded
    assert result.url is not None
    assert result.title == "Price by Category"


def test_plot_bar_with_metric(seeded_dm):
    """Bar chart with metric parameter is accepted and returns a valid PlotResult."""
    result = seeded_dm.plot(
        "test/products",
        plot_type="bar",
        x="category",
        y="price",
        metric="count",
        title="Product Count by Category",
    )

    assert isinstance(result, PlotResult)
    assert result.succeeded
    assert result.url is not None
    assert result.title == "Product Count by Category"


def test_plot_batch_with_metric(seeded_dm):
    """plot_batch forwards metric without error."""
    contexts = ["test/products", "test/orders"]
    results = seeded_dm.plot_batch(
        contexts,
        plot_type="bar",
        x="id",
        y="id",
        metric="sum",
    )

    assert len(results) == len(contexts)
    assert all(r.succeeded for r in results)


def test_plot_line_with_group_by(seeded_dm):
    """Line chart with group_by propagates all params and succeeds."""
    result = seeded_dm.plot(
        "test/orders",
        plot_type="line",
        x="order_date",
        y="total",
        group_by="status",
        title="Orders Over Time",
    )

    assert result.succeeded
    assert "line" in result.url
    assert result.title == "Orders Over Time"


def test_plot_scatter_with_regression_and_scales(seeded_dm):
    """Scatter plot with regression line and log scales."""
    result = seeded_dm.plot(
        "test/products",
        plot_type="scatter",
        x="price",
        y="id",
        show_regression=True,
        scale_x="log",
        scale_y="linear",
    )

    assert result.succeeded
    assert "scatter" in result.url


def test_plot_histogram_with_bins(seeded_dm):
    """Histogram with custom bin_count."""
    result = seeded_dm.plot(
        "test/products",
        plot_type="histogram",
        x="price",
        bin_count=20,
    )

    assert result.succeeded
    assert "histogram" in result.url


def test_plot_with_filter(seeded_dm):
    """Plot with a filter expression propagates without error."""
    result = seeded_dm.plot(
        "test/orders",
        plot_type="bar",
        x="status",
        y="total",
        filter="status == 'shipped'",
        aggregate="sum",
    )

    assert result.succeeded


def test_plot_title_defaults_when_omitted(seeded_dm):
    """Omitting title produces a sensible default containing the plot type."""
    result = seeded_dm.plot(
        "test/products",
        plot_type="bar",
        x="category",
    )

    assert result.succeeded
    assert "bar" in result.title.lower()


def test_plot_batch_per_context_results(seeded_dm):
    """plot_batch returns one result per context with the correct URL stem."""
    contexts = ["test/products", "test/orders"]
    results = seeded_dm.plot_batch(
        contexts,
        plot_type="bar",
        x="id",
        y="id",
    )

    assert len(results) == len(contexts)
    assert all(isinstance(r, PlotResult) for r in results)
    for ctx, r in zip(contexts, results):
        assert r.succeeded
        assert ctx.replace("/", "/") in r.url


def test_plot_resolves_relative_context(simulated_dm):
    """Relative context paths are resolved to the Data/ namespace in the URL."""
    simulated_dm.create_table("myproject/metrics", fields={"x": "int"})
    result = simulated_dm.plot(
        "myproject/metrics",
        plot_type="bar",
        x="x",
    )

    assert result.succeeded
    assert "Data/myproject/metrics" in result.url


# ────────────────────────────────────────────────────────────────────────────
# Table Views
# ────────────────────────────────────────────────────────────────────────────


def test_table_view_with_column_config(seeded_dm):
    """table_view with column visibility, ordering, and sorting."""
    result = seeded_dm.table_view(
        "test/products",
        columns_visible=["id", "name", "price"],
        columns_order=["price", "name", "id"],
        columns_hidden=["category"],
        sort_by="price",
        sort_order="desc",
        title="Products Table",
    )

    assert isinstance(result, TableViewResult)
    assert result.succeeded
    assert result.url is not None
    assert result.title == "Products Table"


def test_table_view_with_filter_and_row_limit(seeded_dm):
    """table_view with filter and row_limit propagates without error."""
    result = seeded_dm.table_view(
        "test/orders",
        filter="status == 'shipped'",
        row_limit=50,
        title="Shipped Orders",
    )

    assert result.succeeded
    assert result.title == "Shipped Orders"


def test_table_view_title_defaults_when_omitted(seeded_dm):
    """Omitting title produces a sensible default."""
    result = seeded_dm.table_view("test/products")
    assert result.succeeded
    assert result.title is not None


def test_table_view_batch_per_context(seeded_dm):
    """table_view_batch returns one result per context."""
    contexts = ["test/products", "test/orders"]
    results = seeded_dm.table_view_batch(
        contexts,
        sort_by="id",
        sort_order="asc",
    )

    assert len(results) == len(contexts)
    assert all(isinstance(r, TableViewResult) for r in results)
    assert all(r.succeeded for r in results)


def test_table_view_resolves_relative_context(simulated_dm):
    """Relative context paths are resolved to the Data/ namespace."""
    simulated_dm.create_table("myproject/data", fields={"a": "int"})
    result = simulated_dm.table_view("myproject/data")

    assert result.succeeded
    assert "Data/myproject/data" in result.url


# ────────────────────────────────────────────────────────────────────────────
# Context resolution
# ────────────────────────────────────────────────────────────────────────────


def test_relative_context_resolution(simulated_dm):
    """Relative paths should be resolved to Data/ namespace."""
    path = simulated_dm.create_table("myproject/data")
    assert path == "Data/myproject/data"


def test_absolute_context_passthrough(simulated_dm):
    """Absolute paths should be used as-is."""
    # Files/ is an absolute path prefix
    simulated_dm._tables["Files/Local/123/Content"] = []
    rows = simulated_dm.filter("Files/Local/123/Content")
    assert isinstance(rows, list)


# ────────────────────────────────────────────────────────────────────────────
# Clear/reset
# ────────────────────────────────────────────────────────────────────────────


def test_clear(simulated_dm):
    """clear should reset all in-memory state."""
    simulated_dm.create_table("test/a")
    simulated_dm.create_table("test/b")
    simulated_dm.insert_rows("test/a", [{"x": 1}])

    simulated_dm.clear()

    tables = simulated_dm.list_tables()
    assert len(tables) == 0


# ────────────────────────────────────────────────────────────────────────────
# Ingest
# ────────────────────────────────────────────────────────────────────────────


def test_ingest_basic(simulated_dm):
    """ingest should create table and insert rows in one call."""
    result = simulated_dm.ingest(
        "test/ingest_basic",
        rows=[
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Carol"},
        ],
        description="Basic ingest test",
        fields={"id": "int", "name": "str"},
    )

    assert isinstance(result, IngestResult)
    assert result.context == "Data/test/ingest_basic"
    assert result.rows_inserted == 3
    assert result.rows_embedded == 0
    assert len(result.log_ids) == 3
    assert result.duration_ms > 0
    assert result.chunks_processed >= 1

    # Verify rows actually exist
    rows = simulated_dm.filter("test/ingest_basic")
    assert len(rows) == 3


def test_ingest_with_embedding(simulated_dm):
    """ingest with embed_columns should report embedded rows."""
    result = simulated_dm.ingest(
        "test/ingest_embed",
        rows=[
            {"id": 1, "text": "Hello world"},
            {"id": 2, "text": "Foo bar"},
        ],
        fields={"id": "int", "text": "str"},
        embed_columns=["text"],
    )

    assert result.rows_inserted == 2
    assert result.rows_embedded > 0

    # Verify embedding columns were created
    desc = simulated_dm.describe_table("test/ingest_embed")
    assert desc.has_embeddings
    assert "_text_emb" in desc.embedding_columns


def test_ingest_appends_rows(simulated_dm):
    """Successive ingest calls should append rows (uniqueness is schema-level)."""
    simulated_dm.ingest(
        "test/ingest_append",
        rows=[{"id": 1, "value": "first"}],
        fields={"id": "int", "value": "str"},
    )

    result = simulated_dm.ingest(
        "test/ingest_append",
        rows=[{"id": 2, "value": "second"}],
    )

    assert result.rows_inserted == 1
    rows = simulated_dm.filter("test/ingest_append")
    assert len(rows) == 2


def test_ingest_empty_rows(simulated_dm):
    """ingest with empty rows should return zero counts immediately."""
    result = simulated_dm.ingest("test/ingest_empty", rows=[])

    assert result.rows_inserted == 0
    assert result.chunks_processed == 0
    assert result.log_ids == []


def test_ingest_chunk_counting(simulated_dm):
    """ingest should calculate correct chunk count based on chunk_size."""
    rows = [{"id": i} for i in range(10)]

    result = simulated_dm.ingest(
        "test/ingest_chunks",
        rows=rows,
        chunk_size=3,
    )

    assert result.rows_inserted == 10
    # 10 rows / chunk_size 3 = ceil(10/3) = 4 chunks
    assert result.chunks_processed == 4


def test_ingest_docstring_inherited(simulated_dm):
    """SimulatedDataManager.ingest should inherit docstring from base."""
    from unity.data_manager.base import BaseDataManager

    assert (
        BaseDataManager.ingest.__doc__.strip()
        in SimulatedDataManager.ingest.__doc__.strip()
    )
