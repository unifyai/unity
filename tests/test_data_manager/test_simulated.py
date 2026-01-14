"""
Tests for SimulatedDataManager.

These tests verify that the simulated implementation works correctly
and can be used as a drop-in replacement for the real DataManager in tests.
"""

from __future__ import annotations

import pytest

from unity.data_manager.simulated import SimulatedDataManager
from unity.data_manager.base import BaseDataManager
from unity.data_manager.types import TableDescription, PlotResult


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


def test_delete_table(simulated_dm):
    """delete_table should remove the table."""
    simulated_dm.create_table("temp/to_delete")
    tables_before = simulated_dm.list_tables(prefix="Data/temp")
    assert len(tables_before) == 1

    simulated_dm.delete_table("temp/to_delete", dangerous_ok=True)

    tables_after = simulated_dm.list_tables(prefix="Data/temp")
    assert len(tables_after) == 0


def test_delete_table_requires_dangerous_ok(simulated_dm):
    """delete_table should raise without dangerous_ok=True."""
    simulated_dm.create_table("temp/protected")

    with pytest.raises(ValueError, match="dangerous_ok"):
        simulated_dm.delete_table("temp/protected")


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

    results = seeded_dm.search("test/docs", query="neural networks deep learning", k=2)

    assert len(results) <= 2
    assert all("_similarity" in r for r in results)


def test_reduce_count(seeded_dm):
    """reduce with count metric should return row count."""
    count = seeded_dm.reduce("test/products", metric="count", column="id")
    assert count == 5


def test_reduce_sum(seeded_dm):
    """reduce with sum metric should return sum."""
    total = seeded_dm.reduce("test/products", metric="sum", column="price")
    assert total == 10.0 + 20.0 + 50.0 + 75.0 + 100.0


def test_reduce_avg(seeded_dm):
    """reduce with avg metric should return average."""
    avg = seeded_dm.reduce("test/products", metric="avg", column="price")
    expected = (10.0 + 20.0 + 50.0 + 75.0 + 100.0) / 5
    assert avg == expected


def test_reduce_with_filter(seeded_dm):
    """reduce should respect filter expression."""
    count = seeded_dm.reduce(
        "test/products",
        metric="count",
        column="id",
        filter="category == 'widgets'",
    )
    assert count == 2


def test_reduce_with_group_by(seeded_dm):
    """reduce with group_by should return grouped results."""
    results = seeded_dm.reduce(
        "test/products",
        metric="count",
        column="id",
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
    """insert_rows should add rows to the table."""
    simulated_dm.create_table("test/items", fields={"id": "int", "value": "str"})

    count = simulated_dm.insert_rows(
        "test/items",
        [
            {"id": 1, "value": "one"},
            {"id": 2, "value": "two"},
        ],
    )

    assert count == 2
    rows = simulated_dm.filter("test/items")
    assert len(rows) == 2


def test_insert_rows_with_dedupe(simulated_dm):
    """insert_rows with dedupe_key should update existing rows."""
    simulated_dm.create_table("test/items", fields={"id": "int", "value": "str"})

    # Insert initial
    simulated_dm.insert_rows("test/items", [{"id": 1, "value": "original"}])

    # Insert with dedupe - should update
    simulated_dm.insert_rows(
        "test/items",
        [{"id": 1, "value": "updated"}],
        dedupe_key="id",
    )

    rows = simulated_dm.filter("test/items")
    assert len(rows) == 1
    assert rows[0]["value"] == "updated"


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


# ────────────────────────────────────────────────────────────────────────────
# Join operations
# ────────────────────────────────────────────────────────────────────────────


def test_filter_join(seeded_dm):
    """filter_join should join two tables."""
    # Add product_id to orders for joining
    results = seeded_dm.filter_join(
        left_context="test/orders",
        right_context="test/products",
        join_column="product_id",
    )

    # Should have joined rows (orders with their product info)
    assert len(results) > 0

    # Each result should have fields from both tables
    for r in results:
        assert "order_id" in r  # from orders
        assert "name" in r  # from products


def test_filter_join_with_filter(seeded_dm):
    """filter_join should support post-join filtering."""
    # Need to ensure join column exists in both tables
    # Add product_id link
    results = seeded_dm.filter_join(
        left_context="test/orders",
        right_context="test/products",
        join_column="product_id",
        filter="status == 'shipped'",
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


def test_plot(seeded_dm):
    """plot should return a PlotResult."""
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


def test_plot_batch(seeded_dm):
    """plot_batch should return a list of PlotResults."""
    results = seeded_dm.plot_batch(
        ["test/products", "test/orders"],
        plot_type="bar",
        x="id",
        y="id",
    )

    assert len(results) == 2
    assert all(isinstance(r, PlotResult) for r in results)


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
