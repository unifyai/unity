"""
Tests for DataManager types (Pydantic models).
"""

from __future__ import annotations


from unity.data_manager.types import (
    TableDescription,
    TableSchema,
    ColumnInfo,
    PlotConfig,
    PlotResult,
    PlotType,
)


# ────────────────────────────────────────────────────────────────────────────
# ColumnInfo
# ────────────────────────────────────────────────────────────────────────────


def test_column_info_basic():
    """ColumnInfo should store column metadata."""
    col = ColumnInfo(name="user_id", dtype="int")

    assert col.name == "user_id"
    assert col.dtype == "int"
    assert col.description is None


def test_column_info_with_description():
    """ColumnInfo should accept optional description."""
    col = ColumnInfo(
        name="email",
        dtype="str",
        description="User email address",
    )

    assert col.description == "User email address"


# ────────────────────────────────────────────────────────────────────────────
# TableSchema
# ────────────────────────────────────────────────────────────────────────────


def test_table_schema_empty():
    """TableSchema should work with no columns."""
    schema = TableSchema()

    assert schema.columns == []
    assert schema.column_names == []
    assert schema.column_types == {}


def test_table_schema_with_columns():
    """TableSchema should provide convenience accessors."""
    schema = TableSchema(
        columns=[
            ColumnInfo(name="id", dtype="int"),
            ColumnInfo(name="name", dtype="str"),
            ColumnInfo(name="active", dtype="bool"),
        ],
    )

    assert schema.column_names == ["id", "name", "active"]
    assert schema.column_types == {"id": "int", "name": "str", "active": "bool"}


def test_table_schema_with_constraints():
    """TableSchema should accept unique_keys and auto_counting."""
    schema = TableSchema(
        columns=[ColumnInfo(name="id", dtype="int")],
        unique_keys={"id": "int"},
        auto_counting={"id": None},
    )

    assert schema.unique_keys == {"id": "int"}
    assert schema.auto_counting == {"id": None}


# ────────────────────────────────────────────────────────────────────────────
# TableDescription
# ────────────────────────────────────────────────────────────────────────────


def test_table_description_basic():
    """TableDescription should store complete table metadata."""
    desc = TableDescription(
        context="Data/MH/arrears",
        description="Tenant arrears data",
        table_schema=TableSchema(
            columns=[
                ColumnInfo(name="tenant_id", dtype="int"),
                ColumnInfo(name="amount", dtype="float"),
            ],
        ),
    )

    assert desc.context == "Data/MH/arrears"
    assert desc.description == "Tenant arrears data"
    assert len(desc.table_schema.columns) == 2


def test_table_description_embeddings():
    """TableDescription should track embedding columns."""
    desc = TableDescription(
        context="Data/docs",
        has_embeddings=True,
        embedding_columns=["_text_emb", "_title_emb"],
    )

    assert desc.has_embeddings is True
    assert "_text_emb" in desc.embedding_columns


def test_table_description_convenience_accessors():
    """TableDescription should provide backward-compatible accessors."""
    desc = TableDescription(
        context="Data/test",
        table_schema=TableSchema(
            columns=[
                ColumnInfo(name="a", dtype="int"),
                ColumnInfo(name="b", dtype="str"),
            ],
            unique_keys={"a": "int"},
        ),
    )

    # Convenience accessors
    assert desc.columns == {"a": "int", "b": "str"}
    assert desc.unique_keys == {"a": "int"}


# ────────────────────────────────────────────────────────────────────────────
# PlotType
# ────────────────────────────────────────────────────────────────────────────


def test_plot_type_enum():
    """PlotType should have expected values."""
    assert PlotType.SCATTER == "scatter"
    assert PlotType.BAR == "bar"
    assert PlotType.HISTOGRAM == "histogram"
    assert PlotType.LINE == "line"


# ────────────────────────────────────────────────────────────────────────────
# PlotConfig
# ────────────────────────────────────────────────────────────────────────────


def test_plot_config_minimal():
    """PlotConfig should work with minimal required fields."""
    config = PlotConfig(plot_type="bar", x_axis="category")

    assert config.plot_type == "bar"
    assert config.x_axis == "category"
    assert config.y_axis is None


def test_plot_config_full():
    """PlotConfig should accept all configuration options."""
    config = PlotConfig(
        plot_type="scatter",
        x_axis="experience",
        y_axis="salary",
        group_by="department",
        aggregate="avg",
        scale_x="log",
        scale_y="linear",
        show_regression=True,
        title="Salary vs Experience",
    )

    assert config.plot_type == "scatter"
    assert config.y_axis == "salary"
    assert config.group_by == "department"
    assert config.show_regression is True


def test_plot_config_histogram():
    """PlotConfig for histogram should accept bin_count."""
    config = PlotConfig(
        plot_type="histogram",
        x_axis="price",
        bin_count=20,
    )

    assert config.plot_type == "histogram"
    assert config.bin_count == 20


# ────────────────────────────────────────────────────────────────────────────
# PlotResult
# ────────────────────────────────────────────────────────────────────────────


def test_plot_result_success():
    """PlotResult should indicate success with URL."""
    result = PlotResult(
        url="https://plots.example.com/abc123",
        token="secret-token",
        expires_in_hours=24,
        title="Revenue Chart",
        context="Data/sales",
    )

    assert result.succeeded is True
    assert result.url == "https://plots.example.com/abc123"
    assert result.error is None


def test_plot_result_failure():
    """PlotResult should indicate failure with error."""
    result = PlotResult(
        error="Invalid column: nonexistent",
        traceback_str="Traceback...",
    )

    assert result.succeeded is False
    assert result.url is None
    assert result.error == "Invalid column: nonexistent"


def test_plot_result_traceback_alias():
    """PlotResult should accept traceback_str via 'traceback' alias."""
    result = PlotResult(
        error="Something went wrong",
        traceback_str="Full traceback here",
    )

    assert result.traceback_str == "Full traceback here"
