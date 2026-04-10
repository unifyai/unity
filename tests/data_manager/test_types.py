"""
Tests for DataManager types (Pydantic models).
"""

from __future__ import annotations


from unity.data_manager.types import (
    TableDescription,
    TableSchema,
    ColumnInfo,
    IngestExecutionConfig,
    IngestResult,
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
# IngestExecutionConfig
# ────────────────────────────────────────────────────────────────────────────


def test_ingest_execution_config_defaults():
    """IngestExecutionConfig should have sensible defaults."""
    cfg = IngestExecutionConfig()

    assert cfg.max_workers == 4
    assert cfg.max_retries == 3
    assert cfg.retry_delay_seconds == 3.0
    assert cfg.fail_fast is False


def test_ingest_execution_config_custom():
    """IngestExecutionConfig should accept custom values."""
    cfg = IngestExecutionConfig(
        max_workers=8,
        max_retries=5,
        retry_delay_seconds=1.0,
        fail_fast=True,
    )

    assert cfg.max_workers == 8
    assert cfg.max_retries == 5
    assert cfg.retry_delay_seconds == 1.0
    assert cfg.fail_fast is True


def test_ingest_execution_config_validation():
    """IngestExecutionConfig should enforce constraints."""
    import pytest

    with pytest.raises(Exception):
        IngestExecutionConfig(max_workers=0)  # ge=1

    with pytest.raises(Exception):
        IngestExecutionConfig(max_retries=-1)  # ge=0

    with pytest.raises(Exception):
        IngestExecutionConfig(retry_delay_seconds=-0.5)  # ge=0.0


# ────────────────────────────────────────────────────────────────────────────
# IngestResult
# ────────────────────────────────────────────────────────────────────────────


def test_ingest_result_defaults():
    """IngestResult should have zero-value defaults except context."""
    result = IngestResult(context="Data/test")

    assert result.context == "Data/test"
    assert result.rows_inserted == 0
    assert result.rows_embedded == 0
    assert result.log_ids == []
    assert result.duration_ms == 0.0
    assert result.chunks_processed == 0


def test_ingest_result_full():
    """IngestResult should store all fields."""
    result = IngestResult(
        context="Data/MidlandHeart/Repairs",
        rows_inserted=5000,
        rows_embedded=5000,
        log_ids=[1, 2, 3],
        duration_ms=1234.5,
        chunks_processed=5,
    )

    assert result.rows_inserted == 5000
    assert result.rows_embedded == 5000
    assert len(result.log_ids) == 3
    assert result.chunks_processed == 5
