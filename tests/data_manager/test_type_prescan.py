"""Tests for the pre-scan type inference and coercion module.

Covers:
- ``prescan_column_types``: stratified sampling, majority voting, edge cases
- ``coerce_value``: per-target-type validation
- ``coerce_rows``: full-row coercion with stats
- ``coerce_empty_strings``: universal empty-string → None rule
"""

from __future__ import annotations

import pytest

from unity.data_manager.ops.type_prescan import (
    _stratified_indices,
    coerce_empty_strings,
    coerce_rows,
    coerce_value,
    prescan_column_types,
)

# =========================================================================
# _stratified_indices
# =========================================================================


class TestStratifiedIndices:

    def test_small_dataset(self):
        """When n <= k, return all indices."""
        assert _stratified_indices(10, 500) == list(range(10))

    def test_large_dataset_bounded(self):
        indices = _stratified_indices(10_000, 500)
        assert len(indices) <= 500
        assert indices[0] == 0
        assert indices[-1] >= 9_500

    def test_includes_head_and_tail(self):
        indices = _stratified_indices(1_000, 100)
        assert 0 in indices
        assert 999 in indices


# =========================================================================
# prescan_column_types
# =========================================================================


class TestPrescanColumnTypes:

    def test_all_datetime(self):
        rows = [{"dt": f"2025-01-{i+1:02d} 12:00:00"} for i in range(50)]
        result = prescan_column_types(rows)
        assert result["dt"] == "datetime"

    def test_majority_datetime_with_garbage(self):
        """90% datetime + 10% garbage → datetime wins by majority."""
        rows = [{"col": f"2025-01-{(i % 28)+1:02d} 12:00:00"} for i in range(90)]
        rows += [{"col": "GARBAGE"} for _ in range(10)]
        result = prescan_column_types(rows)
        assert result["col"] == "datetime"

    def test_all_empty_strings_fallback(self):
        rows = [{"col": ""} for _ in range(50)]
        result = prescan_column_types(rows)
        assert result["col"] == "str"

    def test_all_none_fallback(self):
        rows = [{"col": None} for _ in range(50)]
        result = prescan_column_types(rows)
        assert result["col"] == "str"

    def test_mixed_types_tie_fallback(self):
        """Equal counts of different types → fallback to str."""
        rows = [{"col": "2025-01-01 12:00:00"}] * 25 + [{"col": 42}] * 25
        result = prescan_column_types(rows)
        assert result["col"] in ("datetime", "int", "str")

    def test_small_dataset_under_sample_size(self):
        rows = [{"a": 1, "b": "hello"} for _ in range(5)]
        result = prescan_column_types(rows)
        assert result["a"] == "int"
        assert result["b"] == "str"

    def test_empty_rows(self):
        assert prescan_column_types([]) == {}

    def test_multiple_columns(self):
        rows = [
            {"name": "Alice", "age": 30, "joined": "2025-01-15 09:00:00"}
            for _ in range(20)
        ]
        result = prescan_column_types(rows)
        assert result["name"] == "str"
        assert result["age"] == "int"
        assert result["joined"] == "datetime"


# =========================================================================
# coerce_value
# =========================================================================


class TestCoerceValue:

    def test_none_passthrough(self):
        assert coerce_value(None, "datetime") is None

    @pytest.mark.parametrize(
        "value, target, expected_is_none",
        [
            ("2025-01-01 12:00:00", "datetime", False),
            ("garbage", "datetime", True),
            ("14:30:00", "time", False),
            ("not-a-time", "time", True),
            ("2025-01-01", "date", False),
            ("not-a-date", "date", True),
            ("P1D", "timedelta", False),
            ("not-a-duration", "timedelta", True),
            (42, "int", False),
            ("not_a_number", "int", True),
            (True, "int", True),
            (3.14, "float", False),
            (42, "float", False),
            ("not_a_float", "float", True),
            (True, "bool", False),
            (42, "bool", True),
            ("anything", "str", False),
        ],
        ids=[
            "dt-valid",
            "dt-invalid",
            "time-valid",
            "time-invalid",
            "date-valid",
            "date-invalid",
            "td-valid",
            "td-invalid",
            "int-valid",
            "int-invalid",
            "bool-not-int",
            "float-valid",
            "int-as-float",
            "float-invalid",
            "bool-valid",
            "int-not-bool",
            "str-always-valid",
        ],
    )
    def test_per_type(self, value, target, expected_is_none):
        result = coerce_value(value, target)
        if expected_is_none:
            assert result is None
        else:
            assert result is not None

    def test_unknown_type_passthrough(self):
        """Unknown target types pass through unchanged."""
        assert coerce_value("whatever", "SomeCustomType") == "whatever"


# =========================================================================
# coerce_rows
# =========================================================================


class TestCoerceRows:

    def test_empty_strings_coerced(self):
        rows = [{"a": "", "b": "hello"}]
        coerced, stats = coerce_rows(rows, {"a": "str", "b": "str"})
        assert coerced[0]["a"] is None
        assert coerced[0]["b"] == "hello"
        assert stats.empty_strings_coerced == 1

    def test_type_mismatch_coerced(self):
        rows = [{"dt": "garbage", "dt2": "2025-01-01 12:00:00"}]
        coerced, stats = coerce_rows(rows, {"dt": "datetime", "dt2": "datetime"})
        assert coerced[0]["dt"] is None
        assert coerced[0]["dt2"] == "2025-01-01 12:00:00"
        assert stats.type_coerced == 1

    def test_all_valid_zero_coercions(self):
        rows = [{"x": 1, "y": 2}]
        _, stats = coerce_rows(rows, {"x": "int", "y": "int"})
        assert stats.empty_strings_coerced == 0
        assert stats.type_coerced == 0
        assert stats.total_cells == 2

    def test_stats_counts(self):
        rows = [
            {"a": "", "b": "garbage", "c": "2025-01-01 12:00:00"},
            {"a": "2025-01-01 12:00:00", "b": "", "c": "also garbage"},
        ]
        types = {"a": "datetime", "b": "datetime", "c": "datetime"}
        _, stats = coerce_rows(rows, types)
        assert stats.total_cells == 6
        assert stats.empty_strings_coerced == 2
        assert stats.type_coerced == 2
        assert "a" in stats.coerced_by_column
        assert "b" in stats.coerced_by_column
        assert "c" in stats.coerced_by_column

    def test_preserves_row_structure(self):
        rows = [{"a": 1, "b": "hello", "c": ""}]
        coerced, _ = coerce_rows(rows, {"a": "int", "b": "str", "c": "str"})
        assert set(coerced[0].keys()) == {"a", "b", "c"}
        assert coerced[0]["a"] == 1
        assert coerced[0]["b"] == "hello"
        assert coerced[0]["c"] is None

    def test_none_values_untouched(self):
        rows = [{"a": None}]
        coerced, stats = coerce_rows(rows, {"a": "datetime"})
        assert coerced[0]["a"] is None
        assert stats.type_coerced == 0
        assert stats.empty_strings_coerced == 0


# =========================================================================
# coerce_empty_strings
# =========================================================================


class TestCoerceEmptyStrings:

    def test_basic(self):
        rows = [{"a": "", "b": "hello", "c": ""}]
        coerced, count = coerce_empty_strings(rows)
        assert coerced[0]["a"] is None
        assert coerced[0]["b"] == "hello"
        assert coerced[0]["c"] is None
        assert count == 2

    def test_no_empty_strings(self):
        rows = [{"a": 1, "b": "hello"}]
        _, count = coerce_empty_strings(rows)
        assert count == 0
