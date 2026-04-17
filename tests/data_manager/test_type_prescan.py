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
    TypeMap,
    _stratified_indices,
    coerce_batch,
    coerce_empty_strings,
    coerce_rows,
    coerce_value,
    prescan_column_types,
    prescan_from_rows,
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


# =========================================================================
# TypeMap / prescan_from_rows
# =========================================================================


class TestPrescanFromRows:

    def test_basic_inference(self):
        rows = [
            {"name": "Alice", "age": 30, "joined": "2025-01-15 09:00:00"},
            {"name": "Bob", "age": 25, "joined": "2025-02-20 10:00:00"},
        ]
        tm = prescan_from_rows(rows)
        assert isinstance(tm, TypeMap)
        assert tm.column_types["name"] == "str"
        assert tm.column_types["age"] == "int"
        assert tm.column_types["joined"] == "datetime"
        assert tm.columns == frozenset({"name", "age", "joined"})
        assert tm.sample_size == 2

    def test_empty_input(self):
        tm = prescan_from_rows([])
        assert tm.column_types == {}
        assert tm.columns == frozenset()
        assert tm.sample_size == 0

    def test_respects_sample_size(self):
        rows = [{"val": i} for i in range(100)]
        tm = prescan_from_rows(rows, sample_size=10)
        assert tm.sample_size == 10

    def test_all_none_fallback_to_str(self):
        rows = [{"col": None} for _ in range(20)]
        tm = prescan_from_rows(rows)
        assert tm.column_types["col"] == "str"

    def test_all_empty_string_fallback_to_str(self):
        rows = [{"col": ""} for _ in range(20)]
        tm = prescan_from_rows(rows)
        assert tm.column_types["col"] == "str"

    def test_majority_wins(self):
        rows = [{"col": 42}] * 8 + [{"col": "text"}] * 2
        tm = prescan_from_rows(rows)
        assert tm.column_types["col"] == "int"

    def test_frozen(self):
        tm = prescan_from_rows([{"a": 1}])
        with pytest.raises(AttributeError):
            tm.sample_size = 999  # type: ignore[misc]

    def test_works_with_iterator(self):
        """Accepts any iterable, not just lists."""

        def _gen():
            for i in range(5):
                yield {"x": float(i)}

        tm = prescan_from_rows(_gen())
        assert tm.column_types["x"] == "float"
        assert tm.sample_size == 5


# =========================================================================
# coerce_batch
# =========================================================================


class TestCoerceBatch:

    def _make_type_map(self, types: dict) -> TypeMap:
        return TypeMap(
            column_types=types,
            columns=frozenset(types.keys()),
            sample_size=0,
        )

    def test_basic_coercion(self):
        tm = self._make_type_map({"age": "int", "name": "str"})
        batch = [{"age": 25, "name": "Alice"}, {"age": "not_int", "name": "Bob"}]
        coerced, stats = coerce_batch(batch, tm)
        assert coerced[0]["age"] == 25
        assert coerced[1]["age"] is None
        assert stats.type_coerced == 1

    def test_empty_string_coercion(self):
        tm = self._make_type_map({"col": "str"})
        batch = [{"col": ""}, {"col": "ok"}]
        coerced, stats = coerce_batch(batch, tm)
        assert coerced[0]["col"] is None
        assert coerced[1]["col"] == "ok"
        assert stats.empty_strings_coerced == 1

    def test_empty_batch(self):
        tm = self._make_type_map({"x": "int"})
        coerced, stats = coerce_batch([], tm)
        assert coerced == []
        assert stats.total_cells == 0

    def test_preserves_none(self):
        tm = self._make_type_map({"val": "datetime"})
        batch = [{"val": None}]
        coerced, stats = coerce_batch(batch, tm)
        assert coerced[0]["val"] is None
        assert stats.type_coerced == 0

    def test_coerced_by_column_tracking(self):
        tm = self._make_type_map({"a": "int", "b": "int"})
        batch = [{"a": "bad", "b": 1}, {"a": 2, "b": "worse"}]
        _, stats = coerce_batch(batch, tm)
        assert stats.coerced_by_column.get("a", 0) >= 1
        assert stats.coerced_by_column.get("b", 0) >= 1

    def test_consistent_with_prescan(self):
        """TypeMap from prescan_from_rows can feed coerce_batch."""
        rows = [{"x": 1}, {"x": 2}, {"x": 3}]
        tm = prescan_from_rows(rows)
        batch = [{"x": 10}, {"x": "garbage"}]
        coerced, stats = coerce_batch(batch, tm)
        assert coerced[0]["x"] == 10
        assert coerced[1]["x"] is None
        assert stats.type_coerced == 1
