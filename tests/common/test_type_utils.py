"""Tests for the vendored type utilities in unity.common.type_utils.

Covers:
- ``infer_type_from_value``: content-based type inference for all categories
- ``types_match``: structural compatibility with NoneType-as-weak, Optional, normalization
- ``normalize_type_string``: casing, whitespace, Optional desugaring
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pytest

from unity.common.type_utils import (
    infer_type_from_value,
    normalize_type_string,
    types_match,
)

# =========================================================================
# infer_type_from_value
# =========================================================================


class TestInferTypeFromValue:

    @pytest.mark.parametrize(
        "value, expected",
        [
            (None, "NoneType"),
            (True, "bool"),
            (False, "bool"),
            (42, "int"),
            (0, "int"),
            (-7, "int"),
            (3.14, "float"),
            (0.0, "float"),
        ],
        ids=["none", "true", "false", "int", "zero", "neg-int", "float", "zero-float"],
    )
    def test_primitives(self, value, expected):
        assert infer_type_from_value(value) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            (datetime(2025, 1, 1, 12, 0), "datetime"),
            (date(2025, 1, 1), "date"),
            (time(14, 30, 0), "time"),
            (timedelta(days=1), "timedelta"),
        ],
        ids=["datetime-obj", "date-obj", "time-obj", "timedelta-obj"],
    )
    def test_temporal_objects(self, value, expected):
        assert infer_type_from_value(value) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("2025-08-02 17:11:00", "datetime"),
            ("2025-08-02T17:11:00", "datetime"),
            ("2025-01-01", "date"),
            ("01/31/2023", "date"),
            ("00:09:00", "time"),
            ("14:30", "time"),
            ("2:30 PM", "time"),
            ("P1D", "timedelta"),
            ("PT1H30M", "timedelta"),
            ("hello", "str"),
            ("some random text", "str"),
            ("", "str"),
        ],
        ids=[
            "datetime-space",
            "datetime-iso",
            "date-iso",
            "date-us",
            "time-hms",
            "time-hm",
            "time-ampm",
            "timedelta-iso-day",
            "timedelta-iso-hm",
            "str-plain",
            "str-random",
            "str-empty",
        ],
    )
    def test_strings(self, value, expected):
        assert infer_type_from_value(value) == expected

    def test_list_homogeneous(self):
        assert infer_type_from_value([1, 2, 3]) == "List[int]"

    def test_list_heterogeneous(self):
        result = infer_type_from_value([1, "hello"])
        assert "List[" in result
        assert "int" in result
        assert "str" in result

    def test_list_empty(self):
        assert infer_type_from_value([]) == "List[Any]"

    def test_dict_homogeneous(self):
        assert infer_type_from_value({"a": 1}) == "Dict[str, int]"

    def test_dict_empty(self):
        assert infer_type_from_value({}) == "Dict[Any, Any]"

    def test_tuple_homogeneous(self):
        assert infer_type_from_value((1, 2, 3)) == "Tuple[int, ...]"

    def test_tuple_heterogeneous(self):
        result = infer_type_from_value((1, "hello"))
        assert "Tuple[" in result
        assert "int" in result
        assert "str" in result

    def test_bool_before_int(self):
        """bool is a subclass of int; ensure bool is detected first."""
        assert infer_type_from_value(True) == "bool"
        assert infer_type_from_value(1) == "int"


# =========================================================================
# types_match
# =========================================================================


class TestTypesMatch:

    @pytest.mark.parametrize(
        "field_type, inferred_type, expected",
        [
            ("datetime", "datetime", True),
            ("int", "int", True),
            ("str", "str", True),
            ("datetime", "str", False),
            ("int", "str", False),
            ("float", "int", False),
            # NoneType is weak — matches anything
            ("datetime", "NoneType", True),
            ("NoneType", "int", True),
            ("NoneType", "NoneType", True),
            # Any matches everything
            ("Any", "int", True),
            ("Any", "str", True),
            ("int", "Any", True),
            # enum matches str
            ("enum", "str", True),
            ("enum", "int", False),
            # Normalization
            ("DateTime", "datetime", True),
            ("INT", "int", True),
            # Container family matching
            ("list", "List[int]", True),
            ("List[int]", "List[int]", True),
            ("List[int]", "List[str]", False),
            ("dict", "Dict[str, int]", True),
            ("Dict[str, int]", "Dict[str, int]", True),
        ],
        ids=[
            "exact-datetime",
            "exact-int",
            "exact-str",
            "mismatch-datetime-str",
            "mismatch-int-str",
            "mismatch-float-int",
            "nonetype-weak-field",
            "nonetype-weak-inferred",
            "nonetype-both",
            "any-field",
            "any-field-str",
            "any-inferred",
            "enum-str",
            "enum-int",
            "norm-datetime",
            "norm-int",
            "bare-list",
            "list-exact",
            "list-mismatch",
            "bare-dict",
            "dict-exact",
        ],
    )
    def test_basic_pairs(self, field_type, inferred_type, expected):
        assert types_match(field_type, inferred_type) is expected

    def test_union_with_nonetype(self):
        """Union[datetime, NoneType] should match datetime."""
        assert types_match("Union[datetime, NoneType]", "datetime") is True

    def test_union_with_nonetype_matches_none(self):
        assert types_match("Union[datetime, NoneType]", "NoneType") is True

    def test_union_mismatch(self):
        """Union[datetime, NoneType] should not match str."""
        assert types_match("Union[datetime, NoneType]", "str") is False

    def test_optional_desugared(self):
        """Optional[int] normalizes to Union[int, NoneType]."""
        assert types_match("Optional[int]", "int") is True
        assert types_match("Optional[int]", "NoneType") is True
        assert types_match("Optional[int]", "str") is False

    def test_non_string_field_type(self):
        assert types_match(42, "int") is False
        assert types_match(None, "int") is False


# =========================================================================
# normalize_type_string
# =========================================================================


class TestNormalizeTypeString:

    @pytest.mark.parametrize(
        "input_str, expected",
        [
            ("int", "int"),
            ("Int", "int"),
            ("INT", "int"),
            ("ANY", "Any"),
            ("any", "Any"),
            ("nonetype", "NoneType"),
            ("NoneType", "NoneType"),
            ("ENUM", "enum"),
            ("Optional[int]", "Union[int, NoneType]"),
            ("LIST[INT]", "List[int]"),
            ("Dict[Str, Float]", "Dict[str, float]"),
            ("list", "list"),
            ("List[int]", "List[int]"),
        ],
        ids=[
            "lowercase",
            "titlecase",
            "uppercase",
            "any-upper",
            "any-lower",
            "nonetype-lower",
            "nonetype-proper",
            "enum-upper",
            "optional-desugar",
            "list-upper",
            "dict-mixed",
            "bare-list",
            "parameterized-list",
        ],
    )
    def test_normalization(self, input_str, expected):
        assert normalize_type_string(input_str) == expected

    def test_empty_string(self):
        assert normalize_type_string("") == ""

    def test_deeply_nested(self):
        result = normalize_type_string("List[Dict[str, List[int]]]")
        assert result == "List[Dict[str, List[int]]]"

    def test_json_schema_alias(self):
        assert normalize_type_string("string") == "str"
        assert normalize_type_string("integer") == "int"
        assert normalize_type_string("number") == "float"
        assert normalize_type_string("boolean") == "bool"
