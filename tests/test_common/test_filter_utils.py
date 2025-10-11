from __future__ import annotations

import pytest

from unity.common.filter_utils import normalize_filter_expr


@pytest.mark.parametrize(
    "expr",
    [
        None,
        "",
        "   ",
        "status not in ('completed','cancelled')",
    ],
)
def test_basic_passthrough_and_membership_allowed(expr):
    normalized = normalize_filter_expr(expr)
    if expr is None or not (isinstance(expr, str) and expr.strip()):
        assert normalized is expr


def test_attribute_rewrite():
    expr = (
        "schedule.start_at == '2025-06-13' and status not in ('completed','cancelled')"
    )
    normalized = normalize_filter_expr(expr)
    assert "schedule['start_at']" in normalized


def test_get_rewrite():
    expr = "schedule.get('start_at') is not None"
    normalized = normalize_filter_expr(expr)
    assert "schedule['start_at'] is not None" in normalized


@pytest.mark.parametrize(
    "expr",
    [
        "deadline[:10] == '2025-06-13'",  # slicing should remain as-is (no rewrite yet)
        "1 + 2 == 3 and status == 'queued'",  # arithmetic left unchanged
        "arr[i] == 1",  # computed index left unchanged
    ],
)
def test_unsupported_patterns_left_unchanged(expr: str):
    out = normalize_filter_expr(expr)
    assert out == expr


def test_attribute_chain_rewrites():
    expr = "a.b.c == 1"
    normalized = normalize_filter_expr(expr)
    # Expect nested attribute access rewritten into chained subscripts
    assert "a['b']['c'] == 1" in normalized


def test_string_key_subscript_is_safe():
    expr = "obj['field'] == 1"
    normalized = normalize_filter_expr(expr)
    assert normalized == expr


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("isinstance(x, str)", "type(x) is str"),
        ("isinstance(x, (str, int))", "type(x) in (str, int)"),
    ],
)
def test_isinstance_rewrites(expr: str, expected: str):
    normalized = normalize_filter_expr(expr)
    assert normalized == expected
