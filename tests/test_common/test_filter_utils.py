from __future__ import annotations

import pytest

from unity.common.filter_utils import normalize_and_validate_filter_expr


@pytest.mark.parametrize(
    "expr, expected_safe",
    [
        (None, True),
        ("", True),
        ("   ", True),
        ("status not in ('completed','cancelled')", True),
    ],
)
def test_basic_passthrough_and_membership_allowed(expr, expected_safe):
    normalized, is_safe = normalize_and_validate_filter_expr(expr)
    assert is_safe is expected_safe
    # For None/empty, normalized should equal the input
    if expr is None or not (isinstance(expr, str) and expr.strip()):
        assert normalized is expr


def test_attribute_rewrite_and_safety():
    expr = (
        "schedule.start_at == '2025-06-13' and status not in ('completed','cancelled')"
    )
    normalized, is_safe = normalize_and_validate_filter_expr(expr)
    assert is_safe is True
    assert "schedule['start_at']" in normalized


def test_get_rewrite_and_safety():
    expr = "schedule.get('start_at') is not None"
    normalized, is_safe = normalize_and_validate_filter_expr(expr)
    assert is_safe is True
    assert "schedule['start_at'] is not None" in normalized


@pytest.mark.parametrize(
    "expr",
    [
        "isinstance(deadline, str)",  # disallowed call
        "deadline[:10] == '2025-06-13'",  # slicing disallowed
        "1 + 2 == 3 and status == 'queued'",  # arithmetic disallowed
        "arr[i] == 1",  # computed index subscript disallowed
    ],
)
def test_disallowed_constructs_flag_unsafe(expr: str):
    _, is_safe = normalize_and_validate_filter_expr(expr)
    assert is_safe is False


def test_attribute_chain_rewrites():
    expr = "a.b.c == 1"
    normalized, is_safe = normalize_and_validate_filter_expr(expr)
    assert is_safe is True
    # Expect nested attribute access rewritten into chained subscripts
    assert "a['b']['c'] == 1" in normalized


def test_string_key_subscript_is_safe():
    expr = "obj['field'] == 1"
    normalized, is_safe = normalize_and_validate_filter_expr(expr)
    assert is_safe is True
    assert normalized == expr
