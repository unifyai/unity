"""Tests for backfill utility."""

from __future__ import annotations

import unify
from tests.helpers import _handle_project
from unity.common.backfill import (
    backfill_assistant_field,
    backfill_all_contexts_for_assistant,
)


def _count_logs_missing_assistant(logs) -> int:
    """Count logs where _assistant is missing or None (client-side check)."""
    return sum(1 for lg in logs if lg.entries.get("_assistant") is None)


@_handle_project
def test_backfill_assistant_field():
    """Backfill should update logs missing _assistant."""
    ctx = "TestAssistant/TestBackfill"

    # Create context with required fields
    unify.delete_context(ctx)
    unify.create_context(ctx)
    unify.create_fields({"name": "str", "_assistant": "str"}, context=ctx)

    # Create logs without _assistant
    for i in range(5):
        unify.log(context=ctx, name=f"item_{i}", new=True)

    # Verify they don't have _assistant (client-side check)
    logs = unify.get_logs(context=ctx)
    assert (
        _count_logs_missing_assistant(logs) == 5
    ), f"Expected 5 logs without _assistant, got {_count_logs_missing_assistant(logs)}"

    # Run backfill
    result = backfill_assistant_field(ctx, "TestAssistant")
    assert result["total_updated"] == 5, f"Expected 5 updated, got {result}"
    assert result["context"] == ctx

    # Verify all now have _assistant (client-side check)
    logs_after = unify.get_logs(context=ctx)
    assert (
        _count_logs_missing_assistant(logs_after) == 0
    ), "All logs should now have _assistant"

    # Verify correct value was set
    logs_with = unify.get_logs(context=ctx, filter="_assistant == 'TestAssistant'")
    assert len(logs_with) == 5, f"Expected 5 logs with _assistant, got {len(logs_with)}"


@_handle_project
def test_backfill_assistant_field_empty_context():
    """Backfill should handle empty context gracefully."""
    ctx = "TestAssistant/TestBackfillEmpty"

    # Create empty context
    unify.delete_context(ctx)
    unify.create_context(ctx)
    unify.create_fields({"name": "str", "_assistant": "str"}, context=ctx)

    # Run backfill on empty context
    result = backfill_assistant_field(ctx, "TestAssistant")
    assert result["total_updated"] == 0
    assert result["context"] == ctx


@_handle_project
def test_backfill_assistant_field_with_filter():
    """Backfill should respect additional filter."""
    ctx = "TestAssistant/TestBackfillFilter"

    unify.delete_context(ctx)
    unify.create_context(ctx)
    unify.create_fields(
        {"name": "str", "category": "str", "_assistant": "str"},
        context=ctx,
    )

    # Create logs with different categories
    for i in range(3):
        unify.log(context=ctx, name=f"item_a_{i}", category="A", new=True)
    for i in range(2):
        unify.log(context=ctx, name=f"item_b_{i}", category="B", new=True)

    # Backfill only category A
    result = backfill_assistant_field(
        ctx,
        "TestAssistant",
        filter="category == 'A'",
    )
    assert result["total_updated"] == 3

    # Verify category A has _assistant
    logs_a = unify.get_logs(
        context=ctx,
        filter="category == 'A' and _assistant == 'TestAssistant'",
    )
    assert len(logs_a) == 3

    # Verify category B does not have _assistant (client-side check)
    logs_b = unify.get_logs(context=ctx, filter="category == 'B'")
    assert (
        _count_logs_missing_assistant(logs_b) == 2
    ), "Category B logs should not have _assistant"


@_handle_project
def test_backfill_all_contexts_for_assistant():
    """Backfill should update all contexts for an assistant."""
    assistant = "TestMultiCtx"

    # Create multiple contexts
    ctx1 = f"{assistant}/ContextOne"
    ctx2 = f"{assistant}/ContextTwo"

    for ctx in [ctx1, ctx2]:
        unify.delete_context(ctx)
        unify.create_context(ctx)
        unify.create_fields({"name": "str", "_assistant": "str"}, context=ctx)

    # Create logs in each context
    for i in range(3):
        unify.log(context=ctx1, name=f"ctx1_item_{i}", new=True)
    for i in range(2):
        unify.log(context=ctx2, name=f"ctx2_item_{i}", new=True)

    # Backfill all contexts
    results = backfill_all_contexts_for_assistant(assistant)

    assert ctx1 in results
    assert ctx2 in results
    assert results[ctx1]["total_updated"] == 3
    assert results[ctx2]["total_updated"] == 2


@_handle_project
def test_backfill_idempotent():
    """Running backfill twice should not update already-backfilled logs."""
    ctx = "TestAssistant/TestBackfillIdempotent"

    unify.delete_context(ctx)
    unify.create_context(ctx)
    unify.create_fields({"name": "str", "_assistant": "str"}, context=ctx)

    for i in range(3):
        unify.log(context=ctx, name=f"item_{i}", new=True)

    # First backfill
    result1 = backfill_assistant_field(ctx, "TestAssistant")
    assert result1["total_updated"] == 3

    # Second backfill should find nothing to update
    result2 = backfill_assistant_field(ctx, "TestAssistant")
    assert result2["total_updated"] == 0
