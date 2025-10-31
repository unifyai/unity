from __future__ import annotations

import uuid
import pytest
import unify


@pytest.mark.unit
def test_metrics_count_matches_rows_and_resets_on_context_delete():
    """
    Fundamental reproduction for metric count inconsistency on auto-increment fields.

    Steps:
      1) Create a fresh context with an auto-incrementing "row_id" field.
      2) Verify the metric count starts at 0 (fresh context).
      3) Insert three rows without specifying "row_id" (auto-counting assigns ids).
      4) Verify get_logs(...) returns 3 rows and metric count == 3.
      5) Delete the context and recreate it.
      6) Verify metric count resets to 0 for the recreated context.

    Expected: All assertions should pass. If any fail, metric aggregation is
    inconsistent with context scoping or not reset on deletion.
    """

    ctx = f"tests/local_storage/metrics/{uuid.uuid4().hex}"

    def _create_ctx():
        # Ensure a context with auto-counting on 'row_id'
        unify.create_context(
            ctx,
            unique_keys={"row_id": "int"},
            auto_counting={"row_id": None},
            description="Metrics test context",
        )
        unify.create_fields(
            {
                "row_id": {"type": "int"},
                "name": {"type": "str"},
            },
            context=ctx,
        )

    def _as_int0(v):
        return 0 if v is None else int(v)

    try:
        # Create fresh context
        _create_ctx()

        # Initial metric should be zero in a fresh context
        initial_metric = unify.get_logs_metric(
            metric="count",
            key="row_id",
            context=ctx,
        )
        assert (
            _as_int0(initial_metric) == 0
        ), f"Expected initial metric 0 for fresh context, got {initial_metric} (context={ctx})"

        # Insert three rows; 'id' is auto-incremented by the backend
        unify.log(context=ctx, new=True, name="A")
        unify.log(context=ctx, new=True, name="B")
        unify.log(context=ctx, new=True, name="C")

        rows = unify.get_logs(context=ctx, return_ids_only=False)
        row_count = len(rows)
        metric_after = unify.get_logs_metric(metric="count", key="row_id", context=ctx)

        assert row_count == 3, f"Expected 3 rows, got {row_count} (context={ctx})"
        assert (
            _as_int0(metric_after) == row_count
        ), f"Metric/row mismatch in context={ctx}: metric={_as_int0(metric_after)}, rows={row_count}"

        # Delete context and recreate; metric must reset to 0
        unify.delete_context(ctx)
        _create_ctx()
        metric_reset = unify.get_logs_metric(metric="count", key="row_id", context=ctx)
        assert (
            _as_int0(metric_reset) == 0
        ), f"Expected metric to reset to 0 after context deletion, got {_as_int0(metric_reset)} (context={ctx})"

    finally:
        # Best-effort cleanup
        try:
            unify.delete_context(ctx)
        except Exception:
            pass
