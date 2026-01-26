from __future__ import annotations

import uuid
import unify


def test_count_matches_rows_and_resets():
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
        unify.delete_context(ctx)


def test_max_matches_row_ids_and_resets():
    """
    Fundamental check for the "max" metric on an auto-increment field.

    Steps:
      1) Create a fresh context with auto-incrementing "row_id".
      2) Verify initial max metric is 0 (empty context).
      3) Insert three rows; read back the rows and compute max(row_id) from entries.
      4) Verify get_logs_metric("max", key="row_id") equals that computed max.
      5) Delete the context and recreate; verify max resets to 0.
    """

    ctx = f"tests/local_storage/metrics_max/{uuid.uuid4().hex}"

    def _as_int0(v):
        return 0 if v is None else int(v)

    def _create_ctx():
        unify.create_context(
            ctx,
            unique_keys={"row_id": "int"},
            auto_counting={"row_id": None},
            description="Metrics test context (max)",
        )
        unify.create_fields(
            {
                "row_id": {"type": "int"},
                "name": {"type": "str"},
            },
            context=ctx,
        )

    try:
        _create_ctx()

        initial_max = unify.get_logs_metric(metric="max", key="row_id", context=ctx)
        assert (
            _as_int0(initial_max) == 0
        ), f"Expected initial max 0 for fresh context, got {initial_max} (context={ctx})"

        # Insert three rows; backend assigns row_id automatically
        unify.log(context=ctx, new=True, name="A")
        unify.log(context=ctx, new=True, name="B")
        unify.log(context=ctx, new=True, name="C")

        logs = unify.get_logs(context=ctx, return_ids_only=False)
        # Extract row_id values from returned logs
        row_ids = []
        for lg in logs:
            try:
                entries = getattr(lg, "entries", lg)
            except Exception:
                entries = lg
            rid = entries.get("row_id")
            if isinstance(rid, int):
                row_ids.append(rid)

        assert (
            len(row_ids) == 3
        ), f"Expected 3 row_id values after inserts, got {len(row_ids)} (context={ctx})"

        computed_max = max(row_ids) if row_ids else 0
        metric_max_after = unify.get_logs_metric(
            metric="max",
            key="row_id",
            context=ctx,
        )
        assert (
            _as_int0(metric_max_after) == computed_max
        ), f"Metric max mismatch in context={ctx}: metric={_as_int0(metric_max_after)}, rows_max={computed_max}, rows={sorted(row_ids)}"

        # Delete context and recreate; metric must reset to 0
        unify.delete_context(ctx)
        _create_ctx()
        metric_max_reset = unify.get_logs_metric(
            metric="max",
            key="row_id",
            context=ctx,
        )
        assert (
            _as_int0(metric_max_reset) == 0
        ), f"Expected max metric to reset to 0 after context deletion, got {_as_int0(metric_max_reset)} (context={ctx})"

    finally:
        unify.delete_context(ctx)
