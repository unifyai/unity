from __future__ import annotations

import uuid

import unify

from unity.common.metrics_utils import reduce_logs, SUPPORTED_REDUCTION_METRICS


def _create_test_context() -> str:
    ctx = f"tests/common/metrics_utils/{uuid.uuid4().hex}"
    unify.create_context(
        ctx,
        unique_keys={"row_id": "int"},
        auto_counting={"row_id": None},
        description="metrics_utils test context",
    )
    unify.create_fields(
        {
            "row_id": {"type": "int"},
            "value": {"type": "float"},
            "category": {"type": "str"},
        },
        context=ctx,
    )
    # Three simple rows for deterministic aggregates
    unify.log(context=ctx, new=True, row_id=1, value=1.0, category="a")
    unify.log(context=ctx, new=True, row_id=2, value=2.0, category="a")
    unify.log(context=ctx, new=True, row_id=3, value=3.0, category="b")
    return ctx


def test_reduce_logs_single_key_and_filter():
    ctx = _create_test_context()
    try:
        # sum over all rows
        total = reduce_logs(context=ctx, metric="sum", keys="value")
        assert total == 6.0

        # mean over a filtered subset (row_id > 1 → values 2.0 and 3.0)
        mean_gt1 = reduce_logs(
            context=ctx,
            metric="mean",
            keys="value",
            filter="row_id > 1",
        )
        assert mean_gt1 == 2.5
    finally:
        unify.delete_context(ctx)


def test_reduce_logs_multi_key_and_group_by():
    ctx = _create_test_context()
    try:
        # Multiple keys, no grouping → dict[key -> scalar]
        res = reduce_logs(context=ctx, metric="max", keys=["row_id", "value"])
        assert isinstance(res, dict)
        assert set(res.keys()) == {"row_id", "value"}

        # Grouped by category → nested dict keyed by group values
        grouped = reduce_logs(
            context=ctx,
            metric="sum",
            keys="value",
            group_by="category",
        )
        assert isinstance(grouped, dict)
        # We know categories 'a' and 'b' are present
        assert set(grouped.keys()) == {"a", "b"}

        # Grouped by multiple fields (category, row_id) → still a dict-shaped result
        grouped_multi = reduce_logs(
            context=ctx,
            metric="sum",
            keys=["value"],
            group_by=["category", "row_id"],
        )
        assert isinstance(grouped_multi, dict)
    finally:
        unify.delete_context(ctx)


def test_reduce_logs_rejects_unsupported_metric():
    ctx = _create_test_context()
    try:
        fake_metric = "not_a_real_metric"
        assert fake_metric not in SUPPORTED_REDUCTION_METRICS
        try:
            reduce_logs(context=ctx, metric=fake_metric, keys="row_id")
            assert False, f"Expected ValueError for unsupported metric {fake_metric!r}"
        except ValueError:
            pass
    finally:
        unify.delete_context(ctx)
