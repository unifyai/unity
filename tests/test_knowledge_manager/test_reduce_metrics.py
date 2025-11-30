from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.knowledge_manager.knowledge_manager import KnowledgeManager


@pytest.mark.requires_real_unify
@_handle_project
def test_knowledge_reduce_param_shapes():
    km = KnowledgeManager()

    # Provision a simple numeric table for metrics
    km._create_table(name="MetricsTable")
    km._add_rows(
        table="MetricsTable",
        rows=[
            {"item": "A", "units": 1},
            {"item": "B", "units": 2},
        ],
    )

    # Single key, no grouping
    scalar = km._reduce(table="MetricsTable", metric="sum", keys="units")
    assert isinstance(scalar, (int, float))

    # Multiple keys, no grouping
    multi = km._reduce(
        table="MetricsTable",
        metric="max",
        keys=["units", "row_id"],
    )
    assert isinstance(multi, dict)
    assert set(multi.keys()) == {"units", "row_id"}

    # Single key, group_by string
    grouped_str = km._reduce(
        table="MetricsTable",
        metric="sum",
        keys="units",
        group_by="item",
    )
    assert isinstance(grouped_str, dict)

    # Multiple keys, group_by string
    grouped_str_multi = km._reduce(
        table="MetricsTable",
        metric="min",
        keys=["units"],
        group_by="item",
    )
    assert isinstance(grouped_str_multi, dict)

    # Single key, group_by list
    grouped_list = km._reduce(
        table="MetricsTable",
        metric="sum",
        keys="units",
        group_by=["item", "row_id"],
    )
    assert isinstance(grouped_list, dict)

    # Multiple keys, group_by list
    grouped_list_multi = km._reduce(
        table="MetricsTable",
        metric="mean",
        keys=["units"],
        group_by=["item", "row_id"],
    )
    assert isinstance(grouped_list_multi, dict)

    # Filter as string
    filtered_scalar = km._reduce(
        table="MetricsTable",
        metric="sum",
        keys="units",
        filter="units >= 0",
    )
    assert isinstance(filtered_scalar, (int, float))

    # Filter as per-key dict
    filtered_multi = km._reduce(
        table="MetricsTable",
        metric="sum",
        keys=["units"],
        filter={"units": "units >= 0"},
    )
    assert isinstance(filtered_multi, dict)
