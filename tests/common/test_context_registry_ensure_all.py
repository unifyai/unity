"""``ContextRegistry._ensure_all_contexts`` aggregation bypass on Hive paths.

Per-body paths mint two aggregation shells (``{user}/All/<suffix>`` and
``All/<suffix>``) so cross-assistant and cross-user queries can run.
Hive-scoped paths already are the shared-across-bodies surface, so
minting ``Hives/{hive_id}/All/...`` shells would pollute the Hive cascade
and never serve a real reader.

These tests patch the low-level context-store helpers so the bypass and
the preserved per-body path are exercised without a live Unify backend.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from unity.common.context_registry import ContextRegistry, TableContext


@pytest.fixture
def tasks_table_context() -> TableContext:
    return TableContext(
        name="Tasks",
        description="Task definitions owned by this body or Hive.",
        fields={"task_id": "int", "name": "str"},
    )


def test_ensure_all_contexts_short_circuits_on_hive_path(tasks_table_context):
    """A ``Hives/{id}/Tasks/Activations`` target mints no aggregation shells."""
    with (
        patch(
            "unity.common.context_registry._create_context_with_retry",
        ) as create_ctx,
        patch("unity.common.context_registry.create_fields") as create_fields,
    ):
        ContextRegistry._ensure_all_contexts(
            "Hives/42/Tasks/Activations",
            tasks_table_context,
        )

    create_ctx.assert_not_called()
    create_fields.assert_not_called()


def test_ensure_all_contexts_provisions_shells_for_per_body_path(tasks_table_context):
    """Solo regression: per-body path still mints ``{user}/All`` and ``All`` shells."""
    with (
        patch(
            "unity.common.context_registry._create_context_with_retry",
        ) as create_ctx,
        patch("unity.common.context_registry.create_fields"),
    ):
        ContextRegistry._ensure_all_contexts(
            "u7/7/Tasks/Activations",
            tasks_table_context,
        )

    minted = [call.args[0] for call in create_ctx.call_args_list]
    assert "u7/All/Tasks/Activations" in minted
    assert "All/Tasks/Activations" in minted


def test_ensure_all_contexts_preserves_per_body_behavior_for_knowledge():
    """Knowledge's write-mirror opt-out is orthogonal to the Hive bypass.

    ``include_in_multi_assistant_table=False`` (set on the manager
    instance) gates whether ``add_to_all_context=True`` fires on write; it
    does not gate aggregation-context provisioning. The Hive bypass in
    :meth:`ContextRegistry._ensure_all_contexts` short-circuits only on
    ``Hives/...`` paths, so per-body Knowledge still mints ``{user}/All``
    and ``All`` shells.
    """
    knowledge_tc = TableContext(
        name="Knowledge",
        description="Knowledge manager surface.",
        fields={"knowledge_id": "int"},
    )
    with (
        patch(
            "unity.common.context_registry._create_context_with_retry",
        ) as create_ctx,
        patch("unity.common.context_registry.create_fields"),
    ):
        ContextRegistry._ensure_all_contexts("u7/7/Knowledge", knowledge_tc)

    minted = [call.args[0] for call in create_ctx.call_args_list]
    assert "u7/All/Knowledge" in minted
    assert "All/Knowledge" in minted
