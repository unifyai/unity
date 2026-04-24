"""KnowledgeManager keeps its aggregation opt-out on both solo and Hive.

KnowledgeManager sets ``include_in_multi_assistant_table = False`` so that
per-body Knowledge writes do not mirror into ``{user}/All/Knowledge/...``
or ``All/Knowledge/...`` aggregation siblings. When a body joins a Hive,
the write path lands on ``Hives/{hive_id}/Knowledge/...`` and
``_ensure_all_contexts`` short-circuits, so Hive writes also never mint
``Hives/{h}/All/Knowledge/...`` aggregation siblings.

This test is the canary for that asymmetry: if a future refactor flips
``include_in_multi_assistant_table`` on Knowledge, or if the Hive-path
short-circuit regresses in ``_ensure_all_contexts``, the test fails.
"""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest

from unity.common.context_registry import ContextRegistry, TableContext
from unity.knowledge_manager.knowledge_manager import KnowledgeManager


pytestmark = pytest.mark.usefixtures("pinned_hive_body")


def test_knowledge_manager_init_still_sets_opt_out_flag():
    """``KnowledgeManager.__init__`` sets ``include_in_multi_assistant_table = False``.

    The flag suppresses ``add_to_all_context`` mirroring on writes, which is
    the whole point of the Knowledge aggregation opt-out; a source-level
    check keeps the invariant under review without needing a live Unify
    backend to instantiate the manager.
    """
    source = inspect.getsource(KnowledgeManager.__init__)
    assert "self.include_in_multi_assistant_table = False" in source


def test_hive_knowledge_path_skips_aggregation_provisioning():
    """No ``Hives/{h}/All/Knowledge/...`` shell is minted for a Hive write."""
    table = TableContext(name="Knowledge", description="stub")

    with patch(
        "unity.common.context_registry._create_context_with_retry",
    ) as create_ctx:
        ContextRegistry._ensure_all_contexts("Hives/42/Knowledge", table)

    create_ctx.assert_not_called()


def test_per_body_knowledge_path_keeps_aggregation_shells():
    """Per-body Knowledge keeps its ``{user}/All/...`` and ``All/...`` shells.

    The aggregation shells are provisioned unconditionally at context
    setup; the opt-out only suppresses ``add_to_all_context`` mirroring at
    **write** time. Solo bodies therefore still own the shells, they just
    do not receive row references.
    """
    table = TableContext(name="Knowledge", description="stub")

    with patch(
        "unity.common.context_registry._create_context_with_retry",
    ) as create_ctx:
        ContextRegistry._ensure_all_contexts("u7/7/Knowledge", table)

    created_paths = [call.args[0] for call in create_ctx.call_args_list]
    assert "u7/All/Knowledge" in created_paths
    assert "All/Knowledge" in created_paths
