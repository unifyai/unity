from __future__ import annotations

import asyncio
import pytest
import unify

from tests.helpers import _handle_project
from unity.common.context_store import TableStore
from unity.common.search_utils import table_search_top_k


@pytest.mark.asyncio
@_handle_project
async def test_concurrent_semantic_search_helper_no_failures(capfd):
    """
    Minimal concurrent test on the underlying semantic search helper.

    Intentionally trigger many concurrent ``table_search_top_k`` calls on a
    fresh table to surface races in vector provisioning.

    This mirrors the previous manager-specific test but decouples it from
    ``ContactManager`` so it exercises the shared helper only.

    Note: This test currently fails with n==2 due to a race in vector ensure,
    and serves as a guard for future hardening.
    """

    # Build a per-test context under the active write context
    try:
        ctxs = unify.get_active_context()
        base_ctx = ctxs.get("write") if isinstance(ctxs, dict) else None
    except Exception:
        base_ctx = None
    ctx = f"{base_ctx}/ConcurrentSearch" if base_ctx else "ConcurrentSearch"

    # Provision a minimal schema used for semantic search
    store = TableStore(
        ctx,
        unique_keys={"item_id": "int"},
        auto_counting={"item_id": None},
        description="Minimal table for concurrent semantic search tests",
        fields={
            "item_id": {"type": "int", "mutable": True},
            "first_name": {"type": "str", "mutable": True},
            "bio": {"type": "str", "mutable": True},
        },
    )
    store.ensure_context()

    # Seed a few rows to take the semantic path on the 'bio' column
    unify.log(
        context=ctx,
        first_name="Alice",
        bio="Enjoys email threads and detailed reports",
        new=True,
        mutable=True,
    )
    unify.log(
        context=ctx,
        first_name="Bob",
        bio="Prefers short text messages, hates long emails",
        new=True,
        mutable=True,
    )
    unify.log(
        context=ctx,
        first_name="Carol",
        bio="Commutes by train; likes concise updates over email",
        new=True,
        mutable=True,
    )

    # Use the same reference to force contention on the same embedding column
    references = {"bio": "short emails and texts"}

    async def _one_call():
        # Run the potentially blocking call in a thread to allow true parallelism
        return await asyncio.to_thread(
            table_search_top_k,
            ctx,
            references,
            k=3,
            allowed_fields=["item_id", "first_name", "bio"],
            unique_id_field="item_id",
        )

    # Clear any prior captured output before we start
    try:
        capfd.readouterr()
    except Exception:
        pass

    # Launch two concurrent calls to provoke the race
    tasks = [asyncio.create_task(_one_call()) for _ in range(2)]

    # If any task raises, gather will surface the exception → test fails
    results = await asyncio.gather(*tasks)

    # Basic sanity: we should have results from all tasks
    assert len(results) == 2
    assert all(isinstance(r, list) for r in results)

    # Assert that no vector-provisioning errors were logged during concurrency
    captured = capfd.readouterr()
    combined_output = (captured.out or "") + "\n" + (captured.err or "")
    # These substrings are emitted when the embedding ensure races
    assert "Failed to create derived column" not in combined_output
    assert "UniqueViolation" not in combined_output
