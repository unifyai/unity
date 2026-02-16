"""Storage-loop integration test for FunctionManager + GuidanceManager.

Verifies that the post-completion storage check loop correctly identifies
and stores *both* reusable functions (in FM) and compositional workflow
guidance (in GM) when the trajectory warrants it.

This complements the existing storage tests in ``test_can_compose_and_store.py``
which only assert on FunctionManager storage.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from unity.actor.code_act_actor import CodeActActor

pytestmark = pytest.mark.eval


# ---------------------------------------------------------------------------
# GuidanceManager mock with real method signatures and docstrings.
#
# We avoid MagicMock because the storage-check loop accesses
# ``type(gm).<method>.__doc__`` to wire tool docstrings, and MagicMock's
# metaclass attribute access doesn't expose stable docstrings.
# ---------------------------------------------------------------------------


class _TrackingGuidanceManager:
    """Minimal GuidanceManager stand-in that records ``add_guidance`` calls."""

    def __init__(self) -> None:
        self.add_calls: list[dict] = []

    def search(self, references=None, k=10):
        """Search for guidance entries by semantic similarity to reference content."""
        return []

    def filter(self, filter=None, offset=0, limit=100):
        """Filter guidance entries using a Python filter expression."""
        return []

    def add_guidance(self, *, title, content, function_ids=None):
        """Add a guidance entry describing a compositional workflow or playbook."""
        self.add_calls.append(
            {"title": title, "content": content, "function_ids": function_ids},
        )
        return {"details": {"guidance_id": len(self.add_calls)}}

    def update_guidance(self, *, guidance_id, title=None, content=None, function_ids=None):
        """Update an existing guidance entry."""
        return {"details": {"guidance_id": guidance_id}}

    def delete_guidance(self, *, guidance_id):
        """Delete a guidance entry by ID."""
        return {"deleted": True}


# ---------------------------------------------------------------------------
# Test: storage loop stores both function(s) AND guidance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_storage_loop_stores_both_function_and_guidance():
    """The storage check stores both functions (FM) and guidance (GM).

    The task produces a multi-step data-cleaning pipeline with:
    - Two parameterized, general-purpose utility functions that work on
      *any* dict-based dataset (not tied to a specific schema).
    - A composition function that chains normalize → filter → deduplicate
      with conditional logic and summarization.

    The storage-check librarian should recognise:
    - The parameterized utilities as genuinely reusable → store via FM.
    - The multi-step composition (which fields get which ops, merge
      strategy selection, empty-record filtering, statistics) as a
      non-trivial workflow recipe → store via GM.
    """
    fm = MagicMock()
    fm.search_functions = MagicMock(return_value={"metadata": []})
    fm.filter_functions = MagicMock(return_value={"metadata": []})
    fm.list_functions = MagicMock(return_value={"metadata": []})
    fm.add_functions = MagicMock(return_value={"stored": "added"})
    fm.delete_function = MagicMock(return_value={})

    gm = _TrackingGuidanceManager()

    actor = CodeActActor(
        function_manager=fm,
        guidance_manager=gm,
        timeout=180,
    )
    try:
        handle = await actor.act(
            "I regularly process messy data exports from multiple sources and need "
            "a reliable, reusable cleaning toolkit. Build these general-purpose "
            "utilities (parameterized — not tied to any specific schema):\n\n"
            "1. `transform_field(value, operations: list[str]) -> str`\n"
            "   Chains text operations on a single value. Supported ops:\n"
            "   'strip', 'lower', 'upper', 'title', 'digits_only', 'alpha_only', "
            "'collapse_spaces'.\n"
            "   Returns '' for None/non-string input. Operations applied left-to-right.\n\n"
            "2. `detect_and_merge_duplicates(records: list[dict], match_keys: list[str], "
            "merge: str = 'most_complete') -> tuple[list[dict], list[tuple[int, ...]]]`\n"
            "   Groups records whose match_keys values are identical after strip+lower.\n"
            "   Merge strategies:\n"
            "   - 'most_complete': for each field, pick the non-empty value from the "
            "record that has the most filled fields overall.\n"
            "   - 'first'/'last': keep the first/last occurrence from each group.\n"
            "   Returns (unique_records, merge_groups) where merge_groups lists the "
            "original indices that were merged together.\n\n"
            "3. Compose these into `clean_export(records, field_ops, dedup_keys)` that:\n"
            "   a) Normalizes every record by applying field_ops "
            "(dict mapping field names to operation lists) via transform_field.\n"
            "   b) Removes records where ALL dedup_keys are empty after normalization.\n"
            "   c) Merges duplicates on dedup_keys using 'most_complete' strategy.\n"
            "   d) Returns {'cleaned': [...], 'empty_removed': int, "
            "'duplicates_merged': int, 'input_count': int, 'output_count': int}\n\n"
            "Test the full pipeline with this messy employee dataset:\n"
            "[\n"
            "  {'name': '  JOHN DOE  ', 'email': 'John@example.COM', "
            "'dept': 'Engineering', 'id': '1234'},\n"
            "  {'name': 'john doe', 'email': 'john@example.com', "
            "'dept': '', 'id': ''},\n"
            "  {'name': '', 'email': '', 'dept': '', 'id': ''},\n"
            "  {'name': 'Jane Smith', 'email': 'jane@example.com', "
            "'dept': 'Marketing', 'id': '5678'},\n"
            "  {'name': '  jane SMITH', 'email': ' Jane@Example.com ', "
            "'dept': '', 'id': '5678'},\n"
            "]\n"
            "Using: field_ops={'name': ['strip', 'title'], 'email': ['strip', 'lower'], "
            "'id': ['digits_only']}, dedup_keys=['email']\n\n"
            "Verify: John and Jane's duplicates are merged (keeping dept and id from "
            "the most complete record), the empty record is removed, and final count is 2.",
            can_store=True,
            persist=False,
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=240)
        assert result is not None

        # The storage check should have stored at least one function.
        fm.add_functions.assert_called(), (
            f"Expected FunctionManager.add_functions to be called for the "
            f"reusable data-cleaning utilities."
        )

        # The storage check should have stored guidance about the
        # data-cleaning workflow composition.
        assert gm.add_calls, (
            f"Expected GuidanceManager.add_guidance to be called for the "
            f"data-cleaning pipeline composition. "
            f"FM add_functions was called {fm.add_functions.call_count} time(s), "
            f"but no guidance was stored."
        )
    finally:
        try:
            await actor.close()
        except Exception:
            pass
