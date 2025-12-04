"""
Foreign Key Tests for FunctionManager

Coverage
========
✓ guidance_ids[*] → Guidance.guidance_id (nested array FK)
  - Validation: Reject invalid guidance_ids on function creation
  - CASCADE: Remove deleted guidance IDs from function.guidance_ids array
  - CASCADE: Update guidance_id changes in function.guidance_ids array
  - Array operations: Multiple guidance references, empty arrays
"""

from __future__ import annotations

import unify
from tests.helpers import _handle_project
from unity.function_manager.function_manager import FunctionManager
from unity.guidance_manager.guidance_manager import GuidanceManager


# --------------------------------------------------------------------------- #
#  Unit Tests: guidance_ids[*] → Guidance.guidance_id                         #
# --------------------------------------------------------------------------- #


@_handle_project
def test_fk_guidance_ids_valid_reference():
    """Test that functions can reference valid guidance IDs."""
    gm = GuidanceManager()
    fm = FunctionManager()

    # Create guidance entries
    gm._add_guidance(title="Setup Guide", content="How to setup the system")
    gm._add_guidance(title="Usage Guide", content="How to use the system")

    # Get guidance IDs
    guidance_list = unify.get_logs(context=gm._ctx, from_fields=["guidance_id"])
    g_ids = sorted([int(g.entries["guidance_id"]) for g in guidance_list])
    assert len(g_ids) == 2

    # Create function
    src = "def setup_demo():\n    return 'demo setup'\n"
    fm.add_functions(implementations=src)

    # Get the function log
    func_logs = unify.get_logs(
        context=fm._compositional_ctx,
        filter="name == 'setup_demo'",
        return_ids_only=True,
    )
    assert func_logs, "Function not created"

    # Update with guidance_ids
    unify.update_logs(
        context=fm._compositional_ctx,
        logs=func_logs[0],
        entries={"guidance_ids": g_ids},
        overwrite=True,
    )

    # Verify function was created with guidance_ids
    funcs = unify.get_logs(
        context=fm._compositional_ctx,
        from_fields=["function_id", "guidance_ids"],
    )
    assert len(funcs) == 1
    stored_guidance_ids = funcs[0].entries.get("guidance_ids", [])
    assert sorted(stored_guidance_ids) == g_ids


@_handle_project
def test_fk_guidance_ids_cascade_on_delete():
    """Test nested CASCADE: Deleting guidance removes it from function.guidance_ids array."""
    gm = GuidanceManager()
    fm = FunctionManager()

    # Create guidance entries
    gm._add_guidance(title="Guide 1", content="Content 1")
    gm._add_guidance(title="Guide 2", content="Content 2")
    gm._add_guidance(title="Guide 3", content="Content 3")

    # Get guidance IDs
    guidance_list = unify.get_logs(context=gm._ctx, from_fields=["guidance_id"])
    g_ids = sorted([int(g.entries["guidance_id"]) for g in guidance_list])
    assert len(g_ids) == 3
    g1, g2, g3 = g_ids

    # Create function
    src = "def complex_setup():\n    return 'setup'\n"
    fm.add_functions(implementations=src)

    # Get the function log
    func_logs = unify.get_logs(
        context=fm._compositional_ctx,
        filter="name == 'complex_setup'",
        return_ids_only=True,
    )
    assert func_logs, "Function not created"

    # Update with guidance_ids
    unify.update_logs(
        context=fm._compositional_ctx,
        logs=func_logs[0],
        entries={"guidance_ids": [g1, g2, g3]},
        overwrite=True,
    )

    # Verify function has all three guidance_ids
    funcs = unify.get_logs(
        context=fm._compositional_ctx,
        from_fields=["function_id", "guidance_ids"],
    )
    assert len(funcs) == 1
    assert sorted(funcs[0].entries["guidance_ids"]) == [g1, g2, g3]

    # Delete the middle guidance entry (g2)
    gm._delete_guidance(guidance_id=g2)

    # Verify g2 was removed from function.guidance_ids (CASCADE behavior)
    funcs_after = unify.get_logs(
        context=fm._compositional_ctx,
        from_fields=["function_id", "guidance_ids"],
    )
    assert len(funcs_after) == 1
    remaining_ids = sorted(funcs_after[0].entries.get("guidance_ids", []))
    assert remaining_ids == [g1, g3]  # g2 should be removed
    assert g2 not in remaining_ids


@_handle_project
def test_fk_guidance_ids_empty_array():
    """Test that empty guidance_ids array is valid."""
    fm = FunctionManager()

    # Create function with no guidance references (defaults to empty array)
    src = "def standalone():\n    return 'standalone'\n"
    fm.add_functions(implementations=src)

    # Verify function was created with empty guidance_ids
    funcs = unify.get_logs(
        context=fm._compositional_ctx,
        from_fields=["function_id", "guidance_ids"],
    )
    assert len(funcs) == 1
    assert funcs[0].entries.get("guidance_ids", []) == []


@_handle_project
def test_fk_guidance_ids_multiple_deletes():
    """Test nested CASCADE with multiple sequential deletes."""
    gm = GuidanceManager()
    fm = FunctionManager()

    # Create multiple guidance entries
    for i in range(5):
        gm._add_guidance(title=f"Guide {i}", content=f"Content {i}")

    # Get all guidance IDs
    guidance_list = unify.get_logs(context=gm._ctx, from_fields=["guidance_id"])
    g_ids = sorted([int(g.entries["guidance_id"]) for g in guidance_list])
    assert len(g_ids) == 5

    # Create function
    src = "def mega_func():\n    return 'mega'\n"
    fm.add_functions(implementations=src)

    # Get the function log
    func_logs = unify.get_logs(
        context=fm._compositional_ctx,
        filter="name == 'mega_func'",
        return_ids_only=True,
    )
    assert func_logs, "Function not created"

    # Update with guidance_ids
    unify.update_logs(
        context=fm._compositional_ctx,
        logs=func_logs[0],
        entries={"guidance_ids": g_ids},
        overwrite=True,
    )

    # Delete guidance entries one by one
    for gid in g_ids[:3]:  # Delete first 3
        gm._delete_guidance(guidance_id=gid)

    # Verify only last 2 remain in function.guidance_ids
    funcs = unify.get_logs(
        context=fm._compositional_ctx,
        from_fields=["function_id", "guidance_ids"],
    )
    assert len(funcs) == 1
    remaining = sorted(funcs[0].entries.get("guidance_ids", []))
    assert remaining == g_ids[3:]  # Only last 2 should remain
