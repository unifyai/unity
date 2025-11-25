from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.function_manager.function_manager import FunctionManager
from unity.guidance_manager.guidance_manager import GuidanceManager


@pytest.mark.unit
@_handle_project
def test_guidance_function_ids_roundtrip_and_fetch():
    # Seed two functions
    src_a = (
        "def alpha(a: int) -> int:\n"
        '    """Return value + 1"""\n'
        "    return a + 1\n"
    )
    src_b = (
        "def beta(b: int) -> int:\n" '    """Return value * 2"""\n' "    return b * 2\n"
    )
    fm = FunctionManager()
    fm.add_functions(implementations=[src_a, src_b])
    listing = fm.list_functions()
    alpha_id = listing["alpha"]["function_id"]
    beta_id = listing["beta"]["function_id"]

    # Create guidance that references both functions
    gm = GuidanceManager()
    out = gm._add_guidance(
        title="Math Ops",
        content="Guidance relevant to alpha and beta functions.",
        function_ids=[alpha_id, beta_id],
    )
    gid = out["details"]["guidance_id"]

    # Roundtrip: row stores function_ids
    rows = gm._filter(filter=f"guidance_id == {gid}", limit=1)
    assert rows and rows[0].function_ids == [alpha_id, beta_id]

    # Fetch related functions (without implementations)
    funcs = gm._get_functions_for_guidance(
        guidance_id=gid,
        include_implementations=False,
    )
    names = {f["name"] for f in funcs}
    assert names == {"alpha", "beta"}
    assert all("implementation" not in f for f in funcs)

    # Fetch related functions (with implementations)
    funcs_with_impl = gm._get_functions_for_guidance(
        guidance_id=gid,
        include_implementations=True,
    )
    assert any("implementation" in f for f in funcs_with_impl)


@pytest.mark.unit
@_handle_project
def test_guidance_attach_functions_limit_and_update():
    # Seed two functions
    src_x = "def inc(x: int) -> int:\n" '    """Increment"""\n' "    return x + 1\n"
    src_y = "def dbl(y: int) -> int:\n" '    """Double"""\n' "    return y * 2\n"
    fm = FunctionManager()
    fm.add_functions(implementations=[src_x, src_y])
    listing = fm.list_functions()
    inc_id = listing["inc"]["function_id"]
    dbl_id = listing["dbl"]["function_id"]

    gm = GuidanceManager()
    out = gm._add_guidance(
        title="Calculations",
        content="Useful operations for math.",
        function_ids=[inc_id, dbl_id],
    )
    gid = out["details"]["guidance_id"]

    # Attach with a limit
    payload = gm._attach_functions_for_guidance_to_context(
        guidance_id=gid,
        include_implementations=False,
        limit=1,
    )
    assert isinstance(payload, dict)
    assert payload.get("attached_count") == 1
    assert isinstance(payload.get("functions"), list) and len(payload["functions"]) == 1

    # Update to a single function id
    gm._update_guidance(guidance_id=gid, function_ids=[inc_id])
    rows = gm._filter(filter=f"guidance_id == {gid}", limit=1)
    assert rows and rows[0].function_ids == [inc_id]

    funcs_after = gm._get_functions_for_guidance(guidance_id=gid)
    assert len(funcs_after) == 1 and funcs_after[0]["function_id"] == inc_id


@pytest.mark.unit
@_handle_project
def test_guidance_columns_include_function_ids():
    gm = GuidanceManager()
    cols = gm._list_columns()
    assert "function_ids" in cols
