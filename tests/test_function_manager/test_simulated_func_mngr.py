from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.function_manager.simulated import SimulatedFunctionManager


# --------------------------------------------------------------------------- #
#  add_functions                                                              #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_sim_fm_add_functions_accepts_single_and_multiple():
    fm = SimulatedFunctionManager()

    # single implementation
    one_src = "def alpha(x):\n    return x\n"
    res_one = fm.add_functions(implementations=one_src)
    assert isinstance(res_one, dict) and res_one
    ((name_one, status_one),) = res_one.items()
    assert name_one == "alpha"
    assert "added" in status_one.lower()

    # multiple implementations
    a_src = "def a():\n    return 1\n"
    b_src = "def b(y):\n    return y\n"
    res_many = fm.add_functions(implementations=[a_src, b_src])
    assert set(res_many.keys()) == {"a", "b"}
    assert all("added" in v.lower() for v in res_many.values())


# --------------------------------------------------------------------------- #
#  list_functions                                                             #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_sim_fm_list_functions_with_and_without_implementations():
    fm = SimulatedFunctionManager()

    listing = fm.list_functions()
    assert isinstance(listing, dict) and listing
    # simulated returns a deterministic example entry
    assert "example" in listing
    assert "implementation" not in listing["example"]

    listing_with_impl = fm.list_functions(include_implementations=True)
    assert "implementation" in listing_with_impl["example"]


# --------------------------------------------------------------------------- #
#  get_precondition                                                           #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_sim_fm_get_precondition_returns_none():
    fm = SimulatedFunctionManager()
    assert fm.get_precondition(function_name="does_not_matter") is None


# --------------------------------------------------------------------------- #
#  delete_function                                                            #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_sim_fm_delete_function_acknowledges():
    fm = SimulatedFunctionManager()
    ack = fm.delete_function(function_id=42, delete_dependents=True)
    assert isinstance(ack, dict) and ack
    ((k, v),) = ack.items()
    assert "id=42" in k
    assert "deleted" in v.lower()


# --------------------------------------------------------------------------- #
#  search_functions                                                           #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_sim_fm_search_functions_returns_list_of_dicts():
    fm = SimulatedFunctionManager()
    hits = fm.search_functions(filter="'price' in docstring")
    assert isinstance(hits, list) and hits
    first = hits[0]
    assert isinstance(first, dict)
    for key in ("name", "function_id", "argspec", "docstring"):
        assert key in first


# --------------------------------------------------------------------------- #
#  search_functions_by_similarity                                             #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_sim_fm_search_functions_by_similarity_bounds_and_shape():
    fm = SimulatedFunctionManager()
    n = 2
    sims = fm.search_functions_by_similarity(query="add numbers", n=n)
    assert isinstance(sims, list) and 1 <= len(sims) <= n
    first = sims[0]
    assert isinstance(first, dict)
    for key in ("name", "function_id", "argspec", "score"):
        assert key in first
