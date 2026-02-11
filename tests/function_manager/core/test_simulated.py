from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.function_manager.simulated import SimulatedFunctionManager

# --------------------------------------------------------------------------- #
#  Doc-string inheritance                                                     #
# --------------------------------------------------------------------------- #


def test_docstrings_match_base():
    """
    Public methods in SimulatedFunctionManager should copy the real
    BaseFunctionManager doc-strings one-for-one (via functools.wraps).
    """
    from unity.function_manager.base import BaseFunctionManager
    from unity.function_manager.simulated import SimulatedFunctionManager

    assert (
        BaseFunctionManager.add_functions.__doc__.strip()
        in SimulatedFunctionManager.add_functions.__doc__.strip()
    ), ".add_functions doc-string was not copied correctly"

    assert (
        BaseFunctionManager.list_functions.__doc__.strip()
        in SimulatedFunctionManager.list_functions.__doc__.strip()
    ), ".list_functions doc-string was not copied correctly"


# --------------------------------------------------------------------------- #
#  add_functions                                                              #
# --------------------------------------------------------------------------- #


@_handle_project
def test_add_functions_accepts_single_and_multiple():
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
def test_list_functions_with_and_without_implementations():
    fm = SimulatedFunctionManager()

    listing = fm.list_functions()
    assert isinstance(listing, dict) and listing
    # pick a representative entry and verify shape without relying on specific names
    first_name, first_entry = next(iter(listing.items()))
    assert isinstance(first_entry, dict)
    for key in ("function_id", "argspec", "docstring"):
        assert key in first_entry
    assert "implementation" not in first_entry

    listing_with_impl = fm.list_functions(include_implementations=True)
    assert isinstance(listing_with_impl, dict) and listing_with_impl
    # Ensure the same entry now includes an implementation when requested
    assert first_name in listing_with_impl
    assert "implementation" in listing_with_impl[first_name]


# --------------------------------------------------------------------------- #
#  get_precondition                                                           #
# --------------------------------------------------------------------------- #


@_handle_project
def test_get_precondition_returns_none():
    fm = SimulatedFunctionManager()
    assert fm.get_precondition(function_name="does_not_matter") is None


# --------------------------------------------------------------------------- #
#  delete_function                                                            #
# --------------------------------------------------------------------------- #


@_handle_project
def test_delete_function_acknowledges():
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
def test_search_functions_returns_list_of_dicts():
    fm = SimulatedFunctionManager()
    hits = fm.filter_functions(filter="'price' in docstring")
    assert isinstance(hits, list) and hits
    first = hits[0]
    assert isinstance(first, dict)
    for key in ("name", "function_id", "argspec", "docstring"):
        assert key in first


# --------------------------------------------------------------------------- #
#  search_functions                                             #
# --------------------------------------------------------------------------- #


@_handle_project
def test_search_functions_bounds_and_shape():
    fm = SimulatedFunctionManager()
    n = 2
    sims = fm.search_functions(query="add numbers", n=n)
    assert isinstance(sims, list) and 1 <= len(sims) <= n
    first = sims[0]
    assert isinstance(first, dict)
    for key in ("name", "function_id", "argspec", "score"):
        assert key in first


# --------------------------------------------------------------------------- #
#  execute_function                                                           #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_returns_simulated_response():
    """
    SimulatedFunctionManager.execute_function should return a simulated
    response with the expected structure (result, error, stdout, stderr).
    """
    fm = SimulatedFunctionManager()
    response = await fm.execute_function(
        function_name="my_function",
        call_kwargs={"x": 1, "y": 2},
    )
    assert isinstance(response, dict)
    # Check required keys in response
    assert "result" in response
    assert "error" in response
    assert "stdout" in response
    assert "stderr" in response
    # Error should be None for successful simulation
    assert response["error"] is None
    # Result should contain simulated marker and function info
    result = response["result"]
    assert isinstance(result, dict)
    assert result.get("simulated") is True
    assert result.get("function_name") == "my_function"
    assert result.get("call_kwargs") == {"x": 1, "y": 2}


# --------------------------------------------------------------------------- #
#  clear                                                                      #
# --------------------------------------------------------------------------- #


@_handle_project
def test_clear_sync():
    """
    SimulatedFunctionManager.clear should reset the manager (hard-coded completion)
    and remain usable afterwards.
    """
    fm = SimulatedFunctionManager()
    # Do a synchronous operation to create some prior state
    fm.add_functions(implementations="def _tmp():\n    return 1\n")
    # Clear should not raise and should be quick (no LLM roundtrip requirement)
    fm.clear()
    # Post-clear, read-only operations should still work
    post = fm.list_functions()
    assert isinstance(post, dict) and post is not None


# --------------------------------------------------------------------------- #
#  filter_scope                                                                #
# --------------------------------------------------------------------------- #


pytestmark_eval = pytest.mark.eval


@pytestmark_eval
@_handle_project
def test_filter_scope_respected_in_list_functions():
    """
    A SimulatedFunctionManager with filter_scope="language == 'python'" should
    only return Python functions from list_functions.
    """
    fm = SimulatedFunctionManager(
        description="A mixed catalogue of Python and bash utility functions",
        filter_scope="language == 'python'",
    )
    listing = fm.list_functions()
    assert isinstance(listing, dict) and listing
    # Every entry should report language as 'python' (or omit it, defaulting to python)
    for name, meta in listing.items():
        lang = meta.get("language", "python")
        assert (
            lang == "python"
        ), f"filter_scope='language == \"python\"' but {name!r} has language={lang!r}"


# --------------------------------------------------------------------------- #
#  Setter contract (shared with FunctionManager)                              #
# --------------------------------------------------------------------------- #


def test_filter_scope_setter():
    """filter_scope can be set after construction."""
    fm = SimulatedFunctionManager()
    assert fm.filter_scope is None

    fm.filter_scope = "language == 'python'"
    assert fm.filter_scope == "language == 'python'"

    fm.filter_scope = None
    assert fm.filter_scope is None


def test_exclude_primitive_ids_setter():
    """exclude_primitive_ids can be set after construction."""
    fm = SimulatedFunctionManager()
    assert fm.exclude_primitive_ids is None

    fm.exclude_primitive_ids = frozenset({1, 2, 3})
    assert fm.exclude_primitive_ids == frozenset({1, 2, 3})

    fm.exclude_primitive_ids = None
    assert fm.exclude_primitive_ids is None


def test_exclude_compositional_ids_setter():
    """exclude_compositional_ids can be set after construction."""
    fm = SimulatedFunctionManager()
    assert fm.exclude_compositional_ids is None

    fm.exclude_compositional_ids = frozenset({10, 20})
    assert fm.exclude_compositional_ids == frozenset({10, 20})

    fm.exclude_compositional_ids = None
    assert fm.exclude_compositional_ids is None


def test_setters_do_not_crash_when_called_from_code_act_actor():
    """SimulatedFunctionManager should survive CodeActActor.__init__ setter calls.

    CodeActActor.__init__ collects function_ids from environments and sets
    them on the FM via setters. This must not crash for SimulatedFunctionManager.
    """
    from unity.actor.code_act_actor import CodeActActor
    from unity.actor.environments.state_managers import StateManagerEnvironment

    fm = SimulatedFunctionManager(description="test")

    # Constructing a CodeActActor with environments that have function_ids
    # should set exclusions on the FM via setters, not replace it.
    actor = CodeActActor(
        environments=[StateManagerEnvironment()],
        function_manager=fm,
        timeout=30,
    )

    # The FM instance should be the SAME object (not replaced).
    assert actor.function_manager is fm

    # Exclusion IDs should have been set via the setter.
    assert fm.exclude_primitive_ids is not None
    assert len(fm.exclude_primitive_ids) > 0


def test_setters_update_system_message():
    """Setting exclusions should update the LLM system message."""
    fm = SimulatedFunctionManager()
    original_sys = fm._llm.system_message

    fm.exclude_primitive_ids = frozenset({42})
    updated_sys = fm._llm.system_message

    assert original_sys != updated_sys
    assert "42" in updated_sys
    assert "excluded" in updated_sys.lower()
