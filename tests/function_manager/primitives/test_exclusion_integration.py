"""
Integration tests for FunctionManager environment exclusion.

Verifies that when exclude_primitive_ids is set on a FunctionManager instance,
the excluded primitives do NOT appear in search_functions, list_functions,
filter_functions, or list_primitives results -- using real backend queries.
"""

import pytest

from unity.function_manager.function_manager import FunctionManager
from unity.function_manager.primitives import Primitives, PrimitiveScope, get_registry
from unity.actor.environments.state_managers import StateManagerEnvironment
from unity.common.context_registry import ContextRegistry
from tests.helpers import _handle_project

# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def fm_factory():
    """Factory that creates FunctionManager instances with context cleanup."""
    managers = []

    def _create(**kwargs):
        ContextRegistry.forget(FunctionManager, "Functions/VirtualEnvs")
        ContextRegistry.forget(FunctionManager, "Functions/Compositional")
        ContextRegistry.forget(FunctionManager, "Functions/Primitives")
        ContextRegistry.forget(FunctionManager, "Functions/Meta")
        fm = FunctionManager(**kwargs)
        managers.append(fm)
        return fm

    yield _create

    for fm in managers:
        try:
            fm.clear()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

_CONTACTS_SCOPE = PrimitiveScope(
    scoped_managers=frozenset({"contacts", "files"}),
)


def _get_contacts_ask_id() -> int:
    """Get the stable function_id for primitives.contacts.ask."""
    return get_registry().get_function_id("contacts", "ask")


def _get_contacts_update_id() -> int:
    """Get the stable function_id for primitives.contacts.update."""
    return get_registry().get_function_id("contacts", "update")


# ────────────────────────────────────────────────────────────────────────────
# 1. list_primitives exclusion
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_list_primitives_excludes_tagged_ids(fm_factory):
    """list_primitives() should not return primitives whose IDs are excluded."""
    contacts_ask_id = _get_contacts_ask_id()

    # Baseline: unexcluded FM sees contacts.ask
    fm_all = fm_factory(primitive_scope=_CONTACTS_SCOPE)
    fm_all.sync_primitives()
    prims_all = fm_all.list_primitives()
    assert (
        "primitives.contacts.ask" in prims_all
    ), "Baseline: contacts.ask should be visible"

    # Excluded FM should NOT see contacts.ask
    fm_excl = fm_factory(
        primitive_scope=_CONTACTS_SCOPE,
        exclude_primitive_ids=frozenset({contacts_ask_id}),
    )
    fm_excl.sync_primitives()
    prims_excl = fm_excl.list_primitives()
    assert (
        "primitives.contacts.ask" not in prims_excl
    ), "contacts.ask should be excluded from list_primitives"
    # Other primitives should still be visible
    assert (
        "primitives.contacts.update" in prims_excl
    ), "contacts.update should still be visible"


# ────────────────────────────────────────────────────────────────────────────
# 2. list_functions exclusion
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_list_functions_excludes_tagged_primitive_ids(fm_factory):
    """list_functions() should not return primitives whose IDs are excluded."""
    contacts_ask_id = _get_contacts_ask_id()

    # Baseline
    fm_all = fm_factory(primitive_scope=_CONTACTS_SCOPE)
    fm_all.sync_primitives()
    listing_all = fm_all.list_functions()
    assert "primitives.contacts.ask" in listing_all

    # Excluded
    fm_excl = fm_factory(
        primitive_scope=_CONTACTS_SCOPE,
        exclude_primitive_ids=frozenset({contacts_ask_id}),
    )
    fm_excl.sync_primitives()
    listing_excl = fm_excl.list_functions()
    assert "primitives.contacts.ask" not in listing_excl
    assert "primitives.contacts.update" in listing_excl


# ────────────────────────────────────────────────────────────────────────────
# 3. search_functions exclusion
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_search_functions_excludes_tagged_primitive_ids(fm_factory):
    """search_functions() should not return primitives whose IDs are excluded."""
    contacts_ask_id = _get_contacts_ask_id()

    # Baseline: search should find contacts.ask
    fm_all = fm_factory(primitive_scope=_CONTACTS_SCOPE)
    fm_all.sync_primitives()
    hits_all = fm_all.search_functions(query="ask about contacts", n=20)
    names_all = {h["name"] for h in hits_all}
    assert (
        "primitives.contacts.ask" in names_all
    ), "Baseline: search should find contacts.ask"

    # Excluded: search should NOT find contacts.ask
    fm_excl = fm_factory(
        primitive_scope=_CONTACTS_SCOPE,
        exclude_primitive_ids=frozenset({contacts_ask_id}),
    )
    fm_excl.sync_primitives()
    hits_excl = fm_excl.search_functions(query="ask about contacts", n=20)
    names_excl = {h["name"] for h in hits_excl}
    assert (
        "primitives.contacts.ask" not in names_excl
    ), "contacts.ask should be excluded from search_functions"


# ────────────────────────────────────────────────────────────────────────────
# 4. filter_functions exclusion
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_filter_functions_excludes_tagged_primitive_ids(fm_factory):
    """filter_functions() should not return primitives whose IDs are excluded."""
    contacts_ask_id = _get_contacts_ask_id()

    # Baseline
    fm_all = fm_factory(primitive_scope=_CONTACTS_SCOPE)
    fm_all.sync_primitives()
    hits_all = fm_all.filter_functions(filter="is_primitive == True")
    names_all = {h["name"] for h in hits_all}
    assert "primitives.contacts.ask" in names_all

    # Excluded
    fm_excl = fm_factory(
        primitive_scope=_CONTACTS_SCOPE,
        exclude_primitive_ids=frozenset({contacts_ask_id}),
    )
    fm_excl.sync_primitives()
    hits_excl = fm_excl.filter_functions(filter="is_primitive == True")
    names_excl = {h["name"] for h in hits_excl}
    assert "primitives.contacts.ask" not in names_excl
    assert "primitives.contacts.update" in names_excl


@_handle_project
def test_filter_functions_handles_production_sized_primitive_exclusions(fm_factory):
    """Large primitive exclusion sets should not trigger backend recursion."""
    production_like_ids = frozenset(range(1000, 1120))

    fm = fm_factory(
        primitive_scope=PrimitiveScope.all_managers(),
        exclude_primitive_ids=production_like_ids,
    )
    fm.sync_primitives()

    hits = fm.filter_functions(
        filter="is_primitive == True",
        limit=5,
        include_implementations=False,
    )

    assert isinstance(hits, list)


# ────────────────────────────────────────────────────────────────────────────
# 5. Multiple IDs excluded at once
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_multiple_primitives_excluded(fm_factory):
    """Excluding multiple primitive IDs hides all of them."""
    contacts_ask_id = _get_contacts_ask_id()
    contacts_update_id = _get_contacts_update_id()

    fm_excl = fm_factory(
        primitive_scope=_CONTACTS_SCOPE,
        exclude_primitive_ids=frozenset({contacts_ask_id, contacts_update_id}),
    )
    fm_excl.sync_primitives()
    listing = fm_excl.list_functions()

    assert "primitives.contacts.ask" not in listing
    assert "primitives.contacts.update" not in listing
    # files primitives should still be visible
    file_primitives = [n for n in listing if n.startswith("primitives.files.")]
    assert len(file_primitives) > 0, "files primitives should still be visible"


# ────────────────────────────────────────────────────────────────────────────
# 6. Environment-driven exclusion (CodeActActor path)
# ────────────────────────────────────────────────────────────────────────────


def test_environment_function_ids_match_exclusion_targets():
    """The function_ids tagged on StateManagerEnvironment tools should correspond
    to actual primitives in the FunctionManager, ensuring the exclusion targets
    the right rows."""
    registry = get_registry()
    scope = PrimitiveScope(scoped_managers=frozenset({"contacts"}))

    # Get IDs from environment
    env = StateManagerEnvironment(Primitives(primitive_scope=scope))
    env_ids = {
        meta.function_id
        for meta in env.get_tools().values()
        if meta.function_id is not None
    }

    # Get IDs from collect_primitives (same source as sync_primitives)
    collected = registry.collect_primitives(scope)
    collected_ids = {row["function_id"] for row in collected.values()}

    # They should be identical
    assert (
        env_ids == collected_ids
    ), f"Environment IDs {env_ids} should match collected primitive IDs {collected_ids}"
