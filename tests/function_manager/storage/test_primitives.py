"""
Tests for action primitives in FunctionManager.

Tests the primitives registry, sync mechanism, and semantic search
that includes both user-defined functions and action primitives.

Primitives are stored in a separate context (Functions/Primitives) with
stable hash-based function_id values, while user-defined functions are in
Functions/Compositional with auto-incrementing IDs.
"""

import asyncio

import pytest

from unity.function_manager.function_manager import FunctionManager
from unity.function_manager.primitives import (
    Primitives,
    _AsyncPrimitiveWrapper,
    _create_async_wrapper,
    get_registry,
)
from unity.common.context_registry import ContextRegistry
from tests.helpers import _handle_project

# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def function_manager_factory():
    """
    Factory fixture that creates FunctionManager instances.

    Returns a callable that creates a FunctionManager. This ensures the
    FunctionManager is instantiated AFTER @_handle_project sets up the
    test-specific context, providing proper isolation for parallel tests.
    """
    managers = []

    def _create():
        # Forget FunctionManager's cached contexts to ensure we get
        # fresh contexts for this test's active context (set by @_handle_project)
        ContextRegistry.forget(FunctionManager, "Functions/VirtualEnvs")
        ContextRegistry.forget(FunctionManager, "Functions/Compositional")
        ContextRegistry.forget(FunctionManager, "Functions/Primitives")
        ContextRegistry.forget(FunctionManager, "Functions/Meta")
        fm = FunctionManager()
        managers.append(fm)
        return fm

    yield _create

    # Cleanup all created managers
    for fm in managers:
        try:
            fm.clear()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# 1. Primitives collection tests
# ────────────────────────────────────────────────────────────────────────────


def test_collect_primitives_returns_expected_methods():
    """Registry should return metadata for all auto-discovered methods."""
    from unity.function_manager.primitives import PrimitiveScope
    from unity.function_manager.primitives.registry import get_primitive_sources

    registry = get_registry()
    scope = PrimitiveScope.all_managers()
    primitives = registry.collect_primitives(scope)

    # Should have collected at least some primitives
    assert len(primitives) > 0

    # Verify primitives include expected manager classes (using primitive_class)
    classes_found = set(p["primitive_class"] for p in primitives.values())
    assert any("ContactManager" in c for c in classes_found)
    assert any("FileManager" in c for c in classes_found)
    assert any("DataManager" in c for c in classes_found)

    # Verify primitives match what get_primitive_sources returns
    # (i.e., the auto-discovery is working correctly)
    for cls, method_names in get_primitive_sources():
        class_name = cls.__name__
        for method_name in method_names:
            qualified_name = f"{class_name}.{method_name}"
            assert (
                qualified_name in primitives
            ), f"Expected auto-discovered primitive '{qualified_name}' not found"


def test_collect_primitives_has_required_fields():
    """Each primitive should have the required metadata fields including function_id."""
    from unity.function_manager.primitives import PrimitiveScope

    registry = get_registry()
    scope = PrimitiveScope.all_managers()
    primitives = registry.collect_primitives(scope)

    for data in primitives.values():
        assert "name" in data
        assert "argspec" in data
        assert "docstring" in data
        assert "embedding_text" in data
        assert data.get("is_primitive") is True
        assert "primitive_class" in data
        assert "primitive_method" in data
        # Primitives have explicit integer function_ids
        assert "function_id" in data
        assert isinstance(data["function_id"], int)


def test_collect_primitives_has_stable_ids():
    """Primitive function_ids should be stable hash-based IDs."""
    from unity.function_manager.primitives import PrimitiveScope

    registry = get_registry()
    scope = PrimitiveScope.all_managers()
    primitives = registry.collect_primitives(scope)

    # Verify no duplicate IDs
    ids = [p["function_id"] for p in primitives.values()]
    assert len(ids) == len(set(ids)), "Primitive IDs should be unique"

    # Verify IDs are deterministic (calling twice gives same IDs)
    primitives2 = registry.collect_primitives(scope)
    for name, data in primitives.items():
        assert (
            primitives2[name]["function_id"] == data["function_id"]
        ), f"ID for '{name}' should be stable across calls"

    # Verify IDs are non-negative integers (hash-based)
    for data in primitives.values():
        assert isinstance(data["function_id"], int)
        assert data["function_id"] >= 0


def test_collect_primitives_has_docstrings():
    """Primitives should have non-empty docstrings (from base class)."""
    from unity.function_manager.primitives import PrimitiveScope

    registry = get_registry()
    scope = PrimitiveScope.all_managers()
    primitives = registry.collect_primitives(scope)

    # At least some primitives should have docstrings
    with_docstrings = [
        name for name, p in primitives.items() if p.get("docstring", "").strip()
    ]
    assert (
        len(with_docstrings) > 0
    ), "Expected at least some primitives to have docstrings"


def test_compute_primitives_hash_is_stable():
    """Hash should be deterministic for the same primitives."""
    from unity.function_manager.primitives import PrimitiveScope

    registry = get_registry()
    scope = PrimitiveScope.all_managers()

    hash1 = registry.compute_primitives_hash(primitive_scope=scope)
    hash2 = registry.compute_primitives_hash(primitive_scope=scope)

    assert hash1 == hash2
    assert len(hash1) == 16  # 16 hex chars


def test_compute_primitives_hash_changes_for_different_scopes():
    """Hash should be different for different scopes."""
    from unity.function_manager.primitives import PrimitiveScope

    registry = get_registry()
    scope_all = PrimitiveScope.all_managers()
    scope_files = PrimitiveScope.single("files")

    hash_all = registry.compute_primitives_hash(primitive_scope=scope_all)
    hash_files = registry.compute_primitives_hash(primitive_scope=scope_files)

    assert hash_all != hash_files


def test_compute_primitives_hash_changes_on_modification():
    """Hash should change when primitives are modified."""
    from unity.function_manager.primitives import PrimitiveScope

    registry = get_registry()
    scope = PrimitiveScope.single("files")
    primitives = registry.collect_primitives(scope)

    # Compute original hash
    original_hash = registry.compute_primitives_hash(primitives=primitives)

    # Modify a primitive's docstring
    first_name = next(iter(primitives.keys()))
    modified_primitives = dict(primitives)
    modified_primitives[first_name] = dict(modified_primitives[first_name])
    modified_primitives[first_name]["docstring"] = "MODIFIED DOCSTRING FOR TESTING"

    # Hash should change
    modified_hash = registry.compute_primitives_hash(primitives=modified_primitives)
    assert (
        original_hash != modified_hash
    ), "Hash should change when primitives are modified"


def test_collect_primitives_includes_file_manager():
    """FileManager primitives should be collected from auto-discovery."""
    from unity.function_manager.primitives import PrimitiveScope

    registry = get_registry()
    scope = PrimitiveScope.single("files")
    primitives = registry.collect_primitives(scope)

    assert len(primitives) >= 5, "Expected at least 5 FileManager primitives"

    # Verify all are from files manager (using primitive_class)
    for name, p in primitives.items():
        assert "FileManager" in p["primitive_class"]
        # Old format: "FileManager.method", not "primitives.files.method"
        assert p["name"].startswith("FileManager.")


# ────────────────────────────────────────────────────────────────────────────
# 2. FunctionManager primitives sync tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_sync_primitives_inserts_rows(function_manager_factory):
    """sync_primitives() should insert primitive rows into the Primitives context."""
    function_manager = function_manager_factory()

    # Initially no primitives
    primitives_before = function_manager.list_primitives()

    # Sync primitives
    did_sync = function_manager.sync_primitives()
    assert did_sync is True

    # Now should have primitives with integer IDs
    primitives_after = function_manager.list_primitives()
    assert len(primitives_after) > 0

    # Verify they have integer function_ids
    for name, data in primitives_after.items():
        assert isinstance(data["function_id"], int)


@_handle_project
def test_sync_primitives_is_idempotent(function_manager_factory):
    """Calling sync_primitives() twice should not duplicate rows."""
    function_manager = function_manager_factory()

    # First sync
    function_manager.sync_primitives()
    count1 = len(function_manager.list_primitives())

    # Reset the session flag to force re-check
    function_manager._primitives_synced = False

    # Second sync (should be no-op since hash matches)
    did_sync = function_manager.sync_primitives()
    assert did_sync is False  # No sync needed

    count2 = len(function_manager.list_primitives())
    assert count1 == count2


@_handle_project
def test_list_primitives_returns_primitive_metadata(function_manager_factory):
    """list_primitives() should return primitive metadata with integer function_ids."""
    function_manager = function_manager_factory()

    function_manager.sync_primitives()
    primitives = function_manager.list_primitives()

    for name, data in primitives.items():
        assert data.get("is_primitive") is True
        assert "argspec" in data
        assert "docstring" in data
        # Verify function_id is an integer (not None)
        assert "function_id" in data
        assert isinstance(data["function_id"], int)


@_handle_project
def test_primitives_have_stable_ids_in_database(function_manager_factory):
    """Primitives stored in database should have consistent IDs across syncs."""
    function_manager = function_manager_factory()

    # First sync
    function_manager.sync_primitives()
    first_sync = function_manager.list_primitives()

    # Clear and re-sync
    function_manager.clear()
    function_manager.sync_primitives()
    second_sync = function_manager.list_primitives()

    # IDs should be identical
    for name in first_sync:
        assert name in second_sync, f"Primitive {name} missing after re-sync"
        assert first_sync[name]["function_id"] == second_sync[name]["function_id"], (
            f"Primitive {name} ID changed from {first_sync[name]['function_id']} "
            f"to {second_sync[name]['function_id']}"
        )


# ────────────────────────────────────────────────────────────────────────────
# 3. Semantic search with primitives tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_search_includes_primitives_by_default(function_manager_factory):
    """search_functions should include primitives by default."""
    function_manager = function_manager_factory()

    # Search for something that should match a primitive
    results = function_manager.search_functions(
        query="ask a question to the contact manager",
        n=5,
    )

    # Should have results (primitives get synced automatically)
    assert len(results) > 0

    # At least one result should be a primitive
    has_primitive = any(r.get("is_primitive") for r in results)
    assert has_primitive, "Expected at least one primitive in search results"


@_handle_project
def test_search_can_exclude_primitives(function_manager_factory):
    """search_functions can exclude primitives."""
    function_manager = function_manager_factory()

    # First sync primitives so they exist
    function_manager.sync_primitives()

    # Add a user function
    implementation = '''
def ask_contact_question(question: str) -> str:
    """Ask a question about contacts."""
    return f"Asked: {question}"
'''
    function_manager.add_functions(implementations=[implementation])

    # Search excluding primitives
    results = function_manager.search_functions(
        query="ask question about contacts",
        n=10,
        include_primitives=False,
    )

    # Should not have any primitives
    has_primitive = any(r.get("is_primitive") for r in results)
    assert not has_primitive, "Expected no primitives when include_primitives=False"


@_handle_project
def test_search_ranks_functions_and_primitives_together(function_manager_factory):
    """Search should return both functions and primitives ranked by relevance."""
    function_manager = function_manager_factory()

    # Add a user function related to contacts
    implementation = '''
def update_contact_email(contact_id: int, email: str) -> str:
    """Update a contact's email address."""
    return f"Updated contact {contact_id} email to {email}"
'''
    function_manager.add_functions(implementations=[implementation])

    # Search for contact-related functionality
    results = function_manager.search_functions(
        query="update contact information",
        n=10,
    )

    # Should have both user functions and primitives
    user_funcs = [r for r in results if not r.get("is_primitive")]
    primitives = [r for r in results if r.get("is_primitive")]

    assert len(user_funcs) > 0, "Expected at least one user function"
    assert len(primitives) > 0, "Expected at least one primitive"


# ────────────────────────────────────────────────────────────────────────────
# 4. Clear and re-sync tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_clear_removes_primitives(function_manager_factory):
    """clear() should remove primitives along with user functions."""
    function_manager = function_manager_factory()

    function_manager.sync_primitives()
    assert len(function_manager.list_primitives()) > 0

    function_manager.clear()

    # After clear, primitives should be gone (until next sync)
    # Note: list_primitives doesn't trigger sync, so should be empty
    primitives = function_manager.list_primitives()
    assert len(primitives) == 0


@_handle_project
def test_sync_after_clear_restores_primitives(function_manager_factory):
    """Syncing after clear should restore primitives."""
    function_manager = function_manager_factory()

    function_manager.sync_primitives()
    count_before = len(function_manager.list_primitives())

    function_manager.clear()
    function_manager.sync_primitives()

    count_after = len(function_manager.list_primitives())
    assert count_after == count_before


# ────────────────────────────────────────────────────────────────────────────
# 5. Async patching tests
# ────────────────────────────────────────────────────────────────────────────


def test_manager_spec_has_excluded_methods():
    """ManagerSpec entries should have excluded_methods."""
    registry = get_registry()

    for spec in registry.MANAGERS:
        # excluded_methods should be a frozenset
        assert isinstance(
            spec.excluded_methods,
            frozenset,
        ), f"{spec.manager_alias} excluded_methods should be frozenset"


def test_common_excluded_methods():
    """Common excluded methods should include lifecycle and internal helpers."""
    from unity.function_manager.primitives.registry import _COMMON_EXCLUDED_METHODS

    assert "clear" in _COMMON_EXCLUDED_METHODS
    assert "add_tools" in _COMMON_EXCLUDED_METHODS
    assert "get_tools" in _COMMON_EXCLUDED_METHODS


def test_primitive_methods_respects_exclusions():
    """primitive_methods should exclude methods in config."""
    registry = get_registry()
    methods = registry.primitive_methods(manager_alias="contacts")

    # Should not include excluded methods
    assert "filter_contacts" not in methods
    assert "update_contact" not in methods
    assert "clear" not in methods  # Common exclusion

    # Should include public methods
    assert "ask" in methods
    assert "update" in methods


def test_async_wrapper_auto_detects_sync_methods():
    """Wrapper should auto-detect sync methods without config."""
    from unity.manager_registry import ManagerRegistry

    dm = ManagerRegistry.get_data_manager()

    # Before wrapping, filter is sync
    original_filter = dm.filter
    is_originally_sync = not asyncio.iscoroutinefunction(original_filter)

    # Create wrapper (using manager alias)
    wrapper = _create_async_wrapper(dm, "data")

    # After wrapping, filter should be async
    assert asyncio.iscoroutinefunction(
        wrapper.filter,
    ), "filter should be async after wrapping"

    # Original should remain sync
    assert not asyncio.iscoroutinefunction(
        dm.filter,
    ), "Original filter should remain sync"

    # Verify it was originally sync (this confirms auto-detection worked)
    assert is_originally_sync, "filter should have been sync before wrapping"


def test_async_patching_preserves_docstrings():
    """Patched methods should preserve their original docstrings."""
    primitives = Primitives()
    dm = primitives.data

    # Patched method should have a docstring
    assert dm.filter.__doc__ is not None, "Patched method should have docstring"
    assert len(dm.filter.__doc__) > 0, "Docstring should not be empty"


def test_async_patching_preserves_signatures():
    """Patched methods should preserve their original signatures."""
    import inspect

    primitives = Primitives()
    dm = primitives.data

    # Patched method should have a signature
    sig = inspect.signature(dm.filter)
    assert sig is not None, "Patched method should have signature"

    # Should have expected parameters
    params = list(sig.parameters.keys())
    assert "context" in params, "filter should have 'context' parameter"


def test_primitives_data_is_async_wrapper():
    """primitives.data should return an async wrapper around DataManager."""
    primitives = Primitives()
    dm = primitives.data

    assert isinstance(dm, _AsyncPrimitiveWrapper)


def test_primitives_files_is_async_wrapper():
    """primitives.files should return an async wrapper around FileManager."""
    primitives = Primitives()
    fm = primitives.files

    assert isinstance(fm, _AsyncPrimitiveWrapper)


def test_primitives_returns_async_wrapper():
    """primitives.data should return an async wrapper."""
    primitives = Primitives()

    # Access data
    dm = primitives.data

    # Should be a wrapper
    assert isinstance(dm, _AsyncPrimitiveWrapper), "Should return async wrapper"


def test_async_wrapper_preserves_async_methods():
    """Wrapper should preserve methods that are already async."""
    from unity.manager_registry import ManagerRegistry

    fm = ManagerRegistry.get_file_manager()

    # ask_about_file is already async
    original_ask = fm.ask_about_file
    assert asyncio.iscoroutinefunction(original_ask), "ask_about_file should be async"

    # Create wrapper (using manager alias)
    wrapper = _create_async_wrapper(fm, "files")

    # Wrapped method should also be async
    assert asyncio.iscoroutinefunction(
        wrapper.ask_about_file,
    ), "Wrapped method should be async"

    # Original should be unchanged
    assert asyncio.iscoroutinefunction(
        fm.ask_about_file,
    ), "Original should remain async"
