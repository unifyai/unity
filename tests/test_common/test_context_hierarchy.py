"""
Tests for context hierarchy and aggregation context derivation.

This module tests that:
1. Production contexts derive correct aggregation contexts
2. Test contexts derive aggregation contexts scoped to the test root
3. The @_handle_project decorator sets up proper context hierarchy
"""

from unity.common.log_utils import _derive_all_contexts
from unity.session_details import DEFAULT_USER_CONTEXT

# =============================================================================
# Tests for _derive_all_contexts - Production contexts
# =============================================================================


def test_derive_all_contexts_production_simple():
    """Production context should derive standard aggregation contexts."""
    context = "JohnDoe/MyAssistant/Contacts"
    result = _derive_all_contexts(context)

    assert result == [
        "JohnDoe/All/Contacts",
        "All/Contacts",
    ]


def test_derive_all_contexts_production_nested_suffix():
    """Production context with nested suffix should work correctly."""
    context = "JohnDoe/MyAssistant/Knowledge/Sales"
    result = _derive_all_contexts(context)

    assert result == [
        "JohnDoe/All/Knowledge/Sales",
        "All/Knowledge/Sales",
    ]


def test_derive_all_contexts_too_few_parts():
    """Contexts with fewer than 3 parts should return empty list."""
    assert _derive_all_contexts("Contacts") == []
    assert _derive_all_contexts("JohnDoe/Contacts") == []


# =============================================================================
# Tests for _derive_all_contexts - Test contexts
# =============================================================================


def test_derive_all_contexts_test_simple():
    """Test context should derive aggregation contexts scoped to test root."""
    context = f"tests/test_foo/{DEFAULT_USER_CONTEXT}/Assistant/Contacts"
    result = _derive_all_contexts(context)

    assert result == [
        f"tests/test_foo/{DEFAULT_USER_CONTEXT}/All/Contacts",
        "tests/test_foo/All/Contacts",
    ]


def test_derive_all_contexts_test_nested_path():
    """Test context with nested test path should scope correctly."""
    context = f"tests/test_contact_manager/test_all_ctx/test_my_test/{DEFAULT_USER_CONTEXT}/Assistant/Contacts"
    result = _derive_all_contexts(context)

    assert result == [
        f"tests/test_contact_manager/test_all_ctx/test_my_test/{DEFAULT_USER_CONTEXT}/All/Contacts",
        "tests/test_contact_manager/test_all_ctx/test_my_test/All/Contacts",
    ]


def test_derive_all_contexts_test_nested_suffix():
    """Test context with nested suffix should work correctly."""
    context = f"tests/test_foo/{DEFAULT_USER_CONTEXT}/Assistant/Knowledge/Sales"
    result = _derive_all_contexts(context)

    assert result == [
        f"tests/test_foo/{DEFAULT_USER_CONTEXT}/All/Knowledge/Sales",
        "tests/test_foo/All/Knowledge/Sales",
    ]


def test_derive_all_contexts_test_without_default_user():
    """Test context without DefaultUser marker should return empty list."""
    # This shouldn't happen with @_handle_project, but test the fallback
    context = "tests/test_foo/SomeUser/Assistant/Contacts"
    result = _derive_all_contexts(context)

    # SomeUser is not the DEFAULT_USER_CONTEXT marker, so can't determine structure
    assert result == []


def test_derive_all_contexts_test_insufficient_parts_after_user():
    """Test context without enough parts after User should return empty list."""
    # Only User, no Assistant or Suffix
    context = f"tests/test_foo/{DEFAULT_USER_CONTEXT}/Assistant"
    result = _derive_all_contexts(context)

    # Need User/Assistant/Suffix (3 parts minimum after test_root)
    assert result == []


# =============================================================================
# Tests for context hierarchy structure
# =============================================================================


def test_production_context_structure():
    """Verify production context hierarchy structure."""
    user = "JohnDoe"
    assistant = "MyAssistant"
    manager_ctx = "Contacts"

    # The full context path
    full_ctx = f"{user}/{assistant}/{manager_ctx}"

    # Derived aggregation contexts
    aggregation = _derive_all_contexts(full_ctx)

    # Should have exactly 2 aggregation contexts
    assert len(aggregation) == 2

    # User-level: all assistants for this user
    assert aggregation[0] == f"{user}/All/{manager_ctx}"

    # Global: all users
    assert aggregation[1] == f"All/{manager_ctx}"


def test_test_context_structure():
    """Verify test context hierarchy structure matches production pattern."""
    test_root = "tests/test_contact_manager/test_all_ctx/test_foo"
    user = DEFAULT_USER_CONTEXT
    assistant = "Assistant"
    manager_ctx = "Contacts"

    # The full context path (as created by @_handle_project)
    full_ctx = f"{test_root}/{user}/{assistant}/{manager_ctx}"

    # Derived aggregation contexts
    aggregation = _derive_all_contexts(full_ctx)

    # Should have exactly 2 aggregation contexts
    assert len(aggregation) == 2

    # User-level: all assistants for this user (scoped to test root)
    assert aggregation[0] == f"{test_root}/{user}/All/{manager_ctx}"

    # Global: all users (scoped to test root)
    assert aggregation[1] == f"{test_root}/All/{manager_ctx}"


def test_test_context_isolation():
    """Two different tests should have completely isolated contexts."""
    test_root_a = "tests/test_foo/test_a"
    test_root_b = "tests/test_foo/test_b"

    ctx_a = f"{test_root_a}/{DEFAULT_USER_CONTEXT}/Assistant/Contacts"
    ctx_b = f"{test_root_b}/{DEFAULT_USER_CONTEXT}/Assistant/Contacts"

    aggregation_a = _derive_all_contexts(ctx_a)
    aggregation_b = _derive_all_contexts(ctx_b)

    # Each test should have its own aggregation contexts
    assert aggregation_a[0] != aggregation_b[0]
    assert aggregation_a[1] != aggregation_b[1]

    # Verify the test roots are preserved
    assert test_root_a in aggregation_a[0]
    assert test_root_a in aggregation_a[1]
    assert test_root_b in aggregation_b[0]
    assert test_root_b in aggregation_b[1]
