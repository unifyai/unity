"""
End-to-end tests for RPC with real primitives.

Unlike test_venv_rpc.py which uses mocks, these tests verify the complete chain:
venv_runner → RPC → FunctionManager → real state managers (ContactManager, etc.)

This validates that:
1. The RPC protocol works correctly with actual manager implementations
2. Data created in the main process is accessible via RPC from venv subprocesses
3. Results from real managers are correctly serialized back to the subprocess
"""

import shutil

import pytest

from unity.function_manager.function_manager import FunctionManager
from unity.function_manager.primitives import Primitives
from unity.contact_manager.contact_manager import ContactManager
from unity.common.context_registry import ContextRegistry
from tests.helpers import _handle_project

# Sample pyproject.toml with minimal dependencies
MINIMAL_VENV_CONTENT = """
[project]
name = "test-venv-e2e"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
""".strip()


# ────────────────────────────────────────────────────────────────────────────
# Test Functions that Use Real Primitives
# ────────────────────────────────────────────────────────────────────────────

# Function that calls primitives.contacts.filter_contacts (sync method)
FILTER_CONTACTS_FUNCTION = """
def get_all_contacts() -> dict:
    \"\"\"Get all contacts via RPC using filter_contacts.\"\"\"
    result = primitives.contacts.filter_contacts()
    return result
""".strip()

# Function that counts contacts
COUNT_CONTACTS_FUNCTION = """
def count_all_contacts() -> int:
    \"\"\"Count all contacts via RPC.\"\"\"
    result = primitives.contacts.filter_contacts()
    return len(result.get("contacts", []))
""".strip()

# Function that processes contact data
PROCESS_CONTACTS_FUNCTION = """
def count_contacts_with_email() -> int:
    \"\"\"Count contacts that have an email address.\"\"\"
    result = primitives.contacts.filter_contacts()
    contacts = result.get("contacts", [])
    count = 0
    for contact in contacts:
        if contact.get("email_address"):
            count += 1
    return count
""".strip()

# Function that makes multiple calls to same manager with different params
MULTI_CALL_FUNCTION = """
def get_contact_details() -> dict:
    \"\"\"Get multiple pieces of contact info via separate RPC calls.\"\"\"
    # First call: get all contacts
    all_contacts = primitives.contacts.filter_contacts()
    total_count = len(all_contacts.get("contacts", []))

    # Second call: filter for contacts with email
    # (filter_contacts accepts filter expressions)
    email_contacts = primitives.contacts.filter_contacts(
        filter="email_address != None"
    )
    email_count = len(email_contacts.get("contacts", []))

    return {
        "total_count": total_count,
        "email_count": email_count,
    }
""".strip()


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def function_manager_factory():
    """Factory fixture that creates FunctionManager instances."""
    managers = []

    def _create():
        ContextRegistry.forget(FunctionManager, "Functions/VirtualEnvs")
        ContextRegistry.forget(FunctionManager, "Functions/Compositional")
        ContextRegistry.forget(FunctionManager, "Functions/Primitives")
        ContextRegistry.forget(FunctionManager, "Functions/Meta")
        fm = FunctionManager()
        managers.append(fm)
        return fm

    yield _create

    for fm in managers:
        try:
            fm.clear()
        except Exception:
            pass


@pytest.fixture
def cleanup_venvs(function_manager_factory):
    """Cleanup venvs after test."""
    venv_dirs = []

    def _track(fm, venv_id):
        venv_dirs.append(fm._get_venv_dir(venv_id))

    yield _track

    for venv_dir in venv_dirs:
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────────
# End-to-End Tests with Real ContactManager
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_e2e_rpc_filter_contacts_returns_real_data(
    function_manager_factory,
    cleanup_venvs,
):
    """RPC to filter_contacts should return real contacts created in main process."""
    fm = function_manager_factory()
    cm = ContactManager()

    # Create real contacts in the main process
    contact1_id = cm._create_contact(
        first_name="Alice",
        surname="Smith",
        email_address="alice@example.com",
    )
    contact2_id = cm._create_contact(
        first_name="Bob",
        surname="Jones",
        phone_number="5551234",
    )

    # Set up venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    cleanup_venvs(fm, venv_id)

    # Create real Primitives (will lazily instantiate real ContactManager)
    primitives = Primitives()

    # Execute function in venv with RPC
    result = await fm.execute_in_venv(
        venv_id=venv_id,
        implementation=FILTER_CONTACTS_FUNCTION,
        call_kwargs={},
        is_async=False,
        primitives=primitives,
    )

    # Verify no errors
    assert result["error"] is None, f"Unexpected error: {result['error']}"

    # Verify we got real contact data
    contacts_data = result["result"]
    assert "contacts" in contacts_data
    contacts = contacts_data["contacts"]

    # Should have at least the 2 contacts we created
    assert len(contacts) >= 2

    # Verify our contacts are in the results
    first_names = [c.get("first_name") for c in contacts]
    assert "Alice" in first_names
    assert "Bob" in first_names


@_handle_project
@pytest.mark.asyncio
async def test_e2e_rpc_count_contacts(
    function_manager_factory,
    cleanup_venvs,
):
    """RPC to count contacts should return correct count."""
    fm = function_manager_factory()
    cm = ContactManager()

    # Create contacts
    cm._create_contact(first_name="Charlie", surname="Brown")
    cm._create_contact(first_name="Lucy", surname="Van Pelt")
    cm._create_contact(first_name="Linus", surname="Van Pelt")

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    cleanup_venvs(fm, venv_id)

    primitives = Primitives()

    result = await fm.execute_in_venv(
        venv_id=venv_id,
        implementation=COUNT_CONTACTS_FUNCTION,
        call_kwargs={},
        is_async=False,
        primitives=primitives,
    )

    assert result["error"] is None, f"Unexpected error: {result['error']}"

    # Should count at least 3 contacts
    assert result["result"] >= 3


@_handle_project
@pytest.mark.asyncio
async def test_e2e_rpc_process_contact_data_in_venv(
    function_manager_factory,
    cleanup_venvs,
):
    """Venv function should correctly process real contact data received via RPC."""
    fm = function_manager_factory()
    cm = ContactManager()

    # Create contacts - some with email, some without
    cm._create_contact(
        first_name="Emily",
        email_address="emily@example.com",
    )
    cm._create_contact(
        first_name="Frank",
        email_address="frank@example.com",
    )
    cm._create_contact(
        first_name="George",
        phone_number="5550000",
    )

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    cleanup_venvs(fm, venv_id)

    primitives = Primitives()

    result = await fm.execute_in_venv(
        venv_id=venv_id,
        implementation=PROCESS_CONTACTS_FUNCTION,
        call_kwargs={},
        is_async=False,
        primitives=primitives,
    )

    assert result["error"] is None, f"Unexpected error: {result['error']}"

    # Should count 2 contacts with email
    assert result["result"] >= 2


# ────────────────────────────────────────────────────────────────────────────
# End-to-End Tests with Multiple RPC Calls
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_e2e_rpc_multiple_calls_in_single_function(
    function_manager_factory,
    cleanup_venvs,
):
    """RPC should work for multiple calls within a single function."""
    fm = function_manager_factory()
    cm = ContactManager()

    # Create contacts - some with email, some without
    cm._create_contact(
        first_name="Diana",
        email_address="diana@example.com",
    )
    cm._create_contact(
        first_name="Edward",
        phone_number="5550000",
    )

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    cleanup_venvs(fm, venv_id)

    primitives = Primitives()

    result = await fm.execute_in_venv(
        venv_id=venv_id,
        implementation=MULTI_CALL_FUNCTION,
        call_kwargs={},
        is_async=False,
        primitives=primitives,
    )

    assert result["error"] is None, f"Unexpected error: {result['error']}"

    # Verify both calls returned correct data
    assert result["result"]["total_count"] >= 2  # At least 2 contacts
    assert result["result"]["email_count"] >= 1  # At least 1 with email
    assert result["result"]["email_count"] <= result["result"]["total_count"]


# ────────────────────────────────────────────────────────────────────────────
# Error Propagation Tests with Real Managers
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_e2e_rpc_invalid_method_error(
    function_manager_factory,
    cleanup_venvs,
):
    """RPC to non-existent method should propagate error correctly."""
    fm = function_manager_factory()

    # Function that calls a non-existent method
    bad_function = """
def call_bad_method() -> str:
    result = primitives.contacts.this_method_does_not_exist()
    return result
""".strip()

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    cleanup_venvs(fm, venv_id)

    primitives = Primitives()

    result = await fm.execute_in_venv(
        venv_id=venv_id,
        implementation=bad_function,
        call_kwargs={},
        is_async=False,
        primitives=primitives,
    )

    # Should have an error
    assert result["error"] is not None
    assert (
        "this_method_does_not_exist" in result["error"]
        or "AttributeError" in result["error"]
    )


@_handle_project
@pytest.mark.asyncio
async def test_e2e_rpc_invalid_filter_expression_error(
    function_manager_factory,
    cleanup_venvs,
):
    """RPC should propagate errors from manager methods back to venv."""
    fm = function_manager_factory()

    # Function that calls filter_contacts with an invalid filter expression
    bad_filter_function = """
def call_bad_filter() -> dict:
    result = primitives.contacts.filter_contacts(filter="this is not valid python")
    return result
""".strip()

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    cleanup_venvs(fm, venv_id)

    primitives = Primitives()

    result = await fm.execute_in_venv(
        venv_id=venv_id,
        implementation=bad_filter_function,
        call_kwargs={},
        is_async=False,
        primitives=primitives,
    )

    # Should have an error about the filter
    assert result["error"] is not None


# ────────────────────────────────────────────────────────────────────────────
# Data Consistency Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_e2e_rpc_sees_data_created_after_primitives_init(
    function_manager_factory,
    cleanup_venvs,
):
    """RPC should see data created after Primitives() was instantiated."""
    fm = function_manager_factory()
    cm = ContactManager()

    # Create Primitives FIRST
    primitives = Primitives()

    # THEN create contacts
    cm._create_contact(first_name="CreatedAfter", email_address="after@test.com")

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    cleanup_venvs(fm, venv_id)

    result = await fm.execute_in_venv(
        venv_id=venv_id,
        implementation=FILTER_CONTACTS_FUNCTION,
        call_kwargs={},
        is_async=False,
        primitives=primitives,
    )

    assert result["error"] is None, f"Unexpected error: {result['error']}"

    contacts = result["result"].get("contacts", [])
    first_names = [c.get("first_name") for c in contacts]
    assert "CreatedAfter" in first_names


@_handle_project
@pytest.mark.asyncio
async def test_e2e_rpc_multiple_sequential_calls(
    function_manager_factory,
    cleanup_venvs,
):
    """Multiple sequential RPC calls should all work correctly."""
    fm = function_manager_factory()
    cm = ContactManager()

    # Create initial contact
    cm._create_contact(first_name="Initial")

    # Function that makes multiple RPC calls
    multi_call_function = """
def multi_call() -> list:
    results = []
    for i in range(3):
        r = primitives.contacts.filter_contacts()
        results.append(len(r.get("contacts", [])))
    return results
""".strip()

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    cleanup_venvs(fm, venv_id)

    primitives = Primitives()

    result = await fm.execute_in_venv(
        venv_id=venv_id,
        implementation=multi_call_function,
        call_kwargs={},
        is_async=False,
        primitives=primitives,
    )

    assert result["error"] is None, f"Unexpected error: {result['error']}"

    # All 3 calls should return the same count
    counts = result["result"]
    assert len(counts) == 3
    assert all(c >= 1 for c in counts)
    assert counts[0] == counts[1] == counts[2]
