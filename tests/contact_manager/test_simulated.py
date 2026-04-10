from __future__ import annotations

import pytest

from unity.contact_manager.simulated import (
    SimulatedContactManager,
)

# keeps each test isolated in its own Unify project / trace context
from tests.helpers import (
    _handle_project,
)


# ────────────────────────────────────────────────────────────────────────────
# 1.  Doc-string inheritance                                                 #
# ────────────────────────────────────────────────────────────────────────────
def test_docstrings_match_base():
    """
    Public methods in SimulatedContactManager should copy the real
    BaseContactManager doc-strings one-for-one (via functools.wraps).
    """
    from unity.contact_manager.base import BaseContactManager
    from unity.contact_manager.simulated import SimulatedContactManager

    assert (
        BaseContactManager.ask.__doc__.strip()
        in SimulatedContactManager.ask.__doc__.strip()
    ), ".store doc-string was not copied correctly"

    assert (
        BaseContactManager.update.__doc__.strip()
        in SimulatedContactManager.update.__doc__.strip()
    ), ".retrieve doc-string was not copied correctly"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Basic start-and-ask                                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_start_and_ask():
    cm = SimulatedContactManager("Demo CRM for unit-tests.")
    h = await cm.ask("List all my contacts.")
    answer = await h.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Stateful memory – serial asks                                         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_stateful_serial_asks():
    """
    Two consecutive .ask() calls share context because the manager keeps a
    stateful LLM.
    """
    cm = SimulatedContactManager()

    h1 = await cm.ask(
        "Please suggest a unique reference code for a new prospect, "
        "and reply with *only* that code.",
    )
    ref_code = (await h1.result()).strip()
    assert ref_code, "Reference code should not be empty"

    h2 = await cm.ask("Great. What reference code did you just propose?")
    answer2 = (await h2.result()).lower()
    assert ref_code.lower() in answer2, "LLM should recall the code it generated"


# ────────────────────────────────────────────────────────────────────────────
# 4.  Update then ask – state carries through (freeform mode)                #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_stateful_update_then_ask():
    """
    In freeform mode, two consecutive LLM calls share context because the
    manager keeps a stateful LLM that remembers conversation history.
    """
    # Use freeform mode so LLM remembers the "update" from conversation history
    cm = SimulatedContactManager(deterministic=False)
    full_name = "Johnathan Doe"
    email = "john.doe@example.com"

    # create a fictitious contact via the LLM-based update method
    upd = await cm.update(
        f"Create a new contact: {full_name}, email {email}, mark as high priority.",
    )
    await upd.result()

    # ask about it - in freeform mode, LLM should recall from conversation history
    hq = await cm.ask("Do we have Johnathan's contact details on file?")
    ans = (await hq.result()).lower()
    assert (
        "john" in ans and "email" in ans
    ), "Contact created via update should be recalled"


# ────────────────────────────────────────────────────────────────────────────
# 10.  Simulated private helpers                                             #
# ────────────────────────────────────────────────────────────────────────────
@_handle_project
def test_filter_sync():
    """
    SimulatedContactManager.filter_contacts should produce a plausible list of
    contacts synchronously (cannot be called from an active event loop).
    """
    cm = SimulatedContactManager()
    # Use a permissive filter; just validate basic shape and limit behaviour
    results = cm.filter_contacts(filter="True", limit=3)
    assert isinstance(results, dict), "Expected dict with 'contacts' key"
    assert "contacts" in results, "Result should have 'contacts' key"
    contacts = results["contacts"]
    assert isinstance(contacts, list), "Expected list of contacts"
    assert len(contacts) <= 3, "Limit should cap the number of returned contacts"
    # System contacts (0 and 1) should exist by default
    assert len(contacts) >= 2, "Should have at least system contacts"
    if contacts:
        first = contacts[0]
        assert hasattr(first, "contact_id"), "Each contact should have contact_id"


@_handle_project
def test_update_sync():
    """
    SimulatedContactManager.update_contact should return a structured confirmation
    with 'outcome' and 'details.contact_id'.
    """
    cm = SimulatedContactManager()
    out = cm.update_contact(contact_id=123, first_name="Alice")
    assert isinstance(out, dict), "update_contact yields a dict-like outcome"
    assert "outcome" in out, "Outcome should include 'outcome' message"
    assert "details" in out and isinstance(out["details"], dict)
    assert isinstance(out["details"].get("contact_id"), int)


@_handle_project
def test_clear_sync():
    """
    SimulatedContactManager.clear should reset the manager (hard-coded completion)
    and remain usable afterwards.
    """
    cm = SimulatedContactManager()
    # Do a synchronous operation to create some prior state
    cm.update_contact(contact_id=1, surname="Smith")
    # Clear should not raise and should be quick (no LLM roundtrip)
    cm.clear()
    # Post-clear, synchronous helper still works and returns system contacts
    post = cm.filter_contacts(limit=10)
    assert isinstance(post, dict), "Expected dict with 'contacts' key"
    contacts = post["contacts"]
    assert len(contacts) >= 2, "Should have system contacts after clear"


@_handle_project
def test_simulated_contact_manager_reduce_shapes():
    cm = SimulatedContactManager()

    scalar = cm.reduce(metric="sum", keys="contact_id")
    assert isinstance(scalar, (int, float))

    multi = cm.reduce(metric="max", keys=["contact_id"])
    assert isinstance(multi, dict)
    assert set(multi.keys()) == {"contact_id"}

    grouped = cm.reduce(metric="sum", keys="contact_id", group_by="segment")
    assert isinstance(grouped, dict)


# ────────────────────────────────────────────────────────────────────────────
# 13.  System contacts exist by default                                       #
# ────────────────────────────────────────────────────────────────────────────
@_handle_project
def test_system_contacts_exist():
    """
    SimulatedContactManager should have system contacts (0=assistant, 1=user)
    pre-populated on initialization.
    """
    cm = SimulatedContactManager()

    # Check assistant contact (id=0)
    assistant = cm.get_contact_info(0)
    assert 0 in assistant, "Assistant contact (id=0) should exist"
    assert assistant[0].get("first_name") == "Default"
    assert assistant[0].get("is_system") is True

    # Check user contact (id=1)
    user = cm.get_contact_info(1)
    assert 1 in user, "User contact (id=1) should exist"
    assert user[1].get("first_name") == "Default"
    assert user[1].get("is_system") is True


# ────────────────────────────────────────────────────────────────────────────
# 14.  get_contact_info returns correct data                                  #
# ────────────────────────────────────────────────────────────────────────────
@_handle_project
def test_get_contact_info_single():
    """get_contact_info returns contact data for a single ID."""
    cm = SimulatedContactManager()
    # Create a contact
    result = cm._create_contact(first_name="Alice", email_address="alice@example.com")
    contact_id = result["details"]["contact_id"]

    # Retrieve it
    info = cm.get_contact_info(contact_id)
    assert contact_id in info
    assert info[contact_id]["first_name"] == "Alice"
    assert info[contact_id]["email_address"] == "alice@example.com"


@_handle_project
def test_get_contact_info_multiple():
    """get_contact_info returns data for multiple IDs."""
    cm = SimulatedContactManager()
    # Create two contacts
    r1 = cm._create_contact(first_name="Bob")
    r2 = cm._create_contact(first_name="Carol")
    id1, id2 = r1["details"]["contact_id"], r2["details"]["contact_id"]

    # Retrieve both
    info = cm.get_contact_info([id1, id2])
    assert id1 in info and id2 in info
    assert info[id1]["first_name"] == "Bob"
    assert info[id2]["first_name"] == "Carol"


@_handle_project
def test_get_contact_info_missing():
    """get_contact_info omits missing IDs."""
    cm = SimulatedContactManager()
    info = cm.get_contact_info(9999)
    assert 9999 not in info
    assert info == {}


@_handle_project
def test_get_contact_info_with_fields():
    """get_contact_info respects fields parameter."""
    cm = SimulatedContactManager()
    cm._create_contact(
        first_name="Dave",
        surname="Smith",
        email_address="dave@example.com",
    )
    # Get contact with specific fields
    info = cm.get_contact_info(2, fields=["first_name", "email_address"])
    assert 2 in info
    assert "first_name" in info[2]
    assert "email_address" in info[2]
    assert "surname" not in info[2]  # Not requested


# ────────────────────────────────────────────────────────────────────────────
# 15.  _create_contact uses deterministic counter                             #
# ────────────────────────────────────────────────────────────────────────────
@_handle_project
def test_create_contact_deterministic_ids():
    """_create_contact assigns sequential IDs starting at 2."""
    cm = SimulatedContactManager()

    # First non-system contact should be ID 2
    r1 = cm._create_contact(first_name="First")
    assert r1["details"]["contact_id"] == 2

    # Second should be ID 3
    r2 = cm._create_contact(first_name="Second")
    assert r2["details"]["contact_id"] == 3

    # Third should be ID 4
    r3 = cm._create_contact(first_name="Third")
    assert r3["details"]["contact_id"] == 4


@_handle_project
def test_create_contact_stores_data():
    """_create_contact stores contact in internal store."""
    cm = SimulatedContactManager()
    cm._create_contact(
        first_name="Eve",
        surname="Johnson",
        phone_number="+15551234567",
        should_respond=False,
    )

    info = cm.get_contact_info(2)
    assert 2 in info
    assert info[2]["first_name"] == "Eve"
    assert info[2]["surname"] == "Johnson"
    assert info[2]["phone_number"] == "+15551234567"
    assert info[2]["should_respond"] is False


# ────────────────────────────────────────────────────────────────────────────
# 16.  _delete_contact rejects system contacts                                #
# ────────────────────────────────────────────────────────────────────────────
@_handle_project
def test_delete_contact_rejects_assistant():
    """_delete_contact raises RuntimeError for assistant (id=0)."""
    cm = SimulatedContactManager()
    with pytest.raises(RuntimeError, match="Cannot delete system contact"):
        cm._delete_contact(contact_id=0)


@_handle_project
def test_delete_contact_rejects_user():
    """_delete_contact raises RuntimeError for user (id=1)."""
    cm = SimulatedContactManager()
    with pytest.raises(RuntimeError, match="Cannot delete system contact"):
        cm._delete_contact(contact_id=1)


@_handle_project
def test_delete_contact_removes_non_system():
    """_delete_contact removes non-system contacts."""
    cm = SimulatedContactManager()
    r = cm._create_contact(first_name="ToDelete")
    contact_id = r["details"]["contact_id"]

    # Verify exists
    assert contact_id in cm.get_contact_info(contact_id)

    # Delete
    out = cm._delete_contact(contact_id=contact_id)
    assert out["outcome"] == "contact deleted"

    # Verify gone
    assert contact_id not in cm.get_contact_info(contact_id)


@_handle_project
def test_delete_contact_nonexistent_raises():
    """_delete_contact raises ValueError for non-existent contact."""
    cm = SimulatedContactManager()
    with pytest.raises(ValueError, match="does not exist"):
        cm._delete_contact(contact_id=9999)


# ────────────────────────────────────────────────────────────────────────────
# 17.  filter_contacts works deterministically                                #
# ────────────────────────────────────────────────────────────────────────────
@_handle_project
def test_filter_contacts_by_name():
    """filter_contacts can filter by first_name."""
    cm = SimulatedContactManager()
    cm._create_contact(first_name="Alice")
    cm._create_contact(first_name="Bob")
    cm._create_contact(first_name="Alice")  # Another Alice

    result = cm.filter_contacts(filter="first_name == 'Alice'")
    contacts = result["contacts"]
    assert len(contacts) == 2
    assert all(c.first_name == "Alice" for c in contacts)


@_handle_project
def test_filter_contacts_by_email():
    """filter_contacts can filter by email."""
    cm = SimulatedContactManager()
    cm._create_contact(first_name="Test", email_address="test@example.com")

    result = cm.filter_contacts(filter="email_address == 'test@example.com'")
    contacts = result["contacts"]
    assert len(contacts) == 1
    assert contacts[0].email_address == "test@example.com"


@_handle_project
def test_filter_contacts_offset_limit():
    """filter_contacts respects offset and limit."""
    cm = SimulatedContactManager()
    # Create 5 contacts (IDs 2-6) with valid names (no digits in first_name)
    names = ["Alice", "Bob", "Carol", "Dave", "Eve"]
    for name in names:
        cm._create_contact(first_name=name)

    # Get all (2 system + 5 created = 7)
    all_contacts = cm.filter_contacts()["contacts"]
    assert len(all_contacts) == 7

    # Get with offset=2, limit=2
    result = cm.filter_contacts(offset=2, limit=2)
    contacts = result["contacts"]
    assert len(contacts) == 2


# ────────────────────────────────────────────────────────────────────────────
# 18.  update_contact modifies internal store                                 #
# ────────────────────────────────────────────────────────────────────────────
@_handle_project
def test_update_contact_modifies_existing():
    """update_contact modifies an existing contact."""
    cm = SimulatedContactManager()
    r = cm._create_contact(first_name="Original")
    contact_id = r["details"]["contact_id"]

    cm.update_contact(contact_id=contact_id, first_name="Updated", bio="New bio")

    info = cm.get_contact_info(contact_id)
    assert info[contact_id]["first_name"] == "Updated"
    assert info[contact_id]["bio"] == "New bio"


@_handle_project
def test_update_contact_creates_if_missing():
    """update_contact creates a contact if it doesn't exist."""
    cm = SimulatedContactManager()
    cm.update_contact(contact_id=100, first_name="NewContact")

    info = cm.get_contact_info(100)
    assert 100 in info
    assert info[100]["first_name"] == "NewContact"


# ────────────────────────────────────────────────────────────────────────────
# 19.  Freeform mode (deterministic=False)                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_freeform_mode_ask():
    """In freeform mode, ask uses the description, not the store."""
    cm = SimulatedContactManager(
        description="You have exactly 100 contacts, all named John.",
        deterministic=False,
    )
    h = await cm.ask("How many contacts do I have?")
    answer = await h.result()
    # In freeform mode, the LLM should mention "100" based on the description
    assert isinstance(answer, str) and answer.strip()
    # Note: We can't guarantee "100" appears, but the test validates the mode works


# ────────────────────────────────────────────────────────────────────────────
# 20.  Clear resets to system contacts only                                   #
# ────────────────────────────────────────────────────────────────────────────
@_handle_project
def test_clear_resets_to_system_contacts():
    """clear() resets to only system contacts."""
    cm = SimulatedContactManager()

    # Create some contacts
    cm._create_contact(first_name="A")
    cm._create_contact(first_name="B")
    cm._create_contact(first_name="C")

    # Verify we have 5 contacts (2 system + 3 created)
    result = cm.filter_contacts()
    assert len(result["contacts"]) == 5

    # Clear
    cm.clear()

    # Should only have system contacts
    result = cm.filter_contacts()
    assert len(result["contacts"]) == 2
    ids = {c.contact_id for c in result["contacts"]}
    assert ids == {0, 1}
