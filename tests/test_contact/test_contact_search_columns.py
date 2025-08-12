import pytest

from unity.contact_manager.contact_manager import ContactManager
from tests.helpers import _handle_project


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_contacts_single_reference_basic():
    cm = ContactManager()

    entries = [
        ("Alice", "Enjoys long emails about projects"),
        ("Bob", "Prefers short text messages"),
        ("Carol", "Commutes by train and reads books"),
        ("Derek", "Hard to reach by phone"),
    ]
    for fname, bio in entries:
        cm._create_contact(first_name=fname, bio=bio)

    query = "short messages"
    results = cm._search_contacts(references={"bio": query}, k=3)

    assert results[0].first_name == "Bob"

    cols = cm._list_columns()
    assert "_bio_emb" in cols


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_contacts_multi_columns_json_and_vec_created():
    cm = ContactManager()

    # Distribute signal across two columns
    cm._create_contact(first_name="Eve", bio="Loves detailed emails and reports")
    cm._create_contact(first_name="Frank", bio="Short notes, hates phone calls")
    cm._create_contact(first_name="Grace", bio="Prefers texting and quick pings")

    query = "quick text pings"
    # Provide separate references; ranking should still pick Grace
    refs = {"bio": query, "first_name": "irrelevant"}
    results = cm._search_contacts(references=refs, k=2)

    assert len(results) == 2
    # Grace mentions texting and quick pings – should be the top hit
    assert results[0].first_name == "Grace"

    # Ensure vector columns were created for each referenced source
    cols = cm._list_columns()
    assert "_bio_emb" in cols
    assert "_first_name_emb" in cols


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_contacts_all_columns_default_derivation():
    cm = ContactManager()

    # Populate different fields so the all-columns JSON helps similarity
    cm._create_contact(
        first_name="Helen",
        bio="Reads a lot",
        email_address="helen@example.com",
    )
    cm._create_contact(
        first_name="Ian",
        bio="Responds best to emails",
        email_address="ian@example.com",
    )
    cm._create_contact(
        first_name="Judy",
        bio="Text first please",
        phone_number="1234567890",
    )

    # Build a composite expression spanning multiple fields
    expr = "str({first_name}) + ' ' + str({bio}) + ' ' + str({email_address}) + ' ' + str({phone_number}) + ' ' + str({whatsapp_number})"
    query = "best to emails"
    results = cm._search_contacts(references={expr: query}, k=2)

    assert len(results) == 2
    assert results[0].first_name == "Ian"

    # Ensure a derived embedding column exists for the composite expression
    cols = cm._list_columns()
    assert any(k.startswith("_expr_") and k.endswith("_emb") for k in cols.keys())
