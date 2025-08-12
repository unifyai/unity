import pytest

from unity.contact_manager.contact_manager import ContactManager
from tests.helpers import _handle_project


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_contacts_single_string_vs_single_list_equivalence():
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
    by_str = cm._search_contacts(columns="bio", text=query, k=3)
    by_list = cm._search_contacts(columns=["bio"], text=query, k=3)

    assert [c.contact_id for c in by_str] == [c.contact_id for c in by_list]
    assert by_str[0].first_name == "Bob"

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

    # Use two columns so the source is derived as JSON over both
    selected = ["first_name", "bio"]
    # Sorted order is used internally to form the source/vec names
    selected_sorted = sorted(selected)
    expected_source = "_json_" + "_".join(selected_sorted)  # e.g. __json_bio_first_name
    expected_vec = f"{expected_source}_emb"  # leading underscore by implementation

    query = "quick text pings"
    results = cm._search_contacts(columns=selected, text=query, k=2)

    assert len(results) == 2
    # Grace mentions texting and quick pings – should be the top hit
    assert results[0].first_name == "Grace"

    # Ensure vector column was created for the JSON-derived source
    cols = cm._list_columns()
    assert expected_vec in cols, "Expected multi-column vector column not created"


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

    query = "best to emails"
    results = cm._search_contacts(text=query, k=2)  # columns=None default

    assert len(results) == 2
    assert results[0].first_name == "Ian"

    # Ensure the all-columns derived/vector columns exist
    cols = cm._list_columns()
    # source name: _all_columns_json -> vec: _<source>_emb
    assert (
        "_all_columns_json_emb" in cols
    ), "Expected all-columns vector column not created"
