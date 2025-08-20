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

    # Create a custom column to be included in the composite expression
    from unity.knowledge_manager.types import ColumnType

    cm._create_custom_column(column_name="occupation", column_type=ColumnType.str)

    # Populate different fields so the all-columns JSON helps similarity
    cm._create_contact(
        first_name="Helen",
        bio="Reads a lot",
        email_address="helen@example.com",
        occupation="Designer",
    )
    cm._create_contact(
        first_name="Ian",
        bio="Responds best to emails",
        email_address="ian@example.com",
        occupation="Email specialist",
    )
    cm._create_contact(
        first_name="Judy",
        bio="Text first please",
        phone_number="1234567890",
        occupation="Engineer",
    )

    # Build a composite expression spanning multiple fields including the custom column
    expr = "str({first_name}) + ' ' + str({bio}) + ' ' + str({email_address}) + ' ' + str({phone_number}) + ' ' + str({whatsapp_number}) + ' ' + str({occupation})"
    query = "best to emails"
    results = cm._search_contacts(references={expr: query}, k=2)

    assert len(results) == 2
    assert results[0].first_name == "Ian"

    # Ensure a derived embedding column exists for the composite expression
    cols = cm._list_columns()
    assert any(k.startswith("_expr_") and k.endswith("_emb") for k in cols.keys())


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_contacts_mean_of_cosine_ranking():
    cm = ContactManager()

    # Ensure custom column exists
    from unity.knowledge_manager.types import ColumnType

    cm._create_custom_column(column_name="occupation", column_type=ColumnType.str)

    # A: matches both references
    cm._create_contact(
        first_name="Alex",
        rolling_summary="We had a phone call last week about training",
        occupation="Professional footballer playing striker",
    )
    # B: matches only the occupation reference
    cm._create_contact(
        first_name="Blake",
        rolling_summary="Haven't spoken yet",
        occupation="Retired footballer and youth coach",
    )
    # C: matches only the rolling_summary reference
    cm._create_contact(
        first_name="Casey",
        rolling_summary="Had a phone call last week regarding taxes",
        occupation="Senior accountant focused on audits",
    )

    # Provide multiple references including the custom column and the composite expr
    refs = {
        "occupation": "footballer",
        "rolling_summary": "phone call last week",
    }
    results = cm._search_contacts(references=refs, k=3)
    assert len(results) == 3
    names = [c.first_name for c in results]

    # Ensure Alex (matches both) is ranked above the others
    assert names[0] == "Alex"
    assert names.index("Alex") < names.index("Blake")
    assert names.index("Alex") < names.index("Casey")

    # Ensure columns and vectors were created
    cols = cm._list_columns()
    assert "occupation" in cols
    assert "_occupation_emb" in cols
    assert "_rolling_summary_emb" in cols
    assert any(k.startswith("_sum_cos_") for k in cols.keys())


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_contacts_backfills_when_insufficient_similarity_results():
    cm = ContactManager()

    # Create several contacts with minimal bios so similarity on bio returns < k
    cm._create_contact(first_name="Adam")
    cm._create_contact(first_name="Beatrice")
    cm._create_contact(first_name="Carla", bio="needle in haystack")  # single match
    cm._create_contact(first_name="Darren")
    cm._create_contact(first_name="Evelyn")
    cm._create_contact(first_name="Frank")

    k = 4
    results = cm._search_contacts(references={"bio": "needle"}, k=k)

    assert len(results) == k
    names = [c.first_name for c in results]
    # Carla should be the top semantic match
    assert names[0] == "Carla"
    # Remaining should be backfilled from latest creation order without duplicates
    assert names[2:4] == ["Frank", "Evelyn"]

    # When references is None, skip semantic search and return most recent contacts
    recent_only = cm._search_contacts(references=None, k=3)
    assert [c.first_name for c in recent_only] == ["Frank", "Evelyn", "Darren"]

    # Also verify empty dict behaves the same as None
    recent_only_empty = cm._search_contacts(references={}, k=3)
    assert [c.first_name for c in recent_only_empty] == ["Frank", "Evelyn", "Darren"]


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_contacts_defaults_two_terms_no_crash():
    cm = ContactManager()
    refs = {"bio": "footballer", "rolling_summary": "footballer"}

    # Expect no exception and both the assistant and user to be returned
    results = cm._search_contacts(references=refs, k=2)
    assert isinstance(results, list)
    assert len(results) == 2
