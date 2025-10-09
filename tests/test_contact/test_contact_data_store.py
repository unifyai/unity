from __future__ import annotations

import pytest

from unity.contact_manager.contact_manager import ContactManager
from unity.common.data_store import DataStore
from tests.helpers import _handle_project


@pytest.mark.unit
@_handle_project
def test_data_store_updated_after_create():
    cm = ContactManager()

    # Sanity: resolve the DataStore instance for this context
    ds = DataStore.for_context(cm._ctx, key_fields=("contact_id",))

    # Create a contact
    out = cm._create_contact(first_name="CacheTest", surname="One")
    cid = out["details"]["contact_id"]

    # Verify DataStore has the newly created row (never reading from it elsewhere)
    row = ds[cid]
    assert row["contact_id"] == cid
    assert row.get("first_name") == "CacheTest"
    assert row.get("surname") == "One"


@pytest.mark.unit
@_handle_project
def test_data_store_updated_after_update():
    cm = ContactManager()

    ds = DataStore.for_context(cm._ctx, key_fields=("contact_id",))

    # Create then update
    out = cm._create_contact(first_name="CacheTest", surname="Two")
    cid = out["details"]["contact_id"]

    cm._update_contact(contact_id=cid, surname="Updated")

    # Verify DataStore reflects updated surname
    row = ds[cid]
    assert row["contact_id"] == cid
    assert row.get("first_name") == "CacheTest"
    assert row.get("surname") == "Updated"


@pytest.mark.unit
@_handle_project
def test_data_store_deleted_after_delete():
    cm = ContactManager()

    ds = DataStore.for_context(cm._ctx, key_fields=("contact_id",))

    # Create then delete
    out = cm._create_contact(first_name="CacheTest", surname="DeleteMe")
    cid = out["details"]["contact_id"]

    # Ensure present first
    _ = ds[cid]

    cm._delete_contact(contact_id=cid)

    # Verify removal from DataStore
    with pytest.raises(KeyError):
        _ = ds[cid]


@pytest.mark.unit
@_handle_project
def test_filter_contacts_repopulates_data_store():
    cm = ContactManager()

    ds = DataStore.for_context(cm._ctx, key_fields=("contact_id",))

    # Seed a user contact
    out = cm._create_contact(first_name="CacheTest", surname="Filter")
    cid = out["details"]["contact_id"]

    # Clear DataStore manually (simulate empty cache)
    ds.clear()

    # Read via filter_contacts and ensure cache is repopulated
    rows = cm._filter_contacts(filter=f"contact_id == {cid}")
    assert rows and rows[0].contact_id == cid

    row = ds[cid]
    assert row["contact_id"] == cid
    assert row.get("first_name") == "CacheTest"


@pytest.mark.unit
@_handle_project
def test_search_contacts_repopulates_data_store():
    cm = ContactManager()

    ds = DataStore.for_context(cm._ctx, key_fields=("contact_id",))

    # Seed
    out = cm._create_contact(first_name="CacheTest", bio="emails and texts")
    cid = out["details"]["contact_id"]

    ds.clear()

    # Trigger semantic path (references provided) which writes-through filled rows
    results = cm._search_contacts(references={"bio": "emails"}, k=1)
    assert results and results[0].contact_id == cid

    row = ds[cid]
    assert row["contact_id"] == cid
    assert row.get("first_name") == "CacheTest"


@pytest.mark.unit
@_handle_project
def test_system_contacts_present_in_data_store_after_init():
    cm = ContactManager()

    ds = DataStore.for_context(cm._ctx, key_fields=("contact_id",))

    # Assistant and default user should be cached
    a = ds.get(0)
    u = ds.get(1)
    assert a is not None and a.get("respond_to") is True
    assert u is not None and u.get("respond_to") is True


@pytest.mark.unit
@_handle_project
def test_data_store_hygiene_after_custom_column_delete():
    cm = ContactManager()
    ds = DataStore.for_context(cm._ctx, key_fields=("contact_id",))

    # Create a custom column and a contact that uses it
    cm._create_custom_column(column_name="department", column_type="str")
    cid = cm._create_contact(first_name="Jane", department="Engineering")["details"][
        "contact_id"
    ]

    # Ensure cache has the field
    row = ds[cid]
    assert row.get("department") == "Engineering"

    # Delete the custom column
    cm._delete_custom_column(column_name="department")

    # The cache should be scrubbed of the deleted key
    row2 = ds[cid]
    assert "department" not in row2


@pytest.mark.unit
@_handle_project
def test_data_store_after_merge_contacts():
    cm = ContactManager()
    ds = DataStore.for_context(cm._ctx, key_fields=("contact_id",))

    # Create two contacts and ensure both cached via writes
    cid1 = cm._create_contact(first_name="John", surname="Doe")["details"]["contact_id"]
    cid2 = cm._create_contact(first_name="Johnny", surname="Roe")["details"][
        "contact_id"
    ]

    # Merge: keep cid1, take surname from cid2
    cm._merge_contacts(
        contact_id_1=cid1,
        contact_id_2=cid2,
        overrides={"contact_id": 1, "surname": 2},
    )

    # Kept contact should be present and updated
    kept_row = ds[cid1]
    assert kept_row["first_name"] == "John"
    assert kept_row.get("surname") == "Roe"

    # Deleted contact should be absent from DataStore
    with pytest.raises(KeyError):
        _ = ds[cid2]


@pytest.mark.unit
@_handle_project
def test_data_store_never_contains_vector_columns():
    cm = ContactManager()
    ds = DataStore.for_context(cm._ctx, key_fields=("contact_id",))

    # Create a contact and drive semantic path to create vectors server-side
    cm._create_contact(first_name="VecTest", bio="likes vectors")
    _ = cm._search_contacts(references={"bio": "vectors"}, k=1)

    # Scan snapshot and assert no *_emb keys exist in cached rows
    snap = ds.snapshot()
    for _key, row in snap.items():
        assert all(not str(k).endswith("_emb") for k in row.keys())
