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
