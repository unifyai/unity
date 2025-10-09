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
