"""
Unit-tests covering the add / use / delete cycle for **custom contact columns**.

The tests mirror the style of the existing contact-manager suites and rely on
the private helpers (`_create_custom_column`, `_delete_custom_column`,
`_list_columns`) that were added to the implementation.
"""

from __future__ import annotations

import pytest

from unity.contact_manager.contact_manager import ContactManager

# keeps each test isolated in its own Unify project / trace context
from tests.helpers import _handle_project


# ────────────────────────────────────────────────────────────────────────────
# 1.  Create a custom column                                                #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.unit
@_handle_project
def test_create_custom_column():
    cm = ContactManager()

    before_cols = cm._list_columns()
    assert (
        "nickname" not in before_cols
    ), "Pre-condition failed: nickname already exists"

    cm._create_custom_column(column_name="nickname", column_type="str")

    after_cols = cm._list_columns()
    assert "nickname" in after_cols and after_cols["nickname"] == "str"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Refuse to recreate a required column                                  #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.unit
@_handle_project
def test_create_required_column_raises():
    cm = ContactManager()

    with pytest.raises(AssertionError):
        cm._create_custom_column(column_name="first_name", column_type="str")


# ────────────────────────────────────────────────────────────────────────────
# 3.  Delete a custom column                                                #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.unit
@_handle_project
def test_delete_custom_column():
    cm = ContactManager()

    cm._create_custom_column(column_name="twitter", column_type="str")
    assert "twitter" in cm._list_columns()

    cm._delete_custom_column(column_name="twitter")
    assert "twitter" not in cm._list_columns()


# ────────────────────────────────────────────────────────────────────────────
# 4.  Refuse to delete a required column                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.unit
@_handle_project
def test_delete_required_column_raises():
    cm = ContactManager()

    with pytest.raises(ValueError):
        cm._delete_custom_column(column_name="phone_number")


# ────────────────────────────────────────────────────────────────────────────
# 5.  Create a contact that uses a custom field                             #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.unit
@_handle_project
def test_create_contact_with_custom_field():
    cm = ContactManager()

    cm._create_custom_column(column_name="department", column_type="str")

    cid = cm._create_contact(
        first_name="Jane",
        department="Engineering",
    )[
        "details"
    ]["contact_id"]
    contacts = cm._filter_contacts(filter=f"contact_id == {cid}")
    assert contacts and contacts[0].department == "Engineering"


# ────────────────────────────────────────────────────────────────────────────
# 6.  Update an existing custom field                                       #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.unit
@_handle_project
def test_update_contact_custom_field():
    cm = ContactManager()

    cm._create_custom_column(column_name="age", column_type="int")
    cid = cm._create_contact(first_name="Tom", age=30)["details"]["contact_id"]

    cm._update_contact(contact_id=cid, age=31)
    contact = cm._filter_contacts(filter=f"contact_id == {cid}")[0]
    assert contact.age == 31
