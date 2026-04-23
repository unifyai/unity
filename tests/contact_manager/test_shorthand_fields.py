from __future__ import annotations

import re

from unity.contact_manager.types.contact import Contact
from tests.helpers import _handle_project
from unity.contact_manager.contact_manager import ContactManager


@_handle_project
def test_json_shorthand_aliases():
    c = Contact(
        contact_id=42,
        first_name="Alice",
        surname="Smith",
    )

    dumped = c.model_dump(mode="json", context={"shorthand": True})

    for k in ("cid", "fn", "sn"):
        assert k in dumped, f"expected shorthand key {k} in dump"

    for k in (
        "contact_id",
        "first_name",
        "surname",
    ):
        assert k not in dumped, f"did not expect original key {k} in dump"

    for k in ("resp", "policy", "should_respond", "response_policy"):
        assert (
            k not in dumped
        ), f"{k} is now an overlay field and must not appear on a shared Contact dump"


@_handle_project
def test_json_shorthand_prune():
    c = Contact(
        contact_id=7,
        first_name="Bob",
    )

    dumped = c.model_dump(
        mode="json",
        context={"shorthand": True, "prune_empty": True},
    )

    for k in ("cid", "fn"):
        assert k in dumped, f"expected shorthand key {k} in dump"

    for k in ("sn", "email", "phone", "bio", "rs", "policy", "resp"):
        assert k not in dumped, f"did not expect empty shorthand key {k} in dump"


@_handle_project
def test_custom_column_shorthand():
    cm = ContactManager()

    # Create a custom column and register a shorthand alias on the Contact model
    cm._create_custom_column(column_name="nickname", column_type="str")
    Contact.register_alias("nickname")

    fwd = Contact.shorthand_map()
    assert "nickname" in fwd, "Custom column should have a shorthand alias registered"
    alias = fwd["nickname"]

    # Alias should be snake_case and not collide with built-in aliases
    assert re.fullmatch(r"[a-z][a-z0-9_]*", alias), "Alias must be snake_case"
    builtin_aliases = set(Contact.SHORTHAND_MAP.values())
    assert (
        alias not in builtin_aliases
    ), "Alias should not collide with built-in aliases"

    # Create a contact using the custom field, then ensure JSON shorthand uses the alias
    out = cm._create_contact(first_name="Zoe", nickname="Z")
    cid = out["details"]["contact_id"]
    contact = cm.filter_contacts(filter=f"contact_id == {cid}")["contacts"][0]

    dumped = contact.model_dump(mode="json", context={"shorthand": True})
    assert alias in dumped and dumped[alias] == "Z"
    assert (
        "nickname" not in dumped
    ), "Original custom field key should be aliased in shorthand JSON"


@_handle_project
def test_custom_column_inverse():
    cm = ContactManager()

    # Create another custom column and register a shorthand alias, then verify forward/inverse maps agree
    cm._create_custom_column(column_name="department", column_type="str")
    Contact.register_alias("department")

    fwd = Contact.shorthand_map()
    inv = Contact.shorthand_inverse_map()

    assert "department" in fwd
    dep_alias = fwd["department"]
    assert dep_alias in inv and inv[dep_alias] == "department"
