from __future__ import annotations

import pytest

from unity.contact_manager.types.contact import Contact
from tests.helpers import _handle_project


@pytest.mark.unit
@_handle_project
def test_contact_json_shorthand_aliases_keys_no_prune():
    c = Contact(
        contact_id=42,
        first_name="Alice",
        surname="Smith",
    )

    dumped = c.model_dump(mode="json", context={"shorthand": True})

    # Aliased keys should exist
    for k in ("cid", "fn", "sn", "resp"):
        assert k in dumped, f"expected shorthand key {k} in dump"

    # Original keys should not be present
    for k in (
        "contact_id",
        "first_name",
        "surname",
        "respond_to",
    ):
        assert k not in dumped, f"did not expect original key {k} in dump"


@pytest.mark.unit
@_handle_project
def test_contact_json_shorthand_with_prune_omits_empty_fields():
    c = Contact(
        contact_id=7,
        first_name="Bob",
    )

    dumped = c.model_dump(
        mode="json",
        context={"shorthand": True, "prune_empty": True},
    )

    # Aliased keys present (id, first_name, respond_to)
    for k in ("cid", "fn", "resp"):
        assert k in dumped, f"expected shorthand key {k} in dump"

    # Empty/None fields should be pruned when prune_empty=True → aliases absent
    for k in ("sn", "email", "phone", "whatsapp", "bio", "rs", "policy"):
        assert k not in dumped, f"did not expect empty shorthand key {k} in dump"
