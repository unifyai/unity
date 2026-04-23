"""The per-body ``ContactMembership`` overlay is declared correctly.

Covers the Pydantic shape, the ``TableContext`` declaration on
:class:`ContactManager.Config`, and the foreign key from the overlay's
``contact_id`` back to the shared ``Contacts`` table. The FK must
use the *relative* form so :class:`ContextRegistry` routes Hive
members and solo bodies to the correct storage root automatically.
"""

from __future__ import annotations

from unity.common.context_registry import TableContext
from unity.contact_manager.contact_manager import ContactManager
from unity.contact_manager.types.contact import ContactMembership


def test_overlay_pydantic_shape():
    """The overlay carries the five body-local fields and nothing else."""
    fields = ContactMembership.model_fields
    assert set(fields.keys()) == {
        "contact_id",
        "relationship",
        "should_respond",
        "response_policy",
        "can_edit",
    }


def test_overlay_defaults_are_correct_for_lazy_materialization():
    """Defaults mirror the §4 contract: ``"other"`` / respond / editable."""
    row = ContactMembership(contact_id=7)
    assert row.relationship == "other"
    assert row.should_respond is True
    assert row.response_policy is None
    assert row.can_edit is True


def test_overlay_relationship_is_constrained_to_four_values():
    """Relationship is a ``Literal`` — anything else fails validation."""
    import pydantic

    try:
        ContactMembership(contact_id=7, relationship="nope")  # type: ignore[arg-type]
    except pydantic.ValidationError:
        return
    raise AssertionError(
        "ContactMembership accepted an unexpected relationship value",
    )


def test_contact_manager_config_declares_both_tables():
    """``ContactManager.Config.required_contexts`` lists shared + overlay."""
    names = [c.name for c in ContactManager.Config.required_contexts]
    assert "Contacts" in names
    assert "ContactMembership" in names


def test_contact_membership_table_context_shape():
    """The overlay ``TableContext`` declares the expected shape.

    ``unique_keys`` guarantees a single overlay row per ``contact_id``
    per body. The FK uses the relative form so ``ContextRegistry``
    routes it through :meth:`base_for` on the target table — the same
    path that carries Hive-vs-solo routing uniformly.
    """
    ctx: TableContext = next(
        c
        for c in ContactManager.Config.required_contexts
        if c.name == "ContactMembership"
    )

    assert ctx.unique_keys == {"contact_id": "int"}
    assert ctx.auto_counting is None
    assert ctx.foreign_keys is not None
    assert len(ctx.foreign_keys) == 1

    fk = ctx.foreign_keys[0]
    assert fk["name"] == "contact_id"
    assert fk["references"] == "Contacts.contact_id"
    assert fk.get("on_delete") == "SET NULL"


def test_contacts_shared_table_has_no_inbound_fk_from_membership_removed():
    """The overlay's FK is declared at the overlay's level, not the shared row.

    Regression: the shared ``Contacts`` table must not grow an
    ``ON DELETE`` rule that would surprise Orchestra's Hive-level
    cascade delete. Only the overlay declares the FK, and its
    ``SET NULL`` keeps the shared row authoritative.
    """
    shared_ctx: TableContext = next(
        c for c in ContactManager.Config.required_contexts if c.name == "Contacts"
    )
    assert not shared_ctx.foreign_keys
