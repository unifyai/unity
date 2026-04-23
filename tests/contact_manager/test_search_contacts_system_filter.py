"""``search_contacts`` hides system contacts via the overlay, not magic ids.

The predicate is driven by ``ContactMembership.relationship``: every
body's overlay row with ``relationship IN ("self", "boss")`` joins the
excluded set. An ``is_system != True`` clause remains as the fallback
signal for bodies that have not yet materialized their system overlays.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from unity.contact_manager.search import _system_contact_row_filter


class _FakeLog:
    def __init__(self, entries):
        self.entries = entries


def _fake_membership_rows(pairs):
    return [_FakeLog({"contact_id": cid, "relationship": rel}) for cid, rel in pairs]


def _cm_stub() -> MagicMock:
    """Build a minimal ContactManager stand-in for the predicate builder."""
    cm = MagicMock()
    cm._membership_ctx = "u1/1/ContactMembership"
    return cm


def test_filter_excludes_self_and_boss_overlay_ids(monkeypatch):
    """Overlay rows with ``"self"`` / ``"boss"`` join the exclude list."""
    cm = _cm_stub()

    def fake_get_logs(*, context, filter, limit, from_fields):
        assert context == cm._membership_ctx
        assert "self" in filter and "boss" in filter
        return _fake_membership_rows([(17, "self"), (42, "boss")])

    monkeypatch.setattr("unity.contact_manager.search.unify.get_logs", fake_get_logs)

    predicate = _system_contact_row_filter(cm)

    assert "contact_id not in [17, 42]" in predicate
    assert "is_system != True" in predicate


def test_filter_falls_back_to_is_system_when_no_overlay(monkeypatch):
    """No overlay rows → the predicate is only the ``is_system`` fallback.

    Before a body first materializes its ``self`` / ``boss`` overlays
    the shared ``is_system`` flag is the only signal available, and it
    must still hide system rows from semantic search.
    """
    cm = _cm_stub()

    monkeypatch.setattr(
        "unity.contact_manager.search.unify.get_logs",
        lambda **kwargs: [],
    )

    predicate = _system_contact_row_filter(cm)

    assert predicate == "is_system != True"


def test_filter_survives_overlay_query_failure(monkeypatch):
    """A backend error during the overlay read falls back gracefully."""
    cm = _cm_stub()

    def boom(**kwargs):
        raise RuntimeError("overlay read failed")

    monkeypatch.setattr("unity.contact_manager.search.unify.get_logs", boom)

    predicate = _system_contact_row_filter(cm)

    assert predicate == "is_system != True"


def test_filter_does_not_hardcode_id_zero_or_one(monkeypatch):
    """The predicate never hardcodes ``contact_id != 0`` / ``!= 1``."""
    cm = _cm_stub()

    monkeypatch.setattr(
        "unity.contact_manager.search.unify.get_logs",
        lambda **kwargs: _fake_membership_rows([(17, "self")]),
    )

    predicate = _system_contact_row_filter(cm)

    assert "contact_id != 0" not in predicate
    assert "contact_id != 1" not in predicate


def test_filter_combines_clauses_with_and(monkeypatch):
    """Overlay exclusions and the ``is_system`` fallback AND together.

    Both signals point at "this is a system contact" independently;
    ANDing them means a contact has to pass both checks to survive,
    which is the correct conservative default for hiding system rows.
    """
    cm = _cm_stub()

    monkeypatch.setattr(
        "unity.contact_manager.search.unify.get_logs",
        lambda **kwargs: _fake_membership_rows([(5, "self"), (6, "boss")]),
    )

    predicate = _system_contact_row_filter(cm)

    assert predicate.startswith("is_system != True")
    assert " and contact_id not in [5, 6]" in predicate
