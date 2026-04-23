"""``ContextRegistry.base_for`` and ``_is_absolute_reference``.

``base_for`` is the single source of truth for a manager's storage root. A
body in a Hive resolves Hive-flagged tables under ``Hives/{hive_id}`` so every
member shares the same rows; every other table stays under the per-body
``{user_id}/{assistant_id}`` root. ``_is_absolute_reference`` keeps foreign
keys pointing at the correct target when the referenced table lives in a
different root (a per-body overlay that references a Hive-shared parent).

The tests mock ``_get_active_context`` so they exercise the resolver contract
without a live Unify backend.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from unity.common.context_registry import (
    ContextRegistry,
    TableContext,
    _is_absolute_reference,
)
from unity.session_details import SESSION_DETAILS

_PER_BODY_BASE = "u7/7"


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Each test gets a clean ``SESSION_DETAILS`` and registry."""
    SESSION_DETAILS.reset()
    SESSION_DETAILS.populate(agent_id=7, user_id="u7")
    previous_base = ContextRegistry._base_context
    ContextRegistry._base_context = _PER_BODY_BASE
    yield
    SESSION_DETAILS.reset()
    ContextRegistry._base_context = previous_base


def test_base_for_tasks_returns_hive_root_for_hive_member():
    SESSION_DETAILS.hive_id = 42
    assert ContextRegistry.base_for("Tasks") == "Hives/42"


def test_base_for_tasks_returns_per_body_base_for_solo_body():
    assert SESSION_DETAILS.hive_id is None
    assert ContextRegistry.base_for("Tasks") == _PER_BODY_BASE


def test_base_for_contacts_returns_hive_root_for_hive_member():
    """Contacts is shared across every body in a Hive.

    A body that belongs to a Hive reads and writes the same ``Contacts``
    rows as every other member under ``Hives/{hive_id}/Contacts``. Solo
    bodies continue to resolve ``Contacts`` to their per-body base.
    """
    SESSION_DETAILS.hive_id = 42
    assert ContextRegistry.base_for("Contacts") == "Hives/42"


def test_base_for_non_hive_scoped_table_ignores_hive_membership():
    """A table missing from ``_HIVE_SCOPED_TABLES`` keeps the per-body base.

    Knowledge is not Hive-flagged, so even a body that belongs to a Hive
    resolves it to ``{user_id}/{assistant_id}``. Only table names listed
    in :data:`_HIVE_SCOPED_TABLES` route through the Hive root.
    """
    SESSION_DETAILS.hive_id = 42
    assert ContextRegistry.base_for("Knowledge") == _PER_BODY_BASE


def test_base_for_raises_when_no_base_available():
    """A missing base is a real configuration bug, not a silent fallback."""
    ContextRegistry._base_context = None
    with patch.object(ContextRegistry, "_get_active_context", return_value=""):
        with pytest.raises(RuntimeError, match="no base context available"):
            ContextRegistry.base_for("Tasks")


def test_is_absolute_reference_accepts_hive_prefix():
    assert (
        _is_absolute_reference(
            "Hives/42/Contacts.contact_id",
            _PER_BODY_BASE,
        )
        is True
    )


def test_is_absolute_reference_accepts_matching_user_segment():
    """A reference that already starts with the body's user prefix is absolute."""
    assert _is_absolute_reference("u7/Contacts.contact_id", _PER_BODY_BASE) is True


def test_is_absolute_reference_rejects_bare_table_reference():
    """Relative references (``Table.column``) still rewrite under the caller's base."""
    assert _is_absolute_reference("Contacts.contact_id", _PER_BODY_BASE) is False


def test_get_contexts_for_manager_passes_hive_reference_through_unrewritten():
    """A ``Hives/{id}/<Table>.col`` FK survives context resolution absolutely.

    A per-body overlay that references a Hive-shared parent declares its
    FK as an absolute ``Hives/{hive_id}/<Table>.<col>`` string. The FK
    rewriter in :meth:`ContextRegistry._get_contexts_for_manager` must
    leave those absolute references untouched — prefixing with the
    caller's per-body base would produce ``{user}/{assistant}/Hives/...``
    and the FK would never resolve at read time.
    """

    class _OverlayManager:
        class Config:
            required_contexts = [
                TableContext(
                    name="TestOverlay",
                    description="Per-body overlay pointing at a Hive-shared parent.",
                    foreign_keys=[
                        {
                            "column": "target_id",
                            "references": "Hives/42/TestTarget.id",
                        },
                        {
                            "column": "local_id",
                            "references": "TestLocal.id",
                        },
                    ],
                ),
            ]

    contexts = ContextRegistry._get_contexts_for_manager(_OverlayManager)
    entry = contexts["TestOverlay"]
    fks_by_column = {
        fk["column"]: fk["references"] for fk in entry["resolved_foreign_keys"]
    }

    assert entry["resolved_name"] == f"{_PER_BODY_BASE}/TestOverlay"
    assert fks_by_column["target_id"] == "Hives/42/TestTarget.id"
    assert fks_by_column["local_id"] == f"{_PER_BODY_BASE}/TestLocal.id"


def test_get_contexts_for_manager_resolves_fk_via_target_table_base():
    """Relative FKs resolve against the referenced table's base, not the caller's.

    A per-body overlay (``ContactMembership``) that points at a
    Hive-scoped parent (``Contacts``) declares its FK with a bare
    ``Contacts.contact_id`` reference. The rewriter must route that to
    ``Hives/{hive_id}/Contacts.contact_id`` for Hive members — the
    caller's per-body base would be wrong because the target table lives
    under the Hive root.
    """
    SESSION_DETAILS.hive_id = 42

    class _OverlayManager:
        class Config:
            required_contexts = [
                TableContext(
                    name="ContactMembership",
                    description="Per-body overlay pointing at the shared Contacts table.",
                    foreign_keys=[
                        {
                            "column": "contact_id",
                            "references": "Contacts.contact_id",
                        },
                    ],
                ),
            ]

    contexts = ContextRegistry._get_contexts_for_manager(_OverlayManager)
    entry = contexts["ContactMembership"]

    assert entry["resolved_name"] == f"{_PER_BODY_BASE}/ContactMembership"
    fk = entry["resolved_foreign_keys"][0]
    assert fk["references"] == "Hives/42/Contacts.contact_id"


def test_get_contexts_for_manager_resolves_fk_to_per_body_for_solo_target():
    """A relative FK to a per-body table stays under the per-body base.

    ``TaskScheduler`` declares ``Functions/Compositional.function_id``. When
    ``Tasks`` is Hive-scoped but ``Functions/Compositional`` isn't, the
    rewriter must route the FK to ``{user}/{assistant}/Functions/Compositional.function_id``
    — the Hive base would dangle.
    """
    SESSION_DETAILS.hive_id = 42

    class _HiveTable:
        class Config:
            required_contexts = [
                TableContext(
                    name="Tasks",
                    description="Hive-shared table with an FK into a per-body table.",
                    foreign_keys=[
                        {
                            "column": "function_id",
                            "references": "Functions/Compositional.function_id",
                        },
                    ],
                ),
            ]

    contexts = ContextRegistry._get_contexts_for_manager(_HiveTable)
    entry = contexts["Tasks"]

    assert entry["resolved_name"] == "Hives/42/Tasks"
    fk = entry["resolved_foreign_keys"][0]
    assert fk["references"] == f"{_PER_BODY_BASE}/Functions/Compositional.function_id"
