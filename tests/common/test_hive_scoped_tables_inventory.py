"""Canary for the inventory of Hive-shared tables.

``_HIVE_SCOPED_TABLES`` decides which manager tables share state across every
body in a Hive. Adding a manager to the cluster means also adding its table
name(s) to that frozenset, and removing one means the same in reverse. This
test asserts the full expected inventory so drift — adding a manager to the
cluster without updating the set, or removing one without updating the set —
fails loudly here instead of quietly in production.

Per-body overlay tables (``ContactMembership``, ``SecretBinding``,
``OAuthTokens``) are intentionally *not* in the inventory because they carry
per-body state that layers on top of a Hive-shared parent. VirtualEnv
*catalog* rows are in the inventory; the on-disk venv materialization stays
per-body because the venv path derives from the active Unify context.
"""

from __future__ import annotations

from unity.common.context_registry import _HIVE_SCOPED_TABLES

EXPECTED_HIVE_SCOPED_TABLES: frozenset[str] = frozenset(
    {
        "BlackList",
        "Contacts",
        "Dashboards/Layouts",
        "Dashboards/Tiles",
        "Data",
        "Exchanges",
        "FileRecords",
        "Files",
        "Functions/Compositional",
        "Functions/Meta",
        "Functions/Primitives",
        "Functions/VirtualEnvs",
        "Guidance",
        "Images",
        "Knowledge",
        "SecretDefault",
        "Secrets",
        "Tasks",
        "Transcripts",
    },
)


def test_hive_scoped_tables_inventory_matches_expected():
    """Every manager in the Hive cluster is flagged; nothing else leaks in."""
    missing = EXPECTED_HIVE_SCOPED_TABLES - _HIVE_SCOPED_TABLES
    unexpected = _HIVE_SCOPED_TABLES - EXPECTED_HIVE_SCOPED_TABLES

    assert not missing, (
        "Tables missing from _HIVE_SCOPED_TABLES: "
        f"{sorted(missing)}. Add them to unity/common/context_registry.py "
        "and update this inventory if the new table is truly Hive-shared."
    )
    assert not unexpected, (
        "Unexpected entries in _HIVE_SCOPED_TABLES: "
        f"{sorted(unexpected)}. Remove the flag on per-body tables or update "
        "this inventory when a new Hive-shared table lands."
    )


def test_per_body_overlays_stay_off_the_inventory():
    """Overlay tables must never land on the Hive-shared inventory.

    Overlays (per-body views of a Hive-shared parent) intentionally keep
    bodies isolated from each other's private state; flagging them
    Hive-shared would collapse that isolation.
    """
    per_body_overlays = {"ContactMembership", "SecretBinding", "OAuthTokens"}

    assert not (per_body_overlays & _HIVE_SCOPED_TABLES), (
        "Per-body overlay tables must not be Hive-shared: "
        f"{sorted(per_body_overlays & _HIVE_SCOPED_TABLES)}"
    )
