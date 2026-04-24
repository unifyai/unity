"""Tile bindings resolve against the active body's base context.

DashboardManager tiles carry declarative ``DataBinding`` rows whose
``context`` field starts short-form (e.g. ``"Data/sales_pipeline"``) and
is expanded at write time through ``resolve_binding_contexts``.  The
resolver walks ``unify.get_contexts(prefix=base)``, where ``base`` is
the body's ``ContextRegistry.base_for("Dashboards/Tiles")``:

- Inside a Hive the base is ``Hives/{hive_id}`` and short-form bindings
  resolve to ``Hives/{hive_id}/Data/<table>``.
- On a solo body the base is ``{user_id}/{assistant_id}`` and the same
  short-form resolves to ``{user_id}/{assistant_id}/Data/<table>``.

The test persists a tile in each mode and reads back the stored
``data_binding_contexts`` column to verify both routings.  It runs
against the live backend because the resolver observes real contexts
returned by ``unify.get_contexts``.
"""

from __future__ import annotations

import random

import pytest
import unify

from tests.helpers import bind_body, unique_hive_id
from unity.dashboard_manager.dashboard_manager import DashboardManager
from unity.dashboard_manager.types.tile import FilterBinding
from unity.session_details import SESSION_DETAILS


_TABLE = "sales_pipeline"


def _cleanup(ctx_root: str) -> None:
    """Drop the body-scoped solo root; failures mean the context was never created."""
    try:
        unify.delete_context(context=ctx_root, include_children=True)
    except Exception:
        pass


def _fetch_binding_contexts(dm: DashboardManager, token: str) -> str:
    """Return the stored ``data_binding_contexts`` CSV for ``token``."""
    tile = dm.get_tile(token)
    assert tile is not None, f"tile {token!r} not found in {dm._tiles_ctx}"
    assert tile.data_binding_contexts, (
        f"tile {token!r} has no resolved binding contexts: {tile!r}"
    )
    return tile.data_binding_contexts


@pytest.mark.requires_real_unify
def test_hive_body_resolves_binding_onto_hive_root():
    hive_id = unique_hive_id()
    hive_root = f"Hives/{hive_id}"
    data_ctx = f"{hive_root}/Data/{_TABLE}"

    try:
        bind_body(hive_id=hive_id, agent_id=5001)
        unify.create_context(name=data_ctx)

        dm = DashboardManager()
        result = dm.create_tile(
            "<div id='pipeline'>Loading...</div>",
            title="Pipeline Tile",
            data_bindings=[FilterBinding(context=f"Data/{_TABLE}")],
        )
        assert result.succeeded, f"create_tile failed: {result.error}"

        resolved = _fetch_binding_contexts(dm, result.token)
        assert resolved == data_ctx, (
            f"expected binding to resolve to {data_ctx!r}, got {resolved!r}"
        )
    finally:
        SESSION_DETAILS.reset()
        _cleanup(hive_root)


@pytest.mark.requires_real_unify
def test_solo_body_resolves_binding_onto_its_own_base():
    user_id = f"solo-{random.randint(10_000_000, 99_999_999)}"
    agent_id = random.randint(10_000_000, 99_999_999)
    solo_root = f"{user_id}/{agent_id}"
    data_ctx = f"{solo_root}/Data/{_TABLE}"

    try:
        bind_body(
            hive_id=None,
            agent_id=agent_id,
            user_id=user_id,
            solo_base=solo_root,
        )
        unify.create_context(name=data_ctx)
        unify.create_context(name=f"{solo_root}/Dashboards/Tiles")

        dm = DashboardManager()
        result = dm.create_tile(
            "<div id='pipeline'>Loading...</div>",
            title="Pipeline Tile",
            data_bindings=[FilterBinding(context=f"Data/{_TABLE}")],
        )
        assert result.succeeded, f"create_tile failed: {result.error}"

        resolved = _fetch_binding_contexts(dm, result.token)
        assert resolved == data_ctx, (
            f"expected binding to resolve to {data_ctx!r}, got {resolved!r}"
        )
    finally:
        SESSION_DETAILS.reset()
        _cleanup(solo_root)
