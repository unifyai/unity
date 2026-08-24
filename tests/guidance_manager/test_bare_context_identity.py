"""Guidance writes must never yield rows without a ``guidance_id``.

The Guidance context is provisioned with ``guidance_id`` auto-counting, but a
row write that reaches the backend first auto-creates the context bare, and
every row it then accepts has no identity — unreachable by
``get_guidance``/``update_guidance``/``delete_guidance`` and rendered as the
``-1`` sentinel by reads. These tests pin the two guards: provisioning
refuses a bare context outright, and ``add_guidance`` fails loudly (removing
the orphan row) if a created row ever comes back without its id.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import unisdk

from tests.helpers import _handle_project
from unify.common.context_store import ContextIdentityError
from unify.common.log_utils import MissingRowIdentityError
from unify.guidance_manager import guidance_manager as gm_module
from unify.guidance_manager.guidance_manager import GuidanceManager


@_handle_project
def test_provisioning_refuses_bare_guidance_context():
    base = unisdk.get_active_context()["write"]

    # A write racing provisioning auto-creates the context bare.
    unisdk.log(
        context=f"{base}/Guidance",
        new=True,
        title="orphan",
        content="written before provisioning",
    )

    with pytest.raises(ContextIdentityError):
        GuidanceManager()


@_handle_project
def test_add_guidance_without_assigned_id_fails_loud(monkeypatch):
    gm = GuidanceManager()

    fake_log = SimpleNamespace(
        id=987654,
        entries={"title": "T", "content": "C"},
    )
    deleted: list[dict] = []
    monkeypatch.setattr(gm_module, "unity_log", lambda **kwargs: fake_log)
    monkeypatch.setattr(
        unisdk,
        "delete_logs",
        lambda **kwargs: deleted.append(kwargs),
    )

    with pytest.raises(MissingRowIdentityError):
        gm.add_guidance(title="T", content="C")

    assert deleted and deleted[0]["logs"] == 987654
