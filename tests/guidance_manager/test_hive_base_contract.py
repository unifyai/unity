"""GuidanceManager Hive-base contract.

``"Guidance"`` is in ``_HIVE_SCOPED_TABLES``: a body inside a Hive resolves
it under ``Hives/{hive_id}/Guidance`` so every member reads and writes the
same rows; a solo body keeps its own per-body ``Guidance`` table. Every row
also carries ``authoring_assistant_id`` so reviewers can attribute shared
rows back to the authoring body. The stamp is written once at create time
and is excluded from the update surface.
"""

from __future__ import annotations

import pytest

from unity.common.context_registry import ContextRegistry
from unity.guidance_manager.types.guidance import Guidance
from unity.session_details import SESSION_DETAILS

pytestmark = pytest.mark.usefixtures("pinned_hive_body")


def test_hive_member_resolves_guidance_to_hive_root():
    SESSION_DETAILS.hive_id = 42
    assert ContextRegistry.base_for("Guidance") == "Hives/42"


def test_solo_body_resolves_guidance_to_per_body_root(pinned_hive_body):
    assert SESSION_DETAILS.hive_id is None
    assert ContextRegistry.base_for("Guidance") == pinned_hive_body


def test_guidance_model_declares_authoring_assistant_id():
    """The shared-row model exposes the authoring stamp column."""
    field = Guidance.model_fields.get("authoring_assistant_id")
    assert field is not None
    assert field.default is None


def test_guidance_model_accepts_authoring_assistant_id_stamp():
    g = Guidance(
        title="title",
        content="content",
        authoring_assistant_id=7,
    )
    assert g.authoring_assistant_id == 7
