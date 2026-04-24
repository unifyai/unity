"""ImageManager Hive-base contract.

``"Images"`` is in ``_HIVE_SCOPED_TABLES`` so a body inside a Hive reads
and writes it under ``Hives/{hive_id}/Images``; a solo body keeps its
per-body ``{user}/{assistant}/Images`` root. Image payloads are already
shared-friendly (``Image.data`` is base64 or a URL, not an on-disk path),
so the Hive scoping does not require new byte-routing code. Every row
carries ``authoring_assistant_id`` so Hive-shared rows can be attributed
back to the authoring body.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from unity.common.context_registry import ContextRegistry
from unity.image_manager.types.image import Image
from unity.session_details import SESSION_DETAILS

pytestmark = pytest.mark.usefixtures("pinned_hive_body")


def test_hive_member_resolves_images_to_hive_root():
    SESSION_DETAILS.hive_id = 42
    assert ContextRegistry.base_for("Images") == "Hives/42"


def test_solo_body_resolves_images_to_per_body_root(pinned_hive_body):
    assert SESSION_DETAILS.hive_id is None
    assert ContextRegistry.base_for("Images") == pinned_hive_body


def test_image_model_declares_authoring_assistant_id():
    field = Image.model_fields.get("authoring_assistant_id")
    assert field is not None
    assert field.default is None


def test_image_model_accepts_authoring_assistant_id_stamp():
    img = Image(
        timestamp=datetime.now(timezone.utc),
        data="base64://stub",
        authoring_assistant_id=7,
    )
    assert img.authoring_assistant_id == 7
