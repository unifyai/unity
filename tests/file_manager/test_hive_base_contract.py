"""FileManager Hive-base contract.

``"Files"`` and ``"FileRecords"`` are in ``_HIVE_SCOPED_TABLES`` so a body
inside a Hive resolves both catalog tables under ``Hives/{hive_id}/...``;
a solo body keeps its per-body ``{user}/{assistant}/...`` root. Only the
*catalog* rows are Hive-shared — file bytes stay on the authoring body's
local disk under ``~/Unity/Local/...`` because cross-body byte access
would require routing through an outbound channel (handled in prompt
guidance, not this layer). Every catalog row carries
``authoring_assistant_id`` so reviewers can attribute shared rows to the
authoring body.
"""

from __future__ import annotations

import pytest

from unity.common.context_registry import ContextRegistry
from unity.file_manager.types.file import FileRecord, FileRecordFields, FileRecordRow
from unity.session_details import SESSION_DETAILS

pytestmark = pytest.mark.usefixtures("pinned_hive_body")


@pytest.mark.parametrize("table", ["Files", "FileRecords"])
def test_hive_member_resolves_file_catalog_to_hive_root(table):
    SESSION_DETAILS.hive_id = 42
    assert ContextRegistry.base_for(table) == "Hives/42"


@pytest.mark.parametrize("table", ["Files", "FileRecords"])
def test_solo_body_resolves_file_catalog_to_per_body_root(table, pinned_hive_body):
    assert SESSION_DETAILS.hive_id is None
    assert ContextRegistry.base_for(table) == pinned_hive_body


@pytest.mark.parametrize("model", [FileRecordFields, FileRecordRow, FileRecord])
def test_file_record_models_declare_authoring_assistant_id(model):
    field = model.model_fields.get("authoring_assistant_id")
    assert field is not None
    assert field.default is None


def test_file_record_row_accepts_authoring_assistant_id_stamp():
    row = FileRecordRow(
        file_path="/tmp/report.pdf",
        source_uri="local:///tmp/report.pdf",
        source_provider="Local",
        authoring_assistant_id=7,
    )
    assert row.authoring_assistant_id == 7
