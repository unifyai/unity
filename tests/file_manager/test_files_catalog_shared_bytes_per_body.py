"""Files catalog is Hive-shared; file bytes stay on each body's local disk.

FileManager's Hive contract has two parts:

- ``FileRecords`` and ``Files`` entries route onto the Hive root so two
  bodies enumerating attachments see the same catalog.
- The filesystem adapter (``LocalFileSystemAdapter`` here) writes bytes
  into a body-local root.  Bytes never migrate between bodies; cross-body
  access would require a remote adapter, not a shared Hive namespace.

The test drives the live backend because both claims are DB/filesystem
facts, not pure-Python shape checks.  The shared catalog is what
``unify.get_logs`` returns; the per-body bytes are what each adapter
root actually holds on disk.
"""

from __future__ import annotations

import pytest
import unify

from tests.helpers import bind_body, cleanup_hive_context, unique_hive_id
from unity.file_manager.filesystem_adapters.local_adapter import (
    LocalFileSystemAdapter,
)
from unity.file_manager.managers.file_manager import FileManager
from unity.file_manager.managers.utils.ops import add_or_replace_file_row
from unity.session_details import SESSION_DETAILS


@pytest.mark.requires_real_unify
def test_catalog_shared_bytes_stay_local(tmp_path):
    hive_id = unique_hive_id()
    records_ctx = f"Hives/{hive_id}/FileRecords/Local"

    root_a = tmp_path / "body_a"
    root_b = tmp_path / "body_b"
    root_a.mkdir()
    root_b.mkdir()

    try:
        bind_body(hive_id=hive_id, agent_id=3001)
        adapter_a = LocalFileSystemAdapter(root_a.as_posix())
        fm_a = FileManager(adapter=adapter_a)
        path_a = fm_a.save_attachment(
            attachment_id="att-a",
            filename="a.bin",
            contents=b"A" * 8,
            auto_ingest=False,
        )
        add_or_replace_file_row(
            fm_a,
            entry={
                "file_path": path_a,
                "source_uri": fm_a._resolve_to_uri(path_a),
                "source_provider": "Local",
                "status": "ok",
                "storage_id": "",
            },
        )

        bind_body(hive_id=hive_id, agent_id=4002)
        adapter_b = LocalFileSystemAdapter(root_b.as_posix())
        fm_b = FileManager(adapter=adapter_b)
        path_b = fm_b.save_attachment(
            attachment_id="att-b",
            filename="b.bin",
            contents=b"B" * 8,
            auto_ingest=False,
        )
        add_or_replace_file_row(
            fm_b,
            entry={
                "file_path": path_b,
                "source_uri": fm_b._resolve_to_uri(path_b),
                "source_provider": "Local",
                "status": "ok",
                "storage_id": "",
            },
        )

        rows = unify.get_logs(context=records_ctx, limit=1000)
        by_path = {row.entries.get("file_path"): row.entries for row in rows}
        assert path_a in by_path, (
            f"body A's catalog row missing from shared context; saw {list(by_path)}"
        )
        assert path_b in by_path, (
            f"body B's catalog row missing from shared context; saw {list(by_path)}"
        )
        assert by_path[path_a].get("authoring_assistant_id") == 3001
        assert by_path[path_b].get("authoring_assistant_id") == 4002

        a_bytes_local = (root_a / path_a).read_bytes()
        b_bytes_local = (root_b / path_b).read_bytes()
        assert a_bytes_local == b"A" * 8
        assert b_bytes_local == b"B" * 8

        assert not (root_a / path_b).exists(), (
            "body B's bytes leaked into body A's local root"
        )
        assert not (root_b / path_a).exists(), (
            "body A's bytes leaked into body B's local root"
        )
    finally:
        SESSION_DETAILS.reset()
        cleanup_hive_context(hive_id)
