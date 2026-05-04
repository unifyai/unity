from __future__ import annotations

import uuid

import pytest
import unify

from tests.helpers import _handle_project
from unity.common.context_registry import ContextRegistry
from unity.file_manager.filesystem_adapters.local_adapter import LocalFileSystemAdapter
from unity.file_manager.managers.file_manager import FileManager
from unity.file_manager.types.config import FilePipelineConfig
from unity.manager_registry import ManagerRegistry
from unity.session_details import SESSION_DETAILS


def _configure_spaces() -> tuple[int, int]:
    base_space_id = 40_000_000 + uuid.uuid4().int % 1_000_000_000
    space_ids = (base_space_id, base_space_id + 1)
    SESSION_DETAILS.space_ids = list(space_ids)
    SESSION_DETAILS.space_summaries = [
        {
            "space_id": space_ids[0],
            "name": "Research Library",
            "description": "Shared workspace for research files and notes.",
        },
        {
            "space_id": space_ids[1],
            "name": "Finance Library",
            "description": "Shared workspace for finance files and tables.",
        },
    ]
    ContextRegistry.clear()
    ManagerRegistry.clear()
    return space_ids


def _reset_spaces(space_ids: tuple[int, int], alias: str) -> None:
    for space_id in space_ids:
        for context in (
            f"Spaces/{space_id}/FileRecords/{alias}",
            f"Spaces/{space_id}/Files/{alias}",
        ):
            try:
                unify.delete_context(context)
            except Exception:
                pass
    SESSION_DETAILS.space_ids = []
    SESSION_DETAILS.space_summaries = []
    ContextRegistry.clear()
    ManagerRegistry.clear()


def _compact_no_embedding_config() -> FilePipelineConfig:
    config = FilePipelineConfig()
    config.embed.strategy = "off"
    config.output.return_mode = "compact"
    config.ingest.table_ingest = False
    return config


@_handle_project
def test_file_ingest_routes_to_destination_and_reads_merge_roots(tmp_path):
    space_ids = _configure_spaces()
    personal_path = tmp_path / f"personal-{uuid.uuid4().hex}.txt"
    shared_path = tmp_path / f"shared-{uuid.uuid4().hex}.txt"
    personal_path.write_text("personal research note", encoding="utf-8")
    shared_path.write_text("shared research note", encoding="utf-8")

    manager = FileManager(adapter=LocalFileSystemAdapter(None))

    try:
        manager.ingest_files(
            str(personal_path),
            config=_compact_no_embedding_config(),
        )
        manager.ingest_files(
            str(shared_path),
            config=_compact_no_embedding_config(),
            destination=f"space:{space_ids[0]}",
        )

        personal_rows = manager._data_manager.filter(
            context=manager._ctx,
            filter=f"file_path == {str(shared_path)!r}",
        )
        shared_context = f"Spaces/{space_ids[0]}/FileRecords/{manager._fs_alias}"
        shared_rows = manager._data_manager.filter(
            context=shared_context,
            filter=f"file_path == {str(shared_path)!r}",
        )

        assert personal_rows == []
        assert len(shared_rows) == 1
        shared_storage = manager.describe(str(shared_path))
        assert shared_storage.index_context == shared_context
        if shared_storage.all_context_paths:
            assert all(
                context_path.startswith(f"Spaces/{space_ids[0]}/Files/")
                for context_path in shared_storage.all_context_paths
            )

        merged_paths = {row["file_path"] for row in manager.filter_files()}
        assert str(personal_path) in merged_paths
        assert str(shared_path) in merged_paths
    finally:
        _reset_spaces(space_ids, manager._fs_alias)


@_handle_project
def test_file_invalid_destination_returns_tool_error(tmp_path):
    _configure_spaces()
    file_path = tmp_path / f"bad-{uuid.uuid4().hex}.txt"
    file_path.write_text("bad destination", encoding="utf-8")
    manager = FileManager(adapter=LocalFileSystemAdapter(None))

    try:
        outcome = manager.ingest_files(
            str(file_path),
            config=_compact_no_embedding_config(),
            destination="space:99999999",
        )
    finally:
        SESSION_DETAILS.space_ids = []
        SESSION_DETAILS.space_summaries = []
        ContextRegistry.clear()
        ManagerRegistry.clear()

    assert outcome["error_kind"] == "invalid_destination"
    assert outcome["details"]["destination"] == "space:99999999"


@_handle_project
def test_file_clear_invalid_destination_returns_tool_error():
    _configure_spaces()
    manager = FileManager(adapter=LocalFileSystemAdapter(None))

    try:
        outcome = manager.clear(destination="space:99999999")
    finally:
        SESSION_DETAILS.space_ids = []
        SESSION_DETAILS.space_summaries = []
        ContextRegistry.clear()
        ManagerRegistry.clear()

    assert outcome["error_kind"] == "invalid_destination"
    assert outcome["details"]["destination"] == "space:99999999"


@_handle_project
def test_file_save_attachment_invalid_destination_returns_tool_error():
    _configure_spaces()
    manager = FileManager(adapter=LocalFileSystemAdapter(None))

    try:
        outcome = manager.save_attachment(
            "attachment-id",
            "report.txt",
            b"report",
            auto_ingest=False,
            destination="space:99999999",
        )
    finally:
        SESSION_DETAILS.space_ids = []
        SESSION_DETAILS.space_summaries = []
        ContextRegistry.clear()
        ManagerRegistry.clear()

    assert outcome["error_kind"] == "invalid_destination"
    assert outcome["details"]["destination"] == "space:99999999"


@pytest.mark.parametrize(
    "call",
    [
        lambda manager, path: manager.sync(
            file_path=path,
            destination="space:99999999",
        ),
        lambda manager, path: manager.rename_file(
            file_id_or_path=path,
            new_name="renamed.txt",
            destination="space:99999999",
        ),
        lambda manager, path: manager.move_file(
            file_id_or_path=path,
            new_parent_path="Archive",
            destination="space:99999999",
        ),
        lambda manager, path: manager.delete_file(
            file_id_or_path=path,
            destination="space:99999999",
        ),
        lambda manager, path: manager.ingest_files(
            path,
            config=_compact_no_embedding_config(),
            destination="space:99999999",
        ),
    ],
)
@_handle_project
def test_file_write_tools_return_tool_error_for_invalid_destination(tmp_path, call):
    _configure_spaces()
    file_path = tmp_path / f"bad-{uuid.uuid4().hex}.txt"
    file_path.write_text("bad destination", encoding="utf-8")
    manager = FileManager(adapter=LocalFileSystemAdapter(None))

    try:
        outcome = call(manager, str(file_path))
    finally:
        SESSION_DETAILS.space_ids = []
        SESSION_DETAILS.space_summaries = []
        ContextRegistry.clear()
        ManagerRegistry.clear()

    assert outcome["error_kind"] == "invalid_destination"
    assert outcome["details"]["destination"] == "space:99999999"
