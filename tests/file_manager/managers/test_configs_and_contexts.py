"""
Config and context structure tests for FileManager.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.helpers import _handle_project
from unity.file_manager.types import (
    FilePipelineConfig,
    ParseConfig,
    IngestConfig,
    EmbeddingsConfig,
    FileEmbeddingSpec,
    TableEmbeddingSpec,
    FileBusinessContextSpec,
    TableBusinessContextSpec,
    BusinessContextsConfig,
    OutputConfig,
    DiagnosticsConfig,
)


@_handle_project
def test_per_file_contexts_created(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()
    p = tmp_path / "ctx_test.txt"
    p.write_text("Simple content for per-file context test.")
    name = str(p)

    res = fm.ingest_files(name)
    item = res[name]
    # All returns are now Pydantic models - use attribute access
    assert item.status == "success"

    # Use describe() to get storage map
    storage = fm.describe(file_path=name)

    # Verify per-file context was created
    assert storage.file_id is not None
    assert storage.has_document, "Expected a per-file Content context"
    assert "/Content" in storage.document.context_path


@_handle_project
def test_unified_mode_context_created(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()
    p1 = tmp_path / "u_ctx1.txt"
    p2 = tmp_path / "u_ctx2.txt"
    p1.write_text("Content A")
    p2.write_text("Content B")
    n1 = str(p1)
    n2 = str(p2)

    cfg = FilePipelineConfig(
        ingest=IngestConfig(storage_id="Docs"),
    )
    res = fm.ingest_files([n1, n2], config=cfg)
    # All returns are now Pydantic models - use attribute access
    assert res[n1].status == "success"
    assert res[n2].status == "success"

    # Unified mode: files are indexed but content goes to unified context
    # Verify both files were indexed
    storage1 = fm.describe(file_path=n1)
    storage2 = fm.describe(file_path=n2)
    assert storage1.file_id is not None
    assert storage2.file_id is not None


@pytest.mark.unit
@_handle_project
def test_ingest_files_batching_and_kwargs(file_manager, tmp_path: Path):
    """Test ingest_files respects parse-stage concurrency configuration."""
    fm = file_manager
    fm.clear()
    paths = []
    names = []
    for i in range(3):
        p = tmp_path / f"batch_{i}.txt"
        p.write_text(f"Row {i}")
        paths.append(p)
        names.append(str(p))

    cfg = FilePipelineConfig(parse=ParseConfig(max_concurrent_parses=2))
    results = fm.ingest_files(names, config=cfg)

    assert len(results) == 3
    # All returns are now Pydantic models - use attribute access
    for path, result in results.items():
        assert result.status in ("success", "error")
    assert any(r.status == "success" for r in results.values())


@_handle_project
def test_embedding_specs_smoke(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()
    p = tmp_path / "emb.txt"
    p.write_text("Content for embedding test.")
    name = str(p)

    cfg = FilePipelineConfig(
        ingest=IngestConfig(),  # default per-file storage
        embed=EmbeddingsConfig(
            strategy="after",
            file_specs=[
                FileEmbeddingSpec(
                    file_path="*",
                    context="per_file",
                    tables=[
                        TableEmbeddingSpec(
                            table="*",
                            source_columns=["summary"],
                            target_columns=["_summary_emb"],
                        ),
                    ],
                ),
            ],
        ),
    )

    res = fm.ingest_files(name, config=cfg)
    item = res[name]
    # All returns are now Pydantic models - use attribute access
    assert item.status == "success"
    # Column existence for index embeddings may be model-driven; ensure schema still accessible
    cols = fm.list_columns()
    assert "file_path" in cols and "status" in cols


@_handle_project
def test_file_pipeline_config_loads_extended_retry_policy(tmp_path: Path):
    config_data = {
        "retry": {
            "max_retries": 4,
            "retry_delay_seconds": 1.5,
            "backoff_multiplier": 3.0,
            "max_backoff_seconds": 15.0,
            "jitter_ratio": 0.0,
            "deadline_seconds": 20.0,
            "retry_mode": "all_errors",
            "fail_fast": True,
        },
    }
    config_file = tmp_path / "retry_config.json"
    config_file.write_text(json.dumps(config_data), encoding="utf-8")

    cfg = FilePipelineConfig.from_file(str(config_file))

    assert cfg.retry.max_retries == 4
    assert cfg.retry.retry_delay_seconds == 1.5
    assert cfg.retry.backoff_multiplier == 3.0
    assert cfg.retry.max_backoff_seconds == 15.0
    assert cfg.retry.jitter_ratio == 0.0
    assert cfg.retry.deadline_seconds == 20.0
    assert cfg.retry.retry_mode == "all_errors"
    assert cfg.retry.fail_fast is True


@_handle_project
def test_table_ingest_toggle_off_skips_tables_contexts(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()
    p = tmp_path / "no_tables.txt"
    p.write_text("Plain text; no tables expected.")
    name = str(p)

    cfg = FilePipelineConfig(ingest=IngestConfig(table_ingest=False))
    res = fm.ingest_files(name, config=cfg)
    item = res[name]
    # All returns are now Pydantic models - use attribute access
    assert item.status == "success"

    # Use describe() to check storage layout
    storage = fm.describe(file_path=name)
    assert storage.has_document, "Expected a per-file Content context"
    # When table_ingest=False, no tables should be created
    assert not storage.has_tables, "Expected no Tables when table_ingest=False"


# ==================== Comprehensive Config Testing ====================


def test_file_pipeline_config_defaults():
    """Test FilePipelineConfig default instantiation and values."""
    cfg = FilePipelineConfig()
    assert isinstance(cfg.parse, ParseConfig)
    assert isinstance(cfg.ingest, IngestConfig)
    assert isinstance(cfg.embed, EmbeddingsConfig)
    assert isinstance(cfg.output, OutputConfig)
    assert isinstance(cfg.diagnostics, DiagnosticsConfig)
    assert cfg.parse.max_concurrent_parses == 3
    assert isinstance(cfg.parse.backend_class_paths_by_format, dict)
    assert cfg.parse.backend_class_paths_by_format  # non-empty mapping
    assert cfg.ingest.storage_id is None  # default per-file mode
    assert cfg.embed.strategy == "after"
    assert cfg.output.return_mode == "compact"
    assert cfg.diagnostics.enable_progress is False


def test_table_embedding_spec():
    """Test TableEmbeddingSpec creation and validation."""
    table_spec = TableEmbeddingSpec(
        table="TestTable",
        source_columns=["col1", "col2"],
        target_columns=["_col1_emb", "_col2_emb"],
    )
    assert table_spec.table == "TestTable"
    assert len(table_spec.source_columns) == 2
    assert len(table_spec.target_columns) == 2


def test_file_embedding_spec():
    """Test FileEmbeddingSpec creation and validation."""
    table_spec1 = TableEmbeddingSpec(
        table="Table1",
        source_columns=["col1"],
        target_columns=["_col1_emb"],
    )
    table_spec2 = TableEmbeddingSpec(
        table="Table2",
        source_columns=["col2"],
        target_columns=["_col2_emb"],
    )
    file_spec = FileEmbeddingSpec(
        file_path="test.xlsx",
        context="per_file_table",
        tables=[table_spec1, table_spec2],
    )
    assert file_spec.file_path == "test.xlsx"
    assert file_spec.context == "per_file_table"
    assert len(file_spec.tables) == 2
    assert file_spec.tables[0].table == "Table1"
    assert file_spec.tables[1].table == "Table2"


def test_table_embedding_spec_validation():
    """Test that TableEmbeddingSpec validates column list lengths match."""
    with pytest.raises(ValueError, match="must have the same length"):
        TableEmbeddingSpec(
            table="TestTable",
            source_columns=["col1", "col2"],
            target_columns=["_col1_emb"],  # Mismatch
        )


def test_file_embedding_spec_validation():
    """Test that FileEmbeddingSpec validates tables list is not empty."""
    # Pydantic validation happens first, so we check for the Pydantic error message
    with pytest.raises(ValueError, match="List should have at least 1 item"):
        FileEmbeddingSpec(
            file_path="test.xlsx",
            context="per_file_table",
            tables=[],  # Empty list
        )


def test_table_business_context_spec():
    """Test TableBusinessContextSpec creation and validation."""
    table_spec = TableBusinessContextSpec(
        table="Sheet1",
        column_descriptions={"col1": "Description 1", "col2": "Description 2"},
        table_description="Table description",
        table_rules=["Rule 1 about multiple columns", "Rule 2"],
    )
    assert table_spec.table == "Sheet1"
    assert len(table_spec.column_descriptions) == 2
    assert table_spec.table_description == "Table description"
    assert len(table_spec.table_rules) == 2
    assert table_spec.table_rules[0] == "Rule 1 about multiple columns"


@pytest.mark.unit
def test_file_business_context_spec():
    """Test FileBusinessContextSpec creation and validation with multiple tables."""
    table_spec1 = TableBusinessContextSpec(
        table="Sheet1",
        column_descriptions={"col1": "Description 1"},
        table_description="Table 1 description",
        table_rules=["Table 1 rule"],
    )
    table_spec2 = TableBusinessContextSpec(
        table="Sheet2",
        column_descriptions={"col2": "Description 2"},
        table_description="Table 2 description",
    )
    fc = FileBusinessContextSpec(
        file_path="/path/to/file.xlsx",
        file_rules=["File-level rule about cross-table data"],
        table_contexts=[table_spec1, table_spec2],
    )
    assert fc.file_path == "/path/to/file.xlsx"
    assert len(fc.file_rules) == 1
    assert len(fc.table_contexts) == 2
    assert fc.table_contexts[0].table == "Sheet1"
    assert fc.table_contexts[1].table == "Sheet2"


@pytest.mark.unit
def test_business_contexts_config():
    """Test BusinessContextsConfig with global_rules and file_contexts."""
    table_spec = TableBusinessContextSpec(
        table="Sheet1",
        column_descriptions={"col1": "Description 1"},
        table_rules=["Table rule"],
    )
    file_context = FileBusinessContextSpec(
        file_path="/path/to/file.xlsx",
        file_rules=["File rule"],
        table_contexts=[table_spec],
    )
    config = BusinessContextsConfig(
        global_rules=["Global rule 1", "Global rule 2"],
        file_contexts=[file_context],
    )
    assert len(config.global_rules) == 2
    assert len(config.file_contexts) == 1
    assert config.file_contexts[0].file_path == "/path/to/file.xlsx"
    assert len(config.file_contexts[0].table_contexts) == 1


def test_config_from_file_empty(tmp_path: Path):
    """Test loading an empty config file (all defaults)."""

    config_file = tmp_path / "empty_config.json"
    config_file.write_text("{}")

    cfg = FilePipelineConfig.from_file(str(config_file))
    assert isinstance(cfg, FilePipelineConfig)
    assert cfg.parse.max_concurrent_parses == 3
    assert cfg.ingest.storage_id is None  # default per-file mode


def test_config_from_file_partial(tmp_path: Path):
    """Test loading config with only some sections."""
    import json

    config_data = {
        "parse": {"batch_size": 10},
        "output": {"return_mode": "full"},
    }
    config_file = tmp_path / "partial_config.json"
    config_file.write_text(json.dumps(config_data))

    cfg = FilePipelineConfig.from_file(str(config_file))
    assert cfg.parse.max_concurrent_parses == 10
    assert cfg.output.return_mode == "full"
    # Other sections should have defaults
    assert cfg.ingest.storage_id is None  # default per-file mode
    assert cfg.embed.strategy == "after"


def test_config_from_file_business_contexts(tmp_path: Path):
    """Test loading config with business contexts (new structure)."""
    import json

    config_data = {
        "ingest": {
            "business_contexts": {
                "global_rules": ["Global rule 1", "Global rule 2"],
                "file_contexts": [
                    {
                        "file_path": "/path/to/file.xlsx",
                        "file_rules": ["File-level rule"],
                        "table_contexts": [
                            {
                                "table": "Sheet1",
                                "column_descriptions": {
                                    "col1": "Description 1",
                                    "col2": "Description 2",
                                },
                                "table_description": "Table description",
                                "table_rules": ["Table rule 1", "Table rule 2"],
                            },
                        ],
                    },
                ],
            },
        },
    }
    config_file = tmp_path / "business_contexts.json"
    config_file.write_text(json.dumps(config_data))

    cfg = FilePipelineConfig.from_file(str(config_file))
    assert cfg.ingest.business_contexts is not None
    assert len(cfg.ingest.business_contexts.global_rules) == 2
    assert len(cfg.ingest.business_contexts.file_contexts) == 1
    fc = cfg.ingest.business_contexts.file_contexts[0]
    assert fc.file_path == "/path/to/file.xlsx"
    assert len(fc.file_rules) == 1
    assert len(fc.table_contexts) == 1
    table_spec = fc.table_contexts[0]
    assert table_spec.table == "Sheet1"
    assert table_spec.column_descriptions["col1"] == "Description 1"
    assert table_spec.table_description == "Table description"
    assert len(table_spec.table_rules) == 2


@pytest.mark.unit
def test_config_from_file_business_contexts_legacy(tmp_path: Path):
    """Test loading config with legacy business contexts (list format)."""
    import json

    # Legacy format: business_contexts as a list
    config_data = {
        "ingest": {
            "business_contexts": [
                {
                    "file_path": "/path/to/file.xlsx",
                    "tables": [
                        {
                            "table": "Sheet1",
                            "column_descriptions": {
                                "col1": "Description 1",
                            },
                            "table_description": "Table description",
                        },
                    ],
                },
            ],
        },
    }
    config_file = tmp_path / "business_contexts_legacy.json"
    config_file.write_text(json.dumps(config_data))

    cfg = FilePipelineConfig.from_file(str(config_file))
    # Legacy format should be converted to new structure
    assert cfg.ingest.business_contexts is not None
    assert (
        len(cfg.ingest.business_contexts.global_rules) == 0
    )  # No global rules in legacy
    assert len(cfg.ingest.business_contexts.file_contexts) == 1
    fc = cfg.ingest.business_contexts.file_contexts[0]
    assert fc.file_path == "/path/to/file.xlsx"
    assert len(fc.table_contexts) == 1
    assert fc.table_contexts[0].table == "Sheet1"


def test_config_from_file_embed_specs(tmp_path: Path):
    """Test loading config with embedding specs."""
    import json

    config_data = {
        "embed": {
            "strategy": "along",
            "specs": [
                {
                    "context": "per_file_table",
                    "table": "TestTable",
                    "source_columns": ["col1", "col2"],
                    "target_columns": ["_col1_emb", "_col2_emb"],
                },
            ],
        },
    }
    config_file = tmp_path / "embed_config.json"
    config_file.write_text(json.dumps(config_data))

    # Legacy "specs" format is ignored - file_specs will be empty
    cfg = FilePipelineConfig.from_file(str(config_file))
    assert cfg.embed.strategy == "along"
    # Legacy specs are ignored, so file_specs should be empty
    assert len(cfg.embed.file_specs) == 0


def test_config_from_file_multiple_specs(tmp_path: Path):
    """Test loading config with multiple embedding specs (legacy format)."""
    import json

    config_data = {
        "embed": {
            "specs": [
                {
                    "context": "per_file",
                    "source_columns": ["summary"],
                    "target_columns": ["_summary_emb"],
                },
                {
                    "context": "per_file_table",
                    "table": "Table1",
                    "source_columns": ["col1", "col2"],
                    "target_columns": ["_col1_emb", "_col2_emb"],
                },
            ],
        },
    }
    config_file = tmp_path / "multiple_specs.json"
    config_file.write_text(json.dumps(config_data))

    # Legacy "specs" format is ignored - file_specs will be empty
    cfg = FilePipelineConfig.from_file(str(config_file))
    # Legacy specs are ignored, so file_specs should be empty
    assert len(cfg.embed.file_specs) == 0


def test_config_from_file_file_specs(tmp_path: Path):
    """Test loading config with new file_specs format."""
    import json

    config_data = {
        "embed": {
            "file_specs": [
                {
                    "file_path": "file1.xlsx",
                    "context": "per_file_table",
                    "tables": [
                        {
                            "table": "Sheet1",
                            "source_columns": ["col1", "col2"],
                            "target_columns": ["_col1_emb", "_col2_emb"],
                        },
                        {
                            "table": "Sheet2",
                            "source_columns": ["col3"],
                            "target_columns": ["_col3_emb"],
                        },
                    ],
                },
                {
                    "file_path": "file2.xlsx",
                    "context": "per_file",
                    "tables": [
                        {
                            "table": "*",
                            "source_columns": ["summary"],
                            "target_columns": ["_summary_emb"],
                        },
                    ],
                },
            ],
        },
    }
    config_file = tmp_path / "file_specs_config.json"
    config_file.write_text(json.dumps(config_data))

    cfg = FilePipelineConfig.from_file(str(config_file))
    assert len(cfg.embed.file_specs) == 2
    assert cfg.embed.file_specs[0].file_path == "file1.xlsx"
    assert cfg.embed.file_specs[0].context == "per_file_table"
    assert len(cfg.embed.file_specs[0].tables) == 2
    assert cfg.embed.file_specs[0].tables[0].table == "Sheet1"
    assert cfg.embed.file_specs[0].tables[1].table == "Sheet2"
    assert cfg.embed.file_specs[1].file_path == "file2.xlsx"
    assert cfg.embed.file_specs[1].context == "per_file"


def test_config_from_file_full(tmp_path: Path):
    """Test loading a full config with all sections."""
    import json

    config_data = {
        # batch_size is supported as a back-compat alias for max_concurrent_parses.
        # parser_kwargs/plugins are ignored (legacy/removed); this test ensures unknown keys don't break loading.
        "parse": {"batch_size": 5, "parser_kwargs": {"key": "value"}},
        "ingest": {
            "storage_id": "FullTest",
            "table_rows_batch_size": 3000,
            "business_contexts": {
                "global_rules": ["Global rule"],
                "file_contexts": [
                    {
                        "file_path": "/path/to/file.xlsx",
                        "file_rules": [],
                        "table_contexts": [
                            {
                                "table": "Sheet1",
                                "column_descriptions": {"col1": "Desc1"},
                            },
                        ],
                    },
                ],
            },
        },
        "embed": {
            "strategy": "after",
            "file_specs": [
                {
                    "file_path": "*",
                    "context": "per_file_table",
                    "tables": [
                        {
                            "table": "Sheet1",
                            "source_columns": ["col1"],
                            "target_columns": ["_col1_emb"],
                        },
                    ],
                },
            ],
        },
        "plugins": {
            "pre_parse": ["module.function"],
            "plugin_kwargs": {"module.function": {"arg": "value"}},
        },
        "output": {"return_mode": "full"},
        "diagnostics": {"enable_progress": True},
    }
    config_file = tmp_path / "full_config.json"
    config_file.write_text(json.dumps(config_data))

    cfg = FilePipelineConfig.from_file(str(config_file))
    assert cfg.parse.max_concurrent_parses == 5
    assert cfg.ingest.storage_id is not None  # unified mode (shared storage)
    assert cfg.embed.strategy == "after"
    assert cfg.output.return_mode == "full"
    assert cfg.diagnostics.enable_progress is True
    assert cfg.ingest.business_contexts is not None
    assert len(cfg.ingest.business_contexts.file_contexts) == 1
    assert len(cfg.embed.file_specs) == 1


def test_config_from_file_nonexistent(tmp_path: Path):
    """Test that from_file raises FileNotFoundError for nonexistent file."""
    nonexistent = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError):
        FilePipelineConfig.from_file(str(nonexistent))


def test_config_from_file_invalid_json(tmp_path: Path):
    """Test that from_file raises ValueError for invalid JSON."""
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("{ invalid json }")
    with pytest.raises(ValueError, match="Invalid JSON"):
        FilePipelineConfig.from_file(str(invalid_file))


def test_config_from_file_invalid_spec(tmp_path: Path):
    """Test that from_file validates embedding spec structure."""
    import json

    config_data = {
        "embed": {
            "file_specs": [
                {
                    "file_path": "*",
                    "context": "per_file_table",
                    "tables": [
                        {
                            "table": "TestTable",
                            "source_columns": ["col1", "col2"],
                            "target_columns": ["_col1_emb"],  # Mismatch - should fail
                        },
                    ],
                },
            ],
        },
    }
    config_file = tmp_path / "invalid_spec.json"
    config_file.write_text(json.dumps(config_data))

    with pytest.raises(ValueError, match="must have the same length"):
        FilePipelineConfig.from_file(str(config_file))


@_handle_project
def test_business_context_applied_during_ingestion(file_manager, tmp_path: Path):
    """Test that business context is applied during table ingestion."""
    import json

    fm = file_manager
    fm.clear()

    # Create config file with business context (new structure)
    config_data = {
        "ingest": {
            "business_contexts": {
                "global_rules": [],
                "file_contexts": [
                    {
                        "file_path": "test_file.xlsx",
                        "file_rules": [],
                        "table_contexts": [
                            {
                                "table": "Sheet1",
                                "column_descriptions": {
                                    "Name": "Person's full name",
                                    "Age": "Person's age in years",
                                },
                                "table_description": "Test table with person data",
                                "table_rules": ["Rule about Name and Age columns"],
                            },
                        ],
                    },
                ],
            },
        },
    }
    config_file = tmp_path / "test_config.json"
    config_file.write_text(json.dumps(config_data))

    cfg = FilePipelineConfig.from_file(str(config_file))
    assert cfg.ingest.business_contexts is not None
    assert len(cfg.ingest.business_contexts.file_contexts) == 1
    fc = cfg.ingest.business_contexts.file_contexts[0]
    assert fc.file_path == "test_file.xlsx"
    assert len(fc.table_contexts) == 1
    assert fc.table_contexts[0].table == "Sheet1"
    assert len(fc.table_contexts[0].table_rules) == 1


@_handle_project
def test_embedding_specs_with_multiple_columns_per_table(file_manager, tmp_path: Path):
    """Test embedding specs with multiple columns per table work correctly."""
    fm = file_manager
    fm.clear()

    cfg = FilePipelineConfig()
    cfg.embed.strategy = "along"
    cfg.embed.file_specs = [
        FileEmbeddingSpec(
            file_path="*",
            context="per_file_table",
            tables=[
                TableEmbeddingSpec(
                    table="Sheet1",
                    source_columns=["Name", "City", "Country"],
                    target_columns=["_Name_emb", "_City_emb", "_Country_emb"],
                ),
            ],
        ),
    ]

    # This is a pure config validation test; the actual embedding/execution behavior
    # is covered in the ingest + embed integration tests.

    # Verify config is valid
    assert len(cfg.embed.file_specs) == 1
    assert len(cfg.embed.file_specs[0].tables) == 1
    assert len(cfg.embed.file_specs[0].tables[0].source_columns) == 3
    assert len(cfg.embed.file_specs[0].tables[0].target_columns) == 3


def test_config_programmatic_vs_file_equivalence(tmp_path: Path):
    """Test that programmatic config and file config produce equivalent results."""
    import json

    # Create programmatic config
    cfg_programmatic = FilePipelineConfig()
    cfg_programmatic.ingest.table_rows_batch_size = 2000
    cfg_programmatic.embed.strategy = "along"
    cfg_programmatic.embed.file_specs = [
        FileEmbeddingSpec(
            file_path="*",
            context="per_file_table",
            tables=[
                TableEmbeddingSpec(
                    table="Sheet1",
                    source_columns=["col1"],
                    target_columns=["_col1_emb"],
                ),
            ],
        ),
    ]
    cfg_programmatic.output.return_mode = "compact"

    # Create equivalent config file
    config_data = {
        "ingest": {"table_rows_batch_size": 2000},
        "embed": {
            "strategy": "along",
            "file_specs": [
                {
                    "file_path": "*",
                    "context": "per_file_table",
                    "tables": [
                        {
                            "table": "Sheet1",
                            "source_columns": ["col1"],
                            "target_columns": ["_col1_emb"],
                        },
                    ],
                },
            ],
        },
        "output": {"return_mode": "compact"},
    }
    config_file = tmp_path / "equivalent_config.json"
    config_file.write_text(json.dumps(config_data))

    # Load file config
    cfg_file = FilePipelineConfig.from_file(str(config_file))

    # Compare key attributes
    assert (
        cfg_programmatic.ingest.table_rows_batch_size
        == cfg_file.ingest.table_rows_batch_size
    )
    assert cfg_programmatic.embed.strategy == cfg_file.embed.strategy
    assert cfg_programmatic.output.return_mode == cfg_file.output.return_mode
    assert len(cfg_programmatic.embed.file_specs) == len(cfg_file.embed.file_specs)
    assert (
        cfg_programmatic.embed.file_specs[0].tables[0].table
        == cfg_file.embed.file_specs[0].tables[0].table
    )


def test_config_all_sections_populated(tmp_path: Path):
    """Test config file with all possible sections populated."""
    import json

    config_data = {
        "parse": {
            "batch_size": 10,
            "parser_kwargs": {"custom_option": "value", "another": 123},
        },
        "ingest": {
            "storage_id": "AllSectionsTest",
            "table_rows_batch_size": 5000,
            "content_rows_batch_size": 3000,
            "business_contexts": {
                "global_rules": ["Cross-file rule 1"],
                "file_contexts": [
                    {
                        "file_path": "/full/path/to/file.xlsx",
                        "file_rules": ["File-level rule"],
                        "table_contexts": [
                            {
                                "table": "MainSheet",
                                "column_descriptions": {
                                    "id": "Unique identifier",
                                    "name": "Item name",
                                },
                                "table_description": "Main data table",
                                "table_rules": ["Table rule 1", "Table rule 2"],
                            },
                        ],
                    },
                ],
            },
        },
        "embed": {
            "strategy": "after",
            "large_threshold": 10000,
            "hooks_per_chunk": True,
            "file_specs": [
                {
                    "file_path": "*",
                    "context": "per_file_table",
                    "tables": [
                        {
                            "table": "MainSheet",
                            "source_columns": ["name"],
                            "target_columns": ["_name_emb"],
                        },
                    ],
                },
            ],
        },
        "plugins": {
            "pre_parse": ["module1.pre_parse"],
            "post_parse": ["module1.post_parse"],
            "pre_ingest": ["module2.pre_ingest"],
            "post_ingest": ["module2.post_ingest"],
            "pre_embed": ["module3.pre_embed"],
            "post_embed": ["module3.post_embed"],
            "plugin_kwargs": {
                "module1.pre_parse": {"arg1": "value1"},
                "module2.pre_ingest": {"arg2": 42},
            },
        },
        "output": {"return_mode": "full"},
        "diagnostics": {"enable_progress": True},
    }
    config_file = tmp_path / "all_sections.json"
    config_file.write_text(json.dumps(config_data))

    cfg = FilePipelineConfig.from_file(str(config_file))

    # Verify parse
    assert cfg.parse.max_concurrent_parses == 10

    # Verify ingest
    assert cfg.ingest.storage_id == "AllSectionsTest"
    assert cfg.ingest.table_rows_batch_size == 5000
    assert cfg.ingest.business_contexts is not None
    assert len(cfg.ingest.business_contexts.global_rules) == 1
    assert len(cfg.ingest.business_contexts.file_contexts) == 1
    fc = cfg.ingest.business_contexts.file_contexts[0]
    assert len(fc.file_rules) == 1
    assert len(fc.table_contexts) == 1
    assert len(fc.table_contexts[0].table_rules) == 2

    # Verify embed
    assert cfg.embed.strategy == "after"
    # The test config uses "file_specs" format (updated from legacy)
    assert len(cfg.embed.file_specs) == 1
    assert cfg.embed.file_specs[0].file_path == "*"
    assert cfg.embed.file_specs[0].context == "per_file_table"
    assert len(cfg.embed.file_specs[0].tables) == 1
    assert cfg.embed.file_specs[0].tables[0].table == "MainSheet"

    # Verify output
    assert cfg.output.return_mode == "full"

    # Verify diagnostics
    assert cfg.diagnostics.enable_progress is True


@_handle_project
def test_complete_pipeline_with_business_context_and_consolidated_embeddings(
    file_manager,
    tmp_path: Path,
):
    """Test complete pipeline using JSON config file with business context and consolidated embedding specs."""
    import json

    fm = file_manager
    fm.clear()

    # Create a CSV file
    csv_path = tmp_path / "sales_data.csv"
    csv_path.write_text(
        "OrderID,Customer,Product,Quantity,Price,Date\n"
        "O001,Acme Corp,Widget,10,19.99,2025-01-15\n"
        "O002,Tech Inc,Gadget,5,29.99,2025-01-16\n",
        encoding="utf-8",
    )
    display_name = str(csv_path)

    # Create comprehensive config file demonstrating all new features
    config_data = {
        "parse": {"batch_size": 3},
        "ingest": {
            "mode": "per_file",
            "table_rows_batch_size": 1000,
            "business_contexts": {
                "global_rules": [],
                "file_contexts": [
                    {
                        "file_path": display_name,
                        "file_rules": [],
                        "table_contexts": [
                            {
                                "table": "sales_data",
                                "column_descriptions": {
                                    "OrderID": "Unique order identifier",
                                    "Customer": "Customer company name",
                                    "Product": "Product name or SKU",
                                    "Quantity": "Number of units ordered",
                                    "Price": "Unit price in USD",
                                    "Date": "Order date in YYYY-MM-DD format",
                                },
                                "table_description": "Sales order transactions with customer and product details",
                                "table_rules": [],
                            },
                        ],
                    },
                ],
            },
        },
        "embed": {
            "strategy": "along",
            "file_specs": [
                {
                    "file_path": "*",
                    "context": "per_file_table",
                    "tables": [
                        {
                            "table": "sales_data",
                            "source_columns": ["Customer", "Product", "OrderID"],
                            "target_columns": [
                                "_Customer_emb",
                                "_Product_emb",
                                "_OrderID_emb",
                            ],
                        },
                    ],
                },
            ],
        },
        "output": {"return_mode": "compact"},
        "diagnostics": {"enable_progress": False},
    }
    config_file = tmp_path / "complete_pipeline_config.json"
    config_file.write_text(json.dumps(config_data))

    # Load config from file
    cfg = FilePipelineConfig.from_file(str(config_file))

    # Verify config loaded correctly
    assert cfg.ingest.business_contexts is not None
    assert len(cfg.ingest.business_contexts.file_contexts) == 1
    fc = cfg.ingest.business_contexts.file_contexts[0]
    assert fc.file_path == display_name
    assert len(fc.table_contexts) == 1
    table_spec = fc.table_contexts[0]
    assert table_spec.table == "sales_data"
    assert len(table_spec.column_descriptions) == 6

    assert len(cfg.embed.file_specs) == 1
    assert len(cfg.embed.file_specs[0].tables) == 1
    assert len(cfg.embed.file_specs[0].tables[0].source_columns) == 3
    assert len(cfg.embed.file_specs[0].tables[0].target_columns) == 3
    assert cfg.embed.file_specs[0].tables[0].source_columns == [
        "Customer",
        "Product",
        "OrderID",
    ]

    # Parse file with config
    result = fm.ingest_files(display_name, config=cfg)
    item = result[display_name]
    # All returns are now Pydantic models - use attribute access
    assert item.status == "success"


@_handle_project
def test_excel_with_business_context_and_multiple_column_embeddings(
    file_manager,
    tmp_path: Path,
):
    """Test Excel file parsing with business context and consolidated multi-column embedding spec."""
    import json

    fm = file_manager
    fm.clear()

    # Create an Excel-like scenario (using CSV as proxy for Excel table structure)
    excel_path = tmp_path / "inventory.xlsx"
    # For this test, we'll simulate with CSV since we don't have Excel writer
    # In real usage, this would be an actual .xlsx file
    csv_path = tmp_path / "inventory.csv"
    csv_path.write_text(
        "SKU,ProductName,Category,StockLevel,UnitPrice,Supplier\n"
        "SKU001,Widget A,Electronics,150,24.99,SupplierX\n"
        "SKU002,Gadget B,Electronics,75,39.99,SupplierY\n"
        "SKU003,Tool C,Hardware,200,14.99,SupplierZ\n",
        encoding="utf-8",
    )
    display_name = str(csv_path)

    # Create config file similar to repairs_file_pipeline_config.json pattern
    config_data = {
        "ingest": {
            "table_rows_batch_size": 2000,
            "business_contexts": {
                "global_rules": [],
                "file_contexts": [
                    {
                        "file_path": display_name,
                        "file_rules": [],
                        "table_contexts": [
                            {
                                "table": "inventory",
                                "column_descriptions": {
                                    "SKU": "Stock Keeping Unit - unique product identifier",
                                    "ProductName": "Display name of the product",
                                    "Category": "Product category classification",
                                    "StockLevel": "Current inventory quantity available",
                                    "UnitPrice": "Price per unit in USD",
                                    "Supplier": "Supplier company name",
                                },
                                "table_description": "Inventory management table tracking product stock levels and pricing",
                                "table_rules": [],
                            },
                        ],
                    },
                ],
            },
        },
        "embed": {
            "strategy": "along",
            "file_specs": [
                {
                    "file_path": "*",
                    "context": "per_file_table",
                    "tables": [
                        {
                            "table": "inventory",
                            "source_columns": [
                                "ProductName",
                                "Category",
                                "Supplier",
                            ],
                            "target_columns": [
                                "_ProductName_emb",
                                "_Category_emb",
                                "_Supplier_emb",
                            ],
                        },
                    ],
                },
            ],
        },
        "output": {"return_mode": "compact"},
    }
    config_file = tmp_path / "inventory_config.json"
    config_file.write_text(json.dumps(config_data))

    # Load and use config
    cfg = FilePipelineConfig.from_file(str(config_file))

    # Verify consolidated embedding spec
    assert len(cfg.embed.file_specs) == 1
    file_spec = cfg.embed.file_specs[0]
    assert len(file_spec.tables) == 1
    embed_table_spec = file_spec.tables[0]
    assert len(embed_table_spec.source_columns) == 3
    assert len(embed_table_spec.target_columns) == 3
    assert embed_table_spec.source_columns == ["ProductName", "Category", "Supplier"]

    # Verify business context
    assert cfg.ingest.business_contexts is not None
    assert len(cfg.ingest.business_contexts.file_contexts) == 1
    fc = cfg.ingest.business_contexts.file_contexts[0]
    assert fc.file_path == display_name
    assert len(fc.table_contexts) == 1
    table_spec = fc.table_contexts[0]
    assert table_spec.table == "inventory"
    assert "SKU" in table_spec.column_descriptions
    assert table_spec.table_description is not None

    # Parse with config
    result = fm.ingest_files(display_name, config=cfg)
    item = result[display_name]
    # All returns are now Pydantic models - use attribute access
    assert item.status == "success"


@_handle_project
def test_multiple_tables_per_file_business_context(file_manager, tmp_path: Path):
    """Test FileBusinessContextSpec with multiple tables per file."""
    import json

    fm = file_manager
    fm.clear()

    # Create a CSV file (simulating Excel with multiple sheets)
    csv_path = tmp_path / "multi_table.csv"
    csv_path.write_text(
        "ID,Name,Value\n1,Item1,100\n2,Item2,200\n",
        encoding="utf-8",
    )
    display_name = str(csv_path)

    # Create config with multiple tables for the same file
    config_data = {
        "ingest": {
            "business_contexts": {
                "global_rules": [],
                "file_contexts": [
                    {
                        "file_path": display_name,
                        "file_rules": ["Cross-table rule for this file"],
                        "table_contexts": [
                            {
                                "table": "multi_table",
                                "column_descriptions": {
                                    "ID": "Unique identifier",
                                    "Name": "Item name",
                                    "Value": "Numeric value",
                                },
                                "table_description": "Main data table",
                                "table_rules": ["Table-specific rule"],
                            },
                            {
                                "table": "Summary",
                                "column_descriptions": {
                                    "Total": "Sum of all values",
                                },
                                "table_description": "Summary statistics",
                            },
                        ],
                    },
                ],
            },
        },
    }
    config_file = tmp_path / "multi_table_config.json"
    config_file.write_text(json.dumps(config_data))

    cfg = FilePipelineConfig.from_file(str(config_file))

    # Verify multiple tables in single FileBusinessContextSpec
    assert cfg.ingest.business_contexts is not None
    assert len(cfg.ingest.business_contexts.file_contexts) == 1
    fc = cfg.ingest.business_contexts.file_contexts[0]
    assert fc.file_path == display_name
    assert len(fc.file_rules) == 1
    assert len(fc.table_contexts) == 2
    assert fc.table_contexts[0].table == "multi_table"
    assert fc.table_contexts[1].table == "Summary"
    assert "ID" in fc.table_contexts[0].column_descriptions
    assert "Total" in fc.table_contexts[1].column_descriptions
    assert len(fc.table_contexts[0].table_rules) == 1
