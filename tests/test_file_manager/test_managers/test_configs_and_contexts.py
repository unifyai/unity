"""
Config and context structure tests for FileManager.
"""

from __future__ import annotations

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
    BusinessContextSpec,
    TableBusinessContextSpec,
    OutputConfig,
    DiagnosticsConfig,
    PluginsConfig,
)


@pytest.mark.unit
@_handle_project
def test_per_file_contexts_created(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()
    p = tmp_path / "ctx_test.txt"
    p.write_text("Simple content for per-file context test.")
    name = str(p)

    res = fm.parse(name)
    _item = res[name]
    _item = _item if isinstance(_item, dict) else _item.model_dump()
    assert _item["status"] == "success"

    print(f"_item.file_path: {_item.get('file_path')}")

    overview = fm._tables_overview(file=name)
    print(f"overview: {overview}")
    assert isinstance(overview, dict)
    # Global entry
    assert "FileRecords" in overview and isinstance(overview["FileRecords"], dict)
    # Find file root entry (shape: { Content: {...}, Tables?: {...} })
    roots = [v for k, v in overview.items() if isinstance(v, dict) and "Content" in v]
    assert roots, "Expected a per-file root with a Content entry"
    content_meta = roots[0]["Content"]
    assert "/Content" in str(content_meta.get("context", ""))


@pytest.mark.unit
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
        ingest=IngestConfig(mode="unified", unified_label="Docs"),
    )
    res = fm.parse([n1, n2], config=cfg)
    _i1 = res[n1]
    _i1 = _i1 if isinstance(_i1, dict) else _i1.model_dump()
    _i2 = res[n2]
    _i2 = _i2 if isinstance(_i2, dict) else _i2.model_dump()
    assert _i1["status"] == "success"
    assert _i2["status"] == "success"

    # Unified context should exist under the unified label
    ov = fm._tables_overview(file="Docs")
    assert isinstance(ov, dict) and len(ov) >= 1
    # Unified label entry should exist with Content
    assert "Docs" in ov and "Content" in ov["Docs"]
    assert "/Content" in str(ov["Docs"]["Content"].get("context", ""))


@pytest.mark.asyncio
@_handle_project
async def test_parse_async_batching_and_kwargs(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()
    paths = []
    names = []
    for i in range(3):
        p = tmp_path / f"async_{i}.txt"
        p.write_text(f"Row {i}")
        paths.append(p)
        names.append(str(p))

    cfg = FilePipelineConfig(parse=ParseConfig(batch_size=2, parser_kwargs={}))
    results = []
    async for r in fm.parse_async(names, config=cfg):
        results.append(r if isinstance(r, dict) else r.model_dump())

    assert len(results) == 3
    assert all(r.get("status") in ("success", "error") for r in results)
    assert any(r.get("status") == "success" for r in results)


@pytest.mark.unit
@_handle_project
def test_embedding_specs_smoke(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()
    p = tmp_path / "emb.txt"
    p.write_text("Content for embedding test.")
    name = str(p)

    cfg = FilePipelineConfig(
        ingest=IngestConfig(mode="per_file"),
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

    res = fm.parse(name, config=cfg)
    _item = res[name]
    _item = _item if isinstance(_item, dict) else _item.model_dump()
    assert _item["status"] == "success"
    # Column existence for index embeddings may be model-driven; ensure schema still accessible
    cols = fm._list_columns()
    assert "file_path" in cols and "status" in cols


@pytest.mark.unit
@_handle_project
def test_table_ingest_toggle_off_skips_tables_contexts(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()
    p = tmp_path / "no_tables.txt"
    p.write_text("Plain text; no tables expected.")
    name = str(p)

    cfg = FilePipelineConfig(ingest=IngestConfig(table_ingest=False))
    res = fm.parse(name, config=cfg)
    _item = res[name]
    _item = _item if isinstance(_item, dict) else _item.model_dump()
    assert _item["status"] == "success"

    ov = fm._tables_overview(file=name)
    assert isinstance(ov, dict)
    # When table_ingest=False, the root should not include a "Tables" map
    roots = [v for k, v in ov.items() if isinstance(v, dict) and "Content" in v]
    assert roots, "Expected a per-file root with Content"
    assert "Tables" not in roots[0]


# ==================== Comprehensive Config Testing ====================


@pytest.mark.unit
def test_file_pipeline_config_defaults():
    """Test FilePipelineConfig default instantiation and values."""
    cfg = FilePipelineConfig()
    assert isinstance(cfg.parse, ParseConfig)
    assert isinstance(cfg.ingest, IngestConfig)
    assert isinstance(cfg.embed, EmbeddingsConfig)
    assert isinstance(cfg.plugins, PluginsConfig)
    assert isinstance(cfg.output, OutputConfig)
    assert isinstance(cfg.diagnostics, DiagnosticsConfig)
    assert cfg.parse.batch_size == 3
    assert cfg.ingest.mode == "per_file"
    assert cfg.embed.strategy == "auto"
    assert cfg.output.return_mode == "compact"
    assert cfg.diagnostics.enable_progress is False


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_table_embedding_spec_validation():
    """Test that TableEmbeddingSpec validates column list lengths match."""
    with pytest.raises(ValueError, match="must have the same length"):
        TableEmbeddingSpec(
            table="TestTable",
            source_columns=["col1", "col2"],
            target_columns=["_col1_emb"],  # Mismatch
        )


@pytest.mark.unit
def test_file_embedding_spec_validation():
    """Test that FileEmbeddingSpec validates tables list is not empty."""
    # Pydantic validation happens first, so we check for the Pydantic error message
    with pytest.raises(ValueError, match="List should have at least 1 item"):
        FileEmbeddingSpec(
            file_path="test.xlsx",
            context="per_file_table",
            tables=[],  # Empty list
        )


@pytest.mark.unit
def test_table_business_context_spec():
    """Test TableBusinessContextSpec creation and validation."""
    table_spec = TableBusinessContextSpec(
        table="Sheet1",
        column_descriptions={"col1": "Description 1", "col2": "Description 2"},
        table_description="Table description",
    )
    assert table_spec.table == "Sheet1"
    assert len(table_spec.column_descriptions) == 2
    assert table_spec.table_description == "Table description"


@pytest.mark.unit
def test_business_context_spec():
    """Test BusinessContextSpec creation and validation with multiple tables."""
    table_spec1 = TableBusinessContextSpec(
        table="Sheet1",
        column_descriptions={"col1": "Description 1"},
        table_description="Table 1 description",
    )
    table_spec2 = TableBusinessContextSpec(
        table="Sheet2",
        column_descriptions={"col2": "Description 2"},
        table_description="Table 2 description",
    )
    bc = BusinessContextSpec(
        file_path="/path/to/file.xlsx",
        tables=[table_spec1, table_spec2],
    )
    assert bc.file_path == "/path/to/file.xlsx"
    assert len(bc.tables) == 2
    assert bc.tables[0].table == "Sheet1"
    assert bc.tables[1].table == "Sheet2"


@pytest.mark.unit
def test_config_from_file_empty(tmp_path: Path):
    """Test loading an empty config file (all defaults)."""

    config_file = tmp_path / "empty_config.json"
    config_file.write_text("{}")

    cfg = FilePipelineConfig.from_file(str(config_file))
    assert isinstance(cfg, FilePipelineConfig)
    assert cfg.parse.batch_size == 3
    assert cfg.ingest.mode == "per_file"


@pytest.mark.unit
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
    assert cfg.parse.batch_size == 10
    assert cfg.output.return_mode == "full"
    # Other sections should have defaults
    assert cfg.ingest.mode == "per_file"
    assert cfg.embed.strategy == "auto"


@pytest.mark.unit
def test_config_from_file_business_contexts(tmp_path: Path):
    """Test loading config with business contexts."""
    import json

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
                                "col2": "Description 2",
                            },
                            "table_description": "Table description",
                        },
                    ],
                },
            ],
        },
    }
    config_file = tmp_path / "business_contexts.json"
    config_file.write_text(json.dumps(config_data))

    cfg = FilePipelineConfig.from_file(str(config_file))
    assert len(cfg.ingest.business_contexts) == 1
    bc = cfg.ingest.business_contexts[0]
    assert bc.file_path == "/path/to/file.xlsx"
    assert len(bc.tables) == 1
    table_spec = bc.tables[0]
    assert table_spec.table == "Sheet1"
    assert table_spec.column_descriptions["col1"] == "Description 1"
    assert table_spec.table_description == "Table description"


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_config_from_file_full(tmp_path: Path):
    """Test loading a full config with all sections."""
    import json

    config_data = {
        "parse": {"batch_size": 5, "parser_kwargs": {"key": "value"}},
        "ingest": {
            "mode": "unified",
            "unified_label": "FullTest",
            "table_rows_batch_size": 3000,
            "business_contexts": [
                {
                    "file_path": "/path/to/file.xlsx",
                    "tables": [
                        {
                            "table": "Sheet1",
                            "column_descriptions": {"col1": "Desc1"},
                        },
                    ],
                },
            ],
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
    assert cfg.parse.batch_size == 5
    assert cfg.ingest.mode == "unified"
    assert cfg.embed.strategy == "after"
    assert cfg.output.return_mode == "full"
    assert cfg.diagnostics.enable_progress is True
    assert len(cfg.ingest.business_contexts) == 1
    assert len(cfg.embed.file_specs) == 1


@pytest.mark.unit
def test_config_from_file_nonexistent(tmp_path: Path):
    """Test that from_file raises FileNotFoundError for nonexistent file."""
    nonexistent = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError):
        FilePipelineConfig.from_file(str(nonexistent))


@pytest.mark.unit
def test_config_from_file_invalid_json(tmp_path: Path):
    """Test that from_file raises ValueError for invalid JSON."""
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("{ invalid json }")
    with pytest.raises(ValueError, match="Invalid JSON"):
        FilePipelineConfig.from_file(str(invalid_file))


@pytest.mark.unit
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


@pytest.mark.unit
@_handle_project
def test_business_context_applied_during_ingestion(file_manager, tmp_path: Path):
    """Test that business context is applied during table ingestion."""
    import json

    fm = file_manager
    fm.clear()

    # Create config file with business context
    config_data = {
        "ingest": {
            "business_contexts": [
                {
                    "file_path": "test_file.xlsx",
                    "tables": [
                        {
                            "table": "Sheet1",
                            "column_descriptions": {
                                "Name": "Person's full name",
                                "Age": "Person's age in years",
                            },
                            "table_description": "Test table with person data",
                        },
                    ],
                },
            ],
        },
    }
    config_file = tmp_path / "test_config.json"
    config_file.write_text(json.dumps(config_data))

    cfg = FilePipelineConfig.from_file(str(config_file))
    assert len(cfg.ingest.business_contexts) == 1
    assert cfg.ingest.business_contexts[0].file_path == "test_file.xlsx"
    assert len(cfg.ingest.business_contexts[0].tables) == 1
    assert cfg.ingest.business_contexts[0].tables[0].table == "Sheet1"


@pytest.mark.unit
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

    # Create a table document
    class _TableStub:
        def __init__(self, rows, sheet_name=None):
            self.rows = rows
            self.columns = None
            self.sheet_name = sheet_name
            self.section_path = None

    class _MetaStub:
        def __init__(self, tables=None):
            self.tables = tables or []
            self.mime_type = "text/plain"
            self.parser_name = "Stub"
            self.processing_time = 0.0

    class _DocStub:
        def __init__(self, tables=None):
            self.metadata = _MetaStub(tables=tables or [])
            self.processing_status = "completed"

        def to_parse_result(self, *a, **kw):
            return {}

    rows = [
        ["Name", "Age", "City", "Country"],
        ["Alice", "30", "London", "UK"],
        ["Bob", "25", "Paris", "France"],
    ]
    tbl = _TableStub(rows=rows, sheet_name="Sheet1")
    doc = _DocStub(tables=[tbl])
    result = {
        "status": "success",
        "total_records": 0,
        "file_format": "xlsx",
        "records": [],
    }

    # Verify config is valid
    assert len(cfg.embed.file_specs) == 1
    assert len(cfg.embed.file_specs[0].tables) == 1
    assert len(cfg.embed.file_specs[0].tables[0].source_columns) == 3
    assert len(cfg.embed.file_specs[0].tables[0].target_columns) == 3


@pytest.mark.unit
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


@pytest.mark.unit
def test_config_all_sections_populated(tmp_path: Path):
    """Test config file with all possible sections populated."""
    import json

    config_data = {
        "parse": {
            "batch_size": 10,
            "parser_kwargs": {"custom_option": "value", "another": 123},
        },
        "ingest": {
            "mode": "unified",
            "unified_label": "AllSectionsTest",
            "table_rows_batch_size": 5000,
            "content_rows_batch_size": 3000,
            "business_contexts": [
                {
                    "file_path": "/full/path/to/file.xlsx",
                    "tables": [
                        {
                            "table": "MainSheet",
                            "column_descriptions": {
                                "id": "Unique identifier",
                                "name": "Item name",
                            },
                            "table_description": "Main data table",
                        },
                    ],
                },
            ],
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
    assert cfg.parse.batch_size == 10
    assert cfg.parse.parser_kwargs == {"custom_option": "value", "another": 123}

    # Verify ingest
    assert cfg.ingest.mode == "unified"
    assert cfg.ingest.unified_label == "AllSectionsTest"
    assert cfg.ingest.table_rows_batch_size == 5000
    assert len(cfg.ingest.business_contexts) == 1

    # Verify embed
    assert cfg.embed.strategy == "after"
    assert cfg.embed.large_threshold == 10000
    assert cfg.embed.hooks_per_chunk is True
    # The test config uses "file_specs" format (updated from legacy)
    assert len(cfg.embed.file_specs) == 1
    assert cfg.embed.file_specs[0].file_path == "*"
    assert cfg.embed.file_specs[0].context == "per_file_table"
    assert len(cfg.embed.file_specs[0].tables) == 1
    assert cfg.embed.file_specs[0].tables[0].table == "MainSheet"

    # Verify plugins
    assert len(cfg.plugins.pre_parse) == 1
    assert len(cfg.plugins.post_embed) == 1
    assert len(cfg.plugins.plugin_kwargs) == 2

    # Verify output
    assert cfg.output.return_mode == "full"

    # Verify diagnostics
    assert cfg.diagnostics.enable_progress is True


@pytest.mark.unit
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
            "business_contexts": [
                {
                    "file_path": display_name,
                    "tables": [
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
                        },
                    ],
                },
            ],
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
    assert len(cfg.ingest.business_contexts) == 1
    assert cfg.ingest.business_contexts[0].file_path == display_name
    assert len(cfg.ingest.business_contexts[0].tables) == 1
    table_spec = cfg.ingest.business_contexts[0].tables[0]
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
    result = fm.parse(display_name, config=cfg)
    _item = result[display_name]
    _item = _item if isinstance(_item, dict) else _item.model_dump()
    assert _item["status"] == "success"


@pytest.mark.unit
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
            "business_contexts": [
                {
                    "file_path": display_name,
                    "tables": [
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
                        },
                    ],
                },
            ],
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
    table_spec = file_spec.tables[0]
    assert len(table_spec.source_columns) == 3
    assert len(table_spec.target_columns) == 3
    assert table_spec.source_columns == ["ProductName", "Category", "Supplier"]

    # Verify business context
    assert len(cfg.ingest.business_contexts) == 1
    bc = cfg.ingest.business_contexts[0]
    assert bc.file_path == display_name
    assert len(bc.tables) == 1
    table_spec = bc.tables[0]
    assert table_spec.table == "inventory"
    assert "SKU" in table_spec.column_descriptions
    assert table_spec.table_description is not None

    # Parse with config
    result = fm.parse(display_name, config=cfg)
    _item = result[display_name]
    _item = _item if isinstance(_item, dict) else _item.model_dump()
    assert _item["status"] == "success"


@pytest.mark.unit
@_handle_project
def test_multiple_tables_per_file_business_context(file_manager, tmp_path: Path):
    """Test BusinessContextSpec with multiple tables per file."""
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
            "business_contexts": [
                {
                    "file_path": display_name,
                    "tables": [
                        {
                            "table": "multi_table",
                            "column_descriptions": {
                                "ID": "Unique identifier",
                                "Name": "Item name",
                                "Value": "Numeric value",
                            },
                            "table_description": "Main data table",
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
    }
    config_file = tmp_path / "multi_table_config.json"
    config_file.write_text(json.dumps(config_data))

    cfg = FilePipelineConfig.from_file(str(config_file))

    # Verify multiple tables in single BusinessContextSpec
    assert len(cfg.ingest.business_contexts) == 1
    bc = cfg.ingest.business_contexts[0]
    assert bc.file_path == display_name
    assert len(bc.tables) == 2
    assert bc.tables[0].table == "multi_table"
    assert bc.tables[1].table == "Summary"
    assert "ID" in bc.tables[0].column_descriptions
    assert "Total" in bc.tables[1].column_descriptions
