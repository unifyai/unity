"""Tests for the unified PipelineConfig schema.

Covers:
- Model validation (SourceTableSpec, SourceFileSpec, PipelineConfig)
- ``from_file()`` round-trip through a temp JSON file
- ``to_fm_config()`` mapping to FilePipelineConfig
- ``effective_post_ingest()`` merging of global + table-level rules
- ``resolve_paths()`` relative-to-absolute resolution
- Validation of the real pipeline_config.json against the schema
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unity.customization.types.pipeline_config import (
    PipelineConfig,
    PipelineExecutionConfig,
    SourceFileSpec,
    SourceTableSpec,
)
from unity.data_manager.types.ingest import (
    AutoDerivedColumn,
    ExplicitDerivedColumn,
    PostIngestConfig,
)

# =============================================================================
# Model validation
# =============================================================================


class TestSourceTableSpec:

    def test_minimal(self):
        spec = SourceTableSpec(sheet="Sheet1", context="Project/Data")
        assert spec.sheet == "Sheet1"
        assert spec.context == "Project/Data"
        assert spec.description == ""
        assert spec.chunk_size == 1000
        assert spec.post_ingest is None

    def test_with_post_ingest(self):
        spec = SourceTableSpec(
            sheet="July 2025",
            context="MH/Telematics/July",
            description="Telematics for July",
            chunk_size=500,
            post_ingest=PostIngestConfig(
                derived_columns=[
                    ExplicitDerivedColumn(
                        source_field="Trip travel time",
                        target_name="Trip travel time duration seconds",
                        equation="duration_seconds({lg:{field}})",
                    ),
                ],
            ),
        )
        assert spec.post_ingest is not None
        assert len(spec.post_ingest.derived_columns) == 1


class TestSourceFileSpec:

    def test_basic(self):
        spec = SourceFileSpec(
            file_path="data/file.xlsx",
            tables=[SourceTableSpec(sheet="Sheet1", context="Proj/Data")],
        )
        assert spec.file_path == "data/file.xlsx"
        assert len(spec.tables) == 1


class TestPipelineConfigValidation:

    def test_minimal_config(self):
        cfg = PipelineConfig(
            source_files=[
                SourceFileSpec(
                    file_path="file.xlsx",
                    tables=[
                        SourceTableSpec(sheet="Sheet1", context="Proj/Data"),
                    ],
                ),
            ],
        )
        assert len(cfg.source_files) == 1
        assert cfg.post_ingest.derived_columns == []
        assert cfg.execution.max_table_workers == 8

    def test_extra_keys_rejected(self):
        with pytest.raises(Exception):
            PipelineConfig(
                source_files=[],
                dm_contexts={"bad": "key"},
            )

    def test_from_dict_with_post_ingest(self):
        data = {
            "source_files": [
                {
                    "file_path": "f.xlsx",
                    "tables": [
                        {
                            "sheet": "S1",
                            "context": "X/Y",
                            "post_ingest": {
                                "derived_columns": [
                                    {
                                        "kind": "auto",
                                        "source_type": "datetime",
                                        "target_suffix": "Date",
                                        "equation": "date({lg:{field}})",
                                    },
                                ],
                            },
                        },
                    ],
                },
            ],
            "post_ingest": {
                "derived_columns": [
                    {
                        "kind": "explicit",
                        "source_field": "col_a",
                        "target_name": "col_b",
                        "equation": "f({lg:{field}})",
                    },
                ],
            },
        }
        cfg = PipelineConfig.model_validate(data)
        assert len(cfg.source_files) == 1
        table = cfg.source_files[0].tables[0]
        assert isinstance(table.post_ingest.derived_columns[0], AutoDerivedColumn)
        assert isinstance(cfg.post_ingest.derived_columns[0], ExplicitDerivedColumn)


# =============================================================================
# from_file round-trip
# =============================================================================


class TestFromFile:

    def test_round_trip(self, tmp_path: Path):
        data = {
            "source_files": [
                {
                    "file_path": "data.xlsx",
                    "tables": [
                        {"sheet": "Tab1", "context": "P/D", "chunk_size": 500},
                    ],
                },
            ],
            "execution": {
                "parallel_files": True,
                "max_table_workers": 4,
            },
        }
        config_file = tmp_path / "test_config.json"
        config_file.write_text(json.dumps(data))

        cfg = PipelineConfig.from_file(str(config_file))
        assert len(cfg.source_files) == 1
        assert cfg.source_files[0].tables[0].chunk_size == 500
        assert cfg.execution.parallel_files is True
        assert cfg.execution.max_table_workers == 4

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_file("/nonexistent/path.json")

    def test_invalid_json_raises(self, tmp_path: Path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json {{{")
        with pytest.raises(ValueError, match="Invalid JSON"):
            PipelineConfig.from_file(str(bad_file))


# =============================================================================
# to_fm_config
# =============================================================================


class TestToFmConfig:

    def test_maps_execution_fields(self):
        cfg = PipelineConfig(
            source_files=[],
            execution=PipelineExecutionConfig(
                parallel_files=True,
                max_file_workers=6,
                max_table_workers=10,
                max_embed_workers=12,
            ),
        )
        fm = cfg.to_fm_config()
        assert fm.execution.parallel_files is True
        assert fm.execution.max_file_workers == 6
        assert fm.execution.max_embed_workers == 12
        assert not hasattr(fm.execution, "max_table_workers")

    def test_passes_through_shared_sections(self):
        cfg = PipelineConfig(source_files=[])
        cfg.embed.strategy = "along"
        cfg.retry.max_retries = 5

        fm = cfg.to_fm_config()
        assert fm.embed.strategy == "along"
        assert fm.retry.max_retries == 5


# =============================================================================
# effective_post_ingest
# =============================================================================


class TestEffectivePostIngest:

    def _make_config(
        self,
        global_rules=None,
        table_rules=None,
    ) -> tuple[PipelineConfig, SourceTableSpec]:
        table = SourceTableSpec(
            sheet="S1",
            context="X/Y",
            post_ingest=(
                PostIngestConfig(derived_columns=table_rules) if table_rules else None
            ),
        )
        cfg = PipelineConfig(
            source_files=[SourceFileSpec(file_path="f.xlsx", tables=[table])],
            post_ingest=PostIngestConfig(
                derived_columns=global_rules or [],
            ),
        )
        return cfg, table

    def test_neither_returns_none(self):
        cfg, table = self._make_config()
        assert cfg.effective_post_ingest(table) is None

    def test_global_only(self):
        auto = AutoDerivedColumn(
            source_type="datetime",
            target_suffix="Date",
            equation="date({lg:{field}})",
        )
        cfg, table = self._make_config(global_rules=[auto])
        eff = cfg.effective_post_ingest(table)
        assert eff is not None
        assert len(eff.derived_columns) == 1
        assert eff is cfg.post_ingest

    def test_table_only(self):
        explicit = ExplicitDerivedColumn(
            source_field="col_a",
            target_name="col_b",
            equation="f({lg:{field}})",
        )
        cfg, table = self._make_config(table_rules=[explicit])
        eff = cfg.effective_post_ingest(table)
        assert eff is not None
        assert len(eff.derived_columns) == 1
        assert eff is table.post_ingest

    def test_merge_global_then_table(self):
        auto = AutoDerivedColumn(
            source_type="datetime",
            target_suffix="Date",
            equation="date({lg:{field}})",
        )
        explicit = ExplicitDerivedColumn(
            source_field="col_a",
            target_name="col_b",
            equation="f({lg:{field}})",
        )
        cfg, table = self._make_config(
            global_rules=[auto],
            table_rules=[explicit],
        )
        eff = cfg.effective_post_ingest(table)
        assert eff is not None
        assert len(eff.derived_columns) == 2
        assert isinstance(eff.derived_columns[0], AutoDerivedColumn)
        assert isinstance(eff.derived_columns[1], ExplicitDerivedColumn)


# =============================================================================
# resolve_paths
# =============================================================================


class TestResolvePaths:

    def test_resolves_relative_source_files(self, tmp_path: Path):
        cfg = PipelineConfig(
            source_files=[
                SourceFileSpec(
                    file_path="data/file.xlsx",
                    tables=[SourceTableSpec(sheet="S1", context="X/Y")],
                ),
            ],
        )
        cfg.resolve_paths(tmp_path)
        assert cfg.source_files[0].file_path == str(tmp_path / "data/file.xlsx")

    def test_absolute_paths_unchanged(self, tmp_path: Path):
        cfg = PipelineConfig(
            source_files=[
                SourceFileSpec(
                    file_path="/absolute/path/file.xlsx",
                    tables=[SourceTableSpec(sheet="S1", context="X/Y")],
                ),
            ],
        )
        cfg.resolve_paths(tmp_path)
        assert cfg.source_files[0].file_path == "/absolute/path/file.xlsx"


# =============================================================================
# Real config validation
# =============================================================================


class TestRealConfig:

    def test_midland_heart_config_validates(self):
        """Ensure the actual pipeline_config.json passes schema validation."""
        config_path = (
            Path(__file__).resolve().parents[2]
            / "unity"
            / "customization"
            / "clients"
            / "midland_heart"
            / "pipeline_config.json"
        )
        if not config_path.exists():
            pytest.skip(f"Config not found at {config_path}")

        cfg = PipelineConfig.from_file(str(config_path))

        assert len(cfg.source_files) == 2
        all_tables = [t for sf in cfg.source_files for t in sf.tables]
        assert len(all_tables) == 6

        repairs = cfg.source_files[0].tables[0]
        assert repairs.context == "MidlandHeart/Repairs2025"
        assert repairs.post_ingest is None

        telem_july = cfg.source_files[1].tables[0]
        assert telem_july.context == "MidlandHeart/Telematics2025/July"
        assert telem_july.post_ingest is not None
        assert len(telem_july.post_ingest.derived_columns) == 3

        assert len(cfg.post_ingest.derived_columns) == 1
        assert isinstance(cfg.post_ingest.derived_columns[0], AutoDerivedColumn)

        eff_repairs = cfg.effective_post_ingest(repairs)
        assert eff_repairs is not None
        assert len(eff_repairs.derived_columns) == 1

        eff_telem = cfg.effective_post_ingest(telem_july)
        assert eff_telem is not None
        assert len(eff_telem.derived_columns) == 4
