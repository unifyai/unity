"""Unified pipeline configuration schema.

Provides a single, strongly-typed Pydantic model (``PipelineConfig``) that
validates the entire ``pipeline_config.json`` used by client ingestion scripts.

The schema **composes** the existing FM and DM type hierarchies without
modifying them:

* FM types (``ParseConfig``, ``IngestConfig``, ``EmbeddingsConfig``, etc.)
  from :mod:`unity.file_manager.types.config`.
* DM types (``PostIngestConfig``) from :mod:`unity.data_manager.types.ingest`.

It adds two new models to represent the source-file / table structure that
was previously scattered across ``source_files`` and ``dm_contexts``:

* :class:`SourceTableSpec` -- per-table config (context path, description,
  chunk size, optional per-table post-ingest rules).
* :class:`SourceFileSpec` -- per-file config (file path + list of tables).

Key methods on :class:`PipelineConfig`:

* ``from_file(path)`` -- load and validate a JSON config file.
* ``resolve_paths(project_root)`` -- resolve relative file paths to absolute.
* ``to_fm_config()`` -- produce a ``FilePipelineConfig`` for the FM flow.
* ``effective_post_ingest(table)`` -- merge global + table-level post-ingest
  rules for a specific table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from unity.data_manager.types.ingest import PostIngestConfig
from unity.file_manager.types.config import (
    DiagnosticsConfig,
    EmbeddingsConfig,
    ExecutionConfig,
    FilePipelineConfig,
    IngestConfig,
    OutputConfig,
    ParseConfig,
    RetryConfig,
)


class SourceTableSpec(BaseModel):
    """Per-table configuration within a source file.

    Co-locates the context path, human-readable description, chunking
    parameters, and optional table-scoped post-ingest rules that were
    previously spread across ``dm_contexts`` and the global
    ``post_ingest`` section.
    """

    sheet: str
    context: str
    description: str = ""
    chunk_size: int = 1000
    post_ingest: Optional[PostIngestConfig] = None


class SourceFileSpec(BaseModel):
    """A source file and its per-table specifications."""

    file_path: str
    tables: List[SourceTableSpec]


class PipelineExecutionConfig(BaseModel):
    """Superset of FM and DM execution knobs.

    FM uses ``max_file_workers`` and ``max_embed_workers``.
    DM uses ``max_file_workers`` and ``max_table_workers``.
    """

    parallel_files: bool = False
    max_file_workers: int = 4
    max_table_workers: int = 8
    max_embed_workers: int = 8


class PipelineConfig(BaseModel):
    """Unified, strongly-typed pipeline configuration.

    Validates the entire ``pipeline_config.json`` and provides typed
    access for both the FM and DM ingestion flows.
    """

    model_config = ConfigDict(extra="forbid")

    source_files: List[SourceFileSpec] = Field(default_factory=list)
    parse: ParseConfig = Field(default_factory=ParseConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    post_ingest: PostIngestConfig = Field(default_factory=PostIngestConfig)
    embed: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    execution: PipelineExecutionConfig = Field(
        default_factory=PipelineExecutionConfig,
    )
    retry: RetryConfig = Field(default_factory=RetryConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)

    # -- factory -----------------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> PipelineConfig:
        """Load a JSON config file and return a validated instance.

        Parameters
        ----------
        path : str
            Path to the JSON configuration file.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file contains invalid JSON or fails schema validation.
        """
        config_path = Path(path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}") from e

        return cls.model_validate(data)

    # -- path resolution ---------------------------------------------------

    def resolve_paths(self, project_root: Path) -> None:
        """Resolve relative ``file_path`` values to absolute paths.

        Mutates *source_files*, *embed.file_specs*, and
        *ingest.business_contexts.file_contexts* in place.
        """

        def _resolve(rel: str) -> str:
            p = Path(rel)
            return str(project_root / p) if not p.is_absolute() else rel

        for sf in self.source_files:
            sf.file_path = _resolve(sf.file_path)

        for spec in self.embed.file_specs:
            spec.file_path = _resolve(spec.file_path)

        if self.ingest.business_contexts:
            for fc in self.ingest.business_contexts.file_contexts:
                fc.file_path = _resolve(fc.file_path)

    # -- FM interop --------------------------------------------------------

    def to_fm_config(self) -> FilePipelineConfig:
        """Produce a :class:`FilePipelineConfig` for the FM pipeline.

        Maps the superset :class:`PipelineExecutionConfig` to the FM's
        narrower :class:`ExecutionConfig` and passes through all shared
        sections unchanged.  ``FilePipelineConfig`` is **not** modified.
        """
        return FilePipelineConfig(
            parse=self.parse,
            ingest=self.ingest,
            embed=self.embed,
            output=self.output,
            diagnostics=self.diagnostics,
            execution=ExecutionConfig(
                parallel_files=self.execution.parallel_files,
                max_file_workers=self.execution.max_file_workers,
                max_embed_workers=self.execution.max_embed_workers,
            ),
            retry=self.retry,
        )

    # -- post-ingest merging -----------------------------------------------

    def effective_post_ingest(
        self,
        table: SourceTableSpec,
    ) -> Optional[PostIngestConfig]:
        """Merge global and table-level post-ingest rules for *table*.

        * If neither level defines rules, returns ``None``.
        * If only one level defines rules, returns that level's config.
        * If both levels define rules, returns a **new** config whose
          ``derived_columns`` is the concatenation of global rules
          followed by table-level rules.
        """
        global_rules = self.post_ingest.derived_columns
        table_rules = table.post_ingest.derived_columns if table.post_ingest else []

        if not global_rules and not table_rules:
            return None

        if not table_rules:
            return self.post_ingest

        if not global_rules:
            return table.post_ingest

        return PostIngestConfig(
            derived_columns=list(global_rules) + list(table_rules),
        )
