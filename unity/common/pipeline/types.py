from __future__ import annotations

from typing import Dict, Literal, Optional, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from unity.file_manager.file_parsers.types.contracts import FileParseResult
from unity.file_manager.file_parsers.types.json_types import JsonObject


class InlineRowsHandle(BaseModel):
    """Inline tabular rows kept in-process for small tables."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["inline_rows"] = "inline_rows"
    rows: list[JsonObject] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    row_count: Optional[int] = None


class CsvFileHandle(BaseModel):
    """Reference to a CSV source file that should be streamed at ingest time."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["csv_file"] = "csv_file"
    storage_uri: str
    logical_path: str
    source_local_path: str
    columns: list[str] = Field(default_factory=list)
    encoding: str = "utf-8"
    delimiter: str = ","
    quotechar: str = '"'
    has_header: bool = True
    row_count: Optional[int] = None


class XlsxSheetHandle(BaseModel):
    """Reference to a single XLSX worksheet streamed on demand."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["xlsx_sheet"] = "xlsx_sheet"
    storage_uri: str
    logical_path: str
    source_local_path: str
    sheet_name: str
    columns: list[str] = Field(default_factory=list)
    has_header: bool = True
    row_count: Optional[int] = None


class ObjectStoreArtifactHandle(BaseModel):
    """Reference to a materialized artifact in a local or remote store.

    Currently only ``"jsonl"`` is implemented.  ``"parquet"`` and
    ``"arrow_ipc"`` are reserved for future backends and will raise
    ``NotImplementedError`` in ``LocalArtifactStore`` if used today.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["object_store_artifact"] = "object_store_artifact"
    storage_uri: str
    logical_path: str
    artifact_format: Literal["jsonl", "parquet", "arrow_ipc"] = "jsonl"
    columns: list[str] = Field(default_factory=list)
    row_count: Optional[int] = None


TableInputHandle: TypeAlias = (
    InlineRowsHandle | CsvFileHandle | XlsxSheetHandle | ObjectStoreArtifactHandle
)


class ParsedFileBundle(BaseModel):
    """Pipeline-owned parse wrapper separating semantic output from row transport."""

    result: FileParseResult
    table_inputs: Dict[str, TableInputHandle] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# IngestPlan: the parse -> ingest queue contract
# ---------------------------------------------------------------------------
#
# ``IngestPlan`` is the *manifest* the parse worker publishes to GCS after a
# file has been parsed AND lowered.  It is intentionally pointer-only:
# content rows and table rows NEVER appear inline. Every heavy artifact is
# materialised to object storage by the parse worker and referenced here via
# a ``TableInputHandle`` (``ObjectStoreArtifactHandle`` for content /
# in-memory table bodies, ``CsvFileHandle``/``XlsxSheetHandle`` for
# streamable source files).
#
# This keeps manifests KB-scale for any input size (1000-page PDFs, 1M-row
# spreadsheets) and preserves true streaming end-to-end: the ingest worker
# only touches bulk data when ``DataManager.ingest`` pulls a batch from the
# handle.
#
# The ``parse_summary`` field is a stripped ``FileParseResult`` with
# ``graph=None``, ``full_text=""``, and ``tables=[]``.  It retains only the
# fields ``FileRecord.to_file_record_entry`` needs (logical_path, status,
# error, file_format, mime_type, summary, metadata, trace) so the ingest
# worker can create a ``FileRecords`` entry without shipping the document
# graph across the wire.


class TableMeta(BaseModel):
    """Lightweight per-table metadata carried in an ``IngestPlan``.

    The table's *rows* live out-of-band behind a handle in
    ``IngestPlan.table_inputs[table_id]``.  ``TableMeta`` is only the
    structural + contextual info the ingest worker needs to provision the
    destination context and resolve embed columns / descriptions.
    """

    model_config = ConfigDict(frozen=True)

    table_id: str
    label: str
    columns: list[str] = Field(default_factory=list)
    row_count: Optional[int] = None
    sheet_name: Optional[str] = None
    table_summary: Optional[str] = None


class IngestPlan(BaseModel):
    """Pointer-only plan handed from the parse worker to the ingest worker.

    The manifest is always KB-scale regardless of input size.  Heavy data
    (content rows, table rows) is materialised to GCS by the parse worker
    and referenced via handles.

    Notes
    -----
    - ``parse_summary`` is a stripped ``FileParseResult`` with
      ``graph=None``, ``full_text=""``, and ``tables=[]``.  It exists so the
      ingest worker can create a ``FileRecords`` row via
      ``FileRecord.to_file_record_entry`` without having to ship the
      document graph.
    - ``content_rows_handle`` points at a JSONL artifact of
      ``FileContentRow`` payloads lowered from the document graph by the
      parse worker.  ``None`` means the parse produced no content rows
      (e.g. pure tabular file with ``table_ingest=False``).
    - ``table_inputs[table_id]`` must be a ``TableInputHandle`` that
      streams rows on demand.  Inline handles are permitted but only for
      small fallback tables.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    file_path: str
    parse_status: Literal["success", "error"] = "success"
    parse_summary: FileParseResult
    document_summary: str = ""
    content_rows_handle: Optional[TableInputHandle] = None
    tables_meta: list[TableMeta] = Field(default_factory=list)
    table_inputs: Dict[str, TableInputHandle] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Queue message models (for GKE workers and local queue)
# ---------------------------------------------------------------------------


class AttachmentCallback(BaseModel):
    """Routing metadata for attachment-ingestion dispatch.

    When a ``ParseRequested`` or ``IngestRequested`` message carries this
    callback, the ingest worker publishes a
    ``thread="attachment_ingestion_complete"`` envelope to the
    ``unity-{assistant_id}{env_suffix}`` Pub/Sub topic after ingest finishes
    so the originating ``ConversationManager`` can update ``FileRecords``.
    """

    model_config = ConfigDict(frozen=True)

    assistant_id: str
    env_suffix: str = ""
    display_name: str


class IngestBinding(BaseModel):
    """Common identity fields for any ingest-mode binding.

    Both FM and DM ingest need a Unify ``UNIFY_KEY`` to authenticate
    backend calls. Workers are shared across many assistants, so the
    key cannot live on the pod; it is looked up per message via
    Orchestra admin endpoints.

    ``user_id`` is always required for provenance / routing, and
    ``assistant_id`` is always required for deterministic Orchestra key
    resolution via ``GET /v0/admin/assistant?agent_id=...``.
    """

    model_config = ConfigDict(frozen=True)

    user_id: str
    assistant_id: str


class FmBinding(IngestBinding):
    """Routing info the ingest worker needs to run FM-mode ingest.

    With these fields the worker can activate the right Unify context,
    instantiate a ``FileManager(alias=fm_alias)``, and hand the parsed
    result to ``fm_process_file`` so the data lands under
    ``Files/{alias}/{storage_id}/...`` with a proper ``FileRecords``
    entry (rather than a bare ``DataManager`` context).
    """

    fm_alias: str = "Local"
    logical_path: str


class DmBinding(IngestBinding):
    """Routing info the ingest worker needs to run DM-mode ingest.

    Used when the caller wants raw ``DataManager`` ingestion into a
    specific context while still authenticating as a concrete
    assistant. No ``FileRecords`` entry is created in this mode.
    """

    target_context: str
    create_table_prefix: str = ""


class ParseRequested(BaseModel):
    """Message placed on the parse queue by the coordinator.

    A parse worker picks this up, runs ``FileParser.parse_batch``,
    writes the resulting ``ParsedFileBundle`` manifest to the artifact
    store, and acks the message.

    Message granularity contract: ``file_paths`` SHOULD contain exactly one
    entry per message so that parse work distributes evenly across worker
    pods via Pub/Sub fan-out. The list type is retained for back-compat but
    new dispatch paths enforce a single file per message.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["parse_requested"] = "parse_requested"
    job_id: str
    deployment_id: str = ""
    file_paths: list[str] = Field(default_factory=list)
    manifest_key: str = ""
    transport_mode: str = "source_reference"
    artifact_format: str = "jsonl"
    attachment_callback: Optional[AttachmentCallback] = None
    ingestion_mode: Literal["dm", "fm"] = "dm"
    fm_binding: Optional[FmBinding] = None
    dm_binding: Optional[DmBinding] = None


class IngestRequested(BaseModel):
    """Message placed on the ingest queue after parsing completes.

    An ingest worker picks this up, reads the ``ParsedFileBundle``
    manifest from the artifact store, rehydrates a ``FileParseResult``,
    activates the right Unify context, and dispatches via
    ``fm_process_file`` (FM mode) or ``ingest_artifacts`` (DM mode).
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["ingest_requested"] = "ingest_requested"
    job_id: str
    deployment_id: str = ""
    manifest_key: str = ""
    target_context: str = ""
    batch_size: int = 500
    attachment_callback: Optional[AttachmentCallback] = None
    ingestion_mode: Literal["dm", "fm"] = "dm"
    fm_binding: Optional[FmBinding] = None
    dm_binding: Optional[DmBinding] = None
