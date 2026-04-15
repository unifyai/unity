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
    """Reference to a materialized artifact such as Parquet or Arrow IPC."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["object_store_artifact"] = "object_store_artifact"
    storage_uri: str
    logical_path: str
    artifact_format: Literal["parquet", "arrow_ipc", "jsonl"]
    columns: list[str] = Field(default_factory=list)
    row_count: Optional[int] = None


TableInputHandle: TypeAlias = (
    InlineRowsHandle | CsvFileHandle | XlsxSheetHandle | ObjectStoreArtifactHandle
)


class ParsedFileBundle(BaseModel):
    """Pipeline-owned parse wrapper separating semantic output from row transport."""

    result: FileParseResult
    table_inputs: Dict[str, TableInputHandle] = Field(default_factory=dict)
