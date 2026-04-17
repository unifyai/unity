from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Sequence

from unity.file_manager.file_parsers.settings import FileParserSettings
from unity.file_manager.file_parsers.types.contracts import (
    FileParseMetadata,
    FileParseResult,
    FileParseTrace,
)
from unity.file_manager.file_parsers.types.enums import NodeKind
from unity.file_manager.file_parsers.types.formats import FileFormat, MimeType
from unity.file_manager.file_parsers.types.graph import (
    ContentGraph,
    ContentNode,
    DocumentPayload,
    SheetPayload,
    TablePayload,
)
from unity.file_manager.file_parsers.types.json_types import JsonObject
from unity.file_manager.file_parsers.types.table import ExtractedTable
from unity.file_manager.file_parsers.utils.format_policy import (
    bound_spreadsheet_full_text,
    build_spreadsheet_profile_text,
    fallback_spreadsheet_summary,
)


def normalize_tabular_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def build_spreadsheet_graph(
    *,
    logical_path: str,
    tables: Sequence[ExtractedTable],
    sheet_names: Sequence[str],
) -> ContentGraph:
    """Build a minimal spreadsheet graph for downstream lowering."""

    root_id = "document:0"
    document_title = Path(str(logical_path or "")).name or "spreadsheet"
    nodes: dict[str, ContentNode] = {
        root_id: ContentNode(
            node_id=root_id,
            kind=NodeKind.DOCUMENT,
            title=document_title,
            payload=DocumentPayload(title=document_title),
        ),
    }

    ordered_sheets = [str(name).strip() for name in sheet_names if str(name).strip()]
    if not ordered_sheets:
        ordered_sheets = sorted(
            {str(table.sheet_name or "Sheet 1") for table in list(tables or [])},
        )
    if not ordered_sheets:
        ordered_sheets = ["Sheet 1"]

    sheet_node_ids: dict[str, str] = {}
    for sheet_index, sheet_name in enumerate(ordered_sheets, start=1):
        node_id = f"sheet:{sheet_index}"
        sheet_node_ids[sheet_name] = node_id
        nodes[node_id] = ContentNode(
            node_id=node_id,
            kind=NodeKind.SHEET,
            parent_id=root_id,
            order=sheet_index,
            title=sheet_name,
            payload=SheetPayload(sheet_index=sheet_index, sheet_name=sheet_name),
            meta={"sheet_name": sheet_name},
        )
        nodes[root_id].children_ids.append(node_id)

    for table_index, table in enumerate(list(tables or []), start=1):
        sheet_name = str(table.sheet_name or "Sheet 1")
        parent_id = sheet_node_ids.get(sheet_name)
        if parent_id is None:
            parent_id = f"sheet:{len(sheet_node_ids) + 1}"
            sheet_node_ids[sheet_name] = parent_id
            nodes[parent_id] = ContentNode(
                node_id=parent_id,
                kind=NodeKind.SHEET,
                parent_id=root_id,
                order=len(sheet_node_ids),
                title=sheet_name,
                payload=SheetPayload(
                    sheet_index=len(sheet_node_ids),
                    sheet_name=sheet_name,
                ),
                meta={"sheet_name": sheet_name},
            )
            nodes[root_id].children_ids.append(parent_id)

        table_node = ContentNode(
            node_id=str(table.table_id or f"table:{table_index}"),
            kind=NodeKind.TABLE,
            parent_id=parent_id,
            order=table_index,
            title=str(table.label or f"table_{table_index}"),
            payload=TablePayload(
                label=str(table.label or f"table_{table_index}"),
                columns=list(table.columns or []),
                sample_rows=list(table.sample_rows or []),
                num_rows=table.num_rows,
                num_cols=table.num_cols,
            ),
            meta={
                "table_label": str(table.label or f"table_{table_index}"),
                "sheet_name": sheet_name,
            },
        )
        nodes[table_node.node_id] = table_node
        nodes[parent_id].children_ids.append(table_node.node_id)

    return ContentGraph(root_id=root_id, nodes=nodes)


def finalize_spreadsheet_result(
    *,
    logical_path: str,
    file_format: FileFormat | None,
    mime_type: MimeType | None,
    trace: FileParseTrace,
    settings: FileParserSettings,
    tables: Sequence[ExtractedTable],
    sheet_names: Sequence[str],
) -> FileParseResult:
    """Build the common success result for spreadsheet-style backends."""

    graph = build_spreadsheet_graph(
        logical_path=logical_path,
        tables=tables,
        sheet_names=sheet_names,
    )
    profile_text = build_spreadsheet_profile_text(
        logical_path=logical_path,
        tables=list(tables or []),
        sheet_names=list(sheet_names or []),
        max_tables=settings.TABULAR_PROFILE_MAX_TABLES,
        max_sample_rows=settings.TABULAR_PROFILE_MAX_SAMPLE_ROWS,
    )
    full_text = bound_spreadsheet_full_text(
        profile_text=profile_text,
        settings=settings,
    )
    summary = fallback_spreadsheet_summary(
        logical_path=logical_path,
        tables=list(tables or []),
        sheet_names=list(sheet_names or []),
    )

    trace.counters["nodes"] = len(graph.nodes)
    trace.counters["tables"] = len(list(tables or []))
    trace.counters["sheets"] = len(list(sheet_names or []))

    return FileParseResult(
        logical_path=logical_path,
        status="success",
        file_format=file_format,
        mime_type=mime_type,
        tables=list(tables or []),
        summary=summary,
        full_text=full_text,
        trace=trace,
        graph=graph,
    )


def should_inline_tabular_rows(
    *,
    row_count: int,
    settings: FileParserSettings,
) -> bool:
    return row_count <= max(int(settings.TABULAR_INLINE_ROW_LIMIT), 0)


def take_sample_rows(
    rows: Iterable[JsonObject],
    *,
    settings: FileParserSettings,
) -> list[JsonObject]:
    limit = max(int(settings.TABULAR_SAMPLE_ROWS), 0)
    sample: list[JsonObject] = []
    for row in rows:
        sample.append(dict(row))
        if len(sample) >= limit:
            break
    return sample


def _coerce_metadata(metadata: FileParseMetadata) -> FileParseMetadata:
    return FileParseMetadata.model_validate(
        metadata.model_dump(mode="json", exclude_none=True),
    )
