"""Helpers for preserving shared-row authorship metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from unity.session_details import SESSION_DETAILS

AUTHORING_ASSISTANT_ID_FIELD = "authoring_assistant_id"
SHARED_SCOPED_TABLES: frozenset[str] = frozenset(
    {
        "Tasks",
        "Contacts",
        "Secrets",
        "Knowledge",
        "Guidance",
        "Functions/Compositional",
        "Functions/Meta",
        "Functions/Primitives",
        "Functions/VirtualEnvs",
        "FileRecords",
        "Files",
        "Data",
        "BlackList",
        "Dashboards/Tiles",
        "Dashboards/Layouts",
        "Transcripts",
        "Exchanges",
        "Images",
    },
)
DYNAMIC_AUTHORED_TABLE_PREFIXES: frozenset[str] = frozenset(
    {
        "Data",
        "FileRecords",
        "Files",
        "Knowledge",
    },
)
SHARED_TABLE_MATCHERS: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
    (table_name, tuple(table_name.split("/")))
    for table_name in sorted(
        SHARED_SCOPED_TABLES,
        key=lambda value: len(value.split("/")),
        reverse=True,
    )
)
AUTHORING_ASSISTANT_ID_FIELD_INFO: dict[str, Any] = {
    "type": "int",
    "mutable": False,
    "description": "Assistant id that originally authored this row.",
}


class AuthoredRow(BaseModel):
    """Base model for rows that record which assistant originally wrote them."""

    authoring_assistant_id: int | None = Field(
        default=None,
        description="Assistant id that originally authored this row.",
        json_schema_extra={"mutable": False},
    )


def current_authoring_assistant_id() -> int | None:
    """Return the active assistant id used for authorship stamps."""

    return SESSION_DETAILS.assistant.agent_id


def fields_with_authoring(
    fields: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a field definition map containing the authorship field."""

    merged = dict(fields or {})
    merged.setdefault(
        AUTHORING_ASSISTANT_ID_FIELD,
        dict(AUTHORING_ASSISTANT_ID_FIELD_INFO),
    )
    return merged


def shared_table_for_context(context: str) -> str | None:
    """Return the shared table name embedded in a concrete context path."""

    parts = tuple(context.split("/"))
    for start in range(len(parts)):
        for table_name, table_parts in SHARED_TABLE_MATCHERS:
            end = start + len(table_parts)
            if parts[start:end] != table_parts:
                continue
            if end == len(parts) or table_name in DYNAMIC_AUTHORED_TABLE_PREFIXES:
                return table_name
    return None


def is_shared_authored_context(context: str) -> bool:
    """Return whether a concrete context stores rows that carry authorship."""

    return shared_table_for_context(context) is not None


def stamp_authoring_assistant_id(
    entries: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a row payload stamped with the active assistant as original author."""

    stamped = dict(entries)
    stamped[AUTHORING_ASSISTANT_ID_FIELD] = current_authoring_assistant_id()
    return stamped


def strip_authoring_assistant_id(
    entries: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return an update payload without caller-controlled authorship changes."""

    cleaned = dict(entries or {})
    cleaned.pop(AUTHORING_ASSISTANT_ID_FIELD, None)
    return cleaned
