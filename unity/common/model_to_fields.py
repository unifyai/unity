from __future__ import annotations

import copy
import json
from typing import Any

import jsonref
from pydantic import BaseModel

from ..knowledge_manager.types import ColumnType

# Map JSON Schema types to ColumnType
_JSON_SCHEMA_TYPE_MAP = {
    "string": ColumnType.str,
    "integer": ColumnType.int,
    "number": ColumnType.float,
    "boolean": ColumnType.bool,
    "array": ColumnType.list,
    "object": ColumnType.dict,
}

# JSON Schema formats that map to specific ColumnTypes
_JSON_SCHEMA_FORMAT_MAP = {
    "date-time": ColumnType.datetime,
    "date": ColumnType.date,
    "time": ColumnType.time,
}


def _is_nested_schema(prop: dict[str, Any]) -> bool:
    """Check if a JSON Schema has nested structure requiring full serialization."""
    # Array of objects with properties
    if prop.get("type") == "array":
        items = prop.get("items", {})
        if isinstance(items, dict):
            # Direct object items
            if items.get("properties"):
                return True
            # Items with anyOf (union types like RawImageRef | AnnotatedImageRef)
            if "anyOf" in items:
                for sub in items["anyOf"]:
                    if isinstance(sub, dict) and sub.get("properties"):
                        return True
    # Object with defined properties
    if prop.get("type") == "object" and prop.get("properties"):
        return True
    # Handle anyOf (Optional types) - check if any non-null branch is nested
    if "anyOf" in prop:
        for sub in prop["anyOf"]:
            if isinstance(sub, dict) and sub.get("type") != "null":
                if _is_nested_schema(sub):
                    return True
    return False


def _schema_to_column_type(prop: dict[str, Any]) -> str:
    """Map a JSON Schema property to a ColumnType string."""
    # Check format first (e.g., "date-time" → datetime)
    fmt = prop.get("format")
    if fmt in _JSON_SCHEMA_FORMAT_MAP:
        return _JSON_SCHEMA_FORMAT_MAP[fmt]

    # Handle anyOf (typically Optional types)
    if "anyOf" in prop:
        non_null = [s for s in prop["anyOf"] if s.get("type") != "null"]
        if len(non_null) == 1:
            return _schema_to_column_type(non_null[0])

    # Map basic type
    schema_type = prop.get("type", "string")
    return _JSON_SCHEMA_TYPE_MAP.get(schema_type, ColumnType.str)


def model_to_fields(model: type[BaseModel]) -> dict[str, dict[str, Any]]:
    """
    Translate a Pydantic model class into the structure expected by
    `unify.create_fields`.

    Uses Pydantic's JSON Schema with $ref dereferencing via jsonref.
    Nested object schemas are serialized as JSON strings for Orchestra.

    Supports per-field overrides via ``json_schema_extra``:
    - ``{"unify_type": "..."}`` — override the inferred Orchestra type
    - ``{"unique": True}`` — mark the field as unique in Orchestra
    - ``{"mutable": False}`` — mark the field immutable in Orchestra

    Examples
    --------
    >>> fields_dict = model_to_fields(Contact)
    >>> unify.create_fields(fields_dict, context=ctx)
    """
    # Get Pydantic's JSON Schema and dereference all $refs
    schema = model.model_json_schema()
    schema_json = json.dumps(schema)
    dereferenced = jsonref.loads(schema_json)

    # Convert jsonref proxy objects to plain dicts
    properties = {
        name: copy.deepcopy(dict(prop))
        for name, prop in dereferenced.get("properties", {}).items()
    }

    fields_source = model.model_fields
    result: dict[str, dict[str, Any]] = {}

    for name, prop in properties.items():
        field_info = fields_source.get(name)

        # Check for explicit unify_type override
        unify_type = None
        if field_info:
            extra = getattr(field_info, "json_schema_extra", None)
            if isinstance(extra, dict):
                unify_type = extra.get("unify_type")

        # Build field entry
        if unify_type:
            entry: dict[str, Any] = {"type": unify_type}
        elif _is_nested_schema(prop):
            # Serialize nested schema as JSON string for Orchestra
            entry = {"type": json.dumps(prop)}
        else:
            # Use simple ColumnType
            entry = {"type": _schema_to_column_type(prop)}

        entry["mutable"] = True
        if field_info and isinstance(extra, dict) and "mutable" in extra:
            entry["mutable"] = bool(extra["mutable"])

        if field_info and isinstance(extra, dict) and extra.get("unique"):
            entry["unique"] = True

        # Use Field description (not JSON Schema description which can be very long)
        if field_info and getattr(field_info, "description", None):
            entry["description"] = field_info.description.strip()

        result[name] = entry

    return result
