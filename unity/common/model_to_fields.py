from __future__ import annotations

from typing import Any
from pydantic import BaseModel


def model_to_fields(model: type[BaseModel]) -> dict[str, dict[str, Any]]:
    """
    Translate a Pydantic model class into the structure expected by
    `unify.create_fields`.

    Uses Pydantic's native JSON Schema generation for full type fidelity,
    including parameterized types like List[str].

    Supports per-field type overrides via:
    - Field(..., json_schema_extra={"unify_type": "..."})

    Examples
    --------
    >>> fields_dict = model_to_fields(Contact)
    >>> unify.create_fields(fields_dict, context=ctx)
    """
    schema = model.model_json_schema()
    properties = schema.get("properties", {})
    fields_source = model.model_fields

    result: dict[str, dict[str, Any]] = {}
    for name, prop in properties.items():
        field_info = fields_source.get(name)

        # Check for unify_type override (e.g. "image" type in ImageManager)
        unify_type = None
        if field_info:
            extra = getattr(field_info, "json_schema_extra", None)
            if isinstance(extra, dict):
                unify_type = extra.get("unify_type")

        if unify_type:
            # Use explicit override
            entry: dict[str, Any] = {"type": unify_type}
            if prop.get("description"):
                entry["description"] = prop["description"]
            result[name] = entry
        else:
            # Use full Pydantic JSON Schema (preserves List[str], etc.)
            result[name] = dict(prop)

    return result
