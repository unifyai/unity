from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Mapping, Sequence, Union, get_args, get_origin
from pydantic import BaseModel

from ..knowledge_manager.types import ColumnType


# ---- Helper -----------------------------------------------------------------
def model_to_fields(model: type[BaseModel]) -> dict[str, dict[str, Any]]:
    """
    Translate a Pydantic *model class* into the structure expected by
    `unify.create_fields`.

    Rules implemented
    -----------------
    • Skip any field whose name ends with ``_id``.
    • Infer the closest ``ColumnType`` from the type annotation (one-level deep).
      Unwraps ``Optional[X]`` / ``Union[X, None]`` automatically.
    • Pull the human-readable description from ``Field(..., description=...)``.
      Omit the key when no description was supplied.

    Examples
    --------
    >>> fields_dict = model_to_unify_fields(Contact)
    >>> unify.create_fields(fields_dict, context=ctx)
    """
    fields_source = model.model_fields

    def map_python_type(py_t: Any) -> str:
        """
        Map a (possibly-parameterised) Python type to our ColumnType label.
        """
        origin = get_origin(py_t) or py_t

        # Containers ----------------------------------------------------------
        if isinstance(origin, type) and issubclass(origin, BaseModel):
            return ColumnType.dict
        if origin in (dict, Mapping):
            return ColumnType.dict
        if origin in (list, tuple, set, Sequence):
            return ColumnType.list

        # Scalars -------------------------------------------------------------
        if origin is str:
            return ColumnType.str
        if origin is int:
            return ColumnType.int
        if origin is float:
            return ColumnType.float
        if origin is bool:
            return ColumnType.bool
        if origin is datetime:
            return ColumnType.datetime
        if origin is date:
            return ColumnType.date
        if origin is time:
            return ColumnType.time

        # Fallback
        return ColumnType.str

    unify_fields: dict[str, dict[str, Any]] = {}

    for name, field in fields_source.items():

        # Unwrap Optional / Union[..., None]
        annotation = field.annotation
        origin = get_origin(annotation)
        if origin is Union:
            non_none = [t for t in get_args(annotation) if t is not type(None)]
            if non_none:  # keep first surviving alternative
                annotation = non_none[0]

        column_type = map_python_type(annotation)

        entry: dict[str, Any] = {"type": column_type, "mutable": True}
        if getattr(field, "description", None):
            entry["description"] = field.description.strip()

        unify_fields[name] = entry

    return unify_fields
