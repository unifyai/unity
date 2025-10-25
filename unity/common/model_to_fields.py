from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Mapping, Sequence, Union, get_args, get_origin, Annotated
import json
from types import UnionType  # Python 3.10+
from pydantic import BaseModel

from ..knowledge_manager.types import ColumnType


# ---- Helper -----------------------------------------------------------------
def model_to_fields(model: type[BaseModel]) -> dict[str, dict[str, Any]]:
    """
    Translate a Pydantic *model class* into the structure expected by
    `unify.create_fields`.

    Rules implemented
    -----------------
    • Recursively infer the closest ``ColumnType`` from the type annotation –
      even for deeply nested generics.
      Unwraps ``Optional[X]`` / ``Union[X, None]`` automatically.
    • Pull the human-readable description from ``Field(..., description=...)``.
      Omit the key when no description was supplied.
    • Honor a per-field Unify type override via either
      ``Field(json_schema_extra={"unify_type": "..."})`` or
      ``typing.Annotated[T, {"unify_type": "..."}]``.

    Examples
    --------
    >>> fields_dict = model_to_fields(Contact)
    >>> unify.create_fields(fields_dict, context=ctx)
    """
    fields_source = model.model_fields

    def _extract_unify_type(field_info: Any, annotation: Any) -> str | None:
        """Return an explicit Unify data type override if specified.

        Supported sources:
        - Field(..., json_schema_extra={"unify_type": "..."})
        - typing.Annotated[T, {"unify_type": "..."}] or an object with attribute ``unify_type``
        """
        # A) From Field(..., json_schema_extra={...})
        try:
            extra = getattr(field_info, "json_schema_extra", None)
            if isinstance(extra, dict):
                ut = extra.get("unify_type")
                if isinstance(ut, str) and ut:
                    return ut
        except Exception:
            pass

        # B) From typing.Annotated[..., meta]
        try:
            if get_origin(annotation) is Annotated:
                metas = get_args(annotation)[1:]
                for m in metas:
                    if isinstance(m, dict) and isinstance(m.get("unify_type"), str):
                        return m["unify_type"]
                    if hasattr(m, "unify_type") and isinstance(
                        getattr(m, "unify_type"),
                        str,
                    ):
                        return getattr(m, "unify_type")
        except Exception:
            pass

        return None

    def _unwrap_annotated(py_t: Any) -> Any:
        try:
            if get_origin(py_t) is Annotated:
                args = get_args(py_t)
                if args:
                    return args[0]
        except Exception:
            pass
        return py_t

    def _pydantic_json_schema_for(annotation: Any) -> dict | None:
        """Return a JSON Schema dict for a Pydantic model annotation when possible.

        Behaviour:
        - Preserve Optional/Union[..., None] nullability by returning anyOf with {"type":"null"}.
        - For BaseModel subclasses (including RootModel) → use model_json_schema().
        - For containers like List[Model] → build a minimal array schema with items.
        """
        try:
            ann = _unwrap_annotated(annotation)
            origin = get_origin(ann)

            # Optional / Union[..., None] – preserve nullability
            if origin in (Union, UnionType):
                args = list(get_args(ann))
                has_none = any(a is type(None) for a in args)  # noqa: E721
                non_none = [a for a in args if a is not type(None)]  # noqa: E721
                if has_none and len(non_none) == 1:
                    base = non_none[0]
                    sub = _pydantic_json_schema_for(base)
                    # If nested model/list schema is available, wrap with null
                    if sub is not None:
                        return {"anyOf": [sub, {"type": "null"}]}
                    # Otherwise, synthesise a minimal schema for common primitives/containers
                    try:
                        primitive_map = {
                            str: {"type": "string"},
                            int: {"type": "integer"},
                            float: {"type": "number"},
                            bool: {"type": "boolean"},
                            datetime: {"type": "string", "format": "date-time"},
                            date: {"type": "string", "format": "date"},
                            time: {"type": "string", "format": "time"},
                        }
                        if base in primitive_map:
                            return {"anyOf": [primitive_map[base], {"type": "null"}]}
                        b_origin = get_origin(base)
                        if b_origin in (list, Sequence):
                            return {"anyOf": [{"type": "array"}, {"type": "null"}]}
                        if b_origin in (dict, Mapping) or base is dict:
                            return {"anyOf": [{"type": "object"}, {"type": "null"}]}
                    except Exception:
                        pass
                    # Fallback so ColumnType mapping can handle rare cases
                    return None
                # Other unions not supported here
                return None

            # Direct BaseModel subclass (covers RootModel as well)
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                return ann.model_json_schema()

            # Container[List/Sequence] of BaseModel
            if origin in (list, Sequence):
                (item_type,) = get_args(ann) if get_args(ann) else (None,)
                if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                    return {"type": "array", "items": item_type.model_json_schema()}
                # List of something else → not a nested Pydantic model
                return None
        except Exception:
            return None
        return None

    def infer_column_type(py_t: Any) -> str:
        """Map a (possibly nested) annotation to the closest ``ColumnType`` label."""

        origin = get_origin(py_t)

        # ---- Optional / Union handling ------------------------------------
        if origin in (
            Union,
            UnionType,
        ):  # handles both ``typing.Union`` and ``|`` syntax
            non_none_args = [
                arg for arg in get_args(py_t) if arg is not type(None)
            ]  # noqa: E721
            if not non_none_args:
                return ColumnType.str

            resolved = {infer_column_type(arg) for arg in non_none_args}
            return resolved.pop() if len(resolved) == 1 else ColumnType.str

        # ---- Container & scalar types -------------------------------------
        origin_or_self = origin or py_t

        if isinstance(origin_or_self, type) and issubclass(origin_or_self, BaseModel):
            # Pydantic v2 RootModel[T]: infer from the root field's annotation (if available)
            try:
                if getattr(origin_or_self, "__pydantic_root_model__", False):
                    root_field = getattr(origin_or_self, "model_fields", {}).get("root")
                    if root_field is not None:
                        return infer_column_type(root_field.annotation)
                    # Fallback if root metadata is unavailable
                    return ColumnType.list
            except Exception:
                pass
            # Regular BaseModel → dict
            return ColumnType.dict
        if origin_or_self in (dict, Mapping):
            return ColumnType.dict
        if origin_or_self in (list, tuple, set, Sequence):
            return ColumnType.list

        primitive_map = {
            str: ColumnType.str,
            int: ColumnType.int,
            float: ColumnType.float,
            bool: ColumnType.bool,
            datetime: ColumnType.datetime,
            date: ColumnType.date,
            time: ColumnType.time,
        }
        if origin_or_self in primitive_map:
            return primitive_map[origin_or_self]

        # ---- Recursive inspection of type arguments -----------------------
        # For generics like ``List[Foo]`` we've already classified via the
        # container check. For anything else we inspect the first generic
        # parameter (if there is one) hoping it gives a more specific hint.
        for arg in get_args(py_t):
            nested_type = infer_column_type(arg)
            if nested_type != ColumnType.str:
                return nested_type

        # ---- Fallback ------------------------------------------------------
        return ColumnType.str

    unify_fields: dict[str, dict[str, Any]] = {}

    for name, field in fields_source.items():

        annotation = field.annotation
        # Prefer explicit Unify type override when present
        unify_type_override = _extract_unify_type(field, annotation)
        # Try to obtain a precise JSON Schema for nested Pydantic types when no override is present
        json_schema: dict | None = None
        if unify_type_override is None:
            json_schema = _pydantic_json_schema_for(annotation)
        column_type = unify_type_override or infer_column_type(annotation)

        # If we have a JSON Schema, pass it as the field type (backend now accepts JSON-serialized types)
        entry: dict[str, Any] = {
            "type": (
                json.dumps(json_schema) if json_schema is not None else column_type
            ),
            "mutable": True,
        }
        if getattr(field, "description", None):
            entry["description"] = field.description.strip()

        unify_fields[name] = entry

    return unify_fields
