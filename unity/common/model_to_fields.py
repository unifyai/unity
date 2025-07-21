from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Mapping, Sequence, Union, get_args, get_origin
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

    Examples
    --------
    >>> fields_dict = model_to_fields(Contact)
    >>> unify.create_fields(fields_dict, context=ctx)
    """
    fields_source = model.model_fields

    def infer_column_type(py_t: Any) -> str:
        """
        Recursively map an arbitrary (possibly deeply-nested) Python type
        annotation to the closest ``ColumnType`` label.

        The rules are:
        • ``BaseModel`` subclasses and ``Mapping``/``dict`` → ``dict``
        • Any ``Sequence`` (list, tuple, set, …) → ``list``
        • Primitive scalars → their matching scalar type
        • ``Union``/``|`` types are flattened – if all non-None variants resolve
          to the *same* ``ColumnType`` that label is used, otherwise we fall
          back to ``str``.
        • Anything unknown falls back to ``str``.
        """

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

        # ---- Container types ----------------------------------------------
        origin_or_self = origin or py_t

        if isinstance(origin_or_self, type) and issubclass(origin_or_self, BaseModel):
            return ColumnType.dict
        if origin_or_self in (dict, Mapping):
            return ColumnType.dict
        if origin_or_self in (list, tuple, set, Sequence):
            return ColumnType.list

        # ---- Scalar primitives -------------------------------------------
        if origin_or_self is str:
            return ColumnType.str
        if origin_or_self is int:
            return ColumnType.int
        if origin_or_self is float:
            return ColumnType.float
        if origin_or_self is bool:
            return ColumnType.bool
        if origin_or_self is datetime:
            return ColumnType.datetime
        if origin_or_self is date:
            return ColumnType.date
        if origin_or_self is time:
            return ColumnType.time

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
        column_type = infer_column_type(annotation)

        entry: dict[str, Any] = {"type": column_type, "mutable": True}
        if getattr(field, "description", None):
            entry["description"] = field.description.strip()

        unify_fields[name] = entry

    return unify_fields
