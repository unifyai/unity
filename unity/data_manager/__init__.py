"""
DataManager module.

DataManager provides the canonical data operations layer for any Unify context.
It is the single source of truth for filter/search/reduce/join/insert/update/
delete/vectorize operations.

Usage
-----
>>> from unity.data_manager import DataManager
>>>
>>> dm = DataManager()
>>>
>>> # Create a table
>>> dm.create_table("MH/arrears", fields={"tenant_id": "int", "amount": "float"})
>>>
>>> # Insert data
>>> dm.insert_rows("MH/arrears", [{"tenant_id": 1, "amount": 100.0}])
>>>
>>> # Query data
>>> rows = dm.filter("MH/arrears", filter="amount > 50")
>>>
>>> # Aggregate
>>> total = dm.reduce("MH/arrears", metric="sum", column="amount")

Semantic Ownership
------------------
DataManager owns the ``Data/*`` namespace for pipeline/API-derived datasets.
However, its primitives work on ANY context, including ``Files/*``.

This allows FileManager to delegate its data operations internally while
retaining semantic ownership of file-derived contexts.

No Tool Loops
-------------
DataManager exposes pure primitives with no ask/update tool loops.
High-level orchestration is handled by Actor composing these primitives.

See Also
--------
BaseDataManager : Abstract interface with full docstrings for all methods.
FileManager : File-specific convenience methods that delegate to DataManager.
"""

from typing import TYPE_CHECKING

__all__ = [
    "DataManager",
    "BaseDataManager",
    "TableDescription",
    "TableSchema",
    "ColumnInfo",
]


def __getattr__(name: str):
    """Lazy import to avoid circular import with settings."""
    if name == "DataManager":
        from unity.data_manager.data_manager import DataManager

        return DataManager
    elif name == "BaseDataManager":
        from unity.data_manager.base import BaseDataManager

        return BaseDataManager
    elif name == "TableDescription":
        from unity.data_manager.types.table import TableDescription

        return TableDescription
    elif name == "TableSchema":
        from unity.data_manager.types.table import TableSchema

        return TableSchema
    elif name == "ColumnInfo":
        from unity.data_manager.types.table import ColumnInfo

        return ColumnInfo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from unity.data_manager.data_manager import DataManager
    from unity.data_manager.base import BaseDataManager
    from unity.data_manager.types.table import (
        TableDescription,
        TableSchema,
        ColumnInfo,
    )
