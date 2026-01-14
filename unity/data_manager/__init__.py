"""
DataManager module.

DataManager provides the canonical data operations layer for any Unify context.
It is the single source of truth for filter/search/reduce/join/insert/update/
delete/vectorize/plot operations.

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

from unity.data_manager.data_manager import DataManager
from unity.data_manager.base import BaseDataManager
from unity.data_manager.types.table import (
    TableDescription,
    TableSchema,
    ColumnInfo,
)
from unity.data_manager.types.plot import (
    PlotConfig,
    PlotResult,
    PlotType,
)

__all__ = [
    # Main class
    "DataManager",
    # Abstract base
    "BaseDataManager",
    # Table types
    "TableDescription",
    "TableSchema",
    "ColumnInfo",
    # Plot types
    "PlotConfig",
    "PlotResult",
    "PlotType",
]
