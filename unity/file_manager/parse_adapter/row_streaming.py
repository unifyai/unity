"""Re-export shim -- canonical implementation lives in ``unity.common.pipeline.row_streaming``.

Existing FM callers can continue importing from here without changes.
"""

from unity.common.pipeline.row_streaming import (  # noqa: F401
    iter_table_input_row_batches,
    iter_table_input_rows,
)

__all__ = [
    "iter_table_input_rows",
    "iter_table_input_row_batches",
]
