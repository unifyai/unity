"""
DataManager operation implementations.

This module exports implementation functions that are called by
DataManager methods. The implementations are separated to keep
data_manager.py thin and focused on orchestration.

NOTE: Docstrings for these operations live in base.py on the
abstract methods. Do not duplicate docstrings here.
"""

from unity.data_manager.ops.table_ops import (
    create_table_impl,
    describe_table_impl,
    list_tables_impl,
    delete_table_impl,
)
from unity.data_manager.ops.query_ops import (
    filter_impl,
    search_impl,
    reduce_impl,
)
from unity.data_manager.ops.mutation_ops import (
    insert_rows_impl,
    update_rows_impl,
    delete_rows_impl,
)
from unity.data_manager.ops.join_ops import (
    filter_join_impl,
    search_join_impl,
    filter_multi_join_impl,
    search_multi_join_impl,
)
from unity.data_manager.ops.plot_ops import (
    generate_plot,
    generate_plots_batch,
)

__all__ = [
    # Table operations
    "create_table_impl",
    "describe_table_impl",
    "list_tables_impl",
    "delete_table_impl",
    # Query operations
    "filter_impl",
    "search_impl",
    "reduce_impl",
    # Mutation operations
    "insert_rows_impl",
    "update_rows_impl",
    "delete_rows_impl",
    # Join operations
    "filter_join_impl",
    "search_join_impl",
    "filter_multi_join_impl",
    "search_multi_join_impl",
    # Plot operations
    "generate_plot",
    "generate_plots_batch",
]
