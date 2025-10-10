"""Centralized global docstrings used across Unity base classes."""

# Shared docstrings

CLEAR_METHOD_DOCSTRING = (
    """
    WARNING: Irreversible total data erasure.

    This clear operation will permanently and completely delete all records,
    caches, contexts, tables, and any other stored state managed by this component.
    After it runs, the manager will be in the same state as a brand-new instance
    that has just been initialized for the first time, with absolutely no prior
    state preserved in any backend or context.

    Only call this if you are 100% sure that you want to erase all data. This
    action cannot be undone.
    """
).strip()
