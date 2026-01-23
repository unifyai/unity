"""Centralized global docstrings used across Unity base classes."""

# Shared docstrings

CLEAR_METHOD_DOCSTRING = ("""
    WARNING: Irreversible total data erasure.

    This clear operation will permanently and completely delete all records,
    caches, contexts, tables, and any other stored state managed by this component.
    After it runs, the manager will be in the same state as a brand-new instance
    that has just been initialized for the first time, with absolutely no prior
    state preserved in any backend or context.

    Mandatory confirmation
    ----------------------
    • Before invoking this method, ALWAYS request explicit confirmation/clarification from the user
      to verify that they truly intend to perform this irreversible deletion.
    • When an interactive clarification/confirmation channel exists (for example, a
      request_clarification tool or equivalent UI), use it to obtain a clear affirmative
      response from the user first. If no such channel exists, do not proceed until
      you have explicit confirmation from the user in the outer interaction.

    Only call this after the user has explicitly confirmed and you are 100% sure that
    you want to erase all data. This action cannot be undone.
    """).strip()
