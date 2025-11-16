"""
Common sentinel types used across managers.

The _UnsetSentinel type provides a stable, human-readable repr so that
inspect.signature(...) stringifications in prompts are deterministic
across processes (renders as "<UNSET>").
"""

from __future__ import annotations


class _UnsetSentinel:
    __slots__ = ()

    def __repr__(self) -> str:
        # Stable textual form to avoid process-specific object addresses
        return "<UNSET>"
