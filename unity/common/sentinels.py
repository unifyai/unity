"""
Common sentinel types used across managers.

The _UnsetSentinel type provides a stable, human-readable repr so that
inspect.signature(...) stringifications in prompts are deterministic
across processes (renders as "<UNSET>").

The _DisabledSentinel type marks a manager as explicitly disabled,
distinguishing it from "not provided" (which falls back to defaults).
"""

from __future__ import annotations


class _UnsetSentinel:
    __slots__ = ()

    def __repr__(self) -> str:
        # Stable textual form to avoid process-specific object addresses
        return "<UNSET>"


class _DisabledSentinel:
    """Sentinel indicating a manager should be explicitly disabled (not created)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<DISABLED>"


# Module-level singleton for convenience
DISABLED = _DisabledSentinel()
