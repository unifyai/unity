"""PrimitiveScope: the single knob for controlling which managers are exposed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import FrozenSet

# Canonical manager aliases - the only valid values for scoped_managers.
# This is the authoritative list; ToolSurfaceRegistry.MANAGERS must match.
VALID_MANAGER_ALIASES: frozenset[str] = frozenset(
    {
        "contacts",
        "tasks",
        "transcripts",
        "knowledge",
        "secrets",
        "guidance",
        "web",
        "data",
        "files",
    },
)


@dataclass(frozen=True, slots=True)
class PrimitiveScope:
    """
    Defines which managers are exposed in a deployment.

    This is the single source of truth for scoping. All downstream consumers
    (Primitives, StateManagerEnvironment, FunctionManager, prompt builders)
    read from this object.

    Attributes
    ----------
    scoped_managers : frozenset[str]
        Set of manager aliases to expose. Must be non-empty and contain only
        valid manager aliases from VALID_MANAGER_ALIASES.

    Examples
    --------
    # Files-only deployment
    scope = PrimitiveScope(scoped_managers=frozenset({"files"}))

    # Full deployment
    scope = PrimitiveScope(scoped_managers=VALID_MANAGER_ALIASES)
    """

    scoped_managers: "FrozenSet[str]"

    def __post_init__(self) -> None:
        """Validate scoped_managers."""
        if not self.scoped_managers:
            raise ValueError("scoped_managers must be non-empty")

        invalid = self.scoped_managers - VALID_MANAGER_ALIASES
        if invalid:
            raise ValueError(
                f"Invalid manager aliases: {sorted(invalid)}. "
                f"Valid aliases: {sorted(VALID_MANAGER_ALIASES)}",
            )

    @property
    def scope_key(self) -> str:
        """
        Stable, deterministic key for caching and registry lookups.

        Returns a sorted comma-separated string of manager aliases.
        """
        return ",".join(sorted(self.scoped_managers))

    def includes(self, manager_alias: str) -> bool:
        """Check if a manager alias is in scope."""
        return manager_alias in self.scoped_managers

    @classmethod
    def all_managers(cls) -> "PrimitiveScope":
        """Create a scope with all managers exposed."""
        return cls(scoped_managers=VALID_MANAGER_ALIASES)

    @classmethod
    def single(cls, manager_alias: str) -> "PrimitiveScope":
        """Create a scope with a single manager exposed."""
        return cls(scoped_managers=frozenset({manager_alias}))
