"""
ContactManager-specific settings.

These settings are composed into the global ProductionSettings.
Environment variables use the prefix UNITY_CONTACT_.
"""

from typing import Final

from pydantic_settings import BaseSettings, SettingsConfigDict

# ────────────────────────────────────────────────────────────────────────────
# ContactMembership.relationship value set
# ────────────────────────────────────────────────────────────────────────────
# These constants are the single source of truth for the four relationship
# values the per-body ContactMembership overlay understands. Every routing,
# filter, and provisioning call site imports from here instead of sprinkling
# string literals through the contact_manager package.

RELATIONSHIP_SELF: Final[str] = "self"
RELATIONSHIP_BOSS: Final[str] = "boss"
RELATIONSHIP_COWORKER: Final[str] = "coworker"
RELATIONSHIP_OTHER: Final[str] = "other"

# Structural name of the per-body ContactMembership overlay table.
# One source of truth for the TableContext declaration, the
# ContextRegistry lookup key, and peer-body overlay context paths
# built during cross-Hive merge fan-out.
CONTACT_MEMBERSHIP_TABLE: Final[str] = "ContactMembership"


class ContactSettings(BaseSettings):
    """ContactManager settings.

    Attributes:
        IMPL: Implementation type - "real" or "simulated".
    """

    IMPL: str = "real"

    model_config = SettingsConfigDict(
        env_prefix="UNITY_CONTACT_",
        case_sensitive=True,
        extra="ignore",
    )
