"""
MemoryManager-specific settings.

These settings are composed into the global ProductionSettings.
Environment variables use the prefix UNITY_MEMORY_.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class MemorySettings(BaseSettings):
    """MemoryManager settings.

    Attributes:
        IMPL: Implementation type - "real" or "simulated".
        ENABLED: Whether MemoryManager is enabled. When False, MemoryManager
            is not created during ConversationManager initialization.
        CONTACTS: Auto-detect and update contacts from transcript chunks.
        BIOS: Auto-refresh contact bio columns from transcript chunks.
        ROLLING_SUMMARIES: Auto-refresh rolling conversation summaries.
        RESPONSE_POLICIES: Auto-refresh per-contact response policies.
        KNOWLEDGE: Auto-extract facts to the knowledge base.
        TASKS: Auto-update the task schedule from transcript chunks.
    """

    IMPL: str = "real"
    ENABLED: bool = True

    CONTACTS: bool = False
    BIOS: bool = True
    ROLLING_SUMMARIES: bool = True
    RESPONSE_POLICIES: bool = True
    KNOWLEDGE: bool = False
    TASKS: bool = False

    model_config = SettingsConfigDict(
        env_prefix="UNITY_MEMORY_",
        case_sensitive=True,
        extra="ignore",
    )
