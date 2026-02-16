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
    """

    IMPL: str = "real"
    ENABLED: bool = False

    model_config = SettingsConfigDict(
        env_prefix="UNITY_MEMORY_",
        case_sensitive=True,
        extra="ignore",
    )
