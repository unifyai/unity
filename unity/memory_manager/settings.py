"""
MemoryManager-specific settings.

These settings are composed into the global ProductionSettings.
Environment variables use the prefix UNITY_MEMORY_.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class MemorySettings(BaseSettings):
    """MemoryManager settings.

    Attributes:
        REGISTER_UPDATE_CALLBACKS: Whether to register event bus callbacks
            for automatic memory updates from transcripts.
    """

    REGISTER_UPDATE_CALLBACKS: bool = True

    model_config = SettingsConfigDict(
        env_prefix="UNITY_MEMORY_",
        case_sensitive=True,
        extra="ignore",
    )
