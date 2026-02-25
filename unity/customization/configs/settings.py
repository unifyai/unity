"""
ConfigManager-specific settings.

These settings are composed into the global ProductionSettings.
Environment variables use the prefix UNITY_CONFIG_.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigSettings(BaseSettings):
    """ConfigManager settings.

    Attributes:
        ENABLED: Whether ConfigManager is enabled.
        IMPL: Implementation type - "real" or "simulated".
    """

    ENABLED: bool = True
    IMPL: str = "real"

    model_config = SettingsConfigDict(
        env_prefix="UNITY_CONFIG_",
        case_sensitive=True,
        extra="ignore",
    )
