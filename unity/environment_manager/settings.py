"""
EnvironmentManager-specific settings.

These settings are composed into the global ProductionSettings.
Environment variables use the prefix UNITY_ENVIRONMENT_.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvironmentSettings(BaseSettings):
    """EnvironmentManager settings.

    Attributes:
        ENABLED: Whether EnvironmentManager is enabled.
        IMPL: Implementation type - "real" or "simulated".
    """

    ENABLED: bool = True
    IMPL: str = "real"

    model_config = SettingsConfigDict(
        env_prefix="UNITY_ENVIRONMENT_",
        case_sensitive=True,
        extra="ignore",
    )
