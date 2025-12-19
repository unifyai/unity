"""
GuidanceManager-specific settings.

These settings are composed into the global ProductionSettings.
Environment variables use the prefix UNITY_GUIDANCE_.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class GuidanceSettings(BaseSettings):
    """GuidanceManager settings.

    Attributes:
        ENABLED: Whether GuidanceManager is enabled.
        IMPL: Implementation type - "real" or "simulated".
    """

    ENABLED: bool = False
    IMPL: str = "real"

    model_config = SettingsConfigDict(
        env_prefix="UNITY_GUIDANCE_",
        case_sensitive=True,
        extra="ignore",
    )
