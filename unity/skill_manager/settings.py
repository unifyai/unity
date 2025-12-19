"""
SkillManager-specific settings.

These settings are composed into the global ProductionSettings.
Environment variables use the prefix UNITY_SKILL_.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class SkillSettings(BaseSettings):
    """SkillManager settings.

    Attributes:
        ENABLED: Whether SkillManager is enabled.
        IMPL: Implementation type - "real" or "simulated".
    """

    ENABLED: bool = False
    IMPL: str = "real"

    model_config = SettingsConfigDict(
        env_prefix="UNITY_SKILL_",
        case_sensitive=True,
        extra="ignore",
    )
