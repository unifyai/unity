"""
ContactManager-specific settings.

These settings are composed into the global ProductionSettings.
Environment variables use the prefix UNITY_CONTACT_.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


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
