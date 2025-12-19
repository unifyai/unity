"""
SecretManager-specific settings.

These settings are composed into the global ProductionSettings.
Environment variables use the prefix UNITY_SECRET_.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class SecretSettings(BaseSettings):
    """SecretManager settings.

    Attributes:
        ENABLED: Whether SecretManager is enabled.
        IMPL: Implementation type - "real" or "simulated".
        DOTENV_PATH: Path to the .env file for secret storage.
    """

    ENABLED: bool = False
    IMPL: str = "real"
    DOTENV_PATH: str = ""

    model_config = SettingsConfigDict(
        env_prefix="UNITY_SECRET_",
        case_sensitive=True,
        extra="ignore",
    )
