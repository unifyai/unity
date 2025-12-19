"""
FileManager-specific settings.

These settings are composed into the global ProductionSettings.
Environment variables use the prefix UNITY_FILE_.

Note: Parser-specific settings (EMBEDDING_*, SUMMARY_*, etc.) are in
file_parsers/settings.py as FileParserSettings, which is intentionally
separate to keep parsing concerns modular.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class FileSettings(BaseSettings):
    """FileManager settings.

    Attributes:
        ENABLED: Whether FileManager is enabled.
        IMPL: Implementation type - "real" or "simulated".
        CODESANDBOX_SERVICE_BASE_URL: Base URL for CodeSandbox service.
        CODESANDBOX_SERVICE_PORT: Port for CodeSandbox service.
        CODESANDBOX_API_TOKEN: API token for CodeSandbox.
        INTERACT_API_BASE: Base URL for Interact API.
        INTERACT_KEY: API key for Interact.
        INTERACT_SECRET: API secret for Interact.
        INTERACT_PERSON_ID: Person ID for Interact.
        INTERACT_TENANT: Tenant for Interact.
    """

    ENABLED: bool = False
    IMPL: str = "real"

    # CodeSandbox adapter settings
    CODESANDBOX_SERVICE_BASE_URL: str = ""
    CODESANDBOX_SERVICE_PORT: str = "3100"
    CODESANDBOX_API_TOKEN: str = ""

    # Interact adapter settings
    INTERACT_API_BASE: str = ""
    INTERACT_KEY: str = ""
    INTERACT_SECRET: str = ""
    INTERACT_PERSON_ID: str = ""
    INTERACT_TENANT: str = ""

    model_config = SettingsConfigDict(
        env_prefix="UNITY_FILE_",
        case_sensitive=True,
        extra="ignore",
    )
