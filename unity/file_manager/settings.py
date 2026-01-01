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
        CONSOLE_BASE_URL: Base URL for Unify Console (Plot API).
        PLOT_API_ENDPOINT: Endpoint path for plot creation.
        PLOT_API_TIMEOUT: Timeout in seconds for plot API requests.
        PLOT_API_MAX_RETRIES: Maximum number of retries for plot API requests.
        PLOT_API_RETRY_BACKOFF: Base backoff time in seconds for retries.
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

    # Plot API settings
    CONSOLE_BASE_URL: str = "https://console.unify.ai"
    PLOT_API_ENDPOINT: str = "/api/plot/create"
    PLOT_API_TIMEOUT: float = 30.0
    PLOT_API_MAX_RETRIES: int = 3
    PLOT_API_RETRY_BACKOFF: float = 1.0

    model_config = SettingsConfigDict(
        env_prefix="UNITY_FILE_",
        case_sensitive=True,
        extra="ignore",
    )
