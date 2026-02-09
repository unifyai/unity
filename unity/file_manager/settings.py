"""
FileManager-specific settings.

These settings are composed into the global ProductionSettings.
Environment variables use the prefix UNITY_FILE_.

Note: Parser-specific settings (EMBEDDING_*, SUMMARY_*, etc.) are in
file_parsers/settings.py as FileParserSettings, which is intentionally
separate to keep parsing concerns modular.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def get_local_root() -> str:
    """Return the resolved local file root directory.

    Uses ``SETTINGS.file.LOCAL_ROOT`` when set, otherwise defaults to
    ``~/Unity/Local``.  All code that needs the local file root should
    call this function instead of hard-coding the path.
    """
    from unity.settings import SETTINGS

    explicit = SETTINGS.file.LOCAL_ROOT.strip()
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    return str(Path.home() / "Unity" / "Local")


class FileSettings(BaseSettings):
    """FileManager settings.

    Attributes:
        ENABLED: Whether FileManager is enabled.
        IMPL: Implementation type - "real" or "simulated".
        LOCAL_ROOT: Root directory for local file operations and the
            CodeActActor working directory.  Defaults to ``~/Unity/Local``.
            Override via ``UNITY_FILE_LOCAL_ROOT`` env var.
        CONSOLE_BASE_URL: Base URL for Unify Console (Plot API).
        PLOT_API_ENDPOINT: Endpoint path for plot creation.
        PLOT_API_TIMEOUT: Timeout in seconds for plot API requests.
        PLOT_API_MAX_RETRIES: Maximum number of retries for plot API requests.
        PLOT_API_RETRY_BACKOFF: Base backoff time in seconds for retries.
    """

    ENABLED: bool = False
    IMPL: str = "real"
    LOCAL_ROOT: str = ""

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
