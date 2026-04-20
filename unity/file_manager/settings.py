"""
FileManager-specific settings.

These settings are composed into the global ProductionSettings.
Environment variables use the prefix UNITY_FILE_.

Note: Parser-specific settings (EMBEDDING_*, SUMMARY_*, etc.) are in
file_parsers/settings.py as FileParserSettings, which is intentionally
separate to keep parsing concerns modular.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_local_root() -> str:
    """Return the resolved local file root directory.

    Uses ``SETTINGS.UNITY_LOCAL_ROOT`` when set, otherwise defaults to
    ``~/Unity/Local``.  All code that needs the local file root should
    call this function instead of hard-coding the path.
    """
    from unity.settings import SETTINGS

    explicit = SETTINGS.UNITY_LOCAL_ROOT.strip()
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    return str(Path.home() / "Unity" / "Local")


class FileSettings(BaseSettings):
    """FileManager settings.

    Attributes:
        ENABLED: Whether FileManager is enabled.
        IMPL: Implementation type - "real" or "simulated".
        PLOT_API_ENDPOINT: Endpoint path for plot creation.
        PLOT_API_TIMEOUT: Timeout in seconds for plot API requests.
        PLOT_API_MAX_RETRIES: Maximum number of retries for plot API requests.
        PLOT_API_RETRY_BACKOFF: Base backoff time in seconds for retries.
        TABLE_VIEW_API_ENDPOINT: Endpoint path for table view creation.
        TABLE_VIEW_API_TIMEOUT: Timeout in seconds for table view API requests.
        TABLE_VIEW_API_MAX_RETRIES: Maximum number of retries for table view API requests.
        TABLE_VIEW_API_RETRY_BACKOFF: Base backoff time in seconds for retries.
    """

    ENABLED: bool = False
    IMPL: str = "real"
    IMPLICIT_INGESTION: bool = False
    ATTACHMENT_INGESTION_MAX_WORKERS: int = Field(default=2, ge=1)

    # -- Pipeline worker dispatch (GKE parse/ingest workers) ---
    # When PIPELINE_DISPATCH_ENABLED is true, attachment ingestion uploads the
    # source file to ``PIPELINE_ARTIFACT_BUCKET`` and publishes a
    # ``thread="attachment_parse"`` envelope to the
    # ``unity-parse{ENV_SUFFIX}`` Pub/Sub topic (resolved from
    # ``SETTINGS.GCP_PROJECT_ID`` + ``SETTINGS.ENV_SUFFIX``) instead of running
    # the in-process ``AttachmentIngestionPool``.  Completion events arrive
    # back on the per-assistant topic and are handled by ``CommsManager``.
    PIPELINE_DISPATCH_ENABLED: bool = False
    PIPELINE_ARTIFACT_BUCKET: str = ""

    # Plot API settings
    PLOT_API_ENDPOINT: str = "/logs/plot"
    PLOT_API_TIMEOUT: float = 30.0
    PLOT_API_MAX_RETRIES: int = 3
    PLOT_API_RETRY_BACKOFF: float = 1.0

    # Table View API settings
    TABLE_VIEW_API_ENDPOINT: str = "/logs/table"
    TABLE_VIEW_API_TIMEOUT: float = 30.0
    TABLE_VIEW_API_MAX_RETRIES: int = 3
    TABLE_VIEW_API_RETRY_BACKOFF: float = 1.0

    model_config = SettingsConfigDict(
        env_prefix="UNITY_FILE_",
        case_sensitive=True,
        extra="ignore",
    )
