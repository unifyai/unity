"""Settings for the DashboardManager module."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DashboardSettings(BaseSettings):
    """DashboardManager configuration.

    Supports both 'real' and 'simulated' implementations.
    """

    IMPL: str = Field(
        default="real",
        description="DashboardManager implementation: 'real' or 'simulated'.",
    )

    ENABLED: bool = Field(
        default=True,
        description="Whether the DashboardManager is enabled.",
    )

    model_config = SettingsConfigDict(
        env_prefix="UNITY_DASHBOARD_",
        case_sensitive=True,
        extra="ignore",
    )
