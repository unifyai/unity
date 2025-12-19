"""
WebSearcher-specific settings.

These settings are composed into the global ProductionSettings.
Environment variables use the prefix UNITY_WEB_.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class WebSettings(BaseSettings):
    """WebSearcher settings.

    Attributes:
        ENABLED: Whether WebSearcher is enabled.
        IMPL: Implementation type - "real" or "simulated".
        TAVILY_API_KEY: API key for Tavily web search service.
    """

    ENABLED: bool = False
    IMPL: str = "real"
    TAVILY_API_KEY: str = ""

    model_config = SettingsConfigDict(
        env_prefix="UNITY_WEB_",
        case_sensitive=True,
        extra="ignore",
    )
