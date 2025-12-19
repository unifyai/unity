"""
KnowledgeManager-specific settings.

These settings are composed into the global ProductionSettings.
Environment variables use the prefix UNITY_KNOWLEDGE_.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class KnowledgeSettings(BaseSettings):
    """KnowledgeManager settings.

    Attributes:
        ENABLED: Whether KnowledgeManager is enabled.
        IMPL: Implementation type - "real" or "simulated".
        MODEL_MAX_INPUT_TOKENS: Maximum input tokens for model context.
    """

    ENABLED: bool = False
    IMPL: str = "real"
    MODEL_MAX_INPUT_TOKENS: int = 128000

    model_config = SettingsConfigDict(
        env_prefix="UNITY_KNOWLEDGE_",
        case_sensitive=True,
        extra="ignore",
    )
