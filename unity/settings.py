"""
unity/settings.py
==================

Centralized production environment settings using pydantic-settings.

These settings are used in the deployed system and are inherited by test settings.
All settings can be overridden via environment variables or .env file.
"""

from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_bool_or_str(v: Any) -> bool | str:
    """Parse a value that can be bool, bool-string, or pass-through string."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        lower = v.lower()
        if lower in ("true", "yes", "1"):
            return True
        if lower in ("false", "no", "0"):
            return False
        return v  # Pass through for special cache modes like "read-only"
    return bool(v)


def _parse_bool(v: Any) -> bool:
    """Parse a value as boolean."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "yes", "1", "on")
    return bool(v)


class ProductionSettings(BaseSettings):
    """Production environment settings used in deployed system and tests.

    All settings can be overridden via environment variables.
    Test settings (TestingSettings) inherit from this class.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Core LLM Settings
    # ─────────────────────────────────────────────────────────────────────────
    UNIFY_MODEL: str = "gpt-5.1@openai"
    UNIFY_CACHE: bool | str = True

    # ─────────────────────────────────────────────────────────────────────────
    # Debugging / Observability
    # ─────────────────────────────────────────────────────────────────────────
    LLM_IO_DEBUG: bool = True
    ASYNCIO_DEBUG: bool = False
    ASYNCIO_VERBOSE_DEBUG: bool = False
    PYTEST_LOG_TO_FILE: bool = True

    # ─────────────────────────────────────────────────────────────────────────
    # Feature Flags
    # ─────────────────────────────────────────────────────────────────────────
    UNITY_SEMANTIC_CACHE: bool = False
    UNITY_READONLY_ASK_GUARD: bool = True
    FIRST_ASK_TOOL_IS_SEARCH: bool = True
    FIRST_MUTATION_TOOL_IS_ASK: bool = True

    # ─────────────────────────────────────────────────────────────────────────
    # Logging Control
    # ─────────────────────────────────────────────────────────────────────────
    UNITY_SILENCE_HTTPX: bool = True
    UNITY_SILENCE_URLLIB3: bool = True
    UNITY_SILENCE_OPENAI: bool = True
    UNITY_LOG_ONLY_PROJECT: bool = True
    UNITY_LOG_INCLUDE_PREFIXES: str = "unity"

    # ─────────────────────────────────────────────────────────────────────────
    # Validators
    # ─────────────────────────────────────────────────────────────────────────
    @field_validator("UNIFY_CACHE", mode="before")
    @classmethod
    def parse_cache(cls, v: Any) -> bool | str:
        return _parse_bool_or_str(v)

    @field_validator(
        "LLM_IO_DEBUG",
        "ASYNCIO_DEBUG",
        "ASYNCIO_VERBOSE_DEBUG",
        "PYTEST_LOG_TO_FILE",
        "UNITY_SEMANTIC_CACHE",
        "UNITY_READONLY_ASK_GUARD",
        "FIRST_ASK_TOOL_IS_SEARCH",
        "FIRST_MUTATION_TOOL_IS_ASK",
        "UNITY_SILENCE_HTTPX",
        "UNITY_SILENCE_URLLIB3",
        "UNITY_SILENCE_OPENAI",
        "UNITY_LOG_ONLY_PROJECT",
        mode="before",
    )
    @classmethod
    def parse_bool_fields(cls, v: Any) -> bool:
        return _parse_bool(v)

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


# Singleton instance for production code
SETTINGS = ProductionSettings()
