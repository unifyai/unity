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
    # LLM Provider Credentials
    # ─────────────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_APPLICATION_CREDENTIALS: str = ""
    VERTEXAI_LOCATION: str = ""
    VERTEXAI_PROJECT: str = ""
    UNITY_VALIDATE_LLM_PROVIDERS: bool = True

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
    # Conductor Manager Configuration
    # ─────────────────────────────────────────────────────────────────────────
    # Foundational managers (cannot be disabled, only implementation switched):
    #   - Actor, ContactManager, TranscriptManager, TaskScheduler, ConversationManager
    # Optional managers (can be disabled via ENABLED=False):
    #   - KnowledgeManager, GuidanceManager, SecretManager, SkillManager,
    #     WebSearcher, GlobalFileManager

    # -- Foundational managers (implementation only) --
    # Actor: hierarchical | single_function | code_act | simulated
    UNITY_ACTOR_IMPL: str = "hierarchical"
    # ContactManager: real | simulated
    UNITY_CONTACTS_IMPL: str = "real"
    # TranscriptManager: real | simulated
    UNITY_TRANSCRIPTS_IMPL: str = "real"
    # TaskScheduler: real | simulated
    UNITY_TASKS_IMPL: str = "real"
    # ConversationManager: real | simulated
    UNITY_CONVERSATION_IMPL: str = "real"

    # -- Optional managers (disabled by default for minimal initial rollout) --
    # These will be enabled incrementally as they become stable and fully featured.
    # KnowledgeManager
    UNITY_KNOWLEDGE_ENABLED: bool = False
    UNITY_KNOWLEDGE_IMPL: str = "real"
    # GuidanceManager
    UNITY_GUIDANCE_ENABLED: bool = False
    UNITY_GUIDANCE_IMPL: str = "real"
    # SecretManager
    UNITY_SECRETS_ENABLED: bool = False
    UNITY_SECRETS_IMPL: str = "real"
    # SkillManager
    UNITY_SKILLS_ENABLED: bool = False
    UNITY_SKILLS_IMPL: str = "real"
    # WebSearcher
    UNITY_WEB_SEARCH_ENABLED: bool = False
    UNITY_WEB_SEARCH_IMPL: str = "real"
    # GlobalFileManager
    UNITY_FILES_ENABLED: bool = False
    UNITY_FILES_IMPL: str = "real"

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
        "UNITY_KNOWLEDGE_ENABLED",
        "UNITY_GUIDANCE_ENABLED",
        "UNITY_SECRETS_ENABLED",
        "UNITY_SKILLS_ENABLED",
        "UNITY_WEB_SEARCH_ENABLED",
        "UNITY_FILES_ENABLED",
        "UNITY_VALIDATE_LLM_PROVIDERS",
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

    def validate_llm_providers(self) -> None:
        """Validate that all required LLM provider credentials are set.

        Raises:
            RuntimeError: If any LLM provider credential is missing or empty.
        """
        if not self.UNITY_VALIDATE_LLM_PROVIDERS:
            return
        required = {
            "OPENAI_API_KEY": self.OPENAI_API_KEY,
            "ANTHROPIC_API_KEY": self.ANTHROPIC_API_KEY,
            "GOOGLE_APPLICATION_CREDENTIALS": self.GOOGLE_APPLICATION_CREDENTIALS,
            "VERTEXAI_LOCATION": self.VERTEXAI_LOCATION,
            "VERTEXAI_PROJECT": self.VERTEXAI_PROJECT,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(
                f"Missing required LLM provider credentials: {', '.join(missing)}. "
                "Set these environment variables before initializing Unity.",
            )


# Singleton instance for production code
SETTINGS = ProductionSettings()
