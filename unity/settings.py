"""
unity/settings.py
==================

Centralized production environment settings using pydantic-settings.

These settings are used in the deployed system and are inherited by test settings.
All settings can be overridden via environment variables or .env file.
"""

from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from unity.actor.settings import ActorSettings
from unity.conductor.settings import ConductorSettings
from unity.contact_manager.settings import ContactSettings
from unity.conversation_manager.settings import ConversationSettings
from unity.file_manager.settings import FileSettings
from unity.function_manager.settings import FunctionSettings
from unity.guidance_manager.settings import GuidanceSettings
from unity.image_manager.settings import ImageSettings
from unity.knowledge_manager.settings import KnowledgeSettings
from unity.memory_manager.settings import MemorySettings
from unity.secret_manager.settings import SecretSettings
from unity.task_scheduler.settings import TaskSettings
from unity.transcript_manager.settings import TranscriptSettings
from unity.web_searcher.settings import WebSettings


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
    UNIFY_MODEL: str = "gpt-5.2@openai"
    UNIFY_CACHE: bool | str = True

    # ─────────────────────────────────────────────────────────────────────────
    # LLM Provider Credentials
    # ─────────────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    UNITY_VALIDATE_LLM_PROVIDERS: bool = True

    # ─────────────────────────────────────────────────────────────────────────
    # External Service Credentials
    # ─────────────────────────────────────────────────────────────────────────
    ORCHESTRA_ADMIN_KEY: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Infrastructure URLs
    # ─────────────────────────────────────────────────────────────────────────
    UNIFY_BASE_URL: str = "https://api.unify.ai/v0"

    # ─────────────────────────────────────────────────────────────────────────
    # Logging / Observability
    # ─────────────────────────────────────────────────────────────────────────
    PYTEST_LOG_TO_FILE: bool = True
    # Directory for Unity LOGGER file output (async tool loop, managers, etc.)
    # When set, logs are written to {UNITY_LOG_DIR}/unity.log
    # Default: None (console only)
    UNITY_LOG_DIR: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # OpenTelemetry Tracing
    # ─────────────────────────────────────────────────────────────────────────
    # Master switch for OTel tracing.
    # - UNITY_OTEL=false (default): OTel tracing disabled
    # - UNITY_OTEL=true: OTel tracing enabled, creates TracerProvider if needed
    # - UNITY_OTEL_ENDPOINT: OTLP endpoint for trace export (optional)
    # - UNITY_OTEL_LOG_DIR: Directory for file-based span export (optional)
    #
    # When enabled, manager operations and async tool loops create spans that
    # propagate trace context to downstream libraries (unillm, unify).
    #
    # File-based span export:
    # When UNITY_OTEL_LOG_DIR is set, spans are written to JSONL files keyed
    # by trace_id. This enables full-stack trace correlation with Orchestra
    # (which runs in a separate process but receives the traceparent header).
    UNITY_OTEL: bool = False
    UNITY_OTEL_ENDPOINT: str = ""
    UNITY_OTEL_LOG_DIR: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Debug Modes (performance overhead, development-only)
    # ─────────────────────────────────────────────────────────────────────────
    ASYNCIO_DEBUG: bool = False
    ASYNCIO_DEBUG_VERBOSE: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Test Infrastructure
    # ─────────────────────────────────────────────────────────────────────────
    # Fixed datetime for LLM cache consistency in tests (ISO format string)
    # When set, _get_now() returns this fixed datetime instead of datetime.now()
    UNITY_FIXED_DATETIME: str = ""
    # Log subdirectory for LLM I/O log files (datetime-prefixed for ordering)
    UNITY_LOG_SUBDIR: str = ""
    # Terminal socket name for tmux isolation; also used as log subdir fallback
    # when UNITY_LOG_SUBDIR is not set
    UNITY_TEST_SOCKET: str = ""
    # Explicit repository root for log file placement (e.g., worktrees)
    UNITY_LOG_ROOT: str = ""
    # Test mode flag
    TEST: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Feature Flags
    # ─────────────────────────────────────────────────────────────────────────
    UNITY_SEMANTIC_CACHE: bool = False
    UNITY_READONLY_ASK_GUARD: bool = True
    FIRST_ASK_TOOL_IS_SEARCH: bool = True
    FIRST_MUTATION_TOOL_IS_ASK: bool = True
    STAGING: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Conductor Manager Configuration
    # ─────────────────────────────────────────────────────────────────────────
    # Foundational managers (cannot be disabled, only implementation switched):
    #   - Actor, ContactManager, TranscriptManager, TaskScheduler, ConversationManager
    # Optional managers (can be disabled via ENABLED=False):
    #   - KnowledgeManager, GuidanceManager, SecretManager,
    #     WebSearcher, GlobalFileManager

    # ─────────────────────────────────────────────────────────────────────────
    # Composed Manager Settings
    # ─────────────────────────────────────────────────────────────────────────
    # Each manager owns its settings in its own settings.py file.
    # Access via SETTINGS.contact.IMPL, SETTINGS.transcript.IMPL, etc.
    actor: ActorSettings = Field(default_factory=ActorSettings)
    conductor: ConductorSettings = Field(default_factory=ConductorSettings)
    contact: ContactSettings = Field(default_factory=ContactSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    file: FileSettings = Field(default_factory=FileSettings)
    function: FunctionSettings = Field(default_factory=FunctionSettings)
    guidance: GuidanceSettings = Field(default_factory=GuidanceSettings)
    image: ImageSettings = Field(default_factory=ImageSettings)
    knowledge: KnowledgeSettings = Field(default_factory=KnowledgeSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    secret: SecretSettings = Field(default_factory=SecretSettings)
    task: TaskSettings = Field(default_factory=TaskSettings)
    transcript: TranscriptSettings = Field(default_factory=TranscriptSettings)
    web: WebSettings = Field(default_factory=WebSettings)

    # ─────────────────────────────────────────────────────────────────────────
    # Validators
    # ─────────────────────────────────────────────────────────────────────────
    @field_validator("UNIFY_CACHE", mode="before")
    @classmethod
    def parse_cache(cls, v: Any) -> bool | str:
        return _parse_bool_or_str(v)

    @field_validator(
        "ASYNCIO_DEBUG",
        "ASYNCIO_DEBUG_VERBOSE",
        "PYTEST_LOG_TO_FILE",
        "UNITY_SEMANTIC_CACHE",
        "UNITY_READONLY_ASK_GUARD",
        "FIRST_ASK_TOOL_IS_SEARCH",
        "FIRST_MUTATION_TOOL_IS_ASK",
        "STAGING",
        "TEST",
        "UNITY_VALIDATE_LLM_PROVIDERS",
        "UNITY_OTEL",
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
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(
                f"Missing required LLM provider credentials: {', '.join(missing)}. "
                "Set these environment variables before initializing Unity.",
            )


# Singleton instance for production code
SETTINGS = ProductionSettings()
