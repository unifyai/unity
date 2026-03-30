"""
unity/settings.py
==================

Centralized production environment settings using pydantic-settings.

These settings are used in the deployed system and are inherited by test settings.
All settings can be overridden via environment variables or .env file.
"""

from typing import Any, Literal

from pydantic import Field, field_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from unity.actor.settings import ActorSettings
from unity.contact_manager.settings import ContactSettings
from unity.conversation_manager.settings import ConversationSettings
from unity.data_manager.settings import DataSettings
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


def _parse_bool(v: Any) -> bool:
    """Parse a value as boolean."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "yes", "1", "on")
    return bool(v)


def _parse_deploy_env(v: Any) -> str:
    """Parse the deployment environment setting."""
    if v is None:
        return "production"
    env = str(v).strip().lower() or "production"
    if env not in {"production", "staging", "preview"}:
        raise ValueError("DEPLOY_ENV must be one of production, staging, or preview")
    return env


class ProductionSettings(BaseSettings):
    """Production environment settings used in deployed system and tests.

    All settings can be overridden via environment variables.
    Test settings (TestingSettings) inherit from this class.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Local Workspace
    # ─────────────────────────────────────────────────────────────────────────
    # Root directory for local file operations, CodeActActor working directory,
    # virtual environments, and .env storage.  Defaults to ~/Unity/Local when
    # empty.  Override via UNITY_LOCAL_ROOT env var.
    UNITY_LOCAL_ROOT: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Core LLM Settings
    # ─────────────────────────────────────────────────────────────────────────
    UNIFY_MODEL: str = "claude-4.6-opus@anthropic"

    # ─────────────────────────────────────────────────────────────────────────
    # LLM Provider Credentials
    # ─────────────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: SecretStr = SecretStr("")
    ANTHROPIC_API_KEY: SecretStr = SecretStr("")
    UNITY_VALIDATE_LLM_PROVIDERS: bool = True

    # ─────────────────────────────────────────────────────────────────────────
    # External Service Credentials
    # ─────────────────────────────────────────────────────────────────────────
    ORCHESTRA_ADMIN_KEY: SecretStr = SecretStr("")

    # ─────────────────────────────────────────────────────────────────────────
    # Infrastructure URLs
    # ─────────────────────────────────────────────────────────────────────────
    ORCHESTRA_URL: str = "https://api.unify.ai/v0"

    # ─────────────────────────────────────────────────────────────────────────
    # GCP Project
    # ─────────────────────────────────────────────────────────────────────────
    # GCP project ID for Pub/Sub topics and subscriptions. Override via
    # GCP_PROJECT_ID env var for local development with the Pub/Sub emulator
    # (e.g. "local-test-project" to match Communication's local.sh).
    GCP_PROJECT_ID: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Logging / Observability
    # ─────────────────────────────────────────────────────────────────────────
    PYTEST_LOG_TO_FILE: bool = True
    # Directory for Unity LOGGER file output (async tool loop, managers, etc.)
    # When set, logs are written to {UNITY_LOG_DIR}/unity.log
    # Default: None (console only)
    UNITY_LOG_DIR: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # EventBus Publishing
    # ─────────────────────────────────────────────────────────────────────────
    # Controls whether EventBus publishes events (logging to Unify and local
    # subscriptions/callbacks). Disabled by default for local development to
    # reduce noise. Enable in production deployments.
    EVENTBUS_PUBLISHING_ENABLED: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # EventBus Pub/Sub Streaming (Live Actions)
    # ─────────────────────────────────────────────────────────────────────────
    # When enabled, EventBus.publish() also streams ManagerMethod and ToolLoop
    # events to the assistant's GCP Pub/Sub topic with thread="action_event".
    # This enables real-time frontend rendering of the agent's activity tree
    # without polling Orchestra. Requires GCP credentials and a provisioned
    # Pub/Sub topic. Disabled by default; enable in production deployments.
    EVENTBUS_PUBSUB_STREAMING: bool = False

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
    # Terminal Logging
    # ─────────────────────────────────────────────────────────────────────────
    UNITY_TERMINAL_LOG: bool = True
    UNITY_TERMINAL_LOG_LEVEL: str = "INFO"

    # ─────────────────────────────────────────────────────────────────────────
    # Debug Modes (performance overhead, development-only)
    # ─────────────────────────────────────────────────────────────────────────
    UNITY_ASYNCIO_DEBUG: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Test Infrastructure
    # ─────────────────────────────────────────────────────────────────────────
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
    UNITY_READONLY_ASK_GUARD: bool = True
    FIRST_ASK_TOOL_IS_SEARCH: bool = False
    FIRST_MUTATION_TOOL_IS_ASK: bool = False
    DEPLOY_ENV: Literal["production", "staging", "preview"] = "production"
    DEMO_MODE: bool = False
    DEMO_ID: int | None = None  # Demo assistant metadata ID (if DEMO_MODE is True)

    # ─────────────────────────────────────────────────────────────────────────
    # Manager Configuration
    # ─────────────────────────────────────────────────────────────────────────
    # Foundational managers (cannot be disabled, only implementation switched):
    #   - Actor, ContactManager, TranscriptManager, TaskScheduler, ConversationManager
    # Optional managers (can be disabled via ENABLED=False):
    #   - KnowledgeManager, GuidanceManager, SecretManager,
    #     WebSearcher

    # ─────────────────────────────────────────────────────────────────────────
    # Composed Manager Settings
    # ─────────────────────────────────────────────────────────────────────────
    # Each manager owns its settings in its own settings.py file.
    # Access via SETTINGS.contact.IMPL, SETTINGS.transcript.IMPL, etc.
    actor: ActorSettings = Field(default_factory=ActorSettings)
    contact: ContactSettings = Field(default_factory=ContactSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    data: DataSettings = Field(default_factory=DataSettings)
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
    @field_validator(
        "UNITY_TERMINAL_LOG",
        "UNITY_ASYNCIO_DEBUG",
        "DEMO_MODE",
        "EVENTBUS_PUBLISHING_ENABLED",
        "EVENTBUS_PUBSUB_STREAMING",
        "PYTEST_LOG_TO_FILE",
        "UNITY_READONLY_ASK_GUARD",
        "FIRST_ASK_TOOL_IS_SEARCH",
        "FIRST_MUTATION_TOOL_IS_ASK",
        "TEST",
        "UNITY_VALIDATE_LLM_PROVIDERS",
        "UNITY_OTEL",
        mode="before",
    )
    @classmethod
    def parse_bool_fields(cls, v: Any) -> bool:
        return _parse_bool(v)

    @field_validator("DEPLOY_ENV", mode="before")
    @classmethod
    def parse_deploy_env_field(cls, v: Any) -> str:
        return _parse_deploy_env(v)

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def ENV_SUFFIX(self) -> str:
        """Return the environment suffix used in shared resource names."""
        return "" if self.DEPLOY_ENV == "production" else f"-{self.DEPLOY_ENV}"

    def validate_llm_providers(self) -> None:
        """Validate that at least one LLM provider credential is set.

        Raises:
            RuntimeError: If no LLM provider credentials are set.
        """
        if not self.UNITY_VALIDATE_LLM_PROVIDERS:
            return
        available = {
            "OPENAI_API_KEY": self.OPENAI_API_KEY,
            "ANTHROPIC_API_KEY": self.ANTHROPIC_API_KEY,
        }
        if not any(available.values()):
            raise RuntimeError(
                "At least one LLM provider credential is required. "
                "Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY.",
            )


# Singleton instance for production code
SETTINGS = ProductionSettings()
