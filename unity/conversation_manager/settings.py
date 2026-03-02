"""
ConversationManager-specific settings.

These settings are composed into the global ProductionSettings.
Environment variables use the prefix UNITY_CONVERSATION_.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConversationSettings(BaseSettings):
    """ConversationManager settings.

    Attributes:
        IMPL: Implementation type - "real" or "simulated".
        COMMS_URL: URL for the communications service (reads from UNITY_COMMS_URL).
        JOB_NAME: Job name for the ConversationManager session.
        CONTACT_ID: Default contact ID for simulated ConversationManager.
        FAST_BRAIN_MODEL: LLM model for the voice fast brain (TTS mode).
            Override via UNITY_CONVERSATION_FAST_BRAIN_MODEL.
        FAST_BRAIN_CONTEXT_WINDOW: Maximum number of conversation items
            (utterances, notifications, etc.) the fast brain keeps in its
            rolling context window. Also used as the limit when hydrating
            historical events at call start.
        BLACKLIST_CHECKS_ENABLED: Enable blacklist filtering and unknown contact
            creation for inbound messages. Default False for fast inbound path.
            When False, no BlackListManager or ContactManager initialization
            occurs during message handling.
    """

    FAST_BRAIN_MODEL: str = "gpt-5-mini@openai"
    FAST_BRAIN_CONTEXT_WINDOW: int = 50
    FAST_BRAIN_STRUCTURED_NOTIFICATION_REPLY: bool = False
    IMPL: str = "real"
    COMMS_URL: str = Field(default="", validation_alias="UNITY_COMMS_URL")
    ADAPTERS_URL: str = Field(default="", validation_alias="UNITY_ADAPTERS_URL")
    JOB_NAME: str = ""
    CONTACT_ID: str = "1"
    BLACKLIST_CHECKS_ENABLED: bool = False

    model_config = SettingsConfigDict(
        env_prefix="UNITY_CONVERSATION_",
        case_sensitive=True,
        extra="ignore",
    )
