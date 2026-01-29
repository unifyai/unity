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
        LOG_LEVEL: Logging level for ConversationManager.
        COMMS_URL: URL for the communications service (reads from UNITY_COMMS_URL).
        JOB_NAME: Job name for the ConversationManager session.
        CONTACT_ID: Default contact ID for simulated ConversationManager.
        BLACKLIST_CHECKS_ENABLED: Enable blacklist filtering and unknown contact
            creation for inbound messages. Default False for fast inbound path.
            When False, no BlackListManager or ContactManager initialization
            occurs during message handling.
    """

    IMPL: str = "real"
    LOG_LEVEL: str = "INFO"
    COMMS_URL: str = Field(default="", validation_alias="UNITY_COMMS_URL")
    JOB_NAME: str = ""
    CONTACT_ID: str = "1"
    BLACKLIST_CHECKS_ENABLED: bool = False

    model_config = SettingsConfigDict(
        env_prefix="UNITY_CONVERSATION_",
        case_sensitive=True,
        extra="ignore",
    )
