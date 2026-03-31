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
        SPEECH_URGENCY_PREEMPT_ENABLED: Enable the concurrent fast-brain urgency
            evaluator for voice mode. When a user speaks while the slow brain is
            mid-run, a sidecar LLM call classifies the utterance as urgent
            (preempt) or not (let the queue proceed). Default False.
        SPEECH_DEDUP_ENABLED: Enable pre-speak deduplication gate in the fast
            brain subprocess during voice calls. Before playing queued slow brain
            speech, a lightweight LLM check compares the proposed speech against
            recent fast brain utterances and suppresses it when the information
            has already been communicated. Default True.
        USER_DESKTOP_CONTROL_ENABLED: Enable prompts that claim the assistant
            can remotely control the user's computer. When False (default),
            prompts clarify that the assistant can only control its own VM and
            the user can optionally view/control the assistant's desktop — not
            the other way around.
        NOTIFICATION_REPLY_CONTEXT_WINDOW: Maximum number of conversation items
            the notification reply evaluator keeps. Smaller than the fast brain
            window because the evaluator only needs recent context to detect
            redundant speech.
        ASSISTANT_SESSION_GROUP: API group for AssistantSession resources.
        ASSISTANT_SESSION_VERSION: API version for AssistantSession resources.
        ASSISTANT_SESSION_PLURAL: Resource plural for AssistantSession resources.
        ASSISTANT_SESSION_PROTOCOL_VERSION: Bootstrap protocol version expected
            from AssistantSession-backed startup state.
    """

    FAST_BRAIN_MODEL: str = "gpt-5-mini@openai"
    FAST_BRAIN_CONTEXT_WINDOW: int = 50
    NOTIFICATION_REPLY_CONTEXT_WINDOW: int = 8
    IMPL: str = "real"
    COMMS_URL: str = Field(default="", validation_alias="UNITY_COMMS_URL")
    ADAPTERS_URL: str = Field(default="", validation_alias="UNITY_ADAPTERS_URL")
    JOB_NAME: str = ""
    CONTACT_ID: str = "1"
    BLACKLIST_CHECKS_ENABLED: bool = False
    SPEECH_URGENCY_PREEMPT_ENABLED: bool = True
    SPEECH_DEDUP_ENABLED: bool = True
    USER_DESKTOP_CONTROL_ENABLED: bool = False
    ASSISTANT_SESSION_GROUP: str = "infra.unify.ai"
    ASSISTANT_SESSION_VERSION: str = "v1alpha1"
    ASSISTANT_SESSION_PLURAL: str = "assistantsessions"
    ASSISTANT_SESSION_PROTOCOL_VERSION: str = "v1"
    ASSIGNMENT_POLL_INTERVAL: float = 0.5

    model_config = SettingsConfigDict(
        env_prefix="UNITY_CONVERSATION_",
        case_sensitive=True,
        extra="ignore",
    )
