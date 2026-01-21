from enum import StrEnum
from pydantic import BaseModel, Field


class Thread(StrEnum):
    """
    Enumeration of conversation thread types.

    Threads group related mediums for conversation tracking. Multiple mediums
    may share the same thread (e.g., PHONE_CALL and UNIFY_MEET both use VOICE).
    """

    VOICE = "voice"
    SMS = "sms"
    EMAIL = "email"
    UNIFY_MESSAGE = "unify_message"


class Mode(StrEnum):
    """
    Enumeration of ConversationManager operational modes.

    Modes determine how the ConversationManager handles communication:
    - CALL: Phone call voice mode
    - UNIFY_MEET: Unify Meet voice/video mode
    - TEXT: Asynchronous text-based communication (SMS, email, Unify messages)
    """

    CALL = "call"
    UNIFY_MEET = "unify_meet"
    TEXT = "text"

    @classmethod
    def voice_modes(cls) -> tuple["Mode", ...]:
        """Return all voice-based modes."""
        return (cls.CALL, cls.UNIFY_MEET)

    @property
    def is_voice(self) -> bool:
        """Check if this mode is a voice mode."""
        return self in self.voice_modes()


class MediumInfo(BaseModel):
    """Metadata describing a communication medium."""

    value: str = Field(
        description="The unique string identifier for this medium used in the database",
    )
    description: str = Field(
        description="A natural language description of what this medium represents",
    )
    thread: "Thread" = Field(
        description="The conversation thread type this medium belongs to",
    )
    mode: "Mode" = Field(
        description="The ConversationManager operational mode for this medium",
    )


class Medium(StrEnum):
    """
    Enumeration of all supported communication mediums.
    Acts as a StrEnum for compatibility but provides rich metadata via .info and .description.
    """

    UNIFY_MESSAGE = "unify_message"
    UNIFY_MEET = "unify_meet"
    EMAIL = "email"
    SMS_MESSAGE = "sms_message"
    PHONE_CALL = "phone_call"

    @property
    def info(self) -> MediumInfo:
        """Return the full Pydantic metadata model for this medium."""
        return MEDIUM_REGISTRY[self]

    @property
    def description(self) -> str:
        """Return the natural language description."""
        return self.info.description

    @property
    def thread(self) -> Thread:
        """Return the conversation thread type for this medium."""
        return self.info.thread

    @property
    def mode(self) -> Mode:
        """Return the ConversationManager operational mode for this medium."""
        return self.info.mode


# Registry of metadata for each medium
MEDIUM_REGISTRY: dict[Medium, MediumInfo] = {
    Medium.UNIFY_MESSAGE: MediumInfo(
        value=Medium.UNIFY_MESSAGE,
        description="A text-based chat message sent within the internal Unify assistant interface.",
        thread=Thread.UNIFY_MESSAGE,
        mode=Mode.TEXT,
    ),
    Medium.UNIFY_MEET: MediumInfo(
        value=Medium.UNIFY_MEET,
        description="A live voice or video call conducted directly through the Unify platform.",
        thread=Thread.VOICE,
        mode=Mode.UNIFY_MEET,
    ),
    Medium.EMAIL: MediumInfo(
        value=Medium.EMAIL,
        description="An asynchronous email message.",
        thread=Thread.EMAIL,
        mode=Mode.TEXT,
    ),
    Medium.SMS_MESSAGE: MediumInfo(
        value=Medium.SMS_MESSAGE,
        description="A standard SMS text message sent via cellular network.",
        thread=Thread.SMS,
        mode=Mode.TEXT,
    ),
    Medium.PHONE_CALL: MediumInfo(
        value=Medium.PHONE_CALL,
        description="A standard telephonic voice call.",
        thread=Thread.VOICE,
        mode=Mode.CALL,
    ),
}

# Export valid values for validation/random selection
VALID_MEDIA: tuple[str, ...] = tuple(m.value for m in Medium)
VALID_THREADS: tuple[str, ...] = tuple(t.value for t in Thread)
VALID_MODES: tuple[str, ...] = tuple(m.value for m in Mode)
