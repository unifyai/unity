from enum import StrEnum
from pydantic import BaseModel, Field

from .mode import Mode


class MediumInfo(BaseModel):
    """Metadata describing a communication medium."""

    value: str = Field(
        description="The unique string identifier for this medium used in the database",
    )
    description: str = Field(
        description="A natural language description of what this medium represents",
    )
    mode: Mode = Field(
        description="The ConversationManager operational mode for this medium",
    )


class Medium(StrEnum):
    """
    Enumeration of all supported communication mediums.

    Medium serves as the single source of truth for communication channel types.
    Each medium value can be used directly as a conversation thread key.

    Naming convention uses _MESSAGE/_CALL suffixes to disambiguate related
    mediums (e.g., WHATSAPP_MESSAGE vs WHATSAPP_CALL in future).
    """

    UNIFY_MESSAGE = "unify_message"
    UNIFY_MEET = "unify_meet"
    EMAIL = "email"
    SMS_MESSAGE = "sms_message"
    PHONE_CALL = "phone_call"
    API_MESSAGE = "api_message"

    @property
    def info(self) -> MediumInfo:
        """Return the full Pydantic metadata model for this medium."""
        return MEDIUM_REGISTRY[self]

    @property
    def description(self) -> str:
        """Return the natural language description."""
        return self.info.description

    @property
    def mode(self) -> Mode:
        """Return the ConversationManager operational mode for this medium."""
        return self.info.mode


# Registry of metadata for each medium
MEDIUM_REGISTRY: dict[Medium, MediumInfo] = {
    Medium.UNIFY_MESSAGE: MediumInfo(
        value=Medium.UNIFY_MESSAGE,
        description="A text-based chat message sent within the internal Unify assistant interface.",
        mode=Mode.TEXT,
    ),
    Medium.UNIFY_MEET: MediumInfo(
        value=Medium.UNIFY_MEET,
        description="A live voice or video call conducted directly through the Unify platform.",
        mode=Mode.MEET,
    ),
    Medium.EMAIL: MediumInfo(
        value=Medium.EMAIL,
        description="An asynchronous email message.",
        mode=Mode.TEXT,
    ),
    Medium.SMS_MESSAGE: MediumInfo(
        value=Medium.SMS_MESSAGE,
        description="A standard SMS text message sent via cellular network.",
        mode=Mode.TEXT,
    ),
    Medium.PHONE_CALL: MediumInfo(
        value=Medium.PHONE_CALL,
        description="A standard telephonic voice call.",
        mode=Mode.CALL,
    ),
    Medium.API_MESSAGE: MediumInfo(
        value=Medium.API_MESSAGE,
        description="A programmatic message sent via the REST API.",
        mode=Mode.TEXT,
    ),
}

# Export valid values for validation/random selection
VALID_MEDIA: tuple[str, ...] = tuple(m.value for m in Medium)
