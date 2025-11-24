from enum import StrEnum
from pydantic import BaseModel, Field


class MediumInfo(BaseModel):
    """Metadata describing a communication medium."""

    value: str = Field(
        description="The unique string identifier for this medium used in the database",
    )
    description: str = Field(
        description="A natural language description of what this medium represents",
    )


class Medium(StrEnum):
    """
    Enumeration of all supported communication mediums.
    Acts as a StrEnum for compatibility but provides rich metadata via .info and .description.
    """

    UNIFY_MESSAGE = "unify_message"
    UNIFY_CALL = "unify_call"
    UNIFY_MEET = "unify_meet"
    EMAIL = "email"
    SMS_MESSAGE = "sms_message"
    PHONE_CALL = "phone_call"
    WHATSAPP_MSG = "whatsapp_message"
    WHATSAPP_CALL = "whatsapp_call"
    GOOGLE_MEET = "google_meet"

    @property
    def info(self) -> MediumInfo:
        """Return the full Pydantic metadata model for this medium."""
        return _MEDIUM_REGISTRY[self]

    @property
    def description(self) -> str:
        """Return the natural language description."""
        return self.info.description


# Registry of metadata for each medium
_MEDIUM_REGISTRY: dict[Medium, MediumInfo] = {
    Medium.UNIFY_MESSAGE: MediumInfo(
        value=Medium.UNIFY_MESSAGE,
        description="A text-based chat message sent within the internal Unify assistant interface.",
    ),
    Medium.UNIFY_CALL: MediumInfo(
        value=Medium.UNIFY_CALL,
        description="A live voice call conducted directly through the Unify platform.",
    ),
    Medium.UNIFY_MEET: MediumInfo(
        value=Medium.UNIFY_MEET,
        description="A live video meeting hosted on the Unify platform.",
    ),
    Medium.EMAIL: MediumInfo(
        value=Medium.EMAIL,
        description="An asynchronous email message.",
    ),
    Medium.SMS_MESSAGE: MediumInfo(
        value=Medium.SMS_MESSAGE,
        description="A standard SMS text message sent via cellular network.",
    ),
    Medium.PHONE_CALL: MediumInfo(
        value=Medium.PHONE_CALL,
        description="A standard telephonic voice call.",
    ),
    Medium.WHATSAPP_MSG: MediumInfo(
        value=Medium.WHATSAPP_MSG,
        description="A text message sent via WhatsApp.",
    ),
    Medium.WHATSAPP_CALL: MediumInfo(
        value=Medium.WHATSAPP_CALL,
        description="A voice or video call made via WhatsApp.",
    ),
    Medium.GOOGLE_MEET: MediumInfo(
        value=Medium.GOOGLE_MEET,
        description="A video conference meeting hosted on Google Meet.",
    ),
}

# Export valid values for validation/random selection
VALID_MEDIA: tuple[str, ...] = tuple(m.value for m in Medium)
