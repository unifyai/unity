from enum import StrEnum
from pydantic import BaseModel, Field, model_validator
from datetime import datetime

UNASSIGNED = -1


class Medium(StrEnum):
    SMS_MESSAGE = "sms_message"
    EMAIL = "email"
    WHATSAPP_MSG = "whatsapp_message"
    PHONE_CALL = "phone_call"
    WHATSAPP_CALL = "whatsapp_call"


class Message(BaseModel):
    message_id: int = Field(description="Unique identifier for the message", ge=-1)
    medium: Medium = Field(
        description="The communication channel used for this message",
    )
    sender_id: int = Field(description="ID of the contact who sent the message")
    receiver_ids: list[int] = Field(
        description="IDs of the contact(s) who received the message.",
        min_length=1,
    )
    timestamp: datetime = Field(
        description="When the message was sent/received in ISO-8601 format",
    )
    content: str = Field(description="The actual text content of the message")
    exchange_id: int | None = Field(
        default=None,
        description="ID of the conversation thread this message belongs to",
    )

    @model_validator(mode="before")
    @classmethod
    def _inject_sentinel(cls, data: dict) -> dict:
        """Pre-processing hook to

        1. Ensure *message_id* has the **UNASSIGNED** sentinel when omitted so
           downstream code can rely on its presence.
        2. Maintain **backwards-compatibility** by transparently upgrading the
           legacy ``receiver_id`` → ``receiver_ids`` (wrapping the single id in
           a one-element list).
        """

        # Guarantee sentinel for id ------------------------------------------------
        data.setdefault("message_id", UNASSIGNED)

        # Upgrade old field name ----------------------------------------------------
        if "receiver_id" in data and "receiver_ids" not in data:
            rid = data.pop("receiver_id")
            # treat "None" as an omitted receiver list (rare / test data)
            data["receiver_ids"] = [rid] if rid is not None else []

        return data

    # --------------------------------------------------------------------- #
    #  Backwards-compat attribute                                           #
    # --------------------------------------------------------------------- #
    @property
    def receiver_id(self) -> int:
        """Convenience alias returning the *first* receiver.

        This exists solely for **backwards-compatibility** so that existing
        code/tests referencing ``.receiver_id`` keep working without
        modification.  When there are multiple receivers the first entry is
        returned (historically there was only ever one).
        """

        return self.receiver_ids[0] if self.receiver_ids else UNASSIGNED

    # Don’t serialise the sentinel value when POSTing
    def to_post_json(self) -> dict:
        """Dump payload for POST; omit the dummy id."""
        exclude = {"message_id"} if self.message_id == UNASSIGNED else {}

        payload = self.model_dump(mode="json", exclude=exclude)

        # Emit the deprecated singular key when there's exactly one receiver
        # so tests that filter on it (e.g. "receiver_id == 7") continue to
        # work until they are updated.
        if len(self.receiver_ids) == 1:
            payload["receiver_id"] = self.receiver_ids[0]

        return payload


VALID_MEDIA: tuple[str, ...] = tuple(m.value for m in Medium)
