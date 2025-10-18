from enum import StrEnum
from pydantic import BaseModel, Field, model_validator, model_serializer
from datetime import datetime
from ...image_manager.types import AnnotatedImageRefs

UNASSIGNED = -1


class Medium(StrEnum):
    UNIFY_MESSAGE = "unify_message"
    UNIFY_CALL = "unify_call"
    UNIFY_MEET = "unify_meet"
    EMAIL = "email"
    SMS_MESSAGE = "sms_message"
    PHONE_CALL = "phone_call"
    WHATSAPP_MSG = "whatsapp_message"
    WHATSAPP_CALL = "whatsapp_call"
    GOOGLE_MEET = "google_meet"


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
    exchange_id: int = Field(
        description="ID of the conversation thread this message belongs to",
        ge=-1,
    )
    images: AnnotatedImageRefs = Field(
        default_factory=lambda: AnnotatedImageRefs.model_validate([]),
        description=(
            "List of annotated image references aligned to the text. Each entry must be an AnnotatedImageRef."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _inject_sentinel(cls, data: dict) -> dict:
        """Pre-processing hook to

        Ensure *message_id* has the **UNASSIGNED** sentinel when omitted so
        downstream code can rely on its presence.
        """
        # Guarantee sentinel for id ------------------------------------------------
        data.setdefault("message_id", UNASSIGNED)
        data.setdefault("exchange_id", UNASSIGNED)
        return data

    # Don’t serialise the sentinel value when POSTing
    def to_post_json(self) -> dict:
        """Dump payload for POST; omit the dummy id."""
        exclude = set()
        if self.exchange_id == UNASSIGNED:
            exclude.add("exchange_id")
        if self.message_id == UNASSIGNED:
            exclude.add("message_id")

        payload = self.model_dump(mode="json", exclude=exclude)

        # Ensure structural defaults are present for persistence even when the
        # JSON serializer prunes empty fields. This avoids later reconstruction
        # paths treating missing keys as None (which fails validation).
        payload.setdefault("images", [])

        return payload

    # Only affect JSON-mode serialisation: prune empty fields so tool-loop
    # presentations omit noise like images: [] while the
    # in-memory model remains unchanged and fully populated.
    @model_serializer(mode="wrap")
    def _prune_empty_on_serialize(self, handler):  # type: ignore[no-redef]
        data = handler(self)

        def _is_empty(value):
            try:
                if value is None:
                    return True
                # Treat empty strings as empty; keep False/0 as meaningful
                if isinstance(value, str):
                    return value.strip() == ""
                if isinstance(value, (list, tuple, set, dict)):
                    return len(value) == 0
                return False
            except Exception:
                return False

        def _prune(obj):
            try:
                if isinstance(obj, dict):
                    pruned = {k: _prune(v) for k, v in obj.items()}
                    return {k: v for k, v in pruned.items() if not _is_empty(v)}
                if isinstance(obj, list):
                    pruned_list = [_prune(v) for v in obj]
                    return [v for v in pruned_list if not _is_empty(v)]
                return obj
            except Exception:
                return obj

        try:
            return _prune(data)
        except Exception:
            return data


VALID_MEDIA: tuple[str, ...] = tuple(m.value for m in Medium)
