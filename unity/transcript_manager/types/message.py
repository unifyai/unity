from pydantic import (
    BaseModel,
    Field,
    model_validator,
    model_serializer,
    SerializationInfo,
    SerializerFunctionWrapHandler,
)
from datetime import datetime
from ...image_manager.types import AnnotatedImageRefs
from typing import ClassVar, Optional
from unity.conversation_manager.cm_types import Medium

UNASSIGNED = -1


class Message(BaseModel):
    message_id: int = Field(description="Unique identifier for the message", ge=-1)
    medium: Medium = Field(
        description="The communication channel used for this message",
    )
    sender_id: Optional[int] = Field(
        default=None,
        description="ID of the contact who sent the message (None if contact deleted)",
    )
    receiver_ids: list[Optional[int]] = Field(
        description="IDs of the contact(s) who received the message (None entries if contacts deleted)",
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
    attachments: list[dict] = Field(
        default_factory=list,
        description=(
            "List of file attachments with metadata. Each attachment is a dict with keys: "
            "id, filename, gs_url, content_type, size_bytes."
        ),
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Medium-specific metadata (e.g. email_id for email replies).",
    )

    # Central, single source of truth for shorthand aliases (full → shorthand)
    SHORTHAND_MAP: ClassVar[dict[str, str]] = {
        "message_id": "mid",
        "medium": "med",
        "sender_id": "sid",
        "receiver_ids": "rids",
        "timestamp": "ts",
        "content": "c",
        "exchange_id": "xid",
        "images": "imgs",
        "attachments": "atts",
        "metadata": "meta",
    }

    @classmethod
    def shorthand_map(cls) -> dict[str, str]:
        """Return a copy of the full→shorthand mapping for Message fields."""
        return dict(cls.SHORTHAND_MAP)

    @classmethod
    def shorthand_inverse_map(cls) -> dict[str, str]:
        """Return shorthand→full mapping for Message fields."""
        return {v: k for k, v in cls.SHORTHAND_MAP.items()}

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
        payload.setdefault("attachments", [])

        return payload

    # Only affect JSON-mode serialisation: prune empty fields so tool-loop
    # presentations omit noise like images: [] while the
    # in-memory model remains unchanged and fully populated.
    @model_serializer(mode="wrap")
    def _prune_empty_on_serialize(
        self,
        handler: SerializerFunctionWrapHandler,
        info: SerializationInfo,
    ) -> dict:  # type: ignore[no-redef]
        data = handler(self)

        # Default behaviour: do NOT prune empties; only when explicitly requested via context
        prune = False
        shorthand = False
        try:
            ctx = info.context or {}
            if "prune_empty" in ctx:
                prune = bool(ctx["prune_empty"])  # explicit override
            if "shorthand" in ctx:
                shorthand = bool(ctx["shorthand"])  # explicit aliasing
        except Exception:
            pass

        out = data
        if prune:

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
                out = _prune(out)
            except Exception:
                out = data

        if shorthand and isinstance(out, dict):
            # Minimal, stable aliases for top-level fields
            alias_map = type(self).SHORTHAND_MAP
            try:
                out = {alias_map.get(k, k): v for k, v in out.items()}
            except Exception:
                # best-effort: if mapping fails, keep original keys
                out = out

        return out
