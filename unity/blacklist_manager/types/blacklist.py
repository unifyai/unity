from __future__ import annotations

from typing import ClassVar
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    model_serializer,
    SerializationInfo,
    SerializerFunctionWrapHandler,
)

from unity.conversation_manager.cm_types import Medium

UNASSIGNED = -1


class BlackList(BaseModel):
    """
    Minimal schema representing a single blacklist entry.

    Each row corresponds to one blocked contact detail on a specific communication medium.
    """

    # Stable, minimal aliases for compact renderings (kept consistent with other models)
    SHORTHAND_MAP: ClassVar[dict[str, str]] = {
        "blacklist_id": "bid",
        "medium": "med",
        "contact_detail": "cd",
        "reason": "r",
    }

    blacklist_id: int = Field(
        default=UNASSIGNED,
        description="Unique identifier for the blacklist entry",
        ge=UNASSIGNED,
    )
    medium: Medium = Field(
        description="The communication channel this contact detail applies to (e.g., email, sms_message, phone_call).",
    )
    contact_detail: str = Field(
        description="The concrete contact detail to block (e.g., an email address or a phone number).",
    )
    reason: str = Field(
        description="Why this contact detail is blacklisted (short context).",
    )

    @model_validator(mode="before")
    @classmethod
    def _inject_sentinel(cls, data: dict) -> dict:
        data.setdefault("blacklist_id", UNASSIGNED)
        return data

    def to_post_json(self) -> dict:
        """
        Dump payload for POST operations; omit the sentinel id when unassigned.
        """
        exclude = {"blacklist_id"} if self.blacklist_id == UNASSIGNED else set()
        return self.model_dump(mode="json", exclude=exclude)

    @classmethod
    def shorthand_map(cls) -> dict[str, str]:
        """Return a copy of the full→shorthand mapping for BlackList fields."""
        return dict(cls.SHORTHAND_MAP)

    @classmethod
    def shorthand_inverse_map(cls) -> dict[str, str]:
        """Return shorthand→full mapping for BlackList fields."""
        return {v: k for k, v in cls.SHORTHAND_MAP.items()}

    # Only affect JSON-mode serialization: optional pruning and aliasing when requested via context
    @model_serializer(mode="wrap")
    def _prune_empty_on_serialize(
        self,
        handler: SerializerFunctionWrapHandler,
        info: SerializationInfo,
    ) -> dict:  # type: ignore[no-redef]
        data = handler(self)

        prune = False
        shorthand = False
        try:
            ctx = info.context or {}
            if "prune_empty" in ctx:
                prune = bool(ctx["prune_empty"])
            if "shorthand" in ctx:
                shorthand = bool(ctx["shorthand"])
        except Exception:
            pass

        out = data
        if prune:

            def _is_empty(value):
                try:
                    if value is None:
                        return True
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
            alias_map = type(self).SHORTHAND_MAP
            try:
                out = {alias_map.get(k, k): v for k, v in out.items()}
            except Exception:
                out = out

        return out
