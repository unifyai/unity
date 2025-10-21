from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    model_serializer,
    SerializationInfo,
    SerializerFunctionWrapHandler,
)
from typing import Optional, ClassVar

UNICODE_NAME_RE = r"^[^\W\d_](?:[^\W\d_]|[ .'-])*$"  # ← one reusable constant

UNASSIGNED = -1


class Contact(BaseModel):
    # Central, single source of truth for shorthand aliases (full → shorthand)
    SHORTHAND_MAP: ClassVar[dict[str, str]] = {
        "contact_id": "cid",
        "first_name": "fn",
        "surname": "sn",
        "email_address": "email",
        "phone_number": "phone",
        "whatsapp_number": "whatsapp",
        "bio": "bio",
        "rolling_summary": "rs",
        "respond_to": "resp",
        "response_policy": "policy",
    }

    contact_id: int = Field(
        default=UNASSIGNED,
        description="Unique identifier for the contact",
        ge=UNASSIGNED,
    )
    first_name: Optional[str] = Field(
        default=None,
        description="Contact's first name – letters (any script) plus . ' - and space",
        pattern=UNICODE_NAME_RE,
    )
    surname: Optional[str] = Field(
        default=None,
        description="Contact's surname – letters (any script) plus . ' - and space",
        pattern=UNICODE_NAME_RE,
    )
    email_address: Optional[str] = Field(
        default=None,
        description="Must contain exactly one @ with characters on either side",
        pattern=r"^[^@]+@[^@]+$",
    )
    phone_number: Optional[str] = Field(
        default=None,
        description="Optional leading +, then digits only",
        pattern=r"^\+?[0-9]+$",
    )
    whatsapp_number: Optional[str] = Field(
        default=None,
        description="Optional leading +, then digits only",
        pattern=r"^\+?[0-9]+$",
    )
    bio: Optional[str] = Field(
        default=None,
        description="Concise biographic profile of the contact (role, background, why they matter).",
    )
    rolling_summary: Optional[str] = Field(
        default=None,
        description="Short rolling conversation summary and current objectives with this contact.",
    )
    respond_to: bool = Field(
        default=False,
        description="Whether the assistant should respond to inbound messages or calls from this contact.",
    )
    response_policy: Optional[str] = Field(
        default=None,
        description="Policy dictating how the assistant should respond to this contact.",
    )

    @model_validator(mode="before")
    @classmethod
    def _inject_sentinel(cls, data: dict) -> dict:
        data.setdefault("contact_id", UNASSIGNED)
        return data

    def to_post_json(self) -> dict:
        exclude = {"contact_id"} if self.contact_id == UNASSIGNED else {}
        return self.model_dump(mode="json", exclude=exclude)

    # Shorthand helpers (parity with Message model)
    @classmethod
    def shorthand_map(cls) -> dict[str, str]:
        return dict(cls.SHORTHAND_MAP)

    @classmethod
    def shorthand_inverse_map(cls) -> dict[str, str]:
        return {v: k for k, v in cls.SHORTHAND_MAP.items()}

    @field_validator(
        "first_name",
        "surname",
        "email_address",
        "phone_number",
        "whatsapp_number",
        "bio",
        "rolling_summary",
        mode="before",
    )
    @classmethod
    def _empty_to_none(cls, v):
        """
        Treat blank or whitespace-only strings as missing (None)
        so they skip regex validation entirely.
        """
        if v is not None and isinstance(v, str) and v.strip() == "":
            return None
        return v

    model_config = {"extra": "allow"}

    # Only affect JSON-mode serialisation: prune empty fields and/or alias keys
    # when explicitly requested via context (parity with Message model)
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
            alias_map = type(self).SHORTHAND_MAP
            try:
                out = {alias_map.get(k, k): v for k, v in out.items()}
            except Exception:
                out = out

        return out
