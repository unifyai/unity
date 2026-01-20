import zoneinfo
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


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight contact detail models for outbound communication tools
# ─────────────────────────────────────────────────────────────────────────────


class ContactDetailsBase(BaseModel):
    """Minimal contact identity for lookup or creation."""

    first_name: Optional[str] = Field(
        default=None,
        description="Contact's first name",
        pattern=UNICODE_NAME_RE,
    )
    surname: Optional[str] = Field(
        default=None,
        description="Contact's surname",
        pattern=UNICODE_NAME_RE,
    )


class ContactDetailsPhone(ContactDetailsBase):
    """Contact details with phone number for SMS or calls."""

    phone_number: Optional[str] = Field(
        default=None,
        description="Phone number with optional leading +, then digits only",
        pattern=r"^\+?[0-9]+$",
    )


class ContactDetailsEmail(ContactDetailsBase):
    """Contact details with email address for email communication."""

    email_address: Optional[str] = Field(
        default=None,
        description="Email address (must contain exactly one @)",
        pattern=r"^[^@]+@[^@]+$",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Contact model
# ─────────────────────────────────────────────────────────────────────────────


class Contact(BaseModel):
    # Central, single source of truth for shorthand aliases (full → shorthand)
    SHORTHAND_MAP: ClassVar[dict[str, str]] = {
        "contact_id": "cid",
        "first_name": "fn",
        "surname": "sn",
        "email_address": "email",
        "phone_number": "phone",
        "bio": "bio",
        "rolling_summary": "rs",
        "should_respond": "resp",
        "response_policy": "policy",
        "timezone": "tz",
        "is_system": "sys",
    }

    # Dynamic aliases for custom columns (full → shorthand); managers can
    # register into this mapping at runtime. Kept on the class to avoid
    # per‑instance plumbing.
    SHORTHAND_MAP_DYNAMIC: ClassVar[dict[str, str]] = {}

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
    bio: Optional[str] = Field(
        default=None,
        description="Concise biographic profile of the contact (role, background, why they matter).",
    )
    rolling_summary: Optional[str] = Field(
        default=None,
        description="Short rolling conversation summary and current objectives with this contact.",
    )
    should_respond: bool = Field(
        default=True,
        description="Whether the assistant should respond to inbound messages or calls from this contact.",
    )
    response_policy: Optional[str] = Field(
        default=None,
        description="Policy dictating how the assistant should respond to this contact.",
    )

    # IANA timezone identifier (e.g. "America/New_York", "Europe/London")
    timezone: Optional[str] = Field(
        default=None,
        description="IANA Timezone identifier (e.g., 'America/New_York', 'Europe/London').",
    )

    is_system: bool = Field(
        default=False,
        description="System contact (assistant, user, or org member). Cannot be deleted.",
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
        base = dict(cls.SHORTHAND_MAP)
        try:
            dyn = dict(getattr(cls, "SHORTHAND_MAP_DYNAMIC", {}) or {})
            for k, v in dyn.items():
                if k not in base:
                    base[k] = v
        except Exception:
            pass
        return base

    @classmethod
    def shorthand_inverse_map(cls) -> dict[str, str]:
        fwd = cls.shorthand_map()
        return {v: k for k, v in fwd.items()}

    @field_validator(
        "first_name",
        "surname",
        "email_address",
        "phone_number",
        "bio",
        "rolling_summary",
        "timezone",
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

    @field_validator("timezone", mode="before")
    @classmethod
    def _validate_timezone(cls, v):
        if v is None:
            return None
        v_str = str(v).strip()
        if not v_str:
            return None
        try:
            zoneinfo.ZoneInfo(v_str)
        except Exception:
            raise ValueError(
                f"Invalid timezone identifier '{v}'. Please use a valid IANA timezone (e.g., 'America/New_York').",
            )
        return v_str

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
            alias_map = type(self).shorthand_map()
            try:
                out = {alias_map.get(k, k): v for k, v in out.items()}
            except Exception:
                out = out

        return out

    # ------------------------- dynamic alias helpers -------------------------
    @classmethod
    def derive_unique_alias(cls, column_name: str) -> str:
        import re as _re

        parts = [p for p in str(column_name).split("_") if p]
        base = "".join(p[:2] for p in parts) or str(column_name)[:3]
        base = _re.sub(r"[^a-z0-9_]", "", base.lower())
        if not base or not _re.match(r"^[a-z]", base):
            base = ("c_" + base) if base else "c"
        used = set(cls.shorthand_map().values())
        cand = base
        idx = 1
        while cand in used:
            cand = f"{base}{idx}"
            idx += 1
        return cand

    @classmethod
    def register_alias(cls, column_name: str, shorthand: Optional[str] = None) -> str:
        import re as _re

        if shorthand is None:
            shorthand = cls.derive_unique_alias(column_name)
        if not _re.fullmatch(r"[a-z][a-z0-9_]*", shorthand):
            raise ValueError(
                "shorthand must be snake_case: start with a letter, then letters/digits/underscores",
            )
        fwd = cls.shorthand_map()
        if shorthand in set(fwd.values()):
            raise ValueError(
                f"shorthand '{shorthand}' already exists. Please choose a different alias.",
            )
        try:
            cls.SHORTHAND_MAP_DYNAMIC[column_name] = shorthand
        except Exception:
            pass
        return shorthand
