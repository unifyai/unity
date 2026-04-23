import logging
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
from typing import Literal, Optional, ClassVar

_log = logging.getLogger(__name__)

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


class ContactDetailsWhatsApp(ContactDetailsBase):
    """Contact details with WhatsApp number for WhatsApp communication."""

    whatsapp_number: Optional[str] = Field(
        default=None,
        description="WhatsApp number with optional leading +, then digits only",
        pattern=r"^\+?[0-9]+$",
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
        "whatsapp_number": "wa",
        "discord_id": "disc",
        "bio": "bio",
        "job_title": "jt",
        "rolling_summary": "rs",
        "timezone": "tz",
        "is_system": "sys",
        "assistant_id": "aid",
        "authoring_assistant_id": "auth_aid",
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
        json_schema_extra={"unique": True},
    )
    phone_number: Optional[str] = Field(
        default=None,
        description="Optional leading +, then digits only",
        pattern=r"^\+?[0-9]+$",
        json_schema_extra={"unique": True},
    )
    whatsapp_number: Optional[str] = Field(
        default=None,
        description="WhatsApp number — optional leading +, then digits only",
        pattern=r"^\+?[0-9]+$",
        json_schema_extra={"unique": True},
    )
    discord_id: Optional[str] = Field(
        default=None,
        description="Discord user snowflake ID (digits only)",
        pattern=r"^[0-9]+$",
        json_schema_extra={"unique": True},
    )
    bio: Optional[str] = Field(
        default=None,
        description="Concise biographic profile of the contact (role, background, why they matter).",
    )
    job_title: Optional[str] = Field(
        default=None,
        description="Free-text job title / specialization (e.g. 'Growth marketing').",
    )
    rolling_summary: Optional[str] = Field(
        default=None,
        description="Short rolling conversation summary and current objectives with this contact.",
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

    assistant_id: Optional[int] = Field(
        default=None,
        description=(
            "Populated when this contact represents a Hive-member body "
            "(the assistant the contact *is*); NULL for every non-body contact."
        ),
    )
    authoring_assistant_id: Optional[int] = Field(
        default=None,
        description=(
            "Assistant id that authored this row. Stamped at create time by "
            "the authoring body and never rewritten by subsequent updates or "
            "merges (the survivor keeps its original author)."
        ),
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
        "whatsapp_number",
        "discord_id",
        "bio",
        "job_title",
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
            return v_str
        except Exception:
            _log.warning("Unrecognised timezone '%s', falling back to None", v_str)
            return None

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


# ─────────────────────────────────────────────────────────────────────────────
# Per-body ContactMembership overlay
# ─────────────────────────────────────────────────────────────────────────────


class ContactMembership(BaseModel):
    """A body-local view onto a shared :class:`Contact` row.

    Each Hive member (or solo body) maintains its own
    ``{user}/{assistant}/ContactMembership`` overlay. One row per contact
    expresses *this body's* relationship to the shared contact and its
    per-body response policy; the shared row (``Contacts``) remains the
    single source of truth for identity, name, email, etc.

    The overlay is the home of fields that are legitimately body-local:
    how I relate to this contact (``relationship``) and whether I, as
    this particular body, should proactively respond to them
    (``should_respond`` / ``response_policy``). Two bodies sharing the
    same contact can hold different relationships and different response
    policies without colliding on the shared identity row.
    """

    contact_id: int = Field(
        description="FK to the shared Contact row this overlay describes.",
        ge=0,
    )
    relationship: Literal["self", "boss", "coworker", "other"] = Field(
        default="other",
        description=(
            "This body's relationship to the shared contact. "
            "'self' for the contact that IS this body, 'boss' for the "
            "body's primary human principal, 'coworker' for another body "
            "inside the same Hive, 'other' for everyone else."
        ),
    )
    should_respond: bool = Field(
        default=True,
        description=(
            "Whether *this* body should proactively respond to inbound "
            "conversational traffic from the contact. Gates the "
            "conversational response loop only; scheduled and triggered "
            "tasks fire regardless."
        ),
    )
    response_policy: Optional[str] = Field(
        default=None,
        description=(
            "Body-local response policy for this contact. When None, "
            "callers should fall back to the manager-level default."
        ),
    )
    can_edit: bool = Field(
        default=True,
        description=(
            "Whether this body is allowed to mutate the shared Contact "
            "row. Reserved for future permissions work; defaults to True "
            "so every body can edit by default."
        ),
    )

    model_config = {"extra": "allow"}
