from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional

UNICODE_NAME_RE = r"^[^\W\d_](?:[^\W\d_]|[ .'-])*$"  # ← one reusable constant

UNASSIGNED = -1


class Contact(BaseModel):
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
