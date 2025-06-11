from pydantic import BaseModel, Field
from typing import Optional

UNICODE_NAME_RE = r"^[^\W\d_](?:[^\W\d_]|[ .'-])*$"  # ← one reusable constant


class Contact(BaseModel):
    contact_id: int = Field(description="Unique identifier for the contact")

    first_name: Optional[str] = Field(
        description="Contact's first name – letters (any script) plus . ' - and space",
        pattern=UNICODE_NAME_RE,
    )
    surname: Optional[str] = Field(
        description="Contact's surname – letters (any script) plus . ' - and space",
        pattern=UNICODE_NAME_RE,
    )

    email_address: Optional[str] = Field(
        description="Must contain exactly one @ with characters on either side",
        pattern=r"^[^@]+@[^@]+$",
    )
    phone_number: Optional[str] = Field(
        description="Optional leading +, then digits only",
        pattern=r"^\+?[0-9]+$",
    )
    whatsapp_number: Optional[str] = Field(
        description="Optional leading +, then digits only",
        pattern=r"^\+?[0-9]+$",
    )

    description: Optional[str] = Field(description="Free-form notes about the contact.")

    model_config = {"extra": "allow"}
