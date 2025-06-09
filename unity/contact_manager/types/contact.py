from pydantic import BaseModel, Field
from typing import Optional


class Contact(BaseModel):
    contact_id: int = Field(
        description="Unique identifier for the contact",
    )
    first_name: Optional[str] = Field(
        description="Contact's first name - must start with a capital letter",
        pattern="^[A-Z][a-zA-Z .-]*$",
    )
    surname: Optional[str] = Field(
        description="Contact's surname/family name - must start with a capital letter",
        pattern="^[A-Z][a-zA-Z .-]*$",
    )
    email_address: Optional[str] = Field(
        description="Contact's email address - must contain exactly one @ symbol with characters on either side",
        pattern="^[^@]+@[^@]+$",
    )
    phone_number: Optional[str] = Field(
        description="Contact's phone number - can optionally start with '+' (only if *explicitly* mentioned by the user), but must otherwise contain only digits",
        pattern="^\\+?[0-9]+$",
    )
    whatsapp_number: Optional[str] = Field(
        description="Contact's WhatsApp number - can optionally start with '+' (only if *explicitly* mentioned by the user), but must otherwise contain only digits",
        pattern="^\\+?[0-9]+$",
    )
    description: Optional[str] = Field(
        description="Free-form notes about the contact.",
    )
    model_config = {
        "extra": "allow",
    }
