from pydantic import BaseModel
from typing import Optional


class Contact(BaseModel):
    contact_id: int
    first_name: Optional[str]
    surname: Optional[str]
    email_address: Optional[str]
    phone_number: Optional[str]
    whatsapp_number: Optional[str]
