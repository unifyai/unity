from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field

UNASSIGNED_SECRET_ID = -1


class Secret(BaseModel):
    """Fixed schema for storing secrets.

    The schema is intentionally immutable: columns cannot be added or removed.
    """

    secret_id: int = Field(
        default=UNASSIGNED_SECRET_ID,
        ge=UNASSIGNED_SECRET_ID,
        description=(
            "Stable integer identifier for the secret row (auto-counted). Safe to surface to LLMs."
        ),
    )
    name: str = Field(
        description="Unique identifier for the secret. Used in placeholders like ${name}.",
    )
    value: str = Field(description="The raw secret value. Never expose to LLMs.")
    description: str = Field(
        default="",
        description="Human-readable description of the secret's purpose.",
    )
    description_emb: List[float] = Field(
        default_factory=list,
        description="Vector embedding of the description for semantic search.",
    )
