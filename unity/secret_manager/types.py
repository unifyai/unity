from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class Secret(BaseModel):
    """Fixed schema for storing secrets.

    The schema is intentionally immutable: columns cannot be added or removed.
    """

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
