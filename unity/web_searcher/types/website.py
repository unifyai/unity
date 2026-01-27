from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

UNASSIGNED_WEBSITE_ID = -1


class Website(BaseModel):
    """Fixed schema for storing websites of interest for WebSearcher.

    This schema is stable and intended to be used as the persisted shape in the
    Unify `Websites` table owned by the WebSearcher.
    """

    website_id: int = Field(
        default=UNASSIGNED_WEBSITE_ID,
        ge=UNASSIGNED_WEBSITE_ID,
        description=("Stable integer identifier for the website row (auto-counted)."),
    )

    name: str = Field(
        description=(
            "Human-friendly unique name for the website (e.g., 'The New York Times')."
        ),
        min_length=1,
    )

    host: str = Field(
        description=(
            "Canonical host or site key (e.g., 'nytimes.com' or canonical root URL)."
        ),
    )

    gated: bool = Field(
        description="Whether the website is gated (requires login or paywall).",
    )

    subscribed: bool = Field(
        description="Whether we have an active subscription to this website.",
    )

    credentials: Optional[List[int]] = Field(
        default=None,
        description=(
            "Optional list of foreign keys into Secrets (by secret_id) used for this site."
        ),
    )

    actor_entrypoint: Optional[int] = Field(
        default=None,
        description=(
            "Optional foreign key into Functions (by function_id) used as an Actor entrypoint."
        ),
    )

    notes: str = Field(
        default="",
        description=(
            "Freeform notes on usage, importance, and guidance for when to consult this site."
        ),
    )

    # Semantic embedding for notes (best-effort vector maintained by the manager)
    notes_emb: List[float] = Field(
        default_factory=list,
        description="Vector embedding for notes to support semantic search.",
    )
