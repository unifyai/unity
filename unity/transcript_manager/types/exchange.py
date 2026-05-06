from __future__ import annotations

from typing import Dict, Any
from pydantic import Field

from unity.common.authorship import AuthoredRow


class Exchange(AuthoredRow):
    exchange_id: int = Field(
        description="Unique identifier for the exchange/thread",
        ge=0,
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary exchange-level metadata (e.g., URLs, external refs)",
    )
    medium: str = Field(
        default="",
        description=(
            "Communication medium for the exchange (same semantics as Message.medium)"
        ),
    )
