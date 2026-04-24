from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Exchange(BaseModel):
    """One row per conversation thread.

    ``counterparty_contact_id`` pins the non-body side of the
    conversation at open time — for body-to-body exchanges inside a
    Hive it references the Hive-mate's self-contact row, so the
    dedup query ``(medium, counterparty_contact_id)`` collapses two
    bodies' views of the same conversation onto one row. For external
    contacts it references that contact, and for legacy rows (no
    counterparty known at open time) it stays ``None``.

    ``authoring_assistant_id`` is the body that *opened* the exchange
    and is preserved across subsequent messages from other bodies —
    later writers never rewrite the stamp.
    """

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
    counterparty_contact_id: Optional[int] = Field(
        default=None,
        description=(
            "contact_id of the non-body participant at exchange-open time. "
            "For body-to-body exchanges, the Hive-mate's self-contact id; "
            "for external conversations, the external contact's id. Used as "
            "the dedup index together with ``medium`` so two Hive-mate bodies "
            "converse on one shared exchange row."
        ),
    )
    authoring_assistant_id: Optional[int] = Field(
        default=None,
        description=(
            "Assistant id of the body that opened this exchange. Stamped "
            "once at open time and preserved across subsequent messages "
            "from other bodies — never rewritten."
        ),
    )
