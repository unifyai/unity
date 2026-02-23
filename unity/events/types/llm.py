"""Payload model for LLM completion events."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class LLMPayload(BaseModel):
    """Payload for LLM completion events.

    Published when an LLM call completes through unillm. A single event is
    emitted per LLM call, containing the full request and response data.

    This mirrors what gets logged to logs/unillm/ files, providing complete
    request/response data without derived metrics.
    """

    model_config = ConfigDict(extra="allow")

    # Full request dict sent to the LLM
    request: dict[str, Any] = Field(
        description="Full request kwargs sent to LLM (model, messages, tools, etc.)",
    )

    # Full serialized response from the LLM (None for streaming or errors)
    response: Optional[dict[str, Any]] = Field(
        default=None,
        description="Full serialized response from LLM (None for streaming/errors)",
    )

    # Cost information (only for cache misses)
    provider_cost: Optional[float] = Field(
        default=None,
        description="Raw cost charged by the LLM provider (USD)",
    )
    billed_cost: Optional[float] = Field(
        default=None,
        description="Cost charged to the user (provider_cost × margin, USD)",
    )
