"""Payload model for LLM request/response events."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LLMPayload(BaseModel):
    """Payload for LLM request/response events.

    Published when an LLM call is made through unillm. Events are emitted
    twice per call: once at request time (phase="request") and once when
    the response is received (phase="response").

    For non-streaming calls, the response event includes cache status and
    token usage. For streaming calls, cache_status is None and token counts
    may be unavailable.
    """

    model_config = ConfigDict(extra="allow")

    # Core event metadata
    phase: str = Field(description="Event phase: 'request' or 'response'")
    endpoint: str = Field(description="LLM endpoint (e.g., 'gpt-4o@openai')")
    model: str = Field(description="Model name extracted from endpoint")
    provider: str = Field(description="Provider name extracted from endpoint")
    stream: bool = Field(
        default=False,
        description="Whether this is a streaming request",
    )

    # Cache status (response phase only, non-streaming)
    cache_status: Optional[str] = Field(
        default=None,
        description="'hit', 'miss', or 'error' (response phase, non-streaming only)",
    )

    # Request summary (to avoid storing large message arrays)
    messages_count: int = Field(
        default=0,
        description="Number of messages in the request",
    )
    tools_count: int = Field(
        default=0,
        description="Number of tools provided in the request",
    )

    # Response metadata (response phase only)
    response_model: Optional[str] = Field(
        default=None,
        description="Model name from response (may differ from request)",
    )
    prompt_tokens: Optional[int] = Field(
        default=None,
        description="Number of prompt tokens used",
    )
    completion_tokens: Optional[int] = Field(
        default=None,
        description="Number of completion tokens generated",
    )
    total_tokens: Optional[int] = Field(
        default=None,
        description="Total tokens (prompt + completion)",
    )

    # Error info (if the LLM call failed)
    error: Optional[str] = Field(
        default=None,
        description="Error message if the LLM call failed",
    )

    # Optional content preview for debugging (truncated)
    content_preview: Optional[str] = Field(
        default=None,
        description="Truncated preview of response content (first 200 chars)",
    )
