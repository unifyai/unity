from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ActorConfig(BaseModel):
    """Configuration for the top-level CodeActActor.

    All fields are Optional.  ``None`` means "not configured — use the
    constructor default".  Partial overrides are natural (e.g. set only
    ``can_compose=False`` and leave everything else at defaults).

    Defined in code under ``unity/customization/clients/`` per client,
    resolved at Actor construction time via the cascading
    org -> user -> assistant merge.
    """

    can_compose: Optional[bool] = Field(
        default=None,
        description="Whether the LLM can write and execute arbitrary code.",
    )
    can_store: Optional[bool] = Field(
        default=None,
        description="Whether a post-completion review loop stores reusable functions/guidance.",
    )
    timeout: Optional[float] = Field(
        default=None,
        description="Maximum seconds for the actor to complete.",
        gt=0,
    )
    model: Optional[str] = Field(
        default=None,
        description="LLM model identifier (e.g. 'claude-4.5-opus@anthropic').",
    )
    prompt_caching: Optional[List[str]] = Field(
        default=None,
        description=(
            "Cache targets for Anthropic prompt caching. "
            "Valid values: 'tools', 'system', 'messages'."
        ),
    )
    guidelines: Optional[str] = Field(
        default=None,
        description="Persistent behavioral guidelines applied to every act() invocation.",
    )
    url_mappings: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "URL origin mappings for demo mode. Keys are original origins "
            "(e.g. 'https://www.zoho.com'), values are local replacements "
            "(e.g. 'http://localhost:4001'). Applied via Playwright context.route()."
        ),
    )

    def to_post_json(self) -> dict:
        """Return a JSON-safe dict with only the explicitly set (non-None) fields."""
        return {k: v for k, v in self.model_dump(mode="json").items() if v is not None}
