"""Business context payload for slot-filling prompt composition.

This module provides a structured way to inject domain-specific context into
generic prompt builders without modifying the core instructions. The payload
defines slots for identity, domain knowledge, and response guidelines that
are assembled into the final prompt in a consistent order.

Design principles:
- Business identity (role_description) appears FIRST in the final prompt
- Generic capabilities come SECOND (tool descriptions, retrieval patterns)
- Domain knowledge (domain_rules, retrieval_hints) comes THIRD
- Response guidelines come LAST (recency effect)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BusinessContextPayload:
    """Structured business context for slot-filling prompt composition.

    This dataclass provides explicit slots that prompt builders use to inject
    domain-specific content into the correct positions within a prompt. New
    clients/domains only need to construct a BusinessContextPayload; the
    generic prompt builders remain unchanged.

    Attributes
    ----------
    role_description : str
        Primary identity statement for the agent. This appears FIRST in the
        final prompt to establish the correct persona (primacy effect).
        Example: "You are an expert data analyst for Midland Heart..."

    domain_rules : str
        Domain-specific knowledge including:
        - Data sources and file paths
        - Schema descriptions and column meanings
        - Join logic and cross-referencing rules
        - Which columns are embedded for semantic search

    response_guidelines : str
        Output format and style requirements:
        - Citation format
        - Confidence scores
        - Human-friendly step descriptions
        - Tone and verbosity

    retrieval_hints : str | None
        Optional domain-specific query patterns:
        - Which tables to query for specific question types
        - Path-first references for this domain
        - Temporal data split strategies (e.g., by month)
    """

    role_description: str
    domain_rules: str
    response_guidelines: str
    retrieval_hints: Optional[str] = None
