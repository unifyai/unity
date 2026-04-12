"""Business context payload for slot-filling prompt composition.

This module provides a structured way to inject domain-specific context into
generic prompt builders without modifying the core instructions. The payload
defines slots for identity, domain knowledge, and response guidelines that
are assembled into the final prompt in a consistent order.

Design principles:
- Business identity (role_description) appears FIRST in the final prompt
- Generic capabilities come SECOND (tool descriptions, retrieval patterns)
- Domain knowledge (domain_rules, data_overview, retrieval_hints) comes THIRD
- Response guidelines come LAST (recency effect)

IMPORTANT: Domain harnesses should NOT contain tool names, tool syntax, or
low-level usage examples. Those details belong in tool docstrings. The
BusinessContextPayload is purely business-level context.
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

    IMPORTANT: All fields should be purely business-level. Do NOT include:
    - Tool names or tool syntax
    - Low-level column names or JSON schemas
    - Sample rows or raw data examples
    Instead, use natural-language descriptions of business concepts and relationships.

    Attributes
    ----------
    role_description : str
        Primary identity statement for the agent. This appears FIRST in the
        final prompt to establish the correct persona (primacy effect).
        Example: "You are an expert data analyst for Acme Housing..."

    domain_rules : str
        Domain-specific knowledge in natural language:
        - What datasets exist and their business purpose
        - How entities relate to each other conceptually
        - Cross-referencing logic between datasets
        Do NOT include raw schemas, column lists, or sample data.

    response_guidelines : str
        Output format and style requirements:
        - Citation format
        - Confidence scores
        - Human-friendly step descriptions
        - Tone and verbosity

    retrieval_hints : str | None
        Optional domain-specific query patterns:
        - Which logical datasets to query for specific question types
        - Temporal data organization (e.g., split by month)
        Keep this high-level; tool-specific syntax belongs in docstrings.

    data_overview : str | None
        Optional natural-language description of available datasets:
        - Table names and their business purpose
        - Key fields conceptually (not exhaustive column lists)
        - Which columns support semantic search
        This replaces raw JSON schema dumps in prompts.
    """

    role_description: str
    domain_rules: str
    response_guidelines: str
    retrieval_hints: Optional[str] = None
    data_overview: Optional[str] = None
