"""
tests/async_tool_loop/conftest.py
======================================

Shared pytest fixtures for async tool loop tests.

Provides model parameterization to ensure all LLM-based tests run against
multiple model families (OpenAI and Anthropic) for cross-provider compatibility.
Each config bundles the model identifier with its production-matching kwargs
(reasoning_effort, service_tier, etc.) so tests exercise the exact settings
used when that model is hot-swapped into production.
"""

from __future__ import annotations

import pytest

# Full client configs to test against — mirrors production hot-swap settings.
LLM_CONFIGS = [
    pytest.param(
        {
            "model": "gpt-5.2@openai",
            "reasoning_effort": "high",
            "service_tier": "priority",
        },
        id="gpt-5.2",
    ),
    pytest.param(
        {
            "model": "claude-4.6-opus@anthropic",
            "reasoning_effort": "low",
            "service_tier": "priority",
        },
        id="claude-4.6-opus",
    ),
]


@pytest.fixture(params=LLM_CONFIGS)
def llm_config(request) -> dict[str, str]:
    """Parameterized fixture providing full LLM client configurations.

    Each config bundles the model identifier with all provider-specific
    kwargs (reasoning_effort, service_tier, etc.) so that tests exercise
    the exact settings used in production for each model family.

    Tests using this fixture will run once per config in LLM_CONFIGS.
    """
    return request.param
