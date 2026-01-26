"""
tests/async_tool_loop/conftest.py
======================================

Shared pytest fixtures for async tool loop tests.

Provides model parameterization to ensure all LLM-based tests run against
multiple model families (OpenAI and Vertex AI) for cross-provider compatibility.
"""

from __future__ import annotations

import pytest

# Models to test against for cross-provider compatibility
LLM_MODELS = [
    "gpt-5.2@openai",
    "claude-4.5-opus@anthropic",
]


@pytest.fixture(params=LLM_MODELS)
def model(request) -> str:
    """Parameterized fixture providing LLM model identifiers.

    Tests using this fixture will run once per model in LLM_MODELS.
    """
    return request.param
