"""Tests for SecretManager response_format parameter."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field
from typing import List, Optional

from unity.secret_manager.secret_manager import SecretManager
from unity.secret_manager.simulated import SimulatedSecretManager
from tests.helpers import _handle_project

# ────────────────────────────────────────────────────────────────────────────
# Response format schemas
# ────────────────────────────────────────────────────────────────────────────


class SecretQueryResult(BaseModel):
    """Structured result from a secret query."""

    secrets_count: int = Field(..., description="Number of secrets found")
    secret_names: List[str] = Field(
        ...,
        description="Names/keys of secrets (not values)",
    )
    has_api_keys: bool = Field(
        ...,
        description="Whether any API key secrets exist",
    )
    summary: str = Field(..., description="Brief natural language summary")


class SecretUpdateResult(BaseModel):
    """Structured result after a secret update operation."""

    success: bool = Field(..., description="Whether the update was successful")
    secret_name: Optional[str] = Field(
        None,
        description="Name of the secret that was created/modified",
    )
    action_taken: str = Field(..., description="Description of what was done")


# ────────────────────────────────────────────────────────────────────────────
# Simulated SecretManager tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_simulated_ask_response_format():
    """Simulated SecretManager.ask should return structured output when response_format is provided."""
    sm = SimulatedSecretManager("Demo secrets vault with API keys and passwords.")

    handle = await sm.ask(
        "What secrets do we have stored and are there any API keys?",
        response_format=SecretQueryResult,
    )
    result = await handle.result()

    # Should be valid JSON conforming to the schema
    parsed = SecretQueryResult.model_validate_json(result)

    assert isinstance(parsed.secrets_count, int)
    assert parsed.secrets_count >= 0
    assert isinstance(parsed.secret_names, list)
    assert isinstance(parsed.has_api_keys, bool)
    assert parsed.summary.strip(), "Summary should be non-empty"


@pytest.mark.asyncio
@_handle_project
async def test_simulated_update_response_format():
    """Simulated SecretManager.update should return structured output when response_format is provided."""
    sm = SimulatedSecretManager("Demo secrets for testing updates.")

    handle = await sm.update(
        "Create a new secret named stripe_api_key with a generated value",
        response_format=SecretUpdateResult,
    )
    result = await handle.result()

    parsed = SecretUpdateResult.model_validate_json(result)

    assert isinstance(parsed.success, bool)
    assert parsed.action_taken.strip(), "Action description should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# Real SecretManager tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_real_ask_response_format(secret_manager_context):
    """Real SecretManager.ask should return structured output when response_format is provided."""
    sm = SecretManager()
    # Seed some secrets
    sm._create_secret(name="api_key", value="sk-123", description="Main API key")
    sm._create_secret(
        name="db_password",
        value="secret123",
        description="Database password",
    )

    handle = await sm.ask(
        "How many secrets are stored and list their names?",
        response_format=SecretQueryResult,
    )
    result = await handle.result()

    parsed = SecretQueryResult.model_validate_json(result)

    assert isinstance(parsed.secrets_count, int)
    assert isinstance(parsed.secret_names, list)
    assert parsed.summary.strip(), "Summary should be non-empty"


@pytest.mark.asyncio
@_handle_project
async def test_real_update_response_format(secret_manager_context):
    """Real SecretManager.update should return structured output when response_format is provided."""
    sm = SecretManager()

    handle = await sm.update(
        "Create a new secret named github_token with value ghp_abc123",
        response_format=SecretUpdateResult,
    )
    result = await handle.result()

    parsed = SecretUpdateResult.model_validate_json(result)

    assert isinstance(parsed.success, bool)
    assert parsed.action_taken.strip(), "Action description should be non-empty"
