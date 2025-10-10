from __future__ import annotations

import asyncio
import pytest

from unity.secret_manager.secret_manager import SecretManager


@pytest.mark.unit
def test_create_secret(secret_manager_context):
    sm = SecretManager()
    out = sm._create_secret(
        name="unify_key",
        value="sk-test-123",
        description="Unify API key",
    )
    assert out["outcome"] == "secret created"

    # Read path must never expose values
    rows = sm._filter_secrets(filter="name == 'unify_key'")
    assert rows and rows[0].name == "unify_key"
    assert rows[0].value == ""


@pytest.mark.unit
def test_update_secret(secret_manager_context):
    sm = SecretManager()
    sm._create_secret(name="db_password", value="abc", description="db pass")
    out = sm._update_secret(name="db_password", value="xyz", description="rotated")
    assert out["outcome"] == "secret updated"

    rows = sm._filter_secrets(filter="name == 'db_password'")
    assert rows and rows[0].name == "db_password" and rows[0].description == "rotated"


@pytest.mark.unit
def test_delete_secret(secret_manager_context):
    sm = SecretManager()
    sm._create_secret(name="temp_token", value="t123")
    out = sm._delete_secret(name="temp_token")
    assert out["outcome"] == "secret deleted"
    rows = sm._filter_secrets(filter="name == 'temp_token'")
    assert len(rows) == 0


@pytest.mark.unit
def test_from_placeholder_public_method(secret_manager_context):
    sm = SecretManager()
    sm._create_secret(name="page_username", value="user123")
    sm._create_secret(name="page_password", value="pass456")

    cmd = "input username as ${page_username} and password as ${page_password}"
    resolved = asyncio.get_event_loop().run_until_complete(sm.from_placeholder(cmd))
    assert "user123" in resolved and "pass456" in resolved


@pytest.mark.unit
def test_to_placeholder_public_method(secret_manager_context):
    sm = SecretManager()
    sm._create_secret(name="api_key", value="sk-live-abc", description="api")
    sm._create_secret(name="db_password", value="p@ssw0rd", description="db")

    text = "Use sk-live-abc to authenticate and p@ssw0rd for the database."
    out = asyncio.get_event_loop().run_until_complete(sm.to_placeholder(text))
    assert "${api_key}" in out and "${db_password}" in out


@pytest.mark.unit
def test_secret_manager_clear(secret_manager_context):
    sm = SecretManager()

    # Seed a couple of secrets
    sm._create_secret(name="alpha", value="val_a", description="A secret")
    sm._create_secret(name="beta", value="val_b", description="B secret")

    # Sanity: present before clear
    before = sm._list_secret_keys()
    assert "alpha" in before and "beta" in before

    # Execute clear
    sm.clear()

    # After clear: storage should be re-provisioned and empty
    after = sm._list_secret_keys()
    assert isinstance(after, list)
    assert len(after) == 0
