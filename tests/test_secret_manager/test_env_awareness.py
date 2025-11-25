from __future__ import annotations

import pytest

from unity.secret_manager.secret_manager import SecretManager


@pytest.mark.unit
def test_list_secret_keys(secret_manager_context):
    sm = SecretManager()
    sm._create_secret(name="unify_key", value="sk-xyz", description="api key")
    sm._create_secret(name="db_password", value="abc123", description="db")
    keys = sm._list_secret_keys()  # type: ignore[attr-defined]
    assert "unify_key" in keys and "db_password" in keys


@pytest.mark.unit
def test_requires_value_on_create(secret_manager_context):
    sm = SecretManager()
    with pytest.raises(AssertionError):
        sm._create_secret(name="empty_value", value="", description="should fail")
