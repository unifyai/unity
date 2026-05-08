from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from unity.common import runtime_oauth


def _future_expiry() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()


def _past_expiry() -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()


class _FakeSecretManager:
    def __init__(self, secrets: dict[str, str]) -> None:
        self.secrets = secrets
        self.sync_calls: list[dict[str, Any]] = []
        self.on_sync = None

    def _get_secret_value(self, name: str) -> str | None:
        return self.secrets.get(name)

    def sync_assistant_secrets_if_stale(self, **kwargs: Any) -> bool:
        self.sync_calls.append(kwargs)
        if self.on_sync is not None:
            self.on_sync()
        return True


def _install_secret_manager(monkeypatch, sm: _FakeSecretManager) -> None:
    monkeypatch.setattr(runtime_oauth, "_get_secret_manager", lambda: sm)


def test_get_oauth_access_token_supports_provider_alias(monkeypatch):
    sm = _FakeSecretManager(
        {
            "MICROSOFT_ACCESS_TOKEN": "fresh-ms-token",
            "MICROSOFT_TOKEN_EXPIRES_AT": _future_expiry(),
        },
    )
    _install_secret_manager(monkeypatch, sm)

    assert runtime_oauth.get_oauth_access_token("graph") == "fresh-ms-token"
    assert sm.sync_calls[-1]["force"] is False


def test_get_oauth_access_token_unknown_provider_is_non_secret_error(monkeypatch):
    sm = _FakeSecretManager({})
    _install_secret_manager(monkeypatch, sm)

    with pytest.raises(ValueError, match="Unknown refresh-token OAuth provider"):
        runtime_oauth.get_oauth_access_token("not-a-real-provider")


def test_get_oauth_access_token_missing_token_raises_after_sync(monkeypatch):
    sm = _FakeSecretManager({"MICROSOFT_TOKEN_EXPIRES_AT": _future_expiry()})
    _install_secret_manager(monkeypatch, sm)

    with pytest.raises(ValueError, match="No access token is available"):
        runtime_oauth.get_oauth_access_token("microsoft")

    assert sm.sync_calls[-1]["force"] is True


def test_get_oauth_access_token_forces_sync_when_token_is_expired(monkeypatch):
    sm = _FakeSecretManager(
        {
            "MICROSOFT_ACCESS_TOKEN": "old-ms-token",
            "MICROSOFT_TOKEN_EXPIRES_AT": _past_expiry(),
        },
    )
    _install_secret_manager(monkeypatch, sm)

    def refresh_token() -> None:
        sm.secrets["MICROSOFT_ACCESS_TOKEN"] = "fresh-ms-token"
        sm.secrets["MICROSOFT_TOKEN_EXPIRES_AT"] = _future_expiry()

    sm.on_sync = refresh_token

    assert runtime_oauth.get_oauth_access_token("microsoft") == "fresh-ms-token"
    assert sm.sync_calls[-1]["force"] is True


def test_get_oauth_access_token_forces_sync_when_expiry_is_invalid(monkeypatch):
    sm = _FakeSecretManager(
        {
            "GOOGLE_ACCESS_TOKEN": "old-google-token",
            "GOOGLE_TOKEN_EXPIRES_AT": "not-a-date",
        },
    )
    _install_secret_manager(monkeypatch, sm)

    def refresh_token() -> None:
        sm.secrets["GOOGLE_ACCESS_TOKEN"] = "fresh-google-token"
        sm.secrets["GOOGLE_TOKEN_EXPIRES_AT"] = _future_expiry()

    sm.on_sync = refresh_token

    assert runtime_oauth.get_oauth_access_token("google") == "fresh-google-token"
    assert sm.sync_calls[-1]["force"] is True


def test_get_oauth_access_token_supports_multiple_providers(monkeypatch):
    sm = _FakeSecretManager(
        {
            "MICROSOFT_ACCESS_TOKEN": "fresh-ms-token",
            "MICROSOFT_TOKEN_EXPIRES_AT": _future_expiry(),
            "GOOGLE_ACCESS_TOKEN": "fresh-google-token",
            "GOOGLE_TOKEN_EXPIRES_AT": _future_expiry(),
        },
    )
    _install_secret_manager(monkeypatch, sm)

    assert runtime_oauth.get_oauth_access_token("microsoft") == "fresh-ms-token"
    assert runtime_oauth.get_oauth_access_token("google") == "fresh-google-token"


def test_get_refresh_token_oauth_env_overlay_returns_all_current_values(monkeypatch):
    sm = _FakeSecretManager(
        {
            "MICROSOFT_ACCESS_TOKEN": "fresh-ms-token",
            "MICROSOFT_TOKEN_EXPIRES_AT": _future_expiry(),
            "GOOGLE_ACCESS_TOKEN": "fresh-google-token",
            "GOOGLE_TOKEN_EXPIRES_AT": _future_expiry(),
        },
    )
    _install_secret_manager(monkeypatch, sm)

    overlay = runtime_oauth.get_refresh_token_oauth_env_overlay()

    assert overlay["MICROSOFT_ACCESS_TOKEN"] == "fresh-ms-token"
    assert overlay["GOOGLE_ACCESS_TOKEN"] == "fresh-google-token"
    assert sm.sync_calls[-1]["reason"] == "oauth_env_overlay"
