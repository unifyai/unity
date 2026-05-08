from __future__ import annotations

import os
from threading import Lock

from unity.secret_manager.secret_manager import SecretManager


def _unit_secret_manager() -> SecretManager:
    sm = object.__new__(SecretManager)
    sm._assistant_secret_sync_lock = Lock()
    sm._last_assistant_secret_sync_success_at = None
    sm._last_assistant_secret_sync_failure_at = None
    return sm


def test_sync_assistant_secrets_if_stale_debounces(monkeypatch):
    sm = _unit_secret_manager()
    calls: list[str] = []

    monkeypatch.setattr(
        sm,
        "_sync_assistant_secrets",
        lambda: calls.append("assistant"),
    )
    monkeypatch.setattr(sm, "_sync_dotenv", lambda: calls.append("dotenv"))

    assert sm.sync_assistant_secrets_if_stale(reason="test") is True
    assert sm.sync_assistant_secrets_if_stale(reason="test") is False

    assert calls == ["assistant", "dotenv"]


def test_sync_assistant_secrets_if_stale_force_bypasses_debounce(monkeypatch):
    sm = _unit_secret_manager()
    calls: list[str] = []

    monkeypatch.setattr(
        sm,
        "_sync_assistant_secrets",
        lambda: calls.append("assistant"),
    )
    monkeypatch.setattr(sm, "_sync_dotenv", lambda: calls.append("dotenv"))

    assert sm.sync_assistant_secrets_if_stale(reason="test") is True
    assert sm.sync_assistant_secrets_if_stale(reason="test", force=True) is True

    assert calls == ["assistant", "dotenv", "assistant", "dotenv"]


def test_sync_assistant_secrets_if_stale_observes_failure_cooldown(monkeypatch):
    sm = _unit_secret_manager()
    calls = {"assistant": 0, "dotenv": 0}

    def fail_sync():
        calls["assistant"] += 1
        raise RuntimeError("sync failed")

    def sync_dotenv():
        calls["dotenv"] += 1

    monkeypatch.setattr(sm, "_sync_assistant_secrets", fail_sync)
    monkeypatch.setattr(sm, "_sync_dotenv", sync_dotenv)

    assert sm.sync_assistant_secrets_if_stale(reason="test") is False
    assert sm.sync_assistant_secrets_if_stale(reason="test") is False
    assert calls == {"assistant": 1, "dotenv": 0}


def test_resolve_secret_allowlist_includes_runtime_oauth_secret_names():
    allowlist = SecretManager._resolve_secret_allowlist()

    assert "MICROSOFT_ACCESS_TOKEN" in allowlist
    assert "MICROSOFT_TOKEN_EXPIRES_AT" in allowlist
    assert "GOOGLE_ACCESS_TOKEN" in allowlist
    assert "GOOGLE_TOKEN_EXPIRES_AT" in allowlist


def test_env_merge_and_write_updates_dotenv_and_process_env(monkeypatch, tmp_path):
    sm = _unit_secret_manager()
    dotenv_path = tmp_path / ".env"

    monkeypatch.setattr(sm, "_dotenv_path", lambda: str(dotenv_path))
    monkeypatch.delenv("MICROSOFT_ACCESS_TOKEN", raising=False)

    sm._env_merge_and_write(
        add_or_update={"MICROSOFT_ACCESS_TOKEN": "fresh-token"},
        remove_keys=None,
    )

    assert dotenv_path.read_text() == "MICROSOFT_ACCESS_TOKEN=fresh-token\n"
    assert os.environ["MICROSOFT_ACCESS_TOKEN"] == "fresh-token"
