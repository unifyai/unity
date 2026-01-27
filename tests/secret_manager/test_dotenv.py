from __future__ import annotations

import tempfile
import pathlib


from unity.secret_manager.secret_manager import SecretManager


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return ""


def test_created_and_backfilled_on_init(monkeypatch, secret_manager_context):
    with tempfile.TemporaryDirectory() as td:
        dotenv_path = str(pathlib.Path(td) / ".env")
        monkeypatch.setenv("UNITY_SECRET_DOTENV_PATH", dotenv_path)

        sm = SecretManager()
        # Seed two secrets in storage
        sm._create_secret(name="unify_key", value="sk-xyz", description="api")
        sm._create_secret(name="db_password", value="abc123", description="db")

        # Re-instantiate to trigger backfill-on-init
        sm2 = SecretManager()
        content = _read(dotenv_path)
        assert "unify_key=sk-xyz" in content
        assert "db_password=abc123" in content


def test_updates_on_create_update_delete(monkeypatch, secret_manager_context):
    with tempfile.TemporaryDirectory() as td:
        dotenv_path = str(pathlib.Path(td) / ".env")
        monkeypatch.setenv("UNITY_SECRET_DOTENV_PATH", dotenv_path)

        sm = SecretManager()

        # create
        sm._create_secret(name="api_key", value="sk-live-abc", description="api")
        content = _read(dotenv_path)
        assert "api_key=sk-live-abc" in content

        # update value
        sm._update_secret(name="api_key", value="sk-live-rotated")
        content = _read(dotenv_path)
        assert "api_key=sk-live-rotated" in content
        assert "api_key=sk-live-abc" not in content

        # delete
        sm._delete_secret(name="api_key")
        content = _read(dotenv_path)
        assert "api_key=" not in content
