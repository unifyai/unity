from __future__ import annotations

import tempfile
import pathlib

import unify

from unity.settings import SETTINGS
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
        monkeypatch.setattr(SETTINGS.secret, "DOTENV_PATH", dotenv_path)

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
        monkeypatch.setattr(SETTINGS.secret, "DOTENV_PATH", dotenv_path)

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


def test_externally_added_secret_synced_on_ask(monkeypatch, secret_manager_context):
    """Secrets added outside SecretManager (e.g. via Console UI → Orchestra)
    should reach .env when ask() is called, not only on init."""
    with tempfile.TemporaryDirectory() as td:
        dotenv_path = str(pathlib.Path(td) / ".env")
        monkeypatch.setattr(SETTINGS.secret, "DOTENV_PATH", dotenv_path)

        sm = SecretManager()

        # Add one secret through the normal path (syncs to .env immediately)
        sm._create_secret(
            name="existing_key",
            value="val-existing",
            description="existed",
        )
        content = _read(dotenv_path)
        assert "existing_key=val-existing" in content

        # Simulate an external write — the Console UI writes directly to
        # Orchestra, bypassing SecretManager entirely.  This is the gap:
        # .env has no idea this secret exists.
        unify.log(
            context=sm._ctx,
            name="external_key",
            value="val-external",
            description="added via console",
            new=True,
            mutable=True,
        )

        # .env should NOT have the external secret yet (the old bug)
        content_before = _read(dotenv_path)
        assert (
            "external_key" not in content_before
        ), "external_key should not appear in .env before sync"

        # _sync_dotenv (called at the start of ask()) closes the gap
        sm._sync_dotenv()

        content_after = _read(dotenv_path)
        assert "external_key=val-external" in content_after
        assert "existing_key=val-existing" in content_after
