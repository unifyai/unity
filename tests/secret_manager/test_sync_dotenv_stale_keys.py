"""``_sync_dotenv`` evicts previously-exported keys when the slice shrinks.

The dotenv export surface is the bridge between stored credentials and
the ``os.environ`` that pooled venv subprocesses inherit at spawn time.
When a body unbinds an integration (or a Hive default is retracted),
the credential it exported to ``os.environ`` and ``~/.env`` must be
removed — otherwise the next pooled subprocess will happily inherit
the ghost value and hand it to a tool call after the retirement was
supposed to retire it.

``SecretManager`` tracks the names of the keys it last exported on
``self._exported_env_keys``; on the next sync it diffs that against
the freshly resolved slice and passes the departures into
``_env_merge_and_write`` as ``remove_keys``. These tests pin that
contract against the real dotenv file and the real ``os.environ`` so
a regression shows up immediately.
"""

from __future__ import annotations

import os
import pathlib
import tempfile

from unity.secret_manager.secret_manager import SecretManager
from unity.session_details import SESSION_DETAILS
from unity.settings import SETTINGS


def _read_dotenv(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return ""


def test_hive_unbind_evicts_previously_exported_key(
    secret_manager_context,
    monkeypatch,
):
    """Unbinding a Hive integration drops the exported key from env and .env.

    Hive bodies export only the binding-filtered slice of the vault.
    An unbind removes a row from that slice; ``_sync_dotenv`` must
    therefore remove the corresponding ``KEY=value`` line from
    ``~/.env`` and ``os.environ`` so pooled venvs spawned after the
    next invalidation never see the retired credential.
    """
    with tempfile.TemporaryDirectory() as td:
        dotenv_path = str(pathlib.Path(td) / ".env")
        monkeypatch.setattr(SETTINGS.secret, "DOTENV_PATH", dotenv_path)
        monkeypatch.delenv("salesforce_admin", raising=False)

        sm = SecretManager()
        sm._create_secret(
            name="salesforce_admin",
            value="tok-admin",
            description="admin",
        )

        monkeypatch.setattr(SESSION_DETAILS, "hive_id", 42, raising=False)

        sm._bind_secret(
            integration="salesforce",
            secret_name="salesforce_admin",
        )
        assert "salesforce_admin=tok-admin" in _read_dotenv(dotenv_path)
        assert os.environ.get("salesforce_admin") == "tok-admin"

        sm._unbind_secret(integration="salesforce")

        assert "salesforce_admin" not in _read_dotenv(dotenv_path)
        assert os.environ.get("salesforce_admin") is None


def test_unbind_one_of_two_bindings_preserves_the_other(
    secret_manager_context,
    monkeypatch,
):
    """Only the unbound integration's key is evicted; the rest stay."""
    with tempfile.TemporaryDirectory() as td:
        dotenv_path = str(pathlib.Path(td) / ".env")
        monkeypatch.setattr(SETTINGS.secret, "DOTENV_PATH", dotenv_path)
        monkeypatch.delenv("sf_key", raising=False)
        monkeypatch.delenv("stripe_key", raising=False)

        sm = SecretManager()
        sm._create_secret(name="sf_key", value="tok-sf", description="sf")
        sm._create_secret(
            name="stripe_key",
            value="tok-stripe",
            description="stripe",
        )

        monkeypatch.setattr(SESSION_DETAILS, "hive_id", 99, raising=False)

        sm._bind_secret(integration="salesforce", secret_name="sf_key")
        sm._bind_secret(integration="stripe", secret_name="stripe_key")

        contents = _read_dotenv(dotenv_path)
        assert "sf_key=tok-sf" in contents
        assert "stripe_key=tok-stripe" in contents

        sm._unbind_secret(integration="salesforce")

        contents = _read_dotenv(dotenv_path)
        assert "sf_key" not in contents
        assert "stripe_key=tok-stripe" in contents
        assert os.environ.get("sf_key") is None
        assert os.environ.get("stripe_key") == "tok-stripe"


def test_hive_default_retraction_evicts_exported_key(
    secret_manager_context,
    monkeypatch,
):
    """Replacing a Hive default with a different row evicts the old key.

    The previous default pointed at one credential; the new default
    points at another. The old credential's name leaves the resolved
    slice, and the next ``_sync_dotenv`` must drop it from the exported
    environment instead of leaving a stale ghost behind.
    """
    with tempfile.TemporaryDirectory() as td:
        dotenv_path = str(pathlib.Path(td) / ".env")
        monkeypatch.setattr(SETTINGS.secret, "DOTENV_PATH", dotenv_path)
        monkeypatch.delenv("old_default", raising=False)
        monkeypatch.delenv("new_default", raising=False)

        sm = SecretManager()
        sm._create_secret(
            name="old_default",
            value="tok-old",
            description="old",
        )
        sm._create_secret(
            name="new_default",
            value="tok-new",
            description="new",
        )

        monkeypatch.setattr(SESSION_DETAILS, "hive_id", 5, raising=False)

        sm._set_hive_default(integration="api", secret_name="old_default")
        assert "old_default=tok-old" in _read_dotenv(dotenv_path)

        sm._set_hive_default(integration="api", secret_name="new_default")

        contents = _read_dotenv(dotenv_path)
        assert "old_default" not in contents
        assert "new_default=tok-new" in contents
        assert os.environ.get("old_default") is None
        assert os.environ.get("new_default") == "tok-new"


def test_hive_member_rebind_rotates_exported_key(
    secret_manager_context,
    monkeypatch,
):
    """Rebinding an integration to a different row rotates the exported key.

    One call replaces the binding target; the exported name changes
    from the old row's ``name`` to the new row's ``name``. Neither the
    old nor the new ``KEY=value`` may linger in ``~/.env`` or
    ``os.environ``.
    """
    with tempfile.TemporaryDirectory() as td:
        dotenv_path = str(pathlib.Path(td) / ".env")
        monkeypatch.setattr(SETTINGS.secret, "DOTENV_PATH", dotenv_path)
        monkeypatch.delenv("alpha", raising=False)
        monkeypatch.delenv("beta", raising=False)

        sm = SecretManager()
        sm._create_secret(name="alpha", value="tok-alpha", description="a")
        sm._create_secret(name="beta", value="tok-beta", description="b")

        monkeypatch.setattr(SESSION_DETAILS, "hive_id", 11, raising=False)

        sm._bind_secret(integration="service", secret_name="alpha")
        assert "alpha=tok-alpha" in _read_dotenv(dotenv_path)

        sm._bind_secret(integration="service", secret_name="beta")

        contents = _read_dotenv(dotenv_path)
        assert "alpha" not in contents
        assert "beta=tok-beta" in contents
        assert os.environ.get("alpha") is None
        assert os.environ.get("beta") == "tok-beta"


def test_exported_env_keys_tracking_survives_reinstantiation(
    secret_manager_context,
    monkeypatch,
):
    """A fresh ``SecretManager`` reconstructs its exported-key set from state.

    The tracking attribute lives on the instance, not in durable
    storage; a new instance must pick the right eviction set up from
    the resolver alone. This pins the invariant that the eviction
    logic is *derived from the resolved slice*, not from a cached set
    that could drift if a process restarts mid-rotation.
    """
    with tempfile.TemporaryDirectory() as td:
        dotenv_path = str(pathlib.Path(td) / ".env")
        monkeypatch.setattr(SETTINGS.secret, "DOTENV_PATH", dotenv_path)
        monkeypatch.delenv("gamma", raising=False)

        sm = SecretManager()
        sm._create_secret(name="gamma", value="tok-gamma", description="g")

        monkeypatch.setattr(SESSION_DETAILS, "hive_id", 3, raising=False)

        sm._bind_secret(integration="service", secret_name="gamma")
        assert os.environ.get("gamma") == "tok-gamma"

        sm_restarted = SecretManager()
        assert sm_restarted._exported_env_keys >= {"gamma"}

        sm_restarted._unbind_secret(integration="service")

        assert os.environ.get("gamma") is None
        assert "gamma" not in _read_dotenv(dotenv_path)
