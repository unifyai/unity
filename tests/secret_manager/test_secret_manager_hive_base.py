"""SecretManager resolves its credential tables through Hive or per-body roots.

``"Secrets"`` and ``"SecretDefault"`` are in ``_HIVE_SCOPED_TABLES``, so Hive
members share ``Hives/{hive_id}/Secrets`` plus ``Hives/{hive_id}/SecretDefault``
while solo bodies keep both on ``{user}/{assistant}/``. The binding overlay
(``SecretBinding``) and OAuth token store (``OAuthTokens``) are always per-body
so one Hive body cannot read another body's bindings or tokens.

These tests exercise the resolver contract directly so they don't need a live
Unify backend.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from unity.common.context_registry import ContextRegistry
from unity.secret_manager.secret_manager import SecretManager
from unity.session_details import SESSION_DETAILS

_PER_BODY_BASE = "u7/7"


@pytest.fixture(autouse=True)
def _reset_session_details():
    """Each test gets a clean ``SESSION_DETAILS`` and registry state."""
    SESSION_DETAILS.reset()
    SESSION_DETAILS.populate(agent_id=7, user_id="u7")
    previous_base = ContextRegistry._base_context
    ContextRegistry._base_context = _PER_BODY_BASE
    yield
    SESSION_DETAILS.reset()
    ContextRegistry._base_context = previous_base


def _resolve() -> dict:
    """Return the resolver's output for ``SecretManager`` without constructing one.

    ``SecretManager.__init__`` requires a live Unify backend; the resolver
    alone is enough to assert the storage roots of each required context.
    """
    return ContextRegistry._get_contexts_for_manager(SecretManager)


# ─── Hive-scoped tables ─────────────────────────────────────────────────


def test_hive_member_resolves_secrets_to_hive_root():
    """A Hive member reads and writes ``Secrets`` under ``Hives/{hive_id}``."""
    SESSION_DETAILS.hive_id = 42

    resolved = _resolve()

    assert resolved["Secrets"]["resolved_name"] == "Hives/42/Secrets"


def test_hive_member_resolves_secret_default_to_hive_root():
    """``SecretDefault`` is Hive-wide and lives alongside ``Secrets``."""
    SESSION_DETAILS.hive_id = 42

    resolved = _resolve()

    assert resolved["SecretDefault"]["resolved_name"] == "Hives/42/SecretDefault"


def test_solo_body_resolves_secrets_to_per_body_root():
    """Solo bodies keep ``Secrets`` on their per-body root."""
    assert SESSION_DETAILS.hive_id is None

    resolved = _resolve()

    assert resolved["Secrets"]["resolved_name"] == f"{_PER_BODY_BASE}/Secrets"


def test_solo_body_resolves_secret_default_to_per_body_root():
    """Solo bodies don't actually consult ``SecretDefault``, but the context
    itself must still resolve without a Hive root since there's no Hive to
    anchor it to.
    """
    assert SESSION_DETAILS.hive_id is None

    resolved = _resolve()

    assert (
        resolved["SecretDefault"]["resolved_name"] == f"{_PER_BODY_BASE}/SecretDefault"
    )


# ─── Per-body overlays ──────────────────────────────────────────────────


def test_hive_member_keeps_secret_binding_on_per_body_root():
    """``SecretBinding`` is the per-body overlay and never Hive-shared."""
    SESSION_DETAILS.hive_id = 42

    resolved = _resolve()

    assert (
        resolved["SecretBinding"]["resolved_name"] == f"{_PER_BODY_BASE}/SecretBinding"
    )


def test_hive_member_keeps_oauth_tokens_on_per_body_root():
    """OAuth tokens are per-body so cross-body leakage is impossible."""
    SESSION_DETAILS.hive_id = 42

    resolved = _resolve()

    assert resolved["OAuthTokens"]["resolved_name"] == f"{_PER_BODY_BASE}/OAuthTokens"


def test_solo_body_keeps_secret_binding_on_per_body_root():
    """Solo bodies keep the binding overlay on their per-body root too."""
    assert SESSION_DETAILS.hive_id is None

    resolved = _resolve()

    assert (
        resolved["SecretBinding"]["resolved_name"] == f"{_PER_BODY_BASE}/SecretBinding"
    )


# ─── Foreign-key routing across Hive / per-body split ───────────────────


def test_secret_binding_fk_resolves_to_hive_secrets_for_hive_member():
    """``SecretBinding.secret_id`` FK points at the shared ``Secrets`` table."""
    SESSION_DETAILS.hive_id = 42

    resolved = _resolve()

    fks = resolved["SecretBinding"]["resolved_foreign_keys"]
    assert fks and len(fks) == 1
    assert fks[0]["references"] == "Hives/42/Secrets.secret_id"


def test_secret_binding_fk_resolves_to_per_body_secrets_for_solo():
    """Solo bodies keep everything on their per-body base."""
    assert SESSION_DETAILS.hive_id is None

    resolved = _resolve()

    fks = resolved["SecretBinding"]["resolved_foreign_keys"]
    assert fks and len(fks) == 1
    assert fks[0]["references"] == f"{_PER_BODY_BASE}/Secrets.secret_id"


def test_secret_default_fk_resolves_to_hive_secrets_for_hive_member():
    """``SecretDefault.secret_id`` FK points at the same Hive ``Secrets``."""
    SESSION_DETAILS.hive_id = 42

    resolved = _resolve()

    fks = resolved["SecretDefault"]["resolved_foreign_keys"]
    assert fks and len(fks) == 1
    assert fks[0]["references"] == "Hives/42/Secrets.secret_id"


# ─── Provisioning hits every required context ──────────────────────────


def test_resolver_provisions_every_secret_context():
    """All four required contexts run through the provisioning hook."""
    SESSION_DETAILS.hive_id = 42

    with patch.object(
        ContextRegistry,
        "_create_context_wrapper",
        side_effect=lambda name, entry: entry["resolved_name"],
    ) as provision:
        resolved = ContextRegistry._get_contexts_for_manager(SecretManager)
        for entry in resolved.values():
            ContextRegistry._create_context_wrapper(
                SecretManager.__name__,
                entry,
            )

    resolved_names = {
        call.args[1]["resolved_name"] for call in provision.call_args_list
    }
    assert "Hives/42/Secrets" in resolved_names
    assert "Hives/42/SecretDefault" in resolved_names
    assert f"{_PER_BODY_BASE}/SecretBinding" in resolved_names
    assert f"{_PER_BODY_BASE}/OAuthTokens" in resolved_names
