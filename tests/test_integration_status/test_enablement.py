"""Unit tests for ``unity.integration_status``.

These tests exercise the enablement-detection logic with synthetic registry
rows so we don't need a real DataManager context provisioned.  Live-context
tests live separately under ``tests/secret_manager/`` once we wire the
hook end-to-end against Orchestra.
"""

from __future__ import annotations

import json

import pytest

from unity import integration_status as IS


def _row(
    *,
    slug: str,
    label: str,
    required: list[str],
    optional: list[str] | None = None,
    function_names: list[str] | None = None,
    guidance_titles: list[str] | None = None,
) -> dict:
    return {
        "slug": slug,
        "label": label,
        "category": "test",
        "version": "0.1.0",
        "tier": "api",
        "quality": "bronze",
        "required_secrets_json": json.dumps(required),
        "optional_secrets_json": json.dumps(optional or []),
        "function_names_json": json.dumps(function_names or []),
        "guidance_titles_json": json.dumps(guidance_titles or []),
        "capability_ids_json": "[]",
        "tags_json": "[]",
        "homepage": "",
        "description": f"{label} description",
    }


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    """Reset the per-session cache and reload the registry from the supplied
    rows on each test.  Bypasses DataManager entirely."""
    IS.reset_session_cache()
    yield
    IS.reset_session_cache()


def _seed_registry(monkeypatch, rows: list[dict]) -> None:
    """Patch ``_load_registry`` to return *rows* and prime the cache."""

    def _fake_load() -> list[dict]:
        cache = IS._session_cache()
        cache["registry_loaded"] = True
        cache["registry"] = rows
        return rows

    monkeypatch.setattr(IS, "_load_registry", _fake_load)


# ---------------------------------------------------------------------------
# Empty registry → no detection
# ---------------------------------------------------------------------------


def test_recompute_with_no_registry_returns_empty(monkeypatch):
    _seed_registry(monkeypatch, [])

    IS.recompute_enablement(
        assistant_id=1,
        secrets={"HUBSPOT_PRIVATE_APP_TOKEN": "tok"},
    )

    assert IS.get_enabled_integrations() == []
    assert IS.get_setup_completeness() == {}


def test_summary_empty_when_no_registry(monkeypatch):
    _seed_registry(monkeypatch, [])
    IS.recompute_enablement(assistant_id=1, secrets={})
    assert IS.enabled_summary_for_prompt() == ""


def test_build_guidance_filter_scope_returns_none_when_registry_empty(monkeypatch):
    _seed_registry(monkeypatch, [])
    IS.recompute_enablement(assistant_id=1, secrets={})
    assert IS.build_guidance_filter_scope() is None


# ---------------------------------------------------------------------------
# Single-integration enablement: HubSpot
# ---------------------------------------------------------------------------


def _hubspot_row() -> dict:
    return _row(
        slug="hubspot",
        label="HubSpot",
        required=["HUBSPOT_PRIVATE_APP_TOKEN"],
        optional=["HUBSPOT_PORTAL_ID"],
        function_names=["get_hubspot_contact", "search_hubspot_contacts"],
        guidance_titles=["Hubspot Overview", "Hubspot Crm Contacts"],
    )


def test_hubspot_enabled_when_required_token_present(monkeypatch):
    _seed_registry(monkeypatch, [_hubspot_row()])

    IS.recompute_enablement(
        assistant_id=1,
        secrets={"HUBSPOT_PRIVATE_APP_TOKEN": "tok"},
    )

    assert IS.get_enabled_integrations() == ["hubspot"]
    completeness = IS.get_setup_completeness()
    assert completeness["hubspot"]["status"] == "configured"
    assert completeness["hubspot"]["missing_optional_secrets"] == ["HUBSPOT_PORTAL_ID"]


def test_hubspot_disabled_without_required_token(monkeypatch):
    _seed_registry(monkeypatch, [_hubspot_row()])

    IS.recompute_enablement(assistant_id=1, secrets={"HUBSPOT_PORTAL_ID": "12345"})

    assert IS.get_enabled_integrations() == []


def test_hubspot_fully_connected_when_optional_present(monkeypatch):
    _seed_registry(monkeypatch, [_hubspot_row()])

    IS.recompute_enablement(
        assistant_id=1,
        secrets={
            "HUBSPOT_PRIVATE_APP_TOKEN": "tok",
            "HUBSPOT_PORTAL_ID": "12345",
        },
    )

    completeness = IS.get_setup_completeness()
    assert completeness["hubspot"]["status"] == "fully_connected"
    assert completeness["hubspot"]["missing_optional_secrets"] == []


def test_empty_string_secret_treated_as_missing(monkeypatch):
    """Orchestra returns ``""`` for unset secrets in some paths.  Treat
    empty/whitespace as not-present so we don't false-positive enablement."""
    _seed_registry(monkeypatch, [_hubspot_row()])

    IS.recompute_enablement(
        assistant_id=1,
        secrets={"HUBSPOT_PRIVATE_APP_TOKEN": ""},
    )

    assert IS.get_enabled_integrations() == []


# ---------------------------------------------------------------------------
# Multi-secret AND: Employment Hero (CLIENT_ID + CLIENT_SECRET both required)
# ---------------------------------------------------------------------------


def _eh_row() -> dict:
    return _row(
        slug="employment_hero",
        label="Employment Hero",
        required=[
            "EMPLOYMENTHERO_OAUTH_CLIENT_ID",
            "EMPLOYMENTHERO_OAUTH_CLIENT_SECRET",
        ],
        optional=[
            "EMPLOYMENTHERO_REFRESH_TOKEN",
            "EMPLOYMENTHERO_ORGANISATION_ID",
            "EMPLOYMENTHERO_HUB_DOMAIN",
        ],
        function_names=["get_employmenthero_employee"],
        guidance_titles=["Employmenthero Overview"],
    )


def test_eh_disabled_when_only_client_id_present(monkeypatch):
    _seed_registry(monkeypatch, [_eh_row()])

    IS.recompute_enablement(
        assistant_id=1,
        secrets={"EMPLOYMENTHERO_OAUTH_CLIENT_ID": "id"},
    )

    assert IS.get_enabled_integrations() == []


def test_eh_configured_with_both_required_but_oauth_not_complete(monkeypatch):
    """Once CLIENT_ID + CLIENT_SECRET are pasted, EH is *configured* but the
    OAuth Connect flow that populates REFRESH_TOKEN hasn't run.  We surface
    that gap via ``missing_optional_secrets`` so the prompt can guide the
    user to complete setup."""
    _seed_registry(monkeypatch, [_eh_row()])

    IS.recompute_enablement(
        assistant_id=1,
        secrets={
            "EMPLOYMENTHERO_OAUTH_CLIENT_ID": "id",
            "EMPLOYMENTHERO_OAUTH_CLIENT_SECRET": "secret",
        },
    )

    enabled = IS.get_enabled_integrations()
    assert enabled == ["employment_hero"]
    completeness = IS.get_setup_completeness()
    assert completeness["employment_hero"]["status"] == "configured"
    assert (
        "EMPLOYMENTHERO_REFRESH_TOKEN"
        in completeness["employment_hero"]["missing_optional_secrets"]
    )


def test_eh_fully_connected_with_full_oauth_set(monkeypatch):
    _seed_registry(monkeypatch, [_eh_row()])

    IS.recompute_enablement(
        assistant_id=1,
        secrets={
            "EMPLOYMENTHERO_OAUTH_CLIENT_ID": "id",
            "EMPLOYMENTHERO_OAUTH_CLIENT_SECRET": "secret",
            "EMPLOYMENTHERO_REFRESH_TOKEN": "rt",
            "EMPLOYMENTHERO_ORGANISATION_ID": "org",
            "EMPLOYMENTHERO_HUB_DOMAIN": "acme.employmenthero.com",
        },
    )

    completeness = IS.get_setup_completeness()
    assert completeness["employment_hero"]["status"] == "fully_connected"


# ---------------------------------------------------------------------------
# Mid-session recompute (the core promise of the _sync_assistant_secrets hook)
# ---------------------------------------------------------------------------


def test_mid_session_recompute_picks_up_newly_added_secret(monkeypatch):
    """Simulates the user pasting a HubSpot token mid-session.  The first
    sync call runs with no token; the second runs after the user adds it.
    The enabled set must update without any session restart."""
    _seed_registry(monkeypatch, [_hubspot_row()])

    IS.recompute_enablement(assistant_id=1, secrets={})
    assert IS.get_enabled_integrations() == []

    # User pastes the token.  Next _sync_assistant_secrets fires.
    IS.recompute_enablement(
        assistant_id=1,
        secrets={"HUBSPOT_PRIVATE_APP_TOKEN": "tok"},
    )
    assert IS.get_enabled_integrations() == ["hubspot"]


def test_mid_session_recompute_drops_disabled_integration(monkeypatch):
    """Symmetric: user revokes a token mid-session → integration falls out
    of the enabled set on next sync."""
    _seed_registry(monkeypatch, [_hubspot_row()])

    IS.recompute_enablement(
        assistant_id=1,
        secrets={"HUBSPOT_PRIVATE_APP_TOKEN": "tok"},
    )
    assert IS.get_enabled_integrations() == ["hubspot"]

    IS.recompute_enablement(assistant_id=1, secrets={})
    assert IS.get_enabled_integrations() == []


# ---------------------------------------------------------------------------
# Multi-integration: HubSpot + Employment Hero
# ---------------------------------------------------------------------------


def test_multi_integration_independent_enablement(monkeypatch):
    """HubSpot and EH should enable independently of each other."""
    _seed_registry(monkeypatch, [_hubspot_row(), _eh_row()])

    IS.recompute_enablement(
        assistant_id=1,
        secrets={"HUBSPOT_PRIVATE_APP_TOKEN": "tok"},
    )

    enabled = sorted(IS.get_enabled_integrations())
    assert enabled == ["hubspot"]


# ---------------------------------------------------------------------------
# Allowlist union (drives SecretManager._resolve_secret_allowlist)
# ---------------------------------------------------------------------------


def test_all_known_secret_names_unions_all_required_and_optional(monkeypatch):
    _seed_registry(monkeypatch, [_hubspot_row(), _eh_row()])

    names = IS.all_known_secret_names()

    # Required secrets from both packages.
    assert "HUBSPOT_PRIVATE_APP_TOKEN" in names
    assert "EMPLOYMENTHERO_OAUTH_CLIENT_ID" in names
    assert "EMPLOYMENTHERO_OAUTH_CLIENT_SECRET" in names
    # Optional secrets from both packages.
    assert "HUBSPOT_PORTAL_ID" in names
    assert "EMPLOYMENTHERO_REFRESH_TOKEN" in names


def test_all_known_secret_names_empty_when_registry_empty(monkeypatch):
    _seed_registry(monkeypatch, [])
    assert IS.all_known_secret_names() == set()


# ---------------------------------------------------------------------------
# Prompt-block rendering
# ---------------------------------------------------------------------------


def test_summary_lists_active_and_inactive_with_setup_hint(monkeypatch):
    _seed_registry(monkeypatch, [_hubspot_row(), _eh_row()])

    IS.recompute_enablement(
        assistant_id=1,
        secrets={
            "HUBSPOT_PRIVATE_APP_TOKEN": "tok",
            "HUBSPOT_PORTAL_ID": "12345",
        },
    )

    summary = IS.enabled_summary_for_prompt()
    # HubSpot ends up in Active.
    assert "Active integrations:" in summary
    assert "HubSpot (fully_connected)" in summary
    # Employment Hero ends up in Inactive with the missing required keys
    # spelled out, so the LLM can guide the user.
    assert "Inactive (credentials not configured):" in summary
    assert "Employment Hero" in summary
    assert "EMPLOYMENTHERO_OAUTH_CLIENT_ID" in summary
    assert "EMPLOYMENTHERO_OAUTH_CLIENT_SECRET" in summary


def test_summary_active_with_configured_lists_missing_optional(monkeypatch):
    """When an integration is configured (required met) but missing optional
    secrets like REFRESH_TOKEN, the prompt should call this out so the LLM
    can guide the user through the OAuth Connect step."""
    _seed_registry(monkeypatch, [_eh_row()])

    IS.recompute_enablement(
        assistant_id=1,
        secrets={
            "EMPLOYMENTHERO_OAUTH_CLIENT_ID": "id",
            "EMPLOYMENTHERO_OAUTH_CLIENT_SECRET": "secret",
        },
    )

    summary = IS.enabled_summary_for_prompt()
    assert "Active integrations:" in summary
    assert "configured" in summary
    assert "EMPLOYMENTHERO_REFRESH_TOKEN" in summary


# ---------------------------------------------------------------------------
# Filter scope construction (used to set GuidanceManager.filter_scope)
# ---------------------------------------------------------------------------


def test_build_guidance_filter_scope_returns_empty_set_when_registry_seeded_but_no_enablement(
    monkeypatch,
):
    """When integrations exist on the deployment but none have credentials,
    we want guidance retrieval to actively *exclude* integration guidance,
    not fall through silently.  ``guidance_id in ()`` matches nothing —
    this is intentional."""
    _seed_registry(monkeypatch, [_hubspot_row()])
    IS.recompute_enablement(assistant_id=1, secrets={})

    scope = IS.build_guidance_filter_scope()
    assert scope == "guidance_id in ()"


# ---------------------------------------------------------------------------
# Cache reset
# ---------------------------------------------------------------------------


def test_reset_session_cache_clears_state(monkeypatch):
    _seed_registry(monkeypatch, [_hubspot_row()])

    IS.recompute_enablement(
        assistant_id=1,
        secrets={"HUBSPOT_PRIVATE_APP_TOKEN": "tok"},
    )
    assert IS.get_enabled_integrations() == ["hubspot"]

    IS.reset_session_cache()
    assert IS.get_enabled_integrations() == []
    assert IS.get_setup_completeness() == {}
