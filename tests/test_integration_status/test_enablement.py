"""Unit tests for ``unity.integration_status`` enablement read.

Exercises the pure read API (``get_enabled_integrations``,
``get_setup_completeness``, ``enabled_summary_for_prompt``,
``build_guidance_filter_scope``) against synthetic disk-package metadata
and synthetic secret keysets.

Synthetic state is injected by stubbing two helpers:

* ``discover_available_packages`` from
  ``unity.integration_status.discovery`` — returns the list of
  package-metadata dicts the assistant has on disk.
* ``_read_local_secret_keyset`` on the ``unity.integration_status``
  module — returns the set of currently-present secret names in the
  assistant's local Secrets context.

Manager-coupled helpers (``enabled_function_ids``,
``enabled_guidance_ids``) are tested separately under
``tests/test_integration_status/test_manager_resolution.py`` (TODO) once
we have lightweight FunctionManager / GuidanceManager fakes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from unity import integration_status as IS

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _pkg(
    *,
    slug: str,
    label: str,
    required: list[str],
    optional: list[str] | None = None,
    function_names: list[str] | None = None,
    guidance_titles: list[str] | None = None,
) -> dict:
    """Build a synthetic package-metadata dict matching the shape returned
    by ``discover_available_packages``."""
    return {
        "slug": slug,
        "label": label,
        "category": "test",
        "version": "0.1.0",
        "tier": "api",
        "root_dir": Path("/nonexistent"),
        "required_secrets": required,
        "optional_secrets": optional or [],
        "function_names": function_names or [],
        "guidance_titles": guidance_titles or [],
        "function_dir": None,
        "guidance_dir": None,
    }


def _stub_packages_and_keyset(
    monkeypatch: pytest.MonkeyPatch,
    *,
    packages: list[dict],
    keyset: set[str] | None = None,
) -> None:
    """Stub disk discovery + local keyset reads for an enablement test."""
    from unity.integration_status import discovery as D

    monkeypatch.setattr(D, "discover_available_packages", lambda: packages)
    monkeypatch.setattr(IS, "_read_local_secret_keyset", lambda: set(keyset or set()))


@pytest.fixture(autouse=True)
def _reset_cache():
    IS.reset_session_cache()
    yield
    IS.reset_session_cache()


# ---------------------------------------------------------------------------
# get_enabled_integrations — basic enablement logic
# ---------------------------------------------------------------------------


def test_no_packages_returns_empty(monkeypatch):
    _stub_packages_and_keyset(
        monkeypatch,
        packages=[],
        keyset={"HUBSPOT_PRIVATE_APP_TOKEN"},
    )
    assert IS.get_enabled_integrations() == {}


def test_required_secret_present_enables_package(monkeypatch):
    _stub_packages_and_keyset(
        monkeypatch,
        packages=[
            _pkg(
                slug="hubspot",
                label="HubSpot",
                required=["HUBSPOT_PRIVATE_APP_TOKEN"],
            ),
        ],
        keyset={"HUBSPOT_PRIVATE_APP_TOKEN"},
    )
    enabled = IS.get_enabled_integrations()
    assert set(enabled.keys()) == {"hubspot"}
    assert enabled["hubspot"]["label"] == "HubSpot"


def test_required_secret_missing_disables_package(monkeypatch):
    _stub_packages_and_keyset(
        monkeypatch,
        packages=[
            _pkg(
                slug="hubspot",
                label="HubSpot",
                required=["HUBSPOT_PRIVATE_APP_TOKEN"],
            ),
        ],
        keyset=set(),
    )
    assert IS.get_enabled_integrations() == {}


def test_no_required_secrets_means_always_enabled(monkeypatch):
    """Packages with no declared required secrets are read-only / always-on
    and should be enabled unconditionally."""
    _stub_packages_and_keyset(
        monkeypatch,
        packages=[_pkg(slug="public_data", label="Public Data", required=[])],
        keyset=set(),
    )
    assert "public_data" in IS.get_enabled_integrations()


def test_multi_secret_AND_requires_all_present(monkeypatch):
    eh = _pkg(
        slug="employment_hero",
        label="Employment Hero",
        required=["EH_CLIENT_ID", "EH_CLIENT_SECRET"],
    )
    # Only one of two required secrets present.
    _stub_packages_and_keyset(monkeypatch, packages=[eh], keyset={"EH_CLIENT_ID"})
    assert IS.get_enabled_integrations() == {}

    # Both present → enabled.
    _stub_packages_and_keyset(
        monkeypatch,
        packages=[eh],
        keyset={"EH_CLIENT_ID", "EH_CLIENT_SECRET"},
    )
    assert "employment_hero" in IS.get_enabled_integrations()


def test_multiple_packages_independently_enabled(monkeypatch):
    _stub_packages_and_keyset(
        monkeypatch,
        packages=[
            _pkg(slug="hubspot", label="HubSpot", required=["HUBSPOT_TOKEN"]),
            _pkg(slug="webex", label="Webex", required=["WEBEX_TOKEN"]),
            _pkg(slug="salesforce", label="Salesforce", required=["SF_TOKEN"]),
        ],
        keyset={"HUBSPOT_TOKEN", "WEBEX_TOKEN"},  # SF missing
    )
    enabled = IS.get_enabled_integrations()
    assert set(enabled.keys()) == {"hubspot", "webex"}


# ---------------------------------------------------------------------------
# get_setup_completeness — fully_connected vs configured
# ---------------------------------------------------------------------------


def test_completeness_configured_when_optional_missing(monkeypatch):
    _stub_packages_and_keyset(
        monkeypatch,
        packages=[
            _pkg(
                slug="hubspot",
                label="HubSpot",
                required=["HUBSPOT_TOKEN"],
                optional=["HUBSPOT_PORTAL_ID"],
            ),
        ],
        keyset={"HUBSPOT_TOKEN"},
    )
    comp = IS.get_setup_completeness()
    assert comp["hubspot"]["status"] == "configured"
    assert comp["hubspot"]["missing_optional_secrets"] == ["HUBSPOT_PORTAL_ID"]
    assert comp["hubspot"]["missing_required_secrets"] == []


def test_completeness_fully_connected_when_optional_present(monkeypatch):
    _stub_packages_and_keyset(
        monkeypatch,
        packages=[
            _pkg(
                slug="hubspot",
                label="HubSpot",
                required=["HUBSPOT_TOKEN"],
                optional=["HUBSPOT_PORTAL_ID"],
            ),
        ],
        keyset={"HUBSPOT_TOKEN", "HUBSPOT_PORTAL_ID"},
    )
    comp = IS.get_setup_completeness()
    assert comp["hubspot"]["status"] == "fully_connected"
    assert comp["hubspot"]["missing_optional_secrets"] == []


def test_completeness_only_includes_enabled(monkeypatch):
    _stub_packages_and_keyset(
        monkeypatch,
        packages=[
            _pkg(slug="hubspot", label="HubSpot", required=["HUBSPOT_TOKEN"]),
            _pkg(slug="webex", label="Webex", required=["WEBEX_TOKEN"]),
        ],
        keyset={"HUBSPOT_TOKEN"},
    )
    comp = IS.get_setup_completeness()
    assert set(comp.keys()) == {"hubspot"}


# ---------------------------------------------------------------------------
# enabled_summary_for_prompt — prompt block rendering
# ---------------------------------------------------------------------------


def test_summary_empty_when_no_packages(monkeypatch):
    _stub_packages_and_keyset(monkeypatch, packages=[], keyset=set())
    assert IS.enabled_summary_for_prompt() == ""


def test_summary_renders_active_and_inactive(monkeypatch):
    _stub_packages_and_keyset(
        monkeypatch,
        packages=[
            _pkg(slug="hubspot", label="HubSpot", required=["HUBSPOT_TOKEN"]),
            _pkg(slug="webex", label="Webex", required=["WEBEX_TOKEN"]),
        ],
        keyset={"HUBSPOT_TOKEN"},
    )
    summary = IS.enabled_summary_for_prompt()
    assert "### Integrations" in summary
    assert "HubSpot" in summary
    assert "Webex" in summary
    assert "WEBEX_TOKEN" in summary
    # HubSpot should be in the "Active" section, Webex in "Inactive".
    active_section, inactive_section = summary.split(
        "Inactive (credentials not configured):",
    )
    assert "HubSpot" in active_section
    assert "Webex" in inactive_section


def test_summary_no_inactive_section_when_all_active(monkeypatch):
    _stub_packages_and_keyset(
        monkeypatch,
        packages=[_pkg(slug="hubspot", label="HubSpot", required=["HUBSPOT_TOKEN"])],
        keyset={"HUBSPOT_TOKEN"},
    )
    summary = IS.enabled_summary_for_prompt()
    assert "Active integrations:" in summary
    assert "Inactive" not in summary


# ---------------------------------------------------------------------------
# build_guidance_filter_scope — guidance gating
# ---------------------------------------------------------------------------


def test_filter_scope_none_when_no_packages(monkeypatch):
    """No integration packages on disk → don't filter (preserve existing
    behaviour for non-integration callers)."""
    _stub_packages_and_keyset(monkeypatch, packages=[], keyset={"FOO"})
    assert IS.build_guidance_filter_scope() is None


def test_filter_scope_never_match_when_packages_exist_but_none_enabled(monkeypatch):
    """Packages on disk but none enabled → hide all integration guidance."""
    _stub_packages_and_keyset(
        monkeypatch,
        packages=[_pkg(slug="hubspot", label="HubSpot", required=["HUBSPOT_TOKEN"])],
        keyset=set(),
    )
    assert IS.build_guidance_filter_scope() == "guidance_id in ()"


# ---------------------------------------------------------------------------
# register_available_integrations — startup pass
# ---------------------------------------------------------------------------


def test_register_is_idempotent(monkeypatch):
    """Re-running register_available_integrations doesn't double-process
    packages already in registered_slugs."""
    calls = {"functions": 0, "guidance": 0}

    def fake_register_functions(pkg):
        calls["functions"] += 1
        return 0

    def fake_register_guidance(pkg):
        calls["guidance"] += 1
        return 0

    _stub_packages_and_keyset(
        monkeypatch,
        packages=[_pkg(slug="hubspot", label="HubSpot", required=["X"])],
        keyset=set(),
    )
    monkeypatch.setattr(IS, "_register_functions", fake_register_functions)
    monkeypatch.setattr(IS, "_register_guidance", fake_register_guidance)

    IS.register_available_integrations()
    assert calls == {"functions": 1, "guidance": 1}

    # Second call should be a no-op for the already-registered slug.
    IS.register_available_integrations()
    assert calls == {"functions": 1, "guidance": 1}


def test_register_per_package_failure_does_not_halt_others(monkeypatch):
    """If one package's functions/guidance step raises, the remaining
    packages still get processed."""

    def failing_functions(pkg):
        if pkg["slug"] == "broken":
            raise RuntimeError("simulated failure")
        return 0

    _stub_packages_and_keyset(
        monkeypatch,
        packages=[
            _pkg(slug="broken", label="Broken", required=[]),
            _pkg(slug="hubspot", label="HubSpot", required=[]),
        ],
        keyset=set(),
    )
    monkeypatch.setattr(IS, "_register_functions", failing_functions)
    monkeypatch.setattr(IS, "_register_guidance", lambda pkg: 0)

    # Should not raise.
    IS.register_available_integrations()

    # Both slugs should be marked registered (per-step try/except in the
    # implementation; the broken slug's failure is logged but not
    # propagated, and registration of guidance still ran).
    cache = IS._session_cache()
    assert "broken" in cache["registered_slugs"]
    assert "hubspot" in cache["registered_slugs"]
