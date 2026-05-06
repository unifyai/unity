"""Integration-style tests for ``unity.integration_status.discovery``.

These run against the real unity-deploy package roots so they verify the
import path, manifest parsing, and slug-to-directory contract end to end.
Lightweight (~20ms; no network, no DataManager).
"""

from __future__ import annotations

import importlib

import pytest

from unity.integration_status import discovery as D

# This test module exercises real package discovery, which needs unity_deploy
# importable.  In unity's standalone venv that's not the case; skip cleanly
# rather than emit confusing failures.
_UNITY_DEPLOY_AVAILABLE = importlib.util.find_spec("unity_deploy") is not None
pytestmark = pytest.mark.skipif(
    not _UNITY_DEPLOY_AVAILABLE,
    reason="unity_deploy not installed in this venv; run from unity-deploy/.venv",
)


@pytest.fixture(autouse=True)
def _reset():
    D.reset_discovery_cache()
    yield
    D.reset_discovery_cache()


def test_discover_finds_hubspot_and_employment_hero():
    """The two real packages we ship.  Slug-to-directory contract verified
    by the loader at deploy time; this test ensures runtime discovery sees
    the same view."""
    pkgs = D.discover_available_packages()
    slugs = {p["slug"] for p in pkgs}
    assert "hubspot" in slugs
    assert "employment_hero" in slugs


def test_get_package_for_slug_returns_full_record_for_hubspot():
    pkg = D.get_package_for_slug("hubspot")
    assert pkg is not None
    assert pkg["slug"] == "hubspot"
    assert pkg["label"] == "HubSpot"
    assert "HUBSPOT_PRIVATE_APP_TOKEN" in pkg["required_secrets"]
    # Optional secret declared in the manifest.
    assert "HUBSPOT_PORTAL_ID" in pkg["optional_secrets"]
    # function_dir + guidance_dir resolve to real paths.
    assert pkg["function_dir"] is not None and pkg["function_dir"].is_dir()
    assert pkg["guidance_dir"] is not None and pkg["guidance_dir"].is_dir()


def test_get_package_for_slug_returns_full_record_for_employment_hero():
    pkg = D.get_package_for_slug("employment_hero")
    assert pkg is not None
    assert pkg["slug"] == "employment_hero"
    # Per the manifest, both CLIENT_ID and CLIENT_SECRET are required=true.
    assert "EMPLOYMENTHERO_OAUTH_CLIENT_ID" in pkg["required_secrets"]
    assert "EMPLOYMENTHERO_OAUTH_CLIENT_SECRET" in pkg["required_secrets"]
    # OAuth-managed + override secrets are optional.
    assert "EMPLOYMENTHERO_REFRESH_TOKEN" in pkg["optional_secrets"]


def test_get_package_for_slug_returns_none_for_unknown():
    assert D.get_package_for_slug("definitely_not_real") is None


def test_caching_does_not_double_scan(monkeypatch):
    """Second call should not hit disk again.  We verify this by patching
    the inner reader to count calls."""
    calls = {"n": 0}

    def _counting_reader():
        calls["n"] += 1
        return [{"slug": "fake"}]

    monkeypatch.setattr(D, "_read_packages_from_disk", _counting_reader)
    D.reset_discovery_cache()

    D.discover_available_packages()
    D.discover_available_packages()
    D.discover_available_packages()

    assert calls["n"] == 1


def test_force_reload_re_reads(monkeypatch):
    calls = {"n": 0}

    def _counting_reader():
        calls["n"] += 1
        return [{"slug": "fake"}]

    monkeypatch.setattr(D, "_read_packages_from_disk", _counting_reader)
    D.reset_discovery_cache()

    D.discover_available_packages()
    D.discover_available_packages(force_reload=True)

    assert calls["n"] == 2
