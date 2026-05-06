"""Unit tests for the hot-load path in ``unity.integration_status``.

Hot-load registers an available package's functions, guidance, and registry
row into the running managers in a daemon thread, so a token paste in
Console takes effect on the next message without a deployment cycle.

These tests stub out the heavyweight pieces (FunctionManager,
GuidanceManager, unify backend) so we can verify the orchestration logic in
isolation.  Live-context tests live separately.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

from unity import integration_status as IS
from unity.integration_status import discovery as IS_DISCOVERY

# ---------------------------------------------------------------------------
# Fixtures: synthetic discovery rows
# ---------------------------------------------------------------------------


def _hubspot_discovery_row(*, root: Path | None = None) -> dict:
    return {
        "slug": "hubspot",
        "label": "HubSpot",
        "category": "crm",
        "version": "0.1.0",
        "tier": "api",
        "root_dir": root or Path("/tmp/fake/hubspot"),
        "required_secrets": ["HUBSPOT_PRIVATE_APP_TOKEN"],
        "optional_secrets": ["HUBSPOT_PORTAL_ID"],
        "function_names": ["get_hubspot_contact", "search_hubspot_contacts"],
        "guidance_titles": ["Hubspot Overview", "Hubspot Crm Contacts"],
        "function_dir": (
            (root or Path("/tmp/fake/hubspot")) / "functions" if root else None
        ),
        "guidance_dir": (
            (root or Path("/tmp/fake/hubspot")) / "guidance" if root else None
        ),
    }


def _eh_discovery_row() -> dict:
    return {
        "slug": "employment_hero",
        "label": "Employment Hero",
        "category": "human-resources",
        "version": "0.1.0",
        "tier": "api",
        "root_dir": Path("/tmp/fake/eh"),
        "required_secrets": [
            "EMPLOYMENTHERO_OAUTH_CLIENT_ID",
            "EMPLOYMENTHERO_OAUTH_CLIENT_SECRET",
        ],
        "optional_secrets": [
            "EMPLOYMENTHERO_REFRESH_TOKEN",
            "EMPLOYMENTHERO_ORGANISATION_ID",
        ],
        "function_names": ["get_employmenthero_employee"],
        "guidance_titles": ["Employmenthero Overview"],
        "function_dir": None,
        "guidance_dir": None,
    }


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch):
    """Clear cache + discovery between tests so they don't leak state."""
    IS.reset_session_cache()
    IS_DISCOVERY.reset_discovery_cache()
    yield
    IS.reset_session_cache()
    IS_DISCOVERY.reset_discovery_cache()


def _patch_discovery(monkeypatch, rows: list[dict]) -> None:
    monkeypatch.setattr(
        IS_DISCOVERY,
        "discover_available_packages",
        lambda *, force_reload=False: rows,
    )


# ---------------------------------------------------------------------------
# all_known_secret_names: now unions registry + discovery
# ---------------------------------------------------------------------------


def test_all_known_secret_names_includes_discovered_when_registry_empty(monkeypatch):
    """When the persisted registry is empty (deployment didn't declare any
    integrations) but disk has packages, allowlist still includes their
    secrets — that's what bridges the token-paste-in-Console flow."""
    _patch_discovery(monkeypatch, [_hubspot_discovery_row(), _eh_discovery_row()])

    # Force _load_registry to return empty (mimics no persisted registry).
    monkeypatch.setattr(IS, "_read_persisted_registry", lambda: [])

    names = IS.all_known_secret_names()
    assert "HUBSPOT_PRIVATE_APP_TOKEN" in names
    assert "HUBSPOT_PORTAL_ID" in names
    assert "EMPLOYMENTHERO_OAUTH_CLIENT_ID" in names
    assert "EMPLOYMENTHERO_OAUTH_CLIENT_SECRET" in names
    assert "EMPLOYMENTHERO_REFRESH_TOKEN" in names


def test_all_known_secret_names_empty_when_no_registry_no_discovery(monkeypatch):
    monkeypatch.setattr(IS, "_read_persisted_registry", lambda: [])
    _patch_discovery(monkeypatch, [])
    assert IS.all_known_secret_names() == set()


# ---------------------------------------------------------------------------
# Synthesized registry rows: recompute_enablement works against discovery
# even when the persisted registry is empty
# ---------------------------------------------------------------------------


def test_recompute_works_against_discovery_when_persisted_registry_empty(monkeypatch):
    """The whole point: an assistant whose deployment didn't declare HubSpot
    can still detect HubSpot as enabled once a token is added to Secrets."""
    _patch_discovery(monkeypatch, [_hubspot_discovery_row()])
    monkeypatch.setattr(IS, "_read_persisted_registry", lambda: [])
    # Stub out schedule_hot_load so the test doesn't spawn a thread.
    spawned: list[str] = []
    monkeypatch.setattr(IS, "schedule_hot_load", lambda slug: spawned.append(slug))

    IS.recompute_enablement(
        assistant_id=42,
        secrets={"HUBSPOT_PRIVATE_APP_TOKEN": "tok"},
    )

    assert IS.get_enabled_integrations() == ["hubspot"]
    assert spawned == ["hubspot"]


def test_recompute_does_not_schedule_when_no_required_present(monkeypatch):
    _patch_discovery(monkeypatch, [_hubspot_discovery_row()])
    monkeypatch.setattr(IS, "_read_persisted_registry", lambda: [])
    spawned: list[str] = []
    monkeypatch.setattr(IS, "schedule_hot_load", lambda slug: spawned.append(slug))

    IS.recompute_enablement(assistant_id=42, secrets={})

    assert IS.get_enabled_integrations() == []
    assert spawned == []


def test_recompute_does_not_re_schedule_when_already_loaded(monkeypatch):
    """Once a slug is in ``loaded_slugs`` (a previous hot-load completed),
    recompute_enablement must not queue another load for the same slug."""
    _patch_discovery(monkeypatch, [_hubspot_discovery_row()])
    monkeypatch.setattr(IS, "_read_persisted_registry", lambda: [])
    spawned: list[str] = []
    monkeypatch.setattr(IS, "schedule_hot_load", lambda slug: spawned.append(slug))

    cache = IS._session_cache()
    cache["loaded_slugs"] = {"hubspot"}

    IS.recompute_enablement(
        assistant_id=42,
        secrets={"HUBSPOT_PRIVATE_APP_TOKEN": "tok"},
    )

    assert spawned == []


# ---------------------------------------------------------------------------
# schedule_hot_load: non-blocking + idempotent
# ---------------------------------------------------------------------------


def test_schedule_hot_load_returns_immediately(monkeypatch):
    """schedule_hot_load spawns a daemon thread; the calling thread must not
    wait for the load to finish.  We use a deliberately slow worker to assert
    schedule_hot_load returns well before the load does."""
    barrier_seconds = 5.0

    def _slow_hot_load(slug: str):
        time.sleep(barrier_seconds)

    async def _slow_async(slug: str):
        time.sleep(barrier_seconds)

    monkeypatch.setattr(IS, "hot_load_integration", _slow_async)

    started = time.perf_counter()
    IS.schedule_hot_load("hubspot")
    elapsed = time.perf_counter() - started

    # Schedule should return in well under 1s; the worker is sleeping for 5s.
    assert elapsed < 1.0, f"schedule_hot_load took {elapsed:.2f}s — should be ~0s"


def test_schedule_hot_load_is_idempotent_for_loaded_slug(monkeypatch):
    """If the slug is already in ``loaded_slugs``, schedule must skip the
    thread spawn entirely."""
    threads_spawned: list[str] = []

    real_thread_cls = IS.__dict__.get("threading", None)

    import threading as _threading

    monkeypatch.setattr(
        _threading,
        "Thread",
        lambda *a, **k: pytest.fail("Should not spawn a thread"),
    )

    cache = IS._session_cache()
    cache["loaded_slugs"] = {"hubspot"}

    # Should be a no-op.
    IS.schedule_hot_load("hubspot")


def test_schedule_hot_load_skips_when_already_loading(monkeypatch):
    """Same as above but for the in-flight guard."""
    cache = IS._session_cache()
    cache["loading_slugs"] = {"hubspot"}

    import threading as _threading

    monkeypatch.setattr(
        _threading,
        "Thread",
        lambda *a, **k: pytest.fail("Should not spawn a thread"),
    )

    IS.schedule_hot_load("hubspot")


def test_schedule_hot_load_failure_does_not_propagate(monkeypatch):
    """Any error inside the worker (asyncio.run, etc.) must be logged and
    swallowed; the calling thread must not see it."""

    async def _raises(slug):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(IS, "hot_load_integration", _raises)

    # Should not raise.  Wait a short time for the worker to fail.
    IS.schedule_hot_load("hubspot")
    time.sleep(0.2)

    # ``loading_slugs`` cleanup runs in the worker's finally block.
    cache = IS._session_cache()
    assert "hubspot" not in cache.get("loading_slugs", set())
    # ``loaded_slugs`` is NOT updated on failure → next recompute will retry.
    assert "hubspot" not in cache.get("loaded_slugs", set())


# ---------------------------------------------------------------------------
# hot_load_integration: orchestration with stubbed sub-steps
# ---------------------------------------------------------------------------


def test_hot_load_invokes_all_three_substeps_for_known_slug(monkeypatch):
    pkg = _hubspot_discovery_row()
    monkeypatch.setattr(IS_DISCOVERY, "get_package_for_slug", lambda slug: pkg)

    calls: list[str] = []
    monkeypatch.setattr(IS, "_hot_load_guidance", lambda p: calls.append("guidance"))
    monkeypatch.setattr(
        IS,
        "_hot_load_functions",
        lambda p: calls.append("functions") or 2,
    )
    monkeypatch.setattr(
        IS,
        "_hot_load_registry_row",
        lambda p: calls.append("registry"),
    )

    import asyncio

    asyncio.run(IS.hot_load_integration("hubspot"))

    assert calls == ["guidance", "functions", "registry"]
    cache = IS._session_cache()
    assert "hubspot" in cache.get("loaded_slugs", set())


def test_hot_load_skips_when_already_loaded(monkeypatch):
    monkeypatch.setattr(
        IS_DISCOVERY,
        "get_package_for_slug",
        lambda slug: pytest.fail("Should not consult discovery"),
    )

    cache = IS._session_cache()
    cache["loaded_slugs"] = {"hubspot"}

    import asyncio

    asyncio.run(IS.hot_load_integration("hubspot"))


def test_hot_load_no_op_for_unknown_slug(monkeypatch, caplog):
    monkeypatch.setattr(IS_DISCOVERY, "get_package_for_slug", lambda slug: None)

    import asyncio

    with caplog.at_level(logging.WARNING):
        asyncio.run(IS.hot_load_integration("nonexistent"))

    cache = IS._session_cache()
    assert "nonexistent" not in cache.get("loaded_slugs", set())


def test_hot_load_partial_failure_still_marks_loaded(monkeypatch):
    """If one sub-step fails (e.g. guidance) but others succeed, the slug
    is still marked loaded — partial success is better than re-attempting
    the same failure on every recompute."""
    pkg = _hubspot_discovery_row()
    monkeypatch.setattr(IS_DISCOVERY, "get_package_for_slug", lambda slug: pkg)

    def _explode(p):
        raise RuntimeError("guidance step blew up")

    monkeypatch.setattr(IS, "_hot_load_guidance", _explode)
    monkeypatch.setattr(IS, "_hot_load_functions", lambda p: 0)
    monkeypatch.setattr(IS, "_hot_load_registry_row", lambda p: None)

    import asyncio

    asyncio.run(IS.hot_load_integration("hubspot"))

    cache = IS._session_cache()
    assert "hubspot" in cache.get("loaded_slugs", set())


def test_hot_load_invalidates_registry_cache(monkeypatch):
    """After a successful hot-load, the persisted registry has a fresh row
    that subsequent allowlist computations should see.  Verify the cache
    flag is reset so ``_load_registry`` re-reads."""
    pkg = _hubspot_discovery_row()
    monkeypatch.setattr(IS_DISCOVERY, "get_package_for_slug", lambda slug: pkg)
    monkeypatch.setattr(IS, "_hot_load_guidance", lambda p: 0)
    monkeypatch.setattr(IS, "_hot_load_functions", lambda p: 0)
    monkeypatch.setattr(IS, "_hot_load_registry_row", lambda p: None)

    cache = IS._session_cache()
    cache["registry_loaded"] = True
    cache["registry"] = [{"slug": "stale"}]

    import asyncio

    asyncio.run(IS.hot_load_integration("hubspot"))

    assert cache["registry_loaded"] is False
    assert cache["registry"] == []
