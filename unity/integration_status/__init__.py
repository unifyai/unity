"""Per-session detection of which integrations are enabled for an assistant.

The deploy side (``unity_deploy.assistant_deployments.integrations``) seeds a
flat row per integration into the ``Integrations/Manifests`` DataManager
context.  Each row carries the integration's ``required_secrets``,
``optional_secrets``, ``function_names``, and ``guidance_titles`` (all
JSON-stringified).

At runtime the assistant has its own secret keyset (synced from Orchestra by
``SecretManager._sync_assistant_secrets``).  An integration is *enabled* iff
every secret listed in ``required_secrets_json`` is present in that keyset.
Once we know which integrations are enabled we can:

* Inject an integration-status block into the system prompt so the LLM knows
  what's available + what setup steps remain.
* Set ``GuidanceManager.filter_scope`` to a guidance-id predicate that hides
  guidance for integrations whose credentials aren't configured.
* Drive the per-session cache that ``recompute_enablement`` updates from
  every ``_sync_assistant_secrets`` call.

This module is purely a reader of the registry + secret keyset; it never
writes.  All work is best-effort: if the registry hasn't been seeded (e.g.
first deploy after PR A1 lands but before A2 ships), every helper here
returns an empty/no-op result and the caller falls through to existing
behaviour.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_REGISTRY_CONTEXT_LEAF = "Integrations/Manifests"


# ---------------------------------------------------------------------------
# Per-body session cache
# ---------------------------------------------------------------------------
#
# The cache lives on ``SESSION_DETAILS`` so that mid-session changes (a user
# pasting a new token via Settings → Secrets, then ``_sync_assistant_secrets``
# firing again) propagate to subsequent prompt builds + guidance retrievals
# without a session restart.
#
# We attach a single dict via ``setattr`` rather than a typed field so we can
# evolve the shape without coordinating a unity-side schema change.

_SESSION_CACHE_ATTR = "_integration_status_cache"


def _session_cache() -> dict[str, Any]:
    """Lazily attach + return the per-body integration-status cache.

    The cache shape:

        {
            "registry": [<row>, ...]            # full Integrations/Manifests rows
            "registry_loaded": bool             # have we tried at least once?
            "enabled": {<slug>: <row>}          # only enabled integrations
            "completeness": {<slug>: {...}}     # configured vs fully_connected per slug
            "secret_names": set[str]            # last-observed assistant keyset
        }

    Falls back to a process-local module dict when ``SESSION_DETAILS`` isn't
    available (test paths that exercise this helper in isolation).
    """
    try:
        from unity.session_details import SESSION_DETAILS

        cache = getattr(SESSION_DETAILS, _SESSION_CACHE_ATTR, None)
        if cache is None:
            cache = _empty_cache()
            setattr(SESSION_DETAILS, _SESSION_CACHE_ATTR, cache)
        return cache
    except Exception:
        # Fallback for environments where SESSION_DETAILS isn't constructed yet.
        if not hasattr(_session_cache, "_fallback"):
            _session_cache._fallback = _empty_cache()
        return _session_cache._fallback  # type: ignore[attr-defined]


def _empty_cache() -> dict[str, Any]:
    return {
        "registry": [],
        "registry_loaded": False,
        "enabled": {},
        "completeness": {},
        "secret_names": set(),
        # Hot-load tracking.  ``loaded_slugs`` is set after a successful
        # hot-load completes for this body; ``loading_slugs`` is the set of
        # slugs currently being loaded by a daemon thread.  ``schedule_hot_load``
        # is a no-op when the slug is in either set.
        "loaded_slugs": set(),
        "loading_slugs": set(),
    }


def reset_session_cache() -> None:
    """Drop the cached enablement state for the current session.

    Used by tests to isolate scenarios; production code shouldn't need this."""
    cache = _session_cache()
    cache.update(_empty_cache())


# ---------------------------------------------------------------------------
# Registry lookup
# ---------------------------------------------------------------------------


def _load_registry() -> list[dict[str, Any]]:
    """Pull rows from ``Integrations/Manifests`` once per session, cache.

    The persisted registry only contains rows for integrations that were
    declared in the deployment spec (``integrations=[...]``).  An assistant
    may have credentials for *available* packages that weren't declared —
    the typical case is a manual token paste in Console.  To bridge that
    gap we synthesize in-memory rows from disk discovery when the persisted
    registry is empty.  See :mod:`unity.integration_status.discovery`.
    """
    cache = _session_cache()
    if cache["registry_loaded"]:
        return cache["registry"]

    rows = _read_persisted_registry()
    if not rows:
        rows = _synthesize_rows_from_discovery()
    cache["registry_loaded"] = True
    cache["registry"] = rows
    return rows


def _read_persisted_registry() -> list[dict[str, Any]]:
    try:
        import unify

        active = unify.get_active_context()["read"]
        ctx = f"{active}/{_REGISTRY_CONTEXT_LEAF}"
        logs = unify.get_logs(context=ctx, limit=1000)
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for log in logs or []:
        entries = dict(log.entries or {})
        if entries.get("slug"):
            rows.append(entries)
    return rows


def _synthesize_rows_from_discovery() -> list[dict[str, Any]]:
    """Project disk-discovered packages into registry-row shape.

    These rows live only in the per-body cache; they're never persisted by
    this module.  After a successful hot-load, the persisted registry gets
    its real row written by ``hot_load_integration``."""
    try:
        from unity.integration_status.discovery import discover_available_packages
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for pkg in discover_available_packages():
        rows.append(
            {
                "slug": pkg["slug"],
                "label": pkg["label"],
                "category": pkg.get("category", ""),
                "version": pkg.get("version", ""),
                "tier": pkg.get("tier", ""),
                "required_secrets_json": json.dumps(pkg.get("required_secrets", [])),
                "optional_secrets_json": json.dumps(pkg.get("optional_secrets", [])),
                "function_names_json": json.dumps(pkg.get("function_names", [])),
                "guidance_titles_json": json.dumps(pkg.get("guidance_titles", [])),
                "_synthesized": True,
            },
        )
    return rows


def _row_required(row: dict) -> set[str]:
    return set(json.loads(row.get("required_secrets_json") or "[]"))


def _row_optional(row: dict) -> set[str]:
    return set(json.loads(row.get("optional_secrets_json") or "[]"))


def _row_function_names(row: dict) -> list[str]:
    return list(json.loads(row.get("function_names_json") or "[]"))


def _row_guidance_titles(row: dict) -> list[str]:
    return list(json.loads(row.get("guidance_titles_json") or "[]"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recompute_enablement(*, assistant_id: int, secrets: dict[str, Any]) -> None:
    """Recompute the enabled set + setup-completeness from the secrets dict.

    Called from ``SecretManager._sync_assistant_secrets`` after the dict
    lands.  Idempotent.  Cheap (~10ms on first call when registry needs
    fetching, ~1ms thereafter — pure dict ops over the cached registry).

    Treats values as "present" iff ``isinstance(v, str) and v != ""``.  This
    matches how Orchestra returns secrets: pasting an empty string in the
    Console produces a ``""`` entry that we should treat as not-present.
    """
    cache = _session_cache()
    keyset = {k for k, v in (secrets or {}).items() if isinstance(v, str) and v}
    cache["secret_names"] = keyset

    registry = _load_registry()
    if not registry:
        cache["enabled"] = {}
        cache["completeness"] = {}
        return

    enabled: dict[str, dict] = {}
    completeness: dict[str, dict] = {}
    for row in registry:
        slug = row.get("slug")
        if not slug:
            continue
        required = _row_required(row)
        optional = _row_optional(row)

        if not required.issubset(keyset):
            # Required secrets missing → not enabled.  We still surface this
            # row in the prompt's "inactive" section via _render_status_block,
            # which reads the registry directly so it can describe what's
            # missing in plain English.
            continue

        enabled[slug] = row
        missing_optional = sorted(optional - keyset)
        completeness[slug] = {
            "status": "fully_connected" if not missing_optional else "configured",
            "missing_optional_secrets": missing_optional,
            "missing_required_secrets": [],  # by definition empty when enabled
        }

    # Detect newly-enabled integrations (in current set but not yet hot-loaded)
    # and queue a non-blocking background load for each.  schedule_hot_load
    # spawns a daemon thread so the calling sync path returns immediately.
    just_enabled = set(enabled.keys()) - cache.get("loaded_slugs", set())

    cache["enabled"] = enabled
    cache["completeness"] = completeness

    for slug in just_enabled:
        try:
            schedule_hot_load(slug)
        except Exception:
            logger.exception("Failed to schedule hot-load for %s", slug)


def get_enabled_integrations(_assistant_id: int | None = None) -> list[str]:
    """Return slugs of integrations whose required secrets are all configured.

    Reads the session cache populated by ``recompute_enablement``.  Returns
    ``[]`` when no recompute has happened yet OR when the registry hasn't
    been seeded.
    """
    return list(_session_cache().get("enabled", {}).keys())


def get_setup_completeness(_assistant_id: int | None = None) -> dict[str, dict]:
    """Per enabled integration, return ``{status, missing_optional_secrets,
    missing_required_secrets}``.  Only enabled integrations appear here;
    inactive ones are described separately in the prompt block."""
    return dict(_session_cache().get("completeness", {}))


def enabled_function_ids(_assistant_id: int | None = None) -> set[int]:
    """Resolve the enabled integrations' function names → FunctionManager ids.

    Returns an empty set when nothing is enabled OR when the FunctionManager
    isn't available; callers should treat empty as "don't filter" so the
    no-registry fallback preserves existing behaviour.
    """
    enabled = _session_cache().get("enabled", {})
    if not enabled:
        return set()

    function_names: set[str] = set()
    for row in enabled.values():
        function_names.update(_row_function_names(row))

    if not function_names:
        return set()

    try:
        from unity.manager_registry import ManagerRegistry

        fm = ManagerRegistry.get_function_manager()
    except Exception:
        return set()

    quoted = ", ".join(repr(n) for n in sorted(function_names))
    try:
        rows = fm.filter(filter=f"name in ({quoted})", limit=10000)
    except Exception:
        return set()

    ids: set[int] = set()
    for r in rows or []:
        fid = (
            r.get("function_id")
            if isinstance(r, dict)
            else getattr(r, "function_id", None)
        )
        if isinstance(fid, int):
            ids.add(fid)
    return ids


def enabled_guidance_ids(_assistant_id: int | None = None) -> set[int]:
    """Resolve enabled integrations' guidance titles → GuidanceManager ids.

    Used to set ``GuidanceManager.filter_scope`` to a guidance-id predicate
    that hides entries belonging to disabled integrations.  Returns an empty
    set when the registry is empty or the GuidanceManager isn't available."""
    enabled = _session_cache().get("enabled", {})
    if not enabled:
        return set()

    titles: set[str] = set()
    for row in enabled.values():
        titles.update(_row_guidance_titles(row))

    if not titles:
        return set()

    try:
        from unity.manager_registry import ManagerRegistry

        gm = ManagerRegistry.get_guidance_manager()
    except Exception:
        return set()

    quoted = ", ".join(repr(t) for t in sorted(titles))
    try:
        rows = gm.filter(filter=f"title in ({quoted})", limit=10000)
    except Exception:
        return set()

    ids: set[int] = set()
    for r in rows or []:
        gid = (
            r.guidance_id
            if hasattr(r, "guidance_id")
            else (r.get("guidance_id") if isinstance(r, dict) else None)
        )
        if isinstance(gid, int):
            ids.add(gid)
    return ids


def build_guidance_filter_scope(_assistant_id: int | None = None) -> str | None:
    """Return a ``guidance_id in (...)`` filter scope, or ``None`` to disable.

    Convention: an empty registry (no detection possible) or zero enabled
    integrations means we don't gate retrieval — return ``None`` so existing
    behaviour is preserved.  Actively-enabled integrations with resolved
    guidance ids produce a positive filter."""
    cache = _session_cache()
    if not cache.get("registry"):
        return None
    if not cache.get("enabled"):
        # Registry seeded but no integration enabled → hide all integration
        # guidance.  Use a filter that matches nothing.
        return "guidance_id in ()"
    ids = enabled_guidance_ids()
    if not ids:
        return "guidance_id in ()"
    return "guidance_id in (" + ", ".join(str(i) for i in sorted(ids)) + ")"


def enabled_summary_for_prompt(_assistant_id: int | None = None) -> str:
    """Render the system-prompt status block.

    Re-rendered per turn (cheap; reads only the session cache).  Format::

        ### Integrations

        Active integrations:
        - HubSpot (fully_connected)
        - Employment Hero (configured — OAuth Connect not complete; missing
          EMPLOYMENTHERO_REFRESH_TOKEN, EMPLOYMENTHERO_ORGANISATION_ID.
          Tell the user to click Connect in Settings → Integrations.)

        Inactive (token not configured):
        - Salesforce — needs SALESFORCE_CLIENT_ID, SALESFORCE_CLIENT_SECRET.

    Returns an empty string when there's nothing to say (no registry, or no
    integrations defined for this deployment).
    """
    cache = _session_cache()
    registry = cache.get("registry") or []
    if not registry:
        return ""

    enabled = cache.get("enabled", {})
    completeness = cache.get("completeness", {})
    keyset = cache.get("secret_names", set())

    active_lines: list[str] = []
    inactive_lines: list[str] = []

    for row in sorted(registry, key=lambda r: r.get("slug", "")):
        slug = row.get("slug")
        label = row.get("label") or slug
        if slug in enabled:
            comp = completeness.get(slug, {})
            status = comp.get("status", "configured")
            missing_opt = comp.get("missing_optional_secrets", []) or []
            if status == "fully_connected":
                active_lines.append(f"- {label} (fully_connected)")
            else:
                missing_str = (
                    ", ".join(missing_opt) if missing_opt else "(see manifest)"
                )
                active_lines.append(
                    f"- {label} (configured — setup incomplete; missing optional "
                    f"secrets: {missing_str}.  Suggest the user complete the "
                    f"Connect step in Settings → Integrations if applicable.)",
                )
        else:
            required = _row_required(row)
            missing_required = sorted(required - keyset)
            missing_str = (
                ", ".join(missing_required) if missing_required else "(see manifest)"
            )
            inactive_lines.append(
                f"- {label} — needs {missing_str}.",
            )

    if not active_lines and not inactive_lines:
        return ""

    parts: list[str] = ["### Integrations"]
    if active_lines:
        parts.append("Active integrations:\n" + "\n".join(active_lines))
    else:
        parts.append("Active integrations: (none configured)")
    if inactive_lines:
        parts.append(
            "Inactive (credentials not configured):\n" + "\n".join(inactive_lines),
        )
    parts.append(
        "When a user asks about an inactive integration, tell them which "
        "secrets to add in Settings → Secrets and don't attempt to call its "
        "functions.",
    )
    return "\n\n".join(parts)


def all_known_secret_names() -> set[str]:
    """Return the union of every secret name (required + optional) declared
    by any integration in the registry OR available on disk.

    Used by ``SecretManager._sync_assistant_secrets`` as a registry-derived
    replacement for the hardcoded ``OAUTH_SECRET_ALLOWLIST``.  Includes
    discovery so that secrets for *available* (not-yet-loaded) packages are
    allowlisted too — this is what makes a token paste in Console actually
    sync from Orchestra without requiring a deployment-time declaration."""
    out: set[str] = set()

    # Persisted-registry rows (covers integrations the deployment declared).
    registry = _load_registry()
    for row in registry or []:
        out |= _row_required(row)
        out |= _row_optional(row)

    # Disk-discovered packages (covers available-but-not-declared ones).
    try:
        from unity.integration_status.discovery import discover_available_packages

        for pkg in discover_available_packages():
            out |= set(pkg.get("required_secrets", []))
            out |= set(pkg.get("optional_secrets", []))
    except Exception:
        # Best-effort; unity_deploy not importable in the running env.
        pass

    return out


# ---------------------------------------------------------------------------
# Hot-load: register an available package's functions + guidance + registry
# row into the running managers, in the background.
# ---------------------------------------------------------------------------


def schedule_hot_load(slug: str) -> None:
    """Sync-safe entry point: register the package for ``slug`` in the
    background.  Returns immediately.

    Spawns a daemon thread that runs ``asyncio.run(hot_load_integration(slug))``
    so it works from any caller — sync or async, with or without a running
    event loop (constructor paths in ``SecretManager`` run before the
    assistant's main loop exists).

    Idempotent: a no-op when the slug is already loaded for this session
    or already being loaded by an in-flight thread.  Errors during the
    background load are logged but never propagate to the caller."""
    import threading

    cache = _session_cache()
    if slug in cache.setdefault("loaded_slugs", set()):
        return
    if slug in cache.setdefault("loading_slugs", set()):
        return

    cache["loading_slugs"].add(slug)

    def _worker() -> None:
        try:
            import asyncio

            asyncio.run(hot_load_integration(slug))
        except Exception:
            logger.exception("Hot-load worker failed for slug=%s", slug)
        finally:
            try:
                cache["loading_slugs"].discard(slug)
            except Exception:
                pass

    thread = threading.Thread(
        target=_worker,
        daemon=True,
        name=f"integration-hot-load-{slug}",
    )
    thread.start()


async def hot_load_integration(slug: str) -> None:
    """Load a package's functions + guidance + registry row.

    Per-step idempotent: re-running for the same slug is a no-op for any
    sub-step that's already complete.  Each step runs in its own try/except
    so a partial failure (e.g. one bad function file) doesn't abort the rest.
    """
    cache = _session_cache()

    if slug in cache.get("loaded_slugs", set()):
        return

    try:
        from unity.integration_status.discovery import get_package_for_slug
    except Exception:
        return

    pkg = get_package_for_slug(slug)
    if pkg is None:
        logger.warning("hot_load_integration: package %r not on disk", slug)
        return

    try:
        _hot_load_guidance(pkg)
    except Exception:
        logger.exception("hot_load_integration: guidance step failed for %s", slug)

    try:
        functions_added = _hot_load_functions(pkg)
    except Exception:
        logger.exception("hot_load_integration: functions step failed for %s", slug)
        functions_added = 0

    try:
        _hot_load_registry_row(pkg)
    except Exception:
        logger.exception("hot_load_integration: registry-row step failed for %s", slug)

    cache.setdefault("loaded_slugs", set()).add(slug)

    # Invalidate the registry cache so subsequent ``_load_registry`` calls
    # see the persisted row we just wrote (and stop returning the synthesized
    # variant).
    cache["registry_loaded"] = False
    cache["registry"] = []

    logger.info(
        "Hot-loaded integration slug=%s functions_added=%d guidance=%d",
        slug,
        functions_added,
        len(pkg.get("guidance_titles", [])),
    )


def _hot_load_functions(pkg: dict) -> int:
    """Add a package's @custom_function callables to FunctionManager.

    Uses per-name insert/update (no orphan-delete pass) so we don't disturb
    functions registered by the deployment's own ``function_dirs``.  Returns
    the number of functions actually added or updated.
    """
    function_dir = pkg.get("function_dir")
    if function_dir is None:
        return 0

    from unity.function_manager.custom_functions import collect_custom_functions
    from unity.manager_registry import ManagerRegistry

    source_fns = collect_custom_functions(directory=function_dir)
    if not source_fns:
        return 0

    fm = ManagerRegistry.get_function_manager()
    db_fns = fm._get_custom_functions_from_db()

    changed = 0
    for name, source_data in source_fns.items():
        try:
            # ``custom_functions`` retains ``venv_name`` for the deploy-time
            # sync path that resolves it via the venv catalog.  At hot-load
            # time we don't manage venvs (none of the in-tree integration
            # packages declare one); strip the key so the FM insert API
            # doesn't choke on it.
            source_data = {k: v for k, v in source_data.items() if k != "venv_name"}
            if name in db_fns:
                if db_fns[name].get("custom_hash") != source_data.get("custom_hash"):
                    fm._update_custom_function(
                        function_id=db_fns[name]["function_id"],
                        data=source_data,
                    )
                    changed += 1
            else:
                fm._insert_custom_function(source_data)
                changed += 1
        except Exception:
            logger.exception("Failed to register function %s", name)
    return changed


def _hot_load_guidance(pkg: dict) -> int:
    """Add a package's guidance markdown entries to GuidanceManager.

    Per-title check + insert.  Doesn't update an existing entry's content
    even if the markdown on disk has changed — that's a deploy-time concern
    (``_sync_guidance`` handles drift via SeedMetaStore).  Returns the
    number of new entries added.
    """
    guidance_dir = pkg.get("guidance_dir")
    if guidance_dir is None:
        return 0

    try:
        from unity_deploy.assistant_deployments.integrations.loader import (
            _load_guidance,
        )
    except Exception:
        return 0

    from unity.manager_registry import ManagerRegistry

    entries = _load_guidance(guidance_dir)
    if not entries:
        return 0

    gm = ManagerRegistry.get_guidance_manager()
    added = 0
    for entry in entries:
        try:
            existing = gm.filter(filter=f"title == {entry.title!r}", limit=1)
            if existing:
                continue
            gm.add_guidance(title=entry.title, content=entry.content)
            added += 1
        except Exception:
            logger.exception("Failed to register guidance %r", entry.title)
    return added


def _hot_load_registry_row(pkg: dict) -> None:
    """Persist the integration's registry row into ``Integrations/Manifests``.

    Idempotent on slug: if a row with this slug already exists, update its
    fields; otherwise insert.  Mirrors what unity-deploy's
    ``_sync_integration_registry`` would write at deploy time.
    """
    try:
        import unify
    except Exception:
        return

    try:
        active = unify.get_active_context()["read"]
        ctx = f"{active}/{_REGISTRY_CONTEXT_LEAF}"
        try:
            unify.create_context(ctx)
        except Exception:
            pass
    except Exception:
        return

    row = {
        "slug": pkg["slug"],
        "label": pkg["label"],
        "category": pkg.get("category", ""),
        "version": pkg.get("version", ""),
        "tier": pkg.get("tier", ""),
        "required_secrets_json": json.dumps(pkg.get("required_secrets", [])),
        "optional_secrets_json": json.dumps(pkg.get("optional_secrets", [])),
        "function_names_json": json.dumps(pkg.get("function_names", [])),
        "guidance_titles_json": json.dumps(pkg.get("guidance_titles", [])),
    }

    try:
        existing = unify.get_logs(
            context=ctx,
            filter=f"slug == {pkg['slug']!r}",
            limit=1,
        )
    except Exception:
        existing = []

    if existing:
        try:
            unify.update_logs(
                logs=[existing[0].id],
                context=ctx,
                entries=[row],
                overwrite=True,
            )
        except Exception:
            logger.exception("Failed to update registry row for %s", pkg["slug"])
    else:
        try:
            unify.log(context=ctx, **row)
        except Exception:
            logger.exception("Failed to insert registry row for %s", pkg["slug"])


__all__ = [
    "all_known_secret_names",
    "build_guidance_filter_scope",
    "enabled_function_ids",
    "enabled_guidance_ids",
    "enabled_summary_for_prompt",
    "get_enabled_integrations",
    "get_setup_completeness",
    "hot_load_integration",
    "recompute_enablement",
    "reset_session_cache",
    "schedule_hot_load",
]
