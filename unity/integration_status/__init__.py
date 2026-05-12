"""Integration package registration and enablement detection.

Two orthogonal mechanisms, deliberately decoupled from secret transport
(:mod:`unity.secret_manager`).

1. **Registration (startup, mechanism 1).**
   :func:`register_available_integrations` walks every package manifest under
   unity-deploy's package roots and registers each one's functions + guidance
   with the runtime managers.  Synchronous, main-thread, idempotent.  Called
   once from :mod:`unity.__init__` after :meth:`ContextRegistry.setup`.

2. **Enablement (read-only, mechanism 2).**
   :func:`get_enabled_integrations` is a pure query that returns which
   packages have their required secrets satisfied right now.  No caching
   that can go stale, no thread spawning, no manager re-construction.
   Prompt builders / the actor invoke on demand.

**Secret transport is a third, separate concern** owned by
:meth:`SecretManager._sync_assistant_secrets` (Google / Microsoft OAuth
tokens) and :meth:`SecretManager._sync_dotenv` (everything else, including
Console-pasted integration credentials).  This module never touches secret
values; it only reads the local Secrets keyset to decide which integrations
are enabled.

History: the early-May design entangled all three concerns into one sync
flow with daemon-thread hot-load, registry-derived secret allowlists, and
recompute_enablement called from inside SecretManager.__init__.  That
produced cascading bugs (silent integration-secret wipes, recursive
manager construction, daemon-thread context-resolution failures).  This
module is the post-cleanup shape — see ``unity/integration_status``
commit history for the trail.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-body session cache (tiny — only registration idempotency)
# ---------------------------------------------------------------------------

_SESSION_CACHE_ATTR = "_integration_status_cache"


def _session_cache() -> dict[str, Any]:
    """Return the per-body cache dict, attaching to ``SESSION_DETAILS``.

    Falls back to a process-local module dict when ``SESSION_DETAILS``
    isn't constructed yet (test paths that exercise this helper in
    isolation).
    """
    try:
        from unity.session_details import SESSION_DETAILS

        cache = getattr(SESSION_DETAILS, _SESSION_CACHE_ATTR, None)
        if cache is None:
            cache = _empty_cache()
            setattr(SESSION_DETAILS, _SESSION_CACHE_ATTR, cache)
        return cache
    except Exception:
        if not hasattr(_session_cache, "_fallback"):
            _session_cache._fallback = _empty_cache()  # type: ignore[attr-defined]
        return _session_cache._fallback  # type: ignore[attr-defined]


def _empty_cache() -> dict[str, Any]:
    return {
        # Slugs whose functions + guidance have already been registered
        # with the runtime managers this session.  Idempotency guard for
        # ``register_available_integrations`` re-runs.
        "registered_slugs": set(),
    }


def reset_session_cache() -> None:
    """Drop the cache.  Used by tests for isolation; production code
    shouldn't need this."""
    cache = _session_cache()
    cache.update(_empty_cache())


# ---------------------------------------------------------------------------
# Local secret keyset (read once per enablement query)
# ---------------------------------------------------------------------------


def _read_local_secret_keyset() -> set[str]:
    """Names of non-empty secrets in the assistant's local ``Secrets`` context.

    The local Secrets context is the source of truth for both
    Orchestra-mirrored OAuth tokens (Google / MS, written by
    :meth:`SecretManager._sync_assistant_secrets`) AND directly-pasted
    integration tokens (HubSpot / EmploymentHero / Matterport / Webex /
    Salesforce, written by Console).

    Best-effort: returns an empty set if the SecretManager isn't
    available or the context can't be read.  Never raises.
    """
    try:
        import unify
        from unity.manager_registry import ManagerRegistry

        sm = ManagerRegistry.get_secret_manager()
        rows = unify.get_logs(context=sm._ctx)
    except Exception:
        return set()

    keyset: set[str] = set()
    for lg in rows or []:
        try:
            entries = lg.entries or {}
            nm = entries.get("name")
            val = entries.get("value")
            if isinstance(nm, str) and nm and isinstance(val, str) and val:
                keyset.add(nm)
        except Exception:
            continue
    return keyset


# ---------------------------------------------------------------------------
# Mechanism 1 — Startup registration
# ---------------------------------------------------------------------------


def register_available_integrations() -> None:
    """Walk disk packages and register each ENABLED one's functions +
    guidance with the runtime managers.

    **Gated by enablement.**  Only packages whose required secrets are
    present in the local ``/Secrets`` keyset are registered — this
    prevents every package on disk from polluting FunctionManager /
    GuidanceManager for assistants that never opted into them (e.g. an
    assistant whose deployment declares only HubSpot shouldn't pick up
    Matterport, Webex, etc. tools just because their packages happen to
    be on disk).  Packages with no declared required secrets are
    always-on and are registered unconditionally.

    Deployment-declared packages whose secrets aren't pasted yet are
    still loaded by the deploy seed via ``_sync_functions`` /
    ``_sync_guidance``; this register pass is idempotent over those.

    **Synchronous and idempotent.**  In production, callers schedule
    this as a background task via
    :func:`schedule_register_available_integrations` so the fast-brain
    conversation loop can come online without waiting for the
    (potentially many-hundreds-of-ms) inserts to finish.  Direct
    synchronous use is fine for tests and CLI tools.

    Mid-session token paste does **not** auto-register the integration
    today — adding a secret after startup means the package's functions
    won't appear until the next session.  The pre-May ``schedule_hot_load``
    mechanism handled this lazily but at the cost of a daemon-thread
    bug class we removed; lazy mid-session registration can come back
    as a separate, single-thread, debounced helper if the UX gap shows
    up in practice.
    """
    cache = _session_cache()

    try:
        from unity.integration_status.discovery import discover_available_packages
    except Exception:
        logger.warning(
            "[integrations] register: discovery module unimportable; skipping",
            exc_info=True,
        )
        return

    packages = discover_available_packages()
    if not packages:
        logger.info("[integrations] register: no packages discovered on disk")
        return

    # Gate registration on the local secret keyset.  Computed once per
    # call rather than per-package so we don't re-hit the SecretManager
    # context for each disk package.
    keyset = _read_local_secret_keyset()

    total_funcs = 0
    total_guidance = 0
    registered_now: list[str] = []
    skipped_no_secrets: list[str] = []

    for pkg in packages:
        slug = pkg.get("slug") or ""
        if not slug:
            continue
        already_registered = cache.setdefault("registered_slugs", set())
        if slug in already_registered:
            continue

        required = set(pkg.get("required_secrets", []))
        # A package with required_secrets is registered only when the
        # user has configured every one of them.  Packages with NO
        # required secrets (always-on / read-only) are registered
        # unconditionally.
        if required and not required.issubset(keyset):
            skipped_no_secrets.append(slug)
            continue

        try:
            total_funcs += _register_functions(pkg)
        except Exception:
            logger.exception(
                "[integrations] register: functions step failed for %s",
                slug,
            )

        try:
            total_guidance += _register_guidance(pkg)
        except Exception:
            logger.exception(
                "[integrations] register: guidance step failed for %s",
                slug,
            )

        already_registered.add(slug)
        registered_now.append(slug)

    logger.info(
        "[integrations] register: discovered=%d registered=%s "
        "skipped_no_secrets=%s functions=%d guidance=%d",
        len(packages),
        sorted(registered_now),
        sorted(skipped_no_secrets),
        total_funcs,
        total_guidance,
    )


def schedule_register_available_integrations() -> None:
    """Spawn a single daemon thread that runs
    :func:`register_available_integrations` in the background.

    Returns immediately so the calling startup path (typically
    :mod:`unity.__init__`) stays non-blocking — the assistant's
    conversation / communication "fast brain" comes online without
    waiting for function and guidance inserts to finish.  Integration
    functions become callable as soon as the worker thread completes
    (low hundreds of ms typically); turns that arrive earlier will see
    them missing from the FunctionManager and fall through to existing
    no-tool behaviour.

    Safety vs the May-2026 daemon-thread hot-load we removed:

    * Single thread, single registration pass — not per-slug spawned
      from inside ``recompute_enablement`` running inside
      ``SecretManager.__init__``.
    * Caller is :mod:`unity.__init__` after :meth:`ContextRegistry.setup`
      has populated ``_base_context``, so manager constructions inside
      the worker can resolve their contexts.
    * Captures the calling thread's Unify active context and re-applies
      it inside the worker so ``unify.get_logs(...)`` / context-resolving
      reads in the registration path see the same project + context the
      main thread does.

    Best-effort: spawning failures are logged and never propagate.
    """
    import threading

    try:
        import unify

        captured_ctx = unify.get_active_context()
    except Exception:
        captured_ctx = None

    def _worker() -> None:
        if captured_ctx is not None:
            try:
                import unify

                unify.set_context(
                    captured_ctx["read"],
                    skip_create=True,
                )
            except Exception:
                # Worst case the worker reads from a different context
                # than the main thread; log and continue so a partial
                # context-resolution failure doesn't silently block
                # registration entirely.
                logger.warning(
                    "[integrations] register worker: failed to inherit "
                    "Unify active context; proceeding with worker default",
                    exc_info=True,
                )
        try:
            register_available_integrations()
        except Exception:
            logger.exception(
                "[integrations] register worker: registration failed; "
                "integrations may not be available this session",
            )

    thread = threading.Thread(
        target=_worker,
        daemon=True,
        name="integration-register-startup",
    )
    thread.start()


def _register_functions(pkg: dict) -> int:
    """Add a package's ``@custom_function`` callables to FunctionManager.

    Per-name insert/update (no orphan-delete pass) so we don't disturb
    functions registered by the deployment's own ``function_dirs``.
    Returns the number of functions actually added or updated.
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
            # ``custom_functions`` retains ``venv_name`` for the
            # deploy-time sync path that resolves it via the venv
            # catalog.  At register-time we don't manage venvs (none of
            # the in-tree integration packages declare one); strip the
            # key so the FM insert API doesn't choke on it.
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


def _register_guidance(pkg: dict) -> int:
    """Add a package's guidance markdown entries to GuidanceManager.

    Per-title check + insert.  Doesn't update an existing entry's content
    even if the markdown on disk has changed — that's a deploy-time
    concern (``_sync_guidance`` handles drift via SeedMetaStore).
    Returns the number of new entries added.
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


# ---------------------------------------------------------------------------
# Mechanism 2 — Enablement read (pure, on-demand)
# ---------------------------------------------------------------------------


def get_enabled_integrations() -> dict[str, dict]:
    """Return ``{slug: package_metadata}`` for every disk-discovered package
    whose required secrets are currently in the assistant's local
    ``/Secrets`` context.

    Pure function — reads disk discovery (process-cached in
    :mod:`unity.integration_status.discovery`) plus the local Secrets
    keyset.  Costs ~1ms.  No caching that can go stale; callers get fresh
    state every call.

    A package with no required secrets is considered enabled
    unconditionally (rare; useful for read-only or always-on packages).
    """
    try:
        from unity.integration_status.discovery import discover_available_packages
    except Exception:
        return {}

    keyset = _read_local_secret_keyset()
    enabled: dict[str, dict] = {}
    for pkg in discover_available_packages():
        slug = pkg.get("slug") or ""
        if not slug:
            continue
        required = set(pkg.get("required_secrets", []))
        if not required or required.issubset(keyset):
            enabled[slug] = pkg
    return enabled


def get_setup_completeness() -> dict[str, dict]:
    """Per enabled integration, return setup-completeness metadata.

    Returns ``{slug: {status, missing_optional_secrets,
    missing_required_secrets}}``.  ``status`` is ``"fully_connected"`` if
    every required + optional secret is present, ``"configured"`` if some
    optional ones are still missing.  ``missing_required_secrets`` is
    always empty for enabled integrations (by definition).
    """
    keyset = _read_local_secret_keyset()
    out: dict[str, dict] = {}
    for slug, pkg in get_enabled_integrations().items():
        optional = set(pkg.get("optional_secrets", []))
        missing_opt = sorted(optional - keyset)
        out[slug] = {
            "status": "fully_connected" if not missing_opt else "configured",
            "missing_optional_secrets": missing_opt,
            "missing_required_secrets": [],
        }
    return out


def enabled_function_ids() -> set[int]:
    """Resolve enabled integrations' function names → FunctionManager ids.

    Returns an empty set when nothing is enabled OR when the
    FunctionManager isn't available; callers should treat empty as
    "don't filter" so the no-integrations fallback preserves existing
    behaviour.
    """
    enabled = get_enabled_integrations()
    if not enabled:
        return set()

    function_names: set[str] = set()
    for pkg in enabled.values():
        function_names.update(pkg.get("function_names", []))

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


def enabled_guidance_ids() -> set[int]:
    """Resolve enabled integrations' guidance titles → GuidanceManager ids.

    Used to set ``GuidanceManager.filter_scope`` to a guidance-id
    predicate that hides entries belonging to disabled integrations.
    Returns an empty set when nothing is enabled or the GuidanceManager
    isn't available.
    """
    enabled = get_enabled_integrations()
    if not enabled:
        return set()

    titles: set[str] = set()
    for pkg in enabled.values():
        titles.update(pkg.get("guidance_titles", []))

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


def build_guidance_filter_scope() -> str | None:
    """Return a ``guidance_id in (...)`` filter scope, or ``None`` to disable.

    Convention: when there are no integration packages on disk at all,
    return ``None`` so non-integration callers see existing behaviour.
    When packages exist but none are enabled, return a never-matching
    filter so disabled-integration guidance is hidden.  When some are
    enabled, return a positive filter naming their guidance ids.
    """
    try:
        from unity.integration_status.discovery import discover_available_packages

        packages = discover_available_packages()
    except Exception:
        return None

    if not packages:
        return None

    enabled = get_enabled_integrations()
    if not enabled:
        return "guidance_id in ()"

    ids = enabled_guidance_ids()
    if not ids:
        return "guidance_id in ()"
    return "guidance_id in (" + ", ".join(str(i) for i in sorted(ids)) + ")"


def enabled_summary_for_prompt() -> str:
    """Render the system-prompt status block.

    Re-rendered per turn (cheap; reads disk discovery + local keyset).
    Returns an empty string when there are no integration packages on
    disk.  Format::

        ### Integrations

        Active integrations:
        - HubSpot (fully_connected)
        - Employment Hero (configured — setup incomplete; missing optional
          secrets: EMPLOYMENTHERO_REFRESH_TOKEN.  Suggest the user complete
          the Connect step in Settings → Integrations if applicable.)

        Inactive (credentials not configured):
        - Salesforce — needs SALESFORCE_CLIENT_ID, SALESFORCE_CLIENT_SECRET.
    """
    try:
        from unity.integration_status.discovery import discover_available_packages

        packages = discover_available_packages()
    except Exception:
        return ""

    if not packages:
        return ""

    enabled = get_enabled_integrations()
    completeness = get_setup_completeness()
    keyset = _read_local_secret_keyset()

    active_lines: list[str] = []
    inactive_lines: list[str] = []

    for pkg in sorted(packages, key=lambda p: p.get("slug", "")):
        slug = pkg.get("slug")
        label = pkg.get("label") or slug
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
            required = set(pkg.get("required_secrets", []))
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


__all__ = [
    "build_guidance_filter_scope",
    "enabled_function_ids",
    "enabled_guidance_ids",
    "enabled_summary_for_prompt",
    "get_enabled_integrations",
    "get_setup_completeness",
    "register_available_integrations",
    "reset_session_cache",
    "schedule_register_available_integrations",
]
