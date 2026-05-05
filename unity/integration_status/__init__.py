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

    Returns ``[]`` when the context isn't seeded yet (e.g. first deploy with
    A2 live but A1 not yet shipped, or a new deployment whose seed sync
    hasn't run).  Callers treat empty as "no detection possible — fall
    through to existing behaviour" rather than "every integration disabled"."""
    cache = _session_cache()
    if cache["registry_loaded"]:
        return cache["registry"]

    try:
        import unify

        active = unify.get_active_context()["read"]
        ctx = f"{active}/{_REGISTRY_CONTEXT_LEAF}"
        logs = unify.get_logs(context=ctx, limit=1000)
    except Exception:
        cache["registry_loaded"] = True
        cache["registry"] = []
        return cache["registry"]

    rows: list[dict[str, Any]] = []
    for log in logs or []:
        entries = dict(log.entries or {})
        if entries.get("slug"):
            rows.append(entries)
    cache["registry_loaded"] = True
    cache["registry"] = rows
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

    cache["enabled"] = enabled
    cache["completeness"] = completeness


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
          EMPLOYMENTHERO_REFRESH_TOKEN, EMPLOYMENTHERO_ORGANISATION_ID,
          EMPLOYMENTHERO_HUB_DOMAIN.  Tell the user to click Connect in
          Settings → Integrations.)

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
    by any integration in the registry.

    Used by ``SecretManager._sync_assistant_secrets`` as a registry-derived
    replacement for the hardcoded ``OAUTH_SECRET_ALLOWLIST``.  Falls back to
    an empty set when the registry isn't seeded; the caller should fall back
    to its hardcoded allowlist in that case so OAuth flows still sync."""
    registry = _load_registry()
    if not registry:
        return set()
    out: set[str] = set()
    for row in registry:
        out |= _row_required(row)
        out |= _row_optional(row)
    return out


__all__ = [
    "all_known_secret_names",
    "build_guidance_filter_scope",
    "enabled_function_ids",
    "enabled_guidance_ids",
    "enabled_summary_for_prompt",
    "get_enabled_integrations",
    "get_setup_completeness",
    "recompute_enablement",
    "reset_session_cache",
]
