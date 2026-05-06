"""Runtime-side discovery of integration packages installed on disk.

The deploy side (``unity_deploy.assistant_deployments.integrations``) seeds
*registered* integrations (those declared in a deployment's
``integrations=[...]`` list) into the ``Integrations/Manifests`` DataManager
context.  But assistants can have credentials for *available* packages that
weren't declared on their deployment — that's the gap this module bridges.

At runtime we lazily enumerate every manifest under unity_deploy's package
roots so we can:

* Allowlist secrets for *any* installed package (not just declared ones), so
  ``SecretManager._sync_assistant_secrets`` pulls them from Orchestra.
* Detect when a token for an available-but-not-declared package gets pasted
  into the assistant's secrets, and drive the hot-load that registers the
  package's functions + guidance into the running managers.

Everything is best-effort: if unity_deploy isn't importable on the runtime
container (the only environment where this matters in practice is a unity-only
test env), every helper here returns an empty result and callers fall back to
pre-C behaviour.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cache process-wide.  Discovery is cheap, but allowlist resolution can be
# called many times per session; once is enough.
_DISCOVERY_CACHE: list[dict[str, Any]] | None = None


def discover_available_packages(*, force_reload: bool = False) -> list[dict[str, Any]]:
    """Return one record per integration manifest found on disk.

    Each record::

        {
            "slug": "hubspot",
            "label": "HubSpot",
            "category": "crm",
            "version": "0.1.0",
            "tier": "api",
            "root_dir": Path(...),
            "required_secrets": ["HUBSPOT_PRIVATE_APP_TOKEN"],
            "optional_secrets": ["HUBSPOT_PORTAL_ID"],
            "function_names": [...],
            "guidance_titles": [...],
            "function_dir": Path(...) | None,
            "guidance_dir": Path(...) | None,
        }

    Returns ``[]`` when unity_deploy isn't importable (test envs, broken
    install).  Cached across calls; pass ``force_reload=True`` to invalidate
    (used by tests).
    """
    global _DISCOVERY_CACHE

    if _DISCOVERY_CACHE is not None and not force_reload:
        return _DISCOVERY_CACHE

    rows = _read_packages_from_disk()
    _DISCOVERY_CACHE = rows
    return rows


def _read_packages_from_disk() -> list[dict[str, Any]]:
    try:
        from unity_deploy.assistant_deployments.integrations.discovery import (
            _BUILTIN_DIR,
            _CLIENT_DIR,
        )
        from unity_deploy.assistant_deployments.integrations.discovery import (
            discover_from_directory,
        )
        from unity_deploy.assistant_deployments.integrations.loader import (
            _stem_to_title,
        )
    except Exception:
        logger.debug(
            "unity_deploy package paths not importable from runtime; "
            "discovery returning []",
            exc_info=True,
        )
        return []

    out: list[dict[str, Any]] = []
    for search_root in (_BUILTIN_DIR, _CLIENT_DIR):
        try:
            manifests = discover_from_directory(search_root)
        except Exception:
            logger.warning("Discovery failed for %s", search_root, exc_info=True)
            continue
        for manifest in manifests:
            root = (search_root / manifest.slug).resolve()
            if not root.is_dir():
                # Slug → directory contract from the upstream loader; if it
                # ever drifts, skip rather than crash.
                continue
            row = _project_manifest(manifest, root, _stem_to_title)
            out.append(row)
    return out


def _project_manifest(
    manifest: Any,
    root: Path,
    stem_to_title: Any,
) -> dict[str, Any]:
    """Project a manifest object + its root directory into a discovery row."""
    required = sorted({s.name for s in manifest.secrets if s.required})
    optional = sorted({s.name for s in manifest.secrets if not s.required})

    function_names: set[str] = set()
    guidance_titles: set[str] = set()
    for cap in manifest.capabilities:
        function_names.update(cap.functions)
        guidance_titles.update(stem_to_title(stem) for stem in cap.guidance)

    function_dir = root / "functions"
    guidance_dir = root / "guidance"
    return {
        "slug": manifest.slug,
        "label": manifest.name,
        "category": manifest.sector,
        "version": manifest.version,
        "tier": manifest.tier,
        "root_dir": root,
        "required_secrets": required,
        "optional_secrets": optional,
        "function_names": sorted(function_names),
        "guidance_titles": sorted(guidance_titles),
        "function_dir": function_dir if function_dir.is_dir() else None,
        "guidance_dir": guidance_dir if guidance_dir.is_dir() else None,
    }


def get_package_for_slug(slug: str) -> dict[str, Any] | None:
    """Return the discovery row for ``slug`` or ``None`` if not found."""
    for row in discover_available_packages():
        if row.get("slug") == slug:
            return row
    return None


def reset_discovery_cache() -> None:
    """Drop the process-wide discovery cache.  Tests use this to rebuild
    discovery state without restarting the interpreter."""
    global _DISCOVERY_CACHE
    _DISCOVERY_CACHE = None


__all__ = [
    "discover_available_packages",
    "get_package_for_slug",
    "reset_discovery_cache",
]
