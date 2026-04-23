"""
Backend sync utilities for ContactManager.

Provides fire-and-forget sync of system contact fields (timezone, bio/about)
to the orchestra backend User and Assistant profiles.

Endpoints:
- GET /organizations/members (list org members)
- POST /admin/assistant/update-user (user timezone/bio)
- PATCH /admin/assistant/{assistant_id} (assistant timezone/about)
- GET /hives/{hive_id}/assistants (enumerate Hive member bodies)
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from ..settings import SETTINGS

_log = logging.getLogger(__name__)


def _get_base_url() -> str | None:
    """Return base URL or None if not configured."""
    url = SETTINGS.ORCHESTRA_URL
    return url


def _get_admin_key() -> str | None:
    """Return admin key or None if not configured."""
    return SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()


# ─────────────────────────────────────────────────────────────────────────────
# User Profile Sync → POST /admin/assistant/update-user
# ─────────────────────────────────────────────────────────────────────────────


def sync_user_timezone(assistant_id: int, target_email: str, timezone: str) -> bool:
    """
    Fire-and-forget sync of timezone to user profile via assistant lookup.

    Uses POST /admin/assistant/update-user
    """
    base_url = _get_base_url()
    admin_key = _get_admin_key()
    if not base_url or not admin_key or not assistant_id or not target_email:
        _log.debug("Skipping user timezone sync: missing required params")
        return False

    try:
        from unify.utils import http

        url = f"{base_url}/admin/assistant/update-user"
        headers = {"Authorization": f"Bearer {admin_key}"}
        payload = {
            "assistant_id": int(assistant_id),
            "target_user_email": target_email,
            "timezone": timezone,
        }
        resp = http.post(url, headers=headers, json=payload, timeout=30)
        if 200 <= resp.status_code < 300:
            _log.info(f"Synced user timezone to backend: {timezone}")
            return True
        _log.warning(f"Failed to sync user timezone: {resp.status_code} {resp.text}")
        return False
    except Exception as e:
        _log.warning(f"Error syncing user timezone: {e}")
        return False


def sync_user_bio(assistant_id: int, target_email: str, bio: str) -> bool:
    """
    Fire-and-forget sync of bio to user profile via assistant lookup.

    Uses POST /admin/assistant/update-user
    """
    base_url = _get_base_url()
    admin_key = _get_admin_key()
    if not base_url or not admin_key or not assistant_id or not target_email:
        _log.debug("Skipping user bio sync: missing required params")
        return False

    try:
        from unify.utils import http

        url = f"{base_url}/admin/assistant/update-user"
        headers = {"Authorization": f"Bearer {admin_key}"}
        payload = {
            "assistant_id": int(assistant_id),
            "target_user_email": target_email,
            "bio": bio,
        }
        resp = http.post(url, headers=headers, json=payload, timeout=30)
        if 200 <= resp.status_code < 300:
            _log.info("Synced user bio to backend")
            return True
        _log.warning(f"Failed to sync user bio: {resp.status_code} {resp.text}")
        return False
    except Exception as e:
        _log.warning(f"Error syncing user bio: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Assistant Profile Sync → PATCH /admin/assistant/{assistant_id}
# ─────────────────────────────────────────────────────────────────────────────


def sync_assistant_timezone(assistant_id: int, timezone: str) -> bool:
    """
    Fire-and-forget sync of timezone to assistant profile.

    Uses PATCH /admin/assistant/{assistant_id}
    """
    base_url = _get_base_url()
    admin_key = _get_admin_key()
    if not base_url or not admin_key or not assistant_id:
        _log.debug("Skipping assistant timezone sync: missing required params")
        return False

    try:
        from unify.utils import http

        url = f"{base_url}/admin/assistant/{int(assistant_id)}"
        headers = {"Authorization": f"Bearer {admin_key}"}
        payload = {"timezone": timezone}
        resp = http.patch(url, headers=headers, json=payload, timeout=30)
        if 200 <= resp.status_code < 300:
            _log.info(f"Synced assistant timezone to backend: {timezone}")
            return True
        _log.warning(
            f"Failed to sync assistant timezone: {resp.status_code} {resp.text}",
        )
        return False
    except Exception as e:
        _log.warning(f"Error syncing assistant timezone: {e}")
        return False


def sync_assistant_about(assistant_id: int, about: str) -> bool:
    """
    Fire-and-forget sync of about field to assistant profile.

    Uses PATCH /admin/assistant/{assistant_id}
    Note: ContactManager stores this as 'bio', backend uses 'about'.
    """
    base_url = _get_base_url()
    admin_key = _get_admin_key()
    if not base_url or not admin_key or not assistant_id:
        _log.debug("Skipping assistant about sync: missing required params")
        return False

    try:
        from unify.utils import http

        url = f"{base_url}/admin/assistant/{int(assistant_id)}"
        headers = {"Authorization": f"Bearer {admin_key}"}
        payload = {"about": about}
        resp = http.patch(url, headers=headers, json=payload, timeout=30)
        if 200 <= resp.status_code < 300:
            _log.info("Synced assistant about to backend")
            return True
        _log.warning(f"Failed to sync assistant about: {resp.status_code} {resp.text}")
        return False
    except Exception as e:
        _log.warning(f"Error syncing assistant about: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Hive membership enumeration → GET /hives/{hive_id}/assistants
# ─────────────────────────────────────────────────────────────────────────────


def list_hive_assistants(hive_id: int) -> List[Tuple[str, int]]:
    """Enumerate every ``(user_id, assistant_id)`` pair in a Hive.

    Used by ``ContactManager.merge_contacts`` to fan-out per-body
    ``ContactMembership`` rewrites across every member of the Hive so
    the overlay on each body points at the surviving shared row.

    Fire-and-fail-open: if the Orchestra URL or admin key are missing,
    or the call fails, returns ``[]``. Callers must treat an empty
    result as "no other bodies to rewrite" and proceed with only the
    local overlay, which is still correct for solo bodies.
    """
    base_url = _get_base_url()
    admin_key = _get_admin_key()
    if not base_url or not admin_key or not hive_id:
        _log.debug("Skipping list_hive_assistants: missing base URL / key / hive_id")
        return []

    try:
        from unify.utils import http

        url = f"{base_url}/hives/{int(hive_id)}/assistants"
        headers = {"Authorization": f"Bearer {admin_key}"}
        resp = http.get(url, headers=headers, timeout=30)
        if 200 <= resp.status_code < 300:
            body = resp.json()
            return [
                (str(row["user_id"]), int(row["assistant_id"]))
                for row in body
                if row.get("user_id") is not None
                and row.get("assistant_id") is not None
            ]
        _log.warning(
            f"Failed to list hive assistants: {resp.status_code} {resp.text}",
        )
        return []
    except Exception as e:
        _log.warning(f"Error listing hive assistants: {e}")
        return []


def sync_assistant_job_title(assistant_id: int, job_title: str) -> bool:
    """
    Fire-and-forget sync of ``job_title`` to the assistant profile.

    Uses ``PATCH /admin/assistant/{assistant_id}``. Treats empty/whitespace
    strings as ``None`` so that clearing the field on the contact also clears
    it server-side (the backend additionally normalizes/trims).
    """
    base_url = _get_base_url()
    admin_key = _get_admin_key()
    if not base_url or not admin_key or not assistant_id:
        _log.debug("Skipping assistant job_title sync: missing required params")
        return False

    normalized: str | None = job_title.strip() if isinstance(job_title, str) else None
    if normalized == "":
        normalized = None

    try:
        from unify.utils import http

        url = f"{base_url}/admin/assistant/{int(assistant_id)}"
        headers = {"Authorization": f"Bearer {admin_key}"}
        payload = {"job_title": normalized}
        resp = http.patch(url, headers=headers, json=payload, timeout=30)
        if 200 <= resp.status_code < 300:
            _log.info("Synced assistant job_title to backend")
            return True
        _log.warning(
            f"Failed to sync assistant job_title: {resp.status_code} {resp.text}",
        )
        return False
    except Exception as e:
        _log.warning(f"Error syncing assistant job_title: {e}")
        return False
