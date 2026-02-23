"""
Backend sync utilities for ContactManager.

Provides fire-and-forget sync of system contact fields (timezone, bio/about)
to the orchestra backend User and Assistant profiles.

Endpoints:
- GET /organizations/members (list org members)
- POST /admin/assistant/update-user (user timezone/bio)
- PATCH /admin/assistant/{assistant_id} (assistant timezone/about)
"""

from __future__ import annotations

import logging

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
