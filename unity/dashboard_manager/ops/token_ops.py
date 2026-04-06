"""Token operations for DashboardManager.

Handles token generation and registration with Orchestra's dashboard_token
lookup table. These are called by DashboardManager methods.
"""

from __future__ import annotations

import logging
import secrets

import httpx

from unity.session_details import SESSION_DETAILS
from unity.settings import SETTINGS

logger = logging.getLogger(__name__)


def generate_token() -> str:
    """Generate a cryptographically secure 12-character URL-safe token."""
    return secrets.token_urlsafe(9)[:12]


def _get_auth_headers() -> dict[str, str]:
    """Get authentication headers for Orchestra API requests."""
    unify_key = SESSION_DETAILS.unify_key
    if not unify_key:
        logger.warning("UNIFY_KEY not set - token registration may fail")
    return {
        "Authorization": f"Bearer {unify_key}",
        "Content-Type": "application/json",
    }


def register_token(
    token: str,
    entity_type: str,
    context_name: str,
    project_name: str,
) -> bool:
    """Register a token-to-context mapping in Orchestra.

    Parameters
    ----------
    token : str
        The 12-char URL-safe token (already inserted into Unify context).
    entity_type : str
        Either "tile" or "dashboard".
    context_name : str
        The Unify context name where the content is stored.
    project_name : str
        The Unify project name that owns this context.

    Returns
    -------
    bool
        True if registration succeeded.
    """
    url = f"{SETTINGS.ORCHESTRA_URL}/dashboards/tokens"
    payload = {
        "token": token,
        "entity_type": entity_type,
        "context_name": context_name,
        "project_name": project_name,
    }

    try:
        resp = httpx.post(
            url,
            json=payload,
            headers=_get_auth_headers(),
            timeout=15.0,
        )
        if resp.status_code == 201:
            return True
        if resp.status_code == 409:
            logger.warning("Token %s already registered", token)
            return True
        logger.error(
            "Token registration failed: %s %s",
            resp.status_code,
            resp.text,
        )
        return False
    except httpx.RequestError as e:
        logger.error("Token registration request failed: %s", e)
        return False


def delete_token(token: str) -> bool:
    """Remove a token mapping from Orchestra.

    Parameters
    ----------
    token : str
        The token to deregister.

    Returns
    -------
    bool
        True if deletion succeeded or token did not exist.
    """
    url = f"{SETTINGS.ORCHESTRA_URL}/dashboards/tokens/{token}"
    try:
        resp = httpx.delete(
            url,
            headers=_get_auth_headers(),
            timeout=15.0,
        )
        return resp.status_code in (200, 404)
    except httpx.RequestError as e:
        logger.error("Token deletion request failed: %s", e)
        return False
