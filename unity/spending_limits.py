"""
Spending limit checking for Unity.

This module implements the limit check callback that UniLLM invokes before
each LLM call. It queries Orchestra's admin endpoints to check if spending
limits have been exceeded for the current assistant, user, or organization.

The callback is registered with UniLLM during unity.init() and uses
SESSION_DETAILS to determine the current context.

Limit hierarchy:
- Personal context (user's personal API key): assistant + user limits
- Organization context (org API key): assistant + member + org limits

All checks run in parallel for minimal latency impact.
"""

from __future__ import annotations

import asyncio
import logging
import os
import zoneinfo
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

import httpx

if TYPE_CHECKING:
    from unillm.limit_hooks import LimitCheckRequest, LimitCheckResponse

logger = logging.getLogger(__name__)

# Default timeout for limit check requests (should be fast)
LIMIT_CHECK_TIMEOUT = 5.0


@dataclass
class _LimitCheckResult:
    """Internal result from a single limit check."""

    exceeded: bool
    limit_type: Optional[str] = None
    limit_value: Optional[float] = None
    current_spend: Optional[float] = None
    entity_id: Optional[str] = None
    entity_name: Optional[str] = None


def _get_api_key() -> Optional[str]:
    """Get the admin API key for Orchestra calls."""
    return os.getenv("ORCHESTRA_ADMIN_KEY")


def _get_base_url() -> str:
    """Get the Orchestra API base URL."""
    return os.getenv("ORCHESTRA_URL", "https://api.unify.ai/v0")


def _get_current_month(timezone: str = "UTC") -> str:
    """Get current month string in YYYY-MM format for the given timezone."""
    try:
        tz = zoneinfo.ZoneInfo(timezone)
    except Exception:
        tz = zoneinfo.ZoneInfo("UTC")
    return datetime.now(tz).strftime("%Y-%m")


async def _check_assistant_limit(
    agent_id: str,
    month: str,
    base_url: str,
    api_key: str,
    timeout: float,
) -> _LimitCheckResult:
    """Check if assistant spending limit is exceeded."""
    url = f"{base_url}/admin/assistant/{agent_id}/spend"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"month": month}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

        limit = data.get("limit")
        spend = data.get("cumulative_spend", 0)

        # No limit set = unlimited
        if limit is None:
            return _LimitCheckResult(exceeded=False)

        exceeded = spend >= limit
        return _LimitCheckResult(
            exceeded=exceeded,
            limit_type="assistant",
            limit_value=limit,
            current_spend=spend,
            entity_id=agent_id,
            entity_name=data.get("agent_name"),
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # Assistant not found or no spend data - allow
            return _LimitCheckResult(exceeded=False)
        logger.warning(f"Failed to check assistant limit: {e}")
        return _LimitCheckResult(exceeded=False)  # Fail open
    except Exception as e:
        logger.warning(f"Failed to check assistant limit: {e}")
        return _LimitCheckResult(exceeded=False)  # Fail open


async def _check_user_limit(
    user_id: str,
    month: str,
    base_url: str,
    api_key: str,
    timeout: float,
) -> _LimitCheckResult:
    """Check if user's personal spending limit is exceeded."""
    url = f"{base_url}/admin/user/{user_id}/spend"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"month": month}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

        limit = data.get("limit")
        spend = data.get("cumulative_spend", 0)

        if limit is None:
            return _LimitCheckResult(exceeded=False)

        exceeded = spend >= limit
        return _LimitCheckResult(
            exceeded=exceeded,
            limit_type="user",
            limit_value=limit,
            current_spend=spend,
            entity_id=user_id,
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return _LimitCheckResult(exceeded=False)
        logger.warning(f"Failed to check user limit: {e}")
        return _LimitCheckResult(exceeded=False)
    except Exception as e:
        logger.warning(f"Failed to check user limit: {e}")
        return _LimitCheckResult(exceeded=False)


async def _check_member_limit(
    user_id: str,
    org_id: int,
    month: str,
    base_url: str,
    api_key: str,
    timeout: float,
) -> _LimitCheckResult:
    """Check if organization member's spending limit is exceeded."""
    url = f"{base_url}/admin/organization/{org_id}/members/{user_id}/spend"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"month": month}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

        limit = data.get("limit")
        spend = data.get("cumulative_spend", 0)

        if limit is None:
            return _LimitCheckResult(exceeded=False)

        exceeded = spend >= limit
        return _LimitCheckResult(
            exceeded=exceeded,
            limit_type="member",
            limit_value=limit,
            current_spend=spend,
            entity_id=user_id,
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return _LimitCheckResult(exceeded=False)
        logger.warning(f"Failed to check member limit: {e}")
        return _LimitCheckResult(exceeded=False)
    except Exception as e:
        logger.warning(f"Failed to check member limit: {e}")
        return _LimitCheckResult(exceeded=False)


async def _check_org_limit(
    org_id: int,
    month: str,
    base_url: str,
    api_key: str,
    timeout: float,
) -> _LimitCheckResult:
    """Check if organization spending limit is exceeded."""
    url = f"{base_url}/admin/organization/{org_id}/spend"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"month": month}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

        limit = data.get("limit")
        spend = data.get("cumulative_spend", 0)

        if limit is None:
            return _LimitCheckResult(exceeded=False)

        exceeded = spend >= limit
        return _LimitCheckResult(
            exceeded=exceeded,
            limit_type="organization",
            limit_value=limit,
            current_spend=spend,
            entity_id=str(org_id),
            entity_name=data.get("organization_name"),
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return _LimitCheckResult(exceeded=False)
        logger.warning(f"Failed to check org limit: {e}")
        return _LimitCheckResult(exceeded=False)
    except Exception as e:
        logger.warning(f"Failed to check org limit: {e}")
        return _LimitCheckResult(exceeded=False)


async def check_spending_limits_callback(
    request: "LimitCheckRequest",
) -> "LimitCheckResponse":
    """Limit check callback for UniLLM.

    This is the callback registered with UniLLM via set_limit_check_hook().
    It uses SESSION_DETAILS to determine the current context and checks all
    applicable limits in parallel.

    Args:
        request: Information about the pending LLM call (from UniLLM).

    Returns:
        LimitCheckResponse indicating whether to proceed.
    """
    from unillm.limit_hooks import LimitCheckResponse, LimitType

    from .session_details import SESSION_DETAILS

    # Get API key and base URL
    api_key = _get_api_key()
    if not api_key:
        logger.debug("Spending limit check skipped: no API key")
        return LimitCheckResponse(allowed=True)

    base_url = _get_base_url()
    timeout = LIMIT_CHECK_TIMEOUT

    # Get context from SESSION_DETAILS
    agent_id = None
    if SESSION_DETAILS.assistant_record:
        agent_id = SESSION_DETAILS.assistant_record.get("agent_id")

    user_id = SESSION_DETAILS.user_id
    org_id = SESSION_DETAILS.org_id  # None for personal context

    # Get timezone for month calculation
    timezone = "UTC"
    if SESSION_DETAILS.assistant:
        timezone = SESSION_DETAILS.assistant.timezone or "UTC"

    # Skip if we don't have required context
    if not agent_id or not user_id:
        logger.debug("Spending limit check skipped: missing context")
        return LimitCheckResponse(allowed=True)

    month = _get_current_month(timezone)

    # Build list of limit checks based on context
    checks: List[asyncio.Task] = []

    # Always check assistant limit
    checks.append(
        asyncio.create_task(
            _check_assistant_limit(agent_id, month, base_url, api_key, timeout),
        ),
    )

    is_org_context = org_id is not None
    if is_org_context:
        # Org context: check member + org limits
        checks.append(
            asyncio.create_task(
                _check_member_limit(user_id, org_id, month, base_url, api_key, timeout),
            ),
        )
        checks.append(
            asyncio.create_task(
                _check_org_limit(org_id, month, base_url, api_key, timeout),
            ),
        )
    else:
        # Personal context: check user personal limit
        checks.append(
            asyncio.create_task(
                _check_user_limit(user_id, month, base_url, api_key, timeout),
            ),
        )

    # Wait for all checks in parallel
    results = await asyncio.gather(*checks, return_exceptions=True)

    # Convert limit type string to enum
    def _to_limit_type(type_str: Optional[str]) -> Optional[LimitType]:
        if type_str is None:
            return None
        try:
            return LimitType(type_str)
        except ValueError:
            return None

    # Return first exceeded result
    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"Limit check failed with exception: {result}")
            continue
        if result.exceeded:
            current = (
                f"${result.current_spend:.2f}" if result.current_spend else "unknown"
            )
            limit = f"${result.limit_value:.2f}" if result.limit_value else "unknown"
            reason = f"Monthly spending limit exceeded: {result.limit_type} limit of {limit} reached (current: {current})"
            return LimitCheckResponse(
                allowed=False,
                reason=reason,
                limit_type=_to_limit_type(result.limit_type),
                limit_value=result.limit_value,
                current_spend=result.current_spend,
                entity_id=result.entity_id,
                entity_name=result.entity_name,
            )

    return LimitCheckResponse(allowed=True)


def install_limit_check_hook() -> None:
    """Install the spending limit check hook with UniLLM.

    This function is idempotent - calling it multiple times has no effect
    after the first successful installation.

    Should be called during unity.init() after SESSION_DETAILS is populated.
    """
    # Only install if we have an API key configured
    api_key = _get_api_key()
    if not api_key:
        logger.debug("Limit check hook not installed: no API key")
        return

    try:
        import unillm

        unillm.set_limit_check_hook(check_spending_limits_callback)
        logger.debug("Limit check hook installed")
    except ImportError:
        # unillm not available - skip hook installation
        pass
    except Exception as e:
        # Any other error - skip silently to not break initialization
        logger.debug(f"Failed to install limit check hook: {e}")


def uninstall_limit_check_hook() -> None:
    """Uninstall the spending limit check hook from UniLLM."""
    try:
        import unillm

        unillm.clear_limit_check_hook()
        logger.debug("Limit check hook uninstalled")
    except ImportError:
        pass
    except Exception:
        pass
