"""
Spending limit checking for Unity.

This module implements the limit check callback that UniLLM invokes before
each LLM call. It queries Orchestra's spend endpoints to check if spending
limits have been exceeded for the current assistant, user, or organization.

The callback is registered with UniLLM during unity.init() and uses
SESSION_DETAILS to determine the current context.

Limit hierarchy:
- Personal context (user's personal API key): assistant + user limits
- Organization context (org API key): assistant + member + org limits

All checks run in parallel for minimal latency impact.

Uses ``unify.AsyncSpendClient`` (aiohttp-backed) for connection pooling,
automatic retries, and exponential backoff — matching the reliability
characteristics of the sync ``unify.utils.http`` session.
"""

from __future__ import annotations

import asyncio
import logging
import os
import zoneinfo
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from unify.async_admin import AsyncSpendClient, SpendRequestError

if TYPE_CHECKING:
    from unillm.limit_hooks import LimitCheckRequest, LimitCheckResponse

logger = logging.getLogger(__name__)

LIMIT_CHECK_TIMEOUT = 5.0

_spend_client: Optional[AsyncSpendClient] = None


def _get_api_key() -> Optional[str]:
    """Get the user API key for Orchestra calls."""
    return os.getenv("UNIFY_KEY")


def _get_spend_client() -> AsyncSpendClient:
    """Get or create the shared AsyncSpendClient for limit checks."""
    global _spend_client
    api_key = _get_api_key()
    if _spend_client is None or _spend_client.closed:
        _spend_client = AsyncSpendClient(
            api_key=api_key,
            timeout=LIMIT_CHECK_TIMEOUT,
        )
    return _spend_client


@dataclass
class _LimitCheckResult:
    """Internal result from a single limit check."""

    exceeded: bool
    limit_type: Optional[str] = None
    limit_value: Optional[float] = None
    current_spend: Optional[float] = None
    entity_id: Optional[str] = None
    entity_name: Optional[str] = None
    limit_set_at: Optional[str] = None  # ISO format timestamp
    organization_id: Optional[int] = None  # For member limits
    credit_balance: Optional[float] = None  # Billing account credit balance


def _get_current_month(timezone: str = "UTC") -> str:
    """Get current month string in YYYY-MM format for the given timezone."""
    try:
        tz = zoneinfo.ZoneInfo(timezone)
    except Exception:
        tz = zoneinfo.ZoneInfo("UTC")
    return datetime.now(tz).strftime("%Y-%m")


def _parse_spend_result(
    data: dict,
    limit_type: str,
    entity_id: str,
    *,
    entity_name: Optional[str] = None,
    organization_id: Optional[int] = None,
) -> _LimitCheckResult:
    """Parse a spend endpoint response into a ``_LimitCheckResult``."""
    limit = data.get("limit")
    spend = data.get("cumulative_spend", 0)
    limit_set_at = data.get("limit_set_at")
    credit_balance = data.get("credit_balance")

    if limit is None:
        return _LimitCheckResult(exceeded=False, credit_balance=credit_balance)

    return _LimitCheckResult(
        exceeded=spend >= limit,
        limit_type=limit_type,
        limit_value=limit,
        current_spend=spend,
        entity_id=entity_id,
        entity_name=entity_name or data.get("agent_name"),
        limit_set_at=limit_set_at,
        organization_id=organization_id,
        credit_balance=credit_balance,
    )


async def _check_assistant_limit(
    agent_id: str,
    month: str,
) -> _LimitCheckResult:
    """Check if assistant spending limit is exceeded."""
    try:
        client = _get_spend_client()
        data = await client.get_assistant_spend(agent_id=int(agent_id), month=month)
        return _parse_spend_result(data, "assistant", agent_id)
    except SpendRequestError as e:
        if e.status == 404:
            return _LimitCheckResult(exceeded=False)
        logger.warning(f"Failed to check assistant limit: {type(e).__name__}: {e}")
        return _LimitCheckResult(exceeded=False)
    except Exception as e:
        logger.warning(f"Failed to check assistant limit: {type(e).__name__}: {e}")
        return _LimitCheckResult(exceeded=False)


async def _check_user_limit(
    user_id: str,
    month: str,
) -> _LimitCheckResult:
    """Check if user's personal spending limit is exceeded."""
    try:
        client = _get_spend_client()
        data = await client.get_user_spend(month=month)
        return _parse_spend_result(data, "user", user_id)
    except SpendRequestError as e:
        if e.status == 404:
            return _LimitCheckResult(exceeded=False)
        logger.warning(f"Failed to check user limit: {type(e).__name__}: {e}")
        return _LimitCheckResult(exceeded=False)
    except Exception as e:
        logger.warning(f"Failed to check user limit: {type(e).__name__}: {e}")
        return _LimitCheckResult(exceeded=False)


async def _check_member_limit(
    user_id: str,
    org_id: int,
    month: str,
) -> _LimitCheckResult:
    """Check if organization member's spending limit is exceeded."""
    try:
        client = _get_spend_client()
        data = await client.get_member_spend(
            user_id=user_id,
            org_id=org_id,
            month=month,
        )
        return _parse_spend_result(
            data,
            "member",
            user_id,
            organization_id=org_id,
        )
    except SpendRequestError as e:
        if e.status == 404:
            return _LimitCheckResult(exceeded=False)
        logger.warning(f"Failed to check member limit: {type(e).__name__}: {e}")
        return _LimitCheckResult(exceeded=False)
    except Exception as e:
        logger.warning(f"Failed to check member limit: {type(e).__name__}: {e}")
        return _LimitCheckResult(exceeded=False)


async def _check_org_limit(
    org_id: int,
    month: str,
) -> _LimitCheckResult:
    """Check if organization spending limit is exceeded."""
    try:
        client = _get_spend_client()
        data = await client.get_org_spend(org_id=org_id, month=month)
        return _parse_spend_result(
            data,
            "organization",
            str(org_id),
            entity_name=data.get("organization_name"),
        )
    except SpendRequestError as e:
        if e.status == 404:
            return _LimitCheckResult(exceeded=False)
        logger.warning(f"Failed to check org limit: {type(e).__name__}: {e}")
        return _LimitCheckResult(exceeded=False)
    except Exception as e:
        logger.warning(f"Failed to check org limit: {type(e).__name__}: {e}")
        return _LimitCheckResult(exceeded=False)


async def _notify_limit_reached(
    result: _LimitCheckResult,
    month: str,
) -> None:
    """
    Fire-and-forget notification to Orchestra when a limit is reached.

    This calls Orchestra's spending-limit-reached endpoint which:
    - Deduplicates notifications (won't spam for same limit)
    - Sends email to affected users
    - Records the notification for auditing

    Errors are logged but don't affect the limit check response.
    """
    payload = {
        "limit_type": result.limit_type,
        "entity_id": result.entity_id,
        "limit_value": result.limit_value,
        "current_spend": result.current_spend,
        "month": month,
        "entity_name": result.entity_name,
    }

    if result.limit_set_at:
        payload["limit_set_at"] = result.limit_set_at

    if result.organization_id:
        payload["organization_id"] = result.organization_id

    try:
        client = _get_spend_client()
        data = await client.notify_limit_reached(payload)
        if data.get("notified"):
            logger.info(
                f"Spending limit notification sent for {result.limit_type} "
                f"limit (entity_id={result.entity_id}, limit=${result.limit_value})",
            )
        else:
            logger.debug(
                f"Spending limit notification skipped: {data.get('reason', 'unknown')}",
            )
    except Exception as e:
        logger.warning(f"Failed to send spending limit notification: {e}")


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

    api_key = _get_api_key()
    if not api_key:
        logger.debug("Spending limit check skipped: no API key")
        return LimitCheckResponse(allowed=True)

    agent_id = SESSION_DETAILS.assistant.agent_id

    user_id = SESSION_DETAILS.user_id
    org_id = SESSION_DETAILS.org_id  # None for personal context

    timezone = "UTC"
    if SESSION_DETAILS.assistant:
        timezone = SESSION_DETAILS.assistant.timezone or "UTC"

    if not agent_id or not user_id:
        logger.debug("Spending limit check skipped: missing context")
        return LimitCheckResponse(allowed=True)

    month = _get_current_month(timezone)

    checks: List[asyncio.Task] = []

    checks.append(
        asyncio.create_task(
            _check_assistant_limit(agent_id, month),
        ),
    )

    is_org_context = org_id is not None
    if is_org_context:
        checks.append(
            asyncio.create_task(
                _check_member_limit(user_id, org_id, month),
            ),
        )
        checks.append(
            asyncio.create_task(
                _check_org_limit(org_id, month),
            ),
        )
    else:
        checks.append(
            asyncio.create_task(
                _check_user_limit(user_id, month),
            ),
        )

    results = await asyncio.gather(*checks, return_exceptions=True)

    def _to_limit_type(type_str: Optional[str]) -> Optional[LimitType]:
        if type_str is None:
            return None
        try:
            return LimitType(type_str)
        except ValueError:
            return None

    credit_balance: Optional[float] = None

    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"Limit check failed with exception: {result}")
            continue

        if credit_balance is None and result.credit_balance is not None:
            credit_balance = result.credit_balance

        if result.exceeded:
            current = (
                f"${result.current_spend:.2f}" if result.current_spend else "unknown"
            )
            limit = f"${result.limit_value:.2f}" if result.limit_value else "unknown"
            reason = f"Monthly spending limit exceeded: {result.limit_type} limit of {limit} reached (current: {current})"

            asyncio.create_task(
                _notify_limit_reached(result, month),
            )

            return LimitCheckResponse(
                allowed=False,
                reason=reason,
                limit_type=_to_limit_type(result.limit_type),
                limit_value=result.limit_value,
                current_spend=result.current_spend,
                entity_id=result.entity_id,
                entity_name=result.entity_name,
            )

    if credit_balance is not None and credit_balance <= 0:
        return LimitCheckResponse(
            allowed=False,
            reason=(
                f"Insufficient credits: balance is ${credit_balance:.2f}. "
                "Please add credits to continue."
            ),
        )

    return LimitCheckResponse(allowed=True)


def install_limit_check_hook() -> None:
    """Install the spending limit check hook with UniLLM.

    This function is idempotent - calling it multiple times has no effect
    after the first successful installation.

    Should be called during unity.init() after SESSION_DETAILS is populated.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.debug("Limit check hook not installed: no API key")
        return

    try:
        import unillm

        unillm.set_limit_check_hook(check_spending_limits_callback)
        logger.debug("Limit check hook installed")
    except ImportError:
        pass
    except Exception as e:
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
