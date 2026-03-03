"""
tests/actor/code_act/test_fast_path_watchdog.py
================================================

Eval tests verifying the CodeActActor's fast-path watchdog behaviour.

When the actor is in persist mode and receives ``[Fast-path result]``
interjections, it should:

- Escalate via ``notify()`` when the fast path clearly failed or attempted
  work requiring capabilities it lacks (credentials, multi-step workflows).
- Stay quiet (no escalation) when the fast path succeeded at a simple
  atomic action.
"""

import asyncio

import pytest

from unity.actor.code_act_actor import CodeActActor
from unity.actor.environments.computer import ComputerEnvironment
from unity.function_manager.primitives.runtime import ComputerPrimitives
from unity.manager_registry import ManagerRegistry

pytestmark = pytest.mark.eval


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _make_actor() -> CodeActActor:
    """Create a CodeActActor with mock computer primitives so the
    fast-path awareness prompt section is included."""
    ManagerRegistry.clear()
    cp = ComputerPrimitives(computer_mode="mock")
    computer_env = ComputerEnvironment(cp)
    return CodeActActor(environments=[computer_env], timeout=60)


async def _wait_for_escalation(handle, *, timeout: float = 60) -> dict:
    """Wait for an escalation notification, skipping persist-mode responses."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            break
        try:
            notif = await asyncio.wait_for(
                handle.next_notification(),
                timeout=min(remaining, 10),
            )
        except asyncio.TimeoutError:
            continue

        if not isinstance(notif, dict):
            continue
        # Skip persist-mode turn-complete responses.
        if notif.get("type") == "response":
            continue
        return notif

    raise asyncio.TimeoutError("No escalation notification within timeout")


async def _assert_no_escalation(handle, *, window: float = 15) -> None:
    """Assert no escalation notification arrives within the window,
    tolerating persist-mode response notifications."""
    deadline = asyncio.get_event_loop().time() + window
    while asyncio.get_event_loop().time() < deadline:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            break
        try:
            notif = await asyncio.wait_for(
                handle.next_notification(),
                timeout=min(remaining, 5),
            )
        except asyncio.TimeoutError:
            continue

        if not isinstance(notif, dict):
            continue
        if notif.get("type") == "response":
            continue
        notif_type = notif.get("type", "")
        notif_msg = str(notif.get("message", ""))
        assert notif_type != "escalation" and "escalat" not in notif_msg.lower(), (
            f"Unexpected escalation notification: type={notif_type!r}, "
            f"message={notif_msg!r}"
        )


# ---------------------------------------------------------------------------
#  Test A — credential fast-path failure triggers escalation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_actor_escalates_on_failed_credential_fast_path():
    """The actor should escalate when a fast-path result shows the browser
    agent tried to handle credentials without access to primitives.secrets.
    """
    actor = _make_actor()
    handle = None

    try:
        handle = await actor.act(
            "Interactive desktop session is starting. My boss is on a live "
            "call walking me through a CoStar deal research workflow. The "
            "website requires authentication — credentials are stored in "
            "the Secret Manager (primitives.secrets). Stay available for "
            "desktop interactions and instructions.",
            persist=True,
            can_store=False,
            clarification_enabled=False,
        )

        await asyncio.sleep(5)
        assert not handle.done(), "persist=True should keep the loop alive"

        await handle.interject(
            '[Fast-path result] web_act("Log in to CoStar using stored '
            'credentials from Secret Manager") completed. Result: I need '
            "to find CoStar credentials from Secret Manager. I'll navigate "
            "to Google Cloud Secret Manager or check common secret "
            "management URLs.\n\n"
            "If this result looks wrong or incomplete — especially if the "
            "task falls within your loaded guidance or requires capabilities "
            "the fast path lacks (credentials, secrets, multi-step "
            "workflows) — escalate by calling "
            'notify({"type": "escalation", "message": "<what you can do '
            'better>"}).  Otherwise, no action needed.',
        )

        notification = await _wait_for_escalation(handle, timeout=60)

        assert isinstance(notification, dict)
        notif_type = notification.get("type")
        notif_msg = str(notification.get("message", ""))
        assert notif_type == "escalation" or "escalat" in notif_msg.lower(), (
            f"Expected an escalation notification but got type={notif_type!r}, "
            f"message={notif_msg!r}"
        )

    finally:
        if handle and not handle.done():
            try:
                await handle.stop("test cleanup")
            except Exception:
                pass
        try:
            await actor.close()
        except Exception:
            pass
        ManagerRegistry.clear()


# ---------------------------------------------------------------------------
#  Test B — successful atomic fast-path does NOT trigger escalation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_actor_no_escalation_on_successful_atomic_fast_path():
    """The actor should NOT escalate when a fast-path result shows a simple
    atomic action that completed successfully.
    """
    actor = _make_actor()
    handle = None

    try:
        handle = await actor.act(
            "Interactive desktop session. Stay available for desktop " "interactions.",
            persist=True,
            can_store=False,
            clarification_enabled=False,
        )

        await asyncio.sleep(5)
        assert not handle.done()

        await handle.interject(
            '[Fast-path result] desktop_act("Click the Submit button") '
            "completed. Result: Clicked the Submit button successfully.\n\n"
            "If this result looks wrong or incomplete — especially if the "
            "task falls within your loaded guidance or requires capabilities "
            "the fast path lacks (credentials, secrets, multi-step "
            "workflows) — escalate by calling "
            'notify({"type": "escalation", "message": "<what you can do '
            'better>"}).  Otherwise, no action needed.',
        )

        await _assert_no_escalation(handle, window=15)
        assert not handle.done(), "Actor should still be alive in persist mode"

    finally:
        if handle and not handle.done():
            try:
                await handle.stop("test cleanup")
            except Exception:
                pass
        try:
            await actor.close()
        except Exception:
            pass
        ManagerRegistry.clear()


# ---------------------------------------------------------------------------
#  Test C — multi-step fast-path attempt triggers escalation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_actor_escalates_on_multi_step_fast_path_attempt():
    """The actor should escalate when a fast-path result shows it attempted
    a complex multi-step task that should have been routed through the actor.
    """
    actor = _make_actor()
    handle = None

    try:
        handle = await actor.act(
            "Interactive desktop session for data extraction. Stay available "
            "for browser interactions and data processing tasks.",
            persist=True,
            can_store=False,
            clarification_enabled=False,
        )

        await asyncio.sleep(5)
        assert not handle.done()

        await handle.interject(
            '[Fast-path result] web_act("Copy all the data from the table '
            'on this page and save it to a spreadsheet") completed. Result: '
            "I can see a table with property data on the page. I need to "
            "scroll through all rows, extract each cell, compile into a "
            "structured format, and save to a file. This requires multiple "
            "sequential steps.\n\n"
            "If this result looks wrong or incomplete — especially if the "
            "task falls within your loaded guidance or requires capabilities "
            "the fast path lacks (credentials, secrets, multi-step "
            "workflows) — escalate by calling "
            'notify({"type": "escalation", "message": "<what you can do '
            'better>"}).  Otherwise, no action needed.',
        )

        notification = await _wait_for_escalation(handle, timeout=60)

        assert isinstance(notification, dict)
        notif_type = notification.get("type")
        notif_msg = str(notification.get("message", ""))
        assert notif_type == "escalation" or "escalat" in notif_msg.lower(), (
            f"Expected an escalation notification but got type={notif_type!r}, "
            f"message={notif_msg!r}"
        )

    finally:
        if handle and not handle.done():
            try:
                await handle.stop("test cleanup")
            except Exception:
                pass
        try:
            await actor.close()
        except Exception:
            pass
        ManagerRegistry.clear()
