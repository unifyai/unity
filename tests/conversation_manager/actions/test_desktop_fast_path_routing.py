"""
tests/conversation_manager/actions/test_desktop_fast_path_routing.py
=====================================================================

Eval tests verifying the CM brain routes:
- Atomic desktop action requests to ``desktop_act``
- Complex / multi-step requests to ``act`` even when fast path is available
- Concurrent ``act(persist=True)`` is spun up alongside ``desktop_act`` when
  no act session is already in-flight
- Observation/screenshot requests route through ``act`` (not fast path)

These tests follow the same end-to-end pattern as test_take_action.py and
test_ask_about_contacts.py: the ``initialized_cm`` fixture provides a real
ConversationManager with SimulatedActor, and ``step_until_wait`` runs the
real ``_run_llm()`` with a real LLM call.

A ``ComputerPrimitives(computer_mode="mock")`` singleton is registered so
``cm.computer_primitives`` resolves naturally via ManagerRegistry (no
property patching). The desktop fast-path tool is exposed when screen share
is active — no bootstrap ``act`` call is required.
"""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import (
    assert_efficient,
)
from tests.conversation_manager.conftest import BOSS
from unity.conversation_manager.events import (
    SMSReceived,
    UnifyMessageReceived,
)

pytestmark = pytest.mark.eval


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _ensure_mock_computer_primitives():
    """Create the mock ComputerPrimitives singleton if it doesn't exist yet."""
    from unity.function_manager.primitives.runtime import ComputerPrimitives
    from unity.manager_registry import ManagerRegistry

    if ManagerRegistry.get_instance(ComputerPrimitives) is None:
        ComputerPrimitives(computer_mode="mock")


def _enable_desktop_fast_path(cm_driver):
    """Activate desktop fast-path tool by turning on screen share."""
    cm_driver.cm.assistant_screen_share_active = True


def _setup_desktop_fast_path_from_real_act(cm_driver):
    """Activate desktop fast-path gating and mark existing act sessions as
    desktop-active (for interjection targeting).

    Call this AFTER a ``step_until_wait`` that triggered ``act`` — the
    SimulatedActor handle will be sitting in ``in_flight_actions``.
    """
    cm = cm_driver.cm
    cm.assistant_screen_share_active = True

    for hid, data in cm.in_flight_actions.items():
        if data.get("action_type") == "act":
            cm._act_handles_with_desktop_usage.add(hid)


def _teardown_desktop_fast_path(cm_driver):
    """Reset gating state so subsequent tests start clean.

    NOTE: in_flight_actions cleanup is handled by the ``initialized_cm``
    fixture (``_complete_in_flight_actions``).  We only clear our additions.
    """
    cm = cm_driver.cm
    cm._act_handles_with_desktop_usage.clear()
    cm.assistant_screen_share_active = False


# ---------------------------------------------------------------------------
#  Atomic action requests → desktop_act (with existing act session)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_click_routes_to_desktop_act(initialized_cm):
    """A simple 'click X' request should route to desktop_act.

    Flow: first message triggers ``act`` (creates in-flight action) →
    mark act as desktop-active → second message should route to ``desktop_act``.
    """
    cm = initialized_cm
    _ensure_mock_computer_primitives()

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Search the web for how to configure CRM software settings",
        ),
    )
    assert (
        "act" in cm.all_tool_calls
    ), f"First message should trigger act, got: {cm.all_tool_calls}"

    _setup_desktop_fast_path_from_real_act(cm)
    cm.all_tool_calls.clear()

    try:
        result = await cm.step_until_wait(
            UnifyMessageReceived(
                contact=BOSS,
                content="Click the Submit button on the screen",
            ),
        )

        assert "desktop_act" in cm.all_tool_calls, (
            f"Expected 'desktop_act' for atomic click request, "
            f"got: {cm.all_tool_calls}"
        )
        assert_efficient(result, 5)
    finally:
        _teardown_desktop_fast_path(cm)


# ---------------------------------------------------------------------------
#  Atomic action requests → desktop_act + concurrent act (no existing session)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_click_routes_to_desktop_act_with_concurrent_act(initialized_cm):
    """When no act session is in-flight, an atomic click request should trigger
    both ``desktop_act`` (for the immediate action) AND ``act`` with
    ``persist=True`` (for the full-capability session) in the same turn.
    """
    cm = initialized_cm
    _ensure_mock_computer_primitives()
    _enable_desktop_fast_path(cm)

    try:
        result = await cm.step_until_wait(
            UnifyMessageReceived(
                contact=BOSS,
                content="Click the Submit button on the screen",
            ),
        )

        assert "desktop_act" in cm.all_tool_calls, (
            f"Expected 'desktop_act' for atomic click request, "
            f"got: {cm.all_tool_calls}"
        )
        assert "act" in cm.all_tool_calls, (
            f"Expected concurrent 'act' session alongside desktop_act, "
            f"got: {cm.all_tool_calls}"
        )
        assert_efficient(result, 5)
    finally:
        _teardown_desktop_fast_path(cm)


# ---------------------------------------------------------------------------
#  Observation/screenshot requests → act (not fast path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_observation_routes_to_act_not_fast_path(initialized_cm):
    """A screen observation request should route through act, not a fast path.
    Only atomic actions (click, type, scroll) use the desktop_act fast path."""
    cm = initialized_cm
    _ensure_mock_computer_primitives()

    await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Search the web for the company intranet login page URL",
        ),
    )

    _setup_desktop_fast_path_from_real_act(cm)
    cm.all_tool_calls.clear()

    try:
        result = await cm.step_until_wait(
            UnifyMessageReceived(
                contact=BOSS,
                content="What text is currently visible on the desktop screen?",
            ),
        )

        desktop_calls = [c for c in cm.all_tool_calls if c.startswith("desktop_")]
        assert not desktop_calls, (
            f"Observation request should NOT use desktop fast path, "
            f"but got: {desktop_calls}"
        )
        assert_efficient(result, 5)
    finally:
        _teardown_desktop_fast_path(cm)


# ---------------------------------------------------------------------------
#  Complex requests → act (not fast path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_complex_task_routes_to_act(initialized_cm):
    """A multi-step task should route through act, not the fast path."""
    cm = initialized_cm
    _ensure_mock_computer_primitives()

    await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Search the web for best practices for desktop application configuration",
        ),
    )

    _setup_desktop_fast_path_from_real_act(cm)
    cm.all_tool_calls.clear()

    try:
        result = await cm.step_until_wait(
            UnifyMessageReceived(
                contact=BOSS,
                content=(
                    "Copy all the sales data from the spreadsheet into "
                    "our standard Word template and format it nicely"
                ),
            ),
        )

        desktop_calls = [c for c in cm.all_tool_calls if c.startswith("desktop_")]
        assert not desktop_calls, (
            f"Complex multi-step task should NOT use desktop fast paths, "
            f"but got: {desktop_calls}"
        )
    finally:
        _teardown_desktop_fast_path(cm)
