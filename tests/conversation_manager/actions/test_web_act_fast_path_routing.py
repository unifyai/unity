"""
tests/conversation_manager/actions/test_web_act_fast_path_routing.py
=====================================================================

Eval tests verifying the CM brain routes:
- Web search / navigation requests to ``web_act`` (not ``desktop_act``)
- Native desktop app requests to ``desktop_act`` (not ``web_act``)
- Complex cross-domain requests to ``act`` (not fast paths)
- Browser close requests to ``close_web_session``

Follows the same end-to-end pattern as test_desktop_fast_path_routing.py.
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


def _enable_fast_paths(cm_driver):
    """Activate fast-path tools by turning on screen share."""
    cm_driver.cm.assistant_screen_share_active = True


def _setup_fast_paths_from_real_act(cm_driver):
    """Activate fast-path gating.

    Call this AFTER a ``step_until_wait`` that triggered ``act``.
    """
    cm_driver.cm.assistant_screen_share_active = True


def _teardown_fast_paths(cm_driver):
    """Reset gating state so subsequent tests start clean."""
    cm_driver.cm.assistant_screen_share_active = False


# ---------------------------------------------------------------------------
#  Web search → web_act
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_web_search_routes_to_web_act(initialized_cm):
    """A web search request should route to web_act when fast paths are active."""
    cm = initialized_cm
    _ensure_mock_computer_primitives()

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Help me look up best practices for CRM configuration",
        ),
    )
    assert "act" in cm.all_tool_calls

    _setup_fast_paths_from_real_act(cm)
    cm.all_tool_calls.clear()

    try:
        result = await cm.step_until_wait(
            UnifyMessageReceived(
                contact=BOSS,
                content="Search the web for 'best CRM software 2025 reviews'",
            ),
        )

        assert "web_act" in cm.all_tool_calls, (
            f"Expected 'web_act' for web search request, " f"got: {cm.all_tool_calls}"
        )
        assert_efficient(result, 5)
    finally:
        _teardown_fast_paths(cm)


# ---------------------------------------------------------------------------
#  Web navigation → web_act
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_web_navigation_routes_to_web_act(initialized_cm):
    """Navigating to a URL should route to web_act."""
    cm = initialized_cm
    _ensure_mock_computer_primitives()

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Help me review our competitor's product offerings",
        ),
    )

    _setup_fast_paths_from_real_act(cm)
    cm.all_tool_calls.clear()

    try:
        result = await cm.step_until_wait(
            UnifyMessageReceived(
                contact=BOSS,
                content="Go to example.com and check their pricing page",
            ),
        )

        assert "web_act" in cm.all_tool_calls, (
            f"Expected 'web_act' for web navigation, " f"got: {cm.all_tool_calls}"
        )
        assert_efficient(result, 5)
    finally:
        _teardown_fast_paths(cm)


# ---------------------------------------------------------------------------
#  Native desktop app → desktop_act (NOT web_act)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_native_app_routes_to_desktop_act_not_web_act(initialized_cm):
    """Opening a native desktop app should use desktop_act, not web_act."""
    cm = initialized_cm
    _ensure_mock_computer_primitives()

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Help me set up the desktop environment",
        ),
    )

    _setup_fast_paths_from_real_act(cm)
    cm.all_tool_calls.clear()

    try:
        result = await cm.step_until_wait(
            UnifyMessageReceived(
                contact=BOSS,
                content="Open the Terminal application on the desktop",
            ),
        )

        assert "desktop_act" in cm.all_tool_calls, (
            f"Expected 'desktop_act' for native app request, "
            f"got: {cm.all_tool_calls}"
        )
        assert "web_act" not in cm.all_tool_calls, (
            f"Native app request should NOT use web_act, " f"got: {cm.all_tool_calls}"
        )
        assert_efficient(result, 5)
    finally:
        _teardown_fast_paths(cm)


# ---------------------------------------------------------------------------
#  Complex cross-domain → act (not fast paths)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_complex_cross_domain_routes_to_act(initialized_cm):
    """A request combining web search with contact management should route
    through act, not fast paths."""
    cm = initialized_cm
    _ensure_mock_computer_primitives()

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Help me with some research and data entry",
        ),
    )

    _setup_fast_paths_from_real_act(cm)
    cm.all_tool_calls.clear()

    try:
        result = await cm.step_until_wait(
            UnifyMessageReceived(
                contact=BOSS,
                content=(
                    "Search the web for John Smith's company details and "
                    "then save his email and phone number to my contacts"
                ),
            ),
        )

        web_or_desktop_calls = [
            c
            for c in cm.all_tool_calls
            if c.startswith("web_act") or c.startswith("desktop_")
        ]
        assert not web_or_desktop_calls, (
            f"Complex cross-domain task should NOT use fast paths, "
            f"but got: {web_or_desktop_calls}"
        )
    finally:
        _teardown_fast_paths(cm)
