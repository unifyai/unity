"""
tests/conversation_manager/actions/test_web_act_fast_path_routing.py
=====================================================================

Eval tests verifying the CM brain routes:
- Web search / navigation requests to ``web_act`` (not ``desktop_act``)
- Native desktop app requests to ``desktop_act`` (not ``web_act``)
- Complex cross-domain requests to ``act`` (not fast paths)
- Browser close requests to ``close_web_session``

Follows the same end-to-end pattern as test_desktop_fast_path_routing.py (computer fast-path routing).
"""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import (
    assert_efficient,
    has_steering_tool_call,
)
from tests.conversation_manager.conftest import BOSS
from unity.conversation_manager.events import (
    AssistantScreenShareStarted,
    InboundUnifyMeetUtterance,
    SMSReceived,
    UnifyMeetStarted,
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


def _enable_computer_fast_path(cm_driver):
    """Activate computer fast-path tools by turning on screen share."""
    cm_driver.cm.assistant_screen_share_active = True
    cm_driver.cm.vm_ready = True
    cm_driver.cm.file_sync_complete = True


def _setup_computer_fast_path_from_real_act(cm_driver):
    """Activate computer fast-path gating.

    Call this AFTER a ``step_until_wait`` that triggered ``act``.
    """
    cm_driver.cm.assistant_screen_share_active = True
    cm_driver.cm.vm_ready = True
    cm_driver.cm.file_sync_complete = True


def _teardown_computer_fast_path(cm_driver):
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
    cm.cm.vm_ready = True
    cm.cm.file_sync_complete = True

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Help me look up best practices for CRM configuration",
        ),
    )
    assert "act" in cm.all_tool_calls

    _setup_computer_fast_path_from_real_act(cm)
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
        _teardown_computer_fast_path(cm)


# ---------------------------------------------------------------------------
#  Web navigation → web_act
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_web_navigation_routes_to_web_act(initialized_cm):
    """Navigating to a URL should route to web_act."""
    cm = initialized_cm
    _ensure_mock_computer_primitives()
    cm.cm.vm_ready = True
    cm.cm.file_sync_complete = True

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Help me review our competitor's product offerings",
        ),
    )

    _setup_computer_fast_path_from_real_act(cm)
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
        _teardown_computer_fast_path(cm)


# ---------------------------------------------------------------------------
#  Web page interaction → web_act
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_web_page_scroll_routes_to_web_act(initialized_cm):
    """Scrolling on a web page should route to web_act, not desktop_act."""
    cm = initialized_cm
    _ensure_mock_computer_primitives()
    cm.cm.vm_ready = True
    cm.cm.file_sync_complete = True

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Search the web for our main competitor's pricing strategy",
        ),
    )
    assert "act" in cm.all_tool_calls

    _setup_computer_fast_path_from_real_act(cm)
    cm.all_tool_calls.clear()

    try:
        result = await cm.step_until_wait(
            UnifyMessageReceived(
                contact=BOSS,
                content="Scroll down on the web page to see more content",
            ),
        )

        assert "web_act" in cm.all_tool_calls, (
            f"Expected 'web_act' for web page scroll, " f"got: {cm.all_tool_calls}"
        )
        assert "desktop_act" not in cm.all_tool_calls, (
            f"Web page scroll should NOT use desktop_act, " f"got: {cm.all_tool_calls}"
        )
        assert_efficient(result, 5)
    finally:
        _teardown_computer_fast_path(cm)


# ---------------------------------------------------------------------------
#  Native desktop app → desktop_act (NOT web_act)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_native_app_routes_to_desktop_act_not_web_act(initialized_cm):
    """Opening a native desktop app should use desktop_act, not web_act."""
    cm = initialized_cm
    _ensure_mock_computer_primitives()
    cm.cm.vm_ready = True
    cm.cm.file_sync_complete = True

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Help me set up the desktop environment",
        ),
    )

    _setup_computer_fast_path_from_real_act(cm)
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
        _teardown_computer_fast_path(cm)


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
    cm.cm.vm_ready = True
    cm.cm.file_sync_complete = True

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Help me with some research and data entry",
        ),
    )

    _setup_computer_fast_path_from_real_act(cm)
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
        _teardown_computer_fast_path(cm)


# ---------------------------------------------------------------------------
#  Credentials / secrets → interject_* (NOT web_act)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_login_with_stored_credentials_routes_to_interject(initialized_cm):
    """Logging in with stored credentials must route to interject_*, NOT
    web_act, during an interactive Unify Meet screen-share session.

    web_act's browser agent has no access to primitives.secrets.  Only the
    CodeActActor sandbox can retrieve secret names and use the ${SECRET_NAME}
    placeholder syntax with type_text().  When the user asks to log in using
    stored credentials, the CM must interject the in-flight act session so
    the Actor can execute the credential flow.

    Reproduces the production scenario: Unify Meet with assistant screen
    share, an in-flight act session that has loaded guidance, and a user
    request to log in with saved credentials.
    """
    cm = initialized_cm
    _ensure_mock_computer_primitives()
    cm.cm.vm_ready = True
    cm.cm.file_sync_complete = True

    # Set up Unify Meet with assistant screen share (no LLM run for setup)
    await cm.step(UnifyMeetStarted(contact=BOSS), run_llm=False)
    await cm.step(AssistantScreenShareStarted(), run_llm=False)

    # Bootstrap: first instruction triggers act(persist=True)
    result = await cm.step_until_wait(
        InboundUnifyMeetUtterance(
            contact=BOSS,
            content=(
                "Let's go through CoStar. Open the browser and navigate "
                "to costar.com."
            ),
        ),
    )
    assert (
        "act" in cm.all_tool_calls
    ), f"Bootstrap should trigger act, got: {cm.all_tool_calls}"
    cm.all_tool_calls.clear()

    try:
        # Credential request — should interject the existing act, not web_act
        result = await cm.step_until_wait(
            InboundUnifyMeetUtterance(
                contact=BOSS,
                content=(
                    "Now log in using the stored credentials — you have the "
                    "username and password saved."
                ),
            ),
        )

        assert "web_act" not in cm.all_tool_calls, (
            f"Credential-dependent login should NOT use web_act (browser agent "
            f"has no access to primitives.secrets). "
            f"Got: {cm.all_tool_calls}"
        )
        assert has_steering_tool_call(cm, "interject_"), (
            f"Expected interject_* to relay credential-based login to the "
            f"in-flight act session which has access to primitives.secrets. "
            f"Got: {cm.all_tool_calls}"
        )
    finally:
        _teardown_computer_fast_path(cm)


# ---------------------------------------------------------------------------
#  Browser task (no act session) → web_act + act in same turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_browser_task_without_act_session_routes_to_web_act_and_act(
    initialized_cm,
):
    """When no act session is in-flight and screen share is active, a browser
    task should trigger both ``web_act`` (for the immediate action) AND ``act``
    with ``persist=True`` (for the full-capability session) in the same turn.

    This reproduces the production scenario where "open the browser" was
    incorrectly routed through only ``act``, missing the fast-path entirely.
    """
    cm = initialized_cm
    _ensure_mock_computer_primitives()
    cm.cm.vm_ready = True
    cm.cm.file_sync_complete = True

    await cm.step(UnifyMeetStarted(contact=BOSS), run_llm=False)
    await cm.step(AssistantScreenShareStarted(), run_llm=False)

    try:
        result = await cm.step_until_wait(
            InboundUnifyMeetUtterance(
                contact=BOSS,
                content="Open the browser for me please.",
            ),
        )

        assert "web_act" in cm.all_tool_calls, (
            f"Expected 'web_act' for browser-related request, "
            f"got: {cm.all_tool_calls}"
        )
        assert "act" in cm.all_tool_calls, (
            f"Expected concurrent 'act' session alongside web_act when no "
            f"act session is in-flight, got: {cm.all_tool_calls}"
        )
    finally:
        _teardown_computer_fast_path(cm)
