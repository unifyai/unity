"""An actor is told which routes exist before it is told how to use them.

``primitives.computer.desktop`` and ``primitives.computer.web`` are present on
the namespace whether or not a machine backs them, and the prompt described
both in full regardless. On an assistant without the add-on that is an
instruction manual for something that cannot run: the actor reaches for a
browser, is refused, and reads the refusal as a fault worth retrying rather
than as a capability this assistant does not have.

Naming the routes that *do* work is the part that stops the retry loop --
connected workspaces, integrations, and a plain public-URL fetch cover most of
what a browser was being reached for.

The converse also holds: ``primitives.computer.user_desktop`` is gated on
per-user links and consent, never on the managed-desktop add-on. An
unentitled assistant with a linked user desktop holds a live, sensitive
capability -- the prompt must teach it, and its consent/prohibition
contract, rather than declaring all of computer control unavailable.
"""

from __future__ import annotations

import pytest

from unify.actor.environments.computer import ComputerEnvironment
from unify.session_details import (
    SESSION_DETAILS,
    AssistantDetails,
    UserDesktopLink,
)


@pytest.fixture()
def assistant():
    original = SESSION_DETAILS.assistant
    SESSION_DETAILS.assistant = AssistantDetails()
    yield SESSION_DETAILS.assistant
    SESSION_DETAILS.assistant = original


def _link_user_desktop(assistant, user_id: str = "user-1") -> None:
    assistant.user_desktops[user_id] = UserDesktopLink(
        owner_user_id=user_id,
        url="https://tunnel.example/abc",
        os="macos",
    )


def _context(env_self) -> str:
    """Call the prompt builder without constructing a live environment."""
    return ComputerEnvironment.get_prompt_context(env_self)


def _real_context() -> str:
    """Build the prompt from a live environment (shapes that read env state)."""
    from unify.function_manager.primitives import ComputerPrimitives

    return ComputerEnvironment(
        ComputerPrimitives(computer_mode="mock"),
    ).get_prompt_context()


class TestWithoutADesktop:
    def test_the_section_says_it_is_unavailable(self, assistant):
        assistant.desktop_mode = "none"
        assistant.managed_desktop_status = None

        context = _context(object())

        assert "NOT AVAILABLE" in context

    def test_it_names_the_routes_that_still_work(self, assistant):
        assistant.desktop_mode = "none"
        assistant.managed_desktop_status = None

        context = _context(object())

        assert "primitives.web.fetch" in context
        assert "connected workspace" in context.lower()
        assert "integration" in context.lower()

    def test_it_says_the_failure_is_not_transient(self, assistant):
        # Otherwise the actor retries a capability that will never appear.
        assistant.desktop_mode = "none"
        assistant.managed_desktop_status = None

        context = _context(object())

        assert "not" in context.lower() and "transient" in context.lower()

    def test_the_anonymous_link_case_is_pointed_at_the_fetch(self, assistant):
        # The one thing a browser was genuinely needed for on this path.
        assistant.desktop_mode = "none"
        assistant.managed_desktop_status = None

        assert "anyone with the link" in _context(object()).lower()

    def test_an_addon_switched_off_reads_as_unavailable(self, assistant):
        assistant.desktop_mode = "ubuntu"
        assistant.managed_desktop_status = "inactive"

        assert "NOT AVAILABLE" in _context(object())

    def test_user_desktops_are_not_taught_when_none_is_linked(self, assistant):
        # Teaching an unlinked capability recreates the retry loop the stub
        # exists to prevent.
        assistant.desktop_mode = "none"
        assistant.managed_desktop_status = None

        assert "user_desktop" not in _context(object())


class TestWithoutADesktopButWithALinkedUserDesktop:
    """User desktops work without the add-on, so the section must still
    teach them -- and their safety contract -- while keeping the assistant's
    own desktop and web lanes marked unavailable."""

    @pytest.fixture()
    def linked(self, assistant):
        assistant.desktop_mode = "none"
        assistant.managed_desktop_status = None
        _link_user_desktop(assistant)
        return assistant

    def test_the_own_desktop_lanes_stay_unavailable(self, linked):
        context = _real_context()

        assert "NOT AVAILABLE" in context
        assert "primitives.computer.desktop" in context
        assert "primitives.computer.web" in context
        assert "not" in context.lower() and "transient" in context.lower()

    def test_the_user_desktop_route_is_taught(self, linked):
        context = _real_context()

        assert "primitives.computer.user_desktop" in context
        assert "user_desktop.list_linked()" in context
        assert "session(user_id=...)" in context

    def test_the_consent_contract_is_present(self, linked):
        context = _real_context()

        assert "only on explicit request" in context
        assert "PermissionError" in context
        assert "never modify their machine" in context

    def test_the_file_harvesting_prohibition_is_present(self, linked):
        context = _real_context()

        assert "primitives.computer.user_desktop.files" in context
        assert "Never harvest their files" in context
        assert "~/Unity/Remote/<user_id>/" in context

    def test_the_no_desktop_routes_are_still_named(self, linked):
        context = _real_context()

        assert "primitives.web.fetch" in context
        assert "connected workspace" in context.lower()

    def test_the_managed_desktop_is_not_taught(self, linked):
        context = _real_context()

        assert "#### Managed desktop filesystem" not in context
        assert "new_session" not in context

    def test_screen_reading_guidance_targets_the_session_handle(self, linked):
        context = _real_context()

        assert "### Viewing Computer State" in context
        assert "display(await session.get_screenshot())" in context

    def test_the_addon_pitch_carves_out_the_linked_machine(self, linked):
        # "Say the add-on is needed" must not read as forbidding the one
        # machine the user has explicitly linked.
        context = _real_context()

        assert "Desktop Computer add-on" in context
        assert "a machine a user has linked" in context


class TestWithADesktop:
    @pytest.fixture()
    def entitled(self, assistant):
        assistant.desktop_mode = "ubuntu"
        assistant.managed_desktop_status = "active"
        return assistant

    def test_the_full_section_renders(self, entitled):
        context = _real_context()

        assert "NOT AVAILABLE" not in context
        assert "### Computer Control" in context
        assert "#### Managed desktop filesystem" in context
        assert "primitives.computer.web.new_session" in context

    def test_the_user_desktop_contract_is_present(self, entitled):
        context = _real_context()

        assert "### Your Desktop vs. a User's Desktop" in context
        assert "only on explicit request" in context
        assert "Never harvest their files" in context
