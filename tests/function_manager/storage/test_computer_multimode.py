"""
Tests for the multi-mode ComputerPrimitives API.

Covers:
- Desktop singleton namespace (primitives.computer.desktop.*)
- Web session factory (primitives.computer.web.new_session())
- WebSessionHandle lifecycle (create, use, stop)
- Concurrent web sessions
- ComputerEnvironment tool discovery for the new surface
"""

import asyncio

import pytest
from PIL import Image

from unity.manager_registry import ManagerRegistry


@pytest.fixture(autouse=True)
def _clear_registry():
    ManagerRegistry.clear()
    yield
    ManagerRegistry.clear()


def _make_primitives():
    from unity.function_manager.primitives.runtime import ComputerPrimitives

    return ComputerPrimitives(computer_mode="mock")


# ── Desktop namespace ─────────────────────────────────────────────────


class TestDesktopNamespace:
    """primitives.computer.desktop is a singleton with all standard methods."""

    def test_desktop_property_returns_namespace(self):
        cp = _make_primitives()
        ns = cp.desktop
        assert ns is not None
        assert cp.desktop is ns  # same object on repeated access

    def test_desktop_has_all_methods(self):
        cp = _make_primitives()
        ns = cp.desktop
        for name in (
            "act",
            "observe",
            "query",
            "navigate",
            "get_links",
            "get_content",
            "get_screenshot",
        ):
            assert callable(getattr(ns, name)), f"desktop.{name} should be callable"

    @pytest.mark.asyncio
    async def test_desktop_act(self):
        from unity.function_manager.computer_backends import ActResult

        cp = _make_primitives()
        result = await cp.desktop.act("Click the Start menu")
        assert isinstance(result, ActResult)
        assert result.summary == "done"
        assert result.screenshot  # non-empty base64 screenshot
        assert str(result) == "done"  # backward compat

    @pytest.mark.asyncio
    async def test_desktop_navigate(self):
        cp = _make_primitives()
        result = await cp.desktop.navigate("https://example.com")
        assert result == "success"

    @pytest.mark.asyncio
    async def test_desktop_observe(self):
        cp = _make_primitives()
        result = await cp.desktop.observe("What is on the screen?")
        assert result == "Mock observation"

    @pytest.mark.asyncio
    async def test_desktop_get_screenshot_returns_pil_image(self):
        cp = _make_primitives()
        img = await cp.desktop.get_screenshot()
        assert isinstance(img, Image.Image)

    @pytest.mark.asyncio
    async def test_desktop_get_links(self):
        cp = _make_primitives()
        result = await cp.desktop.get_links()
        assert "links" in result

    @pytest.mark.asyncio
    async def test_desktop_get_content(self):
        cp = _make_primitives()
        result = await cp.desktop.get_content()
        assert "content" in result


# ── Web session factory ───────────────────────────────────────────────


class TestWebSessionFactory:
    """primitives.computer.web is a factory with new_session()."""

    def test_web_property_returns_factory(self):
        from unity.function_manager.primitives.runtime import _WebSessionFactory

        cp = _make_primitives()
        factory = cp.web
        assert isinstance(factory, _WebSessionFactory)
        assert cp.web is factory  # same object on repeated access

    def test_factory_has_new_session(self):
        cp = _make_primitives()
        assert callable(cp.web.new_session)

    def test_factory_has_no_act_or_navigate(self):
        """The factory itself should NOT have convenience methods."""
        cp = _make_primitives()
        assert not hasattr(cp.web, "act")
        assert not hasattr(cp.web, "navigate")
        assert not hasattr(cp.web, "observe")


# ── WebSessionHandle ──────────────────────────────────────────────────


class TestWebSessionHandle:
    """Session handles returned by new_session()."""

    @pytest.mark.asyncio
    async def test_new_session_returns_handle(self):
        from unity.function_manager.primitives.runtime import WebSessionHandle

        cp = _make_primitives()
        session = await cp.web.new_session()
        assert isinstance(session, WebSessionHandle)

    @pytest.mark.asyncio
    async def test_handle_has_all_methods(self):
        cp = _make_primitives()
        session = await cp.web.new_session()
        for name in (
            "act",
            "observe",
            "query",
            "navigate",
            "get_links",
            "get_content",
            "get_screenshot",
            "stop",
        ):
            assert callable(
                getattr(session, name),
            ), f"session.{name} should be callable"

    @pytest.mark.asyncio
    async def test_handle_act(self):
        cp = _make_primitives()
        session = await cp.web.new_session()
        result = await session.act("Click the button")
        assert str(result) == "done"

    @pytest.mark.asyncio
    async def test_handle_navigate(self):
        cp = _make_primitives()
        session = await cp.web.new_session()
        result = await session.navigate("https://example.com")
        assert result == "success"

    @pytest.mark.asyncio
    async def test_handle_observe(self):
        cp = _make_primitives()
        session = await cp.web.new_session()
        result = await session.observe("What is on the page?")
        assert result == "Mock observation"

    @pytest.mark.asyncio
    async def test_handle_get_screenshot_returns_pil_image(self):
        cp = _make_primitives()
        session = await cp.web.new_session()
        img = await session.get_screenshot()
        assert isinstance(img, Image.Image)

    @pytest.mark.asyncio
    async def test_handle_stop(self):
        cp = _make_primitives()
        session = await cp.web.new_session()
        await session.stop()  # should not raise

    @pytest.mark.asyncio
    async def test_visible_true_default(self):
        """new_session() defaults to visible=True."""
        cp = _make_primitives()
        session = await cp.web.new_session()
        assert session._session._mode == "web-vm"

    @pytest.mark.asyncio
    async def test_visible_false(self):
        """new_session(visible=False) creates a headless session."""
        cp = _make_primitives()
        session = await cp.web.new_session(visible=False)
        assert session._session._mode == "web"

    @pytest.mark.asyncio
    async def test_visible_true_explicit(self):
        cp = _make_primitives()
        session = await cp.web.new_session(visible=True)
        assert session._session._mode == "web-vm"


# ── Concurrent web sessions ──────────────────────────────────────────


class TestConcurrentSessions:
    """Multiple web sessions should operate independently."""

    @pytest.mark.asyncio
    async def test_two_sessions_are_independent(self):
        cp = _make_primitives()
        s1 = await cp.web.new_session()
        s2 = await cp.web.new_session()
        assert s1 is not s2
        assert s1._session is not s2._session

    @pytest.mark.asyncio
    async def test_parallel_navigation(self):
        """Navigate two sessions to different URLs concurrently."""
        cp = _make_primitives()
        s1 = await cp.web.new_session()
        s2 = await cp.web.new_session()

        await asyncio.gather(
            s1.navigate("https://alpha.com"),
            s2.navigate("https://beta.com"),
        )

        # Both should have completed without error
        await s1.stop()
        await s2.stop()

    @pytest.mark.asyncio
    async def test_mixed_visible_and_headless(self):
        """Can have visible and headless sessions simultaneously."""
        cp = _make_primitives()
        visible = await cp.web.new_session(visible=True)
        headless = await cp.web.new_session(visible=False)

        assert visible._session._mode == "web-vm"
        assert headless._session._mode == "web"

        await visible.stop()
        await headless.stop()

    @pytest.mark.asyncio
    async def test_desktop_and_web_coexist(self):
        """Desktop namespace and web sessions work simultaneously."""
        cp = _make_primitives()

        desktop_result = await cp.desktop.act("Click something")
        session = await cp.web.new_session()
        web_result = await session.navigate("https://example.com")

        assert str(desktop_result) == "done"
        assert web_result == "success"
        await session.stop()


# ── ComputerEnvironment tool discovery ────────────────────────────────


class TestEnvironmentToolDiscovery:
    """ComputerEnvironment exposes the correct tools for the new API."""

    def test_get_tools_includes_desktop_methods(self):
        from unity.actor.environments.computer import ComputerEnvironment

        cp = _make_primitives()
        env = ComputerEnvironment(cp)
        tools = env.get_tools()
        for method in (
            "act",
            "observe",
            "query",
            "navigate",
            "get_links",
            "get_content",
            "get_screenshot",
        ):
            fq = f"primitives.computer.desktop.{method}"
            assert fq in tools, f"Expected tool {fq} in get_tools()"

    def test_get_tools_includes_web_factory(self):
        from unity.actor.environments.computer import ComputerEnvironment

        cp = _make_primitives()
        env = ComputerEnvironment(cp)
        tools = env.get_tools()
        assert "primitives.computer.web.new_session" in tools

    def test_get_tools_does_not_include_flat_methods(self):
        """Tools should NOT be at primitives.computer.act (flat, no mode prefix)."""
        from unity.actor.environments.computer import ComputerEnvironment

        cp = _make_primitives()
        env = ComputerEnvironment(cp)
        tools = env.get_tools()
        for method in ("act", "observe", "navigate"):
            flat = f"primitives.computer.{method}"
            assert flat not in tools, f"Flat tool name {flat} should not exist"

    def test_new_session_tool_has_docstring(self):
        from unity.actor.environments.computer import ComputerEnvironment

        cp = _make_primitives()
        env = ComputerEnvironment(cp)
        tools = env.get_tools()
        meta = tools["primitives.computer.web.new_session"]
        assert meta.docstring is not None
        assert "visible" in meta.docstring
        assert "browser" in meta.docstring.lower()

    def test_new_session_tool_has_signature(self):
        from unity.actor.environments.computer import ComputerEnvironment

        cp = _make_primitives()
        env = ComputerEnvironment(cp)
        tools = env.get_tools()
        meta = tools["primitives.computer.web.new_session"]
        assert meta.signature is not None
        assert "visible" in meta.signature

    def test_get_tools_includes_list_sessions(self):
        from unity.actor.environments.computer import ComputerEnvironment

        cp = _make_primitives()
        env = ComputerEnvironment(cp)
        tools = env.get_tools()
        assert "primitives.computer.web.list_sessions" in tools

    def test_list_sessions_tool_has_docstring(self):
        from unity.actor.environments.computer import ComputerEnvironment

        cp = _make_primitives()
        env = ComputerEnvironment(cp)
        tools = env.get_tools()
        meta = tools["primitives.computer.web.list_sessions"]
        assert meta.docstring is not None
        assert "visible_only" in meta.docstring
        assert "active_only" in meta.docstring

    def test_prompt_context_mentions_both_interfaces(self):
        from unity.actor.environments.computer import ComputerEnvironment

        cp = _make_primitives()
        env = ComputerEnvironment(cp)
        ctx = env.get_prompt_context()
        assert "primitives.computer.desktop" in ctx
        assert "primitives.computer.web.new_session" in ctx
        assert "visible" in ctx
        assert "stop()" in ctx


# ── list_sessions ─────────────────────────────────────────────────────


class TestListSessions:
    """primitives.computer.web.list_sessions() returns handles from the global registry."""

    def test_list_sessions_empty_initially(self):
        cp = _make_primitives()
        assert cp.web.list_sessions() == []

    @pytest.mark.asyncio
    async def test_list_sessions_returns_created_sessions(self):
        cp = _make_primitives()
        s1 = await cp.web.new_session()
        s2 = await cp.web.new_session()
        sessions = cp.web.list_sessions()
        assert len(sessions) == 2
        assert s1 in sessions
        assert s2 in sessions

    @pytest.mark.asyncio
    async def test_list_sessions_active_only_filters_stopped(self):
        cp = _make_primitives()
        s1 = await cp.web.new_session()
        s2 = await cp.web.new_session()
        await s1.stop()
        active = cp.web.list_sessions(active_only=True)
        assert len(active) == 1
        assert s2 in active
        assert s1 not in active

    @pytest.mark.asyncio
    async def test_list_sessions_visible_only_filters_headless(self):
        cp = _make_primitives()
        visible = await cp.web.new_session(visible=True)
        headless = await cp.web.new_session(visible=False)
        result = cp.web.list_sessions(visible_only=True)
        assert len(result) == 1
        assert visible in result
        assert headless not in result

    @pytest.mark.asyncio
    async def test_list_sessions_combined_filters(self):
        cp = _make_primitives()
        v_active = await cp.web.new_session(visible=True)
        v_stopped = await cp.web.new_session(visible=True)
        h_active = await cp.web.new_session(visible=False)
        await v_stopped.stop()
        result = cp.web.list_sessions(visible_only=True, active_only=True)
        assert result == [v_active]
        assert h_active not in result

    @pytest.mark.asyncio
    async def test_list_sessions_returns_same_handle_objects(self):
        cp = _make_primitives()
        s1 = await cp.web.new_session()
        s2 = await cp.web.new_session()
        sessions = cp.web.list_sessions()
        assert sessions[0] is s1
        assert sessions[1] is s2

    @pytest.mark.asyncio
    async def test_handle_visible_property(self):
        cp = _make_primitives()
        visible = await cp.web.new_session(visible=True)
        headless = await cp.web.new_session(visible=False)
        assert visible.visible is True
        assert headless.visible is False

    @pytest.mark.asyncio
    async def test_handle_active_property(self):
        cp = _make_primitives()
        session = await cp.web.new_session()
        assert session.active is True
        await session.stop()
        assert session.active is False
