"""Tests for subagent environment support (create_env, AgentContext, env forwarding, guidance)."""

import pytest

from unity.actor.code_act_actor import (
    AgentContext,
    get_current_agent_context,
    _CURRENT_AGENT_CONTEXT,
)
from unity.actor.environments import create_env, BaseEnvironment
from unity.actor.environments.actor import (
    _resolve_parent_environments,
)
from unity.actor.execution.session import _CURRENT_ENVIRONMENTS


class TestCreateEnv:
    """Tests for the create_env() factory function."""

    def test_create_env_returns_base_environment(self):
        """create_env should return a BaseEnvironment instance."""

        class DummyService:
            async def do_something(self, task: str) -> str:
                """Does something with a task."""
                return f"done: {task}"

        env = create_env("dummy", DummyService())
        assert isinstance(env, BaseEnvironment)
        assert env.namespace == "dummy"

    def test_create_env_instance_accessible(self):
        """The service instance should be accessible via get_instance()."""

        class MyService:
            def __init__(self):
                self.value = 42

            async def get_value(self) -> int:
                """Returns the stored value."""
                return self.value

        service = MyService()
        env = create_env("myservice", service)
        assert env.get_instance() is service

    def test_create_env_prompt_context_includes_methods(self):
        """get_prompt_context() should list public methods with signatures."""

        class AgentService:
            async def run_subagent(self, task: str, timeout: int = 30) -> str:
                """Spawn a subagent to handle a task."""
                return "result"

            async def search_web(self, query: str) -> dict:
                """Search the web for information."""
                return {}

        env = create_env("agents", AgentService())
        context = env.get_prompt_context()

        assert "agents" in context
        assert "run_subagent" in context
        assert "search_web" in context
        # Should include docstrings
        assert "Spawn a subagent" in context
        assert "Search the web" in context

    def test_create_env_excludes_private_methods(self):
        """Private methods (starting with _) should not appear in prompt context."""

        class ServiceWithPrivate:
            async def public_method(self) -> str:
                """A public method."""
                return "public"

            async def _private_method(self) -> str:
                """A private method."""
                return "private"

        env = create_env("svc", ServiceWithPrivate())
        context = env.get_prompt_context()

        assert "public_method" in context
        assert "_private_method" not in context

    def test_create_env_get_tools_returns_method_metadata(self):
        """get_tools() should return ToolMetadata for each public method."""

        class DummyService:
            async def method(self) -> None:
                pass

        env = create_env("dummy", DummyService())
        tools = env.get_tools()
        assert "dummy.method" in tools
        assert tools["dummy.method"].name == "dummy.method"

    @pytest.mark.asyncio
    async def test_create_env_capture_state(self):
        """capture_state() should return service metadata."""

        class DummyService:
            pass

        env = create_env("myns", DummyService())
        state = await env.capture_state()

        assert state["type"] == "service"
        assert state["namespace"] == "myns"


class TestAgentContext:
    """Tests for AgentContext and get_current_agent_context()."""

    def test_default_context_has_depth_zero(self):
        """Default AgentContext should have depth 0."""
        ctx = AgentContext()
        assert ctx.depth == 0
        assert ctx.handle is None
        assert ctx.agent_id is not None  # Auto-generated

    def test_agent_id_is_unique(self):
        """Each AgentContext should have a unique agent_id."""
        ctx1 = AgentContext()
        ctx2 = AgentContext()
        assert ctx1.agent_id != ctx2.agent_id

    def test_get_current_agent_context_returns_default(self):
        """get_current_agent_context() should return default context when not in act()."""
        ctx = get_current_agent_context()
        assert ctx.depth == 0

    def test_context_depth_can_be_set(self):
        """AgentContext depth can be set explicitly."""
        ctx = AgentContext(depth=3)
        assert ctx.depth == 3

    def test_context_is_mutable(self):
        """AgentContext fields can be updated after creation."""
        ctx = AgentContext(depth=1, handle=None)
        assert ctx.handle is None

        # Simulate setting handle after creation
        mock_handle = object()
        ctx.handle = mock_handle
        assert ctx.handle is mock_handle

    def test_context_var_set_and_reset(self):
        """ContextVar can be set and reset correctly."""
        original = _CURRENT_AGENT_CONTEXT.get()

        # Set a new context
        new_ctx = AgentContext(depth=5)
        token = _CURRENT_AGENT_CONTEXT.set(new_ctx)

        # Verify it's set
        assert get_current_agent_context().depth == 5

        # Reset
        _CURRENT_AGENT_CONTEXT.reset(token)

        # Verify reset worked
        assert get_current_agent_context().depth == original.depth


class TestResolveParentEnvironments:
    """Tests for _resolve_parent_environments()."""

    def _make_env(self, namespace: str) -> BaseEnvironment:
        """Create a dummy environment with the given namespace."""

        class Svc:
            async def do_work(self) -> str:
                """Placeholder."""
                return "ok"

        return create_env(namespace, Svc())

    def test_no_prompt_functions_returns_empty(self):
        """When prompt_functions is None or empty, return empty forwarded list."""
        forwarded, remaining = _resolve_parent_environments(None)
        assert forwarded == []
        assert remaining == []

        forwarded, remaining = _resolve_parent_environments([])
        assert forwarded == []
        assert remaining == []

    def test_no_parent_envs_returns_all_as_remaining(self):
        """When no parent environments are set, everything goes to remaining."""
        forwarded, remaining = _resolve_parent_environments(
            ["colliers", "primitives.files"],
        )
        assert forwarded == []
        assert remaining == ["colliers", "primitives.files"]

    def test_matches_parent_namespace(self):
        """Custom parent namespaces should be forwarded."""
        colliers_env = self._make_env("colliers")
        token = _CURRENT_ENVIRONMENTS.set({"colliers": colliers_env})
        try:
            forwarded, remaining = _resolve_parent_environments(
                ["colliers", "primitives.files"],
            )
            assert len(forwarded) == 1
            assert forwarded[0] is colliers_env
            assert remaining == ["primitives.files"]
        finally:
            _CURRENT_ENVIRONMENTS.reset(token)

    def test_primitives_never_forwarded(self):
        """The 'primitives' namespace must always go through DB resolution."""

        prim_env = self._make_env("primitives")
        token = _CURRENT_ENVIRONMENTS.set({"primitives": prim_env})
        try:
            forwarded, remaining = _resolve_parent_environments(
                ["primitives", "primitives.contacts"],
            )
            assert forwarded == []
            assert remaining == ["primitives", "primitives.contacts"]
        finally:
            _CURRENT_ENVIRONMENTS.reset(token)

    def test_deduplicates_same_namespace(self):
        """Multiple references to the same parent namespace should forward only once."""
        rendering_env = self._make_env("rendering")
        token = _CURRENT_ENVIRONMENTS.set({"rendering": rendering_env})
        try:
            forwarded, remaining = _resolve_parent_environments(
                ["rendering", "rendering.render_pdf"],
            )
            assert len(forwarded) == 1
            assert forwarded[0] is rendering_env
            assert remaining == []
        finally:
            _CURRENT_ENVIRONMENTS.reset(token)

    def test_mixed_parent_and_db(self):
        """Mix of parent envs and DB names are correctly split."""
        colliers_env = self._make_env("colliers")
        rendering_env = self._make_env("rendering")
        token = _CURRENT_ENVIRONMENTS.set(
            {
                "colliers": colliers_env,
                "rendering": rendering_env,
            },
        )
        try:
            forwarded, remaining = _resolve_parent_environments(
                ["colliers", "primitives.contacts.ask", "rendering", "my_function"],
            )
            assert len(forwarded) == 2
            assert colliers_env in forwarded
            assert rendering_env in forwarded
            assert remaining == ["primitives.contacts.ask", "my_function"]
        finally:
            _CURRENT_ENVIRONMENTS.reset(token)


class TestCurrentEnvironmentsContextVar:
    """Tests for the _CURRENT_ENVIRONMENTS ContextVar lifecycle."""

    def test_default_is_empty_dict(self):
        """Default value should be an empty dict."""
        assert _CURRENT_ENVIRONMENTS.get({}) == {}

    def test_set_and_reset(self):
        """ContextVar should be settable and resettable."""
        env = create_env("test_ns", type("Svc", (), {"work": lambda self: None})())
        envs = {"test_ns": env}

        token = _CURRENT_ENVIRONMENTS.set(envs)
        assert _CURRENT_ENVIRONMENTS.get({}) is envs

        _CURRENT_ENVIRONMENTS.reset(token)
        assert _CURRENT_ENVIRONMENTS.get({}) == {}
