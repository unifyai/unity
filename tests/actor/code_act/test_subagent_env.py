"""Tests for subagent environment support (create_env and AgentContext)."""

import pytest

from unity.actor.code_act_actor import (
    AgentContext,
    get_current_agent_context,
    _CURRENT_AGENT_CONTEXT,
)
from unity.actor.environments import create_env, BaseEnvironment


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

    def test_create_env_get_tools_returns_empty(self):
        """get_tools() should return empty dict (ToolMetadata unused by CodeActActor)."""

        class DummyService:
            async def method(self) -> None:
                pass

        env = create_env("dummy", DummyService())
        assert env.get_tools() == {}

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
