"""
Unit tests for MockComputerBackend.

These tests verify that the mock backend:
1. Implements all ComputerBackend abstract methods
2. Returns configurable canned responses
3. Implements additional methods used by tests (barrier, interrupt_current_action, etc.)
4. Works without any external services
"""

import pytest
from pydantic import BaseModel

from unity.function_manager.computer_backends import (
    MockComputerBackend,
    ComputerBackend,
)
from unity.function_manager.computer import Computer


class TestMockComputerBackendInterface:
    """Verify MockComputerBackend implements the ComputerBackend interface."""

    def test_is_computer_backend(self):
        """MockComputerBackend should be a subclass of ComputerBackend."""
        assert issubclass(MockComputerBackend, ComputerBackend)

    def test_can_instantiate_without_args(self):
        """Should instantiate without any arguments."""
        backend = MockComputerBackend()
        assert backend is not None

    def test_can_instantiate_with_kwargs(self):
        """Should accept configuration kwargs."""
        backend = MockComputerBackend(
            url="https://test.com",
            screenshot="test_screenshot",
            act_response="custom_act",
            observe_response="custom_observe",
        )
        assert backend._url == "https://test.com"
        assert backend._screenshot == "test_screenshot"
        assert backend._act_response == "custom_act"
        assert backend._observe_response == "custom_observe"


class TestMockComputerBackendMethods:
    """Verify all ComputerBackend methods work correctly."""

    @pytest.fixture
    def backend(self):
        return MockComputerBackend(
            url="https://example.com",
            screenshot="base64_screenshot",
            act_response="done",
            observe_response="I see a login form",
        )

    @pytest.mark.asyncio
    async def test_act(self, backend):
        """act() should return configured response."""
        result = await backend.act("Click the button")
        assert result == "done"

    @pytest.mark.asyncio
    async def test_observe(self, backend):
        """observe() should return configured response."""
        result = await backend.observe("What do you see?")
        assert result == "I see a login form"

    @pytest.mark.asyncio
    async def test_observe_with_pydantic_model(self, backend):
        """observe() with Pydantic model should try to create instance."""

        class TestModel(BaseModel):
            value: str = "default"

        result = await backend.observe("query", response_format=TestModel)
        assert isinstance(result, TestModel)
        assert result.value == "default"

    @pytest.mark.asyncio
    async def test_query(self, backend):
        """query() should return configured response."""
        result = await backend.query("What happened?")
        assert result == "Mock query response"

    @pytest.mark.asyncio
    async def test_get_screenshot(self, backend):
        """get_screenshot() should return configured screenshot."""
        result = await backend.get_screenshot()
        assert result == "base64_screenshot"

    @pytest.mark.asyncio
    async def test_get_current_url(self, backend):
        """get_current_url() should return configured URL."""
        result = await backend.get_current_url()
        assert result == "https://example.com"

    @pytest.mark.asyncio
    async def test_navigate(self, backend):
        """navigate() should update URL and return success."""
        result = await backend.navigate("https://new-url.com")
        assert result == "success"
        assert await backend.get_current_url() == "https://new-url.com"

    @pytest.mark.asyncio
    async def test_get_links(self, backend):
        """get_links() should return empty links response."""
        result = await backend.get_links()
        assert "links" in result
        assert result["links"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_get_content(self, backend):
        """get_content() should return minimal content response."""
        result = await backend.get_content()
        assert "content" in result
        assert "url" in result
        assert result["format"] == "markdown"

    def test_stop(self, backend):
        """stop() should be a no-op (not raise)."""
        backend.stop()  # Should not raise


class TestMockComputerBackendExtras:
    """Verify additional methods used by tests."""

    @pytest.fixture
    def backend(self):
        return MockComputerBackend()

    @pytest.mark.asyncio
    async def test_barrier(self, backend):
        """barrier() should be a no-op."""
        await backend.barrier()
        await backend.barrier(up_to_seq=5)  # Should not raise

    @pytest.mark.asyncio
    async def test_interrupt_current_action(self, backend):
        """interrupt_current_action() should be a no-op."""
        await backend.interrupt_current_action()

    @pytest.mark.asyncio
    async def test_clear_pending_commands(self, backend):
        """clear_pending_commands() should be a no-op."""
        await backend.clear_pending_commands(run_id=123)

    def test_current_seq(self, backend):
        """current_seq should track command sequence."""
        assert backend.current_seq == 0

    @pytest.mark.asyncio
    async def test_seq_increments(self, backend):
        """Sequence should increment with commands."""
        assert backend.current_seq == 0
        await backend.act("action 1")
        assert backend.current_seq == 1
        await backend.observe("query")
        assert backend.current_seq == 2
        await backend.navigate("https://test.com")
        assert backend.current_seq == 3


class TestComputerWithMockMode:
    """Verify Computer class works with mode='mock'."""

    def test_computer_mock_mode(self):
        """Computer should accept mode='mock'."""
        computer = Computer(mode="mock")
        assert isinstance(computer.backend, MockComputerBackend)

    @pytest.mark.asyncio
    async def test_computer_mock_mode_act(self):
        """Computer with mock mode should delegate to MockComputerBackend."""
        computer = Computer(mode="mock")
        result = await computer.act("Click button")
        assert result == "done"

    @pytest.mark.asyncio
    async def test_computer_mock_mode_get_url(self):
        """Computer with mock mode should return mock URL."""
        computer = Computer(mode="mock")
        url = await computer.get_current_url()
        assert url == "https://mock.example.com"
