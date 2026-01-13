"""Tests for LLM event integration with EventBus.

These tests verify that:
1. The LLMPayload model correctly captures LLM request/response metadata
2. The hook converts unillm LLMEvent to EventBus events
3. LLM events are published to EventBus during actual LLM calls
4. The payload extracts useful information from responses (tokens, preview, etc.)
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
import unillm
from unillm import LLMEvent

from tests.helpers import _handle_project, capture_events
from unity.events.event_bus import EventBus, Event, EVENT_BUS
from unity.events.types.llm import LLMPayload
from unity.events.llm_event_hook import _llm_event_to_eventbus, install_llm_event_hook


# ---------------------------------------------------------------------------
#  1. LLMPayload model tests
# ---------------------------------------------------------------------------


class TestLLMPayloadModel:
    """Tests for the LLMPayload Pydantic model."""

    def test_create_request_payload(self):
        payload = LLMPayload(
            phase="request",
            endpoint="gpt-4o@openai",
            model="gpt-4o",
            provider="openai",
            stream=False,
            messages_count=3,
            tools_count=5,
        )
        assert payload.phase == "request"
        assert payload.endpoint == "gpt-4o@openai"
        assert payload.model == "gpt-4o"
        assert payload.provider == "openai"
        assert payload.stream is False
        assert payload.messages_count == 3
        assert payload.tools_count == 5
        # Response fields should be None for request phase
        assert payload.cache_status is None
        assert payload.response_model is None
        assert payload.prompt_tokens is None

    def test_create_response_payload(self):
        payload = LLMPayload(
            phase="response",
            endpoint="claude-4@anthropic",
            model="claude-4",
            provider="anthropic",
            stream=False,
            cache_status="miss",
            messages_count=2,
            tools_count=0,
            response_model="claude-4-20260115",
            prompt_tokens=150,
            completion_tokens=50,
            total_tokens=200,
            content_preview="Hello, how can I help...",
        )
        assert payload.phase == "response"
        assert payload.cache_status == "miss"
        assert payload.response_model == "claude-4-20260115"
        assert payload.prompt_tokens == 150
        assert payload.completion_tokens == 50
        assert payload.total_tokens == 200
        assert payload.content_preview == "Hello, how can I help..."

    def test_create_error_payload(self):
        payload = LLMPayload(
            phase="response",
            endpoint="gpt-4o@openai",
            model="gpt-4o",
            provider="openai",
            stream=False,
            cache_status="error",
            error="API rate limit exceeded",
            messages_count=1,
            tools_count=0,
        )
        assert payload.cache_status == "error"
        assert payload.error == "API rate limit exceeded"

    def test_streaming_payload(self):
        payload = LLMPayload(
            phase="request",
            endpoint="gpt-4o@openai",
            model="gpt-4o",
            provider="openai",
            stream=True,
            messages_count=1,
            tools_count=0,
        )
        assert payload.stream is True

    def test_payload_allows_extra_fields(self):
        """LLMPayload should accept extra fields for forward compatibility."""
        payload = LLMPayload(
            phase="request",
            endpoint="gpt-4o@openai",
            model="gpt-4o",
            provider="openai",
            custom_field="custom_value",
        )
        assert payload.model_extra.get("custom_field") == "custom_value"


# ---------------------------------------------------------------------------
#  2. Hook conversion tests
# ---------------------------------------------------------------------------


class TestLLMEventToEventBusConversion:
    """Tests for the _llm_event_to_eventbus hook function."""

    @pytest.mark.asyncio
    @_handle_project
    async def test_request_event_conversion(self):
        """Request events should be converted and published to EventBus."""
        async with capture_events("LLM") as captured:
            llm_event = LLMEvent(
                phase="request",
                endpoint="gpt-4o@openai",
                model="gpt-4o",
                provider="openai",
                request_kw={
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Hi"},
                    ],
                    "tools": [{"type": "function", "function": {"name": "search"}}],
                },
                stream=False,
            )
            _llm_event_to_eventbus(llm_event)

            # Give async publish time to complete
            await asyncio.sleep(0.05)

        assert len(captured) == 1
        evt = captured[0]
        assert evt.type == "LLM"
        assert evt.payload["phase"] == "request"
        assert evt.payload["endpoint"] == "gpt-4o@openai"
        assert evt.payload["model"] == "gpt-4o"
        assert evt.payload["provider"] == "openai"
        assert evt.payload["messages_count"] == 2
        assert evt.payload["tools_count"] == 1
        assert evt.payload["stream"] is False

    @pytest.mark.asyncio
    @_handle_project
    async def test_response_event_conversion(self):
        """Response events should include cache status and token usage."""
        # Create a mock response with usage info
        mock_response = MagicMock()
        mock_response.model = "gpt-4o-2024-08-06"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Hello! I'm here to help you."

        async with capture_events("LLM") as captured:
            llm_event = LLMEvent(
                phase="response",
                endpoint="gpt-4o@openai",
                model="gpt-4o",
                provider="openai",
                request_kw={"messages": [{"role": "user", "content": "Hi"}]},
                response=mock_response,
                cache_status="miss",
                stream=False,
            )
            _llm_event_to_eventbus(llm_event)

            await asyncio.sleep(0.05)

        assert len(captured) == 1
        evt = captured[0]
        assert evt.payload["phase"] == "response"
        assert evt.payload["cache_status"] == "miss"
        assert evt.payload["response_model"] == "gpt-4o-2024-08-06"
        assert evt.payload["prompt_tokens"] == 100
        assert evt.payload["completion_tokens"] == 50
        assert evt.payload["total_tokens"] == 150
        assert evt.payload["content_preview"] == "Hello! I'm here to help you."

    @pytest.mark.asyncio
    @_handle_project
    async def test_content_preview_truncation(self):
        """Long content should be truncated in the preview."""
        mock_response = MagicMock()
        mock_response.model = "gpt-4o"
        mock_response.usage = None
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        # Create a long response (over 200 chars)
        long_content = "x" * 300
        mock_response.choices[0].message.content = long_content

        async with capture_events("LLM") as captured:
            llm_event = LLMEvent(
                phase="response",
                endpoint="gpt-4o@openai",
                model="gpt-4o",
                provider="openai",
                request_kw={"messages": []},
                response=mock_response,
                cache_status="hit",
                stream=False,
            )
            _llm_event_to_eventbus(llm_event)

            await asyncio.sleep(0.05)

        assert len(captured) == 1
        preview = captured[0].payload["content_preview"]
        assert len(preview) == 203  # 200 chars + "..."
        assert preview.endswith("...")

    @pytest.mark.asyncio
    @_handle_project
    async def test_error_event_conversion(self):
        """Error events should capture the error message."""
        async with capture_events("LLM") as captured:
            llm_event = LLMEvent(
                phase="response",
                endpoint="gpt-4o@openai",
                model="gpt-4o",
                provider="openai",
                request_kw={"messages": []},
                response=None,
                cache_status="error",
                error=Exception("API rate limit exceeded"),
                stream=False,
            )
            _llm_event_to_eventbus(llm_event)

            await asyncio.sleep(0.05)

        assert len(captured) == 1
        evt = captured[0]
        assert evt.payload["cache_status"] == "error"
        assert evt.payload["error"] == "API rate limit exceeded"

    @pytest.mark.asyncio
    @_handle_project
    async def test_streaming_event_conversion(self):
        """Streaming events should be marked appropriately."""
        async with capture_events("LLM") as captured:
            llm_event = LLMEvent(
                phase="response",
                endpoint="gpt-4o@openai",
                model="gpt-4o",
                provider="openai",
                request_kw={"messages": []},
                response=None,  # Streaming has no single response
                cache_status=None,  # Streaming doesn't use cache
                stream=True,
            )
            _llm_event_to_eventbus(llm_event)

            await asyncio.sleep(0.05)

        assert len(captured) == 1
        evt = captured[0]
        assert evt.payload["stream"] is True
        assert evt.payload["cache_status"] is None


# ---------------------------------------------------------------------------
#  3. Hook installation tests
# ---------------------------------------------------------------------------


class TestHookInstallation:
    """Tests for the hook installation mechanism."""

    def test_hook_installed_during_unity_init(self):
        """The hook should be installed during unity.init()."""
        # unity.init() runs before tests via _handle_project, so hook should be set
        # Note: The hook is a function reference, check it's set and is our hook
        hook = unillm.get_llm_event_hook()
        # After unity.init(), the hook should be _llm_event_to_eventbus
        # We verify by checking the hook is the expected function
        assert hook is _llm_event_to_eventbus

    def test_install_hook_multiple_calls_safe(self):
        """Calling install_llm_event_hook multiple times should not crash."""
        # This tests that the idempotency mechanism works without error
        # The _HOOK_INSTALLED flag prevents actual re-installation
        install_llm_event_hook()
        install_llm_event_hook()
        install_llm_event_hook()
        # Should not raise any errors - the calls are no-ops after first install


# ---------------------------------------------------------------------------
#  4. Integration tests with real LLM calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_llm_call_publishes_events():
    """A real LLM call should publish both request and response events to EventBus."""
    # Ensure the hook is installed (unity.init() should have done this)
    install_llm_event_hook()

    async with capture_events("LLM") as captured:
        client = unillm.AsyncUnify("gpt-4.1-nano@openai", cache=True)
        await client.generate(
            messages=[
                {"role": "user", "content": "Say 'test123' exactly [llm_events_test]"},
            ],
        )

        # Wait for async publish to complete
        await asyncio.sleep(0.1)
        EVENT_BUS.join_published()

    # Should have both request and response events
    assert len(captured) >= 2

    # Find request and response events
    request_events = [e for e in captured if e.payload.get("phase") == "request"]
    response_events = [e for e in captured if e.payload.get("phase") == "response"]

    assert len(request_events) >= 1
    assert len(response_events) >= 1

    # Check request event
    req = request_events[-1]
    assert req.payload["endpoint"] == "gpt-4.1-nano@openai"
    assert req.payload["model"] == "gpt-4.1-nano"
    assert req.payload["provider"] == "openai"
    assert req.payload["messages_count"] == 1
    assert req.payload["stream"] is False

    # Check response event
    resp = response_events[-1]
    assert resp.payload["endpoint"] == "gpt-4.1-nano@openai"
    assert resp.payload["cache_status"] in ("hit", "miss")


@pytest.mark.asyncio
@_handle_project
async def test_llm_call_captures_token_usage():
    """Response events should include token usage from the LLM response."""
    install_llm_event_hook()

    async with capture_events("LLM") as captured:
        client = unillm.AsyncUnify("gpt-4.1-nano@openai", cache=True)
        await client.generate(
            messages=[{"role": "user", "content": "Say 'hi' [token_usage_test]"}],
        )

        await asyncio.sleep(0.1)
        EVENT_BUS.join_published()

    response_events = [e for e in captured if e.payload.get("phase") == "response"]
    assert len(response_events) >= 1

    resp = response_events[-1]
    # Token counts should be present (at least for cache miss)
    # For cache hits, the cached response should also have usage info
    if resp.payload["cache_status"] == "miss":
        # Fresh calls should have token usage
        assert (
            resp.payload.get("prompt_tokens") is not None
            or resp.payload.get("total_tokens") is not None
        )


@pytest.mark.asyncio
@_handle_project
async def test_llm_call_with_tools_captures_tool_count():
    """Request events should capture the number of tools provided."""
    install_llm_event_hook()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform calculations",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    async with capture_events("LLM") as captured:
        client = unillm.AsyncUnify("gpt-4.1-nano@openai", cache=True)
        await client.generate(
            messages=[{"role": "user", "content": "What is 2+2? [tools_test]"}],
            tools=tools,
        )

        await asyncio.sleep(0.1)
        EVENT_BUS.join_published()

    request_events = [e for e in captured if e.payload.get("phase") == "request"]
    assert len(request_events) >= 1

    req = request_events[-1]
    assert req.payload["tools_count"] == 2


@pytest.mark.asyncio
@_handle_project
async def test_multiple_sequential_llm_calls():
    """Multiple LLM calls should each publish their own events."""
    install_llm_event_hook()

    async with capture_events("LLM") as captured:
        client = unillm.AsyncUnify("gpt-4.1-nano@openai", cache=True)

        # Make three calls
        await client.generate(
            messages=[{"role": "user", "content": "Call 1 [seq_test]"}],
        )
        await client.generate(
            messages=[{"role": "user", "content": "Call 2 [seq_test]"}],
        )
        await client.generate(
            messages=[{"role": "user", "content": "Call 3 [seq_test]"}],
        )

        await asyncio.sleep(0.15)
        EVENT_BUS.join_published()

    # Should have 6 events (2 per call: request + response)
    assert len(captured) >= 6

    request_events = [e for e in captured if e.payload.get("phase") == "request"]
    response_events = [e for e in captured if e.payload.get("phase") == "response"]

    assert len(request_events) >= 3
    assert len(response_events) >= 3


@pytest.mark.asyncio
@_handle_project
async def test_llm_events_searchable_in_eventbus():
    """LLM events should be searchable after publishing."""
    install_llm_event_hook()

    client = unillm.AsyncUnify("gpt-4.1-nano@openai", cache=True)
    await client.generate(
        messages=[{"role": "user", "content": "Searchable test [search_test]"}],
    )

    await asyncio.sleep(0.1)
    EVENT_BUS.join_published()

    # Search for LLM events
    events = await EVENT_BUS.search(
        filter='type == "LLM"',
        limit=10,
    )

    assert len(events) >= 2
    llm_events = [e for e in events if e.type == "LLM"]
    assert len(llm_events) >= 2


# ---------------------------------------------------------------------------
#  5. EventBus LLM type registration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_llm_event_type_registered():
    """The LLM event type should be registered in the EventBus."""
    bus = EventBus()

    # Should be able to publish LLM events without error
    event = Event(
        type="LLM",
        payload=LLMPayload(
            phase="request",
            endpoint="test@provider",
            model="test",
            provider="provider",
            messages_count=1,
            tools_count=0,
        ),
    )
    await bus.publish(event)
    bus.join_published()

    # Event should be in the deque
    assert "LLM" in bus._deques
    assert len(bus._deques["LLM"]) >= 1


@pytest.mark.asyncio
@_handle_project
async def test_llm_events_in_search_by_type():
    """LLM events should be retrievable via search with type grouping."""
    bus = EventBus()

    # Publish a couple of LLM events
    for i in range(3):
        await bus.publish(
            Event(
                type="LLM",
                payload=LLMPayload(
                    phase="request" if i % 2 == 0 else "response",
                    endpoint="test@provider",
                    model="test",
                    provider="provider",
                    seq=i,
                ),
            ),
        )
    bus.join_published()

    # Search grouped by type
    results = await bus.search(grouped_by_type=True, limit=10)

    assert "LLM" in results
    assert len(results["LLM"]) >= 3


# ---------------------------------------------------------------------------
#  6. Edge case and resilience tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_hook_handles_missing_response_gracefully():
    """Hook should handle None response without crashing."""
    async with capture_events("LLM") as captured:
        llm_event = LLMEvent(
            phase="response",
            endpoint="gpt-4o@openai",
            model="gpt-4o",
            provider="openai",
            request_kw={"messages": []},
            response=None,
            cache_status="error",
            stream=False,
        )
        _llm_event_to_eventbus(llm_event)

        await asyncio.sleep(0.05)

    # Should still publish an event
    assert len(captured) == 1
    assert captured[0].payload["response_model"] is None
    assert captured[0].payload["prompt_tokens"] is None


@pytest.mark.asyncio
@_handle_project
async def test_hook_handles_empty_request_kw():
    """Hook should handle empty request_kw without crashing."""
    async with capture_events("LLM") as captured:
        llm_event = LLMEvent(
            phase="request",
            endpoint="gpt-4o@openai",
            model="gpt-4o",
            provider="openai",
            request_kw={},  # Empty
            stream=False,
        )
        _llm_event_to_eventbus(llm_event)

        await asyncio.sleep(0.05)

    assert len(captured) == 1
    assert captured[0].payload["messages_count"] == 0
    assert captured[0].payload["tools_count"] == 0


@pytest.mark.asyncio
@_handle_project
async def test_hook_handles_malformed_response():
    """Hook should handle responses without expected attributes."""
    mock_response = MagicMock()
    # Response with no 'model' attribute
    del mock_response.model
    mock_response.usage = None
    mock_response.choices = []  # Empty choices

    async with capture_events("LLM") as captured:
        llm_event = LLMEvent(
            phase="response",
            endpoint="gpt-4o@openai",
            model="gpt-4o",
            provider="openai",
            request_kw={"messages": []},
            response=mock_response,
            cache_status="miss",
            stream=False,
        )
        _llm_event_to_eventbus(llm_event)

        await asyncio.sleep(0.05)

    # Should still publish successfully
    assert len(captured) == 1
    # Fields should be None when extraction fails
    assert captured[0].payload["response_model"] is None
    assert captured[0].payload["content_preview"] is None
