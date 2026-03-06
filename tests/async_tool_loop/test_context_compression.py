from __future__ import annotations

import copy
import inspect

import pytest

from tests.helpers import _handle_project
from unity.common._async_tool.context_compression import (
    CompressedMessage,
    CompressedMessages,
    compress_context,
    compress_messages,
    prepare_messages_for_compression,
    render_compressed_context,
)


class TestCompressContextTool:
    def test_no_parameters(self):
        sig = inspect.signature(compress_context)
        assert len(sig.parameters) == 0

    def test_has_docstring(self):
        assert compress_context.__doc__ is not None
        assert len(compress_context.__doc__) > 20

    def test_returns_string(self):
        result = compress_context()
        assert isinstance(result, str)


class TestPrepareMessagesForCompression:
    def test_strips_image_blocks(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."},
                    },
                ],
            },
        ]
        result = prepare_messages_for_compression(messages)
        assert len(result) == 1
        content = result[0]["content"]
        if isinstance(content, list):
            texts = [b["text"] for b in content if b.get("type") == "text"]
            full_text = " ".join(texts)
        else:
            full_text = content
        assert "1 image" in full_text.lower()
        assert "base64" not in str(content)

    def test_strips_multiple_images(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "data": "abc123..."},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,def456..."},
                    },
                    {
                        "type": "image",
                        "source": {"type": "base64", "data": "ghi789..."},
                    },
                ],
            },
        ]
        result = prepare_messages_for_compression(messages)
        content = result[0]["content"]
        if isinstance(content, list):
            texts = [b["text"] for b in content if b.get("type") == "text"]
            full_text = " ".join(texts)
        else:
            full_text = content
        assert "3 image" in full_text.lower()

    def test_strips_thinking_blocks(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Let me reason about this...",
                        "signature": "base64signatureblob==",
                    },
                    {"type": "text", "text": "Here is the answer."},
                ],
            },
        ]
        result = prepare_messages_for_compression(messages)
        content = result[0]["content"]
        if isinstance(content, list):
            types = [b.get("type") for b in content]
            assert "thinking" not in types
            texts = [b["text"] for b in content if b.get("type") == "text"]
            assert "Here is the answer." in texts
        else:
            assert "thinking" not in content.lower() or "Here is the answer" in content
        assert "signature" not in str(content)

    def test_preserves_text_only_messages(self):
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = prepare_messages_for_compression(messages)
        assert result[0]["content"] == "Hello world"
        assert result[1]["content"] == "Hi there"

    def test_handles_mixed_content(self):
        """Message with text + images + thinking all at once."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "internal reasoning",
                        "signature": "sig==",
                    },
                    {"type": "text", "text": "Here is what I see."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc..."},
                    },
                    {"type": "text", "text": "The image shows a cat."},
                ],
            },
        ]
        result = prepare_messages_for_compression(messages)
        content = result[0]["content"]
        if isinstance(content, list):
            types = [b.get("type") for b in content]
            assert "thinking" not in types
            assert "image_url" not in types
            assert "image" not in types
            texts = [b.get("text", "") for b in content if b.get("type") == "text"]
            full = " ".join(texts)
            assert "Here is what I see." in full
            assert "The image shows a cat." in full
            assert "1 image" in full.lower()
        else:
            assert "base64" not in content
            assert "signature" not in content

    def test_does_not_mutate_input(self):
        original_content = [
            {"type": "text", "text": "hello"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc..."},
            },
        ]
        messages = [{"role": "user", "content": copy.deepcopy(original_content)}]
        original_snapshot = copy.deepcopy(messages)
        prepare_messages_for_compression(messages)
        assert messages == original_snapshot

    def test_preserves_tool_messages(self):
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "search",
                "content": "result data",
            },
        ]
        result = prepare_messages_for_compression(messages)
        assert result[0] == messages[0] or result[0]["content"] == "result data"

    def test_preserves_message_count(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]
        result = prepare_messages_for_compression(messages)
        assert len(result) == len(messages)


@pytest.mark.asyncio
@_handle_project
async def test_compress_returns_compressed_messages(llm_config):
    messages = [
        {
            "role": "user",
            "content": "Please can you find the contact named John, once you find them list their phone number & email",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"name": "John"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "search",
            "content": '{"contact_id": 0, "name": "John", "email": "example@email.com", "phone": "+1234567890", "city": "Giza", "country": "Egypt", "gender": "male"}',
        },
        {
            "role": "assistant",
            "content": "I found the contact named John, their email is example@email.com and their phone number is +1234567890, would you like me to help with anything else?",
        },
    ]
    result = await compress_messages(messages, llm_config["model"])
    assert isinstance(result, CompressedMessages)
    assert len(result.messages) == len(messages)


@pytest.mark.asyncio
@_handle_project
async def test_compress_output_length_matches_input(llm_config):
    messages = [
        {"role": "user", "content": "Hello, how are you doing today?"},
        {
            "role": "assistant",
            "content": "I'm doing great, thank you for asking! How can I help you today?",
        },
        {"role": "user", "content": "What is the weather like?"},
        {
            "role": "assistant",
            "content": "I'm sorry, I don't have access to weather information. Would you like me to help with something else?",
        },
    ]
    result = await compress_messages(messages, llm_config["model"])
    assert len(result.messages) == 4


@pytest.mark.asyncio
@_handle_project
async def test_compress_compacts_tool_call_messages(llm_config):
    messages = [
        {"role": "user", "content": "Find John"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"name": "John"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "search",
            "content": '{"name": "John", "email": "j@test.com"}',
        },
    ]
    result = await compress_messages(messages, llm_config["model"])
    assert len(result.messages) == 3
    tool_call_msg = result.messages[1]
    assert tool_call_msg.role == "assistant"
    assert "search" in tool_call_msg.content.lower()


@pytest.mark.asyncio
@_handle_project
async def test_compress_preserves_image_placeholders(llm_config):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What breed is this dog?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."},
                },
            ],
        },
        {
            "role": "assistant",
            "content": "This is a Golden Retriever, approximately 2-3 years old.",
        },
    ]
    result = await compress_messages(messages, llm_config["model"])
    assert len(result.messages) == 2
    assert "image" in result.messages[0].content.lower()


@pytest.mark.asyncio
@_handle_project
async def test_compress_compacts_verbose_errors(llm_config):
    messages = [
        {"role": "user", "content": "Search for John"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"email": "john@test.com"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "search",
            "content": '{"error": "invalid parameter email, use name or id. Traceback (most recent call last): File search.py line 42 in search raise ValueError(...) ValueError: invalid parameter"}',
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"name": "John"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "name": "search",
            "content": '{"name": "John", "email": "john@test.com"}',
        },
        {
            "role": "assistant",
            "content": "Found John, email: john@test.com",
        },
    ]
    result = await compress_messages(messages, llm_config["model"])
    assert len(result.messages) == 6
    error_msg = result.messages[2]
    assert len(error_msg.content) < len(messages[2]["content"])


class TestRenderCompressedContext:
    def test_basic_format(self):
        compressed = CompressedMessages(
            messages=[
                CompressedMessage(role="user", content="Find John"),
                CompressedMessage(role="assistant", content='search(name="John")'),
                CompressedMessage(role="tool", content="John,j@test.com,+123"),
                CompressedMessage(role="assistant", content="John: j@test.com, +123"),
            ],
        )
        rendered = render_compressed_context(compressed)
        assert "[0] [user]: Find John" in rendered
        assert '[1] [assistant]: search(name="John")' in rendered
        assert "[2] [tool]: John,j@test.com,+123" in rendered
        assert "[3] [assistant]: John: j@test.com, +123" in rendered

    def test_one_line_per_message(self):
        compressed = CompressedMessages(
            messages=[
                CompressedMessage(role="user", content="a"),
                CompressedMessage(role="assistant", content="b"),
                CompressedMessage(role="tool", content="c"),
            ],
        )
        rendered = render_compressed_context(compressed)
        lines = [l for l in rendered.strip().split("\n") if l.strip()]
        assert len(lines) == 3
