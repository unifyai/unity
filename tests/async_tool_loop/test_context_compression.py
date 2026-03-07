from __future__ import annotations

import copy
import inspect

import pytest

import json

from tests.helpers import _handle_project
from unity.common._async_tool.context_compression import (
    CompressedMessage,
    CompressedMessages,
    compress_context,
    compress_messages,
    prepare_messages_for_compression,
    render_compressed_context,
    _eval_transformation,
    _make_update_tool,
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

    def test_preserves_thinking_blocks(self):
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
        assert isinstance(content, list)
        types = [b.get("type") for b in content]
        assert "thinking" in types
        texts = [b["text"] for b in content if b.get("type") == "text"]
        assert "Here is the answer." in texts

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
        assert isinstance(content, list)
        types = [b.get("type") for b in content]
        assert "thinking" in types
        assert "image_url" not in types
        assert "image" not in types
        texts = [b.get("text", "") for b in content if b.get("type") == "text"]
        full = " ".join(texts)
        assert "Here is what I see." in full
        assert "The image shows a cat." in full
        assert "1 image" in full.lower()

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


class TestEvalTransformation:
    def test_overwrite(self):
        result = _eval_transformation('x = "compressed"', "original long content")
        assert result == "compressed"

    def test_surgical_replace(self):
        content = "Hello verbose world with extra stuff"
        result = _eval_transformation(
            'x = x.replace("verbose world with extra stuff", "world")',
            content,
        )
        assert result == "Hello world"

    def test_bare_expression_auto_assigned(self):
        content = "Hello verbose world with extra stuff"
        result = _eval_transformation(
            'x.replace("verbose world with extra stuff", "world")',
            content,
        )
        assert result == "Hello world"

    def test_keep_first_line(self):
        content = "first line\nsecond line\nthird line"
        result = _eval_transformation('x = x.split("\\n")[0]', content)
        assert result == "first line"

    def test_truncate(self):
        content = "a" * 500
        result = _eval_transformation("x = x[:100]", content)
        assert len(result) == 100

    def test_regex_via_re(self):
        content = "data Traceback (most recent call last):\n  File... end"
        result = _eval_transformation(
            r'x = re.sub(r"Traceback[\s\S]*", "traceback omitted", x)',
            content,
        )
        assert "traceback omitted" in result
        assert "most recent" not in result

    def test_multiline_code(self):
        content = "keep\nERROR: bad\nkeep too"
        # Mirrors what arrives after JSON decode: \\n in strings becomes \n
        # (which exec interprets as the newline escape), and actual newlines
        # separate code lines.
        code = "lines = x.split('\\n')\nx = '\\n'.join(l for l in lines if 'ERROR' not in l)"
        result = _eval_transformation(code, content)
        assert result == "keep\nkeep too"

    def test_non_string_result_coerced(self):
        result = _eval_transformation("x = len(x)", "hello")
        assert result == "5"

    def test_runtime_error_raises(self):
        with pytest.raises(ZeroDivisionError):
            _eval_transformation("x = 1 // 0", "content")


class TestMakeUpdateTool:
    _ENDPOINT = "gpt-4o@openai"

    def test_overwrite_entry(self):
        entries = [
            json.dumps({"role": "user", "content": "original"}),
            json.dumps({"role": "assistant", "content": "keep this"}),
        ]
        update = _make_update_tool(entries, self._ENDPOINT)
        result = update(0, 'x = "replaced"')
        assert "replaced" in result
        assert entries[0] == "replaced"
        assert "keep this" in entries[1]

    def test_response_includes_token_usage(self):
        entries = [json.dumps({"role": "user", "content": "some content here"})]
        update = _make_update_tool(entries, self._ENDPOINT)
        result = update(0, 'x = "short"')
        assert "tokens" in result
        assert "%" in result

    def test_surgical_replace(self):
        entries = [
            json.dumps(
                {"role": "tool", "content": "verbose error with traceback details"},
            ),
        ]
        update = _make_update_tool(entries, self._ENDPOINT)
        result = update(
            0,
            'x = x.replace("verbose error with traceback details", "error")',
        )
        assert "error" in result
        assert "verbose" not in entries[0]

    def test_out_of_range_returns_error(self):
        entries = [json.dumps({"role": "user", "content": "only one"})]
        update = _make_update_tool(entries, self._ENDPOINT)
        result = update(5, 'x = "y"')
        assert "Error" in result
        assert "out of range" in result

    def test_invalid_code_returns_error(self):
        entries = [json.dumps({"role": "user", "content": "content"})]
        original = entries[0]
        update = _make_update_tool(entries, self._ENDPOINT)
        result = update(0, "def def def")
        assert "Error" in result
        assert entries[0] == original

    def test_empty_result_becomes_marker(self):
        entries = [json.dumps({"role": "user", "content": "something"})]
        update = _make_update_tool(entries, self._ENDPOINT)
        result = update(0, 'x = ""')
        assert "(empty)" in result
        assert entries[0] == "(empty)"

    def test_multiple_updates(self):
        entries = [
            json.dumps({"role": "user", "content": "first"}),
            json.dumps({"role": "assistant", "content": "second"}),
            json.dumps({"role": "tool", "content": "third"}),
        ]
        update = _make_update_tool(entries, self._ENDPOINT)
        update(0, 'x = "1"')
        update(2, 'x = "3"')
        assert entries[0] == "1"
        assert "second" in entries[1]
        assert entries[2] == "3"

    def test_bare_expression(self):
        entries = [json.dumps({"role": "user", "content": "hello world"})]
        update = _make_update_tool(entries, self._ENDPOINT)
        result = update(0, 'x.replace("hello", "hi")')
        assert "hi world" in result
        assert "hi world" in entries[0]

    def test_structured_transformation(self):
        entries = [
            json.dumps({"role": "tool", "content": "line1\nERROR line2\nline3"}),
        ]
        update = _make_update_tool(entries, self._ENDPOINT)
        code = (
            "msg = json.loads(x)\n"
            "lines = msg['content'].split('\\n')\n"
            "msg['content'] = '\\n'.join(l for l in lines if 'ERROR' not in l)\n"
            "x = json.dumps(msg)"
        )
        result = update(0, code)
        assert "line1" in result
        parsed = json.loads(entries[0])
        assert "ERROR" not in parsed["content"]


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
    original_json = json.dumps(messages[2], default=str)
    error_msg = result.messages[2]
    assert len(error_msg.content) < len(original_json)


class TestRenderCompressedContext:
    def test_basic_format(self):
        compressed = CompressedMessages(
            messages=[
                CompressedMessage(
                    content=json.dumps({"role": "user", "content": "Find John"}),
                ),
                CompressedMessage(
                    content=json.dumps(
                        {"role": "assistant", "content": 'search(name="John")'},
                    ),
                ),
                CompressedMessage(
                    content=json.dumps(
                        {"role": "tool", "content": "John,j@test.com,+123"},
                    ),
                ),
            ],
        )
        rendered = render_compressed_context(compressed)
        assert rendered.startswith("[0] ")
        assert "[1] " in rendered
        assert "[2] " in rendered
        assert "Find John" in rendered
        assert "search" in rendered
        assert "j@test.com" in rendered

    def test_one_line_per_message(self):
        compressed = CompressedMessages(
            messages=[
                CompressedMessage(content="a"),
                CompressedMessage(content="b"),
                CompressedMessage(content="c"),
            ],
        )
        rendered = render_compressed_context(compressed)
        lines = [l for l in rendered.strip().split("\n") if l.strip()]
        assert len(lines) == 3
