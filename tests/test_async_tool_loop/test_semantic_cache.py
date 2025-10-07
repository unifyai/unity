import unify
import asyncio
import pytest

from unity.common.async_tool_loop import start_async_tool_loop
from unity.common._async_tool import semantic_cache as sc
from tests.helpers import _handle_project
from unity.common._async_tool.semantic_cache import _Config
from unity.common.tool_spec import read_only


@pytest.fixture(autouse=True)
def _patch_semantic_cache_config(monkeypatch):
    class _DynamicConfig(_Config):
        @property
        def context(self):
            return f"{unify.get_active_context()['write']}/SemanticCache"

    monkeypatch.setattr(
        "unity.common._async_tool.semantic_cache._CONFIG",
        _DynamicConfig(),
    )


def create_client():
    return unify.AsyncUnify("gpt-4o@openai", temperature=0.0, cache=False)


@pytest.mark.asyncio
@_handle_project
async def test_single_tool_exact_match():
    @read_only
    def say_hello():
        return "Hello from Unity!"

    client = create_client()
    handle = start_async_tool_loop(
        client,
        "Hello, how are you? call the say_hello tool and reply with the result only",
        tools={"say_hello": say_hello},
        semantic_cache=True,
    )
    res = await handle.result()

    # Check that the first call actually made a tool call to say_hello
    say_hello_first_count = 0
    for msg in client.messages:
        if msg.get("role") != "tool":
            continue

        if msg.get("name") == "say_hello":
            say_hello_first_count += 1

    assert (
        say_hello_first_count == 1
    ), f"Expected 1 say_hello tool call in first run, got {say_hello_first_count}"

    sc._SEMANTIC_CACHE_SAVER.wait()

    client = create_client()
    handle = start_async_tool_loop(
        client,
        "Hello, how are you? call the say_hello tool and reply with the result only",
        tools={"say_hello": say_hello},
        semantic_cache=True,
    )
    res = await handle.result()

    # Check that the second call used semantic cache (no say_hello tool calls)
    say_hello_second_count = 0
    for msg in client.messages:
        if msg.get("role") != "tool":
            continue

        if msg.get("name") == "say_hello":
            say_hello_second_count += 1
    assert (
        say_hello_second_count == 0
    ), f"Expected 0 say_hello tool calls in second run (cached), got {say_hello_second_count}"

    assert "Hello from Unity!" in res


@pytest.mark.asyncio
@_handle_project
async def test_single_tool_no_exact_match():
    @read_only
    def say_hello():
        return "Hello from Unity!"

    client = create_client()
    handle = start_async_tool_loop(
        client,
        "Call the say_hello tool and reply with the result only",
        tools={"say_hello": say_hello},
        semantic_cache=True,
    )
    res = await handle.result()

    # Check that the first call actually made a tool call to say_hello
    say_hello_first_count = 0
    for msg in client.messages:
        if msg.get("role") != "tool":
            continue

        if msg.get("name") == "say_hello":
            say_hello_first_count += 1

    assert (
        say_hello_first_count == 1
    ), f"Expected 1 say_hello tool call in first run, got {say_hello_first_count}"

    sc._SEMANTIC_CACHE_SAVER.wait()

    client = create_client()
    handle = start_async_tool_loop(
        client,
        "Could you please call the say_hello tool?",
        tools={"say_hello": say_hello},
        semantic_cache=True,
    )
    res = await handle.result()

    # Check that the second call used semantic cache (no say_hello tool calls)
    say_hello_second_count = 0
    for msg in client.messages:
        if msg.get("role") != "tool":
            continue

        if msg.get("name") == "say_hello":
            say_hello_second_count += 1
    assert (
        say_hello_second_count == 0
    ), f"Expected 0 say_hello tool calls in second run (cached), got {say_hello_second_count}"

    assert "Hello from Unity!" in res


@pytest.mark.asyncio
@_handle_project
async def test_tool_with_different_arguments():

    @read_only
    def search_contact(name: str):
        return f"Contact found: {name}"

    @read_only
    def find_contact(name: str):
        return f"Contact not found: {name}"

    client = create_client()
    handle = start_async_tool_loop(
        client,
        "Can you search for a contact with the name 'John Doe'?",
        tools={"search_contact": search_contact},
        semantic_cache=True,
    )
    res = await handle.result()
    assert "John Doe" in res
    sc._SEMANTIC_CACHE_SAVER.wait()

    client = create_client()
    handle = start_async_tool_loop(
        client,
        "Can you look for a contact with the name 'Jane Doe'?",
        tools={"search_contact": search_contact, "find_contact": find_contact},
        semantic_cache=True,
    )
    res = await handle.result()

    # Should not use result directly from cache
    assert "Jane Doe" in res


@pytest.mark.asyncio
@_handle_project
async def test_tool_is_re_called():
    _call_count = 0

    @read_only
    def current_weather():
        nonlocal _call_count
        if _call_count == 0:
            ret = "The weather is sunny"
        else:
            ret = "The weather is cloudy"
        _call_count += 1
        return ret

    client = create_client()
    handle = start_async_tool_loop(
        client,
        "How is the weather?",
        tools={"current_weather": current_weather},
        semantic_cache=True,
    )
    res = await handle.result()
    assert "The weather is sunny" in res
    assert _call_count == 1

    sc._SEMANTIC_CACHE_SAVER.wait()

    client = create_client()
    handle = start_async_tool_loop(
        client,
        "How is the weather?",
        tools={"current_weather": current_weather},
        semantic_cache=True,
    )
    res = await handle.result()
    assert "cloudy" in res.lower()
    assert _call_count == 2, f"Expected 2 calls, got {_call_count}"


@pytest.mark.asyncio
@_handle_project
async def test_construct_new_user_message():
    @read_only
    async def say_hello():
        await asyncio.sleep(1)
        return "Hello from Unity!"

    @read_only
    async def say_goodbye():
        return "Goodbye from Unity!"

    client = create_client()
    initial_user_message = "Call the say_hello tool and reply with the result only"
    handle = start_async_tool_loop(
        client,
        initial_user_message,
        tools={"say_hello": say_hello, "say_goodbye": say_goodbye},
    )

    await handle.interject("Actually, I meant to call the say_goodbye tool")
    await handle.result()

    msgs = client.messages
    new_user_message = sc._SEMANTIC_CACHE_SAVER._construct_new_user_message(
        initial_user_message,
        msgs,
        client.messages,
    )

    assert "say_goodbye" in new_user_message
    assert "say_hello" not in new_user_message


@pytest.mark.asyncio
@_handle_project
async def test_prune_tools():
    @read_only
    def say_hello(data: str) -> str:
        return f"Hello from Unity!"

    @read_only
    def say_goodbye(data: str) -> str:
        return f"Goodbye from Unity!"

    @read_only
    def find_contact(name: str) -> str:
        return f"Contact found: {name}"

    client = create_client()

    instruction = (
        "1) Call `say_hello` with data='foo' and `say_goodbye` with data='bar' exactly once each (in any order).\n"
        "2) Then call `find_contact` with name='John Doe' exactly once.\n"
        "3) After all tools complete, respond with ONLY the result of `find_contact` (no extra text)."
    )

    handle = start_async_tool_loop(
        client,
        instruction,
        tools={
            "say_hello": say_hello,
            "say_goodbye": say_goodbye,
            "find_contact": find_contact,
        },
    )

    await handle.result()
    cleaned = sc._SEMANTIC_CACHE_SAVER._clean_tool_trajectory(
        "respond with the result of the find_contact tool",
        client.messages,
    )

    assert len(cleaned) == 1, f"Expected 1 tool call, got {len(cleaned)}"
    assert (
        cleaned[0]["request"]["function"]["name"] == "find_contact"
    ), f"Expected find_contact, got {cleaned[0]['request']['function']['name']}"


@pytest.mark.asyncio
@_handle_project
async def test_tool_call_signature_updated():
    @read_only
    def say_hello():
        return "Hello from Unity!"

    client = create_client()
    handle = start_async_tool_loop(
        client,
        "Call the say_hello tool and reply with the result only",
        tools={"say_hello": say_hello},
        semantic_cache=True,
    )
    res = await handle.result()
    assert "Hello from Unity!" in res
    sc._SEMANTIC_CACHE_SAVER.wait()

    @read_only
    def _say_hello_new(user: str):
        return f"Hello from {user}!"

    client = create_client()
    handle = start_async_tool_loop(
        client,
        "Call the say_hello tool with the argument 'Unify' and reply with the result only",
        tools={"say_hello": _say_hello_new},
        semantic_cache=True,
    )
    res = await handle.result()
    assert "Hello from Unify!" in res
