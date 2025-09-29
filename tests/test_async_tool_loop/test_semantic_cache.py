import unify
from unity.common.async_tool_loop import start_async_tool_use_loop
import pytest
from tests.helpers import _handle_project


def say_hello():
    return "Hello from Unity!"


@pytest.mark.asyncio
@_handle_project
async def test_single_tool_exact_match():
    client = unify.AsyncUnify("gpt-4o@openai", temperature=0.0, cache=False)
    handle = start_async_tool_use_loop(
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

    client = unify.AsyncUnify("gpt-4o@openai", temperature=0.0, cache=False)
    handle = start_async_tool_use_loop(
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
    client = unify.AsyncUnify("gpt-4o@openai", temperature=0.0, cache=False)
    handle = start_async_tool_use_loop(
        client,
        "Call the say_hello_there tool and reply with the result only",
        tools={"say_hello_there": say_hello},
        semantic_cache=True,
    )
    res = await handle.result()

    # Check that the first call actually made a tool call to say_hello
    say_hello_first_count = 0
    for msg in client.messages:
        if msg.get("role") != "tool":
            continue

        if msg.get("name") == "say_hello_there":
            say_hello_first_count += 1

    assert (
        say_hello_first_count == 1
    ), f"Expected 1 say_hello tool call in first run, got {say_hello_first_count}"

    client = unify.AsyncUnify("gpt-4o@openai", temperature=0.0, cache=False)
    handle = start_async_tool_use_loop(
        client,
        "Could you please call the say_hello_there tool?",
        tools={"say_hello_there": say_hello},
        semantic_cache=True,
    )
    res = await handle.result()

    # Check that the second call used semantic cache (no say_hello tool calls)
    say_hello_second_count = 0
    for msg in client.messages:
        if msg.get("role") != "tool":
            continue

        if msg.get("name") == "say_hello_there":
            say_hello_second_count += 1
    assert (
        say_hello_second_count == 0
    ), f"Expected 0 say_hello tool calls in second run (cached), got {say_hello_second_count}"

    assert "Hello from Unity!" in res


@pytest.mark.asyncio
@_handle_project
async def test_tool_with_different_arguments():

    def search_contact(name: str):
        return f"Contact found: {name}"

    def find_contact(name: str):
        return f"Contact not found: {name}"

    client = unify.AsyncUnify("gpt-4o@openai", temperature=0.0, cache=False)
    handle = start_async_tool_use_loop(
        client,
        "Can you search for a contact with the name 'John Doe'?",
        tools={"search_contact": search_contact},
        semantic_cache=True,
    )
    res = await handle.result()
    assert "John Doe" in res

    client = unify.AsyncUnify("gpt-4o@openai", temperature=0.0, cache=False)
    handle = start_async_tool_use_loop(
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

    def current_weather():
        nonlocal _call_count
        if _call_count == 0:
            ret = "The weather is sunny"
        else:
            ret = "The weather is cloudy"
        _call_count += 1
        return ret

    client = unify.AsyncUnify("gpt-4o@openai", temperature=0.0, cache=False)
    handle = start_async_tool_use_loop(
        client,
        "How is the weather?",
        tools={"current_weather": current_weather},
        semantic_cache=True,
    )
    res = await handle.result()
    assert "The weather is sunny" in res
    assert _call_count == 1

    client = unify.AsyncUnify("gpt-4o@openai", temperature=0.0, cache=False)
    handle = start_async_tool_use_loop(
        client,
        "How is the weather?",
        tools={"current_weather": current_weather},
        semantic_cache=True,
    )
    res = await handle.result()
    assert "The weather is cloudy" in res
    assert _call_count == 2, f"Expected 2 calls, got {_call_count}"
