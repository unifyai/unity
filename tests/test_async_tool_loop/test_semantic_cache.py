import unify
from unity.common.async_tool_loop import start_async_tool_use_loop
import pytest
from tests.helpers import _handle_project


def say_hello():
    return "Hello from Unity!"


@pytest.mark.asyncio
@_handle_project
async def test_semantic_cache_single_tool_exact_match():
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
async def test_semantic_cache_single_tool_no_exact_match():
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
