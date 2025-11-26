import unify
import asyncio
import pytest
import random
import json

from unity.common.async_tool_loop import start_async_tool_loop
from unity.common._async_tool import semantic_cache as sc
from tests.helpers import _handle_project
from unity.common.llm_client import new_llm_client
from unity.common._async_tool.semantic_cache import _Config, SemanticCacheResult
from unity.common.tool_spec import read_only, normalise_tools, manager_tool
from unity.common.state_managers import BaseStateManager


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
    return new_llm_client(cache=False)


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
        semantic_cache="write",
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
        semantic_cache="read",
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
        semantic_cache="write",
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
        semantic_cache="read",
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
        semantic_cache="write",
    )
    res = await handle.result()
    assert "John Doe" in res
    sc._SEMANTIC_CACHE_SAVER.wait()

    client = create_client()
    handle = start_async_tool_loop(
        client,
        "Can you look for a contact with the name 'Jane Doe'?",
        tools={"search_contact": search_contact, "find_contact": find_contact},
        semantic_cache="read",
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

    query = "How is the weather? Call the current_weather tool and reply with the result only"
    client = create_client()
    handle = start_async_tool_loop(
        client,
        query,
        tools={"current_weather": current_weather},
        semantic_cache="write",
    )
    res = await handle.result()
    assert "The weather is sunny" in res
    assert _call_count == 1

    sc._SEMANTIC_CACHE_SAVER.wait()

    client = create_client()
    handle = start_async_tool_loop(
        client,
        query,
        tools={"current_weather": current_weather},
        semantic_cache="read",
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
        semantic_cache="write",
    )

    await handle.result()
    cleaned = sc._SEMANTIC_CACHE_SAVER._clean_tool_trajectory(
        "respond with the result of the find_contact tool",
        client.messages,
    )

    assert len(cleaned) == 1, f"Expected 1 tool call, got {len(cleaned)}"
    assert (
        cleaned[0]["name"] == "find_contact"
    ), f"Expected find_contact, got {cleaned[0]['name']}"


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
        semantic_cache="write",
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
        semantic_cache="read",
    )
    res = await handle.result()
    assert "Hello from Unify!" in res


@pytest.mark.asyncio
async def test_get_dummy_tool_result_status():
    @read_only
    def say_hello():
        return f"Hello!"

    def say_goodbye():
        return f"Goodbye!"

    tools = {
        "say_hello": say_hello,
        "say_goodbye": say_goodbye,
    }

    tools = normalise_tools(tools)

    def _create_tool_trajectory(n):
        ret = []

        for i in range(n):
            name = random.choice(list(tools.keys()))
            ret.append(
                {
                    "index": i,
                    "name": name,
                    "arguments": "{}",
                    "result": tools[name].fn(),
                },
            )

        return ret

    number_of_calls = 10

    res = SemanticCacheResult(
        original_user_message="",
        closest_user_message="",
        tool_trajectory=_create_tool_trajectory(number_of_calls),
    )

    semantic_closest_match = await sc.get_dummy_tool(res, tools)
    trajectory = json.loads(semantic_closest_match[1]["content"])
    assert len(trajectory) == number_of_calls
    for tool_call in trajectory:
        if tool_call["name"] == "say_hello":
            assert tool_call["result_status"] == "new"

        if tool_call["name"] == "say_goodbye":
            assert tool_call["result_status"] == "cached"


@pytest.mark.asyncio
async def test_get_dummy_tool_parse_arguments():
    @read_only
    def echo(msgs):
        return msgs

    tools = normalise_tools({"echo": echo})

    res = SemanticCacheResult(
        original_user_message="",
        closest_user_message="",
        tool_trajectory=[
            {
                "index": 0,
                "name": "echo",
                "arguments": json.dumps({"msgs": ["Hello", "World"]}),
                "result": "",
            },
        ],
    )

    res = await sc.get_dummy_tool(res, tools)
    trajectory = json.loads(res[1]["content"])
    assert len(trajectory) == 1
    assert trajectory[0]["name"] == "echo"
    assert trajectory[0]["arguments"] == json.dumps({"msgs": ["Hello", "World"]})
    assert trajectory[0]["result"] == ["Hello", "World"]
    assert trajectory[0]["result_status"] == "new"


@pytest.mark.asyncio
async def test_get_dummy_tool_parse_arguments_cached():
    @read_only
    def echo():
        pass

    tools = normalise_tools({"echo": echo})

    res = SemanticCacheResult(
        original_user_message="",
        closest_user_message="",
        tool_trajectory=[
            {
                "index": 0,
                "name": "echo",
                "arguments": json.dumps(
                    {"msgs": "Invalid"},
                ),  # Simulate out-dated arguments
                "result": "Hello!",
            },
        ],
    )

    res = await sc.get_dummy_tool(res, tools)
    trajectory = json.loads(res[1]["content"])
    assert len(trajectory) == 1
    assert trajectory[0]["name"] == "echo"
    assert trajectory[0]["result"] == "Hello!"
    assert trajectory[0]["result_status"] == "cached"


@pytest.mark.asyncio
@_handle_project
async def test_semantic_cache_recursive(monkeypatch):
    class ManagerC(BaseStateManager):
        @read_only
        def _get_answer(self):
            return "I KNOW!"

        def __init__(self):
            super().__init__()
            self.add_tools(
                "ask",
                {
                    "get_answer": self._get_answer,
                },
            )

        @manager_tool
        async def ask(self, text):
            client = create_client()
            client.set_system_message(
                "Whatever the user asks, call the get_answer tool and reply with the result only",
            )
            handle = start_async_tool_loop(
                client,
                text,
                tools=self.get_tools("ask"),
                semantic_cache="write",
                semantic_cache_namespace="ManagerC.ask",
            )
            res = await handle.result()
            return res

    class ManagerB(BaseStateManager):
        def __init__(self):
            super().__init__()
            self.manager_c = ManagerC()
            self.add_tools(
                "ask",
                {
                    "ManagerC_ask": self.manager_c.ask,
                },
            )

        @manager_tool
        async def ask(self, text):
            client = create_client()
            client.set_system_message(
                "Whatever the user asks, call the ManagerC.ask tool and reply with the result only",
            )
            handle = start_async_tool_loop(
                client,
                text,
                tools=self.get_tools("ask"),
                semantic_cache="write",
                semantic_cache_namespace="ManagerB.ask",
            )
            res = await handle.result()
            return res

    class ManagerA(BaseStateManager):
        def __init__(self):
            super().__init__()
            self.manager_b = ManagerB()
            self.add_tools(
                "ask",
                {
                    "ManagerB_ask": self.manager_b.ask,
                },
            )

        @manager_tool
        async def ask(self, text):
            client = create_client()
            client.set_system_message(
                "You are a simulated manager, "
                "whatever the user asks, call the ManagerB.ask tool and reply with the result only",
            )
            handle = start_async_tool_loop(
                client,
                text,
                tools={"ManagerB_ask": self.manager_b.ask},
                semantic_cache="write",
                semantic_cache_namespace="ManagerA.ask",
            )
            res = await handle.result()
            return res

    # Patch _prune_tool_trajectory to return the tool trajectory unchanged
    def _prune_tool_trajectory(self, user_message, tool_trajectory):
        return tool_trajectory

    monkeypatch.setattr(
        sc._SemanticCacheSaver,
        "_prune_tool_trajectory",
        _prune_tool_trajectory,
    )

    query = "What is the answer to the ultimate question of life, the universe, and everything?"
    manager = ManagerA()
    await manager.ask(query)

    sc._SEMANTIC_CACHE_SAVER.wait()

    search_result = sc.search_semantic_cache(query, "ManagerA.ask")
    assert search_result is not None
    history = await sc._rexecute_tools(
        search_result.tool_trajectory,
        normalise_tools(manager.get_tools("ask")),
    )

    # Walk through the returned tool trajectory, check that the result status is all new
    def _check_result_status(trajectory):
        if not isinstance(trajectory, list):
            return
        for tool_call in trajectory:
            if isinstance(tool_call, dict) and "result_status" in tool_call.keys():
                assert tool_call["result_status"] == "new"
                _check_result_status(tool_call["result"])

    _check_result_status(history)
