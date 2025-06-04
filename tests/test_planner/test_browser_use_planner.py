import pytest
import asyncio
import functools
import unify
import os
import json

from unity.common.llm_helpers import start_async_tool_use_loop
from unity.planner.browser_use_planner import BrowserUsePlanner, BrowserUsePlan
from tests.helpers import _handle_project


# Fixtures to create a real LLM client for each test
def make_client(system_message: str):
    client = unify.AsyncUnify(
        os.environ.get("UNIFY_MODEL", "gpt-4o@openai"),
        cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
        traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
    )
    client.set_system_message(system_message)
    return client


@pytest.mark.asyncio
@_handle_project
async def test_start_and_ask_browser_use_plan(monkeypatch):
    planner = BrowserUsePlanner(headless=True)

    ask_called = {"count": 0}
    stop_called = {"count": 0}
    original_ask = BrowserUsePlan.ask
    original_stop = BrowserUsePlan.stop

    @functools.wraps(original_ask)
    async def ask(self, question: str) -> str:
        ask_called["count"] += 1
        return await original_ask(self, question)

    @functools.wraps(original_stop)
    async def stop(self) -> str:
        stop_called["count"] += 1
        print(f"Patched BrowserUsePlan.stop called for plan {self._task_id}")
        return await original_stop(self)

    monkeypatch.setattr(BrowserUsePlan, "ask", ask, raising=True)
    monkeypatch.setattr(BrowserUsePlan, "stop", stop, raising=True)

    def patched_build_tools_for_test():
        dummy_tools = {}
        action_items = {
            "search_google": type(
                "Action",
                (),
                {"description": "Dummy Google Search"},
            )(),
            "extract_content": type(
                "Action",
                (),
                {"description": "Dummy Content Extraction"},
            )(),
            "go_to_url": type("Action", (), {"description": "Dummy URL Navigation"})(),
        }.items()
        for action_name, action_obj in action_items:
            # Use a closure to capture the action_name for each dummy function
            def create_dummy_func(name_closure, desc_closure):
                async def specific_dummy_tool(**kwargs):
                    if "extract" in name_closure.lower():
                        await asyncio.sleep(10)
                        return (
                            "Extracted content from www.tastycola-official.com: "
                            "Tasty Cola Ltd. is a leading beverage company established in 1903, "
                            "known for its flagship Tasty Cola drink. Key products include Classic, Diet, and Zero Sugar versions. "
                            "The company is focused on global expansion. No detailed financial data on this page."
                            " All key information seems to be gathered."  # Hint for the LLM
                        )
                    elif "search" in name_closure.lower():
                        await asyncio.sleep(10)
                        return f"Found several promising search results for '{kwargs.get('query', '')}', including a potential official site: www.tastycola-official.com."
                    elif "go_to_url" in name_closure.lower():
                        await asyncio.sleep(10)
                        return f"Successfully navigated to the official site: 'tastycola-official.com'. The page seems to be about Tasty Cola's products and history."
                    return f"Dummy action '{name_closure}' completed successfully with arguments: {kwargs}."

                specific_dummy_tool.__name__ = name_closure
                specific_dummy_tool.__qualname__ = name_closure
                specific_dummy_tool.__doc__ = (
                    desc_closure
                    or f"This is a dummy implementation for the '{name_closure}' browser tool."
                )
                return specific_dummy_tool

            dummy_tools[action_name] = create_dummy_func(
                action_name,
                getattr(action_obj, "description", None),
            )
        return dummy_tools

    monkeypatch.setattr(planner, "_build_tools", patched_build_tools_for_test)
    if hasattr(planner, "_tools_cache"):
        planner._tools_cache = None
    system = (
        "You are an automated test assistant. Your responses must be precise.\n"
        "1. Call `BrowserUsePlanner_plan` with task_description='perform a search on a Tasty Cola Ltd. using the browser and give me the results'.\n"
        "2. When the user interjects 'ask', you MUST call the dynamic tool that starts with `_ask_` (associated with the active plan created in step 1) exactly once. The question for this `_ask_` tool is: 'Are there any early findings already?'.\n"
        "3. After the `_ask_` tool returns its result, you MUST then call the dynamic tool that starts with `_stop_` (associated with the active plan from step 1) to terminate that plan.\n"
        "4. After the `_stop_` tool returns its result, your next and ONLY response MUST be the single word 'done'. You MUST NOT call any more tools or say anything else."
    )
    client = make_client(system)
    tools = {"BrowserUsePlanner_plan": planner.plan}

    handle = start_async_tool_use_loop(
        client=client,
        message="begin",
        tools=tools,
        max_steps=15,
        timeout=120,
    )
    await asyncio.sleep(2)
    await handle.interject("ask")
    final = await handle.result()

    print(client.messages)
    assert "done" in final.strip().lower()
    assert (
        ask_called["count"] >= 1
    ), "BrowserUsePlan.ask should be invoked at least once"
    assert (
        stop_called["count"] >= 1
    ), "BrowserUsePlan.stop should be invoked at least once"
