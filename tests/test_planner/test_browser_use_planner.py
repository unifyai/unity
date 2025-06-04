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

            def create_dummy_func(name_closure, desc_closure):
                async def specific_dummy_tool(**kwargs):
                    if "extract" in name_closure.lower():
                        await asyncio.sleep(10)
                        return (
                            "Extracted content from www.tastycola-official.com: "
                            "Tasty Cola Ltd. is a leading beverage company established in 1903, "
                            "known for its flagship Tasty Cola drink. Key products include Classic, Diet, and Zero Sugar versions. "
                            "The company is focused on global expansion. No detailed financial data on this page."
                            " All key information seems to be gathered."
                        )
                    elif "search" in name_closure.lower():
                        await asyncio.sleep(10)
                        return f"Found several promising search results for '{kwargs.get('kwargs', '')}', including a potential official site: www.tastycola-official.com."
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
        "1. Call `plan` with task_description='perform a search on a Tasty Cola Ltd. using the browser and give me the results'.\n"
        "2. When the user interjects 'ask', you MUST call the dynamic tool that starts with `_ask_` (associated with the active plan created in step 1) exactly once. The question for this `_ask_` tool is: 'Are there any early findings already?'.\n"
        "3. After the `_ask_` tool returns its result, you MUST then call the dynamic tool that starts with `_stop_` (associated with the active plan from step 1) to terminate that plan.\n"
        "4. After the `_stop_` tool returns its result, your next and ONLY response MUST be the single word 'ask_completed'. You MUST NOT call any more tools or say anything else."
    )
    client = make_client(system)
    tools = {"plan": planner.plan}

    handle = start_async_tool_use_loop(
        client=client,
        message="begin",
        tools=tools,
        max_steps=15,  # Plan → Ask → Stop → Done
        timeout=120,
    )
    await asyncio.sleep(2)
    await handle.interject("ask")
    final = await handle.result()

    print(client.messages)
    assert "ask_completed" in final.strip().lower()
    assert ask_called["count"] == 1, "BrowserUsePlan.ask should be invoked once"
    assert stop_called["count"] == 1, "BrowserUsePlan.stop should be invoked once"

    await planner.close()


@pytest.mark.asyncio
@_handle_project
async def test_interject_browser_use_plan(monkeypatch):
    planner = BrowserUsePlanner(headless=True)

    interjected_log = {"count": 0, "msgs": []}
    original_interject_method = BrowserUsePlan.interject

    @functools.wraps(original_interject_method)
    async def patched_interject(self, instruction: str) -> str:
        interjected_log["count"] += 1
        interjected_log["msgs"].append(instruction)
        print(
            f"Patched BrowserUsePlan.interject called with: '{instruction}' for plan {self._task_id}",
        )
        return await original_interject_method(self, instruction)

    def patched_build_tools_for_test():
        dummy_tools = {}
        action_items = {
            "search_google": type(
                "Action",
                (),
                {"description": "Google Search"},
            )(),
            "extract_content": type(
                "Action",
                (),
                {"description": "Content Extraction"},
            )(),
            "go_to_url": type("Action", (), {"description": "URL Navigation"})(),
        }.items()
        for action_name, action_obj in action_items:

            def create_dummy_func(name_closure, desc_closure):
                async def specific_dummy_tool(**kwargs):
                    if "extract" in name_closure.lower():
                        await asyncio.sleep(5)
                        print(
                            f"Extracting content for {name_closure} with kwargs: {kwargs}",
                        )
                        if "github.com" in kwargs.get("kwargs", ""):
                            return (
                                "Extracted content from https://github.com/unifyai/unify/tree/main: "
                                "The number of stars is 302."
                                " All key information seems to be gathered."
                            )
                        elif "unify.ai" in kwargs.get("kwargs", ""):
                            return (
                                "Extracted content from https://unify.ai: "
                                "Unify AI provides tools and platforms for building and deploying AI applications. "
                                "They focus on making AI development more accessible and efficient. "
                                "The company was founded in 2022 and has 10 employees. "
                                " All key information seems to be gathered."
                            )
                    elif "search" in name_closure.lower():
                        await asyncio.sleep(5)
                        if "github.com" in kwargs.get(
                            "kwargs",
                            "",
                        ) or "stars" in kwargs.get("kwargs", ""):
                            return "unify.ai has 302 github stars."
                        else:
                            return f"Found several promising search results for '{kwargs.get('kwargs', '')}', including the official website: https://unify.ai and their GitHub repository: https://github.com/unifyai/unify/tree/main."
                    elif "go_to_url" in name_closure.lower():
                        await asyncio.sleep(5)
                        url = kwargs.get("kwargs", "")
                        if "google.com" in url:
                            return f"Successfully navigated to 'google.com'."
                        elif "unify.ai" in url:
                            return f"Successfully navigated to the official site: 'https://unify.ai'. The page seems to be about Unify AI's products and mission."
                        elif "github.com" in url:
                            return "Successfully navigated to the Unify AI GitHub repository. I can see information about their projects."
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

    if hasattr(planner, "_tools_cache"):
        planner._tools_cache = None

    monkeypatch.setattr(planner, "_build_tools", patched_build_tools_for_test)
    monkeypatch.setattr(BrowserUsePlan, "interject", patched_interject, raising=True)

    system = (
        "You are an automated test assistant. Your responses must be precise.\n"
        "1. Call `plan` with task_description='perform a search on a unify.ai using the browser and give me the results about the company. You should make use of the extract_content tool to get the information about the company'.\n"
        "2. When the user interjects 'adjust', you MUST call the dynamic tool that starts with `_interject_` (associated with the active plan) exactly once. The content for the interjection should be 'also tell me how many github stars unify.ai has'.\n"
        "3. After the `_interject_` tool returns, your next and ONLY response MUST be the single word 'interjection_processed'. You MUST NOT call any more tools or say anything else."
    )
    client = make_client(system)
    tools = {"plan": planner.plan}

    handle = start_async_tool_use_loop(
        client=client,
        message="begin",
        tools=tools,
        max_steps=15,  # Plan → Interject → Done
        timeout=120,
        log_steps=True,
    )
    await asyncio.sleep(2)
    await handle.interject("adjust")

    final = await handle.result()

    assert "interjection_processed" in final.strip().lower()
    assert (
        interjected_log["count"] == 1
    ), "BrowserUsePlan.interject should be called once"
    assert any(
        "also tell me how many github stars unify.ai has" in msg.lower()
        for msg in interjected_log["msgs"]
    ), "Interjection payload incorrect"

    await planner.close()


@pytest.mark.asyncio
@_handle_project
async def test_pause_and_resume_browser_use_plan(
    monkeypatch,
):
    planner = BrowserUsePlanner(headless=True)

    counts = {"pause": 0, "resume": 0, "stop_after_resume": 0}
    original_pause_method = BrowserUsePlan.pause
    original_resume_method = BrowserUsePlan.resume
    original_stop_method = BrowserUsePlan.stop

    @functools.wraps(original_pause_method)
    async def patched_pause(self) -> str:
        counts["pause"] += 1
        print(f"Patched BrowserUsePlan.pause called for plan {self._task_id}")
        return await original_pause_method(self)

    @functools.wraps(original_resume_method)
    async def patched_resume(self) -> str:
        counts["resume"] += 1
        print(f"Patched BrowserUsePlan.resume called for plan {self._task_id}")
        return await original_resume_method(self)

    @functools.wraps(original_stop_method)
    async def patched_stop_after_resume(self) -> str:
        counts["stop_after_resume"] += 1
        print(
            f"Patched BrowserUsePlan.stop (for pause/resume test) called for plan {self._task_id}",
        )
        return await original_stop_method(self)

    monkeypatch.setattr(BrowserUsePlan, "pause", patched_pause, raising=True)
    monkeypatch.setattr(BrowserUsePlan, "resume", patched_resume, raising=True)
    monkeypatch.setattr(BrowserUsePlan, "stop", patched_stop_after_resume, raising=True)

    def patched_build_tools_for_test():
        dummy_tools: dict[str, callable] = {}
        action_items = {
            "search_google": type(
                "Action",
                (),
                {"description": "Dummy Google Search"},
            )(),
            "go_to_url": type("Action", (), {"description": "Dummy URL Navigation"})(),
            "extract_content": type(
                "Action",
                (),
                {"description": "Dummy Content Extraction"},
            )(),
        }.items()

        for action_name, action_obj in action_items:

            def create_dummy_func(name_closure, desc_closure):
                async def specific_dummy_tool(**kwargs):
                    if "go_to_url" in name_closure:
                        await asyncio.sleep(5)
                        return (
                            f"Successfully navigated to '{kwargs.get('kwargs', '')}'."
                        )
                    elif "search_google" in name_closure:
                        await asyncio.sleep(5)
                        return f"Found search results for '{kwargs.get('kwargs', '')}'."
                    elif "extract_content" in name_closure:
                        await asyncio.sleep(5)
                        return f"Extracted content from '{kwargs.get('kwargs', '')}'."
                    return f"Dummy action '{name_closure}' completed with arguments: {kwargs}."

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
        "1. Call `plan` with task_description='open google.com and search for \"python\" **once**.'.\n"
        "2. When the user interjects 'hold', you MUST call the tool that starts with `_pause_` (associated with the active plan).\n"
        "3. When the user interjects 'go', you MUST call the tool that starts with `_resume_` (associated with the active plan).\n"
        "4. After the `_resume_` tool returns its result, you MUST then call the tool that starts with `_stop_` (associated with the active plan) to terminate that plan.\n"
        "5. After the `_stop_` tool returns, your next and ONLY response MUST be the single word 'pause_resume_completed'. You MUST NOT call any more tools or say anything else."
    )
    client = make_client(system)
    tools = {"plan": planner.plan}

    handle = start_async_tool_use_loop(
        client=client,
        message="run_pause_resume_test",
        tools=tools,
        max_steps=20,  # Plan → Pause → Resume → Stop → Done
        timeout=180,
        log_steps=True,
    )

    await asyncio.sleep(3)
    await handle.interject("hold")

    await asyncio.sleep(2)
    await handle.interject("go")

    final = await handle.result()
    print(f"Test (Pause/Resume): Outer LLM final response: {final}")

    assert "pause_resume_completed" in final.strip().lower()
    assert counts["pause"] == 1, "BrowserUsePlan.pause should be called"
    assert counts["resume"] == 1, "BrowserUsePlan.resume should be called"
    assert (
        counts["stop_after_resume"] == 1
    ), "BrowserUsePlan.stop should be called after resume"

    await planner.close()
