from typing import Tuple, Type
import pytest
import asyncio
import functools
import unify
import os

from unity.common.llm_helpers import start_async_tool_use_loop
from unity.actor.base import BaseActor
from unity.task_scheduler.base import BaseActiveTask
from unity.actor.browser_use_actor import BrowserUseActor, BrowserUsePlan
from unity.actor.tool_loop_actor import ToolLoopActor, ToolLoopPlan
from tests.helpers import _handle_project, SETTINGS


# Fixtures to create a real LLM client for each test
def make_client(system_message: str):
    client = unify.AsyncUnify(
        os.environ.get("UNIFY_MODEL", "gpt-4o@openai"),
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(system_message)
    return client


ActorFixture = Tuple[Type[BaseActor], Type[BaseActiveTask], dict]


@pytest.fixture(
    params=[
        (BrowserUseActor, BrowserUsePlan, {"headless": True}),
        (ToolLoopActor, ToolLoopPlan, {"headless": True}),
    ],
)
def actor_and_plan_types(request) -> ActorFixture:
    return request.param


@pytest.mark.asyncio
@_handle_project
async def test_start_and_ask_plan(monkeypatch, actor_and_plan_types):
    actor_class, plan_class, actor_kwargs = actor_and_plan_types
    actor = actor_class(**actor_kwargs)

    ask_called = {"count": 0}
    stop_called = {"count": 0}
    original_ask = plan_class.ask
    original_stop = plan_class.stop

    @functools.wraps(original_ask)
    async def ask(self, question: str) -> str:
        ask_called["count"] += 1
        return await original_ask(self, question)

    @functools.wraps(original_stop)
    async def stop(self) -> str:
        stop_called["count"] += 1
        return await original_stop(self)

    monkeypatch.setattr(plan_class, "ask", ask, raising=True)
    monkeypatch.setattr(plan_class, "stop", stop, raising=True)

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

    monkeypatch.setattr(actor, "_build_tools", patched_build_tools_for_test)
    if hasattr(actor, "_tools_cache"):
        actor._tools_cache = None
    system = (
        "You are an automated test assistant. Your responses must be precise.\n"
        "1. Call `plan` with task_description='perform a search on a Tasty Cola Ltd. using the browser and give me the results'.\n"
        "2. When the user interjects 'ask', you MUST call the dynamic tool that starts with `_ask_` (associated with the active plan created in step 1) exactly once. The question for this `_ask_` tool is: 'Are there any early findings already?'.\n"
        "3. After the `_ask_` tool returns its result, you MUST then call the dynamic tool that starts with `_stop_` (associated with the active plan from step 1) to terminate that plan.\n"
        "4. After the `_stop_` tool returns its result, your next and ONLY response MUST be the single word 'ask_completed'. You MUST NOT call any more tools or say anything else."
    )
    client = make_client(system)
    tools = {"execute": actor.execute}

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

    assert "ask_completed" in final.strip().lower()
    assert ask_called["count"] == 1, "BrowserUsePlan.ask should be invoked once"
    assert stop_called["count"] == 1, "BrowserUsePlan.stop should be invoked once"

    await actor.close()


@pytest.mark.asyncio
@_handle_project
async def test_interject_plan(monkeypatch, actor_and_plan_types):
    actor_class, plan_class, actor_kwargs = actor_and_plan_types
    actor = actor_class(**actor_kwargs)

    interjected_log = {"count": 0, "msgs": []}
    original_interject_method = plan_class.interject

    @functools.wraps(original_interject_method)
    async def patched_interject(self, instruction: str) -> str:
        interjected_log["count"] += 1
        interjected_log["msgs"].append(instruction)
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

    if hasattr(actor, "_tools_cache"):
        actor._tools_cache = None

    monkeypatch.setattr(actor, "_build_tools", patched_build_tools_for_test)
    monkeypatch.setattr(plan_class, "interject", patched_interject, raising=True)

    system = (
        "You are an automated test assistant. Your responses must be precise.\n"
        "1. Call `plan` with task_description='perform a search on a unify.ai using the browser and give me the results about the company. You should make use of the extract_content tool to get the information about the company'.\n"
        "2. When the user interjects 'adjust', you MUST call the dynamic tool that starts with `_interject_` (associated with the active plan) exactly once. The content for the interjection should be 'also tell me how many github stars unify.ai has'.\n"
        "3. After the `_interject_` tool returns, your next and ONLY response MUST be the single word 'interjection_processed'. You MUST NOT call any more tools or say anything else."
    )
    client = make_client(system)
    tools = {"execute": actor.execute}

    handle = start_async_tool_use_loop(
        client=client,
        message="begin",
        tools=tools,
        max_steps=15,  # Plan → Interject → Done
        timeout=120,
        log_steps=False,
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

    await actor.close()


@pytest.mark.asyncio
@_handle_project
async def test_pause_and_resume_plan(
    monkeypatch,
    actor_and_plan_types,
):
    actor_class, plan_class, actor_kwargs = actor_and_plan_types
    actor = actor_class(**actor_kwargs)

    counts = {"pause": 0, "resume": 0, "stop_after_resume": 0}
    original_pause_method = plan_class.pause
    original_resume_method = plan_class.resume
    original_stop_method = plan_class.stop

    @functools.wraps(original_pause_method)
    async def patched_pause(self) -> str:
        counts["pause"] += 1
        return await original_pause_method(self)

    @functools.wraps(original_resume_method)
    async def patched_resume(self) -> str:
        counts["resume"] += 1
        return await original_resume_method(self)

    @functools.wraps(original_stop_method)
    async def patched_stop_after_resume(self) -> str:
        counts["stop_after_resume"] += 1
        return await original_stop_method(self)

    monkeypatch.setattr(plan_class, "pause", patched_pause, raising=True)
    monkeypatch.setattr(plan_class, "resume", patched_resume, raising=True)
    monkeypatch.setattr(plan_class, "stop", patched_stop_after_resume, raising=True)

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

    monkeypatch.setattr(actor, "_build_tools", patched_build_tools_for_test)
    if hasattr(actor, "_tools_cache"):
        actor._tools_cache = None

    system = (
        "You are an automated test assistant. Your responses must be precise.\n"
        "1. Call `plan` with task_description='open google.com and search for \"python\" **once**.'.\n"
        "2. When the user interjects 'hold', you MUST call the tool that starts with `_pause_` (associated with the active plan).\n"
        "3. When the user interjects 'go', you MUST call the tool that starts with `_resume_` (associated with the active plan).\n"
        "4. After the `_resume_` tool returns its result, you MUST then call the tool that starts with `_stop_` (associated with the active plan) to terminate that plan.\n"
        "5. After the `_stop_` tool returns, your next and ONLY response MUST be the single word 'pause_resume_completed'. You MUST NOT call any more tools or say anything else."
    )
    client = make_client(system)
    tools = {"execute": actor.execute}

    handle = start_async_tool_use_loop(
        client=client,
        message="run_pause_resume_test",
        tools=tools,
        max_steps=20,  # Plan → Pause → Resume → Stop → Done
        timeout=180,
        log_steps=False,
    )

    await asyncio.sleep(3)
    await handle.interject("hold")

    await asyncio.sleep(2)
    await handle.interject("go")

    final = await handle.result()

    assert "pause_resume_completed" in final.strip().lower()
    assert counts["pause"] == 1, "BrowserUsePlan.pause should be called"
    assert counts["resume"] == 1, "BrowserUsePlan.resume should be called"
    assert (
        counts["stop_after_resume"] == 1
    ), "BrowserUsePlan.stop should be called after resume"

    await actor.close()


@pytest.mark.asyncio
@_handle_project
async def test_plan_requests_clarification(
    monkeypatch,
    actor_and_plan_types,
):
    """
    Test that BrowserUsePlan can request and receive clarification via queues.
    """
    actor_class, plan_class, actor_kwargs = actor_and_plan_types
    actor = actor_class(**actor_kwargs)
    clarification_up_q = asyncio.Queue()
    clarification_down_q = asyncio.Queue()

    def patched_build_tools_for_clarification():
        dummy_tools = {}

        async def mock_search_google(query: str) -> str:
            await asyncio.sleep(0.1)
            if "Tasty Cola products" in query:
                return "Found two main products: 'Tasty Cola Classic' and 'Tasty Cola Zero'. Please specify which one to get details for."
            return f"Search results for '{query}'."

        mock_search_google.__doc__ = "Searches Google for the query."

        async def mock_extract_content(url: str) -> str:
            await asyncio.sleep(0.1)
            if "tastycolaclassic.com" in url:
                return "Details for Tasty Cola Classic: The original, sugary delight since 1903."
            if "tastycolazero.com" in url:
                return "Details for Tasty Cola Zero: Same great taste, zero sugar, launched in 2005."
            return f"Extracted content from '{url}'."

        mock_extract_content.__doc__ = "Extracts content from the given URL."

        import inspect

        dummy_tools["search_google"] = mock_search_google
        dummy_tools["search_google"].__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    "query",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=str,
                ),
            ],
            return_annotation=str,
        )
        dummy_tools["extract_content"] = mock_extract_content
        dummy_tools["extract_content"].__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    "url",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=str,
                ),
            ],
            return_annotation=str,
        )
        return dummy_tools

    monkeypatch.setattr(actor, "_build_tools", patched_build_tools_for_clarification)
    if hasattr(actor, "_tools_cache"):
        actor._tools_cache = None

    task_description = (
        "Search for 'Tasty Cola products'. "
        "If multiple main products are found, you MUST ask for clarification on which product to focus on using the 'request_clarification_from_plan_caller' tool. "
        "After clarification, extract content for the specified product's website (assume it's productname.com)."
    )

    plan_handle = await actor.act(
        task_description,
        clarification_up_q=clarification_up_q,
        clarification_down_q=clarification_down_q,
    )

    question_from_plan = ""
    try:
        question_from_plan = await asyncio.wait_for(
            clarification_up_q.get(),
            timeout=60,
        )
    except asyncio.TimeoutError:
        pytest.fail("Test (Clarification): Timed out waiting for question from plan.")

    assert (
        "which product" in question_from_plan.lower()
        or "tasty cola classic" in question_from_plan.lower()
        and "tasty cola zero" in question_from_plan.lower()
    ), f"Unexpected clarification question: {question_from_plan}"

    clarification_answer = "Please focus on Tasty Cola Classic."
    await clarification_down_q.put(clarification_answer)

    final_result = ""
    try:
        final_result = await asyncio.wait_for(plan_handle.result(), timeout=60)
    except asyncio.TimeoutError:
        pytest.fail(
            "Test (Clarification): Timed out waiting for plan final result after providing clarification.",
        )

    assert "tasty cola classic" in final_result.lower()
    assert "original, sugary delight" in final_result.lower()
    assert (
        "tasty cola zero" not in final_result.lower()
        or "Details for Tasty Cola Zero" not in final_result
    )

    await actor.close()
