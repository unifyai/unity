import pytest
import asyncio
import functools
import unify

from unity.common.llm_helpers import start_async_tool_use_loop
from unity.planner.simulated import SimulatedPlanner, SimulatedActiveTask
from tests.helpers import _handle_project, _get_unity_test_env_var


# Fixtures to create a real LLM client for each test
def make_client(system_message: str):
    client = unify.AsyncUnify(
        "gpt-4o@openai",
        cache=_get_unity_test_env_var("UNIFY_CACHE"),
        traced=_get_unity_test_env_var("UNIFY_TRACED"),
    )
    client.set_system_message(system_message)
    return client


@pytest.mark.asyncio
@_handle_project
async def test_start_and_ask_simulated_plan(monkeypatch):
    """
    Test that the outer loop can ask questions to the simulated plan via the dynamic _ask_ helper.
    """
    planner = SimulatedPlanner(steps=1)
    # Count how many times ask is invoked
    ask_called = {"count": 0}
    original_ask = SimulatedActiveTask.ask

    @functools.wraps(original_ask)
    def ask(self, question: str) -> str:
        ask_called["count"] += 1
        return original_ask(self, question)

    monkeypatch.setattr(SimulatedActiveTask, "ask", ask, raising=True)

    system = (
        "You are running inside an automated test.\n"
        "1️⃣ Call `plan` with argument task_description='perform research on a Tasty Cola Ltd.'.\n"
        "2️⃣ When the user says 'ask', call the  `_ask_plan_call_` **once**, and ask if there are any early findings already\n"
        "3️⃣ Finally, regardless of the response to this question, just reply back to the user with exactly 'done', without calling any more tools."
    )
    client = make_client(system)
    handle = start_async_tool_use_loop(
        client=client,
        message="begin",
        tools={"execute": planner.execute},
        max_steps=20,
        timeout=120,
    )
    await asyncio.sleep(5)
    await handle.interject("ask")
    final = await handle.result()
    assert "done" in final.strip().lower()
    # ask should have been called exactly once
    assert ask_called["count"] == 1, "._ask should be invoked exactly once"


@pytest.mark.asyncio
@_handle_project
async def test_interject_simulated_plan(monkeypatch):
    """
    Test that the outer loop can interject instructions into the simulated plan via the `_interject_` helper.
    """
    planner = SimulatedPlanner(steps=1)
    interjected = {"count": 0, "msgs": []}
    original_interject = SimulatedActiveTask.interject

    @functools.wraps(original_interject)
    def interject(self, instruction: str) -> str:
        interjected["count"] += 1
        interjected["msgs"].append(instruction)
        return original_interject(self, instruction)

    monkeypatch.setattr(SimulatedActiveTask, "interject", interject, raising=True)

    system = (
        "You are running inside an automated test.\n"
        "1️⃣ Call `plan` with argument task='perform research on Tasty Cola Ltd.'.\n"
        "2️⃣ When the user says 'adjust', call the helper starting with `_interject_plan_call_` **once**, explaining that we want to know what their revenue is, as part of the research.\n"
        "3️⃣ Finally, reply to the user with 'interjection sent'."
    )
    client = make_client(system)
    handle = start_async_tool_use_loop(
        client=client,
        message="kickoff",
        tools={"execute": planner.execute},
        max_steps=20,
        timeout=120,
    )
    # wait for initial scheduling
    await asyncio.sleep(5)
    await handle.interject("adjust")
    final = await handle.result()
    assert "interjection sent" in final.strip().lower()
    assert interjected["count"] == 1, "._interject should be called exactly once"
    assert "revenue" in interjected["msgs"][0].lower(), "Interjection payload incorrect"


@pytest.mark.asyncio
@_handle_project
async def test_pause_and_resume_simulated_plan(monkeypatch):
    """
    Test that the outer loop can pause and resume the simulated plan via `_pause_` and `_resume_` helpers.
    """
    planner = SimulatedPlanner(steps=2)
    counts = {"pause": 0, "resume": 0}
    original_pause = SimulatedActiveTask.pause

    @functools.wraps(original_pause)
    def pause(self) -> str:
        counts["pause"] += 1
        return original_pause(self)

    original_resume = SimulatedActiveTask.resume

    @functools.wraps(original_resume)
    def resume(self) -> str:
        counts["resume"] += 1
        return original_resume(self)

    monkeypatch.setattr(SimulatedActiveTask, "pause", pause, raising=True)
    monkeypatch.setattr(SimulatedActiveTask, "resume", resume, raising=True)

    system = (
        "You are running inside an automated test.\n"
        "1️⃣ Call `execute` with argument `task_description='perform research on Tasty Cola Ltd.'`.\n"
        "2️⃣ When the user says 'hold', call the helper starting with `_pause_execute_call_`.\n"
        "3️⃣ When the user says 'go', call the helper starting with `_resume_execute_call_`.\n"
        "4️⃣ After resume, reply with 'done'."
    )
    client = make_client(system)
    handle = start_async_tool_use_loop(
        client=client,
        message="run",
        tools={"execute": planner.execute},
        max_steps=30,
        timeout=180,
    )
    await asyncio.sleep(5)
    await handle.interject("hold")
    await asyncio.sleep(5)
    await handle.interject("go")
    final = await handle.result()
    assert "done" in final.strip().lower()
    assert counts == {
        "pause": 1,
        "resume": 1,
    }, "pause/resume should each be called once"


@pytest.mark.asyncio
@_handle_project
async def test_stop_simulated_plan(monkeypatch):
    """
    Test that the outer loop can stop the simulated plan via `_stop_` helper.
    """
    planner = SimulatedPlanner(steps=1)
    stopped = {"count": 0}
    original_stop = SimulatedActiveTask.stop

    @functools.wraps(original_stop)
    def stop(self) -> str:
        stopped["count"] += 1
        return original_stop(self)

    monkeypatch.setattr(SimulatedActiveTask, "stop", stop, raising=True)

    system = (
        "You are running inside an automated test.\n"
        "1️⃣ Call `plan` with argument task='perform research on Tasty Cola Ltd.'.\n"
        "2️⃣ When the user says 'stop it', call the helper starting with `_stop_plan_call_`.\n"
        "3️⃣ Finally, reply with 'stopped'."
    )
    client = make_client(system)
    handle = start_async_tool_use_loop(
        client=client,
        message="begin",
        tools={"execute": planner.execute},
        max_steps=20,
        timeout=120,
    )
    await asyncio.sleep(5)
    await handle.interject("stop it")
    final = await handle.result()
    assert "stopped" in final.strip().lower()
    assert stopped["count"] == 1, "._stop should be called exactly once"


@pytest.mark.asyncio
@_handle_project
async def test_plan_requests_clarification():
    """
    The planner should send a clarification question over `clarification_up_q`
    and wait for the reply on `clarification_down_q` before finishing.
    """
    planner = SimulatedPlanner(steps=1, _requests_clarification=True)

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    # start a plan that needs clarification
    active_task = await planner.execute(
        "Compile the quarterly report",
        clarification_up_q=up_q,
        clarification_down_q=down_q,
    )

    # the plan must ask a clarification question first
    question = await asyncio.wait_for(up_q.get(), timeout=60)
    assert "clarify" in question.lower()

    # provide the clarification answer
    await down_q.put("Yes, please compile the Q1 report now.")

    # the final result should propagate the clarification answer
    result = await active_task.result()
    assert "q1 report" in result.lower()
