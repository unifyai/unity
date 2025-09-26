from __future__ import annotations

import asyncio
import functools
import pytest
import unify

from unity.conductor.simulated import SimulatedConductor
from unity.common.async_tool_loop import AsyncToolUseLoopHandle

# keeps each test isolated in its own Unify project / trace context
from tests.helpers import _handle_project


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Doc-string inheritance                                                 #
# ─────────────────────────────────────────────────────────────────────────────
def test_simulated_cond_docstrings_match_base():
    """
    Public methods in SimulatedConductor should copy the real BaseConductor
    doc-strings one-for-one (via functools.wraps).
    """
    from unity.conductor.base import BaseConductor
    from unity.conductor.simulated import SimulatedConductor

    assert (
        BaseConductor.ask.__doc__.strip() in SimulatedConductor.ask.__doc__.strip()
    ), ".ask doc-string was not copied correctly"

    assert (
        BaseConductor.request.__doc__.strip()
        in SimulatedConductor.request.__doc__.strip()
    ), ".request doc-string was not copied correctly"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Basic start-and-ask                                                    #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_start_and_ask_simulated_cond():
    cond = SimulatedConductor(
        description=(
            "Operations assistant for Acme Real Estate handling listings, client follow-ups, "
            "and internal knowledge."
        ),
    )
    h = await cond.ask(
        "Which follow-ups and tasks are due today across active buyers and listings?",
    )
    answer = await h.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Reasoning steps toggle (ask + request)                                 #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_reasoning_steps_toggle_ask_and_request():
    cond = SimulatedConductor(
        description=(
            "Account management assistant summarizing priorities across active opportunities and customers."
        ),
    )

    # ask() – request hidden messages tuple
    h1 = await cond.ask(
        "List top priorities across active opportunities and existing customers.",
        _return_reasoning_steps=True,
    )
    ans1, msgs1 = await h1.result()
    assert isinstance(ans1, str) and ans1.strip()
    assert isinstance(msgs1, list) and len(msgs1) >= 1

    # request() – also return (answer, messages)
    h2 = await cond.request(
        (
            "Create a high-priority task to call Bob tomorrow at 09:00 and log it against the "
            "'Renewal – Contoso' account."
        ),
        _return_reasoning_steps=True,
    )
    ans2, msgs2 = await h2.result()
    assert isinstance(ans2, str) and ans2.strip()
    assert isinstance(msgs2, list) and len(msgs2) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Write (request) then Read (ask) – state carries via sub-managers        #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_cond_request_then_ask_stateful():
    """
    A `request()` that (likely) touches the TaskScheduler should influence a
    subsequent `ask()` routed to the same sub-manager (stateful LLM behind it).
    """
    cond = SimulatedConductor(
        description=(
            "Project management assistant coordinating FY26 planning tasks and priorities."
        ),
    )
    task_name = "Draft Budget FY26"

    # 1) Request creation of a high-priority task
    h_upd = await cond.request(
        f"Please create a new task called '{task_name}' with high priority.",
    )
    await h_upd.result()

    # 2) Ask about high-priority tasks – the answer should reference our task
    h_q = await cond.ask("Which tasks are high priority right now?")
    answer = (await h_q.result()).lower()

    assert "budget" in answer, "Answer should reference the task added via request()"


# ─────────────────────────────────────────────────────────────────────────────
# Steerable handle tests                                                     #
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Interject                                                             #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_interject(monkeypatch):
    calls = {"interject": 0}
    orig = AsyncToolUseLoopHandle.interject

    @functools.wraps(orig)
    async def wrapped(self, message: str):  # type: ignore[override]
        calls["interject"] += 1
        return await orig(self, message)

    monkeypatch.setattr(AsyncToolUseLoopHandle, "interject", wrapped, raising=True)

    cond = SimulatedConductor(
        description=(
            "SaaS launch week assistant coordinating tasks, contacts, transcripts, and KB."
        ),
    )
    h = await cond.ask(
        "Draft a morning brief covering high-priority tasks, top contacts to reach out to, "
        "and notable messages in the last 24h.",
    )
    await asyncio.sleep(0.05)
    await h.interject("Also call out anything due today and any blockers.")

    await h.result()
    assert calls["interject"] == 1, ".interject should be invoked exactly once"


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Stop                                                                  #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_stop():
    cond = SimulatedConductor(
        description=(
            "Data team sprint planning assistant summarizing backlog, PRs, and stakeholder requests."
        ),
    )
    h = await cond.ask(
        "Produce a comprehensive weekly report across backlog, open PRs, and pending stakeholder requests.",
    )
    await asyncio.sleep(0.05)
    h.stop()
    await h.result()
    assert h.done(), "Handle should report done after stop()"


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Clarification handshake                                                 #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.slow
@pytest.mark.asyncio
@_handle_project
async def test_cond_supports_optional_clarification_channels():
    """
    Conductor provides an optional `request_clarification` tool to the loop when
    caller supplies duplex queues. We do not enforce that the LLM must use it,
    only that the presence of queues does not break execution.
    """
    cond = SimulatedConductor(
        description=(
            "Marketing webinar assistant planning outreach and post-event follow-ups for 'AI for Finance'."
        ),
    )

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    h = await cond.ask(
        (
            "Outline a follow-up sequence for registrants of the 'AI for Finance' webinar; "
            "include channels and timing. Ask for any necessary details before proceeding."
        ),
        clarification_up_q=up_q,
        clarification_down_q=down_q,
    )

    # If the model chooses to ask for clarification, answer it with an LLM-generated reply.
    async def _maybe_answer_clarification():
        try:
            q = await asyncio.wait_for(up_q.get(), timeout=3.0)
            assert isinstance(q, str) and q.strip()
            client = unify.AsyncUnify(
                "o4-mini@openai",
                cache=True,
                traced=False,
            )
            client.set_system_message(
                "You answer clarification questions concisely and specifically. "
                "Provide concrete parameters (counts, timing, channels) when relevant.",
            )
            ans = await client.generate(q)
            await down_q.put(ans)
        except asyncio.TimeoutError:
            # Model didn't ask for clarification – carry on.
            pass

    clar_task = asyncio.create_task(_maybe_answer_clarification())

    # We don't assert that a clarification is asked (LLM-dependent).
    ans = await h.result()
    await clar_task
    assert isinstance(ans, str) and ans.strip(), "Answer should be non-empty"


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Pause → Resume round-trip                                             #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_pause_and_resume(monkeypatch):
    counts = {"pause": 0, "resume": 0}

    original_pause = AsyncToolUseLoopHandle.pause

    @functools.wraps(original_pause)
    def _patched_pause(self):  # type: ignore[override]
        counts["pause"] += 1
        return original_pause(self)

    monkeypatch.setattr(
        AsyncToolUseLoopHandle,
        "pause",
        _patched_pause,
        raising=True,
    )

    original_resume = AsyncToolUseLoopHandle.resume

    @functools.wraps(original_resume)
    def _patched_resume(self):  # type: ignore[override]
        counts["resume"] += 1
        return original_resume(self)

    monkeypatch.setattr(
        AsyncToolUseLoopHandle,
        "resume",
        _patched_resume,
        raising=True,
    )

    cond = SimulatedConductor(
        description=(
            "Sales pipeline assistant tracking deals closing this week and required follow-ups."
        ),
    )
    handle = await cond.ask(
        "Generate a short summary of deals closing this week, related contacts, and required tasks.",
    )

    # Pause before awaiting the result
    handle.pause()

    # Start result() – it should block while paused
    res_task = asyncio.create_task(handle.result())
    await asyncio.sleep(0.1)
    assert not res_task.done(), "result() should block while the handle is paused"

    # Resume and ensure the task now completes
    handle.resume()

    answer = await asyncio.wait_for(res_task, timeout=60)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # Exactly one pause and one resume
    assert counts == {
        "pause": 1,
        "resume": 1,
    }, "pause/resume should each be invoked exactly once"


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Nested ask on handle                                                   #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_ask():
    """
    The internal handle returned by SimulatedConductor.request exposes a
    dynamic ask() method that should produce a nested handle whose result can
    be awaited independently of the parent.
    """
    cond = SimulatedConductor(
        description=(
            "Task triage assistant for today's workload across engineering and support."
        ),
    )
    handle = await cond.request(
        "Prepare a plan to triage today's tasks across engineering and customer support.",
    )

    # Ask a question about the running loop
    nested = await handle.ask(
        "What key steps have been identified so far and who owns them?",
    )
    nested_answer = await nested.result()

    assert isinstance(nested_answer, str) and nested_answer.strip()

    # The original handle should still be awaitable and produce an answer
    parent_answer = await handle.result()
    assert isinstance(parent_answer, str) and parent_answer.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 10.  WebSearcher tool is exposed on ask surface                             #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_simulated_cond_exposes_websearcher_tool_in_prompt():
    cond = SimulatedConductor(
        description=(
            "Research assistant combining internal managers with web research when needed."
        ),
    )

    h = await cond.ask(
        "Find notable updates on vector databases in 2025 and summarize.",
        _return_reasoning_steps=True,
    )
    answer, messages = await h.result()
    assert isinstance(answer, str) and answer.strip()

    # The system prompt should list available tools including SimulatedWebSearcher_ask
    system_msgs = [m.get("content", "") for m in messages if m.get("role") == "system"]
    blob = "\n".join(system_msgs).lower()
    assert (
        "simulatedwebsearcher_ask" in blob
    ), "WebSearcher ask tool should be exposed in the Conductor.ask tool list"
