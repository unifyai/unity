from __future__ import annotations

import time
import re
import pytest

from unity.common.async_tool_loop import (
    start_async_tool_loop,
    AsyncToolLoopHandle,
)
from unity.events.event_bus import EVENT_BUS
from tests.helpers import _handle_project, capture_events
from unity.common.llm_client import new_llm_client

# All tests in this file require EventBus publishing to verify event behavior
pytestmark = pytest.mark.enable_eventbus
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_condition,
)
from unity.contact_manager.contact_manager import ContactManager


@pytest.mark.asyncio
async def test_nested_logging_hierarchy_labels(model):
    """
    Verify that nested async tool loops emit ToolLoop events with hierarchical
    lineage in payload: `hierarchy` (list[str]) and `hierarchy_label` (str).

    We create an outer loop (loop_id="Outer") whose tool starts an inner loop
    (loop_id="Inner"). We assert that events exist for both levels:
    - hierarchy == ["Outer"]
    - hierarchy == ["Outer", "Inner"] with label "Outer -> Inner"
    """

    # ── inner tool: trivial sync function ──────────────────────────────────
    def inner_tool() -> str:  # noqa: D401
        time.sleep(0.1)
        return "inner-ok"

    # ── outer tool: launches a nested loop and returns its handle ──────────
    async def outer_tool() -> AsyncToolLoopHandle:
        inner_client = new_llm_client(model=model)
        inner_client.set_system_message(
            "You are running inside an automated test.\n"
            "1️⃣  Call `inner_tool` (no arguments).\n"
            "2️⃣  Wait for its response.\n"
            "3️⃣  Reply with exactly 'done'.",
        )

        return start_async_tool_loop(
            client=inner_client,
            message="start",
            tools={"inner_tool": inner_tool},
            loop_id="Inner",
            max_steps=10,
            timeout=120,
        )

    outer_tool.__name__ = "outer_tool"
    outer_tool.__qualname__ = "outer_tool"

    # ── top-level loop: uses the outer tool ────────────────────────────────
    client = new_llm_client(model=model)
    client.set_system_message(
        "You are running inside an automated test. Perform the steps exactly:\n"
        "1️⃣  Call `outer_tool` with no arguments.\n"
        "2️⃣  Continue running this tool call, when given the option.\n"
        "3️⃣  Once it is completed, respond with exactly 'outer done'.",
    )

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"outer_tool": outer_tool},
        loop_id="Outer",
        max_steps=10,
        timeout=240,
    )

    # Wait for completion
    final_reply = await handle.result()
    assert final_reply is not None, "Loop should complete with a response"

    # Gather recent ToolLoop events
    events = await EVENT_BUS.search(filter="type == 'ToolLoop'", limit=200)

    # Presence checks for hierarchy payloads
    has_outer_only = any(
        (evt.payload or {}).get("hierarchy") == ["Outer"] for evt in events
    )
    has_outer_inner = any(
        (evt.payload or {}).get("hierarchy") == ["Outer", "Inner"] for evt in events
    )
    has_outer_inner_label = any(
        isinstance((evt.payload or {}).get("hierarchy_label"), str)
        and re.fullmatch(
            r"Outer->Inner(?:\([0-9a-f]{4}\))?",
            (evt.payload or {}).get("hierarchy_label"),
        )
        for evt in events
    )

    assert has_outer_only, "No ToolLoop event recorded with hierarchy ['Outer']"
    assert (
        has_outer_inner
    ), "No ToolLoop event recorded with hierarchy ['Outer', 'Inner']"
    assert (
        has_outer_inner_label
    ), "No ToolLoop event recorded with hierarchy_label 'Outer -> Inner'"


@pytest.mark.asyncio
async def test_single_loop_logging_hierarchy_label(model):
    """
    Verify that a single (non-nested) async tool loop emits ToolLoop events
    with a flat hierarchy and label equal to its loop_id.

    We start a solo loop with loop_id="Solo" and a trivial tool.
    Assertions:
    - hierarchy == ["Solo"] exists
    - hierarchy_label == "Solo" exists
    - no event exists with hierarchy beginning ["Solo", ...] (i.e., nested)
    """

    def noop_tool() -> str:  # noqa: D401
        return "ok"

    client = new_llm_client(model=model)
    client.set_system_message(
        "1️⃣  Call `noop_tool`. 2️⃣ Then reply exactly 'done'.",
    )

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"noop_tool": noop_tool},
        loop_id="Solo",
        max_steps=10,
        timeout=120,
    )

    final_reply = await handle.result()
    assert final_reply is not None, "Loop should complete with a response"

    events = await EVENT_BUS.search(filter="type == 'ToolLoop'", limit=200)

    has_solo = any((evt.payload or {}).get("hierarchy") == ["Solo"] for evt in events)
    has_solo_label = any(
        isinstance((evt.payload or {}).get("hierarchy_label"), str)
        and re.fullmatch(
            r"Solo(?:\([0-9a-f]{4}\))?",
            (evt.payload or {}).get("hierarchy_label"),
        )
        for evt in events
    )
    has_nested_under_solo = any(
        isinstance((evt.payload or {}).get("hierarchy"), list)
        and (evt.payload or {}).get("hierarchy")[:1] == ["Solo"]
        and len((evt.payload or {}).get("hierarchy")) > 1
        for evt in events
    )

    assert has_solo, "No ToolLoop event recorded with hierarchy ['Solo']"
    assert has_solo_label, "No ToolLoop event recorded with hierarchy_label 'Solo'"
    assert not has_nested_under_solo, "Unexpected nested hierarchy found under 'Solo'"


@pytest.mark.asyncio
async def test_nested_steer_interject_logging_has_child_label_and_origin_marker(
    model,
    caplog,
):
    """
    Verify that nested_steer emits a pre-call interject log using the child's loop label,
    and marks the entry as coming via nested_steer.
    """

    # ── inner tool: trivial sync function ──────────────────────────────────
    def inner_tool() -> str:  # noqa: D401
        # Keep the child loop alive long enough for adoption + steering
        time.sleep(0.2)
        return "inner-ok"

    # ── outer tool: launches a nested loop and returns its handle ──────────
    async def outer_tool() -> AsyncToolLoopHandle:
        inner_client = new_llm_client(model=model)
        inner_client.set_system_message(
            "You are running inside an automated test.\n"
            "1️⃣  Call `inner_tool` (no arguments).\n"
            "2️⃣  Wait for its response.\n"
            "3️⃣  Reply with exactly 'done'.",
        )

        return start_async_tool_loop(
            client=inner_client,
            message="start",
            tools={"inner_tool": inner_tool},
            loop_id="Inner",
            max_steps=10,
            timeout=120,
        )

    outer_tool.__name__ = "outer_tool"
    outer_tool.__qualname__ = "outer_tool"

    # ── top-level loop: uses the outer tool ────────────────────────────────
    client = new_llm_client(model=model)
    client.set_system_message(
        "You are running inside an automated test. Perform the steps exactly:\n"
        "1️⃣  Call `outer_tool` with no arguments.\n"
        "2️⃣  Continue running this tool call, when given the option.\n"
        "3️⃣  Once it is completed, respond with exactly 'outer done'.",
    )

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"outer_tool": outer_tool},
        loop_id="Outer",
        max_steps=10,
        timeout=240,
    )

    # Capture logs at DEBUG for the unity logger
    import logging

    caplog.set_level(logging.DEBUG, logger="unity")

    try:
        # Wait until assistant has requested the outer tool
        await _wait_for_tool_request(client, "outer_tool")

        # Wait until the nested inner handle is adopted and visible
        async def _child_adopted():
            try:
                ti = getattr(handle._task, "task_info", {})  # type: ignore[attr-defined]
                if isinstance(ti, dict):
                    return any(
                        getattr(meta, "name", None) == "outer_tool"
                        and getattr(meta, "handle", None) is not None
                        for meta in ti.values()
                    )
            except Exception:
                return False
            return False

        await _wait_for_condition(_child_adopted, poll=0.01, timeout=60.0)

        # Issue nested_steer interject against the child (by handle base name)
        msg = "hello from nested_steer"
        spec = {
            "children": [
                {
                    "handle": "AsyncToolLoopHandle",
                    "steps": [{"method": "interject", "args": msg}],
                },
            ],
        }
        await handle.nested_steer(spec)  # type: ignore[attr-defined]

        # Assert a nested_steer pre-log was emitted with the child loop label and origin marker
        text = caplog.text
        import re

        # Expect a line like: "💬 [Outer->Inner(xxxx)] Interject requested: hello ... – via nested_steer"
        assert re.search(
            r"💬 \[Outer->Inner\([0-9a-f]{4}\)\] Interject requested: .*via nested_steer",
            text,
        ), "Expected nested_steer interject log with child label and origin marker"
    finally:
        # Finish the loop to avoid leaking tasks
        try:
            await handle.result()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_nested_steer_pause_resume_logging_have_child_label_and_origin_marker(
    model,
    caplog,
):
    """
    Verify that nested_steer emits pre-call pause/resume logs using the child's loop label,
    and marks entries as coming via nested_steer.
    """

    # ── inner tool: trivial sync function ──────────────────────────────────
    def inner_tool() -> str:  # noqa: D401
        time.sleep(0.2)
        return "inner-ok"

    # ── outer tool: launches a nested loop and returns its handle ──────────
    async def outer_tool() -> AsyncToolLoopHandle:
        inner_client = new_llm_client(model=model)
        inner_client.set_system_message(
            "You are running inside an automated test.\n"
            "1️⃣  Call `inner_tool` (no arguments).\n"
            "2️⃣  Wait for its response.\n"
            "3️⃣  Reply with exactly 'done'.",
        )

        return start_async_tool_loop(
            client=inner_client,
            message="start",
            tools={"inner_tool": inner_tool},
            loop_id="Inner",
            max_steps=10,
            timeout=120,
        )

    outer_tool.__name__ = "outer_tool"
    outer_tool.__qualname__ = "outer_tool"

    client = new_llm_client(model=model)
    client.set_system_message(
        "You are running inside an automated test. Perform the steps exactly:\n"
        "1️⃣  Call `outer_tool` with no arguments.\n"
        "2️⃣  Continue running this tool call, when given the option.\n"
        "3️⃣  Once it is completed, respond with exactly 'outer done'.",
    )

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"outer_tool": outer_tool},
        loop_id="Outer",
        max_steps=10,
        timeout=240,
    )

    import logging

    caplog.set_level(logging.DEBUG, logger="unity")

    try:
        await _wait_for_tool_request(client, "outer_tool")

        async def _child_adopted():
            try:
                ti = getattr(handle._task, "task_info", {})  # type: ignore[attr-defined]
                if isinstance(ti, dict):
                    return any(
                        getattr(meta, "name", None) == "outer_tool"
                        and getattr(meta, "handle", None) is not None
                        for meta in ti.values()
                    )
            except Exception:
                return False
            return False

        await _wait_for_condition(_child_adopted, poll=0.01, timeout=60.0)

        # Issue pause then resume via nested_steer
        spec = {
            "children": [
                {
                    "handle": "AsyncToolLoopHandle",
                    "steps": [{"method": "pause"}, {"method": "resume"}],
                },
            ],
        }
        await handle.nested_steer(spec)  # type: ignore[attr-defined]

        text = caplog.text
        import re

        # Expect lines like:
        # "⏸️ [Outer->Inner(xxxx)] Pause requested – via nested_steer"
        # "▶️ [Outer->Inner(xxxx)] Resume requested – via nested_steer"
        assert re.search(
            r"⏸️ \[Outer->Inner\([0-9a-f]{4}\)\] Pause requested – via nested_steer",
            text,
        ), "Expected nested_steer pause log with child label and origin marker"
        assert re.search(
            r"▶️ \[Outer->Inner\([0-9a-f]{4}\)\] Resume requested – via nested_steer",
            text,
        ), "Expected nested_steer resume log with child label and origin marker"
    finally:
        try:
            await handle.result()
        except Exception:
            pass


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_replay_logs_with_manager(caplog):
    """
    Verify that deserialization replays seeded assistant/tool messages with:
    - EventBus ToolLoop events carrying origin == 'deserialize'
    - Terminal logs including the loop label and a 'via deserialize' marker.
    """
    cm = ContactManager()
    # Seed a contact to avoid empty search/filter churn
    cm._create_contact(
        first_name="Alpha",
        surname="Tester",
        email_address="alpha@example.com",
        bio="Alpha user",
    )
    handle = await cm.ask("Find contact Alpha and provide a brief answer.")

    # Wait until we have at least one assistant tool_call and one tool message in transcript
    async def _ready():
        try:
            msgs = handle.get_history() or []
        except Exception:
            msgs = []
        has_asst = any(
            m.get("role") == "assistant" and (m.get("tool_calls") or []) for m in msgs
        )
        has_tool = any(m.get("role") == "tool" for m in msgs)
        return has_asst and has_tool

    await _wait_for_condition(_ready, poll=0.02, timeout=120.0)

    snap = handle.serialize()

    # Capture logs; expect replay markers
    import logging as _logging

    caplog.set_level(_logging.DEBUG, logger="unity")

    async with capture_events("ToolLoop") as captured_events:
        resumed = AsyncToolLoopHandle.deserialize(snap)
        # Complete resumed loop to flush all replay logs/events
        await resumed.result()

    # EventBus: at least one ToolLoop event must have origin == 'deserialize'
    has_origin_deser = any(
        (evt.payload or {}).get("origin") == "deserialize" for evt in captured_events
    )
    assert (
        has_origin_deser
    ), "No ToolLoop events found with origin == 'deserialize' after resume"

    # Logs: expect a deserialize banner, a user replay suffix, and an assistant scheduled marker for ContactManager.ask
    text = caplog.text
    banner_re = (
        r"📦 \[ContactManager\.ask\([0-9a-f]{4}\)\] Deserializing \d+ Message\(s\)…"
    )
    assistant_scheduled_re = r"🤖 \[ContactManager\.ask\([0-9a-f]{4}\)\] Assistant scheduled: .* – via deserialize(?: 📦)?"
    user_re = r"🧑‍💻 \[ContactManager\.ask\([0-9a-f]{4}\)\] User Message: .* – via deserialize(?: 📦)?"
    assert re.search(
        banner_re,
        text,
    ), "Expected deserialize banner for ContactManager.ask"
    assert re.search(
        user_re,
        text,
    ), "Expected 'User Message: ... – via deserialize' for ContactManager.ask loop"
    assert re.search(
        assistant_scheduled_re,
        text,
    ), "Expected assistant scheduled replay log with ContactManager.ask label and 'via deserialize' marker"


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_replay_nested_loops(caplog):
    """
    Verify deserialization replay for a nested loop triggered by ContactManager.update
    (whose default tool policy requires `ask` on the first step).
    Assertions:
    - Nested replay logs use a label like 'ContactManager.update->ContactManager.ask(xxxx)' with 'via deserialize'.
    - ToolLoop events include origin == 'deserialize' on replayed messages and show nested hierarchy/label.
    """
    cm = ContactManager()
    # Trigger update with an instruction; first step policy requires `ask`, creating a nested child loop.
    handle = await cm.update("Please check contacts and then do nothing.")

    # Wait until the nested `ask` child has been adopted AND has produced an assistant message
    async def _ask_child_ready_for_snapshot():
        try:
            task_info = getattr(getattr(handle, "_task", None), "task_info", {})  # type: ignore[attr-defined]
            if isinstance(task_info, dict):
                for meta in task_info.values():
                    nm = getattr(meta, "name", None)
                    hd = getattr(meta, "handle", None)
                    if nm == "ask" and hd is not None:
                        # Ensure it's still alive (if it finished, we missed the window)
                        if hasattr(hd, "done") and hd.done():
                            return False
                        # Check history for assistant message so we have something to assert replay on
                        hist = hd.get_history()
                        if any(m.get("role") == "assistant" for m in hist):
                            return True
        except Exception:
            return False
        return False

    await _wait_for_condition(_ask_child_ready_for_snapshot, poll=0.02, timeout=60.0)

    # Take recursive snapshot to capture the in-flight child
    snap = handle.serialize(recursive=True)

    # Capture logs at DEBUG for replay
    import logging as _logging

    caplog.set_level(_logging.DEBUG, logger="unity")

    async with capture_events("ToolLoop") as captured_events:
        resumed = AsyncToolLoopHandle.deserialize(snap)
        await resumed.result()

    # Logs: tighten to require nested deserialize banner, nested user replay suffix, and nested assistant scheduled marker
    text = caplog.text
    nested_banner = r"📦 \[ContactManager\.update->ContactManager\.ask\([0-9a-f]{4}\)\] Deserializing \d+ Message\(s\)…"
    nested_user = r"🧑‍💻 \[ContactManager\.update->ContactManager\.ask\([0-9a-f]{4}\)\] User Message: .* – via deserialize(?: 📦)?"
    nested_assistant = r"🤖 \[ContactManager\.update->ContactManager\.ask\([0-9a-f]{4}\)\] Assistant scheduled: .* – via deserialize(?: 📦)?"
    assert re.search(
        nested_banner,
        text,
    ), "Expected nested child deserialize banner for ContactManager.update->ContactManager.ask"
    assert re.search(
        nested_user,
        text,
    ), "Expected nested child 'User Message: ... – via deserialize' line"
    assert re.search(
        nested_assistant,
        text,
    ), "Expected nested child assistant scheduled replay marker line"

    # Events: at least one nested ToolLoop event with origin == 'deserialize'
    has_nested_deser = any(
        isinstance((evt.payload or {}).get("hierarchy"), list)
        and len((evt.payload or {}).get("hierarchy")) >= 2
        and (evt.payload or {}).get("origin") == "deserialize"
        for evt in captured_events
    )
    assert (
        has_nested_deser
    ), "No nested ToolLoop events with origin == 'deserialize' after resume"

    # And a nested hierarchy_label with the expected format
    has_nested_label = any(
        isinstance((evt.payload or {}).get("hierarchy_label"), str)
        and re.fullmatch(
            r"ContactManager\.update->ContactManager\.ask\([0-9a-f]{4}\)",
            (evt.payload or {}).get("hierarchy_label"),
        )
        for evt in captured_events
    )
    assert (
        has_nested_label
    ), "No 'hierarchy_label' like 'ContactManager.update->ContactManager.ask(xxxx)' after resume"


@pytest.mark.asyncio
async def test_litellm_logs_are_suppressed(model, caplog):
    """
    Verify that LiteLLM logs are suppressed by our logging configuration.

    LiteLLM creates loggers like 'LiteLLM', 'LiteLLM Proxy', 'LiteLLM Router'.
    These should be muted to WARNING level so INFO logs don't pollute output.
    This test catches regressions if LiteLLM changes logger names or adds new ones.
    """
    import logging

    # Capture all logs at DEBUG level to ensure we catch everything
    caplog.set_level(logging.DEBUG)

    def noop_tool() -> str:
        return "ok"

    client = new_llm_client(model=model)
    client.set_system_message("Call noop_tool, then reply 'done'.")

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"noop_tool": noop_tool},
        loop_id="LiteLLMTest",
        max_steps=5,
        timeout=60,
    )

    await handle.result()

    # Check that no log records come from LiteLLM loggers
    litellm_logs = [
        record for record in caplog.records if "litellm" in record.name.lower()
    ]

    assert not litellm_logs, (
        f"Found {len(litellm_logs)} log(s) from LiteLLM loggers that should be suppressed. "
        f"Logger names: {sorted(set(r.name for r in litellm_logs))}"
    )
