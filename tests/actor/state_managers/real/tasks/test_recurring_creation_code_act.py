from __future__ import annotations

import pytest

from tests.actor.state_managers.utils import make_code_act_actor
from unity.task_scheduler.types.status import Status

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_code_act_creates_live_recurring_task_with_null_entrypoint():
    async with make_code_act_actor(impl="real", exposed_managers={"tasks"}) as (
        actor,
        primitives,
        calls,
    ):
        handle = await actor.act(
            (
                "Create exactly one live scheduled recurring task using "
                "primitives.tasks.update. Name it exactly 'Controlled weekly AI report'. "
                "Description: Every Monday at 12:00 UTC, research important AI and "
                "agentic AI work from the previous week, summarize the key developments, "
                "and email me a concise report. Set the first run for the next Monday "
                "at 12:00 UTC and repeat weekly. Do not create or attach any entrypoint "
                "function, do not mark it offline, and do not execute it now."
            ),
            clarification_enabled=False,
        )
        result = await handle.result()

        assert result is not None
        assert "primitives.tasks.update" in set(calls)

        rows = primitives.tasks._filter_tasks(filter="task_id >= 0")
        task = [row for row in rows if row.name == "Controlled weekly AI report"][0]
        assert task.offline is False
        assert task.entrypoint is None
        assert task.schedule is not None
        assert task.repeat is not None
        assert task.status == Status.scheduled


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_code_act_creates_live_triggerable_task_with_null_entrypoint():
    async with make_code_act_actor(impl="real", exposed_managers={"tasks"}) as (
        actor,
        primitives,
        calls,
    ):
        handle = await actor.act(
            (
                "Create exactly one live triggerable task using primitives.tasks.update. "
                "Name it exactly 'Controlled invoice email follow-up'. Description: "
                "Whenever an inbound email about invoices arrives, summarize the email, "
                "identify the needed action, and draft a reply for review. Use an email "
                "trigger, leave entrypoint null, do not mark it offline, and do not "
                "execute it now."
            ),
            clarification_enabled=False,
        )
        result = await handle.result()

        assert result is not None
        assert "primitives.tasks.update" in set(calls)

        rows = primitives.tasks._filter_tasks(filter="task_id >= 0")
        task = [
            row for row in rows if row.name == "Controlled invoice email follow-up"
        ][0]
        assert task.offline is False
        assert task.entrypoint is None
        assert task.trigger is not None
        assert task.status == Status.triggerable
