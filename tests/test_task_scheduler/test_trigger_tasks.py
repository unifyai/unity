"""
Unit-tests for the new **event-triggered** task functionality.
"""

from __future__ import annotations
from datetime import datetime, timezone, timedelta

import pytest

from tests.helpers import _handle_project
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.status import Status
from unity.task_scheduler.types.schedule import Schedule
from unity.task_scheduler.types.trigger import Trigger, Medium


# --------------------------------------------------------------------------- #
# 1.  Creation – triggerable tasks                                            #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_create_triggerable_task():
    ts = TaskScheduler()

    trig = Trigger(
        medium=Medium.EMAIL,
        from_contact_ids=[42],
        interrupt=True,
        recurring=False,
    )

    task_id = ts._create_task(
        name="Wait for client reply",
        description="Begin work once the customer emails us back.",
        trigger=trig,
    )["details"]["task_id"]

    row = ts._filter_tasks(filter=f"task_id == {task_id}", limit=1)[0]
    assert row["status"] == Status.triggerable
    assert row["schedule"] is None
    assert row["trigger"]["medium"] == Medium.EMAIL
    assert row["trigger"]["from_contact_ids"] == [42]


# --------------------------------------------------------------------------- #
# 2.  Mutual exclusion (schedule × trigger)                                   #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_schedule_and_trigger_mutually_exclusive():
    ts = TaskScheduler()

    trig = Trigger(medium=Medium.SMS_MESSAGE)
    tsq = TaskScheduler()
    qid_tmp = tsq._allocate_new_queue_id()
    sched = Schedule(
        queue_id=qid_tmp,
        start_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )

    with pytest.raises(ValueError):
        ts._create_task(
            name="Bad combo task",
            description="Should fail – both schedule and trigger.",
            schedule=sched,
            trigger=trig,
        )


# --------------------------------------------------------------------------- #
# 3.  Adding a trigger to a *scheduled* task should fail                      #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_update_trigger_on_scheduled_task_raises():
    ts = TaskScheduler()

    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    qid = ts._allocate_new_queue_id()
    tid = ts._create_task(
        name="Morning maintenance window",
        description="Auto-patch servers tomorrow.",
        schedule=Schedule(queue_id=qid, start_at=future),
    )["details"]["task_id"]

    with pytest.raises(ValueError):
        ts._update_task_trigger(
            task_id=tid,
            new_trigger=Trigger(medium=Medium.WHATSAPP_MSG),
        )


# --------------------------------------------------------------------------- #
# 4.  Removing the trigger ⇢ status falls back to *queued*                    #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_clear_trigger_transitions_status():
    ts = TaskScheduler()
    trig = Trigger(medium=Medium.PHONE_CALL, interrupt=True)

    tid = ts._create_task(
        name="Answer support hotline",
        description="Pick up when the phone rings.",
        trigger=trig,
    )["details"]["task_id"]

    ts._update_task_trigger(task_id=tid, new_trigger=None)

    row = ts._filter_tasks(filter=f"task_id == {tid}", limit=1)[0]
    assert row["trigger"] is None
    assert row["status"] == Status.queued


# --------------------------------------------------------------------------- #
# 5.  Disallow start_at edits on trigger-based tasks                          #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_start_at_on_trigger_task_raises():
    ts = TaskScheduler()
    trig = Trigger(medium=Medium.WHATSAPP_CALL)

    tid = ts._create_task(
        name="Respond to WhatsApp call",
        description="Jump on the WhatsApp call when it comes in.",
        trigger=trig,
    )["details"]["task_id"]

    with pytest.raises(ValueError):
        ts._update_task_start_at(
            task_id=tid,
            new_start_at=datetime.now(timezone.utc) + timedelta(hours=2),
        )


# --------------------------------------------------------------------------- #
# 6.  Queue operations must reject trigger-tasks                              #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_update_queue_rejects_trigger_tasks():
    ts = TaskScheduler()

    # Normal queued task
    qid = ts._allocate_new_queue_id()
    ts._create_task(
        name="Prep deck",
        description="Slides.",
        schedule=Schedule(queue_id=qid),
    )

    # Trigger-based task
    trig_tid = ts._create_task(
        name="Pickup support call",
        description="Answer phone when VIP calls.",
        trigger=Trigger(medium=Medium.PHONE_CALL),
    )["details"]["task_id"]

    with pytest.raises(ValueError):
        ts._set_queue(queue_id=qid, order=[trig_tid, 0])


# --------------------------------------------------------------------------- #
# 7.  Instance cloning when a triggerable task is started                     #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.asyncio
async def test_triggerable_start_clones_instance():
    """
    Starting a **triggerable** task should:

    • promote the *oldest* instance (`instance_id` 0) to **active**
    • create a **new** row with the same `task_id` but `instance_id` 1
      that remains in the *triggerable* state
    """
    ts = TaskScheduler()

    trig = Trigger(medium=Medium.EMAIL, recurring=False)
    tid = ts._create_task(
        name="Wait for invoice approval",
        description="Start when finance emails us.",
        trigger=trig,
    )["details"]["task_id"]

    # One physical row before activation
    rows_before = ts._filter_tasks(filter=f"task_id == {tid}")
    assert len(rows_before) == 1 and rows_before[0]["instance_id"] == 0

    # Activate
    handle = await ts.execute(text=str(tid))

    # Two rows should now exist: 0 (active) and 1 (still triggerable)
    rows_after = ts._filter_tasks(filter=f"task_id == {tid}")
    assert len(rows_after) == 2

    status_by_inst = {r["instance_id"]: r["status"] for r in rows_after}
    assert status_by_inst[0] == "active"
    assert status_by_inst[1] == "triggerable"

    # Clean-up (avoid background thread leaks)
    handle.stop(cancel=True)
    await handle.result()
