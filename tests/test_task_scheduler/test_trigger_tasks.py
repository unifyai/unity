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

    row = ts._search_tasks(filter=f"task_id == {task_id}", limit=1)[0]
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
    sched = Schedule(start_at=datetime.now(timezone.utc) + timedelta(hours=1))

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
    tid = ts._create_task(
        name="Morning maintenance window",
        description="Auto-patch servers tomorrow.",
        schedule=Schedule(start_at=future),
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

    row = ts._search_tasks(filter=f"task_id == {tid}", limit=1)[0]
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
    ts._create_task(
        name="Prep deck",
        description="Slides.",
    )

    # Trigger-based task
    trig_tid = ts._create_task(
        name="Pickup support call",
        description="Answer phone when VIP calls.",
        trigger=Trigger(medium=Medium.PHONE_CALL),
    )["details"]["task_id"]

    with pytest.raises(ValueError):
        ts._update_task_queue(original=[0], new=[trig_tid, 0])
