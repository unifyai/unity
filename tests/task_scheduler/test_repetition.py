from datetime import datetime, timedelta, timezone

from tests.helpers import _handle_project
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.repetition import (
    Frequency,
    RepeatPattern,
    Weekday,
    next_repeated_start_at,
)
from unity.task_scheduler.types.schedule import Schedule
from unity.task_scheduler.types.status import Status


def test_next_repeated_start_at_skips_past_occurrences():
    pattern = RepeatPattern(frequency=Frequency.DAILY, interval=1)
    previous_start = datetime(2026, 4, 10, 9, 0, tzinfo=timezone.utc)
    now = datetime(2026, 4, 12, 10, 0, tzinfo=timezone.utc)

    next_start = next_repeated_start_at(
        previous_start=previous_start,
        patterns=[pattern],
        current_occurrence_index=0,
        now=now,
    )

    assert next_start == datetime(2026, 4, 13, 9, 0, tzinfo=timezone.utc)


def test_next_repeated_start_at_honors_weekdays_and_count():
    pattern = RepeatPattern(
        frequency=Frequency.WEEKLY,
        interval=1,
        weekdays=[Weekday.MO, Weekday.WE],
        count=2,
    )
    previous_start = datetime(2026, 4, 6, 9, 0, tzinfo=timezone.utc)  # Monday

    next_start = next_repeated_start_at(
        previous_start=previous_start,
        patterns=[pattern],
        current_occurrence_index=0,
        now=datetime(2026, 4, 6, 9, 1, tzinfo=timezone.utc),
    )
    assert next_start == datetime(2026, 4, 8, 9, 0, tzinfo=timezone.utc)

    exhausted = next_repeated_start_at(
        previous_start=previous_start,
        patterns=[pattern],
        current_occurrence_index=1,
        now=datetime(2026, 4, 8, 9, 1, tzinfo=timezone.utc),
    )
    assert exhausted is None


@_handle_project
def test_clone_task_instance_rearms_recurring_scheduled_task():
    scheduler = TaskScheduler()
    initial_start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(
        hours=1,
    )
    scheduler._create_task(
        name="Daily summary",
        description="Send the daily summary email.",
        status=Status.scheduled,
        schedule=Schedule(start_at=initial_start.isoformat()),
        repeat=[RepeatPattern(frequency=Frequency.DAILY)],
    )

    current = scheduler._get_task_or_raise(0)
    scheduler._clone_task_instance(current)

    task_rows = scheduler._filter_tasks(filter="task_id == 0")
    latest = max(task_rows, key=lambda task: task.instance_id)
    assert len(task_rows) == 2
    assert latest.instance_id == 1
    assert latest.status == Status.scheduled
    assert latest.schedule_start_at == initial_start + timedelta(days=1)


@_handle_project
def test_clone_task_instance_stops_when_repeat_count_is_exhausted():
    scheduler = TaskScheduler()
    initial_start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(
        hours=1,
    )
    scheduler._create_task(
        name="One repeat only",
        description="Run once and do not re-arm.",
        status=Status.scheduled,
        schedule=Schedule(start_at=initial_start.isoformat()),
        repeat=[RepeatPattern(frequency=Frequency.DAILY, count=1)],
    )

    current = scheduler._get_task_or_raise(0)
    scheduler._clone_task_instance(current)

    task_rows = scheduler._filter_tasks(filter="task_id == 0")
    assert len(task_rows) == 1
