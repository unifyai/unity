from datetime import datetime, timedelta, timezone

import pytest

from tests.helpers import _handle_project
from unity.actor.simulated import SimulatedActor
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.repetition import (
    Frequency,
    RepeatPattern,
    Weekday,
    normalize_repeat_patterns,
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


def test_next_repeated_start_at_supports_minutely_cadence():
    pattern = RepeatPattern(frequency=Frequency.MINUTELY, interval=30)
    previous_start = datetime(2026, 5, 19, 9, 30, tzinfo=timezone.utc)

    next_start = next_repeated_start_at(
        previous_start=previous_start,
        patterns=[pattern],
        current_occurrence_index=0,
        now=datetime(2026, 5, 19, 9, 31, tzinfo=timezone.utc),
    )

    assert next_start == datetime(2026, 5, 19, 10, 0, tzinfo=timezone.utc)


def test_next_repeated_start_at_skips_missed_subdaily_occurrences():
    pattern = RepeatPattern(frequency=Frequency.MINUTELY, interval=30)
    previous_start = datetime(2026, 5, 19, 9, 30, tzinfo=timezone.utc)

    next_start = next_repeated_start_at(
        previous_start=previous_start,
        patterns=[pattern],
        current_occurrence_index=0,
        now=datetime(2026, 5, 19, 10, 7, tzinfo=timezone.utc),
    )

    assert next_start == datetime(2026, 5, 19, 10, 30, tzinfo=timezone.utc)


def test_next_repeated_start_at_honors_subdaily_count_and_until():
    previous_start = datetime(2026, 5, 19, 9, 30, tzinfo=timezone.utc)

    exhausted_by_count = next_repeated_start_at(
        previous_start=previous_start,
        patterns=[RepeatPattern(frequency=Frequency.MINUTELY, interval=30, count=1)],
        current_occurrence_index=0,
        now=datetime(2026, 5, 19, 9, 31, tzinfo=timezone.utc),
    )
    assert exhausted_by_count is None

    exhausted_by_until = next_repeated_start_at(
        previous_start=previous_start,
        patterns=[
            RepeatPattern(
                frequency=Frequency.MINUTELY,
                interval=30,
                until=datetime(2026, 5, 19, 9, 45, tzinfo=timezone.utc),
            ),
        ],
        current_occurrence_index=0,
        now=datetime(2026, 5, 19, 9, 31, tzinfo=timezone.utc),
    )
    assert exhausted_by_until is None


def test_next_repeated_start_at_supports_same_day_daily_time_slots():
    patterns = [
        RepeatPattern(frequency=Frequency.DAILY, interval=1, time_of_day="09:30"),
        RepeatPattern(frequency=Frequency.DAILY, interval=1, time_of_day="10:00"),
        RepeatPattern(frequency=Frequency.DAILY, interval=1, time_of_day="10:30"),
    ]
    previous_start = datetime(2026, 5, 19, 9, 30, tzinfo=timezone.utc)

    next_start = next_repeated_start_at(
        previous_start=previous_start,
        patterns=patterns,
        current_occurrence_index=0,
        now=datetime(2026, 5, 19, 9, 31, tzinfo=timezone.utc),
    )

    assert next_start == datetime(2026, 5, 19, 10, 0, tzinfo=timezone.utc)


def test_normalize_repeat_patterns_collapses_full_day_half_hour_slots():
    patterns = [
        RepeatPattern(
            frequency=Frequency.DAILY,
            interval=1,
            time_of_day=f"{hour:02d}:{minute:02d}:00",
        )
        for hour in range(24)
        for minute in (0, 30)
    ]

    normalized = normalize_repeat_patterns(patterns)

    assert normalized == [RepeatPattern(frequency=Frequency.MINUTELY, interval=30)]


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
    assert latest.entrypoint is None


@_handle_project
def test_entrypoint_review_patches_future_description_driven_instances():
    scheduler = TaskScheduler()
    initial_start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(
        hours=1,
    )
    scheduler._create_task(
        name="Daily description-driven summary",
        description="Summarize updates every day.",
        status=Status.scheduled,
        schedule=Schedule(start_at=initial_start.isoformat()),
        repeat=[RepeatPattern(frequency=Frequency.DAILY)],
    )

    current = scheduler._get_task_or_raise(0)
    scheduler._clone_task_instance(current)
    result = scheduler._attach_entrypoint_to_future_instances(
        task_id=0,
        completed_instance_id=0,
        function_id=321,
        rationale="The successful run revealed a stable workflow.",
    )

    rows = scheduler._filter_tasks(filter="task_id == 0")
    current_row = min(rows, key=lambda task: task.instance_id)
    future_row = max(rows, key=lambda task: task.instance_id)

    assert result["outcome"] == "attached"
    assert current_row.entrypoint is None
    assert future_row.entrypoint == 321
    assert future_row.offline is True


@pytest.mark.asyncio
@_handle_project
async def test_recurring_execution_clones_before_entrypoint_review_patch():
    scheduler = TaskScheduler(actor=SimulatedActor(steps=0))
    initial_start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(
        hours=1,
    )
    scheduler._create_task(
        name="Daily report",
        description="Run the daily report from the task description.",
        status=Status.scheduled,
        schedule=Schedule(start_at=initial_start.isoformat()),
        repeat=[RepeatPattern(frequency=Frequency.DAILY)],
    )

    handle = await scheduler.execute(task_id=0)
    await handle.result()

    rows_after_run = sorted(
        scheduler._filter_tasks(filter="task_id == 0"),
        key=lambda task: task.instance_id,
    )
    assert [row.instance_id for row in rows_after_run] == [0, 1]
    assert rows_after_run[0].entrypoint is None
    assert rows_after_run[1].entrypoint is None

    result = scheduler._attach_entrypoint_to_future_instances(
        task_id=0,
        completed_instance_id=0,
        function_id=321,
        rationale="The completed run was stable enough to reuse.",
    )
    assert result["outcome"] == "attached"

    patched_next = [
        row
        for row in scheduler._filter_tasks(filter="task_id == 0")
        if row.instance_id == 1
    ][0]
    assert patched_next.entrypoint == 321
    assert patched_next.offline is True

    scheduler._clone_task_instance(patched_next)
    cloned_from_patched = [
        row
        for row in scheduler._filter_tasks(filter="task_id == 0")
        if row.instance_id == 2
    ][0]
    assert cloned_from_patched.entrypoint == 321
    assert cloned_from_patched.offline is True


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
