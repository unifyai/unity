"""
Schema describing how a task repeats over time. The model serializes to and
from JSON for storage and transport.
"""

from __future__ import annotations

import calendar
from enum import Enum
from typing import List, Optional
from datetime import datetime, time, timedelta
from pydantic import BaseModel, Field, field_validator


class Frequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class Weekday(str, Enum):
    MO = "MO"
    TU = "TU"
    WE = "WE"
    TH = "TH"
    FR = "FR"
    SA = "SA"
    SU = "SU"


class RepeatPattern(BaseModel):
    """
    A very small subset of RFC-5545 RRULE expressed as first-class fields:

    * **frequency** – base unit of recurrence.
    * **interval**  – "every *n* units"; defaults to 1.
    * **weekdays**  – which days of the week (only when `frequency=weekly`).
    * **count**     – stop after *count* occurrences.
    * **until**     – or stop at this date/time (ISO-8601).
    * **time_of_day** – local *clock* time at which each occurrence starts.

    Anything more elaborate can still be represented by creating multiple
    `RepeatPattern` instances for a single task.
    """

    frequency: Frequency = Field(..., description="Base unit of recurrence")
    interval: int = Field(
        default=1,
        ge=1,
        description="Number of frequency units between each repeat",
    )
    weekdays: Optional[List[Weekday]] = Field(
        default=None,
        description="Applicable only when frequency == weekly; " "ignored otherwise",
    )
    count: Optional[int] = Field(
        default=None,
        ge=1,
        description="Total number of occurrences before stopping",
    )
    until: Optional[datetime] = Field(
        default=None,
        description="Hard cut-off date/time after which no repeats occur",
    )
    time_of_day: Optional[time] = Field(
        default=None,
        description=(
            "Clock time at which the task should start on each occurrence "
            "(e.g. 09:00).  When omitted the time is inherited from the "
            "queue head or resolved dynamically by the scheduler."
        ),
    )

    @field_validator("weekdays")
    def _weekdays_only_for_weekly(cls, v, info):
        if v is not None and info.data.get("frequency") != Frequency.WEEKLY:
            raise ValueError("`weekdays` only makes sense with weekly frequency")
        return v

    @field_validator("time_of_day")
    def _time_without_date(cls, v):
        """
        Disallow accidental full datetimes – the field must be a *time* only.
        """
        if isinstance(v, datetime):
            raise ValueError(
                "`time_of_day` must be a `datetime.time`, not a full datetime",
            )
        return v


_WEEKDAY_TO_INDEX = {
    Weekday.MO: 0,
    Weekday.TU: 1,
    Weekday.WE: 2,
    Weekday.TH: 3,
    Weekday.FR: 4,
    Weekday.SA: 5,
    Weekday.SU: 6,
}


def next_repeated_start_at(
    *,
    previous_start: datetime,
    patterns: list[RepeatPattern] | None,
    current_occurrence_index: int = 0,
    now: datetime | None = None,
) -> datetime | None:
    """Return the earliest future occurrence across the supplied repeat patterns.

    The scheduler stores only the currently due queue-head timestamp on each task
    instance. After one occurrence fires, this helper advances the repeat rule
    until it finds the next start that is still in the future relative to *now*.

    Parameters
    ----------
    previous_start:
        The timestamp of the occurrence that just fired.
    patterns:
        Repeat rules attached to the task. `None` or an empty list disables
        re-arming.
    current_occurrence_index:
        Zero-based instance index for the occurrence that just fired. This is
        used to respect `RepeatPattern.count`.
    now:
        Optional wall-clock reference. Defaults to the current time in the same
        timezone as `previous_start`.
    """

    if not patterns:
        return None
    reference_now = now or datetime.now(previous_start.tzinfo)
    candidates = [
        candidate
        for candidate in (
            _next_pattern_occurrence(
                previous_start=previous_start,
                pattern=pattern,
                current_occurrence_index=current_occurrence_index,
                reference_now=reference_now,
            )
            for pattern in patterns
        )
        if candidate is not None
    ]
    if not candidates:
        return None
    return min(candidates)


def _next_pattern_occurrence(
    *,
    previous_start: datetime,
    pattern: RepeatPattern,
    current_occurrence_index: int,
    reference_now: datetime,
) -> datetime | None:
    """Advance one pattern until it yields a future occurrence or exhausts."""

    occurrence_count = current_occurrence_index + 1
    if pattern.count is not None and occurrence_count >= pattern.count:
        return None
    candidate = previous_start
    for _ in range(2048):
        candidate = _advance_one_occurrence(candidate, pattern)
        if pattern.until is not None and candidate > _normalize_until(
            pattern.until,
            candidate,
        ):
            return None
        if candidate > reference_now:
            return candidate
    raise ValueError("Could not compute the next repeated task occurrence.")


def _advance_one_occurrence(current: datetime, pattern: RepeatPattern) -> datetime:
    """Return the immediate next occurrence for one pattern."""

    if pattern.frequency == Frequency.DAILY:
        candidate = current + timedelta(days=pattern.interval)
        return _apply_time_of_day(candidate, pattern.time_of_day, current)
    if pattern.frequency == Frequency.WEEKLY:
        if not pattern.weekdays:
            candidate = current + timedelta(weeks=pattern.interval)
            return _apply_time_of_day(candidate, pattern.time_of_day, current)
        return _next_weekday_occurrence(current, pattern)
    if pattern.frequency == Frequency.MONTHLY:
        candidate = _add_months(current, pattern.interval)
        return _apply_time_of_day(candidate, pattern.time_of_day, current)
    if pattern.frequency == Frequency.YEARLY:
        candidate = _add_years(current, pattern.interval)
        return _apply_time_of_day(candidate, pattern.time_of_day, current)
    raise ValueError(f"Unsupported repeat frequency: {pattern.frequency}")


def _next_weekday_occurrence(current: datetime, pattern: RepeatPattern) -> datetime:
    """Return the next matching weekday occurrence inside the weekly cadence."""

    assert pattern.weekdays, "weekly weekday search requires explicit weekdays"
    allowed_weekdays = sorted(
        {_WEEKDAY_TO_INDEX[weekday] for weekday in pattern.weekdays},
    )
    base_week_start = current.date() - timedelta(days=current.weekday())
    search_date = current.date()
    while True:
        search_date += timedelta(days=1)
        weeks_since_base = (search_date - base_week_start).days // 7
        if weeks_since_base % pattern.interval != 0:
            continue
        if search_date.weekday() in allowed_weekdays:
            candidate = datetime.combine(search_date, current.timetz())
            if current.tzinfo is not None:
                candidate = candidate.replace(tzinfo=current.tzinfo)
            return _apply_time_of_day(candidate, pattern.time_of_day, current)


def _apply_time_of_day(
    candidate: datetime,
    override: time | None,
    reference: datetime,
) -> datetime:
    """Apply `time_of_day` when present, otherwise preserve the reference time."""

    target_time = override or reference.timetz().replace(tzinfo=None)
    updated = datetime.combine(candidate.date(), target_time)
    if candidate.tzinfo is not None:
        updated = updated.replace(tzinfo=candidate.tzinfo)
    return updated


def _add_months(value: datetime, months: int) -> datetime:
    """Advance a datetime by whole months, clamping the day when needed."""

    month_index = (value.month - 1) + months
    year = value.year + month_index // 12
    month = (month_index % 12) + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _add_years(value: datetime, years: int) -> datetime:
    """Advance a datetime by whole years, clamping leap-day overflow."""

    year = value.year + years
    day = min(value.day, calendar.monthrange(year, value.month)[1])
    return value.replace(year=year, day=day)


def _normalize_until(until: datetime, reference: datetime) -> datetime:
    """Compare `until` in the same timezone shape as the candidate."""

    if until.tzinfo is None and reference.tzinfo is not None:
        return until.replace(tzinfo=reference.tzinfo)
    if until.tzinfo is not None and reference.tzinfo is None:
        return until.replace(tzinfo=None)
    return until
