"""
Schema describing how a task repeats over time. The model serializes to and
from JSON for storage and transport.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from datetime import datetime, time
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
