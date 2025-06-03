"""
A strongly-typed, validation-first schema for describing how a task
repeats over time.  The model serialises cleanly to / from JSON so it can
be stored in `unify` logs without ambiguity.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional
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

    Anything more elaborate can still be represented by creating multiple
    `RepeatPattern` instances for a single task.
    """

    frequency: Frequency = Field(..., description="Base unit of recurrence")
    interval: int = Field(
        1,
        ge=1,
        description="Number of frequency units between each repeat",
    )
    weekdays: Optional[List[Weekday]] = Field(
        None,
        description="Applicable only when frequency == weekly; " "ignored otherwise",
    )
    count: Optional[int] = Field(
        None,
        ge=1,
        description="Total number of occurrences before stopping",
    )
    until: Optional[datetime] = Field(
        None,
        description="Hard cut-off date/time after which no repeats occur",
    )

    # ------------------------------------------------------------------ #
    #  Validators                                                         #
    # ------------------------------------------------------------------ #

    @field_validator("weekdays")
    def _weekdays_only_for_weekly(cls, v, info):
        if v is not None and info.data.get("frequency") != Frequency.WEEKLY:
            raise ValueError("`weekdays` only makes sense with weekly frequency")
        return v
