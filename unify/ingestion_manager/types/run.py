"""What an ingestion reports back: one run, its progress, and what to do next.

Every ingestion is a **run** with an id, whichever tier executed it. That is the
point of the design: an inline run and a dispatched cloud run are asked about the
same way, so a plan that submits work does not have to know where it went in order
to find out how it ended.

Runs and their events are ordinary rows in contexts this manager owns, read through
the same logging path as every other catalogue. There is deliberately no separate
ledger machinery, no cost accounting and no bespoke progress files: a run's history
is rows, queryable like anything else. The data and files themselves still go to
object storage, which is what lets a parse worker and an ingest worker hand off
remotely -- but observing the run does not.

Nothing here holds bulk data. Row payloads are staged as artifacts and referenced
by key, so a run row stays small however large the ingestion: a single row
carrying an embedded payload is both a write the backend can reject outright and a
read that costs more than the answer is worth.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from unify.common.authorship import AuthoredRow

# Lifecycle of a run. `paused` is distinct from `cancelled`: a paused run keeps its
# queued work and can be resumed, a cancelled one cannot.
RunState = Literal[
    "queued",
    "running",
    "paused",
    "succeeded",
    "failed",
    "cancelled",
]

# Terminal states, past which polling learns nothing further.
TERMINAL_STATES: frozenset[str] = frozenset({"succeeded", "failed", "cancelled"})

# What a retry should re-attempt.
#
#   dlq    -- only the items that were parked after exhausting their retries.
#             The common case, and the safe one: nothing that already succeeded
#             is touched.
#   stale  -- items claimed by a worker that then stopped reporting. Recovers a
#             crashed worker's share without re-running healthy work.
#   all    -- every item, including ones that succeeded. Only when the target was
#             emptied or the parse changed.
RetryScope = Literal["dlq", "stale", "all"]


class TableReconciliation(BaseModel):
    """Whether what landed in a table matches what the source held.

    Row counts alone were not enough to catch the failure this exists for: a
    run reported 13,000 rows committed and every data column in every one of
    them held the string "None". The count was right and the table was useless,
    so a check that only counts agrees with a run that wrote nothing.

    Hence ``empty_columns``: a column that is blank in every sampled row is
    reported, because a table whose columns are uniformly blank did not ingest
    its data whatever its row count says.
    """

    model_config = ConfigDict(extra="forbid")

    context: str = Field(description="Table that was checked.")
    source_rows: Optional[int] = Field(
        default=None,
        description="Rows the source was measured to hold, when known.",
    )
    stored_rows: int = Field(default=0, description="Rows now in the table.")
    empty_columns: List[str] = Field(
        default_factory=list,
        description=(
            "Columns blank in every sampled row. Non-empty here means the rows "
            "arrived without their values, which a row count cannot detect."
        ),
    )
    sampled_rows: int = Field(
        default=0,
        description="Rows inspected to judge column population.",
    )

    @property
    def complete(self) -> bool:
        """Whether the table holds what the source did, with data in it."""
        if self.empty_columns:
            return False
        if self.source_rows is None:
            return self.stored_rows > 0
        return self.stored_rows >= self.source_rows

    @property
    def summary(self) -> str:
        """One line a person or an actor can act on."""
        if self.empty_columns:
            shown = ", ".join(self.empty_columns[:5])
            return (
                f"{self.context}: {self.stored_rows} row(s) stored but "
                f"{len(self.empty_columns)} column(s) are blank in every row "
                f"sampled ({shown}). The rows arrived without their values, so "
                "the count is not evidence of a successful ingest."
            )
        if self.source_rows is None:
            return f"{self.context}: {self.stored_rows} row(s) stored."
        if self.stored_rows >= self.source_rows:
            return (
                f"{self.context}: complete -- {self.stored_rows} of "
                f"{self.source_rows} row(s)."
            )
        missing = self.source_rows - self.stored_rows
        return (
            f"{self.context}: {self.stored_rows} of {self.source_rows} row(s), "
            f"{missing} missing."
        )


class AttemptState(BaseModel):
    """Whether an attempt still holds a file, and whether it can be taken over.

    A stalled run and a slow one look identical from progress alone: both sit at
    the same row count with a terminal-looking silence. The difference is whether
    the lease behind the work is still being renewed, which is the only fact that
    distinguishes "working, be patient" from "dead, take it over".

    Deliberately carries no worker or pod identity. ``attempt_id`` is an opaque
    handle the engine generates; naming the machine holding a lease would widen
    what a prompt injection can reach for no diagnostic gain, since the
    actionable question is only whether takeover is safe.
    """

    model_config = ConfigDict(extra="forbid")

    attempt_id: str = Field(description="Opaque handle for the holding attempt.")
    heartbeat_age_s: Optional[float] = Field(
        default=None,
        description=(
            "Seconds since the holder last renewed. Growing past the lease TTL "
            "is what makes an attempt recoverable."
        ),
    )
    expired: bool = Field(
        default=False,
        description=(
            "Whether the lease has lapsed. True means no live writer owns this "
            "work and a retry can take it over without contending."
        ),
    )
    takeover_count: int = Field(
        default=0,
        description=(
            "Times this work has already changed hands. Repeated takeovers mean "
            "attempts are dying rather than the work being slow."
        ),
    )


class FileProgress(BaseModel):
    """How one file within a run is doing.

    A run's stage counters say two of fifteen files have parsed; they do not say
    *which* two, how far either has got, or where a stuck one stopped. Splitting
    a batch into one run per file was the only way to recover that, which traded
    away the single handle that made the batch observable at all.

    Built from the checkpoints the dispatched path already writes, so this
    surfaces what was measured rather than measuring anything new.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Source file this describes.")
    observed: bool = Field(
        default=True,
        description=(
            "Whether anything actually measured this file. False means only the "
            "path is known and every other field is unset -- the case for a run "
            "executing on the worker fleet, which does not report per file. "
            "False is not 'queued': absence of a measurement is not evidence "
            "that no work happened."
        ),
    )
    state: Optional[RunState] = Field(
        default=None,
        description="Where this file has got to, or null when not observed.",
    )
    claimed: Optional[bool] = Field(
        default=None,
        description=(
            "Whether a worker has taken this file up, or null when not "
            "observed. A file that is queued and unclaimed is waiting for "
            "capacity; one that is queued and claimed is working and has "
            "simply not committed yet. Only the first is a reason to look for "
            "a cause -- and only when it was actually observed."
        ),
    )
    rows_written: int = Field(default=0, description="Rows committed so far.")
    declared_rows: Optional[int] = Field(
        default=None,
        description="Rows the source was measured to hold, when known.",
    )
    context: Optional[str] = Field(
        default=None,
        description="Destination this file's rows are landing in.",
    )
    parked: int = Field(
        default=0,
        description="Records set aside after exhausting retries.",
    )
    error: Optional[str] = None
    attempts: List[AttemptState] = Field(
        default_factory=list,
        description=(
            "Leases currently recorded against this file's tables. Empty means "
            "nothing holds it. An entry with expired=True is recoverable; one "
            "without is being actively worked and must not be disturbed."
        ),
    )

    @property
    def recoverable(self) -> bool:
        """Whether a retry could take this file over without contending.

        True only when something holds it and everything holding it has lapsed.
        A file nothing holds is not 'recoverable' -- there is nothing to
        recover, and reporting it as such invites a retry that fixes nothing.
        """
        return bool(self.attempts) and all(a.expired for a in self.attempts)

    @property
    def fraction(self) -> Optional[float]:
        """Committed over declared, or ``None`` when the total is not yet known."""
        if not self.declared_rows:
            return None
        return min(self.rows_written / self.declared_rows, 1.0)


class StageProgress(BaseModel):
    """How one stage of a run is doing.

    Stages differ by source -- a folder of spreadsheets parses before it ingests,
    a list of rows only ingests -- so this is a list rather than a fixed shape.
    """

    model_config = ConfigDict(extra="forbid")

    stage: str = Field(description="Stage name, e.g. 'parse', 'ingest', 'embed'.")
    state: RunState
    done: int = Field(default=0, description="Items finished.")
    total: Optional[int] = Field(
        default=None,
        description="Items expected, when knowable up front.",
    )
    error: Optional[str] = None


class LogEntry(BaseModel):
    """One recorded event from a run."""

    model_config = ConfigDict(extra="forbid")

    at: str = Field(description="ISO timestamp.")
    stage: Optional[str] = None
    level: Literal["info", "warning", "error"] = "info"
    message: str


class IngestionEventRow(AuthoredRow):
    """Row stored in the ``Ingestion/Events`` context.

    One append-only line per thing that happened. Stage progress is folded from
    these rather than kept as a separate counter, so there is one source of truth: a
    counter maintained alongside events drifts the moment a worker dies between
    incrementing it and recording why.
    """

    model_config = ConfigDict(extra="forbid")

    run_key: str
    at: str
    stage: Optional[str] = None
    level: Literal["info", "warning", "error"] = "info"
    message: str = ""
    # Progress carried on the event so a reader can reconstruct where a stage got
    # to without replaying the work.
    state: Optional[str] = None
    done: Optional[int] = None
    total: Optional[int] = None
    # Per-file identity and measurement. Without these an event records that a
    # run committed rows and never which file they came from -- which is what
    # left a fifteen-file batch reporting one aggregate and nothing else.
    source_path: Optional[str] = None
    context: Optional[str] = None
    rows_written: Optional[int] = None
    declared_rows: Optional[int] = None


class IngestionRunRecord(AuthoredRow):
    """Row stored in the ``Ingestion/Runs`` context.

    ``run_id`` is deliberately absent: the backend auto-counts it, and including
    it here would let a caller try to set it.
    """

    model_config = ConfigDict(extra="forbid")

    # Stable handle the actor holds and polls with. Distinct from the auto-counted
    # row id so it can be minted before the row exists and survive a re-run.
    run_key: str

    state: RunState = "queued"
    # Which tier ran it. Recorded because "why did this take an hour" is usually
    # answered by it, and because a resume has to know where to look.
    executed_as: Optional[Literal["inline", "dispatched"]] = None

    source_kind: str = ""
    target_kind: str = ""

    # Artifact key of the staged request, so a retry or resume can rebuild the
    # work without the caller reconstructing it. A *key*, never the request
    # itself: a rows source can be arbitrarily large, and embedding it here would
    # put bulk data in a log row.
    request_key: str = ""
    # Exact size once measured -- the count the tier decision was made on, and
    # what the completion check holds the durable checkpoint against.
    declared_rows: Optional[int] = None

    # Concrete context paths this run wrote. The reason a canvas can be built over
    # ingested files without anyone hardcoding the storage layout: the run says
    # where its output went.
    contexts: List[str] = Field(default_factory=list)
    rows_written: int = 0
    files_processed: int = 0

    error: Optional[str] = None
    # Items parked after exhausting retries -- the depth `retry(only="dlq")` clears.
    parked: int = 0

    # Set when the dispatched tier ran it, for correlating with cloud-side records.
    dispatch_id: Optional[str] = None

    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class IngestionRun(BaseModel):
    """Handle returned by ``submit``, before the work has necessarily finished.

    ``submit`` does not block, so this is what a plan carries forward. Poll it with
    ``get_status``, or hand it to ``wait`` when the plan genuinely cannot continue
    without the data.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    state: RunState
    executed_as: Optional[Literal["inline", "dispatched"]] = None
    # Populated once known; empty while a run is still queued.
    contexts: List[str] = Field(default_factory=list)


class RunStatus(BaseModel):
    """Everything worth knowing about a run, and what to do about it.

    ``next_step`` exists because a status a caller has to interpret will eventually
    be interpreted wrongly. It states the one action that makes sense now -- keep
    waiting, retry the parked items, fix the target, or nothing -- so recovery does
    not depend on the reader reconstructing the rule from the other fields.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    state: RunState
    executed_as: Optional[Literal["inline", "dispatched"]] = None

    stages: List[StageProgress] = Field(default_factory=list)
    files: List[FileProgress] = Field(
        default_factory=list,
        description=(
            "One entry per source file, so a batch stays one run without losing "
            "the per-file detail that would otherwise require one run each."
        ),
    )
    contexts: List[str] = Field(default_factory=list)
    rows_written: int = 0
    files_processed: int = 0
    parked: int = 0
    error: Optional[str] = None

    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    next_step: str = Field(
        description=(
            "The single action that makes sense now, in plain words -- including "
            "'nothing, this finished' when that is the answer."
        ),
    )

    @property
    def is_terminal(self) -> bool:
        """Whether polling again could tell you anything new."""
        return self.state in TERMINAL_STATES


class RetryResult(BaseModel):
    """Outcome of asking for a retry."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    scope: str
    # Zero is a meaningful answer, not a failure: it means there was nothing in
    # that scope to retry.
    requeued: int = 0
    state: RunState = "queued"
    detail: Optional[str] = None
    files: List[str] = Field(
        default_factory=list,
        description=(
            "Files this retry was aimed at. Empty means the whole run. Echoed "
            "back because a retry that silently matched nothing is "
            "indistinguishable from one that worked."
        ),
    )


class IngestionSummary(BaseModel):
    """One run as it appears in a listing."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    state: RunState
    source_kind: str = ""
    target_kind: str = ""
    contexts: List[str] = Field(default_factory=list)
    rows_written: int = 0
    parked: int = 0
    created_at: Optional[str] = None
    finished_at: Optional[str] = None
