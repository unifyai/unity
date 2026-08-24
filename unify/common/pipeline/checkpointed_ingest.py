"""Resumable table ingestion, shared by every tier that ingests.

This is the one place rows are written. In-process execution and the worker fleet
both call it, which is what makes the tier choice a question of latency rather
than of guarantees: the artifacts, the leases, the checkpoints and the completion
check are the same code, so a run interrupted anywhere resumes anywhere.

Three invariants hold it together, and each exists because of a specific way
ingestion goes wrong when it is interrupted:

**Checkpoint after every committed chunk.** Resume then re-does at most one
chunk. Without it a crash at 90% either restarts from zero or, worse, appends
what it already wrote.

**Hold a fenced lease per table.** Two attempts on one table both writing
progress will interleave their counts, and the survivor's checkpoint ends up
lower than the rows actually committed -- so the next resume skips too few and
duplicates. The lease makes at-most-one-writer true; the fence makes a
superseded writer *notice*.

**Verify against the declared count before reporting success.** A run whose
checkpoint never reached the count its source declared has silently lost rows.
Checking is what turns that from an invisible shortfall into a resume, and then
into a loud failure.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ._utils import utc_now_iso
from .artifact_store import (
    ArtifactStore,
    LeaseNotAcquired,
    StaleLeaseError,
)
from .instrumentation import PipelineInstrumentation
from .orchestration import ArtifactWorkItem, ArtifactWorkResult, ingest_artifacts
from .types import IngestCheckpoint, TableInputHandle
from .work_queue import CancellationCheck, PipelineCancelled, RetryWorkItem

logger = logging.getLogger(__name__)

# Column carried on every ingested row so a re-attempt upserts rather than
# appends. Private and reserved: it is identity for resume, not user data.
INGEST_KEY_COLUMN = "_unity_ingest_key"

# Marker in a surrender message. The per-chunk hook raises ``RetryWorkItem``,
# but the retry wrapper captures it into a result string rather than letting it
# propagate, so the sentinel is how the caller recognises "stop, do not fail"
# and re-raises it.
SURRENDER_SENTINEL = "surrendering in-flight chunk"


class DuplicateLiveAttempt(RuntimeError):
    """Another live attempt already owns this work.

    Not a failure: the usual cause is an ordinary duplicate delivery, and the
    correct response is to let the holder finish rather than to contend.
    """

    def __init__(self, message: str, *, stage: str = "ingest", lease: Any = None):
        super().__init__(message)
        self.stage = stage
        self.lease = lease


@dataclass(frozen=True)
class TableWork:
    """One table to ingest, with everything a resume needs to redo it.

    ``declared_rows`` is required rather than optional, and that is deliberate.
    It is the number the completion check holds the durable checkpoint against,
    so a table without one cannot be verified -- and an unverifiable ingest is
    exactly the silent-shortfall case this module exists to prevent.
    """

    table_id: str
    label: str
    context: str
    handle: TableInputHandle
    declared_rows: int
    # The source file this table was extracted from. Reporting only: the
    # engine never reads it, but a progress number is meaningless to a
    # caller who cannot tell which of fifteen files it belongs to.
    source_path: str = ""
    columns: List[str] = field(default_factory=list)
    chunk_size: int = 500
    # Rows are the wrong unit for a payload limit: the same count carries wildly
    # different bytes depending on how wide the table is, so one number is
    # either wasteful on a 6-column table or oversized on a 50-column one. When
    # a sample is available the size is derived from it instead; this stays as
    # the floor for the case where nothing has been measured yet.
    description: Optional[str] = None
    column_descriptions: Optional[Dict[str, str]] = None
    # Caller-declared row identity. When set, these columns are the unique keys
    # the insert upserts on, so a re-run of the same source updates rows in
    # place instead of appending a second copy -- the current-state-table case.
    # When absent, identity falls back to the reserved per-run resume key below,
    # and a re-run appends -- the time-series case.
    unique_keys: Optional[Dict[str, str]] = None
    # Explicit column schema. Merged under any column descriptions; omit to let
    # the backend infer from the rows.
    fields: Optional[Dict[str, Any]] = None
    embed_columns: Optional[List[str]] = None
    embed_strategy: str = "off"
    post_ingest: Any = None
    infer_untyped_fields: bool = False
    # Prefix for the reserved upsert key. Scopes identity to one job and table so
    # two jobs writing the same context cannot collide on a row key and overwrite
    # each other's rows while each reports success. Only consulted when the
    # caller declared no unique keys of its own.
    ingest_key_prefix: str = ""


@dataclass
class TableOutcome:
    """What happened to one table."""

    table_id: str
    label: str
    context: str
    rows_committed: int
    declared_rows: int
    success: bool
    already_complete: bool = False
    error: Optional[str] = None

    @property
    def is_short(self) -> bool:
        return self.rows_committed < self.declared_rows


@dataclass
class IngestOutcome:
    """What happened to the whole set."""

    tables: List[TableOutcome] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)

    @property
    def rows_committed(self) -> int:
        return sum(table.rows_committed for table in self.tables)

    @property
    def shortfalls(self) -> List[TableOutcome]:
        """Tables whose durable progress is behind what their source declared."""
        return [table for table in self.tables if table.is_short]

    @property
    def failed(self) -> List[TableOutcome]:
        return [table for table in self.tables if not table.success]


class IncompleteIngest(RuntimeError):
    """Committed progress is short of what the source declared.

    Raised instead of returning so a caller cannot finalise past it by ignoring
    a field. Carries the per-table detail because "which table, and by how many"
    is the whole question when deciding whether to resume or give up.
    """

    def __init__(self, shortfalls: List[TableOutcome]):
        detail = "; ".join(
            f"{table.table_id}: {table.rows_committed}/{table.declared_rows}"
            for table in shortfalls
        )
        super().__init__(f"ingest incomplete: {detail}")
        self.shortfalls = shortfalls
        self.detail = detail


class CheckpointedIngest:
    """Ingest table handles into contexts, resumably.

    One instance per attempt at one job. ``attempt_id`` identifies this attempt
    for the lifetime of the object: it is what the lease fences on and what a
    checkpoint records, so a later attempt can tell whose progress it is reading.
    """

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        job_id: str,
        attempt_id: Optional[str] = None,
        lease_ttl_seconds: int = 900,
        lease_steal_after_seconds: int = 30,
        max_workers: int = 8,
    ):
        self._store = artifact_store
        self._job_id = job_id
        self._attempt_id = attempt_id or uuid.uuid4().hex
        self._lease_ttl = lease_ttl_seconds
        self._lease_steal_after = lease_steal_after_seconds
        self._max_workers = max_workers
        # Leases this attempt holds, so an exception that skips per-table
        # cleanup does not strand one for a successor to wait out.
        self._held: Dict[str, Any] = {}

    @property
    def attempt_id(self) -> str:
        return self._attempt_id

    def run(
        self,
        work: List[TableWork],
        *,
        dm: Any,
        destination: Optional[str] = None,
        source_path: str = "",
        instrumentation: Optional[PipelineInstrumentation] = None,
        is_cancelled: Optional[CancellationCheck] = None,
        should_surrender: Optional[Callable[[], bool]] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        verify: bool = True,
        storage_client: Any = None,
        retry_config: Any = None,
    ) -> IngestOutcome:
        """Ingest every table in *work*, resuming each from its checkpoint.

        ``should_surrender`` is checked between chunks and is how a worker being
        shut down stops cleanly. Surrendering at a chunk boundary with the lease
        released is what lets a replacement resume immediately instead of waiting
        out a TTL, so a rollout costs one chunk rather than a whole table.

        ``verify`` runs the completion check and raises :class:`IncompleteIngest`
        on a shortfall. Leave it on unless the caller implements its own bounded
        resume around it -- turning it off is how a silent under-ingest gets out.
        """
        if not work:
            return IngestOutcome()

        # With no ledgers attached every instrumentation method is a no-op, so
        # a caller that does not want diagnostics need not supply one.
        instrumentation = instrumentation or PipelineInstrumentation(
            run_id=self._job_id,
        )
        outcomes: Dict[str, TableOutcome] = {}

        items = [
            self._work_item(
                entry,
                dm=dm,
                destination=destination,
                storage_client=storage_client,
            )
            for entry in work
        ]

        try:
            results = ingest_artifacts(
                work_items=items,
                ingest_fn=lambda item: self._ingest_one(
                    item,
                    is_cancelled=is_cancelled,
                    should_surrender=should_surrender,
                    on_progress=on_progress,
                ),
                instrumentation=instrumentation,
                source_path=source_path or self._job_id,
                max_workers=self._max_workers,
                retry_config=retry_config,
                is_cancelled=is_cancelled,
            )
        finally:
            self._release_all(reason="run finished")

        by_table = {entry.table_id: entry for entry in work}
        for result in results:
            entry = self._entry_for(result, by_table)
            if entry is None:
                continue
            outcomes[entry.table_id] = self._outcome_for(entry, result)

        # A table that never produced a result at all is short by its whole
        # count, not absent: reporting only what ran would understate the gap.
        for entry in work:
            outcomes.setdefault(
                entry.table_id,
                TableOutcome(
                    table_id=entry.table_id,
                    label=entry.label,
                    context=entry.context,
                    rows_committed=self._committed(entry.table_id),
                    declared_rows=entry.declared_rows,
                    success=False,
                    error="no result recorded",
                ),
            )

        outcome = IngestOutcome(
            tables=[outcomes[entry.table_id] for entry in work],
            contexts=list(dict.fromkeys(entry.context for entry in work)),
        )

        self._reraise_surrender(outcome)
        if verify and outcome.shortfalls:
            raise IncompleteIngest(outcome.shortfalls)
        return outcome

    # -- per-table execution --------------------------------------------------

    def _work_item(
        self,
        entry: TableWork,
        *,
        dm: Any,
        destination: Optional[str],
        storage_client: Any = None,
    ) -> ArtifactWorkItem:
        checkpoint = self._store.read_checkpoint(self._job_id, entry.table_id)
        skip_rows = checkpoint.rows_committed if checkpoint else 0
        initial_chunks = checkpoint.chunks_committed if checkpoint else 0
        if checkpoint:
            logger.info(
                "[ingest] Resuming table=%s from checkpoint: %d rows, %d chunks",
                entry.table_id,
                skip_rows,
                initial_chunks,
            )

        fields: Dict[str, Any] = dict(entry.fields or {})
        for name, description in (entry.column_descriptions or {}).items():
            fields.setdefault(name, {"description": description})
        if not entry.unique_keys:
            # The reserved key is declared and made unique so a re-attempt
            # upserts. Without it a resume that misjudges its offset appends
            # duplicates instead of overwriting them. Caller-declared keys make
            # it redundant: they identify the row whichever attempt writes it.
            fields.setdefault(INGEST_KEY_COLUMN, "str")

        return ArtifactWorkItem(
            kind="table",
            label=entry.label,
            stage_name="ingest_table",
            table_id=entry.table_id or None,
            columns=list(entry.columns or getattr(entry.handle, "columns", []) or []),
            row_count=entry.declared_rows,
            payload={
                "entry": entry,
                "dm": dm,
                "destination": destination,
                "fields": fields,
                "skip_rows": skip_rows,
                "initial_chunks": initial_chunks,
                "storage_client": storage_client,
            },
            meta={
                "row_count": entry.declared_rows,
                "table_label": entry.label,
                "context": entry.context,
                "source_handle_type": type(entry.handle).__name__,
            },
        )

    def _ingest_one(
        self,
        item: ArtifactWorkItem,
        *,
        is_cancelled: Optional[CancellationCheck],
        should_surrender: Optional[Callable[[], bool]],
        on_progress: Optional[Callable[[str, int, int], None]],
    ) -> Dict[str, Any]:
        payload = item.payload
        entry: TableWork = payload["entry"]

        # Duplicate delivery of a table that is already fully committed has no
        # work to do. Answering "done" acks it, where acquiring a lease would
        # churn on short retries for nothing. Gated on a known total, so a
        # *stalled* checkpoint still falls through to a real ingest and the
        # completion check remains the backstop.
        committed = self._committed(entry.table_id)
        if committed >= entry.declared_rows:
            logger.info(
                "[ingest] Table %s already complete (%d/%d); acking duplicate",
                entry.table_id,
                committed,
                entry.declared_rows,
            )
            return {"row_count": committed, "already_complete": True}

        lease = self._acquire(entry.table_id)
        try:
            return self._insert(
                entry,
                payload=payload,
                lease=lease,
                is_cancelled=is_cancelled,
                should_surrender=should_surrender,
                on_progress=on_progress,
            )
        finally:
            self._release(entry.table_id, reason="table finished")

    def _insert(
        self,
        entry: TableWork,
        *,
        payload: Dict[str, Any],
        lease: Any,
        is_cancelled: Optional[CancellationCheck],
        should_surrender: Optional[Callable[[], bool]],
        on_progress: Optional[Callable[[str, int, int], None]],
    ) -> Dict[str, Any]:
        state = {
            "rows": int(payload["skip_rows"]),
            "chunks": int(payload["initial_chunks"]),
        }

        def _on_chunk(task: Any, result: Any) -> None:
            if not str(getattr(task, "task_type", "")).startswith("insert_chunk"):
                return
            value = getattr(result, "value", None) or {}
            state["rows"] += (
                int(value.get("row_count", 0) or 0) if isinstance(value, dict) else 0
            )
            state["chunks"] += 1
            # Refreshed before the checkpoint so a long table cannot have its
            # lease expire underneath it and write progress it no longer owns.
            self._refresh(entry.table_id)
            self._store.write_checkpoint(
                self._job_id,
                entry.table_id,
                IngestCheckpoint(
                    job_id=self._job_id,
                    artifact_id=entry.table_id,
                    chunks_committed=state["chunks"],
                    rows_committed=state["rows"],
                    last_updated=utc_now_iso(),
                    attempt_id=self._attempt_id,
                    lease_generation=getattr(lease, "generation", None),
                ),
                attempt_id=self._attempt_id,
                lease_generation=getattr(lease, "generation", None),
            )
            if on_progress:
                on_progress(entry.table_id, state["rows"], entry.declared_rows)
            if is_cancelled and is_cancelled():
                raise PipelineCancelled(
                    f"Job {self._job_id} cancelled after {state['rows']} rows",
                )

        def _before_chunk(**_kwargs: Any) -> None:
            if is_cancelled and is_cancelled():
                raise PipelineCancelled(
                    f"Job {self._job_id} cancelled before next chunk",
                )
            if should_surrender and should_surrender():
                # Between chunks is the only safe place to stop: the checkpoint
                # is current, so a successor resumes having lost nothing.
                raise RetryWorkItem(
                    f"Job {self._job_id} {SURRENDER_SENTINEL}; "
                    "will resume from checkpoint",
                )

        dm = payload["dm"]
        result = dm.ingest(
            entry.context,
            None,
            table_input_handle=entry.handle,
            description=entry.description,
            fields=payload["fields"],
            unique_keys=entry.unique_keys or {INGEST_KEY_COLUMN: "str"},
            infer_untyped_fields=entry.infer_untyped_fields,
            # Sized from the rows actually present rather than a fixed count.
            # The same row count is a modest payload on a six-column table and
            # an unbounded one on a fifty-column table, which is why raising the
            # count alone was the wrong lever.
            chunk_size=_effective_chunk_size(entry),
            embed_columns=entry.embed_columns,
            embed_strategy=entry.embed_strategy,
            post_ingest=entry.post_ingest,
            destination=payload["destination"],
            skip_rows=payload["skip_rows"],
            # Passed so the insert path can refuse a total that disagrees with
            # what the source declared, rather than leaving the shortfall for the
            # completion check to find after the fact.
            expected_total_rows=entry.declared_rows or None,
            # The reserved resume key applies only when the caller declared no
            # identity of its own; with caller keys, a re-delivered chunk
            # already upserts on them.
            private_ingest_key_column=("" if entry.unique_keys else INGEST_KEY_COLUMN),
            private_ingest_key_prefix=(
                ""
                if entry.unique_keys
                else (entry.ingest_key_prefix or f"{self._job_id}:{entry.table_id}")
            ),
            storage_client=payload.get("storage_client"),
            on_task_complete=_on_chunk,
            before_insert_chunk=_before_chunk,
        )
        return {
            "row_count": state["rows"],
            "context": entry.context,
            "ingest_result": result,
        }

    # -- leases ---------------------------------------------------------------

    def _lease_key(self, table_id: str) -> str:
        return f"jobs/{self._job_id}/leases/{table_id or 'table'}"

    def _acquire(self, table_id: str) -> Any:
        key = self._lease_key(table_id)
        try:
            lease = self._store.acquire_lease(
                key,
                owner_id=f"ingest-{self._attempt_id}",
                attempt_id=self._attempt_id,
                stage="ingest",
                ttl_seconds=self._lease_ttl,
                steal_expired_after_seconds=self._lease_steal_after,
            )
        except LeaseNotAcquired as exc:
            # Surfaced as its own type rather than a generic failure: the caller
            # should let the holder finish, not retry into contention.
            raise DuplicateLiveAttempt(
                f"Table {table_id!r} is already being ingested by a live attempt",
                lease=exc.lease,
            ) from exc
        self._held[key] = lease
        return lease

    def _refresh(self, table_id: str) -> None:
        key = self._lease_key(table_id)
        lease = self._held.get(key)
        if lease is None:
            return
        self._held[key] = self._store.refresh_lease(
            key,
            owner_id=lease.owner_id,
            attempt_id=lease.attempt_id,
            generation=lease.generation,
            ttl_seconds=self._lease_ttl,
        )

    def _release(self, table_id: str, *, reason: str) -> None:
        key = self._lease_key(table_id)
        lease = self._held.pop(key, None)
        if lease is None:
            return
        try:
            self._store.release_lease(
                key,
                owner_id=lease.owner_id,
                attempt_id=lease.attempt_id,
                generation=lease.generation,
            )
        except StaleLeaseError:
            # Already taken over. Leaving it alone is correct: the successor
            # owns it, and releasing would hand the work to a third writer.
            logger.info("[ingest] Lease %s already taken over; not released", key)

    def _release_all(self, *, reason: str) -> None:
        for key in list(self._held):
            table_id = key.rsplit("/", 1)[-1]
            self._release(table_id, reason=reason)

    # -- reading back ---------------------------------------------------------

    def _committed(self, table_id: str) -> int:
        checkpoint = self._store.read_checkpoint(self._job_id, table_id)
        return int(checkpoint.rows_committed) if checkpoint else 0

    def _entry_for(
        self,
        result: ArtifactWorkResult,
        by_table: Dict[str, TableWork],
    ) -> Optional[TableWork]:
        for entry in by_table.values():
            if entry.label == result.label:
                return entry
        return None

    def _outcome_for(
        self,
        entry: TableWork,
        result: ArtifactWorkResult,
    ) -> TableOutcome:
        value = result.value if isinstance(result.value, dict) else {}
        return TableOutcome(
            table_id=entry.table_id,
            label=entry.label,
            context=entry.context,
            # Read from the checkpoint rather than from the result: the
            # checkpoint is what a resume will trust, so reporting anything else
            # would let the two disagree about where the table got to.
            rows_committed=self._committed(entry.table_id),
            declared_rows=entry.declared_rows,
            success=result.success,
            already_complete=bool(value.get("already_complete")),
            error=result.error,
        )

    def _reraise_surrender(self, outcome: IngestOutcome) -> None:
        """Turn a captured surrender back into the control-flow it was.

        The retry wrapper catches every exception into a result string, so a
        surrender arrives looking like an ordinary failure. Left that way it
        would fail the run instead of parking it for a clean resume.
        """
        for table in outcome.failed:
            if table.error and SURRENDER_SENTINEL in table.error:
                raise RetryWorkItem(table.error)


def incomplete_tables(
    work: List[TableWork],
    *,
    artifact_store: ArtifactStore,
    job_id: str,
) -> List[TableOutcome]:
    """Report which tables' durable progress is short of their declared count.

    Reads only checkpoints, so it answers "did this job actually land what it
    was given" without re-running anything. That makes it usable as a
    finalisation gate *and* as an out-of-band audit -- and because both callers
    measure the same way, an operator's answer and the worker's cannot disagree.
    """
    shorts: List[TableOutcome] = []
    for entry in work:
        checkpoint = artifact_store.read_checkpoint(job_id, entry.table_id)
        committed = int(getattr(checkpoint, "rows_committed", 0) or 0)
        if committed < entry.declared_rows:
            shorts.append(
                TableOutcome(
                    table_id=entry.table_id,
                    label=entry.label,
                    context=entry.context,
                    rows_committed=committed,
                    declared_rows=entry.declared_rows,
                    success=False,
                    error="committed fewer rows than the source declared",
                ),
            )
    return shorts


def wait_for_lease_release(
    store: ArtifactStore,
    key: str,
    *,
    timeout_s: float,
    poll_s: float = 1.0,
) -> bool:
    """Wait for a lease to disappear, returning whether it did.

    Used where two recovery paths could otherwise publish at once. Serialising
    on the lease is what stops a retry and a stale-recovery from both claiming
    one table, which is the failure that freezes a checkpoint and under-ingests
    without reporting anything.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            store.get_json(key)
        except Exception:
            return True
        time.sleep(poll_s)
    return False


def _effective_chunk_size(entry: "TableWork") -> int:
    """Chunk size for one table, scaled by how wide it is.

    Estimated from the column count rather than measured from rows: a handle is
    a stream, and reading it twice to size the writes that consume it would risk
    losing a row to save a request. Width is the part of payload size knowable
    without touching the data.

    It is an estimate and named as one -- a table of long free text and a table
    of integers with the same column count do not weigh the same. It is still a
    better unit than a fixed row count, which is a modest payload on six columns
    and an unbounded one on fifty.
    """
    columns = len(entry.columns or [])
    if columns <= 0:
        return entry.chunk_size
    projected_row_bytes = max(columns * EST_BYTES_PER_FIELD, 1)
    return max(
        CHUNK_ROWS_MIN,
        min(CHUNK_ROWS_MAX, int(CHUNK_TARGET_BYTES / projected_row_bytes)),
    )


# ── payload-sized chunking ───────────────────────────────────────────────────

# A write is bounded by the bytes it carries, not the rows. The backend caps a
# single call at 1000 rows, so that stays the ceiling; the floor keeps a very
# wide table from degenerating into one row per call.
CHUNK_ROWS_MAX = 1000
CHUNK_ROWS_MIN = 25
# Target serialised payload per call. Chosen to sit well inside a request the
# backend accepts comfortably rather than at its limit, because the sample is an
# estimate and later rows can be wider than the ones measured.
CHUNK_TARGET_BYTES = 2 * 1024 * 1024
# Rough mean serialised size of one field including its key and punctuation.
# Deliberately generous: over-estimating shrinks the request, which is the safe
# direction, while under-estimating grows it toward the limit the estimate
# exists to stay inside.
EST_BYTES_PER_FIELD = 120


def chunk_rows_for(
    sample: Sequence[dict],
    *,
    target_bytes: int = CHUNK_TARGET_BYTES,
) -> int:
    """Rows per commit for a table shaped like *sample*.

    Derived from measured row size so a 6-column table and a 50-column one both
    send a similar payload, instead of one number being wasteful for the first
    and oversized for the second. Raising the row count alone was rejected for
    exactly that reason: the same 10,000 rows is a modest request for a narrow
    table and an unbounded one for a wide one.

    Falls back to the declared default when there is nothing to measure, and
    clamps to the backend's per-call ceiling either way.
    """
    import json

    rows = [r for r in list(sample)[:50] if isinstance(r, dict)]
    if not rows:
        return 500
    measured = sum(len(json.dumps(r, default=str).encode()) for r in rows)
    mean = max(measured / len(rows), 1.0)
    return max(CHUNK_ROWS_MIN, min(CHUNK_ROWS_MAX, int(target_bytes / mean)))
