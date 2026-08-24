"""IngestionManager over Unify contexts, the shared ingest core, and the fleet.

Storage mirrors the other catalogue managers: runs and their events are rows in
contexts this manager declares, and all row I/O goes through DataManager so
destination routing and retry behaviour are inherited rather than reimplemented.

Two things here are load-bearing.

**A run is recorded before any work starts.** A failure is then always something
with an id that can be inspected and resumed, rather than an exception that went
past. It is also why ``submit`` returns immediately: the record is the handle, and
the work follows it.

**Both tiers run the same code.** Rows are written by
:class:`~unify.common.pipeline.checkpointed_ingest.CheckpointedIngest` whether the
work runs in this process or on the fleet, so leases, checkpoints and the
completion check are not reimplemented per tier and cannot drift between them.
The difference is only which artifact store and queue are bound -- and because
both write the same layout, a run interrupted in process is adoptable by the
fleet exactly as one interrupted on a worker is.

Neither parser nor inserter is reimplemented. Files go through the existing parse
pipeline with its per-format backends; rows go through DataManager's chunked
insert. This manager decides *where* work runs, stages what it needs, and records
what happened.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import re
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from unify.common.context_names import assert_all_valid
from unify.common.context_registry import ContextRegistry, TableContext
from unify.common.model_to_fields import model_to_fields
from unify.common.pipeline import (
    CheckpointedIngest,
    DuplicateLiveAttempt,
    IncompleteIngest,
    InlineRowsHandle,
    LocalArtifactStore,
    TableWork,
)
from unify.common.pipeline.work_queue import PipelineCancelled, RetryWorkItem
from unify.data_manager.types.ingest import PostIngestConfig
from unify.ingestion_manager.base import BaseIngestionManager
from unify.ingestion_manager.policy import choose_tier, next_step, stages_from_events
from unify.ingestion_manager.settings import IngestionSettings
from unify.ingestion_manager.types.request import (
    EmbedSpec,
    IngestionRequest,
    IngestionSource,
    IngestionTarget,
)
from unify.ingestion_manager.types.run import (
    AttemptState,
    FileProgress,
    TableReconciliation,
    TERMINAL_STATES,
    IngestionEventRow,
    IngestionRun,
    IngestionRunRecord,
    IngestionSummary,
    LogEntry,
    RetryResult,
    RetryScope,
    RunState,
    RunStatus,
)

logger = logging.getLogger(__name__)

# Rows inspected per table when judging whether a column carries data. Enough
# for a column that is populated to show it, small enough to cost one read.
_RECONCILE_SAMPLE = 25
# Bookkeeping columns that are legitimately absent from source data, so their
# emptiness says nothing about whether the ingest worked.
_RECONCILE_IGNORED = frozenset({"row_id", "authoring_assistant_id"})


def _is_blank(value: Any) -> bool:
    """Whether a stored value carries nothing.

    The string ``"None"`` counts. A row whose columns hold the four characters
    of Python's ``str(None)`` is as empty as one holding nulls, and treating it
    as populated is what let a table of 13,000 valueless rows read as complete.
    """
    if value is None:
        return True
    text = str(value).strip()
    return text in ("", "-", "None", "null", "NaN")


RUNS_TABLE = "Ingestion/Runs"
EVENTS_TABLE = "Ingestion/Events"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_key() -> str:
    """Opaque, unguessable handle minted before the row exists.

    Separate from the auto-counted row id so work can be recorded against it
    immediately, without first reading back what id the backend assigned.
    """
    return secrets.token_urlsafe(9)


# Identifiers a caller may look a run up by: a token_urlsafe run key or a
# numeric row id. Anything else cannot match a run, and rejecting it here keeps
# the value out of the filter expression it would otherwise be spliced into.
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class _StaleSummary:
    """Live versus lapsed attempts, for deciding whether takeover is safe."""

    live: int
    lapsed: int

    @property
    def recoverable(self) -> bool:
        """Something has lapsed and nothing is still renewing.

        Both halves matter: taking over while a writer is live contends for the
        lease, and 'taking over' when nothing holds anything fixes nothing while
        reporting that it did.
        """
        return self.lapsed > 0 and self.live == 0


def _age_seconds(stamp: Any) -> Optional[float]:
    """Seconds since *stamp*, or ``None`` when it cannot be read.

    An unreadable timestamp must not read as "just renewed": that would present
    a dead attempt as healthy, which is the direction that strands work.
    """
    moment = _parse_moment(stamp)
    if moment is None:
        return None
    return max((_utcnow() - moment).total_seconds(), 0.0)


def _is_past(stamp: Any) -> bool:
    """Whether *stamp* has passed. An unreadable expiry counts as passed.

    A lease whose expiry cannot be parsed cannot be honoured, and treating it as
    live would strand the work permanently; expired is the recoverable reading.
    """
    moment = _parse_moment(stamp)
    if moment is None:
        return True
    return _utcnow() >= moment


def _parse_moment(stamp: Any) -> Optional[datetime]:
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(str(stamp))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_id(value: str) -> str:
    """A path reduced to a stable identifier fragment for checkpoint keys."""
    return "".join(
        char if char.isalnum() or char in ("-", "_") else "_" for char in value
    )[-48:]


def _handle_can_yield(handle: Any) -> bool:
    """Whether a transport handle can actually produce the rows it claims.

    Every handle carries a ``row_count``, but only the source-referencing ones
    can stream it: ``build_table_handles`` falls back to an empty
    ``InlineRowsHandle`` when a format defers its rows and is neither CSV nor
    XLSX, so a count without inline rows and without a source to read is a
    promise the transport cannot keep.
    """
    if isinstance(handle, InlineRowsHandle):
        return bool(handle.rows)
    return True


class IngestionManager(BaseIngestionManager):
    """Ingestion over Unify contexts, the shared ingest core and the fleet."""

    class Config:
        """Context registration for the Ingestion namespace."""

        required_contexts = [
            TableContext(
                name=RUNS_TABLE,
                description=(
                    "One row per ingestion. Records what was asked for, where it "
                    "ran, the contexts it wrote and how it ended, so a failure is "
                    "recoverable rather than merely reported."
                ),
                fields=model_to_fields(IngestionRunRecord),
                unique_keys={"run_id": "int"},
                auto_counting={"run_id": None},
            ),
            TableContext(
                name=EVENTS_TABLE,
                description=(
                    "Append-only log of what happened during each run. Stage "
                    "progress is folded from these rows rather than counted "
                    "separately, so there is one source of truth."
                ),
                fields=model_to_fields(IngestionEventRow),
                unique_keys={"event_id": "int"},
                auto_counting={"event_id": None},
            ),
        ]

    def __init__(self) -> None:
        super().__init__()
        self._settings = IngestionSettings()
        self._pool = ThreadPoolExecutor(
            max_workers=self._settings.INLINE_WORKERS,
            thread_name_prefix="ingestion",
        )
        self._lock = threading.RLock()
        self._store = self._build_artifact_store()
        # Control flags for runs executing in this process, keyed by run key.
        # In-process is the one tier where cancel and pause can act between
        # chunks immediately rather than through a queue -- and an inline run
        # dies with the process anyway, so nothing durable is needed to steer it.
        self._inline_control: Dict[str, Dict[str, bool]] = {}
        # Run keys whose two-phase dispatch (stage, upload, publish) is in
        # flight in this process. Uploads only ever run in the submitting
        # process, so a queued dispatched-tier row with no dispatch_id and no
        # entry here is not "still uploading" -- its submitter died before
        # publish, and no worker will ever pick it up. Membership is what lets
        # a status read make that call deterministically, without timers.
        self._dispatching: set[str] = set()
        # Cached answer to "is the fleet actually reachable", resolved on first
        # use rather than at construction so a manager can be built in a process
        # that never ingests without paying for a probe.
        self._fleet_probe: Optional[bool] = None
        logger.debug("IngestionManager initialized")

    # ── plumbing ──────────────────────────────────────────────────────────

    def _build_artifact_store(self) -> Any:
        """Bind the artifact store this deployment ingests through.

        The hosted deployment overrides this with its object-store adapter. The
        local one is a full implementation of the same port, not a stub: it
        fences and checkpoints, so a self-host run is as resumable as a hosted
        one and the two are not different code paths.
        """
        from unify.session_details import SESSION_DETAILS

        local_dir = getattr(SESSION_DETAILS, "local_dir", None)
        # Never fall back to the working directory: a CWD-relative store
        # scatters staged requests and checkpoints wherever the process was
        # launched from -- a repo checkout, a pod entrypoint's directory --
        # and a resume from a different CWD then finds nothing.
        root = (
            Path(local_dir) / "Ingestion"
            if local_dir
            else Path.home() / ".unity" / "ingestion"
        )
        return LocalArtifactStore(root_dir=root)

    def _get_dm(self):
        # Resolved per call rather than held, to avoid an import cycle at
        # construction time. Matches the other catalogue managers.
        from unify.manager_registry import ManagerRegistry

        return ManagerRegistry.get_data_manager()

    def _get_fm(self):
        from unify.manager_registry import ManagerRegistry

        return ManagerRegistry.get_file_manager()

    def _destinations_for(self, request: IngestionRequest) -> List[str]:
        """Every context path this request will try to create.

        A table target names one path outright. A collection derives one per
        file, so the names are only knowable here -- which is exactly why they
        have to be checked here rather than trusted downstream.
        """
        names: List[str] = []
        target = request.target
        if getattr(target, "kind", "") == "table":
            names.append(str(target.context))
        elif getattr(target, "kind", "") == "collection" and target.name:
            names.append(str(target.name))
        # The run's own bookkeeping contexts are created by the same rule, and a
        # bad ``destination`` would fail there first with a less obvious message.
        names.append(self._write_table(RUNS_TABLE, request.destination))
        names.append(self._write_table(EVENTS_TABLE, request.destination))
        return names

    def _write_table(self, table: str, destination: Optional[str]) -> str:
        root = ContextRegistry.write_root(self, table, destination=destination)
        return f"{root.strip('/')}/{table}"

    def _read_tables(self, table: str) -> List[str]:
        return [
            f"{root.strip('/')}/{table}"
            for root in ContextRegistry.read_roots(self, table)
        ]

    def _find_run(self, run_id: str) -> tuple[Optional[Dict[str, Any]], str]:
        """Locate a run by its id or its key, and the context holding it."""
        if not _RUN_ID_PATTERN.fullmatch(str(run_id)):
            # Not a shape any run identifier can take. Answering "not found"
            # also keeps the value out of the filter expressions below.
            return None, ""
        dm = self._get_dm()
        for context in self._read_tables(RUNS_TABLE):
            # Accepts either identifier: the actor holds whichever `submit` handed
            # back, and making it guess which one this expects would be a trap.
            for expression in (f"run_key == '{run_id}'", f"run_id == {run_id}"):
                if expression.startswith("run_id ==") and not str(run_id).isdigit():
                    continue
                rows = dm.filter(context, filter=expression, limit=1)
                if rows:
                    return rows[0], context
        return None, ""

    def _record_event(
        self,
        run_key: str,
        *,
        destination: Optional[str],
        stage: Optional[str] = None,
        level: str = "info",
        message: str = "",
        state: Optional[str] = None,
        done: Optional[int] = None,
        total: Optional[int] = None,
        source_path: Optional[str] = None,
        context: Optional[str] = None,
        rows_written: Optional[int] = None,
        declared_rows: Optional[int] = None,
    ) -> None:
        row = IngestionEventRow(
            run_key=run_key,
            at=_now(),
            stage=stage,
            level=level,  # type: ignore[arg-type]
            message=message,
            state=state,
            done=done,
            total=total,
            source_path=source_path,
            context=context,
            rows_written=rows_written,
            declared_rows=declared_rows,
        )
        self._get_dm().insert_rows(
            self._write_table(EVENTS_TABLE, destination),
            [row.model_dump(exclude_none=True)],
        )

    def _update_run(
        self,
        run_key: str,
        context: str,
        updates: Dict[str, Any],
    ) -> None:
        self._get_dm().update_rows(context, updates, filter=f"run_key == '{run_key}'")

    # ── submitting ────────────────────────────────────────────────────────

    @functools.wraps(BaseIngestionManager.submit, updated=())
    def submit(
        self,
        source: IngestionSource,
        target: IngestionTarget,
        *,
        embed: Optional[EmbedSpec] = None,
        post_ingest: Optional[PostIngestConfig] = None,
        destination: Optional[str] = None,
    ) -> IngestionRun:
        # Validation happens by constructing the request: the impossible
        # source/target pairing and every unsafe name are rejected here, before a
        # row exists, so a refused request leaves nothing behind.
        request = IngestionRequest(
            source=source,
            target=target,
            embed=embed,
            post_ingest=post_ingest,
            destination=destination,
        )

        # Every destination this run will create, checked against the backend's
        # own naming rule before anything is published. The backend reports a
        # violation by naming the rule and not the value, from a worker pod the
        # caller cannot read, four retries into a dispatch -- so an unacceptable
        # name became a poison message the fleet retried indefinitely rather
        # than a refusal at the call that named it.
        assert_all_valid(self._destinations_for(request), what="destination")

        # Counted before the tier is chosen, so the decision rests on a
        # measurement. A stored table is counted by one server-side aggregate
        # rather than by reading it, which is what keeps the count cheap enough
        # to take before committing to anything.
        declared = self._count_source(request)
        fleet = self._fleet_reachable()
        tier = choose_tier(
            request,
            self._settings,
            row_count=declared,
            has_fleet=fleet,
        )

        key = _run_key()
        runs_context = self._write_table(RUNS_TABLE, destination)
        # Staged before the run row so the row can point at it. The request may
        # carry a large rows payload, and a row holding that payload is both a
        # write the backend can reject and a read that costs more than it answers.
        request_key = self._stage_request(key, request)

        record = IngestionRunRecord(
            run_key=key,
            state="queued",
            executed_as=tier,
            source_kind=source.kind,
            target_kind=target.kind,
            request_key=request_key,
            declared_rows=declared,
            created_at=_now(),
        )
        self._get_dm().insert_rows(runs_context, [record.model_dump(exclude_none=True)])

        if not fleet and source.kind in {"files", "folder"}:
            # Files are meant to parse off this process, and a deployment whose
            # control plane cannot reach its backends reads as having no fleet
            # at all -- correctly, since dispatching there would send work
            # nowhere. But the fallback then does the very thing the boundary
            # exists to prevent, and it did so silently on staging for three
            # files of 130-346 MB. Recording it means the run itself says so.
            #
            # Stageless on purpose: this is a fact about the run rather than
            # progress through one, and naming a stage would fabricate progress
            # for a stage that has not started.
            self._record_event(
                key,
                destination=destination,
                level="warning",
                message=(
                    "No worker fleet is reachable, so these files parse in the "
                    "assistant's own process. Configure the pipeline control "
                    "plane to move file parsing off it."
                ),
            )

        self._start(key, runs_context, request, tier=tier, declared=declared)
        return IngestionRun(run_id=key, state="queued", executed_as=tier)

    def _count_source(self, request: IngestionRequest) -> Optional[int]:
        """Measure the source exactly, or return ``None`` when it cannot be.

        Rows in hand are counted directly. A stored table is counted server-side,
        which is one cheap query even on a large context -- unlike reading it,
        which would cost the whole table just to decide where to run.

        Files return ``None``: what a file holds is unknowable before parsing it,
        and guessing from bytes or count is the error this design removes rather
        than refines.
        """
        source = request.source
        if source.kind == "rows":
            return len(source.rows)
        if source.kind == "table":
            return int(
                self._get_dm().reduce(
                    source.context,
                    metric="count",
                    filter=source.filter,
                )
                or 0,
            )
        return None

    def _stage_request(self, run_key: str, request: IngestionRequest) -> str:
        """Persist the request as an artifact and return its key.

        Staged rather than embedded so a retry or resume can rebuild the work
        without the caller reconstructing it, and without a bulk payload ever
        landing in a log row. This is also what a worker reads when the run is
        dispatched, so one representation serves both tiers.
        """
        key = f"jobs/{run_key}/request.json"
        self._store.put_json(key, request.model_dump(mode="json"))
        return key

    def _load_request(self, row: Dict[str, Any]) -> IngestionRequest:
        return IngestionRequest.model_validate(
            self._store.get_json(row["request_key"]),
        )

    def _start(
        self,
        run_key: str,
        runs_context: str,
        request: IngestionRequest,
        *,
        tier: str,
        declared: Optional[int],
    ) -> None:
        if tier == "inline":
            # Registered before the pool picks the run up, so a cancel that
            # arrives while it is still queued is seen at the very first check.
            with self._lock:
                self._inline_control[run_key] = {"cancel": False, "pause": False}
            self._pool.submit(
                self._execute,
                run_key,
                runs_context,
                request,
                declared,
            )
            return
        # Registered before the pool picks it up, for the same reason as the
        # inline flags: a status read racing the pool must see the dispatch as
        # in flight, not as dead.
        with self._lock:
            self._dispatching.add(run_key)
        # Off the caller's thread: staging and uploading a multi-hundred-MB
        # source takes as long as the uplink takes, and `submit` promises a
        # handle immediately. Failures land on the run row, exactly as an
        # inline run's do.
        self._pool.submit(self._dispatch_guarded, run_key, runs_context, request)

    # ── execution ─────────────────────────────────────────────────────────

    def _fleet_reachable(self) -> bool:
        """Whether a worker fleet can actually take work right now.

        Configured is not the same as reachable, and the difference matters in
        one direction only: dispatching to a plane that cannot publish leaves a
        run queued forever, while running in process when a fleet exists costs
        latency and nothing else. So this asks, and a negative answer routes the
        work here.

        Cached for the process's life. The probe is a network round trip and the
        tier decision happens on every submit; a plane that appears later is
        picked up by the next process, and anything it left behind is adoptable
        because both tiers write the same layout.
        """
        if not self._settings.resolved_pipeline_url():
            return False
        with self._lock:
            if self._fleet_probe is None:
                from unify.ingestion_manager.dispatch import probe

                self._fleet_probe = probe(
                    base_url=self._settings.resolved_pipeline_url(),
                )
            return self._fleet_probe

    def _control(self, run_key: str) -> Dict[str, bool]:
        with self._lock:
            return self._inline_control.setdefault(
                run_key,
                {"cancel": False, "pause": False},
            )

    @staticmethod
    @contextlib.contextmanager
    def _pod_work(label: str, run_key: str):
        """Declare pool work to the runtime for as long as it runs.

        `submit` promises a handle immediately, so both tiers hand the real work
        to a pool thread and return. That work therefore outlives the call that
        started it -- and outlives the actor plan whose own ACTIVE_WORK record
        was the only thing telling the runtime not to retire the pod. Row writes
        go through DataManager, which publishes nothing, so without this the
        work is invisible to every idle clock and an inactivity shutdown can
        land in the middle of it.
        """
        from unify.events.active_work import ACTIVE_WORK

        handle = ACTIVE_WORK.begin(
            label=label,
            metadata={"run_key": run_key},
        )
        try:
            yield handle
        finally:
            handle.end()

    def _execute(
        self,
        run_key: str,
        runs_context: str,
        request: IngestionRequest,
        declared: Optional[int],
    ) -> None:
        """Run a request in this process, recording progress as it commits.

        Every exit path settles the ledger. A run left at ``running`` after its
        thread died is indistinguishable from one still working, and that is
        the single failure that makes the whole ledger untrustworthy. A
        cancellation or pause observed mid-run exits without writing -- the
        verb that requested it already wrote the state, and overwriting a
        terminal ``cancelled`` with ``succeeded`` would erase what happened.
        """
        destination = request.destination
        control = self._control(run_key)

        with self._pod_work("ingestion_inline", run_key):
            self._execute_guarded(run_key, runs_context, request, declared)

    def _execute_guarded(
        self,
        run_key: str,
        runs_context: str,
        request: IngestionRequest,
        declared: Optional[int],
    ) -> None:
        destination = request.destination
        control = self._control(run_key)

        try:
            # A cancel or pause that lands while the run is still queued takes
            # effect before any work: the state the verb wrote stands.
            if control["cancel"] or control["pause"]:
                return

            self._update_run(
                run_key,
                runs_context,
                {"state": "running", "started_at": _now()},
            )
            self._record_event(
                run_key,
                destination=destination,
                stage="ingest",
                state="running",
                total=declared,
                message=(
                    f"Storing from {request.source.kind} into {request.target.kind}."
                ),
            )

            try:
                if request.source.kind in {"files", "folder"}:
                    rows, contexts, files = self._ingest_files_inline(
                        run_key,
                        request,
                        control=control,
                    )
                else:
                    outcome = self._ingest_rows(
                        run_key,
                        request,
                        declared=declared,
                        control=control,
                    )
                    rows, contexts, files = (
                        outcome.rows_committed,
                        outcome.contexts,
                        0,
                    )
            except PipelineCancelled:
                # `cancel()` already wrote the terminal state; this records how
                # far the work got before it stopped.
                self._record_event(
                    run_key,
                    destination=destination,
                    stage="ingest",
                    state="cancelled",
                    message="Stopped at a chunk boundary; committed work kept.",
                )
                return
            except RetryWorkItem:
                # A pause surrendered the in-flight work at a checkpoint.
                # `pause()` wrote the state; resume continues from the mark.
                self._record_event(
                    run_key,
                    destination=destination,
                    stage="ingest",
                    state="paused",
                    message="Paused at a checkpoint; resume() continues from it.",
                )
                return
            except IncompleteIngest as shortfall:
                # Distinct from an ordinary failure: rows did land, and the run
                # is resumable from its checkpoint. Saying so is what turns a
                # silent under-ingest into something recoverable.
                logger.error(
                    "Ingestion run %s incomplete: %s",
                    run_key,
                    shortfall.detail,
                )
                self._record_event(
                    run_key,
                    destination=destination,
                    stage="ingest",
                    level="error",
                    state="failed",
                    message=(
                        f"Committed less than the source declared "
                        f"({shortfall.detail}). Resume to continue from the "
                        "last checkpoint."
                    ),
                )
                self._update_run(
                    run_key,
                    runs_context,
                    {
                        "state": "failed",
                        "error": str(shortfall),
                        "parked": len(shortfall.shortfalls),
                        "finished_at": _now(),
                    },
                )
                return
            except DuplicateLiveAttempt:
                # Another attempt owns the work. Leaving the run alone is
                # correct -- the holder will finish it and write the terminal
                # state.
                logger.info(
                    "Ingestion run %s is already being executed; standing down",
                    run_key,
                )
                return
            except Exception as error:  # noqa: BLE001 -- see docstring
                logger.exception("Ingestion run %s failed", run_key)
                self._record_event(
                    run_key,
                    destination=destination,
                    stage="ingest",
                    level="error",
                    state="failed",
                    message=str(error),
                )
                self._update_run(
                    run_key,
                    runs_context,
                    {"state": "failed", "error": str(error), "finished_at": _now()},
                )
                return

            self._record_event(
                run_key,
                destination=destination,
                stage="ingest",
                state="succeeded",
                done=rows,
                total=declared or rows,
                message=(
                    f"Committed {rows} row(s) to {', '.join(contexts) or 'no context'}."
                ),
            )
            updates: Dict[str, Any] = {
                "state": "succeeded",
                "contexts": contexts,
                "rows_written": rows,
                "finished_at": _now(),
            }
            if files:
                updates["files_processed"] = files
            self._update_run(run_key, runs_context, updates)
        finally:
            with self._lock:
                self._inline_control.pop(run_key, None)

    def _ingest_rows(
        self,
        run_key: str,
        request: IngestionRequest,
        *,
        declared: Optional[int],
        control: Dict[str, bool],
    ) -> Any:
        """Ingest a rows or table source through the shared checkpointed core."""
        source = request.source
        target = request.target

        handle = (
            InlineRowsHandle(
                rows=list(source.rows),
                columns=list(source.rows[0].keys()) if source.rows else [],
                row_count=len(source.rows),
            )
            if source.kind == "rows"
            else self._handle_for_table(source, declared=int(declared or 0))
        )

        work = self._table_work(
            table_id=f"run-{run_key}",
            label=target.context,
            handle=handle,
            declared=int(declared or handle.row_count or 0),
            request=request,
            source_path=getattr(handle, "logical_path", "") or target.context,
        )
        return self._run_engine(run_key, [work], request=request, control=control)

    def _table_work(
        self,
        *,
        table_id: str,
        label: str,
        handle: Any,
        declared: int,
        request: IngestionRequest,
        source_path: str = "",
    ) -> TableWork:
        """One unit of engine work, carrying the target's declared identity.

        ``unique_keys`` and ``fields`` come from the target because they are
        statements about the *destination table*, not about any one batch: the
        keys are what make a re-run an upsert instead of an append.
        """
        target = request.target
        embed = request.embed
        return TableWork(
            table_id=table_id,
            label=label,
            context=target.context,
            handle=handle,
            declared_rows=declared,
            source_path=source_path,
            columns=list(handle.columns or []),
            chunk_size=500,
            description=target.description,
            unique_keys=target.unique_keys,
            fields=target.fields,
            embed_columns=embed.columns if embed else None,
            embed_strategy=embed.strategy if embed else "off",
            post_ingest=request.post_ingest,
            infer_untyped_fields=target.infer_untyped_fields,
        )

    def _run_engine(
        self,
        run_key: str,
        work: List[TableWork],
        *,
        request: IngestionRequest,
        control: Dict[str, bool],
    ) -> Any:
        engine = CheckpointedIngest(
            artifact_store=self._store,
            job_id=run_key,
            lease_ttl_seconds=self._settings.LEASE_TTL_SECONDS,
            lease_steal_after_seconds=self._settings.LEASE_STEAL_AFTER_SECONDS,
        )
        # Progress arrives keyed by table, and a table id means nothing to a
        # caller. Resolving it back to the file here is what lets a batch report
        # per-file numbers instead of one aggregate.
        by_table = {unit.table_id: unit for unit in work}

        def _progress(table_id: str, done: int, total: int) -> None:
            unit = by_table.get(table_id)
            self._record_event(
                run_key,
                destination=request.destination,
                stage="ingest",
                state="running",
                done=done,
                total=total or None,
                message=f"Committed {done} row(s).",
                source_path=(unit.source_path or unit.label) if unit else None,
                context=unit.context if unit else None,
                rows_written=done,
                declared_rows=(unit.declared_rows or None) if unit else (total or None),
            )

        return engine.run(
            work,
            dm=self._get_dm(),
            destination=request.destination,
            source_path=run_key,
            # Checked between chunks. Cancel abandons the rest; pause surrenders
            # at the checkpoint so resume() re-does at most one chunk.
            is_cancelled=lambda: control["cancel"],
            should_surrender=lambda: control["pause"],
            on_progress=_progress,
        )

    def _ingest_files_inline(
        self,
        run_key: str,
        request: IngestionRequest,
        *,
        control: Dict[str, bool],
    ) -> tuple[int, List[str], int]:
        """Parse and store files in this process.

        Only reachable when no worker fleet is configured -- with one, files
        always dispatch, because parsing shares this process's memory limit.
        Accepting that risk here is deliberate: a deployment without workers
        (local development, a bare self-host) still has to be able to store an
        attachment, and refusing would fail every file it receives.

        A table target goes through the shared checkpointed engine, so it is
        resumable chunk by chunk like any rows ingestion. A collection target
        runs the file pipeline, whose unit of recovery is the file: a retry
        re-stores whole files, and ``replace_existing`` keeps that idempotent.
        """
        paths = self._resolve_paths(request.source)
        if not paths:
            raise RuntimeError(
                "No files matched this source; nothing was stored.",
            )

        destination = request.destination
        if request.target.kind == "table":
            return self._files_into_table(
                run_key,
                paths,
                request=request,
                control=control,
            )

        from unify.file_manager.types.config import (
            EmbeddingsConfig,
            FilePipelineConfig,
            IngestConfig,
        )

        target = request.target
        config = FilePipelineConfig(
            ingest=IngestConfig(
                storage_id=target.name,
                table_ingest=target.extract_tables,
            ),
            embed=EmbeddingsConfig(
                strategy=request.embed.strategy if request.embed else "after",
            ),
        )

        result = self._get_fm().ingest_files(
            list(paths),
            config=config,
            destination=destination,
        )

        contexts: List[str] = []
        rows = 0
        succeeded = 0
        failures: List[str] = []
        for path, entry in result.files.items():
            if entry.status != "success":
                failures.append(f"{path}: {entry.error or 'failed'}")
                self._record_event(
                    run_key,
                    destination=destination,
                    stage="parse",
                    level="error",
                    message=f"{path}: {entry.error or 'failed'}",
                )
                continue
            succeeded += 1
            content = getattr(entry, "content_ref", None)
            if content is not None and content.context:
                contexts.append(content.context)
                rows += content.record_count
            for table in getattr(entry, "tables_ref", None) or []:
                contexts.append(table.context)
                rows += table.row_count
            self._record_event(
                run_key,
                destination=destination,
                stage="ingest",
                state="running",
                done=succeeded,
                total=len(paths),
                message=f"Stored {path}.",
            )

        if failures and not succeeded:
            raise RuntimeError(
                f"Every file failed to store: {'; '.join(failures)}",
            )
        return rows, list(dict.fromkeys(contexts)), succeeded

    def _files_into_table(
        self,
        run_key: str,
        paths: List[str],
        *,
        request: IngestionRequest,
        control: Dict[str, bool],
    ) -> tuple[int, List[str], int]:
        """Merge the tables found in *paths* into one queryable context.

        Parsing is per file and never raises -- a per-file failure is recorded
        and the rest proceed. The extracted rows then flow through the shared
        engine, one work unit per extracted table, so the write half is
        checkpointed and verified exactly like any other ingestion.

        Transport handles come from ``build_table_handles``, the same helper the
        parse worker lowers through, rather than being built from
        ``table.rows`` here. That is not tidiness. A parser inlines rows only
        below ``TABULAR_INLINE_ROW_LIMIT`` and above it returns the columns, the
        dialect and the row count with ``rows`` deliberately empty, because the
        rows are meant to be streamed from the source. Reading ``rows`` directly
        therefore saw nothing for every table of consequence, and this tier
        failed each one as "no tabular content" -- a 346 MB CSV of 622k rows
        included. Sharing the helper also means ``declared_rows`` is the
        parser's count rather than the length of whatever happened to be
        inlined, which is what the completion check verifies the durable
        checkpoint against.
        """
        from unify.common.pipeline.transport import build_table_handles
        from unify.file_manager.file_parsers.file_parser import FileParser
        from unify.file_manager.file_parsers.types.contracts import FileParseRequest

        destination = request.destination
        parser = FileParser()
        work: List[TableWork] = []
        parsed = 0
        tables_seen = 0
        failures: List[str] = []
        unusable: List[str] = []

        for path in paths:
            if control["cancel"]:
                raise PipelineCancelled(f"Run {run_key} cancelled during parse")
            result = parser.parse(
                FileParseRequest(logical_path=path, source_local_path=path),
            )
            if result.status != "success":
                failures.append(f"{path}: {result.error or 'parse failed'}")
                self._record_event(
                    run_key,
                    destination=destination,
                    stage="parse",
                    level="error",
                    message=f"{path}: {result.error or 'parse failed'}",
                )
                continue
            parsed += 1
            self._record_event(
                run_key,
                destination=destination,
                stage="parse",
                state="running",
                done=parsed,
                total=len(paths),
                message=f"Parsed {path} ({len(result.tables)} table(s)).",
            )
            handles = build_table_handles(result, job_id=run_key)
            for table in result.tables:
                tables_seen += 1
                handle = handles.get(table.table_id)
                declared = table.num_rows
                if declared is None:
                    declared = getattr(handle, "row_count", None)
                if not declared:
                    # An empty sheet is not a failure; there is simply nothing
                    # to write for it.
                    continue
                if handle is None or not _handle_can_yield(handle):
                    # The parser counted rows this transport cannot reach, so
                    # writing what is reachable would be a silent under-ingest.
                    # The dispatched tier refuses the same case for the same
                    # reason.
                    unusable.append(f"{path}:{table.label or table.table_id}")
                    continue
                work.append(
                    self._table_work(
                        # Stable per source table, so a resume of this run finds
                        # the same checkpoint whatever order parsing returned.
                        table_id=f"run-{run_key}-{_safe_id(path)}-{table.table_id}",
                        label=table.label or path,
                        handle=handle,
                        declared=int(declared),
                        request=request,
                        source_path=path,
                    ),
                )

        if not parsed:
            raise RuntimeError(
                f"Every file failed to parse: {'; '.join(failures)}",
            )
        if unusable:
            raise RuntimeError(
                "Parsed tables whose rows this ingestion cannot read: "
                f"{'; '.join(unusable)}. Nothing was stored, because storing "
                "the readable part would under-ingest without reporting it.",
            )
        if not tables_seen:
            raise RuntimeError(
                "Parsing found no tables to store. A table target needs tabular "
                "content; use CollectionTarget to keep these documents whole.",
            )
        if not work:
            raise RuntimeError(
                f"Parsed {tables_seen} table(s) and every one was empty; there "
                "were no rows to store.",
            )

        outcome = self._run_engine(run_key, work, request=request, control=control)
        return outcome.rows_committed, outcome.contexts, parsed

    def _handle_for_table(self, source: Any, *, declared: int) -> Any:
        """Read a stored table into a handle the engine can stream from.

        Paged by offset because the backend serves at most a page per read, so a
        single large read would silently return a prefix.

        The rows do end up in memory, unbounded by anything here: every table
        source runs in process today, because the fleet's unit of work is a
        staged file and no rows job type exists yet. ``MAX_INLINE_ROWS`` is the
        boundary such a job type would restore; until then a very large table
        costs this process memory in exchange for actually executing.

        One consequence worth knowing: a resumed run re-reads the source rather
        than a frozen copy of it, so if the source has been written to in between,
        the checkpoint's offset no longer points at the same rows. Freezing it
        would need the rows staged through the artifact store, which the port's
        materialise call is not shaped for today.
        """
        dm = self._get_dm()
        page = self._settings.EVENTS_PAGE_SIZE
        rows: List[Dict[str, Any]] = []
        offset = 0
        while len(rows) < declared:
            batch = dm.filter(
                source.context,
                filter=source.filter,
                columns=source.columns,
                limit=min(page, declared - len(rows)),
                offset=offset,
            )
            if not batch:
                break
            rows.extend(batch)
            offset += len(batch)
            if len(batch) < page:
                break

        return InlineRowsHandle(
            rows=rows,
            columns=list(rows[0].keys()) if rows else list(source.columns or []),
            row_count=len(rows),
        )

    def _dispatch_guarded(
        self,
        run_key: str,
        runs_context: str,
        request: IngestionRequest,
    ) -> None:
        """Run the dispatch, landing any failure on the run row.

        A dispatch that raises without recording anything leaves the row
        `queued` forever -- the one state whose next step is "keep polling",
        which is exactly wrong for a run that will never start. The failure is
        the run's outcome, so it is written where every other outcome lives.
        """
        try:
            with self._pod_work("ingestion_dispatch_upload", run_key):
                self._dispatch(run_key, runs_context, request)
        except Exception as error:
            self._update_run(
                run_key,
                runs_context,
                {
                    "state": "failed",
                    "error": str(error),
                    "finished_at": _now(),
                },
            )
            self._record_event(
                run_key,
                destination=request.destination,
                stage="parse",
                level="error",
                message=f"Dispatch to the worker fleet failed: {error}",
            )
        finally:
            with self._lock:
                self._dispatching.discard(run_key)

    def _dispatch(
        self,
        run_key: str,
        runs_context: str,
        request: IngestionRequest,
    ) -> None:
        """Hand a run to the pipeline control plane.

        The staged request is what the fleet reads, so nothing about the work is
        re-described here. If no control plane is configured the tier decision
        would not have chosen dispatch, so reaching this without one is a
        misconfiguration rather than a size problem, and it says so.
        The staged request travels with the run rather than being re-described:
        the control plane stages it alongside the sources, and the workers read
        the same document an in-process resume would.
        """
        from unify.ingestion_manager.dispatch import dispatch_run

        if not self._settings.resolved_pipeline_url():
            raise RuntimeError(
                "This run needs the worker fleet and no pipeline control plane is "
                "reachable (neither UNIFY_INGESTION_PIPELINE_URL nor "
                "UNIFY_COMMS_URL is set). Files are "
                "always parsed off the assistant's process, so configure a control "
                "plane or run the self-host worker services.",
            )

        paths = self._resolve_paths(request.source)
        if not paths:
            # The fleet's unit of work is a staged file, and publishing a
            # dispatch with zero jobs succeeds while its status folds to
            # `queued` forever. A source that stages no files cannot dispatch;
            # refusing here turns an infinite hang into a run-row failure.
            raise RuntimeError(
                f"A {request.source.kind!r} source stages no files, and the "
                "worker fleet only executes staged files. This run should have "
                "been routed in process; submit it again.",
            )

        dispatch_id = dispatch_run(
            base_url=self._settings.resolved_pipeline_url(),
            run_key=run_key,
            request=request,
            request_key=f"jobs/{run_key}/request.json",
            request_payload=request.model_dump(mode="json"),
            paths=paths,
            # Where the fleet should journal this run's events: the same two
            # contexts an in-process run writes, so `get_status` reads one
            # history whichever tier executed the work.
            observability={
                "run_key": run_key,
                "runs_context": runs_context,
                "events_context": self._write_table(
                    EVENTS_TABLE,
                    request.destination,
                ),
            },
        )
        self._update_run(run_key, runs_context, {"dispatch_id": dispatch_id})
        self._record_event(
            run_key,
            destination=request.destination,
            stage="parse",
            state="queued",
            message=f"Dispatched to the worker fleet as {dispatch_id}.",
        )

    def _resolve_paths(self, source: Any) -> List[str]:
        """List the files a source names, walking a folder when it is one.

        A walk needs no parsing, so this is measurement rather than prediction --
        it fixes the membership of the set the fleet will process, so a file added
        mid-run is not silently half-included.
        """
        if source.kind == "files":
            paths = list(source.paths)
            self._sync_if_absent(paths)
            return paths
        if source.kind != "folder":
            return []
        root = Path(source.path)
        self._sync_if_absent([str(root)])
        walker = root.rglob if source.recursive else root.glob
        return [str(path) for path in sorted(walker(source.pattern)) if path.is_file()]

    def _sync_if_absent(self, paths: List[str]) -> None:
        """Pull the managed desktop's writes before deciding a path is missing.

        The desktop and this workspace share one tree, but they share it through
        a sync that runs around desktop *execution*. A file the assistant just
        downloaded in a browser therefore exists on the desktop and not yet
        here, and ingesting it moments later measures an empty set -- the same
        shape of silent shortfall as a listing that omits what it was not asked
        to include.

        Only an absent path under the shared root triggers this, so the ordinary
        case costs nothing: paths that are already present, and paths that were
        never on the desktop, do not touch the network.
        """
        missing = [p for p in paths if p and not Path(p).exists()]
        if not missing:
            return
        from unify.file_manager.settings import get_local_root

        try:
            root = Path(get_local_root()).resolve()
        except Exception:
            return
        if not any(self._under(root, Path(p)) for p in missing):
            return

        from unify.common.asyncio_compat import run_coro_sync

        manager = self._sync_manager()
        if manager is None:
            return
        logger.info(
            "Pulling desktop changes before ingesting %d absent path(s).",
            len(missing),
        )
        try:
            run_coro_sync(manager.sync_remote_changes)
        except Exception as exc:
            # A sync that cannot run is not a reason to abandon the ingestion:
            # the paths may be present for some other reason, and the run's own
            # reporting is a better place to discover they are not.
            logger.warning("Pre-ingestion sync failed: %s: %s", type(exc).__name__, exc)

    @staticmethod
    def _under(root: Path, candidate: Path) -> bool:
        try:
            candidate.resolve().relative_to(root)
        except (ValueError, OSError):
            return False
        return True

    def _sync_manager(self) -> Any:
        """The live file-sync manager, or ``None`` when nothing is syncing."""
        from unify.manager_registry import ManagerRegistry

        file_manager = ManagerRegistry.get_file_manager()
        adapter = getattr(file_manager, "_adapter", None)
        manager = getattr(adapter, "_sync_manager", None)
        if manager is None or not getattr(manager, "_started", False):
            return None
        return manager

    # ── observing ─────────────────────────────────────────────────────────

    def _attempt_states(self, run_key: str) -> Dict[str, List[AttemptState]]:
        """Leases recorded against this run, grouped by checkpoint fragment.

        Read from the lease store rather than inferred from progress, because
        progress cannot tell a stalled attempt from a slow one: both sit at the
        same row count. Only whether the lease is still being renewed
        distinguishes them.
        """
        states: Dict[str, List[AttemptState]] = {}
        for key in self._store.list_keys(f"jobs/{run_key}/leases/"):
            table_id = key.rsplit("/", 1)[-1]
            try:
                data = self._store.get_json(key)
            except Exception:  # noqa: BLE001 -- a vanished lease is not held
                continue
            heartbeat_age = _age_seconds(data.get("heartbeat_at"))
            states.setdefault(table_id, []).append(
                AttemptState(
                    attempt_id=str(data.get("attempt_id") or ""),
                    heartbeat_age_s=heartbeat_age,
                    expired=_is_past(data.get("expires_at")),
                    takeover_count=int(data.get("takeover_count") or 0),
                ),
            )
        return states

    def _files_for(
        self,
        row: Dict[str, Any],
        events: List[Dict[str, Any]],
    ) -> List[FileProgress]:
        """Per-file progress, assembled from what the run already recorded.

        The stage counters say two of fifteen files parsed; they never said
        which two. Everything needed to answer that is already written -- the
        staged request names the files, and per-file events carry state, rows
        and destination -- so this reads rather than measures.

        On the in-process tier a file with no event yet is genuinely queued and
        unclaimed: the same process writes those events, so their absence is
        evidence. Unclaimed means waiting for capacity and claimed-but-
        uncommitted means working, and collapsing the two is what made a starved
        batch indistinguishable from a slow one.

        On the dispatched tier absence of an event is **not** evidence -- the
        fleet executes the work and does not write per-file events, so nothing
        here measured the file at all. Those are reported ``observed=False``
        with every measurement left unset, rather than as "queued": claiming a
        file is waiting when nobody looked is the more expensive error of the
        two, because it reads as a finding.
        """
        # A dispatched run's per-file truth lives with the fleet, which reports
        # only aggregates back. Until it emits per-file events, the honest answer
        # here is that these were not observed.
        dispatched = bool(row.get("dispatch_id"))
        paths = [str(p) for p in (row.get("source_paths") or [])]
        attempts = self._attempt_states(str(row.get("run_key") or ""))
        by_path: Dict[str, Dict[str, Any]] = {}
        for event in events:
            path = str(event.get("source_path") or "")
            if not path:
                continue
            # Later events supersede earlier ones for the same file.
            by_path[path] = event

        progress: List[FileProgress] = []
        for path in paths or sorted(by_path):
            event = by_path.get(path) or {}
            if not event and dispatched:
                progress.append(FileProgress(path=path, observed=False))
                continue
            rows_written = int(event.get("rows_written") or 0)
            has_event = bool(event)
            progress.append(
                FileProgress(
                    path=path,
                    observed=True,
                    state=str(  # type: ignore[arg-type]
                        event.get("state")
                        or ("queued" if not has_event else "running"),
                    ),
                    # A committed row proves a worker took it up. An event
                    # without rows still proves a claim; no event at all does
                    # not.
                    claimed=has_event or rows_written > 0,
                    rows_written=rows_written,
                    declared_rows=(
                        int(event["declared_rows"])
                        if event.get("declared_rows") is not None
                        else None
                    ),
                    context=event.get("context") or None,
                    parked=int(event.get("parked") or 0),
                    error=event.get("error") or None,
                    attempts=[
                        state
                        for fragment, states in attempts.items()
                        if _safe_id(path) in fragment
                        for state in states
                    ],
                ),
            )
        return progress

    @functools.wraps(BaseIngestionManager.reconcile, updated=())
    def reconcile(self, run_id: str) -> List[TableReconciliation]:
        status = self.get_status(run_id)
        dm = self._get_dm()
        expected = {f.context: f.declared_rows for f in status.files if f.context}

        results: List[TableReconciliation] = []
        for context in status.contexts:
            stored = int(dm.reduce(context, metric="count") or 0)
            sample = dm.filter(context, limit=_RECONCILE_SAMPLE) or []

            # Which columns carry nothing in any sampled row. A count alone
            # cannot see this, and the failure it exists for looked healthy by
            # count: a run reported every row committed while each row held the
            # string "None" in every data column.
            seen: Dict[str, bool] = {}
            for row in sample:
                values = {
                    **(row.get("entries") or {}),
                    **(row.get("derived_entries") or {}),
                }
                for name, value in values.items():
                    if name.startswith("_") or name in _RECONCILE_IGNORED:
                        continue
                    seen[name] = seen.get(name, False) or not _is_blank(value)

            results.append(
                TableReconciliation(
                    context=context,
                    source_rows=expected.get(context),
                    stored_rows=stored,
                    empty_columns=sorted(
                        n for n, populated in seen.items() if not populated
                    ),
                    sampled_rows=len(sample),
                ),
            )
        return results

    @functools.wraps(BaseIngestionManager.get_status, updated=())
    def get_status(self, run_id: str) -> RunStatus:
        row, runs_context = self._find_run(run_id)
        if row is None:
            raise ValueError(f"No ingestion run {run_id!r}.")

        row = self._fold_fleet_status(row, runs_context)
        events = self._events_for(row["run_key"])
        contexts = row.get("contexts") or []
        parked = int(row.get("parked") or 0)
        state = row.get("state") or "queued"
        files = self._files_for(row, events)

        return RunStatus(
            # The key the caller was handed, never the row's auto-counted id.
            # Lookups accept either, but a status that renames the run it
            # describes is a trap: a plan holding `z9YfymthyUKj` was told
            # `run_id='14'`, and every log line, notification and retry then
            # names an identifier that appears nowhere else in the plan.
            run_id=str(row["run_key"]),
            state=state,  # type: ignore[arg-type]
            files=files,
            executed_as=row.get("executed_as"),
            stages=stages_from_events(events, run_state=state),
            contexts=contexts,
            rows_written=int(row.get("rows_written") or 0),
            files_processed=int(row.get("files_processed") or 0),
            parked=parked,
            error=row.get("error"),
            created_at=row.get("created_at"),
            started_at=row.get("started_at"),
            finished_at=row.get("finished_at"),
            next_step=next_step(
                state=state,
                parked=parked,
                error=row.get("error"),
                executed_as=row.get("executed_as"),
                contexts=contexts,
                files_claimed=sum(1 for f in files if f.claimed),
                files_total=len(files) or None,
            ),
        )

    def _fold_fleet_status(
        self,
        row: Dict[str, Any],
        runs_context: str,
    ) -> Dict[str, Any]:
        """Reconcile a dispatched run's row with the fleet's view of it.

        The workers own the truth about a dispatched run while it executes, and
        nothing else updates the row -- so a read is the moment to reconcile.
        The fleet's answer is advisory until terminal; a terminal answer is
        written back so later reads need not ask again and `wait` can end.

        An unreachable control plane leaves the row as it stands: a stale
        answer that says so via `next_step` beats an exception on a read path.
        """
        dispatch_id = row.get("dispatch_id")
        if row.get("state") in TERMINAL_STATES:
            return row
        if not dispatch_id:
            if row.get("executed_as") != "dispatched":
                return row
            with self._lock:
                still_dispatching = row["run_key"] in self._dispatching
            if still_dispatching:
                # Sources are still being staged and uploaded; the fleet has
                # not heard of this run yet, and that is fine.
                return row
            # Uploads only run in the submitting process, so a dispatched row
            # with no dispatch id and no upload in flight here was orphaned by
            # its submitter dying before publish. No worker will ever pick it
            # up; leaving it `queued` tells the caller to poll forever.
            failed = dict(row)
            failed["state"] = "failed"
            failed["error"] = (
                "The dispatch never reached the worker fleet (the submitting "
                "process ended before the run was published). Nothing was "
                "stored; submit again."
            )
            failed["finished_at"] = _now()
            self._update_run(
                row["run_key"],
                runs_context,
                {
                    "state": failed["state"],
                    "error": failed["error"],
                    "finished_at": failed["finished_at"],
                },
            )
            return failed
        if not self._settings.resolved_pipeline_url():
            return row

        from unify.ingestion_manager.dispatch import fetch_status

        try:
            fleet = fetch_status(
                base_url=self._settings.resolved_pipeline_url(),
                dispatch_id=str(dispatch_id),
            )
        except Exception as error:  # noqa: BLE001 -- read path stays readable
            logger.warning(
                "Pipeline control plane unreachable for %s: %s",
                dispatch_id,
                error,
            )
            return row

        state = fleet.get("state")
        if not state or state == row.get("state"):
            merged = dict(row)
        else:
            merged = dict(row)
            merged["state"] = state
        for field in ("rows_written", "files_processed", "parked"):
            value = fleet.get(field)
            if isinstance(value, int):
                merged[field] = value
        contexts = fleet.get("contexts")
        if isinstance(contexts, list) and contexts:
            merged["contexts"] = contexts
        error_text = fleet.get("error")
        if error_text:
            merged["error"] = error_text

        if merged.get("state") in TERMINAL_STATES:
            merged.setdefault("finished_at", _now())
            self._update_run(
                row["run_key"],
                runs_context,
                {
                    key: merged.get(key)
                    for key in (
                        "state",
                        "rows_written",
                        "files_processed",
                        "parked",
                        "contexts",
                        "error",
                        "finished_at",
                    )
                    if merged.get(key) is not None
                },
            )
        return merged

    def _events_for(self, run_key: str) -> List[Dict[str, Any]]:
        """Read a run's events, paging until they are exhausted.

        Paged rather than fetched with a large limit because the backend serves at
        most a page at a time: asking for more returns a prefix, which would read
        as the whole history and quietly lose the end of a long run -- the part
        that says how it finished.
        """
        dm = self._get_dm()
        page = self._settings.EVENTS_PAGE_SIZE
        events: List[Dict[str, Any]] = []
        for context in self._read_tables(EVENTS_TABLE):
            offset = 0
            while True:
                batch = dm.filter(
                    context,
                    filter=f"run_key == '{run_key}'",
                    limit=page,
                    offset=offset,
                )
                if not batch:
                    break
                events.extend(batch)
                offset += len(batch)
                if len(batch) < page:
                    break
        return sorted(events, key=lambda event: event.get("at") or "")

    @functools.wraps(BaseIngestionManager.get_logs, updated=())
    def get_logs(
        self,
        run_id: str,
        *,
        stage: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[LogEntry]:
        row, _ = self._find_run(run_id)
        if row is None:
            raise ValueError(f"No ingestion run {run_id!r}.")
        events = self._events_for(row["run_key"])
        if stage:
            events = [event for event in events if event.get("stage") == stage]
        window = events[offset : offset + limit]
        return [
            LogEntry(
                at=event.get("at") or "",
                stage=event.get("stage"),
                level=event.get("level") or "info",
                message=event.get("message") or "",
            )
            for event in window
        ]

    @functools.wraps(BaseIngestionManager.wait, updated=())
    def wait(self, run_id: str, *, timeout_s: Optional[float] = None) -> RunStatus:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        # Backs off to a second so a long run does not spend the wait hammering
        # the backend, while a short one still returns promptly.
        interval = 0.2
        while True:
            status = self.get_status(run_id)
            if status.is_terminal:
                return status
            if deadline is not None and time.monotonic() >= deadline:
                return status
            time.sleep(interval)
            interval = min(interval * 1.5, 1.0)

    @functools.wraps(BaseIngestionManager.list_runs, updated=())
    def list_runs(
        self,
        *,
        state: Optional[RunState] = None,
        context: Optional[str] = None,
        limit: int = 50,
    ) -> List[IngestionSummary]:
        dm = self._get_dm()
        rows: List[Dict[str, Any]] = []
        for runs_context in self._read_tables(RUNS_TABLE):
            rows.extend(
                dm.filter(
                    runs_context,
                    filter=f"state == '{state}'" if state else None,
                    limit=min(limit, self._settings.EVENTS_PAGE_SIZE),
                ),
            )
        if context:
            rows = [row for row in rows if context in (row.get("contexts") or [])]
        rows.sort(key=lambda row: row.get("created_at") or "", reverse=True)
        return [
            IngestionSummary(
                run_id=str(row.get("run_id", row.get("run_key", ""))),
                state=row.get("state") or "queued",  # type: ignore[arg-type]
                source_kind=row.get("source_kind") or "",
                target_kind=row.get("target_kind") or "",
                contexts=row.get("contexts") or [],
                rows_written=int(row.get("rows_written") or 0),
                parked=int(row.get("parked") or 0),
                created_at=row.get("created_at"),
                finished_at=row.get("finished_at"),
            )
            for row in rows[:limit]
        ]

    # ── recovering ────────────────────────────────────────────────────────

    @functools.wraps(BaseIngestionManager.retry, updated=())
    def retry(
        self,
        run_id: str,
        *,
        only: RetryScope = "dlq",
        files: Optional[Sequence[str]] = None,
    ) -> RetryResult:
        row, runs_context = self._find_run(run_id)
        if row is None:
            raise ValueError(f"No ingestion run {run_id!r}.")

        targeted = self._resolve_retry_files(row, files)

        state = row.get("state") or "queued"
        if state not in TERMINAL_STATES:
            # A live run already has an attempt working; starting a second one
            # contends for the lease at best, and a scope of "all" would clear
            # the checkpoints out from under the live writer. Paused runs have
            # their own verb.
            #
            # The exception is a stale takeover, which exists precisely for a
            # run that is nominally running and has actually stopped. Requiring
            # a terminal state there made the one case the scope is for
            # unreachable: a stuck run never becomes terminal on its own, so the
            # only options left were waiting forever or cancelling and losing
            # the committed work.
            stale = self._stale_attempt_summary(row, targeted)
            if only != "stale" or not stale.recoverable:
                action = "resume()" if state == "paused" else "wait for it, or cancel()"
                detail = f"This run is {state}; to continue it, {action}."
                if only == "stale" and stale.live:
                    detail = (
                        f"This run is {state} and {stale.live} attempt(s) are "
                        "still renewing their lease, so the work is progressing "
                        "rather than stuck. Taking it over would contend with a "
                        "live writer; wait, or cancel() to stop it."
                    )
                elif only == "stale":
                    detail = (
                        f"This run is {state} and nothing holds a lapsed lease, "
                        "so there is no stalled attempt to take over."
                    )
                return RetryResult(
                    run_id=run_id,
                    scope=only,
                    requeued=0,
                    state=state,  # type: ignore[arg-type]
                    detail=detail,
                    files=list(targeted),
                )

        parked = int(row.get("parked") or 0)
        if only == "dlq" and parked == 0:
            # Zero is an answer, not a failure. Saying so plainly stops a caller
            # concluding the retry itself broke.
            return RetryResult(
                run_id=run_id,
                scope=only,
                requeued=0,
                state=row.get("state") or "queued",  # type: ignore[arg-type]
                detail="Nothing is parked on this run, so there was nothing to retry.",
                files=list(targeted),
            )

        request = self._load_request(row)
        if only == "all" and not row.get("dispatch_id"):
            # The checkpoints are what make a resume skip committed work, so
            # re-attempting everything means discarding them. Done explicitly
            # here rather than left implicit, because it is the one scope that
            # rewrites rows that were already correct. A dispatched run's
            # checkpoints live on the fleet's store; the control plane owns
            # clearing those as part of the retry it serialises.
            #
            # Narrowed to the named files when there are any: the other
            # fourteen files' marks describe rows that committed correctly, and
            # discarding them would turn a one-file retry into a re-ingest of
            # the batch.
            self._store.delete_checkpoints(
                row["run_key"],
                artifact_ids=(
                    self._checkpoint_ids_for(row["run_key"], targeted)
                    if targeted
                    else None
                ),
            )

        self._update_run(
            row["run_key"],
            runs_context,
            {"state": "queued", "error": None, "parked": 0},
        )
        aimed = f" for {len(targeted)} file(s)" if targeted else ""
        self._record_event(
            row["run_key"],
            destination=request.destination,
            stage="ingest",
            message=f"Retrying ({only}){aimed}.",
            state="queued",
        )

        dispatch_id = row.get("dispatch_id")
        if dispatch_id:
            # Asked of the control plane rather than re-published from here: it
            # owns the transition, so a retry cannot race a stale-recovery into
            # two live attempts on one table.
            from unify.ingestion_manager.dispatch import request_retry

            request_retry(
                base_url=self._require_pipeline_url(dispatch_id),
                dispatch_id=dispatch_id,
                scope=only,
                files=list(targeted) or None,
            )
        else:
            self._start(
                row["run_key"],
                runs_context,
                request,
                tier="inline",
                declared=row.get("declared_rows"),
            )

        return RetryResult(
            run_id=run_id,
            scope=only,
            # Files when they were named, else the parked count. The old
            # max(parked, 1) reported one requeued item for a scope that had
            # asked for none, which reads as success.
            requeued=len(targeted) or parked or 1,
            state="queued",
            files=list(targeted),
        )

    def _stale_attempt_summary(
        self,
        row: Dict[str, Any],
        targeted: Sequence[str],
    ) -> _StaleSummary:
        """How many attempts on this run are lapsed versus still renewing.

        Scoped to *targeted* when files were named, so one stuck file can be
        taken over while the rest of a batch keeps working -- the whole point of
        aiming a retry at a file.
        """
        attempts = self._attempt_states(str(row.get("run_key") or ""))
        fragments = [_safe_id(path) for path in targeted]
        live = 0
        lapsed = 0
        for fragment, states in attempts.items():
            if fragments and not any(f and f in fragment for f in fragments):
                continue
            for state in states:
                if state.expired:
                    lapsed += 1
                else:
                    live += 1
        return _StaleSummary(live=live, lapsed=lapsed)

    def _resolve_retry_files(
        self,
        row: Dict[str, Any],
        files: Optional[Sequence[str]],
    ) -> List[str]:
        """The run's own paths matching *files*, refusing any that do not.

        A path with a typo would otherwise select nothing and retry the whole
        run, or retry nothing at all and report success either way. Refusing and
        naming what the run does hold is the only outcome a caller can act on.
        """
        if files is None:
            return []
        known = [str(p) for p in (row.get("source_paths") or [])]
        requested = [str(f) for f in files]
        if not requested:
            raise ValueError(
                "files=[] would retry nothing. Omit the argument to retry the "
                "whole run.",
            )
        unknown = [f for f in requested if f not in known]
        if unknown:
            listed = ", ".join(repr(k) for k in known[:10]) or "none recorded"
            raise ValueError(
                f"This run has no file(s) {', '.join(repr(u) for u in unknown)}. "
                f"It holds: {listed}.",
            )
        return requested

    def _checkpoint_ids_for(self, run_key: str, paths: Sequence[str]) -> List[str]:
        """Checkpoint ids belonging to *paths*.

        One file can yield several tables, so the mapping is discovered from the
        stored keys rather than reconstructed -- which would mean re-parsing the
        file to learn what tables it produced.
        """
        prefix = f"jobs/{run_key}/checkpoints/"
        fragments = [_safe_id(path) for path in paths]
        ids: List[str] = []
        for key in self._store.list_keys(prefix):
            artifact_id = key.rsplit("/", 1)[-1]
            if any(fragment and fragment in artifact_id for fragment in fragments):
                ids.append(artifact_id)
        return ids

    def _require_pipeline_url(self, dispatch_id: Any) -> str:
        """The control plane URL, or a plain refusal when none is configured.

        A dispatched run's work lives on the fleet, so a recovery verb without
        a reachable control plane cannot act on it -- and an empty base URL
        would otherwise surface as an obscure malformed-request error.
        """
        url = self._settings.resolved_pipeline_url()
        if not url:
            raise RuntimeError(
                f"Run {dispatch_id} was dispatched to the worker fleet, but no "
                "pipeline control plane is configured "
                "reachable, so it cannot be "
                "steered from here.",
            )
        return url

    @functools.wraps(BaseIngestionManager.cancel, updated=())
    def cancel(self, run_id: str) -> bool:
        return self._transition(run_id, "cancelled", "Cancelled by request.")

    @functools.wraps(BaseIngestionManager.pause, updated=())
    def pause(self, run_id: str) -> bool:
        return self._transition(run_id, "paused", "Paused by request.")

    @functools.wraps(BaseIngestionManager.resume, updated=())
    def resume(self, run_id: str) -> bool:
        row, runs_context = self._find_run(run_id)
        if row is None or row.get("state") != "paused":
            # Only a paused run can resume. Refusing to "resume" a failed one
            # keeps the two recovery paths distinct: that case wants retry.
            return False
        request = self._load_request(row)
        self._update_run(row["run_key"], runs_context, {"state": "queued"})
        self._record_event(
            row["run_key"],
            destination=request.destination,
            message="Resuming from the last checkpoint.",
            state="queued",
        )

        dispatch_id = row.get("dispatch_id")
        if dispatch_id:
            from unify.ingestion_manager.dispatch import request_resume

            request_resume(
                base_url=self._require_pipeline_url(dispatch_id),
                dispatch_id=dispatch_id,
            )
        else:
            self._start(
                row["run_key"],
                runs_context,
                request,
                tier="inline",
                declared=row.get("declared_rows"),
            )
        return True

    def _transition(self, run_id: str, state: str, message: str) -> bool:
        """Move a live run to *state*, telling the fleet when it owns the work.

        The run row is the record either way, but a dispatched run also has
        messages in flight -- so recording the state without telling the control
        plane would leave workers writing to a run the ledger calls cancelled.
        """
        row, runs_context = self._find_run(run_id)
        if row is None or row.get("state") in TERMINAL_STATES:
            return False
        request = self._load_request(row)

        dispatch_id = row.get("dispatch_id")
        if dispatch_id:
            from unify.ingestion_manager.dispatch import (
                request_cancel,
                request_pause,
            )

            ask = request_cancel if state == "cancelled" else request_pause
            ask(
                base_url=self._require_pipeline_url(dispatch_id),
                dispatch_id=dispatch_id,
            )
        else:
            if (
                row.get("state") == "running"
                and row.get("source_kind") in ("files", "folder")
                and row.get("target_kind") == "collection"
            ):
                # The document pipeline has no chunk boundaries to stop at, so
                # a running collection ingest cannot be steered mid-flight.
                # Refusing is honest; recording "cancelled" while the work
                # completes anyway would make the ledger lie either way.
                return False
            # An in-process run is steered through its control flags, checked
            # between chunks. Cancel abandons the rest; pause surrenders at the
            # checkpoint so resume() re-does at most one chunk.
            with self._lock:
                control = self._inline_control.get(row["run_key"])
                if control is not None:
                    control["cancel" if state == "cancelled" else "pause"] = True

        updates: Dict[str, Any] = {"state": state}
        if state == "cancelled":
            updates["finished_at"] = _now()
        self._update_run(row["run_key"], runs_context, updates)
        self._record_event(
            row["run_key"],
            destination=request.destination,
            message=message,
            state=state,
        )
        return True
