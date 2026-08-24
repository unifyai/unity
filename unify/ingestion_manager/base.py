"""The IngestionManager contract.

Storing data is one verb with two questions: where does it come from, and where
should it land. Those vary independently, so they are two arguments rather than one
config, and every combination of them means something:

===============  ====================  ==========================================
source           target                result
===============  ====================  ==========================================
``RowsSource``   ``TableTarget``        rows become one queryable table
``TableSource``  ``TableTarget``        stored rows reshaped into another table
``FilesSource``  ``CollectionTarget``  documents kept whole, tables inside them
                                        extracted alongside
``FilesSource``  ``TableTarget``        spreadsheets or CSVs merged into one table
``FolderSource`` either                 the same, over an open-ended set of files
===============  ====================  ==========================================

Choosing a target has one rule: **a table is for querying columns, a collection is
for keeping documents whole.** If the intent is to filter, aggregate or chart it,
that is a table. If the intent is to read, search or cite it, that is a collection.

## Nothing blocks

``submit`` records the request and returns a handle immediately. Work happens in the
background whichever tier runs it, so a plan is never held open by a large
ingestion. Poll ``get_status`` when there is other work to do, or call ``wait`` when
there genuinely is not.

## Where work runs is not a parameter

There is no mode or tier argument, and adding one would be a mistake rather than a
convenience. Files are parsed away from the assistant's own process whenever a
worker fleet is reachable, because parsing loads the file and its model into
whatever process does it and no number predicts that cost in advance; a deployment
without a fleet parses in process rather than refusing the file. Rows and stored
tables run in process only under a *measured* row ceiling, where queue round-trip
would cost more than the work.

Both paths write the same artifacts and the same checkpoints, so the choice affects
latency and nothing else. A run has the same id, the same states and the same
recovery verbs either way, and asking about one reads identically.

## Runs survive the process that started them

Every run is recorded before any work starts, and progress is checkpointed as it
commits — so an interrupted run resumes from its last committed chunk instead of
starting over or double-writing. A failure is always something with an id that can
be inspected and retried, never an error that scrolled past.

A run also cannot report success while holding less than its source declared: the
committed total is checked against the measured count, and a shortfall keeps the run
open for resume and then fails loudly rather than passing as a quiet under-ingest.

Items that exhaust their retries are parked rather than dropped, and
``retry(only="dlq")`` re-attempts exactly those, leaving committed work untouched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from unify.ingestion_manager.types.request import (
    EmbedSpec,
    IngestionSource,
    IngestionTarget,
)
from unify.ingestion_manager.types.run import (
    IngestionRun,
    IngestionSummary,
    LogEntry,
    RetryResult,
    RetryScope,
    RunState,
    RunStatus,
)

# Imported for the signature of `post_ingest`, which is the same shape a direct
# table write accepts, so derived columns are requested identically either way.
from unify.data_manager.types.ingest import PostIngestConfig


class BaseIngestionManager(ABC):
    """Store data and files from any source, observably and recoverably."""

    # ── submitting ────────────────────────────────────────────────────────

    @abstractmethod
    def submit(
        self,
        source: IngestionSource,
        target: IngestionTarget,
        *,
        embed: Optional[EmbedSpec] = None,
        post_ingest: Optional[PostIngestConfig] = None,
        destination: Optional[str] = None,
    ) -> IngestionRun:
        """Start storing data, and return a handle without waiting for it.

        Returns as soon as the run is recorded — before the work runs, so that a
        crash a moment later still leaves something with an id to resume. The
        returned ``run_id`` is how everything afterwards refers to this work, and
        the run's ``contexts`` report the exact paths it wrote, which is what lets
        a view be built over the result without anyone predicting the storage
        layout in advance.

        Parameters
        ----------
        source : RowsSource | FilesSource | FolderSource | TableSource
            Where the data comes from. ``RowsSource`` for anything already in hand,
            including a response just fetched from an external service.
            ``FilesSource`` for specific files, ``FolderSource`` for an open-ended
            set of them, ``TableSource`` to reshape what is already stored.
        target : TableTarget | CollectionTarget
            Where it lands. ``TableTarget`` names one queryable context and is the
            right choice when the data will be filtered, aggregated or charted.
            ``CollectionTarget`` keeps documents whole and extracts any tables
            found inside them; leave its ``name`` unset to give each file its own
            namespace, or set it to group a related set under a stable name that
            can be added to later.
        embed : EmbedSpec | None
            Text columns to make semantically searchable. Requested the same way
            for rows and for file content.
        post_ingest : PostIngestConfig | None
            Derived columns computed once the data is stored.
        destination : str | None
            Ownership root: ``"personal"`` (the default) or ``"team:<id>"``. The
            privacy floor is personal — when it is unclear whether something
            belongs to a team, ask rather than widening the audience.

        Returns
        -------
        IngestionRun
            Handle carrying ``run_id`` and the current state. ``contexts`` fills in
            as the run progresses.

        Raises
        ------
        ValueError
            If the source and target cannot mean anything together — rows have no
            documents to keep whole, so they cannot go to a collection. The message
            names the alternative.
        """

    # ── observing ─────────────────────────────────────────────────────────

    @abstractmethod
    def reconcile(self, run_id: str) -> List["TableReconciliation"]:
        """Check what landed against what the source held, per table.

        The honest closing step of an ingestion, and the one a row count cannot
        perform alone: a run can report every row committed while each row holds
        no values, in which case the count agrees with a table that is useless.
        So this reports both how many rows are stored and which columns are
        blank in every row sampled.

        Use it after a run reaches a terminal state, and before telling anyone
        the data is ready. ``complete`` on each result is the single question --
        it is false both when rows are missing and when the rows that arrived
        came without their values.

        Parameters
        ----------
        run_id : str
            The handle returned by ``submit``.
        """

    @abstractmethod
    def get_status(self, run_id: str) -> RunStatus:
        """Report where a run has got to, and what to do about it.

        Carries per-stage progress, how many rows and files have been handled, how
        many items are parked, and any error. ``next_step`` states the single action
        that makes sense now in plain words — including that nothing is needed when
        the run simply finished — so recovery does not depend on reconstructing the
        rule from the other fields.

        Safe to poll. Check ``is_terminal`` to know when polling can stop.

        Parameters
        ----------
        run_id : str
            The handle returned by ``submit``, or listed by ``list_runs``.
        """

    @abstractmethod
    def get_logs(
        self,
        run_id: str,
        *,
        stage: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[LogEntry]:
        """Read what a run recorded, oldest first.

        For the question ``get_status`` cannot answer: *why* a stage failed. Narrow
        to one stage when a status has already identified where the failure was.

        A long run records more than one read returns. ``offset`` continues from
        where the last read stopped, so a full history is a loop rather than a
        larger ``limit`` — raising the limit past what the backend serves would
        silently return a prefix and read as the whole story.

        Parameters
        ----------
        run_id : str
            The run whose history to read.
        stage : str | None
            Restrict to one stage's entries once a status has identified where
            the failure was.
        limit : int
            Maximum entries returned by one read.
        offset : int
            Where to continue from; page through a long history in a loop
            rather than raising ``limit``.
        """

    @abstractmethod
    def wait(
        self,
        run_id: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> RunStatus:
        """Block until a run finishes, or until the timeout passes.

        Use only when the plan genuinely cannot proceed without the data — for
        example when the next step builds a view over it. Otherwise prefer
        ``get_status``, which leaves the plan free to do other work.

        Returns the final status on completion, or the current one if the timeout
        passes first. A timeout is not a failure: the run continues, and the same
        ``run_id`` still tracks it.

        Parameters
        ----------
        run_id : str
            The run to wait on.
        timeout_s : float | None
            Stop waiting after this many seconds; ``None`` waits until the run
            reaches a terminal state.
        """

    @abstractmethod
    def list_runs(
        self,
        *,
        state: Optional[RunState] = None,
        context: Optional[str] = None,
        limit: int = 50,
    ) -> List[IngestionSummary]:
        """List recent runs, newest first.

        Filter by ``state`` to find what needs attention, or by ``context`` to see
        what has been written to a particular place — which is how to tell whether
        a table is stale, and what last wrote to it.

        Parameters
        ----------
        state : RunState | None
            Only runs currently in this state.
        context : str | None
            Only runs that wrote to this context path.
        limit : int
            Maximum summaries returned, newest first.
        """

    # ── recovering ────────────────────────────────────────────────────────

    @abstractmethod
    def retry(
        self,
        run_id: str,
        *,
        only: RetryScope = "dlq",
        files: Optional[Sequence[str]] = None,
    ) -> RetryResult:
        """Re-attempt part of a run.

        ``only="dlq"`` re-attempts just the items that exhausted their retries and
        were parked. This is the default because it is the safe one: work that
        already succeeded is not touched, so retrying twice cannot duplicate data.

        ``only="stale"`` re-attempts items claimed by a worker that then stopped
        reporting, which recovers a crashed worker's share without disturbing
        healthy work. It is the one scope that applies to a run still marked
        running, because that is the state a stalled run sits in: a stuck
        attempt never reaches a terminal state on its own, so requiring one
        would leave waiting forever or cancelling as the only options. It
        refuses while any targeted attempt is still renewing its lease, and
        says which of the two it found.

        ``only="all"`` re-attempts everything, including items that succeeded. Only
        appropriate when the target was emptied or the way the data is parsed
        changed, since it will re-write rows that were already correct.

        Otherwise only a finished run can be retried. A queued or running run
        already has a live attempt, and a paused one is continued with
        ``resume`` — asking here returns zero requeued items and says which verb
        applies.

        ``files`` narrows any of those scopes to named source files, for the case
        where fourteen of fifteen landed and one did not. Rows that already
        committed are skipped on the way back to where the file stopped, so
        retrying a file that partly succeeded does not duplicate its rows — and
        with ``only="all"`` only that file's progress marks are discarded, never
        the batch's.

        A file the run does not hold is refused, and the refusal lists what it
        does hold. A mistyped path would otherwise select nothing and report
        success, which is indistinguishable from having worked.

        A result of zero requeued items is an answer, not a failure: there was
        nothing in that scope to retry.

        Parameters
        ----------
        run_id : str
            The finished run to re-attempt.
        only : {"dlq", "stale", "all"}
            The scope to re-attempt: parked items only (the safe default),
            items whose worker stopped reporting, or everything including
            work that already succeeded.
        files : sequence of str, optional
            Source files to aim the retry at. Omit for the whole run.
        """

    @abstractmethod
    def cancel(self, run_id: str) -> bool:
        """Stop a run and abandon its remaining work.

        Work already completed stays; nothing is rolled back. A cancelled run
        cannot be resumed — use ``pause`` to stop something that should continue
        later. Returns whether the run was in a state that could be cancelled.

        Parameters
        ----------
        run_id : str
            The run to stop.
        """

    @abstractmethod
    def pause(self, run_id: str) -> bool:
        """Stop a run but keep its remaining work.

        For relieving pressure — a rate limit being approached, a noisy neighbour —
        without losing progress. Resume when the reason has passed.

        Parameters
        ----------
        run_id : str
            The run to pause.
        """

    @abstractmethod
    def resume(self, run_id: str) -> bool:
        """Continue a paused run from where it stopped.

        Only a paused run can be resumed. To re-attempt a failed one, use
        ``retry``.

        Parameters
        ----------
        run_id : str
            The paused run to continue.
        """
