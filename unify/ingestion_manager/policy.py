"""Decisions that are pure functions of a request or a run.

Kept out of the manager so they can be reasoned about and tested without a
backend, and so the two that matter most -- where a request runs, and what a
caller should do next -- are stated once each rather than inferred from control
flow at several call sites.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from unify.ingestion_manager.settings import IngestionSettings
from unify.ingestion_manager.types.request import IngestionRequest
from unify.ingestion_manager.types.run import TERMINAL_STATES, StageProgress

Tier = Literal["inline", "dispatched"]


def choose_tier(
    request: IngestionRequest,
    settings: Optional[IngestionSettings] = None,
    *,
    row_count: Optional[int] = None,
    has_fleet: Optional[bool] = None,
) -> Tier:
    """Decide where a request runs. Deterministic, and never a caller's choice.

    Two rules, and the reason they differ is the whole design:

    **Files always dispatch.** Parsing loads the file and its model into the
    process that does it, and a thread does not change that -- threads share one
    address space and one memory limit, so a parse that overruns takes down the
    assistant sharing the process, not just the ingestion. There is also no
    number that predicts the risk: bytes and file count say nothing about page
    count or density. So the answer cannot be a threshold; it has to be a
    boundary, and the boundary is the process.

    **Rows and tables run in process, whatever their size.** The fleet's unit
    of work is a staged *file*: dispatching publishes one parse message per
    uploaded source, so a rows or table request -- which stages no file --
    would publish zero jobs and sit ``queued`` forever, unrecoverable by
    anything short of reading the manifest. Until the fleet grows a rows job
    type, in process is the only tier that can actually execute this work, and
    it executes it correctly: the inline engine checkpoints, verifies against
    the declared count, and resumes, so size costs the assistant latency and
    contention rather than correctness. ``MAX_INLINE_ROWS`` remains the
    documented ceiling a rows job type will restore the boundary at.

    ``row_count`` is the exact count: ``len(rows)`` for rows in hand, or one
    server-side aggregate for a stored table.

    Both tiers write the same artifacts and the same checkpoints, so choosing
    dispatch for work that would have been quick costs latency, and nothing else:
    the run is still resumable, still recoverable, and still asked about the same
    way.
    """
    config = settings or IngestionSettings()
    fleet = bool(config.PIPELINE_URL) if has_fleet is None else has_fleet

    # No fleet reachable: in process is the only tier there is. Safe rather than
    # merely tolerable -- the artifacts and checkpoints land in the layout a
    # fleet reads, so one configured later can adopt whatever was left behind.
    if not fleet:
        return "inline"

    if request.source.kind in {"files", "folder"}:
        return "dispatched"

    return "inline"


def next_step(
    *,
    state: str,
    parked: int,
    error: Optional[str],
    executed_as: Optional[str],
    contexts: List[str],
    files_claimed: Optional[int] = None,
    files_total: Optional[int] = None,
) -> str:
    """State the one action that makes sense for a run in this condition.

    Exists because a status a caller has to interpret will eventually be
    interpreted wrongly, and the cost of that is either a retry that duplicates
    data or a failure nobody notices. Every branch names a concrete next action,
    including that there is nothing to do.
    """
    if state == "queued":
        where = "on the worker fleet" if executed_as == "dispatched" else "in process"
        # "Queued" covers two conditions that need different responses, and
        # collapsing them made a starved batch indistinguishable from a slow one:
        # polling is right for work that has been taken up, and useless for work
        # nothing has claimed, where the answer is that capacity is held
        # elsewhere.
        if files_total and files_claimed is not None and files_claimed < files_total:
            waiting = files_total - files_claimed
            return (
                f"Queued {where}: {files_claimed} of {files_total} file(s) taken "
                f"up by a worker, {waiting} still unclaimed. Unclaimed work is "
                "waiting on capacity rather than making slow progress, so "
                "polling will not change it -- check whether other runs are "
                "holding the fleet before resubmitting anything. Per-file state "
                "is in status.files."
            )
        return f"Nothing yet -- this is queued to run {where}. Poll get_status again."

    if state == "running":
        return (
            "Still running. Poll get_status again, or call wait() if the plan "
            "cannot continue without the data."
        )

    if state == "paused":
        return (
            "Paused with work outstanding. Call resume() to continue from the "
            "last checkpoint, or cancel() to abandon it."
        )

    if state == "cancelled":
        return (
            "Cancelled, so the remaining work was abandoned. Anything already "
            "committed is still there. Submit again to redo the rest."
        )

    if state == "failed":
        if parked > 0:
            return (
                f"Failed with {parked} item(s) parked after exhausting their "
                'retries. Call retry(only="dlq") to re-attempt just those -- work '
                "that already committed is left alone."
            )
        detail = f" The reported cause: {error}" if error else ""
        return (
            "Failed before anything could be parked, so a retry would fail the "
            f"same way. Read get_logs() to find the cause, fix it, then submit "
            f"again.{detail}"
        )

    if state == "succeeded":
        if parked > 0:
            # Worth flagging loudly: a run that finished with parked items looks
            # successful in a listing, and the missing rows are easy to miss.
            return (
                f"Finished, but {parked} item(s) were parked and are not in the "
                'result. Call retry(only="dlq") to bring them in.'
            )
        if contexts:
            return f"Nothing -- this finished. The data is in {', '.join(contexts)}."
        return "Nothing -- this finished."

    return "Unrecognised state; read get_logs() to see what happened."


def stages_from_events(
    events: List[dict],
    *,
    run_state: Optional[str] = None,
) -> List[StageProgress]:
    """Fold recorded events into current per-stage progress.

    Progress is derived rather than stored so there is one source of truth. A
    separately maintained counter drifts from the events the moment a worker dies
    between incrementing it and recording why.

    ``run_state`` closes the stages when the run itself is over. Stage events are
    append-only and a stage that simply ran to completion records no closing
    event, so a terminal run otherwise reports stages still ``running`` -- a
    failed run showed ``parse: running`` beside ``ingest: failed`` even though
    parsing had finished and it was the write that failed. A stage that reached
    its total finished; one that did not ended however the run did.
    """
    ordered: dict[str, StageProgress] = {}
    for event in events:
        stage = event.get("stage")
        if not stage:
            continue
        current = ordered.get(stage)
        if current is None:
            current = StageProgress(stage=stage, state="running")
            ordered[stage] = current
        state = event.get("state")
        if state:
            current.state = state
        done = event.get("done")
        if isinstance(done, int):
            current.done = done
        total = event.get("total")
        if isinstance(total, int):
            current.total = total
        if event.get("level") == "error" and event.get("message"):
            current.error = event["message"]

    if run_state in TERMINAL_STATES:
        for progress in ordered.values():
            if progress.state not in TERMINAL_STATES:
                reached_total = (
                    progress.total is not None and progress.done >= progress.total
                )
                progress.state = "succeeded" if reached_total else run_state

    return list(ordered.values())
