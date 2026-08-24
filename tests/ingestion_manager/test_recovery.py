"""Retrying one file, and taking over an attempt that stopped.

Two gaps this pins, both found by trying to recover a real staging run:

* **Retry was run-scoped.** Fourteen files of fifteen landed; the only verbs
  available re-attempted the batch. With ``only="all"`` that discards the marks
  belonging to the fourteen that were already correct, so recovering one file
  meant re-ingesting everything.
* **A stuck run could not be recovered at all.** ``retry`` required a terminal
  state, and a stalled run sits in ``running`` forever -- nothing moves it on.
  The ``"stale"`` scope existed in the contract for exactly this case and was
  unreachable, leaving "wait indefinitely" and "cancel and lose the committed
  rows" as the only options.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from unify.common.pipeline.artifact_store import LocalArtifactStore
from unify.ingestion_manager.ingestion_manager import (
    _StaleSummary,
    _age_seconds,
    _is_past,
)
from unify.ingestion_manager.types import AttemptState, FileProgress


class TestScopingACheckpointDiscard:
    def test_discarding_one_files_marks_leaves_the_others(self):
        # The bug this exists for: a one-file retry that cleared the batch's
        # progress turned into a re-ingest of every file that was already right.
        with tempfile.TemporaryDirectory() as root:
            store = LocalArtifactStore(root_dir=root)
            marks = pathlib.Path(root) / "jobs" / "job1" / "checkpoints"
            marks.mkdir(parents=True)
            for name in ("run-job1-dimJobDetails-t0", "run-job1-DimDates-t0"):
                (marks / name).write_text("{}")

            store.delete_checkpoints(
                "job1",
                artifact_ids=["run-job1-dimJobDetails-t0"],
            )

            assert [p.name for p in marks.iterdir()] == ["run-job1-DimDates-t0"]

    def test_an_unscoped_discard_still_clears_everything(self):
        # only="all" on a whole run must keep working: it is the scope that
        # exists for a target that was emptied.
        with tempfile.TemporaryDirectory() as root:
            store = LocalArtifactStore(root_dir=root)
            marks = pathlib.Path(root) / "jobs" / "job1" / "checkpoints"
            marks.mkdir(parents=True)
            (marks / "t0").write_text("{}")

            store.delete_checkpoints("job1")

            assert not marks.exists()

    def test_an_empty_id_list_discards_nothing(self):
        # Distinct from None. Passing no ids must not be read as "all of them".
        with tempfile.TemporaryDirectory() as root:
            store = LocalArtifactStore(root_dir=root)
            marks = pathlib.Path(root) / "jobs" / "job1" / "checkpoints"
            marks.mkdir(parents=True)
            (marks / "t0").write_text("{}")

            store.delete_checkpoints("job1", artifact_ids=[])

            assert [p.name for p in marks.iterdir()] == ["t0"]


class TestWhenTakeoverIsSafe:
    def test_a_lapsed_attempt_with_nothing_live_is_recoverable(self):
        assert _StaleSummary(live=0, lapsed=1).recoverable is True

    def test_a_live_attempt_blocks_takeover(self):
        # Contending for a lease the holder is still renewing corrupts nothing
        # but wastes a worker and freezes the checkpoint behind the loser.
        assert _StaleSummary(live=1, lapsed=1).recoverable is False

    def test_nothing_held_is_not_recoverable(self):
        # "Took over 0 attempts" reports success for an action that did nothing.
        assert _StaleSummary(live=0, lapsed=0).recoverable is False


class TestReadingALease:
    def test_an_unreadable_expiry_counts_as_expired(self):
        # A lease that cannot be honoured must not read as live: that strands
        # the work permanently, where the wrong reading the other way costs one
        # contended takeover.
        assert _is_past("not-a-timestamp") is True
        assert _is_past(None) is True
        assert _is_past("") is True

    def test_a_past_expiry_is_past_and_a_future_one_is_not(self):
        assert _is_past("2020-01-01T00:00:00+00:00") is True
        assert _is_past("2999-01-01T00:00:00+00:00") is False

    def test_a_naive_timestamp_is_read_as_utc(self):
        # Leases are written with isoformat() and some bindings drop the zone.
        # Reading one as local time would shift staleness by the offset.
        assert _is_past("2020-01-01T00:00:00") is True

    def test_an_unreadable_heartbeat_is_unknown_not_fresh(self):
        # None means "cannot say". Zero would mean "renewed just now", which
        # presents a dead attempt as healthy.
        assert _age_seconds("nonsense") is None
        assert _age_seconds(None) is None

    def test_a_heartbeat_age_is_never_negative(self):
        # Clock skew between a worker and this process must not produce a
        # negative age that reads as a fresher renewal than is possible.
        assert _age_seconds("2999-01-01T00:00:00+00:00") == 0.0


class TestWhatAFileReportsAboutItsAttempts:
    def test_a_file_nothing_holds_is_not_recoverable(self):
        assert FileProgress(path="a.csv").recoverable is False

    def test_a_file_whose_every_attempt_lapsed_is_recoverable(self):
        progress = FileProgress(
            path="a.csv",
            attempts=[AttemptState(attempt_id="x", expired=True)],
        )

        assert progress.recoverable is True

    def test_one_live_attempt_among_lapsed_ones_blocks_it(self):
        progress = FileProgress(
            path="a.csv",
            attempts=[
                AttemptState(attempt_id="x", expired=True),
                AttemptState(attempt_id="y", expired=False),
            ],
        )

        assert progress.recoverable is False

    def test_an_attempt_carries_no_worker_identity(self):
        # Deliberate: naming the machine holding a lease widens what a prompt
        # injection can reach, and the actionable question is only whether
        # takeover is safe.
        assert "owner_id" not in AttemptState.model_fields
        assert "pod" not in AttemptState.model_fields


class TestNotObservedIsNotQueued:
    def test_an_unobserved_file_reports_no_state_at_all(self):
        # The dispatched tier does not write per-file events, so absence of one
        # is not evidence. Reporting "queued" there was a measurement nobody
        # took, presented as a finding.
        progress = FileProgress(path="a.csv", observed=False)

        assert progress.state is None
        assert progress.claimed is None

    def test_an_observed_queued_file_says_so(self):
        progress = FileProgress(
            path="a.csv",
            observed=True,
            state="queued",
            claimed=False,
        )

        assert progress.state == "queued"
        assert progress.claimed is False


class TestRefusingAMistypedFile:
    """``_resolve_retry_files`` touches no instance state, so it is called unbound."""

    @staticmethod
    def _resolve(row, files):
        from unify.ingestion_manager.ingestion_manager import IngestionManager

        return IngestionManager._resolve_retry_files(None, row, files)

    def test_a_known_file_is_accepted(self):
        row = {"source_paths": ["a.csv", "b.csv"]}

        assert self._resolve(row, ["b.csv"]) == ["b.csv"]

    def test_omitting_files_means_the_whole_run(self):
        assert self._resolve({"source_paths": ["a.csv"]}, None) == []

    def test_an_unknown_file_is_refused_and_the_real_ones_listed(self):
        # Silently matching nothing would either retry the whole run or retry
        # nothing, and report success either way.
        with pytest.raises(ValueError) as excinfo:
            self._resolve({"source_paths": ["a.csv", "b.csv"]}, ["typo.csv"])

        message = str(excinfo.value)
        assert "typo.csv" in message
        assert "a.csv" in message

    def test_an_empty_list_is_refused_rather_than_widened(self):
        # files=[] is a caller bug. Treating it as "the whole run" would run the
        # largest possible action in response to the smallest possible request.
        with pytest.raises(ValueError):
            self._resolve({"source_paths": ["a.csv"]}, [])
