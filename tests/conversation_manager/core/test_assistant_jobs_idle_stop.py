"""Tests for the public assistant_jobs wrapper surface."""

from __future__ import annotations

from unity.conversation_manager import assistant_jobs
from unity.deploy_runtime import register_deploy_runtime, reset_deploy_runtime


class _RecordingJobsBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def mark_job_label(self, *args, **kwargs):
        self.calls.append(("mark_job_label", args, kwargs))
        return True

    def log_job_startup(self, *args, **kwargs):
        self.calls.append(("log_job_startup", args, kwargs))

    def update_liveview_url(self, *args, **kwargs):
        self.calls.append(("update_liveview_url", args, kwargs))

    def mark_job_done(self, *args, **kwargs):
        self.calls.append(("mark_job_done", args, kwargs))


def teardown_function() -> None:
    reset_deploy_runtime()


def test_mark_job_done_delegates_to_registered_backend():
    backend = _RecordingJobsBackend()
    register_deploy_runtime(jobs=backend)

    assistant_jobs.mark_job_done(
        "unity-job-1",
        inactivity_timeout=30.0,
        shutdown_reason="idle_timeout",
    )

    assert backend.calls == [
        (
            "mark_job_done",
            ("unity-job-1",),
            {"inactivity_timeout": 30.0, "shutdown_reason": "idle_timeout"},
        ),
    ]


def test_mark_job_label_delegates_to_registered_backend():
    backend = _RecordingJobsBackend()
    register_deploy_runtime(jobs=backend)

    result = assistant_jobs.mark_job_label(
        "unity-job-1",
        "running",
        assistant_id="42",
        ack_ts="1711800000",
    )

    assert result is True
    assert backend.calls == [
        (
            "mark_job_label",
            ("unity-job-1", "running"),
            {
                "assistant_id": "42",
                "ack_ts": "1711800000",
                "timeout": 30,
                "retries": 0,
            },
        ),
    ]
