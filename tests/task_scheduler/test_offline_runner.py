"""Tests for the headless offline task runner."""

from types import SimpleNamespace


def _seed_env(monkeypatch):
    monkeypatch.setenv("ASSISTANT_ID", "42")
    monkeypatch.setenv("UNITY_OFFLINE_TASK_RUN_KEY", "offline:scheduled:42:101:rev")
    monkeypatch.setenv("UNITY_OFFLINE_TASK_ID", "101")
    monkeypatch.setenv("UNITY_OFFLINE_TASK_FUNCTION_ID", "777")
    monkeypatch.setenv("UNITY_OFFLINE_TASK_REQUEST", "Send the daily summary email.")
    monkeypatch.setenv("UNITY_OFFLINE_TASK_SOURCE_TASK_LOG_ID", "555")
    monkeypatch.setenv("UNITY_OFFLINE_TASK_ACTIVATION_REVISION", "rev-123")
    monkeypatch.setenv("ORCHESTRA_URL", "https://orchestra.test")
    monkeypatch.setenv("ORCHESTRA_ADMIN_KEY", "admin-key")


def test_offline_runner_marks_run_completed(monkeypatch):
    """Successful executions should mark the run completed with a summary."""

    from unity.task_scheduler import offline_runner

    _seed_env(monkeypatch)
    updates = []

    class _FakeHandle:
        async def result(self):
            return SimpleNamespace(
                result={"ok": True},
                stdout="done",
                stderr="",
                error="",
            )

    class _FakeActor:
        def __init__(self, *args, **kwargs):
            pass

        async def act(self, *, request, function_id):
            assert request == "Send the daily summary email."
            assert function_id == 777
            return _FakeHandle()

        async def close(self):
            return None

    monkeypatch.setattr(offline_runner, "SingleFunctionActor", _FakeActor)
    monkeypatch.setattr(
        offline_runner,
        "_update_task_run",
        lambda assistant_id, run_key, payload: updates.append(
            (assistant_id, run_key, payload),
        ),
    )

    exit_code = offline_runner.main()

    assert exit_code == 0
    assert updates[0][0] == "42"
    assert updates[0][2]["state"] == "running"
    assert updates[1][2]["state"] == "completed"
    assert "result_summary" in updates[1][2]


def test_offline_runner_marks_run_failed_when_function_errors(monkeypatch):
    """ExecutionResult.error should mark the run failed without raising."""

    from unity.task_scheduler import offline_runner

    _seed_env(monkeypatch)
    updates = []

    class _FakeHandle:
        async def result(self):
            return SimpleNamespace(
                result=None,
                stdout="",
                stderr="trace",
                error="boom",
            )

    class _FakeActor:
        def __init__(self, *args, **kwargs):
            pass

        async def act(self, *, request, function_id):
            return _FakeHandle()

        async def close(self):
            return None

    monkeypatch.setattr(offline_runner, "SingleFunctionActor", _FakeActor)
    monkeypatch.setattr(
        offline_runner,
        "_update_task_run",
        lambda assistant_id, run_key, payload: updates.append(
            (assistant_id, run_key, payload),
        ),
    )

    exit_code = offline_runner.main()

    assert exit_code == 1
    assert updates[0][0] == "42"
    assert updates[0][2]["state"] == "running"
    assert updates[1][2]["state"] == "failed"
    assert updates[1][2]["error"] == "boom"
