from types import SimpleNamespace

from unity.task_scheduler.machine_state import TASK_MACHINE_STATE_PROJECT
from unity.task_scheduler.storage import TasksStore


def test_tasks_store_get_rows_passes_explicit_project_override(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_get_logs(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("unity.task_scheduler.storage.unify.get_logs", _fake_get_logs)

    store = TasksStore("Tasks/Activations", project=TASK_MACHINE_STATE_PROJECT)
    rows = store.get_rows(limit=5)

    assert rows == []
    assert captured["project"] == TASK_MACHINE_STATE_PROJECT
    assert captured["context"] == "Tasks/Activations"


def test_tasks_store_defaults_to_active_project(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_get_logs(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("unity.task_scheduler.storage.unify.get_logs", _fake_get_logs)
    monkeypatch.setattr(
        "unity.task_scheduler.storage.unify.active_project",
        lambda: "Assistants",
    )

    store = TasksStore("42/7/Tasks")
    store.get_rows(limit=5)

    assert captured["project"] == "Assistants"


def test_tasks_store_log_passes_explicit_project_to_helper(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_log(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id=1)

    monkeypatch.setattr("unity.task_scheduler.storage.unity_log", _fake_log)

    store = TasksStore("42/7/Tasks", project=TASK_MACHINE_STATE_PROJECT)
    result = store.log(entries={"task_id": 101})

    assert result.id == 1
    assert captured["project"] == TASK_MACHINE_STATE_PROJECT
    assert captured["context"] == "42/7/Tasks"
