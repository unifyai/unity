import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "deploy"
    / "scripts"
    / "stress_test"
    / "cleanup_stale_jobs.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "cleanup_stale_jobs_script",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cleanup_jobs_requests_binding_scoped_release(monkeypatch):
    module = _load_script_module()
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        response = MagicMock()
        response.ok = True
        response.raise_for_status.return_value = None
        response.json.return_value = {"released": True}
        return response

    monkeypatch.setattr(module.requests, "post", fake_post)
    monkeypatch.setattr(module, "comms_url", "https://comms.test")
    monkeypatch.setattr(module, "admin_key", "secret")
    monkeypatch.setattr(module, "namespace", "preview")

    module.cleanup_jobs(
        [
            {
                "job_name": "unity-job-1",
                "assistant_id": "assistant-123",
                "labels": {module.BINDING_ID_LABEL: "binding-123"},
            },
        ],
    )

    assert len(calls) == 2
    assert calls[0][0] == "https://comms.test/infra/job/stop"
    assert calls[1][0] == "https://comms.test/infra/vm/pool/release"
    assert calls[1][1]["json"] == {
        "assistant_id": "assistant-123",
        "binding_id": "binding-123",
        "job_name": "unity-job-1",
    }


def test_cleanup_jobs_skips_release_without_binding_id(monkeypatch):
    module = _load_script_module()
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        response = MagicMock()
        response.ok = True
        response.raise_for_status.return_value = None
        response.json.return_value = {"released": True}
        return response

    monkeypatch.setattr(module.requests, "post", fake_post)
    monkeypatch.setattr(module, "comms_url", "https://comms.test")
    monkeypatch.setattr(module, "admin_key", "secret")
    monkeypatch.setattr(module, "namespace", "preview")

    module.cleanup_jobs(
        [
            {
                "job_name": "unity-job-1",
                "assistant_id": "assistant-123",
                "labels": {},
            },
        ],
    )

    assert len(calls) == 1
    assert calls[0][0] == "https://comms.test/infra/job/stop"
