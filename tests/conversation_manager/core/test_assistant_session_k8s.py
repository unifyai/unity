from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from unity.conversation_manager import (
    assistant_session_k8s as assistant_session_k8s_module,
)


def _fake_job(*, labels: dict | None = None, annotations: dict | None = None):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            labels=labels or {},
            annotations=annotations or {},
        ),
    )


def _fake_secret(
    *,
    payload: dict,
    name: str = "assistant-session-bootstrap-42-activation-42",
    annotations: dict | None = None,
):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name=name,
            annotations=annotations or {},
        ),
        data={
            "startup.json": base64.b64encode(
                json.dumps(payload).encode("utf-8"),
            ).decode("utf-8"),
        },
    )


def test_read_job_assignment_record_uses_controller_binding_metadata(monkeypatch):
    batch_api = MagicMock()
    batch_api.read_namespaced_job.return_value = _fake_job(
        annotations={
            assistant_session_k8s_module.SESSION_REF_ANNOTATION: "assistant-session-42",
            assistant_session_k8s_module.BINDING_ID_ANNOTATION: "binding-42",
        },
    )

    monkeypatch.setattr(assistant_session_k8s_module, "_load_clients", lambda: None)
    monkeypatch.setattr(assistant_session_k8s_module, "_namespace", lambda: "preview")
    monkeypatch.setattr(assistant_session_k8s_module, "_batch_api", batch_api)

    record = assistant_session_k8s_module.read_job_assignment_record("unity-job-42")

    assert record == assistant_session_k8s_module.JobAssignmentRecord(
        session_name="assistant-session-42",
        binding_id="binding-42",
    )
    batch_api.read_namespaced_job.assert_called_once_with(
        name="unity-job-42",
        namespace="preview",
    )


def test_read_session_bootstrap_secret_record_returns_owner_annotations(monkeypatch):
    payload = {"assistant_id": "42", "api_key": "user-key"}
    core_api = MagicMock()
    core_api.read_namespaced_secret.return_value = _fake_secret(
        payload=payload,
        annotations={
            assistant_session_k8s_module.SESSION_REF_ANNOTATION: "assistant-session-42",
            assistant_session_k8s_module.ACTIVATION_ID_ANNOTATION: "activation-42",
        },
    )

    monkeypatch.setattr(assistant_session_k8s_module, "_load_clients", lambda: None)
    monkeypatch.setattr(assistant_session_k8s_module, "_namespace", lambda: "preview")
    monkeypatch.setattr(assistant_session_k8s_module, "_core_api", core_api)

    record = assistant_session_k8s_module.read_session_bootstrap_secret_record(
        "assistant-session-bootstrap-42-activation-42",
    )

    assert record == assistant_session_k8s_module.BootstrapSecretRecord(
        name="assistant-session-bootstrap-42-activation-42",
        payload=payload,
        owner_session_name="assistant-session-42",
        owner_activation_id="activation-42",
    )
    core_api.read_namespaced_secret.assert_called_once_with(
        name="assistant-session-bootstrap-42-activation-42",
        namespace="preview",
    )
