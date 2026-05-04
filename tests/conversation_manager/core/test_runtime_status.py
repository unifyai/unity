from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from unity.conversation_manager.prompt_builders import build_system_prompt
from unity.conversation_manager.runtime_status import (
    deployment_runtime_reconcile_prompt_note,
)

pytestmark = pytest.mark.no_unify_context


@dataclass(frozen=True)
class _Status:
    current_phase: str


class _StatusHandle:
    def __init__(self, phase: str) -> None:
        self._phase = phase

    def snapshot(self) -> _Status:
        return _Status(current_phase=self._phase)


def test_runtime_status_note_is_user_facing_while_syncing():
    cm = SimpleNamespace(
        deployment_runtime_reconcile_status=_StatusHandle("syncing_seed_data"),
    )

    note = deployment_runtime_reconcile_prompt_note(cm)

    assert note is not None
    assert "setup is still finishing" in note
    assert "runtime reconciliation" not in note.lower()


def test_runtime_status_note_surfaces_failure_without_internal_jargon():
    cm = SimpleNamespace(deployment_runtime_reconcile_status=_Status("failed"))

    note = deployment_runtime_reconcile_prompt_note(cm)

    assert note is not None
    assert "setup failed" in note
    assert "runtime reconciliation" not in note.lower()


def test_system_prompt_includes_runtime_setup_note_when_present():
    prompt = build_system_prompt(
        bio="A helpful assistant.",
        contact_id=1,
        first_name="Alice",
        surname="Smith",
        runtime_setup_note="Some setup is still finishing.",
    ).flatten()

    assert "Setup readiness" in prompt
    assert "Some setup is still finishing." in prompt
