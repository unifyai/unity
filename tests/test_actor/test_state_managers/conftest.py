from __future__ import annotations

import os
import re

from typing import Any, Literal

import pytest

from unity.actor.hierarchical_actor import VerificationAssessment
from unity.manager_registry import ManagerRegistry


def _in_tree(request: pytest.FixtureRequest, segment: str) -> bool:
    try:
        p = str(getattr(request.node, "fspath", "") or "")
    except Exception:
        p = ""
    return segment in p


def _apply_impl_overrides(
    monkeypatch: pytest.MonkeyPatch,
    *,
    impl: Literal["real", "simulated"],
) -> None:
    """Apply IMPL overrides for all state managers used by Actor primitives."""
    # Env vars (documented contract): each manager settings uses prefix UNITY_<X>_.
    monkeypatch.setenv("UNITY_CONTACT_IMPL", impl)
    monkeypatch.setenv("UNITY_TASK_IMPL", impl)
    monkeypatch.setenv("UNITY_TRANSCRIPT_IMPL", impl)
    monkeypatch.setenv("UNITY_KNOWLEDGE_IMPL", impl)
    monkeypatch.setenv("UNITY_GUIDANCE_IMPL", impl)
    monkeypatch.setenv("UNITY_SECRET_IMPL", impl)
    monkeypatch.setenv("UNITY_WEB_IMPL", impl)
    monkeypatch.setenv("UNITY_FILE_IMPL", impl)

    # Optional managers are disabled by default; enable them for simulated manager tests.
    # This keeps routing tests meaningful (the Actor can actually call these tools).
    monkeypatch.setenv("UNITY_FILE_ENABLED", "true")
    monkeypatch.setenv("UNITY_GUIDANCE_ENABLED", "true")
    monkeypatch.setenv("UNITY_WEB_ENABLED", "true")
    monkeypatch.setenv("UNITY_KNOWLEDGE_ENABLED", "true")

    # Also update the already-instantiated SETTINGS singleton so ManagerRegistry's
    # settings accessors (lambda: SETTINGS.<x>) see the new IMPL values.
    from unity.settings import SETTINGS

    monkeypatch.setattr(SETTINGS.contact, "IMPL", impl, raising=False)
    monkeypatch.setattr(SETTINGS.task, "IMPL", impl, raising=False)
    monkeypatch.setattr(SETTINGS.transcript, "IMPL", impl, raising=False)
    monkeypatch.setattr(SETTINGS.knowledge, "IMPL", impl, raising=False)
    monkeypatch.setattr(SETTINGS.guidance, "IMPL", impl, raising=False)
    monkeypatch.setattr(SETTINGS.secret, "IMPL", impl, raising=False)
    monkeypatch.setattr(SETTINGS.web, "IMPL", impl, raising=False)
    monkeypatch.setattr(SETTINGS.file, "IMPL", impl, raising=False)

    monkeypatch.setattr(SETTINGS.file, "ENABLED", True, raising=False)
    monkeypatch.setattr(SETTINGS.guidance, "ENABLED", True, raising=False)
    monkeypatch.setattr(SETTINGS.web, "ENABLED", True, raising=False)
    monkeypatch.setattr(SETTINGS.knowledge, "ENABLED", True, raising=False)

    # Ensure subsequent ManagerRegistry.get_* returns fresh instances per test.
    ManagerRegistry.clear()


@pytest.fixture(autouse=True)
def configure_simulated_managers(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Autouse fixture for tests under test_simulated/: force simulated managers."""
    if not _in_tree(request, os.path.join("test_state_managers", "test_simulated")):
        return
    _apply_impl_overrides(monkeypatch, impl="simulated")


@pytest.fixture(autouse=True)
def configure_real_managers(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Autouse fixture for tests under test_real/: force real managers + stub network."""
    if not _in_tree(request, os.path.join("test_state_managers", "test_real")):
        return

    # Prevent network access during manager initialization (mirrors Conductor real tests).
    import unity

    monkeypatch.setattr(unity, "ASSISTANT", None, raising=False)
    monkeypatch.setattr(unity, "_list_all_assistants", lambda: [], raising=False)

    _apply_impl_overrides(monkeypatch, impl="real")


@pytest.fixture
def mock_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch verification to always return success."""

    async def _mock_check_state(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> VerificationAssessment:
        return VerificationAssessment(status="ok", reason="Mock verification success")

    monkeypatch.setattr(
        "unity.actor.hierarchical_actor.HierarchicalActor._check_state_against_goal",
        _mock_check_state,
        raising=True,
    )
