"""Prompt helpers for optional background runtime setup status."""

from __future__ import annotations

from typing import Any

INCOMPLETE_PHASES = {
    "not_started",
    "starting",
    "syncing_seed_data",
    "syncing_custom_functions",
}

INCOMPLETE_SETUP_NOTE = (
    "Some assistant setup is still finishing in the background. "
    "Deployment-defined data, guidance, secrets, or custom tools may still be "
    "syncing. I can answer general questions normally. If my boss asks for "
    "information that depends on not-yet-ready data or tools, I should be "
    "transparent: say that those systems are still being prepared, avoid "
    "claiming fresh results, and offer to continue once setup completes."
)

FAILED_SETUP_NOTE = (
    "Background assistant setup failed. Some deployment-defined data or custom "
    "tools may be unavailable. If my boss asks about affected capabilities, I "
    "should explain that setup hit an error and avoid pretending unavailable "
    "data or tools are ready."
)


def _snapshot_status(status: Any) -> Any:
    snapshot = getattr(status, "snapshot", None)
    if callable(snapshot):
        return snapshot()
    return status


def deployment_runtime_reconcile_prompt_note(cm: Any) -> str | None:
    """Translate optional runtime setup status into system-prompt guidance."""

    status = _snapshot_status(
        getattr(cm, "deployment_runtime_reconcile_status", None),
    )
    if status is None:
        return None

    phase = getattr(status, "current_phase", None) or getattr(status, "phase", None)
    if phase == "failed":
        return FAILED_SETUP_NOTE
    if phase in INCOMPLETE_PHASES:
        return INCOMPLETE_SETUP_NOTE
    return None
