"""
Configuration management for the ConversationManager sandbox.

This module is sandbox-only. It persists and restores the user's last-selected
actor configuration via a project-local gitignored file: `.cm_sandbox_config`.

Design goals:
- **Project-local**: configuration should not bleed across repos/worktrees.
- **Minimal dependencies**: JSON via stdlib only.
- **Robust to corruption**: invalid files fall back to safe defaults.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import unify

from sandboxes.conversation_manager.agent_service_bootstrap import (
    diagnose_agent_service_setup,
)

LG = logging.getLogger("conversation_manager_sandbox")

ActorType = Literal["simulated", "codeact_simulated", "codeact_real"]


@dataclass(frozen=True)
class ActorConfig:
    """User-facing sandbox configuration for actor + manager + computer backend mode."""

    actor_type: ActorType = "simulated"
    version: int = 1
    last_updated: Optional[str] = None

    @property
    def managers_mode(self) -> Literal["simulated", "real"]:
        return "real" if self.actor_type == "codeact_real" else "simulated"

    @property
    def computer_backend_mode(self) -> Literal["none", "mock", "real"]:
        if self.actor_type == "simulated":
            return "none"
        if self.actor_type == "codeact_simulated":
            return "mock"
        return "real"

    def to_json_obj(self) -> dict:
        return {
            "actor_type": self.actor_type,
            "last_updated": self.last_updated,
            "version": int(self.version),
        }

    @classmethod
    def from_json_obj(cls, obj: dict) -> "ActorConfig":
        actor_type = obj.get("actor_type") or "simulated"
        if actor_type not in {"simulated", "codeact_simulated", "codeact_real"}:
            actor_type = "simulated"
        version = int(obj.get("version") or 1)
        last_updated = obj.get("last_updated")
        return cls(
            actor_type=actor_type,  # type: ignore[arg-type]
            version=version,
            last_updated=last_updated if isinstance(last_updated, str) else None,
        )


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    error: Optional[str] = None
    failed_component: Optional[str] = None
    help_text: Optional[str] = None


@dataclass(frozen=True)
class StateSnapshot:
    """Rollback handle for sandbox config switching."""

    project_name: str
    commit_hash: str
    created_at: float


class ConfigurationManager:
    """
    Load/save/validate sandbox configuration and snapshot/restore state.

    Notes
    -----
    - Snapshot/restore uses Unify project commits as the durable rollback surface.
    - This module does not depend on any UI; callers decide how to render prompts.
    """

    def __init__(
        self,
        *,
        project_name: str,
        project_root: Path,
        filename: str = ".cm_sandbox_config",
    ) -> None:
        self._project_name = str(project_name)
        self._project_root = Path(project_root)
        self._path = Path(project_root) / filename

    @property
    def path(self) -> Path:
        return self._path

    def load_config(self) -> ActorConfig:
        """Load the last-used configuration from disk, or return safe defaults."""
        try:
            if not self._path.exists():
                LG.info(
                    "[config] No saved configuration found. Using default: SandboxSimulatedActor",
                )
                return ActorConfig(actor_type="simulated", version=1, last_updated=None)
            raw = self._path.read_text(encoding="utf-8")
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError("config is not an object")
            return ActorConfig.from_json_obj(obj)
        except Exception as exc:
            LG.warning(
                "[config] Configuration file corrupted, using default (%s)",
                exc,
            )
            return ActorConfig(actor_type="simulated", version=1, last_updated=None)

    def save_config(self, config: ActorConfig) -> None:
        """Persist configuration to disk (atomic best-effort write)."""
        parent = self._path.parent
        parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        data = {
            **config.to_json_obj(),
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "version": int(config.version),
        }
        tmp.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self._path)

    def validate_config(
        self,
        config: ActorConfig,
        *,
        agent_server_url: str = "http://localhost:3000",
    ) -> ValidationResult:
        """
        Validate that required infrastructure for the selected config is available.

        This is intentionally conservative: we only validate hard prerequisites.
        """
        if config.actor_type == "codeact_real":
            if not os.environ.get("UNIFY_KEY"):
                return ValidationResult(
                    ok=False,
                    failed_component="Real Computer Interface",
                    error="UNIFY_KEY is not set (required for agent-service authentication)",
                    help_text=diagnose_agent_service_setup(
                        repo_root=self._project_root,
                        agent_server_url=agent_server_url,
                    ).help_text,
                )
            # Real computer interface requires agent-service.
            ok = self._validate_agent_service(agent_server_url)
            if not ok:
                diag = diagnose_agent_service_setup(
                    repo_root=self._project_root,
                    agent_server_url=agent_server_url,
                )
                return ValidationResult(
                    ok=False,
                    failed_component="Real Computer Interface",
                    error=diag.summary,
                    help_text=diag.help_text,
                )

            # Real managers require project connectivity. This is a lightweight probe; it
            # will fail early if the backend session is misconfigured.
            try:
                _ = unify.get_project_commits(self._project_name)
            except Exception as exc:
                return ValidationResult(
                    ok=False,
                    failed_component="Real Managers",
                    error=f"Failed to access Unify project: {type(exc).__name__}: {exc}",
                )

        return ValidationResult(ok=True)

    def snapshot_state(self) -> StateSnapshot:
        """Create a project snapshot (commit) for rollback during config switching."""
        created_at = time.time()
        commit = unify.commit_project(
            self._project_name,
            commit_message=(
                "ConversationManager sandbox auto-snapshot "
                f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(created_at))}"
            ),
        )
        commit_hash = str(commit.get("commit_hash") or "")
        if not commit_hash:
            raise RuntimeError("commit_project returned no commit_hash")
        return StateSnapshot(
            project_name=self._project_name,
            commit_hash=commit_hash,
            created_at=created_at,
        )

    def restore_snapshot(self, snapshot: StateSnapshot) -> None:
        """Rollback the project to a prior snapshot (commit hash)."""
        if snapshot.project_name != self._project_name:
            raise ValueError("snapshot project_name does not match current project")
        unify.rollback_project(self._project_name, snapshot.commit_hash)

    def _validate_agent_service(self, agent_server_url: str) -> bool:
        # Avoid introducing a hard dependency on `requests`; prefer httpx if available.
        try:
            import httpx  # type: ignore
        except Exception:
            httpx = None  # type: ignore

        # agent-service requires auth; use UNIFY_KEY if available.
        auth_key = os.environ.get("UNIFY_KEY")
        if not auth_key:
            return False

        url = str(agent_server_url).rstrip("/") + "/sessions"
        try:
            if httpx is None:
                # Fall back to stdlib if httpx isn't installed.
                from urllib.request import Request
                from urllib.request import urlopen

                req = Request(url, headers={"authorization": f"Bearer {auth_key}"})
                with urlopen(req, timeout=5.0) as resp:  # nosec B310
                    return int(getattr(resp, "status", 0) or 0) == 200

            resp = httpx.get(
                url,
                timeout=5.0,
                headers={"authorization": f"Bearer {auth_key}"},
            )
            return int(resp.status_code) == 200
        except Exception:
            return False
