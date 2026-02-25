from __future__ import annotations

import functools
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BaseEnvironmentManager
from .types.environment import Environment
from .environment_manager import (
    _ensure_dependencies,
    _parse_env_path,
    _write_files_to_package,
    _import_and_resolve,
)

if TYPE_CHECKING:
    from ...actor.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)


class SimulatedEnvironmentManager(BaseEnvironmentManager):
    """In-memory EnvironmentManager for testing and simulated sandboxes."""

    def __init__(
        self,
        description: str = "simulated environment manager",
        *,
        simulation_guidance: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._description = description
        self._simulation_guidance = simulation_guidance
        self._entries: Dict[int, Environment] = {}
        self._next_id: int = 1

    @functools.wraps(BaseEnvironmentManager.upload_environment, updated=())
    def upload_environment(
        self,
        *,
        name: str,
        files: Dict[str, str],
        dependencies: Optional[List[str]] = None,
        env: str,
    ) -> int:
        if ":" not in env:
            raise ValueError(
                f"Invalid env path {env!r}: must be 'module:attribute'.",
            )
        eid = self._next_id
        self._next_id += 1
        self._entries[eid] = Environment(
            environment_id=eid,
            name=name,
            env=env,
            files=files,
            dependencies=dependencies or [],
        )
        return eid

    @functools.wraps(BaseEnvironmentManager.list_environments, updated=())
    def list_environments(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[dict]:
        rows = list(self._entries.values())
        if filter is not None:
            matched = []
            for e in rows:
                try:
                    if eval(filter, {"__builtins__": {}}, e.model_dump()):
                        matched.append(e)
                except Exception:
                    pass
            rows = matched
        return [e.model_dump() for e in rows[offset : offset + limit]]

    @functools.wraps(BaseEnvironmentManager.delete_environment, updated=())
    def delete_environment(self, *, environment_id: int) -> dict:
        if environment_id not in self._entries:
            raise ValueError(
                f"No environment found with environment_id {environment_id}.",
            )
        del self._entries[environment_id]
        return {
            "outcome": "environment deleted",
            "details": {"environment_id": environment_id},
        }

    @functools.wraps(BaseEnvironmentManager.load_environment, updated=())
    def load_environment(self, environment_id: int) -> "BaseEnvironment":
        pass

        if environment_id not in self._entries:
            raise ValueError(
                f"No environment found with environment_id {environment_id}.",
            )
        env_def = self._entries[environment_id]
        return self._reconstruct(env_def)

    @functools.wraps(BaseEnvironmentManager.load_all_environments, updated=())
    def load_all_environments(self) -> List["BaseEnvironment"]:
        envs: List["BaseEnvironment"] = []
        for env_def in self._entries.values():
            try:
                envs.append(self._reconstruct(env_def))
            except Exception:
                logger.exception(
                    f"Failed to load environment {env_def.name!r} "
                    f"(id={env_def.environment_id}), skipping.",
                )
        return envs

    def _reconstruct(self, env_def: Environment) -> "BaseEnvironment":
        from ...actor.environments.base import BaseEnvironment

        if env_def.dependencies:
            _ensure_dependencies(env_def.dependencies)

        pkg_dir = _write_files_to_package(
            environment_id=env_def.environment_id,
            files=env_def.files,
        )

        module_name, attr_name = _parse_env_path(env_def.env)
        instance = _import_and_resolve(
            pkg_dir=pkg_dir,
            module_name=module_name,
            attr_name=attr_name,
        )

        if not isinstance(instance, BaseEnvironment):
            raise TypeError(
                f"Resolved object from {env_def.env!r} is {type(instance).__name__}, "
                f"not a BaseEnvironment subclass.",
            )

        return instance

    @functools.wraps(BaseEnvironmentManager.clear, updated=())
    def clear(self) -> None:
        self._entries.clear()
        self._next_id = 1
