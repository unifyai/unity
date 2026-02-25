from __future__ import annotations

import functools
import importlib
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import unify

from ...common.log_utils import log as unity_log
from ...common.model_to_fields import model_to_fields
from ...common.context_store import TableStore
from ...common.filter_utils import normalize_filter_expr
from ...common.context_registry import TableContext, ContextRegistry
from .base import BaseEnvironmentManager
from .types.environment import Environment

if TYPE_CHECKING:
    from ...actor.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)

# Default root for extracted environment packages
_DEFAULT_ENV_ROOT = Path.home() / ".unity" / "environments"


def _pkg_name_from_specifier(spec: str) -> str:
    """Extract the importable package name from a pip specifier.

    Handles simple cases like ``"openpyxl>=3.1"`` → ``"openpyxl"``.
    For packages where pip name != import name, this will fail gracefully
    and the dependency will be installed unconditionally.
    """
    return re.split(r"[><=!~\[;]", spec)[0].strip().replace("-", "_")


class EnvironmentManager(BaseEnvironmentManager):
    """Concrete EnvironmentManager backed by Unify contexts."""

    class Config:
        required_contexts = [
            TableContext(
                name="Environments/Packages",
                description="Stored custom environment packages for actor construction.",
                fields=model_to_fields(Environment),
                unique_keys={"environment_id": "int"},
                auto_counting={"environment_id": None},
            ),
        ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.include_in_multi_assistant_table = True
        self._ctx = ContextRegistry.get_context(self, "Environments/Packages")
        self._BUILTIN_FIELDS: Tuple[str, ...] = tuple(Environment.model_fields.keys())

        self._provision_storage()

    def _provision_storage(self) -> None:
        self._store = TableStore(
            self._ctx,
            unique_keys={"environment_id": "int"},
            auto_counting={"environment_id": None},
            description="Stored custom environment packages for actor construction.",
            fields=model_to_fields(Environment),
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

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
                f"Invalid env path {env!r}: must be 'module:attribute' "
                f"(e.g. 'my_env:my_env').",
            )

        e = Environment(
            name=name,
            env=env,
            files=files,
            dependencies=dependencies or [],
        )
        payload = e.to_post_json()
        log = unity_log(
            context=self._ctx,
            **payload,
            new=True,
            mutable=True,
            add_to_all_context=self.include_in_multi_assistant_table,
        )
        env_id = log.entries["environment_id"]
        logger.info(f"Uploaded environment {name!r} (id={env_id})")
        return env_id

    @functools.wraps(BaseEnvironmentManager.list_environments, updated=())
    def list_environments(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[dict]:
        from_fields = list(self._BUILTIN_FIELDS)
        normalized = normalize_filter_expr(filter)
        logs = unify.get_logs(
            context=self._ctx,
            filter=normalized,
            offset=offset,
            limit=limit,
            from_fields=from_fields,
        )
        return [lg.entries for lg in logs]

    @functools.wraps(BaseEnvironmentManager.delete_environment, updated=())
    def delete_environment(self, *, environment_id: int) -> dict:
        ids = unify.get_logs(
            context=self._ctx,
            filter=f"environment_id == {int(environment_id)}",
            limit=2,
            return_ids_only=True,
        )
        if not ids:
            raise ValueError(
                f"No environment found with environment_id {environment_id}.",
            )
        if len(ids) > 1:
            raise RuntimeError(
                f"Multiple rows found with environment_id {environment_id}. "
                "Data integrity issue.",
            )
        unify.delete_logs(context=self._ctx, logs=ids[0])
        return {
            "outcome": "environment deleted",
            "details": {"environment_id": environment_id},
        }

    @functools.wraps(BaseEnvironmentManager.clear, updated=())
    def clear(self) -> None:
        unify.delete_context(self._ctx)
        ContextRegistry.refresh(self, "Environments/Packages")
        self._provision_storage()

    # ------------------------------------------------------------------
    # Loading / reconstruction
    # ------------------------------------------------------------------

    @functools.wraps(BaseEnvironmentManager.load_environment, updated=())
    def load_environment(self, environment_id: int) -> "BaseEnvironment":
        rows = unify.get_logs(
            context=self._ctx,
            filter=f"environment_id == {int(environment_id)}",
            limit=1,
            from_fields=list(self._BUILTIN_FIELDS),
        )
        if not rows:
            raise ValueError(
                f"No environment found with environment_id {environment_id}.",
            )
        env_def = Environment(**rows[0].entries)
        return self._reconstruct(env_def)

    @functools.wraps(BaseEnvironmentManager.load_all_environments, updated=())
    def load_all_environments(self) -> List["BaseEnvironment"]:
        logs = unify.get_logs(
            context=self._ctx,
            limit=1000,
            from_fields=list(self._BUILTIN_FIELDS),
        )
        envs: List["BaseEnvironment"] = []
        for lg in logs:
            try:
                env_def = Environment(**lg.entries)
                envs.append(self._reconstruct(env_def))
            except Exception:
                logger.exception(
                    f"Failed to load environment {lg.entries.get('name', '?')!r} "
                    f"(id={lg.entries.get('environment_id', '?')}), skipping.",
                )
        return envs

    def _reconstruct(self, env_def: Environment) -> "BaseEnvironment":
        """Install deps, write files, import module, resolve instance."""
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


# ------------------------------------------------------------------
# Module-level helpers (stateless, testable independently)
# ------------------------------------------------------------------


def _parse_env_path(env: str) -> Tuple[str, str]:
    """Split ``"module:attribute"`` into ``(module_name, attr_name)``."""
    if ":" not in env:
        raise ValueError(
            f"Invalid env path {env!r}: must be 'module:attribute'.",
        )
    module_name, attr_name = env.split(":", 1)
    if not module_name or not attr_name:
        raise ValueError(
            f"Invalid env path {env!r}: both module and attribute must be non-empty.",
        )
    return module_name, attr_name


def _ensure_dependencies(dependencies: List[str]) -> None:
    """Pip-install any missing dependencies into the host environment."""
    missing: List[str] = []
    for dep in dependencies:
        pkg = _pkg_name_from_specifier(dep)
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(dep)

    if not missing:
        return

    logger.info(f"Installing missing environment dependencies: {missing}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", *missing],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    # Invalidate import caches so freshly installed packages are discoverable
    importlib.invalidate_caches()


def _write_files_to_package(
    *,
    environment_id: int,
    files: Dict[str, str],
    root: Optional[Path] = None,
) -> Path:
    """Write source files to a deterministic directory and return it.

    The directory is ``~/.unity/environments/{environment_id}/``.
    Files are only rewritten if their content has changed.
    """
    pkg_dir = (root or _DEFAULT_ENV_ROOT) / str(environment_id)
    pkg_dir.mkdir(parents=True, exist_ok=True)

    for filename, source in files.items():
        filepath = pkg_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if filepath.exists():
            try:
                existing = filepath.read_text(encoding="utf-8")
            except Exception:
                pass
        if existing != source:
            filepath.write_text(source, encoding="utf-8")

    return pkg_dir


def _import_and_resolve(
    *,
    pkg_dir: Path,
    module_name: str,
    attr_name: str,
) -> Any:
    """Import a module from ``pkg_dir`` and resolve ``attr_name`` on it."""
    pkg_str = str(pkg_dir)
    if pkg_str not in sys.path:
        sys.path.insert(0, pkg_str)

    # Force re-import if module was previously loaded (e.g., after file update)
    if module_name in sys.modules:
        del sys.modules[module_name]

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Could not import module {module_name!r} from {pkg_dir}. "
            f"Available files: {[f.name for f in pkg_dir.iterdir()]}.",
        ) from exc

    if not hasattr(module, attr_name):
        raise AttributeError(
            f"Module {module_name!r} has no attribute {attr_name!r}. "
            f"Available attributes: {[a for a in dir(module) if not a.startswith('_')]}.",
        )

    return getattr(module, attr_name)
