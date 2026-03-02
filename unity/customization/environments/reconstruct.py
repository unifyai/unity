"""
Standalone helpers for reconstructing live ``BaseEnvironment`` instances
from serialized ``Environment`` definitions (source files + dependencies).
"""

from __future__ import annotations

import importlib
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from unity.common.hierarchical_logger import ICONS

from .types.environment import Environment

if TYPE_CHECKING:
    from ...actor.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)
_ICON = ICONS["customization"]

_DEFAULT_ENV_ROOT = Path.home() / ".unity" / "environments"


def _pkg_name_from_specifier(spec: str) -> str:
    """Extract the importable package name from a pip specifier.

    Handles simple cases like ``"openpyxl>=3.1"`` -> ``"openpyxl"``.
    For packages where pip name != import name, this will fail gracefully
    and the dependency will be installed unconditionally.
    """
    return re.split(r"[><=!~\[;]", spec)[0].strip().replace("-", "_")


def parse_env_path(env: str) -> Tuple[str, str]:
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


def ensure_dependencies(dependencies: List[str]) -> None:
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

    logger.info(f"{_ICON} Installing missing environment dependencies: {missing}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", *missing],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    importlib.invalidate_caches()


def write_files_to_package(
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


def import_and_resolve(
    *,
    pkg_dir: Path,
    module_name: str,
    attr_name: str,
) -> Any:
    """Import a module from ``pkg_dir`` and resolve ``attr_name`` on it."""
    pkg_str = str(pkg_dir)
    if pkg_str not in sys.path:
        sys.path.insert(0, pkg_str)

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


def reconstruct(env_def: Environment) -> "BaseEnvironment":
    """Reconstruct a live ``BaseEnvironment`` from a serialized definition.

    Ensures dependencies are installed, writes source files to a temp
    package directory, imports the entry module, and resolves the
    ``module:attribute`` path to a ``BaseEnvironment`` instance.
    """
    from unity.actor.environments.base import BaseEnvironment

    if env_def.dependencies:
        ensure_dependencies(env_def.dependencies)

    pkg_dir = write_files_to_package(
        environment_id=env_def.environment_id,
        files=env_def.files,
    )

    module_name, attr_name = parse_env_path(env_def.env)
    instance = import_and_resolve(
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
