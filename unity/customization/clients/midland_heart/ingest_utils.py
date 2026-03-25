"""Shared utilities for Midland Heart ingestion scripts.

Provides environment bootstrapping, project activation, config loading,
and progress reporter creation used by both ``ingest_fm.py`` and
``ingest_dm.py``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from unity.customization.types.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Resolve the project root (4 levels up from this file)."""
    return Path(__file__).resolve().parents[4]


_SDK_NOISE_PREFIXES = (
    "unify_requests",
    "unify",
    "unillm",
    "UnifyAsyncLogger",
    "httpx",
    "httpcore",
    "urllib3",
)


def initialize_environment(
    *,
    debug: bool = False,
    log_file: Optional[str] = None,
    sdk_log: bool = True,
) -> Path:
    """Load ``.env``, configure logging, and add project root to ``sys.path``.

    Logging behaviour mirrors the sandbox convention:

    * **Terminal** -- app-level logs only; SDK/HTTP noise from ``unify``,
      ``unillm``, ``httpx`` etc. is suppressed (only WARNING+ passes).
    * **Log file** (optional) -- receives everything at the configured level.
      SDK noise is routed to a companion ``*_unify.log`` file when a main
      log file is provided and *sdk_log* is True.

    Parameters
    ----------
    debug : bool
        If True, set root level to DEBUG; otherwise INFO.
    log_file : str | None
        Optional path to a main log file.  SDK noise is written to a sibling
        ``<stem>_unify.log`` file when provided.
    sdk_log : bool
        If False, skip creating the ``*_unify.log`` companion file entirely.
        SDK noise is still suppressed from the terminal and main log file.

    Returns the resolved project root path.
    """
    project_root = get_project_root()

    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    env_file = project_root / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(str(env_file), override=False)
        except ImportError:
            pass

    level = logging.DEBUG if debug else logging.INFO

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    class _SuppressSDKNoise(logging.Filter):
        """Allow SDK loggers through only at WARNING+."""

        def filter(self, record: logging.LogRecord) -> bool:
            name = record.name or ""
            if any(name.startswith(p) for p in _SDK_NOISE_PREFIXES):
                return record.levelno >= logging.WARNING
            return True

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.addFilter(_SuppressSDKNoise())
    root_logger.addHandler(console)

    if log_file:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.addFilter(_SuppressSDKNoise())
        root_logger.addHandler(fh)

        if sdk_log:
            stem = Path(log_file).stem
            parent = Path(log_file).parent
            unify_log = str(parent / f"{stem}_unify.log")
            fh_sdk = logging.FileHandler(unify_log, mode="w", encoding="utf-8")
            fh_sdk.setFormatter(fmt)
            root_logger.addHandler(fh_sdk)
            print(f"  Unify/SDK logs -> {unify_log}")

    return project_root


def activate_project(
    project_name: str = "MidlandHeart",
    *,
    overwrite: bool = False,
) -> None:
    """Activate a Unify project for ingestion.

    Calls ``unity.init()`` and optionally deletes/recreates the project.
    """
    if overwrite:
        try:
            from unify import delete_project

            delete_project(project_name)
            logger.info("Deleted existing project '%s'", project_name)
        except Exception:
            pass

    os.environ.setdefault("UNIFY_PROJECT_NAME", project_name)

    from unity.session_details import SESSION_DETAILS

    SESSION_DETAILS.populate_from_env()

    import unity

    unity.init(project_name=project_name, overwrite=overwrite)
    logger.info(
        "Activated project '%s' (context: %s/%s)",
        project_name,
        SESSION_DETAILS.user_context,
        SESSION_DETAILS.assistant_context,
    )


def load_pipeline_config(
    config_path: Optional[str] = None,
    *,
    project_root: Optional[Path] = None,
) -> "PipelineConfig":
    """Load, validate, and resolve the shared pipeline JSON config.

    Parameters
    ----------
    config_path : str | None
        Path to the JSON file. If None, defaults to ``pipeline_config.json``
        in the same directory as this module.
    project_root : Path | None
        Project root for resolving relative ``file_path`` entries.
        Defaults to ``get_project_root()``.

    Returns
    -------
    PipelineConfig
        A validated, path-resolved configuration object.
    """
    from unity.customization.types.pipeline_config import PipelineConfig

    if config_path is None:
        config_path = str(Path(__file__).parent / "pipeline_config.json")

    config = PipelineConfig.from_file(config_path)

    root = project_root or get_project_root()
    config.resolve_paths(root)

    logger.info("Loaded pipeline config from %s", config_path)
    return config


def create_run_directory(
    script_name: str,
    *,
    base_dir: Optional[str] = None,
) -> Path:
    """Create and return a timestamped run directory.

    Layout::

        <base_dir>/<script_name>/<YYYY-MM-DDTHH-MM-SS>/
            progress.jsonl      (created by reporter)
            error_details/      (created on first error)

    Parameters
    ----------
    script_name : str
        Short name (e.g. ``"ingest_dm"``). Becomes the parent folder name.
    base_dir : str | None
        Root for all run directories. Defaults to ``./logs/pipeline/``
        relative to the current working directory.

    Returns
    -------
    Path
        The created run directory (guaranteed to exist).
    """
    import time as _time

    root = Path(base_dir) if base_dir else Path("logs") / "pipeline"
    ts = _time.strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = root / script_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_pipeline_reporter(
    diagnostics: Dict[str, Any],
    script_name: str,
    *,
    run_dir: Optional[Path] = None,
    progress_file_override: Optional[str] = None,
    verbosity_override: Optional[str] = None,
):
    """Create a composite ProgressReporter from the diagnostics config section.

    All artefacts for this run (progress JSONL, error detail files, optional
    log files) live under a single timestamped directory::

        <run_dir>/
            progress.jsonl
            error_details/
                <phase>_0001.json
                ...

    Returns a ``CompositeReporter`` that writes to both console and JSONL
    file when progress is enabled, or a ``NoOpReporter`` when disabled.

    Parameters
    ----------
    diagnostics : dict
        The ``diagnostics`` section from pipeline_config.json.
    script_name : str
        Used when auto-creating the run directory.
    run_dir : Path | None
        Pre-created run directory.  If ``None``, one is created via
        :func:`create_run_directory`.
    progress_file_override : str | None
        CLI override that places the progress file at an explicit path
        (bypasses the run directory for the JSONL file).
    verbosity_override : str | None
        CLI override for the verbosity level.
    """
    from unity.file_manager.managers.utils.progress import (
        CompositeReporter,
        ConsoleReporter,
        NoOpReporter,
        create_reporter,
    )

    enable = diagnostics.get("enable_progress", False)
    if not enable:
        return NoOpReporter(), run_dir

    if run_dir is None:
        run_dir = create_run_directory(script_name)

    mode = diagnostics.get("progress_mode", "json_file")
    verbosity = verbosity_override or diagnostics.get("verbosity", "medium")

    if progress_file_override:
        progress_file = progress_file_override
        error_dir = str(Path(progress_file_override).parent / "error_details")
    else:
        progress_file = str(run_dir / "progress.jsonl")
        error_dir = str(run_dir / "error_details")

    json_reporter = create_reporter(
        mode=mode,
        file_path=progress_file,
        verbosity=verbosity,
        error_dir=error_dir,
    )
    console_reporter = ConsoleReporter(emoji=False, show_timestamps=True)

    logger.info("Run directory: %s", run_dir)
    logger.info("Progress file: %s (verbosity=%s)", progress_file, verbosity)
    return CompositeReporter([json_reporter, console_reporter]), run_dir
