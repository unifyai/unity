#!/usr/bin/env python3
"""
Generic FileManager pipeline execution script.

This script loads a JSON configuration file and executes a FileManager parsing pipeline.
It supports partial configs (only define what you need, defaults fill the rest) and can
process multiple files as specified in the config.

Usage:
    python intranet/scripts/run_file_pipeline.py \\
        --config "/path/to/config.json" \\
        --project Repairs \\
        [--overwrite] \\
        [--parallel] \\
        [--progress json_file|callback|off] \\
        [--progress-file /path/to/progress.jsonl] \\
        [--verbosity low|medium|high] \\
        [--fail-fast]

The config file should follow the FilePipelineConfig schema and can include:
- parse: parser configuration
- ingest: ingestion configuration (including business_contexts)
- embed: embedding configuration (with specs using source_columns/target_columns lists)
- plugins: plugin hooks
- output: output mode configuration
- diagnostics: diagnostic output configuration (progress_mode, verbosity, progress_file)
- execution: parallel processing configuration
- retry: retry behavior configuration

Progress Reporting:
- json_file: Append JSON-lines to a file (auto-generated if --progress-file not provided)
- callback: Invoke a user-provided callback (not available via CLI)
- off: Disable progress reporting

Terminal output only includes metadata (file paths, summary, errors).
All progress events go to the progress file for structured analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Set

# Local helpers live in intranet.scripts.utils
from utils import initialize_script_environment, activate_project

# ---------------------------------------------------------------------------
# Boot-strap env / PYTHONPATH before importing Unify/Unity modules
# ---------------------------------------------------------------------------
if not initialize_script_environment():
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generic FileManager pipeline execution using JSON config",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="Repairs",
        help='Unify project name (default: "Repairs")',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Delete existing project data first",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Enable parallel file processing (overrides config)",
    )
    parser.add_argument(
        "--progress",
        type=str,
        choices=["json_file", "off"],
        default=None,
        help="Progress reporting mode (overrides config). Use json_file to write events to a file, off to disable.",
    )
    parser.add_argument(
        "--progress-file",
        type=str,
        default=None,
        help="Path for JSON-lines progress file. If --progress=json_file but no path provided, auto-generates one.",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="Verbosity level for progress events: low (minimal), medium (detailed), high (verbose).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Stop pipeline on first failure (overrides config)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable DEBUG logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Configure logging level based on --debug flag
    #
    # NOTE:
    # initialize_script_environment() / Unify may have already installed
    # handlers on the root logger before we get here. To ensure --debug
    # actually surfaces DEBUG logs (including [TaskFn] debug lines), we use
    # basicConfig(..., force=True) so that existing handlers are replaced
    # with our configuration.
    import logging

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,  # Replace any existing handlers so level actually takes effect
    )

    if args.debug:
        print("🔍 Debug logging enabled")

    print("🔧 Initializing Unify project and environment…")
    activate_project(args.project, overwrite=args.overwrite)

    # Now safe to import Unify/Unity modules
    from unity.file_manager.managers.local import LocalFileManager
    from unity.file_manager.types import FilePipelineConfig

    project_root = Path(__file__).parent.parent.parent.resolve()

    # Set up a Local FileManager rooted at the project root so absolute paths
    # under this directory are accepted by the Local adapter
    fm = LocalFileManager(str(project_root))
    print(f"🧩 FileManager initialised (filesystem: Local, root: {project_root})")

    # Load and validate config file
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return 1

    print(f"📋 Loading configuration from: {config_path}")
    try:
        cfg = FilePipelineConfig.from_file(str(config_path))
    except Exception as e:
        print(f"❌ Failed to load config file: {e}")
        return 1

    # Apply command line overrides
    if args.parallel:
        cfg.execution.parallel_files = True
        print("   ⚙️  Parallel file processing enabled (--parallel)")

    # Progress configuration
    if args.progress:
        cfg.diagnostics.enable_progress = args.progress != "off"
        cfg.diagnostics.progress_mode = args.progress
        print(f"   ⚙️  Progress mode set to: {args.progress}")
    if args.progress_file:
        cfg.diagnostics.progress_file = args.progress_file
        print(f"   ⚙️  Progress file set to: {args.progress_file}")
    if args.verbosity:
        cfg.diagnostics.verbosity = args.verbosity
        print(f"   ⚙️  Verbosity set to: {args.verbosity}")

    if args.fail_fast:
        cfg.retry.fail_fast = True
        print("   ⚙️  Fail-fast mode enabled (--fail-fast)")

    print("✅ Configuration loaded and validated")

    # Print config summary so user can verify settings
    print("\n📊 Configuration Summary:")
    print(f"   Parse:")
    print(f"      • max_concurrent_parses: {cfg.parse.max_concurrent_parses}")
    backend_overrides = {
        k: v for k, v in cfg.parse.backend_class_paths_by_format.items()
    }
    print(
        f"      • backend_class_paths_by_format: {len(backend_overrides)} format(s) configured",
    )
    print(f"   Ingest:")
    print(f"      • storage_id: {cfg.ingest.storage_id or 'per-file (auto)'}")
    print(f"      • table_ingest: {cfg.ingest.table_ingest}")
    print(f"      • content_rows_batch_size: {cfg.ingest.content_rows_batch_size}")
    print(f"      • table_rows_batch_size: {cfg.ingest.table_rows_batch_size}")
    # storage_id already printed above
    print(f"   Embed:")
    print(f"      • strategy: {cfg.embed.strategy}")
    embed_specs_count = len(cfg.embed.file_specs) if cfg.embed.file_specs else 0
    print(f"      • specs: {embed_specs_count} spec(s) defined")
    print(f"   Execution:")
    print(f"      • parallel_files: {cfg.execution.parallel_files}")
    print(f"      • max_file_workers: {cfg.execution.max_file_workers}")
    print(f"   Retry:")
    print(f"      • max_retries: {cfg.retry.max_retries}")
    print(f"      • fail_fast: {cfg.retry.fail_fast}")
    print(f"   Diagnostics:")
    print(f"      • enable_progress: {cfg.diagnostics.enable_progress}")
    print(f"      • progress_mode: {cfg.diagnostics.progress_mode}")
    print(f"      • verbosity: {cfg.diagnostics.verbosity}")
    print(f"   Output:")
    print(f"      • return_mode: {cfg.output.return_mode}")

    # Inform user about progress file location
    if cfg.diagnostics.enable_progress and cfg.diagnostics.progress_mode == "json_file":
        import time

        progress_file = cfg.diagnostics.progress_file
        if not progress_file:
            # Generate default path
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            progress_file = f"./pipeline_progress_{timestamp}.jsonl"
            cfg.diagnostics.progress_file = progress_file
        progress_path = Path(progress_file).expanduser().resolve()
        print(f"📝 Progress events will be logged to: {progress_path}")

    # Extract file paths from business_contexts.file_contexts
    file_paths: Set[str] = set()
    if cfg.ingest.business_contexts and cfg.ingest.business_contexts.file_contexts:
        for fc in cfg.ingest.business_contexts.file_contexts:
            # Resolve relative paths relative to config file location, then project root
            fp = Path(fc.file_path)
            if not fp.is_absolute():
                # First try relative to config file
                config_dir = config_path.parent
                candidate = config_dir / fp
                if candidate.exists():
                    fp = candidate
                else:
                    # Fall back to project root
                    fp = project_root / fp
            else:
                fp = Path(fp)
            file_paths.add(str(fp.resolve()))

    if not file_paths:
        print(
            "⚠️  No file paths found in business_contexts.file_contexts. Please specify file_path in file_contexts.",
        )
        return 1

    file_paths_list = sorted(file_paths)
    print(f"📄 Processing {len(file_paths_list)} file(s):")
    for fp in file_paths_list:
        print(f"   • {fp}")

    print("🧪 Starting parse → ingest → embed…")
    try:
        parse_result = fm.ingest_files(file_paths_list, config=cfg)
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Basic success check and summary
    def _extract_status(res: object) -> str | None:
        if res is None:
            return None
        if isinstance(res, dict):
            return res.get("status")
        # Pydantic model (v2): prefer model_dump
        try:
            md = getattr(res, "model_dump", None)
            if callable(md):
                d = md()
                if isinstance(d, dict):
                    return d.get("status")
        except Exception:
            pass
        # Fallback to attribute access
        try:
            return getattr(res, "status", None)
        except Exception:
            return None

    def _extract_error(res: object) -> str | None:
        """Extract error message from result object."""
        if res is None:
            return None
        if isinstance(res, dict):
            return res.get("error")
        # Pydantic model (v2): prefer model_dump
        try:
            md = getattr(res, "model_dump", None)
            if callable(md):
                d = md()
                if isinstance(d, dict):
                    return d.get("error")
        except Exception:
            pass
        # Fallback to attribute access
        try:
            return getattr(res, "error", None)
        except Exception:
            return None

    # Helper to extract traceback from result
    def _extract_traceback(res: object) -> str | None:
        """Extract traceback string from result object."""
        if res is None:
            return None
        if isinstance(res, dict):
            return res.get("traceback")
        try:
            return getattr(res, "traceback", None)
        except Exception:
            return None

    any_success = any(
        _extract_status(res) == "success" for res in parse_result.values()
    )
    if not any_success:
        # Print summarized statuses and errors for debugging
        try:
            statuses = {k: _extract_status(v) for k, v in parse_result.items()}
            errors = {
                k: _extract_error(v)
                for k, v in parse_result.items()
                if _extract_status(v) in ("failed", "error") and _extract_error(v)
            }
            print(
                f"❌ Processing failed – no successful results returned. Statuses: {statuses}",
            )
            if errors:
                print("\n📋 Error Summary:")
                print("-" * 60)
                for file_path, error_msg in errors.items():
                    print(f"   ❌ {file_path}")
                    print(f"      Error: {error_msg}")
                    # Show traceback if available
                    tb = _extract_traceback(parse_result.get(file_path))
                    if tb:
                        print(f"      Traceback:\n{tb}")
                print("-" * 60)
                # Point user to progress file for more details
                if (
                    cfg.diagnostics.enable_progress
                    and cfg.diagnostics.progress_mode == "json_file"
                ):
                    progress_path = cfg.diagnostics.progress_file
                    if progress_path:
                        print(
                            f"\n📝 For detailed progress events, see: {progress_path}",
                        )
        except Exception:
            print("❌ Processing failed – no successful results returned.")
        return 1

    # Discover per-file table contexts for processed files
    discovered_tables_by_file: dict[str, List[str]] = {}
    for file_path in file_paths_list:
        try:
            overview = fm._tables_overview(file=file_path)  # type: ignore[attr-defined]
        except Exception:
            overview = {}

        discovered_tables: List[str] = []
        for key, val in (overview or {}).items():
            if key == "FileRecords" or not isinstance(val, dict):
                continue
            tables = val.get("Tables")
            if not isinstance(tables, dict):
                continue
            for label, info in tables.items():
                ctx = (info or {}).get("context")
                if isinstance(ctx, str) and ctx:
                    discovered_tables.append(ctx)

        if discovered_tables:
            discovered_tables_by_file[file_path] = discovered_tables

    if discovered_tables_by_file:
        print("🗂️  Per-file Tables contexts created:")
        for file_path, tables in discovered_tables_by_file.items():
            print(f"   {file_path}:")
            for ctx in sorted(tables):
                print(f"      • {ctx}")
    else:
        print(
            "⚠️  No per-file table contexts detected. Were tables present in the files?",
        )

    print("🎉 Pipeline execution complete.")

    # Remind user where progress logs are
    if cfg.diagnostics.enable_progress and cfg.diagnostics.progress_mode == "json_file":
        progress_path = Path(cfg.diagnostics.progress_file).expanduser().resolve()
        if progress_path.exists():
            print(f"✅ Progress events saved to: {progress_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
