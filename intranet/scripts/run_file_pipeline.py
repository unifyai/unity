#!/usr/bin/env python3
"""
Generic FileManager pipeline execution script.

This script loads a JSON configuration file and executes a FileManager parsing pipeline.
It supports partial configs (only define what you need, defaults fill the rest) and can
process multiple files as specified in the config.

Usage:
    python intranet/scripts/run_file_pipeline.py \
        --config "/path/to/config.json" \
        --project Repairs \
        [--overwrite]

The config file should follow the FilePipelineConfig schema and can include:
- parse: parser configuration
- ingest: ingestion configuration (including business_contexts)
- embed: embedding configuration (with specs using source_columns/target_columns lists)
- plugins: plugin hooks
- output: output mode configuration
- diagnostics: diagnostic output configuration
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Set, Optional

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
        "--log-file",
        type=str,
        default=None,
        help="Optional path to log file. Output will be written to both stdout and the log file with colors preserved.",
    )
    args = parser.parse_args()

    # Set up logging to file if requested (preserving colors via ANSI codes)
    # We use Rich's Console for file output to preserve formatting
    # but keep stdout/stderr as-is so Rich progress bars work on terminal
    log_file_console: Optional[object] = None
    if args.log_file:
        log_path = Path(args.log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Use Rich Console for file output (preserves ANSI colors)
        from rich.console import Console as RichConsole

        log_file_handle = open(log_path, "w", encoding="utf-8")
        log_file_console = RichConsole(file=log_file_handle, force_terminal=True)

        # Create a Tee-like class that writes to both stdout and Rich Console file
        # Important: Must delegate terminal detection methods so Rich progress bars work
        class TeeOutput:
            def __init__(self, original_stream, rich_console):
                self.original_stream = original_stream
                self.rich_console = rich_console

            def write(self, obj):
                # Write to original stream (for terminal/progress bars)
                self.original_stream.write(obj)
                self.original_stream.flush()
                # Also write to Rich Console file (preserves formatting)
                if self.rich_console:
                    # Rich Console handles its own buffering
                    self.rich_console.print(obj, end="", markup=False)

            def flush(self):
                self.original_stream.flush()
                if self.rich_console:
                    self.rich_console.file.flush()

            def isatty(self):
                # Delegate to original stream so Rich detects terminal correctly
                return self.original_stream.isatty()

            def __getattr__(self, name):
                # Delegate other attributes to original stream
                return getattr(self.original_stream, name)

        # Replace stdout/stderr with TeeOutput
        import sys

        sys.stdout = TeeOutput(sys.stdout, log_file_console)  # type: ignore
        sys.stderr = TeeOutput(sys.stderr, log_file_console)  # type: ignore
        print(f"📝 Logging output to: {log_path}")

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

    print("✅ Configuration loaded and validated")

    # Extract file paths from business_contexts
    file_paths: Set[str] = set()
    if cfg.ingest.business_contexts:
        for bc in cfg.ingest.business_contexts:
            # Resolve relative paths relative to config file location, then project root
            fp = Path(bc.file_path)
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
            "⚠️  No file paths found in business_contexts. Please specify file_path in business_contexts.",
        )
        return 1

    file_paths_list = sorted(file_paths)
    print(f"📄 Processing {len(file_paths_list)} file(s):")
    for fp in file_paths_list:
        print(f"   • {fp}")

    print("🧪 Starting parse → ingest → embed…")
    try:
        parse_result = fm.parse(file_paths_list, config=cfg)
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
                print("   Errors:")
                for file_path, error_msg in errors.items():
                    print(f"      • {file_path}: {error_msg}")
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

    # Close log file if opened
    if log_file_console:
        log_file_console.file.close()
        print(f"✅ Log file saved: {args.log_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
