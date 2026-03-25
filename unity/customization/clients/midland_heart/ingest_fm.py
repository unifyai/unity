#!/usr/bin/env python3
"""Ingest Midland Heart spreadsheets via the FileManager pipeline.

Reads ``pipeline_config.json``, translates it into a ``FilePipelineConfig``
via :meth:`PipelineConfig.to_fm_config`, and calls
``FileManager.ingest_files()``.  Telemetry is handled internally by the
FM pipeline via its ``DiagnosticsConfig``.

Usage::

    .venv/bin/python unity/customization/clients/midland_heart/ingest_fm.py
    .venv/bin/python unity/customization/clients/midland_heart/ingest_fm.py \\
        --parallel --verbosity high --progress-file ./logs/fm_progress.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest Midland Heart spreadsheets via FileManager pipeline",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to pipeline_config.json (default: alongside this script)",
    )
    parser.add_argument("--project", default="MidlandHeart", help="Unify project name")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the project",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process files in parallel",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip embedding (ingest rows only, no vectorization)",
    )
    parser.add_argument(
        "--verbosity",
        choices=["low", "medium", "high"],
        default=None,
        help="Progress verbosity level",
    )
    parser.add_argument(
        "--progress-file",
        default=None,
        help="Progress JSONL output path (overrides run dir)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file path (default: <run_dir>/run.log)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    parser.add_argument(
        "--no-sdk-log",
        action="store_true",
        help="Disable the companion *_unify.log SDK log file",
    )
    args = parser.parse_args()

    # Bootstrap environment
    from ingest_utils import (
        initialize_environment,
        activate_project,
        load_pipeline_config,
        create_run_directory,
    )

    run_dir = create_run_directory("ingest_fm")
    log_file = args.log_file or str(run_dir / "run.log")

    project_root = initialize_environment(
        debug=args.debug,
        log_file=log_file,
        sdk_log=not args.no_sdk_log,
    )

    logger.info("=== Midland Heart FM Ingestion ===")
    logger.info("Run directory: %s", run_dir)
    start_time = time.perf_counter()

    # Load shared config (strongly typed)
    config = load_pipeline_config(args.config, project_root=project_root)

    # Collect file paths from source_files
    file_paths = [sf.file_path for sf in config.source_files]
    logger.info("Source files: %s", file_paths)

    # Activate project
    activate_project(args.project, overwrite=args.overwrite)

    # Produce a FilePipelineConfig from the unified schema, then apply
    # CLI overrides directly on the typed object (no JSON round-trip).
    cfg = config.to_fm_config()

    if args.no_embed:
        cfg.embed.strategy = "off"
    if args.parallel:
        cfg.execution.parallel_files = True
    cfg.diagnostics.enable_progress = True
    if args.verbosity:
        cfg.diagnostics.verbosity = args.verbosity
    if args.progress_file:
        cfg.diagnostics.progress_file = args.progress_file

    logger.info(
        "FM config: parallel=%s, embed_strategy=%s, progress=%s",
        cfg.execution.parallel_files,
        cfg.embed.strategy,
        cfg.diagnostics.enable_progress,
    )

    # Instantiate LocalFileManager and run pipeline
    from unity.file_manager.managers.local import LocalFileManager

    fm = LocalFileManager(str(project_root))
    logger.info("Running FM ingest pipeline for %d file(s)...", len(file_paths))

    result = fm.ingest_files(file_paths, config=cfg)

    # Print summary
    elapsed = time.perf_counter() - start_time
    success = sum(
        1 for f in result.files.values() if getattr(f, "status", "") == "success"
    )
    errors = sum(
        1 for f in result.files.values() if getattr(f, "status", "") == "error"
    )

    logger.info("=== FM Ingestion Complete ===")
    logger.info("  Total time: %.1fs", elapsed)
    logger.info("  Files: %d success, %d error", success, errors)
    logger.info("  Total rows ingested: %d", result.total_records)

    if errors > 0:
        for path, f in result.files.items():
            if getattr(f, "status", "") == "error":
                logger.error("  FAILED: %s -- %s", path, getattr(f, "error", "unknown"))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
