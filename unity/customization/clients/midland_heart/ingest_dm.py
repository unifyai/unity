#!/usr/bin/env python3
"""Ingest Midland Heart spreadsheets via the DataManager pipeline.

Parses Excel files using ``FileParser.parse_batch()`` (same pattern as
``FileManager.ingest_files``), then ingests each ``ExtractedTable`` into
``MidlandHeart/*`` contexts via ``DataManager.ingest()``.

Produces contexts named as if data came from REST APIs (e.g.
``MidlandHeart/Repairs2025``, ``MidlandHeart/Telematics2025/July``)
rather than file-centric paths (``Files/Local/*/Tables/*``).

Telemetry is wired up manually using the same ``ProgressReporter`` system
that the FM pipeline uses, so progress JSONL files are structurally
identical.

Usage::

    .venv/bin/python unity/customization/clients/midland_heart/ingest_dm.py
    .venv/bin/python unity/customization/clients/midland_heart/ingest_dm.py \\
        --no-embed --verbosity high --debug
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _resolve_embed_columns(
    config: Dict[str, Any],
    file_path: str,
    sheet_name: str,
) -> Optional[List[str]]:
    """Look up embed source columns for a given file + sheet from config."""
    embed_cfg = config.get("embed", {})
    for spec in embed_cfg.get("file_specs", []):
        if spec["file_path"] in file_path or spec["file_path"] == "*":
            for table_spec in spec.get("tables", []):
                if table_spec["table"] == sheet_name:
                    return table_spec.get("source_columns")
    return None


def _resolve_table_description(
    config: Dict[str, Any],
    sheet_name: str,
) -> Optional[str]:
    """Look up table description from business_contexts or dm_contexts."""
    dm_ctx = config.get("dm_contexts", {}).get(sheet_name, {})
    if dm_ctx.get("description"):
        return dm_ctx["description"]

    ingest_cfg = config.get("ingest", {})
    biz = ingest_cfg.get("business_contexts", config.get("business_contexts", {}))
    for fc in biz.get("file_contexts", []):
        for tc in fc.get("table_contexts", []):
            if tc.get("table") == sheet_name:
                return tc.get("table_description")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest Midland Heart spreadsheets via DataManager pipeline",
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
        "--no-embed",
        action="store_true",
        help="Skip embedding (ingest rows only, no vectorization)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Ingest tables in parallel",
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
    parser.add_argument(
        "--skip-all-context",
        action="store_true",
        help="Skip add_to_all_context (faster bulk load; add references later)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override per-table chunk_size from config (e.g. 250 for faster API calls)",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=None,
        metavar="LABEL",
        help=(
            "Only ingest the listed table(s) by sheet label (case-sensitive "
            "substring match).  E.g. --tables 'July 2025' to ingest only "
            "the July telematics sheet."
        ),
    )
    args = parser.parse_args()

    # Bootstrap environment
    from ingest_utils import (
        initialize_environment,
        activate_project,
        load_pipeline_config,
        create_pipeline_reporter,
        create_run_directory,
    )

    # Create run directory first so log file goes inside it
    run_dir = create_run_directory("ingest_dm")
    log_file = args.log_file or str(run_dir / "run.log")

    project_root = initialize_environment(
        debug=args.debug,
        log_file=log_file,
        sdk_log=not args.no_sdk_log,
    )

    logger.info("=== Midland Heart DM Ingestion ===")
    logger.info("Run directory: %s", run_dir)
    pipeline_start = time.perf_counter()

    # Load shared config
    config = load_pipeline_config(args.config, project_root=project_root)

    # Activate project
    activate_project(args.project, overwrite=args.overwrite)

    # Create progress reporter (uses same run_dir)
    diagnostics = config.get("diagnostics", {})
    reporter, run_dir = create_pipeline_reporter(
        diagnostics,
        "ingest_dm",
        run_dir=run_dir,
        progress_file_override=args.progress_file,
        verbosity_override=args.verbosity,
    )
    verbosity = args.verbosity or diagnostics.get("verbosity", "medium")

    from unity.file_manager.managers.utils.progress import create_progress_event

    # ------------------------------------------------------------------ #
    # Step 1: Parse Excel files using FileParser.parse_batch()
    #         (replicates file_manager.py lines 863-910)
    # ------------------------------------------------------------------ #

    from unity.file_manager.file_parsers import FileParser
    from unity.file_manager.file_parsers.types.contracts import (
        FileParseRequest,
        FileParseResult,
    )

    file_parser = FileParser()

    source_files = config["source_files"]
    parse_requests: list[FileParseRequest] = []
    for entry in source_files:
        fp = entry["file_path"]
        parse_requests.append(
            FileParseRequest(logical_path=fp, source_local_path=fp),
        )
        logger.info("Queued for parsing: %s", fp)

    # Emit parse/started events
    parse_start = time.perf_counter()
    for req in parse_requests:
        reporter.report(
            create_progress_event(
                req.logical_path,
                "parse",
                "started",
                duration_ms=0.0,
                elapsed_ms=0.0,
                verbosity=verbosity,
            ),
        )

    logger.info("Parsing %d file(s)...", len(parse_requests))
    parse_results: list[FileParseResult] = file_parser.parse_batch(
        parse_requests,
        raises_on_error=False,
    )

    parse_duration_ms = (time.perf_counter() - parse_start) * 1000

    # Emit per-file parse/completed or parse/failed events
    for pr in parse_results:
        lp = str(getattr(pr, "logical_path", "") or "")
        status = str(getattr(pr, "status", "error"))
        elapsed_ms = (time.perf_counter() - pipeline_start) * 1000

        if status == "success":
            table_count = len(pr.tables) if pr.tables else 0
            reporter.report(
                create_progress_event(
                    lp,
                    "parse",
                    "completed",
                    duration_ms=parse_duration_ms,
                    elapsed_ms=elapsed_ms,
                    meta={"table_count": table_count},
                    verbosity=verbosity,
                ),
            )
            logger.info(
                "Parsed %s: %d table(s) extracted",
                lp,
                table_count,
            )
        else:
            error_msg = str(getattr(pr, "error", "") or "unknown parse error")
            reporter.report(
                create_progress_event(
                    lp,
                    "parse",
                    "failed",
                    duration_ms=parse_duration_ms,
                    elapsed_ms=elapsed_ms,
                    error=error_msg,
                    verbosity=verbosity,
                ),
            )
            logger.error("Parse failed for %s: %s", lp, error_msg)

    # ------------------------------------------------------------------ #
    # Step 2: Ingest each ExtractedTable via DataManager.ingest()
    # ------------------------------------------------------------------ #

    from unity.data_manager.data_manager import DataManager

    dm = DataManager()
    dm_contexts = config.get("dm_contexts", {})
    embed_cfg = config.get("embed", {})
    embed_strategy = "off" if args.no_embed else embed_cfg.get("strategy", "along")
    infer_untyped = config.get("ingest", {}).get("infer_untyped_fields", False)
    max_table_workers = config.get("execution", {}).get(
        "max_table_workers",
        config.get("execution", {}).get("max_file_workers", 8),
    )
    total_tables = 0
    total_rows = 0
    failed_tables = 0
    ingested_contexts: list[str] = []

    def _make_chunk_callback(lp: str, sheet_name: str):
        """Build an on_task_complete callback that emits chunk-level progress."""

        def _on_chunk(task, result):
            meta = dict(task.metadata)
            meta["table_label"] = sheet_name
            meta["task_type"] = task.task_type
            meta["task_id"] = task.id
            meta["success"] = result.success
            meta["duration_ms"] = result.duration_ms
            meta["retries"] = result.retries
            if result.value and isinstance(result.value, dict):
                val = dict(result.value)
                if "inserted_ids" in val:
                    val["inserted_count"] = len(val.pop("inserted_ids"))
                meta.update(val)

            phase = f"ingest_table/{task.task_type}"
            status = "completed" if result.success else "failed"
            elapsed_ms = (time.perf_counter() - pipeline_start) * 1000

            reporter.report(
                create_progress_event(
                    lp,
                    phase,
                    status,
                    duration_ms=result.duration_ms,
                    elapsed_ms=elapsed_ms,
                    error=result.error if not result.success else None,
                    meta=meta,
                    verbosity=verbosity,
                ),
            )
            logger.info(
                "  [%s] %s %s (%.0fms) batch=%s/%s",
                sheet_name,
                task.task_type,
                status,
                result.duration_ms,
                meta.get("chunk_index", "?"),
                meta.get("total_chunks", "?"),
            )

        return _on_chunk

    def _ingest_table(
        table,
        lp: str,
    ) -> tuple[bool, str, int]:
        """Ingest a single table. Returns (success, context_path, row_count)."""
        sheet_name = table.sheet_name or table.label

        ctx_entry = dm_contexts.get(sheet_name)
        if ctx_entry is None:
            logger.warning(
                "No dm_contexts mapping for sheet '%s' -- skipping",
                sheet_name,
            )
            return (True, "", 0)

        context_path = ctx_entry["context"]
        description = _resolve_table_description(config, sheet_name)
        embed_columns = (
            _resolve_embed_columns(config, lp, sheet_name)
            if not args.no_embed
            else None
        )
        chunk_size = args.chunk_size or ctx_entry.get("chunk_size", 1000)
        rows = table.rows

        if not rows:
            logger.info(
                "Sheet '%s' has 0 rows -- skipping %s",
                sheet_name,
                context_path,
            )
            return (True, context_path, 0)

        n_chunks = (len(rows) + chunk_size - 1) // chunk_size

        logger.info(
            "Ingesting %d rows from '%s' -> %s (%d chunks of %d)",
            len(rows),
            sheet_name,
            context_path,
            n_chunks,
            chunk_size,
        )

        table_start = time.perf_counter()
        elapsed_ms = (table_start - pipeline_start) * 1000
        reporter.report(
            create_progress_event(
                lp,
                "ingest_table",
                "started",
                elapsed_ms=elapsed_ms,
                meta={
                    "table_label": sheet_name,
                    "row_count": len(rows),
                    "chunk_size": chunk_size,
                    "total_chunks": n_chunks,
                    "embed_columns": embed_columns,
                    "embed_strategy": embed_strategy if embed_columns else "off",
                },
                verbosity=verbosity,
            ),
        )

        chunk_callback = _make_chunk_callback(lp, sheet_name)

        try:
            result = dm.ingest(
                context_path,
                rows,
                description=description,
                embed_columns=embed_columns,
                embed_strategy=embed_strategy if embed_columns else "off",
                chunk_size=chunk_size,
                infer_untyped_fields=infer_untyped,
                add_to_all_context=not args.skip_all_context,
                on_task_complete=chunk_callback,
            )

            table_duration_ms = (time.perf_counter() - table_start) * 1000
            elapsed_ms = (time.perf_counter() - pipeline_start) * 1000

            reporter.report(
                create_progress_event(
                    lp,
                    "ingest_table",
                    "completed",
                    duration_ms=table_duration_ms,
                    elapsed_ms=elapsed_ms,
                    meta={
                        "table_label": sheet_name,
                        "row_count": len(rows),
                        "rows_inserted": result.rows_inserted,
                        "rows_embedded": result.rows_embedded,
                        "chunks_processed": result.chunks_processed,
                        "total_chunks": n_chunks,
                        "context": context_path,
                    },
                    verbosity=verbosity,
                ),
            )

            logger.info(
                "  -> %s: %d inserted, %d embedded, %d/%d chunks (%.1fs)",
                context_path,
                result.rows_inserted,
                result.rows_embedded,
                result.chunks_processed,
                n_chunks,
                table_duration_ms / 1000,
            )
            return (True, context_path, result.rows_inserted)

        except Exception as e:
            table_duration_ms = (time.perf_counter() - table_start) * 1000
            elapsed_ms = (time.perf_counter() - pipeline_start) * 1000

            import traceback

            tb = traceback.format_exc()

            reporter.report(
                create_progress_event(
                    lp,
                    "ingest_table",
                    "failed",
                    duration_ms=table_duration_ms,
                    elapsed_ms=elapsed_ms,
                    error=str(e),
                    traceback_str=tb,
                    meta={"table_label": sheet_name, "row_count": len(rows)},
                    verbosity=verbosity,
                ),
            )

            logger.error(
                "  -> FAILED %s: %s (%.1fs)",
                context_path,
                e,
                table_duration_ms / 1000,
            )
            return (False, context_path, 0)

    # Collect all (table, logical_path) pairs across all files
    all_work: list[tuple] = []  # (table, logical_path)
    file_tracking: dict[str, dict] = {}  # lp -> {start, tables, rows, failures}

    for pr in parse_results:
        if pr.status != "success" or not pr.tables:
            continue
        lp = str(pr.logical_path)
        file_tracking[lp] = {
            "start": time.perf_counter(),
            "tables": 0,
            "rows": 0,
            "failures": 0,
        }
        for t in pr.tables:
            label = t.sheet_name or t.label
            if dm_contexts.get(label) is None:
                continue
            if args.tables and not any(f in label for f in args.tables):
                logger.info("Skipping '%s' (not in --tables filter)", label)
                continue
            all_work.append((t, lp))

    logger.info(
        "Ingestion plan: %d tables across %d files (parallel=%s)",
        len(all_work),
        len(file_tracking),
        args.parallel,
    )

    def _collect(ok: bool, ctx: str, rows_inserted: int, lp: str):
        nonlocal total_tables, total_rows, failed_tables
        ft = file_tracking[lp]
        if ok and ctx:
            ft["tables"] += 1
            ft["rows"] += rows_inserted
            total_tables += 1
            total_rows += rows_inserted
            ingested_contexts.append(ctx)
        elif not ok:
            ft["failures"] += 1
            failed_tables += 1

    if args.parallel and len(all_work) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info(
            "Parallel ingestion: %d tables, max_workers=%d",
            len(all_work),
            max_table_workers,
        )
        with ThreadPoolExecutor(max_workers=max_table_workers) as pool:
            futures = {pool.submit(_ingest_table, t, lp): lp for t, lp in all_work}
            for fut in as_completed(futures):
                lp = futures[fut]
                ok, ctx, rows_inserted = fut.result()
                _collect(ok, ctx, rows_inserted, lp)
    else:
        for table, lp in all_work:
            ok, ctx, rows_inserted = _ingest_table(table, lp)
            _collect(ok, ctx, rows_inserted, lp)

    # Emit file_complete events
    for lp, ft in file_tracking.items():
        file_duration_ms = (time.perf_counter() - ft["start"]) * 1000
        elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
        file_status = "completed" if ft["failures"] == 0 else "failed"

        reporter.report(
            create_progress_event(
                lp,
                "file_complete",
                file_status,
                duration_ms=file_duration_ms,
                elapsed_ms=elapsed_ms,
                meta={
                    "tables_ingested": ft["tables"],
                    "rows_ingested": ft["rows"],
                    "ingest_failures": ft["failures"],
                },
                verbosity=verbosity,
            ),
        )

    reporter.flush()

    elapsed = time.perf_counter() - pipeline_start

    logger.info("=== DM Ingestion Complete ===")
    logger.info("  Run directory: %s", run_dir)
    logger.info("  Total time: %.1fs", elapsed)
    logger.info("  Tables ingested: %d", total_tables)
    logger.info("  Total rows: %d", total_rows)
    logger.info("  Failed tables: %d", failed_tables)
    logger.info("  Embed strategy: %s", embed_strategy)
    logger.info("  Contexts created:")
    for ctx in ingested_contexts:
        logger.info("    - %s", ctx)

    return 1 if failed_tables > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
