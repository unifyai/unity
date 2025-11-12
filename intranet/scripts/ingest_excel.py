#!/usr/bin/env python3
"""
Generic Excel ingestion script using the FileManager pipeline.

This script:
- Initializes a Unify project (default: "Repairs")
- Parses a single Excel file via FileManager and ingests rows into per‑file tables
- Ensures embeddings are created alongside inserts for selected table columns
- Prints a brief overview of created per‑file table contexts

Usage:
    python intranet/scripts/ingest_excel.py \
        --project Repairs \
        --file "/abs/path/to/your.xlsx" \
        [--overwrite]

Notes:
- The script is generic; it accepts any Excel file path.
- Per-file layout is used; Excel sheets populate per‑file Tables contexts.
- Embeddings are configured for specific columns on a specific sheet label.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Local helpers live in intranet.scripts.utils
from utils import initialize_script_environment, activate_project


# ---------------------------------------------------------------------------
# Boot-strap env / PYTHONPATH before importing Unify/Unity modules
# ---------------------------------------------------------------------------
if not initialize_script_environment():
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generic Excel ingestion using FileManager",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="Repairs",
        help='Unify project name (default: "Repairs")',
    )
    parser.add_argument(
        "--file",
        type=str,
        default=str(
            Path("intranet/repairs/MDH Repairs data July & Aug 25 - DL V1.xlsx"),
        ),
        help="Absolute path to the Excel file to ingest",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Delete existing project data first",
    )
    args = parser.parse_args()

    print("🔧 Initializing Unify project and environment…")
    activate_project(args.project, overwrite=args.overwrite)

    # Now safe to import Unify/Unity modules
    from unity.file_manager.managers.local import LocalFileManager
    from unity.file_manager.types import FilePipelineConfig, EmbeddingSpec

    project_root = Path(__file__).parent.parent.parent.resolve()

    # Set up a Local FileManager rooted at the project root so absolute paths
    # under this directory are accepted by the Local adapter
    fm = LocalFileManager(str(project_root))
    print(f"🧩 FileManager initialised (filesystem: Local, root: {project_root})")

    file_path = str(Path(args.file).expanduser().resolve())
    print(f"📄 Ingesting file: {file_path}")

    # Build pipeline configuration:
    # - Per-file layout (default)
    # - Larger batch sizes for faster ingestion
    # - Use ingest-and-embed-along strategy for the given table + columns
    # - Compact output to keep responses light
    cfg = FilePipelineConfig()
    cfg.ingest.table_rows_batch_size = 2000
    cfg.ingest.content_rows_batch_size = 2000
    # Explicitly select along strategy (chunked ingest+embed within a single file)
    cfg.embed.strategy = "along"
    # Table label is provided unsafed; FileManager will match against its safe() label
    cfg.embed.specs = [
        EmbeddingSpec(
            context="per_file_table",
            table="Raised 01-07-2025 to 31-08-2025",
            source_column="WorksOrderDescription",
            target_column="_WorksOrderDescription_emb",
        ),
        EmbeddingSpec(
            context="per_file_table",
            table="Raised 01-07-2025 to 31-08-2025",
            source_column="FullAddress",
            target_column="_FullAddress_emb",
        ),
        EmbeddingSpec(
            context="per_file_table",
            table="Raised 01-07-2025 to 31-08-2025",
            source_column="OperativeName",
            target_column="_OperativeName_emb",
        ),
        EmbeddingSpec(
            context="per_file_table",
            table="Raised 01-07-2025 to 31-08-2025",
            source_column="OperativeWhoCompletedJob",
            target_column="_OperativeWhoCompletedJob_emb",
        ),
    ]
    cfg.output.return_mode = "compact"

    print("🧪 Starting parse → ingest → embed…")
    parse_result = fm.parse(file_path, config=cfg)

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

    any_success = any(
        _extract_status(res) == "success" for res in parse_result.values()
    )
    if not any_success:
        # Print summarized statuses for debugging
        try:
            statuses = {k: _extract_status(v) for k, v in parse_result.items()}
            print(
                f"❌ Parsing failed – no successful results returned. Statuses: {statuses}",
            )
        except Exception:
            print(f"❌ Parsing failed – no successful results returned.")
        return 1

    # Discover per-file table contexts for this file using the manager's overview
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
        print("🗂️  Per-file Tables contexts (for this file):")
        for ctx in sorted(discovered_tables):
            print(f"   • {ctx}")
    else:
        print(
            "⚠️  No per-file table contexts detected for this file. Was a table present in the Excel file?",
        )

    print("🎉 Excel ingestion complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
