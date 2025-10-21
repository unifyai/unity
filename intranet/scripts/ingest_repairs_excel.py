#!/usr/bin/env python3
"""
One-off ingestion script for the Repairs Excel dataset.

This script:
  - Initializes a Unify project (default: "Repairs")
  - Parses a single Excel file using the FileManager pipeline
  - Automatically creates a per-table context and logs rows (via FileManager)
  - Ensures an embedding column for WorksOrderDescription on the created table context

Usage:
    python intranet/scripts/ingest_repairs_excel.py \
        --project Repairs \
        --file "/home/hmahmood24/unity/intranet/repairs/MDH Repairs data July & Aug 25 - DL V1.xlsx" \
        [--overwrite]

Notes:
  - This is a one-off, case-specific script intended for rapid use.
  - It relies on the FileManager's built-in per-table ingestion for spreadsheets.
  - Embeddings are created only for the WorksOrderDescription column in the table context.
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


def _find_new_table_contexts(all_context_names: List[str], base_ctx: str) -> List[str]:
    """Return contexts created under the FileManager tables namespace for this run."""
    prefix = f"{base_ctx}/Tables__"
    return [c for c in all_context_names if c.startswith(prefix)]


def main() -> int:
    parser = argparse.ArgumentParser(description="One-off Repairs Excel ingestion")
    parser.add_argument(
        "--project",
        type=str,
        default="Repairs",
        help="Unify project name (default: Repairs)",
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

    # Activate project (idempotent); ensures active context and Traces exist
    activate_project(args.project, overwrite=args.overwrite)

    # Now safe to import Unify/Unity modules
    import unify
    from unity.file_manager.managers.local import LocalFileManager

    project_root = Path(__file__).parent.parent.parent.resolve()

    # Set up a Local FileManager rooted at the project root so absolute paths
    # under this directory are accepted by the Local adapter
    fm = LocalFileManager(str(project_root))

    # Snapshot contexts before parse so we can detect newly created table contexts
    before_ctxs = set((unify.get_contexts() or {}).keys())

    file_path = str(Path(args.file).expanduser().resolve())
    print(f"📄 Ingesting file: {file_path}")

    # Parse the file (synchronous). This will:
    #  - Export a temp copy
    #  - Parse via Docling (when available)
    #  - Log a file record in the FileManager context
    #  - Log per-table rows into dedicated table contexts (for CSV/XLSX)
    display = fm.import_file(file_path)
    parse_result = fm.parse(display, table_rows_batch_size=1)

    # Basic success check
    any_success = any(
        (res or {}).get("status") == "success" for res in parse_result.values()
    )
    if not any_success:
        print(f"❌ Parsing failed – no successful results returned: {parse_result}")
        return 1

    # Compute new contexts created during parse and pick table contexts
    after_ctxs_all = set((unify.get_contexts() or {}).keys())
    new_ctxs = sorted(list(after_ctxs_all - before_ctxs))

    # The FileManager base context for this instance is private; access it directly
    base_ctx_for_fm = getattr(fm, "_ctx", "")
    table_ctxs = [c for c in new_ctxs if c.startswith(f"{base_ctx_for_fm}/Tables__")]

    if not table_ctxs:
        print(
            "⚠️  No per-table contexts detected. Was a table present in the Excel file?",
        )
        # Continue anyway; nothing to embed
        return 0

    # Expecting a single sheet → one context; if multiple, take the first
    table_ctx = table_ctxs[0]
    print(f"🗂️  Table context: {table_ctx}")

    # # Create/ensure an embedding column from WorksOrderDescription
    # # Hardcoded for this one-off flow
    # target_vector_col = "_works_order_description_emb"
    # source_text_col = "WorksOrderDescription"

    # try:
    #     # Will no-op if the vector column already exists
    #     ensure_vector_column(
    #         context=table_ctx,
    #         embed_column=target_vector_col,
    #         source_column=source_text_col,
    #     )
    #     print(
    #         f"✅ Embedding ensured for column '{source_text_col}' → '{target_vector_col}'",
    #     )
    # except Exception as e:
    #     print(f"❌ Failed to ensure embedding column: {e}")
    #     return 1

    print("🎉 Repairs Excel ingestion complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
