#!/usr/bin/env python3
"""04_initialize_system.py
Standalone system initializer for Midland Heart RAG.
Run this once after changing the schema or adding documents to:
  • (Re)create the project context
  • Initialise / refactor the single-table schema
  • Ingest all documents under intranet/policies (or DOCUMENTS_PATH env var)
  • Pre-embed the configured summary embeddings

Usage:
    python scripts/04_initialize_system.py [--use-tool-loops] [--overwrite]

Flags:
    --use-tool-loops   Perform initialisation via LLM-driven tool loops (slower, robust)
    --overwrite        Drop the existing "Intranet" project before rebuilding
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Local helpers live in intranet.scripts.utils
from utils import initialize_script_environment, get_config_values, activate_project

# ---------------------------------------------------------------------------
# Boot-strap env / PYTHONPATH
# ---------------------------------------------------------------------------
if not initialize_script_environment():
    sys.exit(1)

from intranet.core.system_utils import SystemInitializer


async def main():
    cfg = get_config_values()

    parser = argparse.ArgumentParser(description="Initialise Midland Heart RAG system")
    parser.add_argument(
        "--use-tool-loops",
        action="store_true",
        default=False,
        help="Run initialisation via LLM tool loops",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Delete existing project data first",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of documents to process in parallel (default: 5)",
    )
    args = parser.parse_args()

    # (Re)activate project
    activate_project("Intranet", overwrite=args.overwrite)

    # Collect config and pass schema path
    cfg["schema_path"] = str(Path(__file__).parent.parent / "flat_schema.json")

    initializer = SystemInitializer(use_tool_loops=args.use_tool_loops)
    result = await initializer.initialize_system(
        cfg,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
    )

    if result.get("success"):
        print("🎉 System initialisation completed!")
        # Future server starts can skip init
        import os

        os.environ["RAG_SKIP_INIT"] = "true"
        sys.exit(0)
    else:
        print(f"❌ Initialisation failed: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
