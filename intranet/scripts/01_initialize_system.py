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
import signal
import threading
from pathlib import Path

# Local helpers live in intranet.scripts.utils
from utils import initialize_script_environment, get_config_values, activate_project

# ---------------------------------------------------------------------------
# Boot-strap env / PYTHONPATH
# ---------------------------------------------------------------------------
if not initialize_script_environment():
    sys.exit(1)

from intranet.core.system_utils import SystemInitializer


_shutdown_requested = False
_loop_ref: asyncio.AbstractEventLoop | None = None
_shutdown_event: asyncio.Event | None = None


def _signal_handler(signum, _frame):
    """Catch SIGINT/SIGTERM and request a clean exit with a short grace period."""
    global _shutdown_requested
    _shutdown_requested = True

    sig_names = {signal.SIGINT: "SIGINT (CtrlC)", signal.SIGTERM: "SIGTERM"}
    name = sig_names.get(signum, f"Signal {signum}")
    print(f"\n🛑 Received {name} – requesting shutdown…")
    # Wake any waiter in the running loop
    if _loop_ref and _shutdown_event:
        try:
            _loop_ref.call_soon_threadsafe(_shutdown_event.set)
        except Exception:
            pass

    # Hard-exit after 10s if tasks don't complete
    def _force_exit():
        if _shutdown_requested:
            print("⏳ Graceful shutdown timed out – forcing exit.")
            # Avoid async teardown deadlocks
            sys.exit(1)

    t = threading.Timer(10.0, _force_exit)
    t.daemon = True
    t.start()


async def _run_or_shutdown(coro):
    """Run *coro* until completion or until a shutdown signal arrives."""
    global _shutdown_event
    _shutdown_event = asyncio.Event()
    task = asyncio.create_task(coro)
    done, pending = await asyncio.wait(
        {task, asyncio.create_task(_shutdown_event.wait())},
        return_when=asyncio.FIRST_COMPLETED,
    )
    # If shutdown fired first, cancel the job
    if _shutdown_event.is_set() and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return None, True
    # Else return result
    if task in done:
        return await task, False
    return None, False


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
    result, was_shutdown = await _run_or_shutdown(
        initializer.initialize_system(
            cfg,
            overwrite=args.overwrite,
            batch_size=args.batch_size,
        ),
    )
    if was_shutdown:
        print("🧹 Shutdown requested – exiting before completion.")
        sys.exit(130)  # 130 = terminated by CtrlC

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
    # Register handlers early
    for _sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(_sig, _signal_handler)
    _loop_ref = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(_loop_ref)
        _loop_ref.run_until_complete(main())
    finally:
        try:
            _loop_ref.close()
        except Exception:
            pass
