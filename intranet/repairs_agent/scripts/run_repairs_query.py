#!/usr/bin/env python3
"""run_repairs_query.py
Run bespoke queries using the LLM-free BespokeRepairsAgent.

Results are automatically logged to .repairs_queries/{timestamp}/ directory.
Terminal output shows only summary; full results are in log files.

Usage:
    python -m intranet.repairs_agent.scripts.run_repairs_query --list
    python -m intranet.repairs_agent.scripts.run_repairs_query --query jobs_completed_per_day
    python -m intranet.repairs_agent.scripts.run_repairs_query --query no_access_rate --params '{"return_absolute": true}'

See README.md in this directory for comprehensive usage examples.
"""

import asyncio
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add repo root to path for intranet imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPAIRS_AGENT_DIR = SCRIPT_DIR.parent
INTRANET_DIR = REPAIRS_AGENT_DIR.parent
REPO_ROOT = INTRANET_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from intranet.scripts.utils import initialize_script_environment, activate_project

# ---------------------------------------------------------------------------
# Boot-strap env / PYTHONPATH
# ---------------------------------------------------------------------------
if not initialize_script_environment():
    print("❌ Failed to initialize script environment")
    sys.exit(1)

# Import after environment is set up
from intranet.repairs_agent.static import (
    BespokeRepairsAgent,
    get_registered_count,
)

# Trigger query registration from the new canonical location
import intranet.repairs_agent.metrics  # noqa: F401

# Import the query logger (from same directory)
try:
    from .query_logger import QueryLogger, DebugLogCapture
except ImportError:
    from query_logger import QueryLogger, DebugLogCapture

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Configure logging based on verbosity level."""
    if debug:
        level = logging.DEBUG
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    elif verbose:
        level = logging.INFO
        log_format = "%(asctime)s | %(levelname)-8s | %(message)s"
    else:
        level = logging.WARNING
        log_format = "%(levelname)s: %(message)s"

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set specific logger levels
    if verbose or debug:
        logging.getLogger("intranet").setLevel(level)
        logging.getLogger("unity").setLevel(level)
    else:
        # Suppress info logs from libraries in non-verbose mode
        logging.getLogger("intranet").setLevel(logging.WARNING)
        logging.getLogger("unity").setLevel(logging.WARNING)


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------


def print_header() -> None:
    """Print script header."""
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║         Midland Heart Repairs - Bespoke Query Runner           ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()


def print_queries_list(queries: list) -> None:
    """Print formatted list of available queries."""
    print(f"📊 Available Metrics ({len(queries)} registered):")
    print("─" * 65)

    for i, q in enumerate(queries, 1):
        print(f"  {i:2}. {q['query_id']}")
        print(f"      └─ {q['description']}")
        print()

    print("─" * 65)
    print(
        "💡 Usage: python -m intranet.repairs_agent.scripts.run_repairs_query --query <query_id>",
    )
    print()


def print_result(result: Dict[str, Any], pretty: bool = True) -> None:
    """Print formatted query result."""
    print()
    print("📈 Query Results:")
    print("─" * 65)

    if pretty:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(result)

    print("─" * 65)


def print_summary(result: Dict[str, Any], elapsed: float) -> None:
    """Print result summary."""
    print()
    print("📊 Summary:")

    if isinstance(result, dict):
        metric_name = result.get("metric_name", "Unknown")
        total = result.get("total", "N/A")
        group_by = result.get("group_by", "N/A")

        print(f"   • Metric: {metric_name}")
        print(f"   • Total: {total}")
        print(f"   • Grouped by: {group_by}")

        # Handle new analysis format
        if "analysis" in result:
            timings = result.get("timings", {})
            print(f"   • Query time: {timings.get('query_ms', 'N/A')}ms")
            print(f"   • Analysis time: {timings.get('analysis_ms', 'N/A')}ms")
        else:
            # Legacy format
            num_results = len(result.get("results", []))
            print(f"   • Groups: {num_results}")

        # Show plot information if present
        plots = result.get("plots", [])
        if plots:
            print(f"   • Plots generated: {len(plots)}")
            for plot in plots:
                # Handle both dict and Pydantic model
                if hasattr(plot, "model_dump"):
                    plot = plot.model_dump()
                title = plot.get("title", "Untitled")
                url = plot.get("url")
                error = plot.get("error")
                if url:
                    print(f"      ✓ {title}: {url}")
                elif error:
                    print(f"      ✗ {title}: FAILED - {error}")

        # Show any warnings from metadata (legacy format)
        metadata = result.get("metadata") or {}
        if metadata.get("status") == "data_not_available":
            print(f"   ⚠️  Warning: {metadata.get('reason', 'Data not available')}")
        if metadata.get("note"):
            print(f"   ℹ️  Note: {metadata['note']}")

    print(f"   • Duration: {_format_duration(elapsed)}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Main entry point. Returns exit code."""
    parser = argparse.ArgumentParser(
        description="Run bespoke queries with the LLM-free BespokeRepairsAgent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list
  %(prog)s --query jobs_completed_per_day
  %(prog)s --query first_time_fix_rate --params '{"group_by": "patch"}'
  %(prog)s --query no_access_rate --params '{"return_absolute": true}'
  %(prog)s --query distance_travelled_per_day --verbose
  %(prog)s --query jobs_completed_per_day --project Intranet
  %(prog)s --query first_time_fix_rate --include-plots

See README.md for comprehensive usage documentation.
        """,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available queries",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query ID to execute",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="{}",
        help='JSON string of query parameters (e.g., \'{"start_date": "2025-07-01"}\')',
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty print JSON output (default: True)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Output raw JSON without formatting",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO level)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (DEBUG level)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="RepairsAgent5M",
        help="Project context to activate (default: RepairsAgent5M)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Custom root directory for log files (default: current working directory)",
    )
    parser.add_argument(
        "--log-subdir",
        type=str,
        default=None,
        help="Subdirectory name for logs (e.g., '{datetime}_{socket}' for per-terminal isolation)",
    )
    parser.add_argument(
        "--metric-subdir",
        type=str,
        default=None,
        help="Metric subdirectory for nested structure (e.g., 'jobs_completed_per_day' in full-matrix mode)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip generating summary file (used when parallel_queries.sh handles summaries)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable file logging, only print to terminal",
    )
    parser.add_argument(
        "--include-plots",
        action="store_true",
        default=False,
        help="Generate visualization URLs for query results",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Explicitly disable plot generation (default behavior)",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        default=False,
        help="Skip LLM analysis and return raw metric results only",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose, debug=args.debug)
    logger = logging.getLogger(__name__)

    # Print header
    print_header()

    logger.info(f"Script started at {datetime.now().isoformat()}")
    logger.debug(f"Arguments: {args}")

    # Activate project context
    try:
        logger.info(f"Activating project context: {args.project}")
        activate_project(args.project)
        logger.debug("Project context activated successfully")
        print(f"✓ Project context: {args.project}")
    except Exception as e:
        logger.error(f"Failed to activate project '{args.project}': {e}")
        print(f"❌ Failed to activate project context '{args.project}': {e}")
        return 1

    # Initialize agent
    try:
        logger.info("Initializing BespokeRepairsAgent...")
        init_start = time.perf_counter()
        agent = BespokeRepairsAgent(skip_analysis=args.skip_analysis)
        init_elapsed = time.perf_counter() - init_start
        logger.info(f"Agent initialized in {_format_duration(init_elapsed)}")
        analysis_mode = "raw data only" if args.skip_analysis else "with LLM analysis"
        print(
            f"✓ Agent initialized ({get_registered_count()} queries registered, {analysis_mode})",
        )
    except Exception as e:
        logger.exception("Failed to initialize agent")
        print(f"❌ Failed to initialize agent: {e}")
        return 1

    # List queries
    if args.list:
        logger.info("Listing available queries")
        queries = agent.list_queries()
        print_queries_list(queries)
        return 0

    # Validate query argument
    if not args.query:
        print("❌ Error: Either --list or --query is required")
        print()
        print(
            "💡 Try: python -m intranet.repairs_agent.scripts.run_repairs_query --list",
        )
        return 1

    # Parse params
    try:
        params = json.loads(args.params)
        logger.debug(f"Parsed parameters: {params}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON params: {e}")
        print(f"❌ Invalid JSON params: {e}")
        print()
        print("💡 JSON must use double quotes. Example:")
        print('   --params \'{"start_date": "2025-07-01", "group_by": "patch"}\'')
        return 1

    # Determine include_plots setting
    include_plots = args.include_plots and not args.no_plots
    if include_plots:
        params["include_plots"] = True
        logger.info("Plot generation enabled")

    # Initialize query logger
    query_logger: Optional[QueryLogger] = None
    if not args.no_log:
        log_root = Path(args.log_dir) if args.log_dir else None
        query_logger = QueryLogger(
            root_dir=log_root,
            log_subdir=args.log_subdir,
            metric_subdir=args.metric_subdir,
        )
        query_logger.start_run()

    # Execute query
    print(f"🔍 Executing query: {args.query}")
    if params:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"   Parameters: {param_str}")
    print()

    query_start = time.perf_counter()
    result = None
    success = True
    error_msg = None
    debug_output = ""

    # Capture debug logs from the metrics module during query execution
    with DebugLogCapture("intranet.repairs_agent.metrics.core") as debug_capture:
        try:
            logger.info(f"Executing query: {args.query}")
            result = await agent.ask(args.query, **params)
            query_elapsed = time.perf_counter() - query_start
            debug_output = debug_capture.get_output()

            # Log to file
            log_path = None
            if query_logger:
                log_path = query_logger.log_query(
                    query_id=args.query,
                    params=params,
                    result=result,
                    elapsed=query_elapsed,
                    success=True,
                    debug_log=debug_output,
                )
                if log_path:
                    print(f"📝 Results logged to: {log_path.name}")

            # Output to terminal
            if args.raw:
                print(json.dumps(result, default=str))
            elif args.no_log:
                # Full output only if not logging to file
                print_result(result, pretty=args.pretty)
                print_summary(result, query_elapsed)
            else:
                # Brief terminal output when logging to file
                print_summary(result, query_elapsed)

            print(
                f"✅ Query completed successfully in {_format_duration(query_elapsed)}",
            )
            logger.info(f"Query completed in {_format_duration(query_elapsed)}")

        except ValueError as e:
            query_elapsed = time.perf_counter() - query_start
            debug_output = debug_capture.get_output()
            success = False
            error_msg = str(e)
            logger.error(
                f"Query validation error after {_format_duration(query_elapsed)}: {e}",
            )

            if query_logger:
                import traceback

                query_logger.log_query(
                    query_id=args.query,
                    params=params,
                    result=None,
                    elapsed=query_elapsed,
                    success=False,
                    error=f"ValueError: {e}\n\n{traceback.format_exc()}",
                    debug_log=debug_output,
                )

            print(f"❌ {e}")
            print()
            print("💡 Use --list to see all available queries")

        except RuntimeError as e:
            query_elapsed = time.perf_counter() - query_start
            debug_output = debug_capture.get_output()
            success = False
            error_msg = str(e)
            logger.error(
                f"Query runtime error after {_format_duration(query_elapsed)}: {e}",
            )

            if query_logger:
                import traceback

                query_logger.log_query(
                    query_id=args.query,
                    params=params,
                    result=None,
                    elapsed=query_elapsed,
                    success=False,
                    error=f"RuntimeError: {e}\n\n{traceback.format_exc()}",
                    debug_log=debug_output,
                )

            print(f"❌ Query failed: {e}")
            print()
            print("💡 Try running with --verbose or --debug for more details")

        except KeyboardInterrupt:
            query_elapsed = time.perf_counter() - query_start
            debug_output = debug_capture.get_output()
            success = False
            error_msg = "Interrupted by user"
            logger.warning(
                f"Query interrupted by user after {_format_duration(query_elapsed)}",
            )

            if query_logger:
                query_logger.log_query(
                    query_id=args.query,
                    params=params,
                    result=None,
                    elapsed=query_elapsed,
                    success=False,
                    error="KeyboardInterrupt: Query interrupted by user",
                    debug_log=debug_output,
                )

            print()
            print(f"⚠️  Query interrupted after {_format_duration(query_elapsed)}")

        except Exception as e:
            query_elapsed = time.perf_counter() - query_start
            debug_output = debug_capture.get_output()
            success = False
            error_msg = f"{type(e).__name__}: {e}"
            logger.exception(
                f"Unexpected error after {_format_duration(query_elapsed)}",
            )

            if query_logger:
                import traceback

                query_logger.log_query(
                    query_id=args.query,
                    params=params,
                    result=None,
                    elapsed=query_elapsed,
                    success=False,
                    error=f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                    debug_log=debug_output,
                )

            print(f"❌ Unexpected error: {type(e).__name__}: {e}")
            print()
            print("💡 Run with --debug for full stack trace")

    # Finish logging run
    if query_logger:
        summary_path = query_logger.finish_run(skip_summary=args.no_summary)
        print()
        print(query_logger.get_terminal_summary())

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
