"""
Sync metric functions to FunctionManager.

This script extracts metric function implementations from the metrics module
and syncs them to FunctionManager's catalogue so CodeActActor can discover
and use them.

Usage:
    python -m intranet.repairs_agent.dynamic.sync
    python -m intranet.repairs_agent.dynamic.sync --dry-run
    python -m intranet.repairs_agent.dynamic.sync --overwrite
"""

from __future__ import annotations

import argparse
import ast
import inspect
import logging
import sys
import textwrap
from pathlib import Path
from typing import Callable, Dict, List, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from unity.function_manager.function_manager import FunctionManager

logger = logging.getLogger(__name__)


# =============================================================================
# METRIC FUNCTION DISCOVERY
# =============================================================================


def get_registered_metrics() -> Dict[str, Callable]:
    """
    Get all registered metric functions from the metrics module.

    Returns
    -------
    Dict[str, Callable]
        Mapping of query_id -> function object
    """
    # Import the registry to get registered functions
    from intranet.repairs_agent.static.registry import _REGISTRY

    # Trigger registration by importing the metrics module
    import intranet.repairs.queries  # noqa: F401

    return {spec.query_id: spec.fn for spec in _REGISTRY.values()}


def extract_function_source(func: Callable) -> str:
    """
    Extract the source code of a function.

    Parameters
    ----------
    func : Callable
        The function to extract source from

    Returns
    -------
    str
        The function's source code as a string
    """
    # Get the underlying function if it's wrapped
    actual_func = func
    while hasattr(actual_func, "__wrapped__"):
        actual_func = actual_func.__wrapped__

    source = inspect.getsource(actual_func)

    # Dedent to remove any leading whitespace
    source = textwrap.dedent(source)

    return source


def extract_function_with_docstring(func: Callable) -> Tuple[str, str]:
    """
    Extract function source and its docstring.

    Parameters
    ----------
    func : Callable
        The function to extract from

    Returns
    -------
    Tuple[str, str]
        (source_code, docstring)
    """
    source = extract_function_source(func)
    docstring = inspect.getdoc(func) or ""
    return source, docstring


def prepare_standalone_function(
    func: Callable,
    func_name: str,
) -> str:
    """
    Prepare a metric function as a standalone implementation.

    The function needs to be modified to:
    1. Remove decorators (@register, @metric_timer)
    2. Include necessary imports inline (simplified)
    3. Keep the enriched docstring

    Parameters
    ----------
    func : Callable
        The function object
    func_name : str
        The registered name of the function

    Returns
    -------
    str
        Standalone function source code
    """
    source = extract_function_source(func)

    # Parse and remove decorators
    tree = ast.parse(source)
    if tree.body and isinstance(tree.body[0], ast.AsyncFunctionDef | ast.FunctionDef):
        func_def = tree.body[0]
        # Remove decorators
        func_def.decorator_list = []

    # Convert back to source
    import ast

    try:
        # Python 3.9+
        clean_source = ast.unparse(tree)
    except AttributeError:
        # Fallback: manually strip decorator lines
        lines = source.split("\n")
        clean_lines = []
        in_decorator = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("@"):
                in_decorator = True
                continue
            if in_decorator and (
                stripped.startswith("async def") or stripped.startswith("def")
            ):
                in_decorator = False
            if (
                not in_decorator
                or stripped.startswith("async def")
                or stripped.startswith("def")
            ):
                clean_lines.append(line)
                in_decorator = False
        clean_source = "\n".join(clean_lines)

    return clean_source


# =============================================================================
# SYNC LOGIC
# =============================================================================


def sync_metrics_to_function_manager(
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    verbose: bool = False,
) -> Dict[str, str]:
    """
    Sync all registered metric functions to FunctionManager.

    Parameters
    ----------
    dry_run : bool
        If True, print what would be synced without actually syncing
    overwrite : bool
        If True, overwrite existing functions
    verbose : bool
        If True, print detailed progress

    Returns
    -------
    Dict[str, str]
        Results mapping function name to status
    """
    # Get all registered metrics
    metrics = get_registered_metrics()
    logger.info(f"Found {len(metrics)} registered metric functions")

    if verbose:
        for name in sorted(metrics.keys()):
            print(f"  - {name}")

    # Prepare function sources
    implementations: List[str] = []
    names: List[str] = []

    for name, func in metrics.items():
        try:
            source = prepare_standalone_function(func, name)
            implementations.append(source)
            names.append(name)

            if verbose:
                print(f"\n{'='*60}")
                print(f"Function: {name}")
                print(f"{'='*60}")
                # Show first 20 lines
                preview = "\n".join(source.split("\n")[:20])
                print(preview)
                if len(source.split("\n")) > 20:
                    print("... (truncated)")

        except Exception as e:
            logger.error(f"Failed to extract source for {name}: {e}")
            continue

    if dry_run:
        print(f"\n[DRY RUN] Would sync {len(implementations)} functions:")
        for name in names:
            print(f"  - {name}")
        return {name: "dry_run" for name in names}

    # Initialize FunctionManager and sync
    logger.info("Initializing FunctionManager...")
    fm = FunctionManager()

    logger.info(f"Syncing {len(implementations)} functions...")
    results = fm.add_functions(
        implementations=implementations,
        overwrite=overwrite,
    )

    # Log results
    added = sum(1 for v in results.values() if v == "added")
    updated = sum(1 for v in results.values() if v == "updated")
    skipped = sum(1 for v in results.values() if v == "skipped")
    errors = sum(1 for v in results.values() if v == "error")

    logger.info(
        f"Sync complete: {added} added, {updated} updated, {skipped} skipped, {errors} errors",
    )

    return results


# =============================================================================
# CLI
# =============================================================================


def main():
    """CLI entry point for sync script."""
    parser = argparse.ArgumentParser(
        description="Sync metric functions to FunctionManager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run - see what would be synced
    python -m intranet.repairs_agent.dynamic.sync --dry-run

    # Sync with verbose output
    python -m intranet.repairs_agent.dynamic.sync --verbose

    # Overwrite existing functions
    python -m intranet.repairs_agent.dynamic.sync --overwrite

    # Full verbose sync with overwrite
    python -m intranet.repairs_agent.dynamic.sync --verbose --overwrite
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be synced without actually syncing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing functions in FunctionManager",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress and function previews",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("Repairs Agent - Metric Function Sync")
    print("=" * 60)

    try:
        results = sync_metrics_to_function_manager(
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )

        print("\n" + "=" * 60)
        print("Results:")
        print("=" * 60)

        for name, status in sorted(results.items()):
            status_icon = {
                "added": "✓",
                "updated": "↻",
                "skipped": "○",
                "error": "✗",
                "dry_run": "?",
            }.get(status, "?")
            print(f"  {status_icon} {name}: {status}")

        # Exit with error if any failures
        if any(v == "error" for v in results.values()):
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Sync failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
