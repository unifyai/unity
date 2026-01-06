"""
Sync metric functions to FunctionManager.

This script extracts metric function implementations from the metrics module
and syncs them to FunctionManager's catalogue so CodeActActor can discover
and use them.

Usage:
    python -m intranet.repairs_agent.dynamic.sync
    python -m intranet.repairs_agent.dynamic.sync --dry-run
    python -m intranet.repairs_agent.dynamic.sync --overwrite-functions
    python -m intranet.repairs_agent.dynamic.sync --project RepairsAgent
    python -m intranet.repairs_agent.dynamic.sync --project RepairsAgent --overwrite-project
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

# ---------------------------------------------------------------------------
# Bootstrap: Add paths for imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPAIRS_AGENT_DIR = SCRIPT_DIR.parent
INTRANET_DIR = REPAIRS_AGENT_DIR.parent
PROJECT_ROOT = INTRANET_DIR.parent
SCRIPTS_DIR = INTRANET_DIR / "scripts"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from intranet.scripts.utils import activate_project, initialize_script_environment

# ---------------------------------------------------------------------------
# Initialize script environment (loads .env, sets up paths)
# ---------------------------------------------------------------------------
if not initialize_script_environment():
    print("❌ Failed to initialize script environment")
    sys.exit(1)

# Import after environment is set up
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
    # Import the registry from the static agent module
    from intranet.repairs_agent.static.registry import _REGISTRY

    # Trigger registration by importing the metrics module
    # This ensures all @register decorators are run
    import intranet.repairs_agent.metrics  # noqa: F401

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


# Known enum types and their string value mappings
# These are used to replace default argument values like GroupBy.OPERATIVE -> "operative"
ENUM_VALUE_MAPPINGS = {
    "GroupBy.OPERATIVE": "operative",
    "GroupBy.PATCH": "patch",
    "GroupBy.REGION": "region",
    "GroupBy.TRADE": "trade",
    "GroupBy.TOTAL": "total",
    "TimePeriod.DAY": "day",
    "TimePeriod.WEEK": "week",
    "TimePeriod.MONTH": "month",
}


class AnnotationStringifier(ast.NodeTransformer):
    """
    AST transformer that:
    1. Converts type annotations to string literals (forward references)
    2. Converts enum default values to their string equivalents

    This allows functions with custom types (GroupBy, MetricResult, etc.)
    to be exec'd in an environment that doesn't have those types defined.
    """

    def _convert_enum_default(self, node: ast.expr) -> ast.expr:
        """Convert enum default value (e.g., GroupBy.OPERATIVE) to string."""
        if isinstance(node, ast.Attribute):
            # Check if it's an enum attribute like GroupBy.OPERATIVE
            unparsed = ast.unparse(node)
            if unparsed in ENUM_VALUE_MAPPINGS:
                return ast.Constant(value=ENUM_VALUE_MAPPINGS[unparsed])
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        """Convert argument annotations to strings."""
        if node.annotation is not None:
            node.annotation = ast.Constant(value=ast.unparse(node.annotation))
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Convert return annotation and default values."""
        if node.returns is not None:
            node.returns = ast.Constant(value=ast.unparse(node.returns))

        # Convert default values for arguments
        node.args.defaults = [self._convert_enum_default(d) for d in node.args.defaults]
        node.args.kw_defaults = [
            self._convert_enum_default(d) if d is not None else None
            for d in node.args.kw_defaults
        ]

        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> ast.AsyncFunctionDef:
        """Convert return annotation and default values."""
        if node.returns is not None:
            node.returns = ast.Constant(value=ast.unparse(node.returns))

        # Convert default values for arguments
        node.args.defaults = [self._convert_enum_default(d) for d in node.args.defaults]
        node.args.kw_defaults = [
            self._convert_enum_default(d) if d is not None else None
            for d in node.args.kw_defaults
        ]

        self.generic_visit(node)
        return node


def prepare_standalone_function(
    func: Callable,
    func_name: str,
) -> str:
    """
    Prepare a metric function as a standalone implementation.

    The function needs to be modified to:
    1. Remove decorators (@register, @metric_timer)
    2. Convert type annotations to string literals (forward references)
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
        Standalone function source code ready for FunctionManager
    """
    source = extract_function_source(func)

    # Parse the source
    tree = ast.parse(source)

    if tree.body and isinstance(tree.body[0], ast.AsyncFunctionDef | ast.FunctionDef):
        func_def = tree.body[0]
        # Remove decorators
        func_def.decorator_list = []

    # Convert type annotations to strings so they don't need to be defined
    # This allows exec() to work without GroupBy, MetricResult, etc.
    transformer = AnnotationStringifier()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    # Convert back to source
    try:
        # Python 3.9+
        clean_source = ast.unparse(tree)
    except AttributeError:
        # Fallback: manually strip decorator lines (annotations won't be converted)
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
# HELPER FUNCTION DISCOVERY
# =============================================================================


def get_helper_functions() -> Dict[str, Callable]:
    """
    Get all helper functions from the helpers module.

    Returns
    -------
    Dict[str, Callable]
        Mapping of function_name -> function object
    """
    from intranet.repairs_agent.metrics import helpers
    from intranet.repairs_agent.metrics.helpers import HELPER_FUNCTIONS

    return {name: getattr(helpers, name) for name in HELPER_FUNCTIONS}


# =============================================================================
# SYNC LOGIC
# =============================================================================


def sync_metrics_to_function_manager(
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    verbose: bool = False,
    include_helpers: bool = True,
) -> Dict[str, str]:
    """
    Sync all registered metric functions and helpers to FunctionManager.

    Parameters
    ----------
    dry_run : bool
        If True, print what would be synced without actually syncing
    overwrite : bool
        If True, overwrite existing functions
    verbose : bool
        If True, print detailed progress
    include_helpers : bool
        If True, also sync helper functions (default: True)

    Returns
    -------
    Dict[str, str]
        Results mapping function name to status
    """
    # Get all registered metrics
    metrics = get_registered_metrics()
    logger.info(f"Found {len(metrics)} registered metric functions")

    # Get helper functions if requested
    helpers = {}
    if include_helpers:
        helpers = get_helper_functions()
        logger.info(f"Found {len(helpers)} helper functions")

    if verbose:
        print("\n[Metrics]")
        for name in sorted(metrics.keys()):
            print(f"  - {name}")
        if include_helpers:
            print("\n[Helpers]")
            for name in sorted(helpers.keys()):
                print(f"  - {name}")

    # Prepare function sources
    implementations: List[str] = []
    names: List[str] = []

    # Process metrics
    for name, func in metrics.items():
        try:
            source = prepare_standalone_function(func, name)
            implementations.append(source)
            names.append(name)

            if verbose:
                print(f"\n{'='*60}")
                print(f"Metric: {name}")
                print(f"{'='*60}")
                # Show first 20 lines
                preview = "\n".join(source.split("\n")[:20])
                print(preview)
                if len(source.split("\n")) > 20:
                    print("... (truncated)")

        except Exception as e:
            logger.error(f"Failed to extract source for metric {name}: {e}")
            continue

    # Process helpers
    if include_helpers:
        for name, func in helpers.items():
            try:
                source = extract_function_source(func)
                implementations.append(source)
                names.append(name)

                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Helper: {name}")
                    print(f"{'='*60}")
                    preview = "\n".join(source.split("\n")[:20])
                    print(preview)
                    if len(source.split("\n")) > 20:
                        print("... (truncated)")

            except Exception as e:
                logger.error(f"Failed to extract source for helper {name}: {e}")
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

    # Overwrite existing functions in FunctionManager
    python -m intranet.repairs_agent.dynamic.sync --overwrite-functions

    # Use a different project context
    python -m intranet.repairs_agent.dynamic.sync --project MyProject

    # Full verbose sync with overwrite
    python -m intranet.repairs_agent.dynamic.sync --verbose --overwrite-functions
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be synced without actually syncing",
    )
    parser.add_argument(
        "--overwrite-functions",
        action="store_true",
        help="Overwrite existing functions in FunctionManager",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="RepairsAgent5M",
        help="Project context to activate (default: RepairsAgent5M)",
    )
    parser.add_argument(
        "--overwrite-project",
        action="store_true",
        help="Overwrite project contexts on activation (default: False)",
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
    parser.add_argument(
        "--no-helpers",
        action="store_true",
        help="Skip syncing helper functions (sync only metrics)",
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

    # Activate project context before using FunctionManager
    try:
        logger.info(f"Activating project context: {args.project}")
        activate_project(args.project, overwrite=args.overwrite_project)
        logger.debug("Project context activated successfully")
        print(f"✓ Project context: {args.project}")
    except Exception as e:
        logger.error(f"Failed to activate project '{args.project}': {e}")
        print(f"❌ Failed to activate project context '{args.project}': {e}")
        sys.exit(1)

    try:
        results = sync_metrics_to_function_manager(
            dry_run=args.dry_run,
            overwrite=args.overwrite_functions,
            verbose=args.verbose,
            include_helpers=not args.no_helpers,
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
