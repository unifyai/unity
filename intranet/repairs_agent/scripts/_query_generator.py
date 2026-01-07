#!/usr/bin/env python3
"""
Query generator for parallel repairs query execution.

Generates query specifications with parameter combinations for parallel execution.
Output is JSON for easy parsing by shell scripts.

Usage:
    python _query_generator.py --list-queries          # List available queries
    python _query_generator.py --all                   # All queries with default params
    python _query_generator.py --all --expand-params   # All queries × all param combinations
    python _query_generator.py --all --full-matrix     # Full matrix: all queries × all params, nested dirs
    python _query_generator.py --query jobs_completed
    python _query_generator.py --query jobs_completed --expand-params
"""

import json
import sys
import argparse
from typing import Any, Dict, List

# Available queries and their descriptions
# NOTE: Skipped metrics (merchant_stops, merchant_dwell_time, travel_time, complaints_rate)
# are NOT included here as they cannot be executed.
QUERIES = {
    "jobs_completed": "Jobs completed (groupable by operative/patch/region/day)",
    "no_access_rate": "No Access % / Absolute number",
    "first_time_fix_rate": "First Time Fix % / Absolute number",
    "follow_on_required_rate": "Follow on Required % / Absolute number",
    "follow_on_materials_rate": "Follow on Required for Materials %",
    "job_completed_on_time_rate": "Job completed on time % / Absolute number",
    "jobs_issued": "Jobs issued (groupable by operative/patch/region/day)",
    "jobs_requiring_materials_rate": "% of jobs that require materials",
    "avg_repairs_per_property": "Average repairs per property",
    "appointment_adherence_rate": "Appointment adherence rate",
    "total_distance_travelled": "Total distance travelled (groupable by vehicle/day)",
}

# Parameter variations for different query types
# NOTE: "day" added for temporal grouping; "total" removed (use None for no grouping)
GROUP_BY_VALUES = ["operative", "patch", "region", "day"]

# Queries that support group_by parameter (repairs data)
GROUPABLE_QUERIES = {
    "jobs_completed",
    "no_access_rate",
    "first_time_fix_rate",
    "follow_on_required_rate",
    "follow_on_materials_rate",
    "job_completed_on_time_rate",
    "jobs_issued",
    "jobs_requiring_materials_rate",
    "appointment_adherence_rate",
}

# Queries that support return_absolute parameter
RATE_QUERIES = {
    "no_access_rate",
    "first_time_fix_rate",
    "follow_on_required_rate",
    "job_completed_on_time_rate",
    "appointment_adherence_rate",
}

# Telematics queries (different grouping - vehicle/day only)
TELEMATICS_QUERIES = {
    "total_distance_travelled",
}


def generate_param_combinations(query_id: str) -> List[Dict[str, Any]]:
    """Generate all parameter combinations for a query.

    NOTE: Does NOT expand include_plots - that's controlled by --include-plots flag.
    """
    combinations = []

    if query_id in GROUPABLE_QUERIES:
        # Generate combinations for each group_by value
        for group_by in GROUP_BY_VALUES:
            params = {"group_by": group_by}

            if query_id in RATE_QUERIES:
                # For rate queries, test both percentage and absolute
                combinations.append({**params, "return_absolute": False})
                combinations.append({**params, "return_absolute": True})
            else:
                combinations.append(params)
    elif query_id in TELEMATICS_QUERIES:
        # Telematics queries: operative (vehicle) or day grouping
        combinations.append({"group_by": "operative"})
        combinations.append({"group_by": "day"})
    else:
        # Queries without special parameters
        combinations.append({})

    return combinations


def get_default_params(query_id: str) -> Dict[str, Any]:
    """Get default parameters for a query."""
    if query_id in GROUPABLE_QUERIES:
        return {"group_by": "operative"}
    elif query_id in TELEMATICS_QUERIES:
        return {"group_by": "operative"}
    return {}


def _params_to_log_filename(params: Dict[str, Any]) -> str:
    """Convert params dict to a log filename component."""
    if not params:
        return "default"
    parts = []
    for key, value in sorted(params.items()):
        # Format value as string
        if isinstance(value, bool):
            val_str = "true" if value else "false"
        else:
            val_str = str(value)
        parts.append(f"{key}_{val_str}")
    return "__".join(parts)


def generate_query_specs(
    query_ids: List[str],
    expand_params: bool = False,
    full_matrix: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate query specifications.

    Args:
        query_ids: List of query IDs to generate specs for
        expand_params: If True, generate all parameter combinations
        full_matrix: If True, generate all combinations with metric subdirs
                    (implies expand_params=True)

    Returns:
        List of query specs with query_id, params, and description.
        In full_matrix mode, also includes 'metric_subdir' for nested logging.
    """
    specs = []

    # full_matrix implies expand_params
    if full_matrix:
        expand_params = True

    for query_id in query_ids:
        if query_id not in QUERIES:
            continue

        description = QUERIES[query_id]

        if expand_params:
            param_combos = generate_param_combinations(query_id)
        else:
            param_combos = [get_default_params(query_id)]

        for params in param_combos:
            spec = {
                "query_id": query_id,
                "params": params,
                "description": description,
            }
            # Generate a unique session name
            if params:
                param_suffix = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
                spec["session_name"] = f"{query_id}__{param_suffix}"
            else:
                spec["session_name"] = query_id

            # In full_matrix mode, add metric_subdir for nested directory structure
            if full_matrix:
                spec["metric_subdir"] = query_id
                # Also add a log filename hint
                spec["log_filename"] = _params_to_log_filename(params)

            specs.append(spec)

    return specs


def main():
    parser = argparse.ArgumentParser(
        description="Generate query specifications for parallel execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list-queries",
        action="store_true",
        help="List available query IDs",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate specs for all queries",
    )
    parser.add_argument(
        "--query",
        type=str,
        action="append",
        dest="queries",
        help="Query ID to include (can be repeated)",
    )
    parser.add_argument(
        "--expand-params",
        action="store_true",
        help="Expand all parameter combinations",
    )
    parser.add_argument(
        "--full-matrix",
        action="store_true",
        help="Full matrix mode: all queries × all params with metric subdirs for nested logging",
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl", "names"],
        default="json",
        help="Output format (default: json)",
    )

    args = parser.parse_args()

    if args.list_queries:
        for query_id, desc in sorted(QUERIES.items()):
            print(f"{query_id}: {desc}")
        return 0

    # Determine which queries to process
    if args.all:
        query_ids = list(QUERIES.keys())
    elif args.queries:
        query_ids = args.queries
    else:
        parser.print_help()
        return 1

    # Generate specs
    specs = generate_query_specs(
        query_ids,
        expand_params=args.expand_params,
        full_matrix=args.full_matrix,
    )

    # Output
    if args.format == "json":
        print(json.dumps(specs, indent=2))
    elif args.format == "jsonl":
        for spec in specs:
            print(json.dumps(spec))
    elif args.format == "names":
        for spec in specs:
            print(spec["session_name"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
