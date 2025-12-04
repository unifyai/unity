#!/usr/bin/env python3
"""run_bespoke_repairs_query.py
Run bespoke queries using the LLM-free BespokeRepairsAgent.

Usage:
    python scripts/run_bespoke_repairs_query.py --list
    python scripts/run_bespoke_repairs_query.py --query repairs_per_day_per_operative
    python scripts/run_bespoke_repairs_query.py --query repairs_per_day_per_operative --params '{"start_date": "2025-07-01", "end_date": "2025-07-31"}'
    python scripts/run_bespoke_repairs_query.py --query operative_repairs_vs_trips --params '{"operative": "John Smith"}'
"""

import asyncio
import argparse
import json
import sys

from utils import initialize_script_environment, activate_project

# ---------------------------------------------------------------------------
# Boot-strap env / PYTHONPATH
# ---------------------------------------------------------------------------
if not initialize_script_environment():
    sys.exit(1)

# Import after environment is set up
from intranet.core.bespoke_repairs_agent import BespokeRepairsAgent

# Trigger query registration
import intranet.repairs.queries  # noqa


async def main():
    parser = argparse.ArgumentParser(
        description="Run bespoke queries with the LLM-free BespokeRepairsAgent",
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
    args = parser.parse_args()

    # Activate project context
    activate_project("Intranet")

    agent = BespokeRepairsAgent()

    # List queries
    if args.list:
        print("\nAvailable queries:")
        print("-" * 60)
        for q in agent.list_queries():
            print(f"  {q['query_id']}")
            print(f"    {q['description']}")
            print()
        return

    # Execute query
    if not args.query:
        parser.error("Either --list or --query is required")

    # Parse params
    try:
        params = json.loads(args.params)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON params: {e}")
        sys.exit(1)

    print(f"\n🔍 Executing query: {args.query}")
    if params:
        print(f"   Parameters: {params}")
    print("-" * 60)

    try:
        result = await agent.ask(args.query, **params)

        if args.pretty:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(result)

        print("-" * 60)
        print("✅ Query completed successfully")

    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Query failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
