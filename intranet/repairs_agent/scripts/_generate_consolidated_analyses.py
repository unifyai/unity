#!/usr/bin/env python3
"""
Generate consolidated analyses for full-matrix runs.

This script reads all log files in a metric subdirectory, extracts the raw results,
and generates a single consolidated _analysis.md file with insights across all
parameter combinations.

Usage:
    python _generate_consolidated_analyses.py <run_log_dir>

Example:
    python _generate_consolidated_analyses.py .repairs_queries/2026-01-07T13-29-54_repairs_dev_pts_30/
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add repo root to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPAIRS_AGENT_DIR = SCRIPT_DIR.parent
INTRANET_DIR = REPAIRS_AGENT_DIR.parent
REPO_ROOT = INTRANET_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from intranet.scripts.utils import initialize_script_environment

if not initialize_script_environment():
    print("❌ Failed to initialize script environment")
    sys.exit(1)

from unity.common.llm_client import new_llm_client

from intranet.repairs_agent.config.prompt_builder import (
    build_analyst_system_prompt,
    build_consolidated_analyst_prompt,
)
from intranet.repairs_agent.static.registry import get_query


def extract_raw_data_from_log(content: str) -> Optional[Dict[str, Any]]:
    """Extract the RESULT JSON section from a log file."""
    # Look for RESULT section (the actual format in our logs)
    match = re.search(
        r"RESULT\n-+\n(\{.+?\})\n\n-+",
        content,
        re.DOTALL,
    )
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Fallback: look for any JSON object with metric_name key
    match = re.search(
        r'(\{"metric_name":.+?"plots":\s*\[.*?\]\s*\})',
        content,
        re.DOTALL,
    )
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None


def extract_params_from_log(content: str) -> Dict[str, Any]:
    """Extract parameters from a log file."""
    params = {}
    match = re.search(r"PARAMETERS\n-+\n(.+?)\n\n", content, re.DOTALL)
    if match:
        for line in match.group(1).strip().split("\n"):
            line = line.strip()
            if line == "(none)":
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                # Try to parse as JSON for booleans/numbers
                try:
                    value = json.loads(value.lower())
                except (json.JSONDecodeError, AttributeError):
                    pass
                params[key] = value
    return params


def extract_params_from_filename(filename: str) -> Dict[str, Any]:
    """Extract parameters from log filename as fallback."""
    params = {}
    # Pattern: metric__param1_value1_param2_value2.log
    parts = filename.replace(".log", "").split("__")
    if len(parts) > 1:
        param_str = parts[1]
        # Split on underscores, pair up key-value
        tokens = param_str.split("_")
        i = 0
        while i < len(tokens) - 1:
            key = tokens[i]
            value = tokens[i + 1]
            # Try to parse value
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            params[key] = value
            i += 2
    return params


async def generate_consolidated_analysis(
    metric_dir: Path,
    metric_name: str,
    metric_description: str,
    metric_docstring: Optional[str] = None,
) -> Optional[Path]:
    """
    Generate consolidated analysis for all combinations of a metric.

    Reads all .log files in the metric directory, extracts raw results,
    and generates a single consolidated _analysis.md file.
    """
    # Collect all results from log files
    all_results: List[Dict[str, Any]] = []

    for log_file in sorted(metric_dir.glob("*.log")):
        if log_file.name.startswith("_"):
            continue  # Skip summary files

        try:
            content = log_file.read_text()
        except Exception as e:
            print(f"    ⚠ Could not read {log_file.name}: {e}")
            continue

        # Extract raw data
        raw_data = extract_raw_data_from_log(content)
        if not raw_data:
            # Skip files without raw data (might be error logs)
            continue

        # Extract params
        params = extract_params_from_log(content)
        if not params:
            # Fallback to filename parsing
            params = extract_params_from_filename(log_file.name)

        all_results.append(
            {
                "params": params,
                "raw_results": raw_data,
                "log_file": log_file.name,
            },
        )

    if not all_results:
        return None

    # Build prompts
    system_prompt = build_analyst_system_prompt()
    user_prompt = build_consolidated_analyst_prompt(
        metric_name=metric_name,
        metric_description=metric_description,
        metric_docstring=metric_docstring,
        all_results=all_results,
    )

    # Call LLM
    client = new_llm_client()
    response = await client.generate(
        user_message=user_prompt,
        system_message=system_prompt,
    )

    # Write analysis file
    analysis_path = metric_dir / "_analysis.md"
    analysis_content = f"""# {metric_name.replace('_', ' ').title()} - Consolidated Analysis

> Generated from {len(all_results)} parameter combinations
> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

---

{response}

---

## Data Sources

| # | Parameters | Log File |
|---|------------|----------|
"""

    for i, item in enumerate(all_results, 1):
        params_str = (
            ", ".join(f"{k}={v}" for k, v in item["params"].items()) or "(defaults)"
        )
        analysis_content += f"| {i} | {params_str} | `{item['log_file']}` |\n"

    analysis_path.write_text(analysis_content)

    return analysis_path


async def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate consolidated analyses for full-matrix runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate analyses for all metrics in a run
  python _generate_consolidated_analyses.py .repairs_queries/2026-01-07T13-29-54_repairs_dev_pts_30/

  # Generate analysis for specific metrics only
  python _generate_consolidated_analyses.py .repairs_queries/2026-01-07T13-29-54_repairs_dev_pts_30/ \\
    --metrics total_distance_travelled avg_repairs_per_property
""",
    )
    parser.add_argument(
        "run_log_dir",
        type=Path,
        help="Path to the run log directory containing metric subdirectories",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        help="Only generate analyses for these specific metrics (by name)",
    )

    args = parser.parse_args()

    run_dir = args.run_log_dir
    filter_metrics = set(args.metrics) if args.metrics else None

    if not run_dir.exists():
        print(f"❌ Directory not found: {run_dir}")
        return 1

    if filter_metrics:
        print(
            f"📊 Generating analyses for {len(filter_metrics)} metric(s) in: {run_dir.name}",
        )
    else:
        print(f"📊 Generating consolidated analyses for: {run_dir.name}")
    print()

    # Trigger metric registration
    import intranet.repairs_agent.metrics  # noqa: F401

    # Find all metric subdirectories
    generated_count = 0
    skipped_count = 0

    for metric_dir in sorted(run_dir.iterdir()):
        if not metric_dir.is_dir():
            continue
        if metric_dir.name.startswith("_"):
            continue

        metric_name = metric_dir.name

        # Skip if filtering and metric not in filter list
        if filter_metrics and metric_name not in filter_metrics:
            continue

        spec = get_query(metric_name)

        if not spec:
            print(f"  ⚠ Unknown metric: {metric_name} (skipping)")
            skipped_count += 1
            continue

        print(f"  📈 {metric_name}...", end=" ", flush=True)

        try:
            docstring = inspect.getdoc(spec.fn)

            analysis_path = await generate_consolidated_analysis(
                metric_dir=metric_dir,
                metric_name=metric_name,
                metric_description=spec.description,
                metric_docstring=docstring,
            )

            if analysis_path:
                print(f"✓ {analysis_path.name}")
                generated_count += 1
            else:
                print("⚠ No results found")
                skipped_count += 1

        except Exception as e:
            print(f"✗ Error: {e}")
            skipped_count += 1

    print()
    print(f"✅ Generated {generated_count} consolidated analyses")
    if skipped_count:
        print(f"⚠  Skipped {skipped_count} metrics")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
