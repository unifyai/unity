"""
Daily repairs queries.

Each query is a registered async function. Use execute_tool/execute_tools
for tool calls and write any custom Python logic you need.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict

from intranet.core.bespoke_repairs_agent import register
from intranet.core.trajectory_executor import execute_tool


@register(
    "repairs_per_day_per_operative",
    "Total repairs completed per day per operative",
)
async def repairs_per_day_per_operative(
    tools: Dict[str, Any],
    start_date: str = None,
    end_date: str = None,
):
    """
    Get repairs per day per operative.

    Parameters
    ----------
    tools : Dict
        Tools from FileManager
    start_date : str, optional
        Start date filter (YYYY-MM-DD)
    end_date : str, optional
        End date filter (YYYY-MM-DD)

    Returns
    -------
    List[Dict]
        Aggregated results: [{"date": ..., "operative": ..., "count": ...}, ...]
    """
    # Build filter
    filter_expr = "CompletionStatus == 'Completed'"
    if start_date and end_date:
        filter_expr += (
            f" AND CompletedDate >= '{start_date}' AND CompletedDate <= '{end_date}'"
        )

    # Tool call
    data = await execute_tool(
        "_filter",
        {
            "table": "Repairs",
            "filter": filter_expr,
            "columns": ["CompletedDate", "OperativeWhoCompletedJob"],
        },
        tools,
    )

    # Custom aggregation logic
    rows = data if isinstance(data, list) else data.get("rows", [])
    counts = defaultdict(int)
    for row in rows:
        key = (row.get("CompletedDate"), row.get("OperativeWhoCompletedJob"))
        counts[key] += 1

    # Return aggregated results
    return [
        {"date": k[0], "operative": k[1], "count": v} for k, v in sorted(counts.items())
    ]


@register(
    "repairs_by_status",
    "Breakdown of repairs by completion status",
)
async def repairs_by_status(tools: Dict[str, Any]):
    """
    Get breakdown of repairs by completion status.

    Returns
    -------
    Dict[str, int]
        Status -> count mapping
    """
    # Tool call
    data = await execute_tool(
        "_filter",
        {
            "table": "Repairs",
            "columns": ["CompletionStatus"],
        },
        tools,
    )

    # Custom aggregation logic
    rows = data if isinstance(data, list) else data.get("rows", [])
    status_counts = defaultdict(int)
    for row in rows:
        status = row.get("CompletionStatus", "Unknown")
        status_counts[status] += 1

    return dict(status_counts)
