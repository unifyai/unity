"""
Cross-reference queries between repairs and telematics.

Demonstrates multiple tool calls with custom Python logic in between.
"""

from __future__ import annotations

from typing import Any, Dict

from intranet.core.bespoke_repairs_agent import register
from intranet.core.trajectory_executor import execute_tool


@register(
    "operative_repairs_vs_trips",
    "Cross-reference repairs completed vs telematics trips for an operative",
)
async def operative_repairs_vs_trips(tools: Dict[str, Any], operative: str):
    """
    Cross-reference repairs with telematics trips for a specific operative.

    Parameters
    ----------
    tools : Dict
        Tools from FileManager
    operative : str
        Operative name to look up

    Returns
    -------
    Dict
        Summary with repair count, trip count, and dates
    """
    # First tool call - get repairs for operative
    repairs = await execute_tool(
        "_filter",
        {
            "table": "Repairs",
            "filter": (
                f"OperativeWhoCompletedJob == '{operative}' "
                "AND CompletionStatus == 'Completed'"
            ),
            "columns": ["CompletedDate", "JobTicketReference", "FullAddress"],
        },
        tools,
    )

    # Custom Python - extract repair dates
    repair_rows = repairs if isinstance(repairs, list) else repairs.get("rows", [])
    repair_dates = {
        r.get("CompletedDate") for r in repair_rows if r.get("CompletedDate")
    }

    # Second tool call - get telematics trips
    # Vehicle column contains operative name in the format: "REG - OperativeName"
    telematics = await execute_tool(
        "_filter",
        {
            "table": "Telematics_July_2025",
            "filter": f"Vehicle LIKE '%{operative}%'",
            "columns": ["Vehicle", "Departure", "Arrival"],
        },
        tools,
    )

    # Custom Python - extract trip dates
    trip_rows = (
        telematics if isinstance(telematics, list) else telematics.get("rows", [])
    )
    trip_dates = set()
    for t in trip_rows:
        departure = t.get("Departure", "")
        if departure:
            # Extract date from datetime string
            date_part = (
                departure.split("T")[0] if "T" in departure else departure.split(" ")[0]
            )
            trip_dates.add(date_part)

    # Return combined analysis
    return {
        "operative": operative,
        "total_repairs": len(repair_rows),
        "total_trips": len(trip_rows),
        "repair_dates": sorted(repair_dates),
        "trip_dates": sorted(trip_dates),
        "days_with_both": sorted(repair_dates & trip_dates),
    }
