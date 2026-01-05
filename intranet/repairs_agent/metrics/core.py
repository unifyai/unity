"""
Core metric functions for repairs analysis.

This module will contain all 16 refactored metric functions after migration
from intranet/repairs/queries/metrics.py.

Each metric follows the discovery-first pattern and uses helpers from helpers.py.

Metrics:
--------
1.  jobs_completed_per_day - Jobs completed per man per day
2.  no_access_rate - No Access % / Absolute number
3.  first_time_fix_rate - First Time Fix % / Absolute Number
4.  follow_on_required_rate - Follow on Required % / Absolute Number
5.  follow_on_materials_rate - Follow on Required for Materials %
6.  job_completed_on_time_rate - Job completed on time % / Absolute Number
7.  merchant_stops_per_day - No of merchant stops per day
8.  avg_duration_at_merchant - Average duration at a Merchant per day
9.  distance_travelled_per_day - Distance Travelled per day
10. avg_time_travelling - Average time travelling per day
11. repairs_completed_per_day - Repairs completed per day
12. jobs_issued_per_day - Jobs issued per day
13. jobs_requiring_materials_rate - % of jobs completed that require materials
14. avg_repairs_per_property - Average no of repairs per property completed
15. complaints_rate - Complaints as % of total jobs completed
16. appointment_adherence_rate - Appointment adherence rate

Status: PENDING MIGRATION (Phase 2-3)
Currently metrics are in: intranet/repairs/queries/metrics.py
"""

from __future__ import annotations

# Placeholder - metrics will be migrated here in Phase 3
# For now, metrics are still in intranet/repairs/queries/metrics.py
# and register to intranet.repairs_agent.static.registry

# Future imports after migration:
# from .helpers import (
#     discover_repairs_table,
#     discover_telematics_tables,
#     resolve_group_by,
#     build_filter,
#     extract_count,
#     normalize_grouped_result,
#     compute_percentage,
# )
# from .types import GroupBy, MetricResult, TimePeriod, ToolsDict
# from intranet.repairs_agent.static.registry import register
