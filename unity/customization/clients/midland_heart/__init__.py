"""
Midland Heart — client customization.

Pre-seeds the assistant with:
- A thin role identity (ActorConfig.guidelines)
- 22 stored functions: 11 repairs KPI metrics + 11 shared helpers,
  all using ``data_primitives`` against ``MidlandHeart/*`` contexts
- Four guidance entries covering KPI workflows, data schemas, and discovery

Data lives in ``MidlandHeart/Repairs2025`` and
``MidlandHeart/Telematics2025/<Month>`` contexts, seeded externally via
``seed_data.seed_all()`` using ``DataManager.ingest()``.
"""

from pathlib import Path

from unity.customization.configs.types.actor_config import ActorConfig
from unity.customization.clients import register_org

# ---------------------------------------------------------------------------
# Actor config — thin role identity only
# ---------------------------------------------------------------------------

_MH_CONFIG = ActorConfig(
    guidelines=(
        "You are the Midland Heart Repairs & Maintenance Assistant, "
        "specialising in reactive repairs KPI analysis, telematics fleet "
        "management, and property maintenance performance reporting."
    ),
)

# ---------------------------------------------------------------------------
# Pre-seeded functions (via function_dir)
# ---------------------------------------------------------------------------

_MH_FUNCTION_DIR = Path(__file__).parent / "functions"

# ---------------------------------------------------------------------------
# Guidance entries — focused, independently searchable recipes
# ---------------------------------------------------------------------------

_KPI_WORKFLOW_GUIDANCE = """\
Analyse Midland Heart repairs performance using the pre-registered KPI
metric functions.

Available KPI functions — all accept (data_primitives, group_by=None,
start_date=None, end_date=None, time_period="day", include_plots=False):
  repairs_kpi_jobs_completed, repairs_kpi_jobs_issued,
  repairs_kpi_no_access_rate (+ return_absolute),
  repairs_kpi_first_time_fix_rate (+ return_absolute),
  repairs_kpi_follow_on_required_rate (+ return_absolute),
  repairs_kpi_follow_on_materials_rate (+ return_absolute),
  repairs_kpi_job_completed_on_time_rate (+ return_absolute),
  repairs_kpi_jobs_requiring_materials_rate (+ return_absolute),
  repairs_kpi_avg_repairs_per_property,
  repairs_kpi_appointment_adherence_rate (+ return_absolute),
  repairs_kpi_total_distance_travelled

Workflow:
1. Call the relevant KPI function, optionally with group_by for breakdowns.
   group_by accepts: "operative", "patch", "region", "trade", "day".
2. Pass start_date/end_date (YYYY-MM-DD) for date-range filtering.
3. Set include_plots=True to generate accompanying bar charts.
4. Rate metrics accept return_absolute=True for raw counts instead of %.
5. Results are standardised dicts with keys: metric_name, group_by,
   time_period, start_date, end_date, results, total, metadata, plots.
6. Compose multiple KPI calls for dashboard-style summaries.

Helper functions for building custom queries:
  discover_repairs_table, discover_telematics_tables,
  resolve_group_by, build_filter,
  extract_count, extract_sum, normalize_grouped_result,
  compute_percentage, build_metric_result,
  extract_plot_url, extract_plot_succeeded

All data lives in MidlandHeart/* contexts — use data_primitives
exclusively (not files_primitives).\
"""

_REPAIRS_SCHEMA_GUIDANCE = """\
Repairs data schema — MidlandHeart/Repairs2025 context.

One row per job ticket line (a works order may have multiple ticket lines).
Source: MDH Repairs Data July–November 2025.

Key columns:
  JobTicketReference (str) — unique ticket line ID (e.g. "3866651/2/T1")
  WorksOrderRef (str) — parent works order (e.g. "3866651/2")
  WorksOrderRaisedDate (datetime) — when the order was raised
  WorksOrderIssuedDate (datetime) — when issued to operative
  WorksOrderStatusDescription (str) — "Closed", "Open", "Cancelled", etc.
  WorksOrderPriorityDescription (str) — priority level
  WorksOrderTargetDate (datetime) — contractual completion deadline
  WorksOrderReportedCompletedDate (datetime) — actual completion
  WorksOrderDescription (str) — free-text description of the fault
  OperativeName (str) — assigned operative
  PropertyReference (str) — property identifier
  FullAddress (str) — full property address
  ScheduledAppointmentStart/End (datetime) — booked appointment window
  ArrivedOnSite (datetime) — operative arrival time
  CompletedVisit (datetime) — visit completion time
  NoAccess (bool) — True if operative could not gain access
  FollowOn (bool) — True if a follow-on visit was required
  FirstTimeFix (bool) — True if fixed on first visit
  SDRCategoryA/B (str) — Standard Description of Repairs classification
  RepairsRegion (str) — geographical region
  RepairsPatch (str) — operative patch assignment
  PropertyType (str) — dwelling type
  NumberOfBedrooms (int) — bedroom count\
"""

_TELEMATICS_SCHEMA_GUIDANCE = """\
Telematics data schema — MidlandHeart/Telematics2025/<Month> contexts.

One row per trip/sub-trip. Source: MDH Telematics Data July–November 2025.
Each month is a separate context (July, August, September, October, November).

Columns:
  Trip (int) — trip sequence number within the month
  Driver (str) — driver name and ID (may be null for unassigned)
  Vehicle (str) — vehicle registration and assigned driver
  Departure (datetime) — trip start time
  Arrival (datetime) — trip end time
  StartLocation (str) — departure address
  EndLocation (str) — arrival address
  Subtrip travel time (timedelta) — duration of sub-trip segment
  Trip travel time (timedelta) — total trip duration
  Idling time (timedelta) — time spent idling
  Business distance (float) — business miles
  Private distance (float) — private miles
  Total distance (float) — total miles
  Avg speed (float) — average speed
  Max speed (float) — maximum speed recorded

Use discover_telematics_tables(data_primitives) to list available months.
Sum across months for aggregate fleet metrics.\
"""

_DISCOVERY_GUIDANCE = """\
Data discovery guide for Midland Heart contexts.

All Midland Heart data lives under the MidlandHeart/ prefix:
  MidlandHeart/Repairs2025 — reactive repairs (all months combined)
  MidlandHeart/Telematics2025/July — July 2025 vehicle trips
  MidlandHeart/Telematics2025/August — August 2025 vehicle trips
  MidlandHeart/Telematics2025/September — September 2025 vehicle trips
  MidlandHeart/Telematics2025/October — October 2025 vehicle trips
  MidlandHeart/Telematics2025/November — November 2025 vehicle trips

Discovery via data_primitives:
  primitives.data.describe_table("MidlandHeart/Repairs2025")
    → returns schema, column types, row count
  primitives.data.list_tables(prefix="MidlandHeart/Telematics2025")
    → returns list of monthly telematics context paths

For ad-hoc queries beyond pre-registered KPIs, use:
  primitives.data.filter(context, filter=...) — row-level filtering
  primitives.data.reduce(context, metric=..., column=...) — aggregation
  primitives.data.plot(contexts=[...], plot_type=...) — charting
  primitives.data.table_view(context, ...) — interactive table views\
"""

_MH_GUIDANCE = [
    {
        "title": "Repairs KPI Analysis Workflow",
        "content": _KPI_WORKFLOW_GUIDANCE,
    },
    {
        "title": "Repairs Data Schema",
        "content": _REPAIRS_SCHEMA_GUIDANCE,
    },
    {
        "title": "Telematics Data Schema",
        "content": _TELEMATICS_SCHEMA_GUIDANCE,
    },
    {
        "title": "Data Discovery Guide",
        "content": _DISCOVERY_GUIDANCE,
    },
]

# ---------------------------------------------------------------------------
# Org-level registration (Midland Heart, org ID 3)
# ---------------------------------------------------------------------------

_MH_ORG_ID = 3

register_org(
    _MH_ORG_ID,
    config=_MH_CONFIG,
    function_dir=_MH_FUNCTION_DIR,
    guidance=_MH_GUIDANCE,
)
