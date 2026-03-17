"""
Midland Heart — client customization.

Pre-seeds the assistant with:
- A thin role identity (ActorConfig.guidelines)
- 22 stored functions: 11 repairs KPI metrics + 11 shared helpers,
  all using ``data_primitives`` against ``MidlandHeart/*`` contexts
- Five guidance entries covering KPI workflows, data schemas, discovery,
  and cross-referencing patterns

Data lives in ``MidlandHeart/Repairs2025`` and
``MidlandHeart/Telematics2025/<Month>`` contexts, seeded externally via
``ingest_dm.py`` using ``DataManager.ingest()``.

Registered at assistant level (agent_id 690) so only this specific
assistant receives the customization.
"""

from pathlib import Path

from unity.customization.configs.types.actor_config import ActorConfig
from unity.customization.clients import register_assistant

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
How to analyse Midland Heart repairs performance using pre-registered
KPI metric functions.

These functions are stored in FunctionManager as memoized compositional
functions. Each takes ``data_primitives`` (i.e. ``primitives.data``) as
its first argument and returns a standardised result dict.

Productivity metrics:
  repairs_kpi_jobs_completed — count of completed repair jobs
  repairs_kpi_jobs_issued — count of issued/incoming repair jobs

Quality and efficiency rates (accept return_absolute=True for raw counts):
  repairs_kpi_first_time_fix_rate — % fixed on first visit
  repairs_kpi_no_access_rate — % where operative couldn't access property
  repairs_kpi_follow_on_required_rate — % requiring follow-on visits
  repairs_kpi_follow_on_materials_rate — % of follow-ons due to materials
  repairs_kpi_jobs_requiring_materials_rate — % needing materials (proxy)
  repairs_kpi_job_completed_on_time_rate — % completed by target date
  repairs_kpi_appointment_adherence_rate — % arriving in appointment window

Property and fleet:
  repairs_kpi_avg_repairs_per_property — repeat-repair analysis by address
  repairs_kpi_total_distance_travelled — total business miles (telematics)

Common parameters (all functions):
  group_by: "operative" | "patch" | "region" | "trade" | "day" | None
  start_date / end_date: "YYYY-MM-DD" bounds (data covers Jul–Nov 2025)
  include_plots: True to generate bar charts via primitives.data.plot()
  time_period: metadata label (default "day")

Calling convention:
  result = await repairs_kpi_jobs_completed(
      data_primitives=primitives.data,
      group_by="operative",
      start_date="2025-07-01",
      end_date="2025-11-30",
      include_plots=True,
  )

Output shape (all metrics):
  {metric_name, group_by, time_period, start_date, end_date,
   results (list of per-group dicts), total, metadata, plots}

For dashboard-style summaries, call multiple KPI functions and combine
their results. Rate metrics accept return_absolute=True to get raw
numerator/denominator counts instead of percentages.

Helper functions (also in FunctionManager, for building custom queries):
  discover_repairs_table — returns context path + schema for Repairs2025
  discover_telematics_tables — lists monthly telematics context paths
  resolve_group_by — maps "operative" etc. to actual column names
  build_filter — composes filter expressions with date-range bounds
  extract_count / extract_sum — parse reduce() results
  normalize_grouped_result — flatten grouped reduce output
  compute_percentage / build_metric_result — formatting utilities

All Midland Heart data lives in MidlandHeart/* contexts and is accessed
via data_primitives (primitives.data).\
"""

_REPAIRS_SCHEMA_GUIDANCE = """\
Repairs data schema — MidlandHeart/Repairs2025 context.

One row per job ticket line. A works order may have multiple ticket
lines. Source: MDH Repairs Data July–November 2025.

Identity and tracking:
  JobTicketReference — unique ticket line ID, e.g. "3866651/2/T1".
    Use for deduplication; WorksOrderRef duplicates may be valid follow-ons.
  WorksOrderRef — parent works order, e.g. "3866651/2"

Dates and lifecycle:
  WorksOrderRaisedDate — when the order was created
  WorksOrderIssuedDate — when issued to operative
  WorksOrderTargetDate — contractual completion deadline
  WorksOrderReportedCompletedDate — actual completion timestamp
  VisitDate — when operative accepted the job on tablet
  ScheduledAppointmentStart — booked appointment window start
  ScheduledAppointmentEnd — booked appointment window end
  ArrivedOnSite — operative arrival (tablet GPS timestamp)
  CompletedVisit — visit completion (tablet timestamp)
  JobTicketLinePlannedStartDate — scheduling team planned start
  JobTicketLinePlannedEndDate — scheduling team planned end

Status and description:
  WorksOrderStatusDescription — "Complete", "Closed", "Open",
    "Cancelled", "Issued", etc. "Complete" = operative finished;
    "Closed" = financials completed. Both count as done for KPIs.
  WorksOrderPriorityDescription — priority level
  WorksOrderDescription — free-text fault description

Operative and assignment:
  OperativeName — assigned operative name
  OperativeWhoCompletedJob — operative who actually completed it.
    Use this column for group_by="operative" and for matching to the
    Vehicle column in telematics data.
  Trade — operative trade classification (plumber, electrician, etc.)

Visit outcome flags:
  NoAccess — reason if operative couldn't access property (None = accessed)
  FollowOn — "Yes" if a follow-on visit was required
  FollowOnDescription — reason for follow-on (includes "MATERIALS REQUIRED")
  FollowOnNotes — additional follow-on notes
  FirstTimeFix — "Yes" if fixed on first visit
  SecondTimeFix — indicator if fixed on second visit
  ThirdTimeFix — indicator if fixed on third visit

Property and geography:
  PropertyReference — property identifier
  FullAddress — full property address (matchable with telematics locations)
  RepairsRegion — North, South, Central North, Central South, etc.
  RepairsPatch — sub-region patch assignment
  PropertyType — dwelling type
  NumberOfBedrooms — bedroom count
  SchemeName — housing scheme/estate name
  LocalAuthorityName — local authority
  WardName — electoral ward

Classification:
  SDRCategoryA — Standard Description of Repairs, level A
  SDRCategoryB — Standard Description of Repairs, level B

Business rules:
- "Completed jobs" = WorksOrderStatusDescription in {Complete, Closed}.
- ArrivedOnSite and CompletedVisit are tablet timestamps; some operatives
  click through quickly, producing very short visit durations.
- VisitDate is the default date column for date-range filtering in KPIs.
- Derived day columns exist for temporal grouping:
  WorksOrderReportedCompletedDate_Day, WorksOrderIssuedDate_Day,
  ArrivedOnSite_Date, ScheduledAppointmentStart_Date.\
"""

_TELEMATICS_SCHEMA_GUIDANCE = """\
Telematics data schema — MidlandHeart/Telematics2025/<Month> contexts.

One row per trip or sub-trip segment. Source: MDH Telematics Data
July–November 2025. Each month is a separate context:
  MidlandHeart/Telematics2025/July
  MidlandHeart/Telematics2025/August
  MidlandHeart/Telematics2025/September
  MidlandHeart/Telematics2025/October
  MidlandHeart/Telematics2025/November

Columns:
  Trip — trip sequence number within the month
  Driver — driver name (may be abbreviated or null for unassigned)
  Vehicle — vehicle registration + operative full name. This is the
    best identifier for linking to OperativeWhoCompletedJob in
    repairs data.
  Departure — trip start time
  Arrival — trip end time
  StartLocation — departure address (matchable with FullAddress)
  EndLocation — arrival address (may include "Stopped with Ignition
    ON" events)
  Subtrip travel time — duration of sub-trip segment
  Trip travel time — total trip duration
  Idling time — engine running but stationary
  Business distance — business miles
  Private distance — private miles (typically 0)
  Total distance — total miles
  Avg speed — average speed in mph
  Max speed — maximum speed in mph

Usage notes:
- Use discover_telematics_tables(data_primitives) to list available months.
- Sum across months for aggregate fleet metrics (the KPI function
  repairs_kpi_total_distance_travelled does this automatically).
- Distance and speed fields may be empty for non-travel segments
  (stop events, ignition-on idling).
- For group_by="operative" on telematics, the Vehicle column is used
  (since it contains the operative's full name).\
"""

_DISCOVERY_AND_COMPOSITION_GUIDANCE = """\
Data discovery, composition patterns, and worked examples for Midland
Heart repairs and telematics analysis.

Context paths (all under MidlandHeart/, data covers July–November 2025):
  MidlandHeart/Repairs2025 — reactive repairs (all months, single table)
  MidlandHeart/Telematics2025/July through .../November — per-month trips

Discovery:
  primitives.data.describe_table("MidlandHeart/Repairs2025")
    → schema, column types, descriptions, row count
  primitives.data.list_tables(prefix="MidlandHeart/Telematics2025")
    → list of monthly telematics context paths
  Or use the helpers: discover_repairs_table(primitives.data),
  discover_telematics_tables(primitives.data)

Available data primitives:
  primitives.data.filter(context, filter=..., columns=[...], limit=N)
  primitives.data.reduce(context, metric="count"|"sum"|"mean", columns=..., group_by=...)
  primitives.data.search(context, query="...", columns=[...], limit=N)
  primitives.data.plot(context, plot_type="bar"|"line"|"scatter", x=..., aggregate=...)
  primitives.data.table_view(context, title=..., columns=[...], filter=...)

Composition patterns:

  1. KPI function as a building block — call one or more KPI functions
     and combine or post-process their results:

       ftf = await repairs_kpi_first_time_fix_rate(primitives.data, group_by="operative")
       na  = await repairs_kpi_no_access_rate(primitives.data, group_by="operative")
       # merge by operative name for a combined performance view

  2. KPI + raw primitive drill-down — use a KPI result to identify an
     outlier, then drill into the raw data for context:

       result = await repairs_kpi_no_access_rate(primitives.data, group_by="patch")
       worst_patch = max(result["results"], key=lambda r: r["percentage"])
       rows = await primitives.data.filter(
           "MidlandHeart/Repairs2025",
           filter=f"`RepairsPatch` == '{worst_patch['group']}' and `NoAccess` != 'None'",
           columns=["FullAddress", "OperativeWhoCompletedJob", "NoAccess"],
           limit=50,
       )

  3. Read function code to build a custom query — inspect a KPI
     function's docstring or source to understand its filter expressions,
     then adapt them for a bespoke query:

       # repairs_kpi_first_time_fix_rate uses: `FirstTimeFix` == 'Yes'
       # adapt to get FTF by property type:
       ftf_by_type = await primitives.data.reduce(
           "MidlandHeart/Repairs2025",
           metric="count", columns="JobTicketReference",
           filter="`FirstTimeFix` == 'Yes'",
           group_by="PropertyType",
       )

  4. Compose helpers into new metrics — reuse discover, build_filter,
     compute_percentage etc. to create a metric that doesn't exist yet:

       table = await discover_repairs_table(primitives.data)
       ctx = table["table"]
       f = build_filter(
           ["`WorksOrderStatusDescription` in ['Complete', 'Closed']",
            "`Trade` == 'Plumber'"],
           start_date="2025-09-01",
       )
       count = await primitives.data.reduce(ctx, metric="count",
           columns="JobTicketReference", filter=f)

  5. Cross-reference repairs and telematics — Vehicle in telematics
     contains "registration + operative full name", matchable against
     OperativeWhoCompletedJob in repairs. FullAddress in repairs can
     match StartLocation/EndLocation in telematics for visit verification.

Anti-patterns to avoid:
- Hardcoding context paths instead of using discover_repairs_table /
  discover_telematics_tables (the path may change between environments).
- Grouping by OperativeName when OperativeWhoCompletedJob is the correct
  column for who actually did the work.
- Querying telematics months one by one and manually summing when
  repairs_kpi_total_distance_travelled already handles aggregation.
- Filtering by date without specifying the date column — VisitDate is the
  default, but WorksOrderIssuedDate or ArrivedOnSite may be more
  appropriate depending on the question.
- Forgetting that NoAccess and FollowOn store string values ("Yes"/reason),
  not booleans — filter with != 'None' and != '', not == True.
- Filtering datetime columns with only != 'None' (string comparison).
  Datetime columns store actual Python None for missing values, so always
  combine both guards: != 'None' and is not None.\
"""

_BUSINESS_RULES_GUIDANCE = """\
Key business rules and data quality notes for Midland Heart data.

Completed jobs definition:
  WorksOrderStatusDescription in {"Complete", "Closed"} counts as done.
  "Complete" means the operative has finished the work. "Closed" means
  financial closure is also done. Both are valid for KPI denominators.

Deduplication:
  Use JobTicketReference for deduplication. WorksOrderRef can appear
  multiple times legitimately (multiple ticket lines per works order,
  or follow-on visits under the same order).

Timestamp reliability:
  ArrivedOnSite and CompletedVisit are captured via the operative's
  tablet. Some operatives click through quickly, leading to very short
  or near-zero visit durations. Treat suspiciously short durations
  (under 5 minutes) with caution in analysis.

Appointment data:
  ScheduledAppointmentStart/End represent the booked window.
  JobTicketLinePlannedStartDate/EndDate are set by the scheduling team
  and may differ from the appointment window.
  ScheduledAppointmentStart_Date values of "1900-01-02" indicate missing
  or placeholder data — exclude from appointment adherence calculations.

Follow-on analysis:
  FollowOnDescription contains the reason for a follow-on. The substring
  "MATERIALS REQUIRED" indicates a materials-driven follow-on (used by
  repairs_kpi_follow_on_materials_rate).

Telematics data quality:
  Distance, speed, and travel time fields may be empty or null for
  non-travel segments (idling events, ignition-on stops). Filter with
  `Business distance` != 'None' and `Business distance` is not None
  for meaningful distance aggregations.

Null handling in datetime columns:
  Datetime columns (e.g. WorksOrderReportedCompletedDate, ArrivedOnSite)
  store actual Python None for missing values, not the string "None".
  Always combine both guards: != 'None' (for string artefacts) and
  is not None (for actual nulls).

Date coverage:
  All data spans July 2025 through November 2025. Queries outside this
  range will return empty results.\
"""

_MH_GUIDANCE = [
    {
        "title": "Repairs KPI Analysis Workflow",
        "content": _KPI_WORKFLOW_GUIDANCE,
    },
    {
        "title": "Repairs Data Schema — MidlandHeart/Repairs2025",
        "content": _REPAIRS_SCHEMA_GUIDANCE,
    },
    {
        "title": "Telematics Data Schema — MidlandHeart/Telematics2025",
        "content": _TELEMATICS_SCHEMA_GUIDANCE,
    },
    {
        "title": "Data Discovery, Composition Patterns, and Examples",
        "content": _DISCOVERY_AND_COMPOSITION_GUIDANCE,
    },
    {
        "title": "Midland Heart Business Rules and Data Quality",
        "content": _BUSINESS_RULES_GUIDANCE,
    },
]

# ---------------------------------------------------------------------------
# Assistant-level registration (Haris's MH demo assistant)
# ---------------------------------------------------------------------------

_MH_ASSISTANT_ID = 690

register_assistant(
    _MH_ASSISTANT_ID,
    config=_MH_CONFIG,
    function_dir=_MH_FUNCTION_DIR,
    guidance=_MH_GUIDANCE,
)
