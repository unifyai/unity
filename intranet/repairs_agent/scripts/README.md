# Midland Heart Repairs - Bespoke Query Runner

This directory contains the LLM-free bespoke query runner for Midland Heart repairs data analysis.

## Overview

The `run_repairs_query.py` script executes pre-defined performance metrics queries against the repairs and telematics datasets. By default, an LLM analyst transforms raw metric results into qualitative business insights. Results are automatically logged to timestamped directories for audit trails.

## Directory Structure

```
intranet/repairs_agent/scripts/
├── README.md                      # This file
├── run_repairs_query.py   # Single query runner script
├── query_logger.py        # File-based logging module
├── parallel_queries.sh            # Parallel query runner (tmux sessions)
├── watch_queries.sh               # Watch running query sessions (live)
├── list_queries.sh                # List all query sockets and sessions
├── kill_failed_queries.sh         # Kill failed query sessions
├── kill_server_queries.sh         # Kill query tmux server(s)
├── _repairs_common.sh             # Shared shell utilities
└── _query_generator.py            # Query specification generator
```

## Quick Start

```bash
# From the repository root directory:

# List all available metrics
python intranet/repairs_agent/scripts/run_repairs_query.py --list

# Run a single query
python intranet/repairs_agent/scripts/run_repairs_query.py --query jobs_completed

# Run all queries in parallel (tmux sessions)
./intranet/repairs_agent/scripts/parallel_queries.sh --all

# Watch running queries
./intranet/repairs_agent/scripts/watch_queries.sh
```

---

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--list` | flag | - | List all available queries |
| `--query` | string | - | Query ID to execute |
| `--params` | JSON string | `{}` | Query parameters as JSON |
| `--project` | string | `RepairsAgent` | Project context to activate |
| `--log-dir` | string | cwd | Custom root directory for log files |
| `--metric-subdir` | string | - | Metric subdirectory for nested structure (full-matrix mode) |
| `--no-summary` | flag | - | Skip generating summary file (used by parallel runner) |
| `--no-log` | flag | - | Disable file logging, full output to terminal |
| `--raw` | flag | - | Output raw JSON without formatting |
| `--pretty` | flag | `True` | Pretty print JSON output |
| `-v, --verbose` | flag | - | Enable verbose logging (INFO level) |
| `--debug` | flag | - | Enable debug logging (DEBUG level) |
| `--include-plots` | flag | - | Generate visualization URLs for query results |
| `--no-plots` | flag | - | Explicitly disable plot generation (default behavior) |
| `--skip-analysis` | flag | - | Skip LLM analysis and return raw metric results only |

---

## Available Metrics

### Fully Implemented (11 metrics)

| # | Query ID | Description |
|---|----------|-------------|
| 1 | `jobs_completed` | Jobs completed (groupable by operative/patch/region/day) |
| 2 | `no_access_rate` | No Access % / Absolute number |
| 3 | `first_time_fix_rate` | First Time Fix % / Absolute number |
| 4 | `follow_on_required_rate` | Follow on Required % / Absolute number |
| 5 | `follow_on_materials_rate` | Follow on Required for Materials % |
| 6 | `job_completed_on_time_rate` | Job completed on time % / Absolute number |
| 7 | `jobs_issued` | Jobs issued (groupable by operative/patch/region/day) |
| 8 | `jobs_requiring_materials_rate` | % of jobs that require materials |
| 9 | `avg_repairs_per_property` | Average repairs per property |
| 10 | `appointment_adherence_rate` | Percentage of appointments attended within scheduled window |
| 11 | `total_distance_travelled` | Total distance travelled (groupable by vehicle/day) |

### Skipped Metrics (not runnable - blocked by data availability)

| # | Query ID | Reason |
|---|----------|--------|
| 1 | `merchant_stops` | No exhaustive merchant name/address list |
| 2 | `merchant_dwell_time` | No exhaustive merchant name/address list |
| 3 | `travel_time` | HH:MM:SS string parsing not supported by backend |
| 4 | `complaints_rate` | No complaints column in available data |

---

## Query Parameters

### Common Parameters

Most metrics support these parameters:

| Parameter | Type | Values | Description |
|-----------|------|--------|-------------|
| `group_by` | string | `operative`, `patch`, `region`, `total` | Dimension to group results by |
| `start_date` | string | `YYYY-MM-DD` | Filter start date |
| `end_date` | string | `YYYY-MM-DD` | Filter end date |
| `time_period` | string | `day`, `week`, `month`, `quarter`, `year` | Time granularity |
| `return_absolute` | boolean | `true`, `false` | Return counts instead of percentages |

---

## Usage Examples

### Basic Queries

```bash
# List all available metrics
python intranet/repairs_agent/scripts/run_repairs_query.py --list

# Get jobs completed (default: grouped by operative)
python intranet/repairs_agent/scripts/run_repairs_query.py --query jobs_completed

# Get total repairs completed (no grouping)
python intranet/repairs_agent/scripts/run_repairs_query.py --query jobs_completed

# Get jobs issued per day
python intranet/repairs_agent/scripts/run_repairs_query.py --query jobs_issued
```

### Grouping Options

```bash
# Group by operative (default)
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query jobs_completed \
    --params '{"group_by": "operative"}'

# Group by patch
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query jobs_completed \
    --params '{"group_by": "patch"}'

# Group by region
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query jobs_completed \
    --params '{"group_by": "region"}'

# No grouping (total only)
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query jobs_completed \
    --params '{"group_by": "total"}'
```

### Rate Metrics (Percentages vs Counts)

```bash
# Get no-access rate as percentage (default)
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query no_access_rate

# Get no-access as absolute count
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query no_access_rate \
    --params '{"return_absolute": true}'

# First time fix rate grouped by patch
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query first_time_fix_rate \
    --params '{"group_by": "patch"}'

# First time fix absolute count grouped by region
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query first_time_fix_rate \
    --params '{"group_by": "region", "return_absolute": true}'

# Follow-on required rate by operative
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query follow_on_required_rate \
    --params '{"group_by": "operative"}'

# Job completed on time rate
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query job_completed_on_time_rate
```

### Telematics Metrics

```bash
# Get total distance travelled
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query total_distance_travelled

# Distance by vehicle/operative
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query total_distance_travelled \
    --params '{"group_by": "operative"}'

# Average time travelling
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query travel_time

# Merchant stops per day
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query merchant_stops
```

### Property Analysis

```bash
# Average repairs per property
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query avg_repairs_per_property

# With time period
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query avg_repairs_per_property \
    --params '{"time_period": "month"}'
```

### Combined Parameters

```bash
# Multiple parameters combined
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query first_time_fix_rate \
    --params '{"group_by": "patch", "return_absolute": false, "time_period": "month"}'

# With date filtering (when supported)
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query no_access_rate \
    --params '{"group_by": "operative", "start_date": "2025-07-01", "end_date": "2025-07-31"}'
```

### Output Control

```bash
# Full output to terminal (disable file logging)
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query jobs_completed \
    --no-log

# Raw JSON output (for piping to other tools)
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query jobs_completed \
    --raw

# Custom log directory
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query jobs_completed \
    --log-dir /tmp/repairs_logs
```

### LLM Analysis (Default)

By default, raw metric results are passed to an LLM analyst that generates business insights:

```bash
# Run with LLM analysis (default behavior)
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query first_time_fix_rate \
    --params '{"group_by": "operative"}'

# Skip LLM analysis - return raw data only
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query first_time_fix_rate \
    --params '{"group_by": "operative"}' \
    --skip-analysis
```

The LLM analyst provides:
- **Executive Summary** - Key findings in 2-3 sentences
- **Detailed Analysis** - What the numbers tell us
- **Performance Insights** - Trends, outliers, comparisons
- **Recommendations** - Actionable next steps

### Plot Visualization

Generate visualization URLs with your query results using the `--include-plots` flag:

```bash
# Generate plots for first time fix rate by operative
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query first_time_fix_rate \
    --params '{"group_by": "operative"}' \
    --include-plots

# Generate plots for jobs completed grouped by patch
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query jobs_completed \
    --params '{"group_by": "patch"}' \
    --include-plots

# Generate trend plots (total over time)
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query jobs_completed \
    --params '{"group_by": "total"}' \
    --include-plots
```

Plot visualizations are automatically configured based on the metric and grouping dimension. The summary output will show:
- Plot title and URL for successful generations
- Error messages for any failed plot generations

Example output with plots:
```
📊 Summary:
   • Metric: first_time_fix_rate
   • Total: 85.5
   • Groups: 150
   • Grouped by: operative
   • Plots generated: 1
      ✓ First-Time Fix Rate by Operative: https://console.unify.ai/plot/view/abc123
   • Duration: 3.45s
```

### Debugging

```bash
# Verbose output (INFO level)
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query jobs_completed \
    --verbose

# Debug output (DEBUG level - very detailed)
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query jobs_completed \
    --debug

# Verbose with file logging disabled
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query jobs_completed \
    --verbose --no-log
```

### Project Context

```bash
# Use default project (RepairsAgent)
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query jobs_completed

# Use different project context
python intranet/repairs_agent/scripts/run_repairs_query.py \
    --query jobs_completed \
    --project Intranet
```

---

## Log File Structure

When running queries via `parallel_queries.sh`, results are saved with per-terminal isolation:

### Standard Mode (`--all` or `--expand-params`)

```
.repairs_queries/
└── 2025-12-18T19-30-45_repairs_dev_pts_0/   # Timestamped + socket name
    ├── jobs_completed.log            # Query result log
    ├── no_access_rate__return_absolute_true.log  # With params in filename
    ├── first_time_fix_rate__group_by_patch.log
    └── _run_summary.log                      # Summary of all queries in this run
```

### Full-Matrix Mode (`--full-matrix`)

In full-matrix mode, logs are organized by metric with per-metric summaries:

```
.repairs_queries/
└── 2025-12-18T19-30-45_repairs_dev_pts_0/
    ├── jobs_completed/              # Subdirectory per metric
    │   ├── group_by_operative.log
    │   ├── group_by_patch.log
    │   ├── group_by_region.log
    │   ├── group_by_total.log
    │   └── _metric_summary.log              # Summary for this metric only
    ├── no_access_rate/
    │   ├── group_by_operative__return_absolute_false.log
    │   ├── group_by_operative__return_absolute_true.log
    │   ├── group_by_patch__return_absolute_false.log
    │   ├── ...
    │   └── _metric_summary.log
    ├── first_time_fix_rate/
    │   └── ...
    └── _run_summary.log                     # Global summary across all metrics
```

### Single Query Mode

When running single queries directly (not via `parallel_queries.sh`):

```
.repairs_queries/
└── 2025-12-18T19-30-45.123456Z/             # Timestamped only
    └── query.log
```

### Log File Contents

Each `.log` file contains the LLM analysis (or raw results if `--skip-analysis` is used):

```
================================================================================
BESPOKE REPAIRS QUERY LOG
================================================================================

Query ID:    jobs_completed
Timestamp:   2025-12-18T19:30:45.123456+00:00
Duration:    8.45s
Status:      SUCCESS

----------------------------------------
PARAMETERS
----------------------------------------
  group_by: operative

----------------------------------------
ANALYSIS
----------------------------------------

## Executive Summary

The jobs completed analysis reveals strong overall productivity with 44,025 total
jobs completed across 150 operatives. However, there is significant variation in
individual performance...

## Detailed Analysis

| Rank | Operative | Jobs Completed | % of Total |
|------|-----------|----------------|------------|
| 1 | John Smith | 520 | 1.2% |
| 2 | Sarah Jones | 498 | 1.1% |
| ... | ... | ... | ... |

## Performance Insights

- **Top Performers**: The top 10% of operatives completed 28% of all jobs
- **Potential Concerns**: 12 operatives completed fewer than 100 jobs...

## Recommendations

1. Review workload distribution to balance assignments more evenly
2. Investigate operatives with significantly below-average completion rates...

----------------------------------------
TIMINGS
----------------------------------------
  Query:    2490ms
  Analysis: 5960ms
  Total:    8450ms

----------------------------------------
VISUALIZATIONS
----------------------------------------
  ✓ Jobs Completed by Operative: https://console.unify.ai/plot/view/abc123

----------------------------------------
RAW DATA (for reference)
----------------------------------------
{
  "metric_name": "jobs_completed",
  "group_by": "operative",
  "total": 44025.0,
  "results": [
    {"group": "Aaron Ward", "count": 390},
    ...
  ]
}

================================================================================
END OF LOG - jobs_completed
================================================================================
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (invalid params, query failure, etc.) |
| 130 | Interrupted (Ctrl+C) |

---

## Troubleshooting

### Common Issues

**"Unknown query_id"**
```bash
# List available queries to see valid IDs
python intranet/repairs_agent/scripts/run_repairs_query.py --list
```

**"Invalid JSON params"**
```bash
# JSON must use double quotes, not single quotes
# Wrong:
--params "{'group_by': 'patch'}"

# Correct:
--params '{"group_by": "patch"}'
```

**"Failed to activate project context"**
```bash
# Try with verbose to see details
python intranet/repairs_agent/scripts/run_repairs_query.py --query jobs_completed --verbose
```

### Getting Help

```bash
# Show help
python intranet/repairs_agent/scripts/run_repairs_query.py --help
```

---

## Data Sources

The metrics query against two Excel files:

1. **Repairs Data**: `MDH Repairs Data July - Nov 25 - DL V1.xlsx`
   - Table: `Raised_01-07-2025_to_30-11-2025`
   - Contains job tickets, completion status, operatives, patches, regions

2. **Telematics Data**: `MDH Telematics Data July - Nov 25 - DL V1.xlsx`
   - Tables: `July_2025`, `August_2025`, `September_2025`, `October_2025`, `November_2025`
   - Contains vehicle trips, distances, travel times, locations

---

## Parallel Query Runner

The `parallel_queries.sh` script runs multiple queries in parallel using tmux sessions, similar to `tests/parallel_run.sh`.

### Status Indicators

Sessions are named with status prefixes:
- `r ⏳ query_name` - Running (pending)
- `p ✅ query_name` - Passed (completed successfully)
- `f ❌ query_name` - Failed (error occurred)

### Parallel Runner Arguments

| Argument | Description |
|----------|-------------|
| `--all` | Run all available queries |
| `--query, -q ID` | Run specific query (can be repeated) |
| `--expand-params` | Expand all parameter combinations per query (flat log structure) |
| `--full-matrix` | Full matrix: all queries × all params with nested directories per metric |
| `-j, --jobs N` | Max concurrent sessions (default: 10) |
| `-w, --wait [N]` | Wait for completion (optional timeout in seconds) |
| `--project NAME` | Project context (default: RepairsAgent) |
| `-v, --verbose` | Enable verbose logging in queries |
| `--debug` | Enable debug logging in queries |
| `--no-log` | Disable file logging in queries |
| `--log-dir PATH` | Custom log directory |
| `--include-plots` | Generate visualization URLs for all query results |
| `--skip-analysis` | Skip LLM analysis and return raw metric results only |
| `--consolidated-analysis` | Generate ONE consolidated analysis per metric instead of individual analyses (requires `--full-matrix`) |

### Parallel Runner Examples

```bash
# Run all queries with default parameters (one session per query)
./intranet/repairs_agent/scripts/parallel_queries.sh --all

# Run all queries × all parameter combinations (flat log structure)
# (e.g., jobs_completed with group_by=operative, patch, region, total)
./intranet/repairs_agent/scripts/parallel_queries.sh --all --expand-params

# Full matrix mode: all queries × all params with nested directories per metric
# Creates subdirectories for each metric with per-metric summaries
./intranet/repairs_agent/scripts/parallel_queries.sh --all --full-matrix -j 8 -w

# Run specific queries
./intranet/repairs_agent/scripts/parallel_queries.sh \
    --query jobs_completed \
    --query no_access_rate \
    --query first_time_fix_rate

# Limit concurrency to 5 parallel sessions
./intranet/repairs_agent/scripts/parallel_queries.sh --all -j 5

# Wait for completion (blocking)
./intranet/repairs_agent/scripts/parallel_queries.sh --all -w

# Wait with timeout (300 seconds)
./intranet/repairs_agent/scripts/parallel_queries.sh --all -w 300

# Run with verbose output in each query
./intranet/repairs_agent/scripts/parallel_queries.sh --all --verbose

# Expand params for specific queries (flat structure)
./intranet/repairs_agent/scripts/parallel_queries.sh \
    --query first_time_fix_rate \
    --query no_access_rate \
    --expand-params

# Full combination: all queries, all params, limited concurrency, wait for completion
./intranet/repairs_agent/scripts/parallel_queries.sh --all --expand-params -j 8 -w

# Run all queries with plot generation and LLM analysis (default)
./intranet/repairs_agent/scripts/parallel_queries.sh --all --include-plots -w

# Run all queries with plots but skip LLM analysis (raw data only)
./intranet/repairs_agent/scripts/parallel_queries.sh --all --include-plots --skip-analysis -w

# Full matrix with individual LLM analysis per combination
./intranet/repairs_agent/scripts/parallel_queries.sh --all --full-matrix --include-plots -w

# Full matrix with CONSOLIDATED analysis (one analysis per metric, not per combination)
./intranet/repairs_agent/scripts/parallel_queries.sh --all --full-matrix --include-plots --consolidated-analysis -w
```

### Watching Queries

Use `watch_queries.sh` to monitor running sessions with live updates:

```bash
# Continuous watch (auto-refresh every 2 seconds)
./intranet/repairs_agent/scripts/watch_queries.sh

# One-time summary
./intranet/repairs_agent/scripts/watch_queries.sh --summary

# Custom refresh interval
./intranet/repairs_agent/scripts/watch_queries.sh --interval 5
```

### Listing All Query Runs

Use `list_queries.sh` to see all query sockets across terminals:

```bash
# List all sockets with active sessions
./intranet/repairs_agent/scripts/list_queries.sh

# Include empty sockets
./intranet/repairs_agent/scripts/list_queries.sh --all

# Just socket names (for scripting)
./intranet/repairs_agent/scripts/list_queries.sh --quiet

# Attach to a session from any socket
tmux -L $(./intranet/repairs_agent/scripts/list_queries.sh -q | head -1) attach
```

Output shows:
- Socket name with terminal indicator (current vs other)
- All sessions with colored status indicators
- Summary counts (running/passed/failed)
- Progress bar for running queries

### Killing Failed Sessions

Use `kill_failed_queries.sh` to clean up failed sessions:

```bash
# Kill failed sessions in current terminal
./intranet/repairs_agent/scripts/kill_failed_queries.sh

# Dry run - see what would be killed
./intranet/repairs_agent/scripts/kill_failed_queries.sh --dry-run

# Kill failed sessions across ALL terminals
./intranet/repairs_agent/scripts/kill_failed_queries.sh --all

# Kill failed sessions in a specific socket
./intranet/repairs_agent/scripts/kill_failed_queries.sh --socket repairs_dev_pts_0
```

### Killing the Server

Use `kill_server_queries.sh` to stop all sessions and clean up:

```bash
# Kill current terminal's server (with confirmation)
./intranet/repairs_agent/scripts/kill_server_queries.sh

# Kill ALL repairs servers across all terminals
./intranet/repairs_agent/scripts/kill_server_queries.sh --all

# Kill specific socket's server
./intranet/repairs_agent/scripts/kill_server_queries.sh --socket repairs_dev_pts_0

# Force kill without confirmation
./intranet/repairs_agent/scripts/kill_server_queries.sh --all --force
```

### Manual Session Management

```bash
# List all sessions for this terminal
tmux -L repairs_dev_pts_0 ls

# Attach to a specific session
tmux -L repairs_dev_pts_0 attach -t 'r ⏳ jobs_completed'

# Kill a specific session
tmux -L repairs_dev_pts_0 kill-session -t 'f ❌ complaints_rate'

# Kill all sessions (via tmux directly)
tmux -L repairs_dev_pts_0 kill-server
```

### Parallelization Modes

| Mode | Sessions Created | Log Structure | Use Case |
|------|------------------|---------------|----------|
| `--all` | 15 (one per query) | Flat | Quick overview of all metrics |
| `--all --expand-params` | ~60 (queries × param combos) | Flat | Comprehensive analysis |
| `--all --full-matrix` | ~60 (queries × param combos) | Nested (per-metric dirs) | Full analysis with per-metric summaries |
| `--query Q1 --query Q2` | 2 | Flat | Focused analysis |
| `--query Q --expand-params` | ~4-8 per query | Flat | Deep dive into one metric |

### Exit Codes (Parallel Runner)

| Code | Meaning |
|------|---------|
| 0 | All queries passed |
| 1 | One or more queries failed |
| 2 | Timeout reached |
| 130 | Interrupted (Ctrl+C) |

---

## Related Files

- `intranet/repairs_agent/metrics/core.py` - Metric function implementations (16 metrics)
- `intranet/repairs_agent/metrics/types.py` - Pydantic models and enums (GroupBy, TimePeriod, MetricResult, PlotResult)
- `intranet/repairs_agent/metrics/helpers.py` - Helper functions for metric calculations
- `intranet/repairs_agent/static/agent.py` - BespokeRepairsAgent with LLM analysis
- `intranet/repairs_agent/static/registry.py` - Query registry
- `intranet/repairs_agent/config/prompt_builder.py` - Analyst prompt builders
