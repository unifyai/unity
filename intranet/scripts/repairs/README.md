# Midland Heart Repairs - Bespoke Query Runner

This directory contains the LLM-free bespoke query runner for Midland Heart repairs data analysis.

## Overview

The `run_repairs_query.py` script executes pre-defined performance metrics queries against the repairs and telematics datasets without requiring LLM orchestration. Results are automatically logged to timestamped directories for audit trails.

## Directory Structure

```
intranet/scripts/repairs/
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
python intranet/scripts/repairs/run_repairs_query.py --list

# Run a single query
python intranet/scripts/repairs/run_repairs_query.py --query jobs_completed_per_day

# Run all queries in parallel (tmux sessions)
./intranet/scripts/repairs/parallel_queries.sh --all

# Watch running queries
./intranet/scripts/repairs/watch_queries.sh
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
| `--no-log` | flag | - | Disable file logging, full output to terminal |
| `--raw` | flag | - | Output raw JSON without formatting |
| `--pretty` | flag | `True` | Pretty print JSON output |
| `-v, --verbose` | flag | - | Enable verbose logging (INFO level) |
| `--debug` | flag | - | Enable debug logging (DEBUG level) |

---

## Available Metrics (15 Total)

### Repairs Data Metrics

| # | Query ID | Description |
|---|----------|-------------|
| 1 | `jobs_completed_per_day` | Jobs completed per man per day |
| 2 | `no_access_rate` | No Access % / Absolute number |
| 3 | `first_time_fix_rate` | First Time Fix % / Absolute number |
| 4 | `follow_on_required_rate` | Follow on Required % / Absolute number |
| 5 | `follow_on_materials_rate` | Follow on Required for Materials % |
| 6 | `job_completed_on_time_rate` | Job completed on time % / Absolute number |
| 7 | `repairs_completed_per_day` | Repairs completed per day (aggregate) |
| 8 | `jobs_issued_per_day` | Jobs issued per day |
| 9 | `jobs_requiring_materials_rate` | % of jobs that require materials |
| 10 | `avg_repairs_per_property` | Average repairs per property |
| 11 | `complaints_rate` | Complaints as % of total jobs (data not available) |

### Telematics Data Metrics

| # | Query ID | Description |
|---|----------|-------------|
| 12 | `distance_travelled_per_day` | Distance travelled per day |
| 13 | `avg_time_travelling` | Average time travelling per day |
| 14 | `merchant_stops_per_day` | Number of merchant stops per day |
| 15 | `avg_duration_at_merchant` | Average duration at merchant |

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
python intranet/scripts/repairs/run_repairs_query.py --list

# Get jobs completed (default: grouped by operative)
python intranet/scripts/repairs/run_repairs_query.py --query jobs_completed_per_day

# Get total repairs completed (no grouping)
python intranet/scripts/repairs/run_repairs_query.py --query repairs_completed_per_day

# Get jobs issued per day
python intranet/scripts/repairs/run_repairs_query.py --query jobs_issued_per_day
```

### Grouping Options

```bash
# Group by operative (default)
python intranet/scripts/repairs/run_repairs_query.py \
    --query jobs_completed_per_day \
    --params '{"group_by": "operative"}'

# Group by patch
python intranet/scripts/repairs/run_repairs_query.py \
    --query jobs_completed_per_day \
    --params '{"group_by": "patch"}'

# Group by region
python intranet/scripts/repairs/run_repairs_query.py \
    --query jobs_completed_per_day \
    --params '{"group_by": "region"}'

# No grouping (total only)
python intranet/scripts/repairs/run_repairs_query.py \
    --query jobs_completed_per_day \
    --params '{"group_by": "total"}'
```

### Rate Metrics (Percentages vs Counts)

```bash
# Get no-access rate as percentage (default)
python intranet/scripts/repairs/run_repairs_query.py \
    --query no_access_rate

# Get no-access as absolute count
python intranet/scripts/repairs/run_repairs_query.py \
    --query no_access_rate \
    --params '{"return_absolute": true}'

# First time fix rate grouped by patch
python intranet/scripts/repairs/run_repairs_query.py \
    --query first_time_fix_rate \
    --params '{"group_by": "patch"}'

# First time fix absolute count grouped by region
python intranet/scripts/repairs/run_repairs_query.py \
    --query first_time_fix_rate \
    --params '{"group_by": "region", "return_absolute": true}'

# Follow-on required rate by operative
python intranet/scripts/repairs/run_repairs_query.py \
    --query follow_on_required_rate \
    --params '{"group_by": "operative"}'

# Job completed on time rate
python intranet/scripts/repairs/run_repairs_query.py \
    --query job_completed_on_time_rate
```

### Telematics Metrics

```bash
# Get total distance travelled
python intranet/scripts/repairs/run_repairs_query.py \
    --query distance_travelled_per_day

# Distance by vehicle/operative
python intranet/scripts/repairs/run_repairs_query.py \
    --query distance_travelled_per_day \
    --params '{"group_by": "operative"}'

# Average time travelling
python intranet/scripts/repairs/run_repairs_query.py \
    --query avg_time_travelling

# Merchant stops per day
python intranet/scripts/repairs/run_repairs_query.py \
    --query merchant_stops_per_day
```

### Property Analysis

```bash
# Average repairs per property
python intranet/scripts/repairs/run_repairs_query.py \
    --query avg_repairs_per_property

# With time period
python intranet/scripts/repairs/run_repairs_query.py \
    --query avg_repairs_per_property \
    --params '{"time_period": "month"}'
```

### Combined Parameters

```bash
# Multiple parameters combined
python intranet/scripts/repairs/run_repairs_query.py \
    --query first_time_fix_rate \
    --params '{"group_by": "patch", "return_absolute": false, "time_period": "month"}'

# With date filtering (when supported)
python intranet/scripts/repairs/run_repairs_query.py \
    --query no_access_rate \
    --params '{"group_by": "operative", "start_date": "2025-07-01", "end_date": "2025-07-31"}'
```

### Output Control

```bash
# Full output to terminal (disable file logging)
python intranet/scripts/repairs/run_repairs_query.py \
    --query jobs_completed_per_day \
    --no-log

# Raw JSON output (for piping to other tools)
python intranet/scripts/repairs/run_repairs_query.py \
    --query jobs_completed_per_day \
    --raw

# Custom log directory
python intranet/scripts/repairs/run_repairs_query.py \
    --query jobs_completed_per_day \
    --log-dir /tmp/repairs_logs
```

### Debugging

```bash
# Verbose output (INFO level)
python intranet/scripts/repairs/run_repairs_query.py \
    --query jobs_completed_per_day \
    --verbose

# Debug output (DEBUG level - very detailed)
python intranet/scripts/repairs/run_repairs_query.py \
    --query jobs_completed_per_day \
    --debug

# Verbose with file logging disabled
python intranet/scripts/repairs/run_repairs_query.py \
    --query jobs_completed_per_day \
    --verbose --no-log
```

### Project Context

```bash
# Use default project (RepairsAgent)
python intranet/scripts/repairs/run_repairs_query.py \
    --query jobs_completed_per_day

# Use different project context
python intranet/scripts/repairs/run_repairs_query.py \
    --query jobs_completed_per_day \
    --project Intranet
```

---

## Log File Structure

When running queries via `parallel_queries.sh`, results are saved with per-terminal isolation:

```
.repairs_queries/
└── 2025-12-18T19-30-45_repairs_dev_pts_0/   # Timestamped + socket name
    ├── jobs_completed_per_day.log            # Query result log
    ├── no_access_rate__return_absolute_true.log  # With params in filename
    ├── first_time_fix_rate__group_by_patch.log
    └── _run_summary.log                      # Summary of all queries in this run
```

When running single queries directly (not via `parallel_queries.sh`):

```
.repairs_queries/
└── 2025-12-18T19-30-45.123456Z/             # Timestamped only
    └── query.log
```

### Log File Contents

Each `.log` file contains:

```
================================================================================
BESPOKE REPAIRS QUERY LOG
================================================================================

Query ID:    jobs_completed_per_day
Timestamp:   2025-12-18T19:30:45.123456+00:00
Duration:    2.49s
Status:      SUCCESS

----------------------------------------
PARAMETERS
----------------------------------------
  group_by: operative

----------------------------------------
RESULT
----------------------------------------
{
  "metric_name": "jobs_completed_per_day",
  "group_by": "operative",
  "total": 44025.0,
  "results": [
    {"group": "Aaron Ward", "count": 390},
    ...
  ]
}

----------------------------------------
SUMMARY
----------------------------------------
  Metric:     jobs_completed_per_day
  Total:      44025.0
  Groups:     150
  Group By:   operative

================================================================================
END OF LOG - jobs_completed_per_day
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
python intranet/scripts/repairs/run_repairs_query.py --list
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
python intranet/scripts/repairs/run_repairs_query.py --query jobs_completed_per_day --verbose
```

### Getting Help

```bash
# Show help
python intranet/scripts/repairs/run_repairs_query.py --help
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
| `--expand-params` | Expand all parameter combinations per query |
| `-j, --jobs N` | Max concurrent sessions (default: 10) |
| `-w, --wait [N]` | Wait for completion (optional timeout in seconds) |
| `--project NAME` | Project context (default: RepairsAgent) |
| `-v, --verbose` | Enable verbose logging in queries |
| `--debug` | Enable debug logging in queries |
| `--no-log` | Disable file logging in queries |
| `--log-dir PATH` | Custom log directory |

### Parallel Runner Examples

```bash
# Run all queries with default parameters (one session per query)
./intranet/scripts/repairs/parallel_queries.sh --all

# Run all queries × all parameter combinations
# (e.g., jobs_completed_per_day with group_by=operative, patch, region, total)
./intranet/scripts/repairs/parallel_queries.sh --all --expand-params

# Run specific queries
./intranet/scripts/repairs/parallel_queries.sh \
    --query jobs_completed_per_day \
    --query no_access_rate \
    --query first_time_fix_rate

# Limit concurrency to 5 parallel sessions
./intranet/scripts/repairs/parallel_queries.sh --all -j 5

# Wait for completion (blocking)
./intranet/scripts/repairs/parallel_queries.sh --all -w

# Wait with timeout (300 seconds)
./intranet/scripts/repairs/parallel_queries.sh --all -w 300

# Run with verbose output in each query
./intranet/scripts/repairs/parallel_queries.sh --all --verbose

# Expand params for specific queries
./intranet/scripts/repairs/parallel_queries.sh \
    --query first_time_fix_rate \
    --query no_access_rate \
    --expand-params

# Full combination: all queries, all params, limited concurrency, wait for completion
./intranet/scripts/repairs/parallel_queries.sh --all --expand-params -j 8 -w
```

### Watching Queries

Use `watch_queries.sh` to monitor running sessions with live updates:

```bash
# Continuous watch (auto-refresh every 2 seconds)
./intranet/scripts/repairs/watch_queries.sh

# One-time summary
./intranet/scripts/repairs/watch_queries.sh --summary

# Custom refresh interval
./intranet/scripts/repairs/watch_queries.sh --interval 5
```

### Listing All Query Runs

Use `list_queries.sh` to see all query sockets across terminals:

```bash
# List all sockets with active sessions
./intranet/scripts/repairs/list_queries.sh

# Include empty sockets
./intranet/scripts/repairs/list_queries.sh --all

# Just socket names (for scripting)
./intranet/scripts/repairs/list_queries.sh --quiet

# Attach to a session from any socket
tmux -L $(./intranet/scripts/repairs/list_queries.sh -q | head -1) attach
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
./intranet/scripts/repairs/kill_failed_queries.sh

# Dry run - see what would be killed
./intranet/scripts/repairs/kill_failed_queries.sh --dry-run

# Kill failed sessions across ALL terminals
./intranet/scripts/repairs/kill_failed_queries.sh --all

# Kill failed sessions in a specific socket
./intranet/scripts/repairs/kill_failed_queries.sh --socket repairs_dev_pts_0
```

### Killing the Server

Use `kill_server_queries.sh` to stop all sessions and clean up:

```bash
# Kill current terminal's server (with confirmation)
./intranet/scripts/repairs/kill_server_queries.sh

# Kill ALL repairs servers across all terminals
./intranet/scripts/repairs/kill_server_queries.sh --all

# Kill specific socket's server
./intranet/scripts/repairs/kill_server_queries.sh --socket repairs_dev_pts_0

# Force kill without confirmation
./intranet/scripts/repairs/kill_server_queries.sh --all --force
```

### Manual Session Management

```bash
# List all sessions for this terminal
tmux -L repairs_dev_pts_0 ls

# Attach to a specific session
tmux -L repairs_dev_pts_0 attach -t 'r ⏳ jobs_completed_per_day'

# Kill a specific session
tmux -L repairs_dev_pts_0 kill-session -t 'f ❌ complaints_rate'

# Kill all sessions (via tmux directly)
tmux -L repairs_dev_pts_0 kill-server
```

### Parallelization Modes

| Mode | Sessions Created | Use Case |
|------|------------------|----------|
| `--all` | 15 (one per query) | Quick overview of all metrics |
| `--all --expand-params` | ~60 (queries × param combos) | Comprehensive analysis |
| `--query Q1 --query Q2` | 2 | Focused analysis |
| `--query Q --expand-params` | ~4-8 per query | Deep dive into one metric |

### Exit Codes (Parallel Runner)

| Code | Meaning |
|------|---------|
| 0 | All queries passed |
| 1 | One or more queries failed |
| 2 | Timeout reached |
| 130 | Interrupted (Ctrl+C) |

---

## Related Files

- `intranet/repairs/queries/metrics.py` - Metric function implementations
- `intranet/repairs/queries/_types.py` - Type definitions (GroupBy, TimePeriod, MetricResult)
- `intranet/core/bespoke_repairs_agent.py` - Agent and query registry
