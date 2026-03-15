# Midland Heart Client Customization

Reactive repairs KPI analysis, telematics fleet management, and property
maintenance performance reporting for Midland Heart (org ID 3).

## Directory layout

```
midland_heart/
  __init__.py            # Org registration: ActorConfig, guidance, function_dir
  functions/
    metrics.py           # 11 KPI metric functions (@custom_function)
    helpers.py           # 11 shared helper functions (@custom_function)
    types.py             # Enums and TypedDicts used by metrics/helpers
  pipeline_config.json   # Shared config: source files, DM contexts, embedding, execution
  seed_data.py           # Programmatic seeding via DataManager.ingest()
  ingest_dm.py           # CLI: DataManager ingestion path
  ingest_fm.py           # CLI: FileManager ingestion path
  ingest_utils.py        # Shared bootstrap: env, project, config, logging, progress
  README.md              # This file
```

## Org registration

`__init__.py` registers the Midland Heart org via `register_org()`:

- **ActorConfig** -- thin role identity for repairs & maintenance KPI analysis.
- **function_dir** -- points to `functions/` containing 22 `@custom_function`
  definitions (11 KPI metrics + 11 shared helpers).
- **Guidance** -- four entries:
  1. Repairs KPI Analysis Workflow (available metrics, parameters, helpers)
  2. Repairs Data Schema (column definitions for Repairs2025)
  3. Telematics Data Schema (column definitions for Telematics2025/*)
  4. Data Discovery Guide (context paths, discovery primitives)

## KPI functions

All metric functions accept `(data_primitives, group_by=None, start_date=None,
end_date=None, time_period="day", include_plots=False)` and return a
standardised result dict with keys: `metric_name`, `group_by`, `time_period`,
`start_date`, `end_date`, `results`, `total`, `metadata`, `plots`.

| Metric | Description |
|---|---|
| `repairs_kpi_jobs_completed` | Count of completed jobs |
| `repairs_kpi_jobs_issued` | Count of issued jobs per period |
| `repairs_kpi_no_access_rate` | No-access percentage (+ `return_absolute`) |
| `repairs_kpi_first_time_fix_rate` | First-time fix percentage (+ `return_absolute`) |
| `repairs_kpi_follow_on_required_rate` | Follow-on required percentage (+ `return_absolute`) |
| `repairs_kpi_follow_on_materials_rate` | Follow-on for materials percentage (+ `return_absolute`) |
| `repairs_kpi_job_completed_on_time_rate` | On-time completion percentage (+ `return_absolute`) |
| `repairs_kpi_jobs_requiring_materials_rate` | Jobs needing materials percentage (+ `return_absolute`) |
| `repairs_kpi_avg_repairs_per_property` | Average repairs per property |
| `repairs_kpi_appointment_adherence_rate` | Appointment adherence percentage (+ `return_absolute`) |
| `repairs_kpi_total_distance_travelled` | Total fleet distance (telematics) |

Group-by dimensions: `"operative"`, `"patch"`, `"region"`, `"trade"`, `"day"`.

Helper functions: `discover_repairs_table`, `discover_telematics_tables`,
`resolve_group_by`, `build_filter`, `extract_count`, `extract_sum`,
`normalize_grouped_result`, `compute_percentage`, `build_metric_result`,
`extract_plot_url`, `extract_plot_succeeded`.

## Data contexts

| Context | Source |
|---|---|
| `MidlandHeart/Repairs2025` | Repairs spreadsheet, single sheet |
| `MidlandHeart/Telematics2025/July` | Telematics spreadsheet, per-month |
| `MidlandHeart/Telematics2025/August` | " |
| `MidlandHeart/Telematics2025/September` | " |
| `MidlandHeart/Telematics2025/October` | " |
| `MidlandHeart/Telematics2025/November` | " |

Source spreadsheets live in `intranet/repairs/` at the project root.

Context paths are relative to the DataManager's base context. The DM's
`_resolve_context` prepends the active user/assistant scope, producing
fully-qualified paths like `<user>/<assistant>/Data/MidlandHeart/Repairs2025`.

## Ingestion

Two paths exist for loading data from the source spreadsheets into Unify
contexts. Both read the same `pipeline_config.json`.

### Programmatic seeding (`seed_data.py`)

For use from ConversationManager flows or test fixtures:

```python
from unity.data_manager.data_manager import DataManager
from unity.customization.clients.midland_heart.seed_data import seed_all

dm = DataManager()
results = seed_all(dm)
for ctx, result in results.items():
    print(f"{ctx}: {result.rows_inserted} rows")
```

### DataManager CLI (`ingest_dm.py`)

Creates data-centric contexts (`MidlandHeart/*` under the DM base context).
Parses Excel files using `FileParser.parse_batch()`, then ingests each
extracted table via `DataManager.ingest()`. Telemetry is wired manually
using the `ProgressReporter` system.

```bash
# Default: sequential, high verbosity
.venv/bin/python unity/customization/clients/midland_heart/ingest_dm.py \
    --config unity/customization/clients/midland_heart/pipeline_config.json \
    --project MidlandHeart

# Parallel tables, skip embedding, debug logging, suppress SDK noise
.venv/bin/python unity/customization/clients/midland_heart/ingest_dm.py \
    --config unity/customization/clients/midland_heart/pipeline_config.json \
    --project MidlandHeart \
    --parallel --no-embed --debug --no-sdk-log

# Ingest only specific tables by sheet name substring
.venv/bin/python unity/customization/clients/midland_heart/ingest_dm.py \
    --config unity/customization/clients/midland_heart/pipeline_config.json \
    --tables "July 2025" "August 2025"

# Fresh project, custom chunk size
.venv/bin/python unity/customization/clients/midland_heart/ingest_dm.py \
    --config unity/customization/clients/midland_heart/pipeline_config.json \
    --overwrite --chunk-size 500
```

### FileManager CLI (`ingest_fm.py`)

Creates file-centric contexts (`Files/Local/{id}/Tables/{sheet}`). Telemetry
is handled internally by the FM pipeline's `DiagnosticsConfig`.

```bash
# Default: sequential, high verbosity
.venv/bin/python unity/customization/clients/midland_heart/ingest_fm.py

# Parallel, custom progress file
.venv/bin/python unity/customization/clients/midland_heart/ingest_fm.py \
    --parallel --progress-file ./logs/fm_progress.jsonl

# Fresh project
.venv/bin/python unity/customization/clients/midland_heart/ingest_fm.py \
    --overwrite --project MidlandHeart_Dev
```

### CLI arguments

Arguments shared by both scripts:

| Argument | Default | Description |
|---|---|---|
| `--config` | `pipeline_config.json` (sibling) | Path to shared JSON config |
| `--project` | `MidlandHeart` | Unify project name |
| `--overwrite` | off | Delete and recreate project first |
| `--parallel` | off | Parallel processing (FM: files, DM: tables) |
| `--no-embed` | off | Skip embedding |
| `--verbosity` | `high` (from config) | `low` / `medium` / `high` |
| `--progress-file` | auto-timestamped JSONL | Override progress output path |
| `--log-file` | `<run_dir>/run.log` | Override log file path |
| `--debug` | off | Enable DEBUG-level Python logging |
| `--no-sdk-log` | off | Suppress the companion `*_unify.log` SDK log file |

Arguments specific to `ingest_dm.py`:

| Argument | Default | Description |
|---|---|---|
| `--skip-all-context` | off | Skip `add_to_all_context` for faster bulk load |
| `--chunk-size` | per-table from config (1000) | Override rows-per-chunk |
| `--tables` | all | Only ingest listed tables (case-sensitive substring match) |

### Progress output

Both scripts write to a timestamped run directory under `logs/pipeline/`:

```
logs/pipeline/
  ingest_dm/
    2026-03-13T17-38-41/
      run.log              # Python logger output
      run_unify.log        # SDK/HTTP traces (unless --no-sdk-log)
      progress.jsonl       # Structured progress events
      error_details/       # Full tracebacks for failed tasks
  ingest_fm/
    ...
```

Each JSONL line follows the `ProgressEvent` schema:

```json
{"timestamp":"...","file_path":"...","phase":"parse","status":"started","duration_ms":0.0,"elapsed_ms":0.0}
{"timestamp":"...","file_path":"...","phase":"ingest_table","status":"completed","duration_ms":13000.0,"elapsed_ms":15500.0,"meta":{"table_label":"July 2025","row_count":1200,"rows_inserted":1200}}
```

Phases: `parse`, `ingest_table`, `ingest_table/create_table`,
`ingest_table/insert_chunk_NNNN`, `ingest_table/embed_chunk_NNNN`,
`file_complete`.

## Pipeline config (`pipeline_config.json`)

Top-level sections:

| Section | Purpose |
|---|---|
| `source_files` | Excel file paths and sheet names to process |
| `dm_contexts` | Sheet-to-context mapping with descriptions and chunk sizes (DM path) |
| `ingest` | FM-specific ingest settings, business context descriptions, column descriptions |
| `embed` | Embedding strategy (`along`/`after`/`off`), per-file source and target column specs |
| `execution` | Worker counts for parallel processing |
| `retry` | Retry policy (max_retries, delay, fail_fast) |
| `diagnostics` | Progress reporting mode and verbosity |
| `parse` | Parser concurrency settings |
| `output` | FM return mode (`compact`/`full`/`none`) |

## Tests

Tests live in `tests/customization/clients/midland_heart/`:

- `test_scaffold.py` -- structural tests: org registration, config, guidance,
  function directory, helper contracts, seed_data importability.
- `test_kpi_functions.py` -- integration tests: loads KPI functions into a
  FunctionManager, verifies dependency edge recording, and executes a
  representative metric against stub `data_primitives`.
