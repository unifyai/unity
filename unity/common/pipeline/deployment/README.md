## `deployment/` — offline deployment bundle ingestion

This subpackage turns ad-hoc client data-dump ingestion into a first-class typed workflow. Instead of running manual scripts on a laptop, operators submit typed `DeploymentBundle` manifests through a managed runner with status tracking, stage reporting, and cost/observability output.

## Core concepts

### DeploymentBundle

A typed manifest describing everything needed for one client data-dump ingest:

- `deployment_identity`: client name, deployment name, project, environment, tenant
- `actor_config`: optional actor/assistant configuration artifact
- `seed_data`: contact, knowledge, or guidance seed artifacts
- `integration_assets`: function directories, integration configs
- `data_pipeline_config`: `FilePipelineConfig` for parse/ingest behavior
- `source_artifact_manifest`: the actual data files to ingest

Each artifact entry is a `DeploymentBundleArtifact` with a `kind`, `logical_name`, `source_path`, and optional `artifact_uri` (populated after staging).

### DeploymentIngestionJob

A tracked execution record for one bundle ingest. Lifecycle:

```
queued → running → success
                 → error
```

Jobs carry:
- `stage_status`: dict of `DeploymentIngestionStageStatus` for named stages (e.g., `prepare_bundle`, `ingest_data`)
- `cost_ledger_path`: path to the JSONL cost ledger for the run
- `observability_refs`: pointers to progress file, log file, run ledger, cost ledger

### Execution model

Jobs are executed by a `DeploymentIngestionExecutor` callback that receives the resolved bundle and a `StageReporter` function. The callback:

1. Reports stage transitions (`running`, `success`, `error`) with optional metadata
2. Runs the actual parse + ingest logic
3. Returns a `DeploymentIngestionOutcome` with cost and observability references

The runner handles all job lifecycle management (status updates, error handling, stage timestamping).

## Module layout

### `types.py` — DTOs and protocols

All Pydantic models, Literal types, and Protocol definitions:

- `DeploymentBundle`, `DeploymentIdentity`, `DeploymentBundleArtifact`
- `DeploymentIngestionJob`, `DeploymentIngestionStageStatus`, `DeploymentIngestionOutcome`
- `DeploymentQueuePayload` (typed queue message for queue-backed submission)
- `DeploymentBundleRuntimeConfig` (adapter/backend selection)
- `DeploymentBundleStore`, `DeploymentJobStore` (storage protocols)
- `DeploymentIngestionExecutor`, `StageReporter` (execution protocols)

### `local.py` — filesystem-backed local implementations

- `LocalDeploymentBundleStore`: writes bundle manifests as JSON, stages source artifacts by copying them into a content-addressed directory
- `LocalDeploymentJobStore`: writes job snapshots as JSON with a `threading.Lock` for concurrent access and an atomic `update_job` read-modify-write helper
- `LocalDeploymentIngestionRunner`: submits jobs to a `ThreadPoolExecutor`, tracks futures, handles stage reporting and job lifecycle updates
- `LocalQueuedDeploymentCoordinator`: async submission through a `WorkQueue` with `LocalQueueWorker` drain semantics

### `runtime.py` — runtime assembly

`build_local_deployment_runtime()` wires up all local implementations from a single `DeploymentBundleRuntimeConfig`. Returns a `LocalDeploymentRuntime` frozen dataclass with all resolved components. Raises `NotImplementedError` for non-local backends (GCP adapters are future work in private repos).

### `__init__.py` — barrel re-exports

All public types and implementations are re-exported for convenient import from `unity.common.pipeline.deployment`.

## Operator workflow

### 1. Prepare a bundle

```python
from unity.common.pipeline import (
    DeploymentBundle,
    DeploymentBundleArtifact,
    DeploymentIdentity,
)

bundle = DeploymentBundle(
    deployment_identity=DeploymentIdentity(
        client="acme_corp",
        deployment="v3",
        project="AcmeCorp",
        environment="local",
    ),
    data_pipeline_config=DeploymentBundleArtifact(
        kind="data_pipeline_config",
        logical_name="pipeline_config.json",
        source_path="/path/to/pipeline_config.json",
    ),
    source_artifact_manifest=[
        DeploymentBundleArtifact(
            kind="source_data",
            logical_name="repairs.csv",
            source_path="/path/to/repairs.csv",
        ),
    ],
)
```

### 2. Build a local runtime

```python
from unity.common.pipeline import (
    DeploymentBundleRuntimeConfig,
    build_local_deployment_runtime,
)

runtime = build_local_deployment_runtime(
    DeploymentBundleRuntimeConfig(local_root_dir=".deployments"),
)
```

### 3. Stage artifacts and submit

```python
prepared = runtime.bundle_store.prepare_bundle(bundle)

def execute(bundle, report_stage):
    report_stage("prepare", "running")
    # ... parse and ingest logic ...
    report_stage("prepare", "success")
    report_stage("ingest", "running")
    # ... ingest logic ...
    report_stage("ingest", "success")
    return DeploymentIngestionOutcome(
        cost_ledger_path=".deployments/cost.jsonl",
    )

job = runtime.runner.submit(prepared, run_mode="hybrid", execute=execute)
final = runtime.runner.wait(job.job_id, timeout=300.0)
print(final.status, final.stage_status)
```

### 4. Queue-backed submission (async)

```python
job = await runtime.queued_coordinator.submit(
    prepared, run_mode="data_manager",
)
# Worker drains the queue
processed = await runtime.queued_coordinator.drain_once(
    execute=execute, max_messages=1,
)
```

## Default storage layout

With `local_root_dir=".deployments"`:

```
.deployments/
├── deployment-bundles/
│   └── <bundle_id>.json          # Bundle manifest
├── deployment-bundle-artifacts/
│   └── <bundle_id>/              # Staged source artifacts
│       ├── pipeline_config.json
│       ├── repairs.csv
│       └── functions/
└── deployment-jobs/
    └── <job_id>.json             # Job status snapshot
```

## Extending for GCP

The `DeploymentBundleRuntimeConfig` Literal types already define the extension points:

- `bundle_store_backend`: `"local"` | `"gcs"`
- `job_store_backend`: `"local"` | `"gcs"`
- `executor_backend`: `"local"` | `"cloud_tasks"`
- `queue_backend`: `"in_memory"` | `"pubsub"` | `"cloud_tasks"`

GCP implementations plug into the same `DeploymentBundleStore` and `DeploymentJobStore` protocols. The runtime builder in private repos would select the appropriate adapter based on config.
