# Job Watcher

A lightweight K8s operator that watches Unity pod terminations and runs
crash-safe exit cleanup.  Built on [kopf](https://kopf.dev/) (Kubernetes
Operator Pythonic Framework).

## Why

VM release and `AssistantJobs` record cleanup must happen regardless of
how a Unity container exits (graceful, OOMKill, segfault, node failure).
Running this logic inside the container is unreliable in crash scenarios.

The job-watcher runs **outside** Unity containers on the same GKE cluster
and is the **sole owner** of VM release and `running=False` record
updates.  The Unity container's `mark_job_done()` only handles session
duration metrics and the K8s label patch.

kopf manages the watch stream lifecycle, reconnection, error isolation,
and liveness probes — reacting within seconds of any pod termination.

## How it works

The watcher registers a kopf event handler on pods with
`label_selector=app=unity`.  kopf manages the underlying K8s watch
stream (persistent push connection, automatic reconnection,
`resourceVersion` tracking).

When a pod reaches `Succeeded` or `Failed`:
- Fetches `AssistantJobs` records for the pod's `assistant-id` label
  and sets `running=False`.
- Calls the comms service to release any pool VM assigned to that
  assistant (with retries and disk-detach fallback).

Each handler invocation is isolated — if one cleanup fails, it doesn't
affect processing of other events.  kopf handles retries and error
tracking automatically.

## Code structure

All AssistantJobs operations (record queries/mutations, label patching,
VM release, disk detach) live in a single shared module:
`unity/conversation_manager/assistant_jobs_api.py`.  This file uses the
`unify` SDK for Orchestra log operations and `requests` for comms-service
calls, with no Unity-specific dependencies.  It is used by both:

- **`assistant_jobs.py`** (Unity container) — thin wrapper that reads
  `SESSION_DETAILS`/`SETTINGS`, records Prometheus metrics, and delegates
  all HTTP operations to `assistant_jobs_api`.
- **`watcher.py`** (this operator) — thin kopf handler that reads env
  vars and delegates cleanup to `assistant_jobs_api`.

The Dockerfile copies `assistant_jobs_api.py` from the Unity source tree
at build time and clones/installs the `unify` SDK from GitHub (the deploy
scripts set the Docker build context to the repo root and pass
`GITHUB_TOKEN` for private repo access).

## Responsibility split

| Component | When it runs | What it does |
|---|---|---|
| `mark_job_done()` (in Unity container) | Graceful exit | K8s label patch (`unity-status=done`) + session duration metric |
| **job-watcher** (this) | Any exit (crash-safe) | `running=False` in AssistantJobs + VM release |
| `expire_all_stale_jobs()` (adapters) | Periodic sweep | Safety net for anything the watcher missed |

The watcher and the adapter sweep are idempotent.  Running both is harmless.

## Deployment

### Prerequisites

- `kubectl` configured for the unity GKE cluster:
  ```bash
  gcloud container clusters get-credentials unity \
    --region us-central1 \
    --project responsive-city-458413-a2
  ```
- `unity-config` ConfigMap and `unity-secrets` Secret exist in the target
  namespace (they already do for Unity jobs).
- `GITHUB_TOKEN` env var set (needed to clone the private `unify` repo
  during Docker build).

### Build and deploy (staging)

```bash
cd scripts/job-watcher
GITHUB_TOKEN=ghp_... staging/deploy.sh --build
```

### Build and deploy (production)

```bash
cd scripts/job-watcher
GITHUB_TOKEN=ghp_... production/deploy.sh --build
```

### Deploy only (image already pushed)

```bash
staging/deploy.sh
# or
production/deploy.sh
```

### Verify

```bash
# Check the pod is running
kubectl get pods -n staging -l app=job-watcher

# Tail logs
kubectl logs -n staging -l app=job-watcher -f

# Check health
kubectl exec -n staging deploy/job-watcher -- curl -s localhost:8080/healthz
```

## Resource footprint

| Resource | Request | Limit |
|---|---|---|
| CPU | 50m | 100m |
| Memory | 64Mi | 128Mi |

## Environment variables

Injected via `unity-config` (ConfigMap) and `unity-secrets` (Secret):

| Variable | Source | Purpose |
|---|---|---|
| `ORCHESTRA_URL` | unity-config | Orchestra API base URL |
| `UNITY_COMMS_URL` | unity-config | Comms service base URL |
| `SHARED_UNIFY_KEY` | unity-secrets | Auth for Orchestra logs API |
| `ORCHESTRA_ADMIN_KEY` | unity-secrets | Auth for comms infra endpoints |
