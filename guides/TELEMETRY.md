# Telemetry Reference

Everything currently being recorded to Prometheus / GCP Cloud Monitoring across
the **adapters**, **communication app**, and **Unity** services.

This document is the single source of truth for the metrics pipeline. Update it
whenever metrics are added, modified, or removed.

---

## Architecture Overview

```
┌────────────────────────────┐   ┌─────────────────────────────┐
│  Adapters  (Cloud Run)     │   │  Comms App  (Cloud Run)     │
│  prometheus_client /metrics│   │  prometheus_client /metrics  │
│                            │   │                             │
│  GMP sidecar scrapes :8080 │   │  GMP sidecar scrapes :8080  │
└────────────┬───────────────┘   └──────────────┬──────────────┘
             │                                   │
             ▼                                   ▼
┌──────────────────────────────────────────────────────────────┐
│              GCP Managed Service for Prometheus               │
│                     (Monarch backend)                         │
│                                                              │
│  Custom app metrics  →  prometheus.googleapis.com/*           │
│  Built-in GCP metrics →  run.googleapis.com/*                │
│                          kubernetes.io/*                      │
│                          pubsub.googleapis.com/*              │
└──────────────────────────────┬───────────────────────────────┘
                               │
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
┌────────────────────────┐        ┌──────────────────────────┐
│  Grafana               │        │  GCP Metrics Explorer    │
│  (GKE, staging ns)     │        │  (PromQL mode)           │
│                        │        │                          │
│  Data sources:         │        │  Quick ad-hoc queries    │
│  1. Google Cloud       │        │  without Grafana         │
│     Monitoring         │        └──────────────────────────┘
└────────────────────────┘

┌────────────────────────────┐
│  Unity  (GKE Jobs)         │
│  OpenTelemetry SDK         │
│  → GCP Monitoring exporter │
│    (HTTPS push, 5s)        │
└────────────────────────────┘
```

### Collection Mechanisms

| Service | Library | Export Method | Interval |
|---------|---------|--------------|----------|
| **Adapters** (Cloud Run) | `prometheus_client` | GMP sidecar scrapes `/metrics` | ~30 s |
| **Comms App** (Cloud Run) | `prometheus_client` | GMP sidecar scrapes `/metrics` | ~30 s |
| **Unity** (GKE Jobs) | `opentelemetry-sdk` + `opentelemetry-exporter-gcp-monitoring` | HTTPS push to GCP Monitoring API | 5 s |

Unity uses push-based export (not a sidecar) because its containers are
ephemeral GKE Jobs that may terminate before a scrape cycle completes.

### Grafana

Deployed on GKE (`staging` namespace) with a Google Cloud Monitoring data source.
Accessible at `https://grafana.staging.internal.saas.unify.ai`.

Deployment manifests: `unity/scripts/grafana/`

### GMP Sidecar (Cloud Run)

Both Cloud Run services run the
`us-docker.pkg.dev/cloud-ops-agents-artifacts/cloud-run-gmp-sidecar/cloud-run-gmp-sidecar:1.2.0`
sidecar container. Setup script: `communication/scripts/setup_gmp_sidecar.py`

---

## 1. Adapters — Custom Metrics

Source: `communication/common/metrics.py`, `communication/adapters/helpers.py`, `communication/adapters/main.py`

### 1.1 HTTP Request Duration

| | |
|---|---|
| **Prometheus name** | `http_request_duration_seconds` |
| **Type** | Histogram |
| **Labels** | `service="adapters"`, `method`, `endpoint`, `status_code` |
| **Buckets** | 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30 |
| **Description** | Latency of every HTTP request by method, route pattern, and status code. |
| **Recorded by** | Middleware in `common/metrics.py` (automatic for all routes, skips `/metrics`) |

### 1.2 HTTP Request Count

| | |
|---|---|
| **Prometheus name** | `http_requests_total` |
| **Type** | Counter |
| **Labels** | `service="adapters"`, `method`, `endpoint`, `status_code` |
| **Description** | Total request count by method, route pattern, and status code. |
| **Recorded by** | Same middleware as above |

### 1.3 Orchestra `get_assistant()` Duration

| | |
|---|---|
| **Prometheus name** | `orchestra_get_assistant_duration_seconds` |
| **Type** | Histogram |
| **Labels** | `lookup_type` (`id` / `email` / `phone`), `status` (`success` / `error`) |
| **Buckets** | 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10 |
| **Description** | Time calling Orchestra `GET /admin/assistant`. Filter `status="success"` for healthy latency. |
| **Recorded by** | `adapters/helpers.py` → `get_assistant()` (`perf_counter` in `finally` block) |

### 1.4 `mark_job_running()` Duration

| | |
|---|---|
| **Prometheus name** | `mark_job_running_duration_seconds` |
| **Type** | Histogram |
| **Labels** | `status` (`success` / `error`) |
| **Buckets** | 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10 |
| **Description** | Time marking an AssistantJob as running via Orchestra `/logs`. |
| **Recorded by** | `adapters/helpers.py` → `mark_job_running()` (`perf_counter` in `finally` block) |

### 1.5 `build_webhook_context()` Duration

| | |
|---|---|
| **Prometheus name** | `build_webhook_context_duration_seconds` |
| **Type** | Histogram |
| **Labels** | `channel` (phone / msg / whatsapp / email / teams / unify_message / etc.), `job_started` (`true` / `false`), `status` (`success` / `error`) |
| **Buckets** | 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60 |
| **Description** | Total time from inbound request to webhook context built. `job_started=true` includes the full path (mark_job_running + start_unity_job + create_job); `job_started=false` is just get_assistant + contact validation. |
| **Recorded by** | `adapters/helpers.py` → `build_webhook_context()` |

### Metrics Endpoint

`GET /metrics` mounted on the FastAPI app via `setup_metrics(app, service_name="adapters")`.
Returns Prometheus text exposition format. Also includes `ProcessCollector` (process-level CPU/memory/fd stats).

---

## 2. Comms App — Custom Metrics

Source: `communication/common/metrics.py`, `communication/communication/main.py`

### 2.1 HTTP Request Duration

| | |
|---|---|
| **Prometheus name** | `http_request_duration_seconds` |
| **Type** | Histogram |
| **Labels** | `service="comms"`, `method`, `endpoint`, `status_code` |
| **Buckets** | 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30 |
| **Description** | Latency of every HTTP request (same metric name as adapters, differentiated by `service` label). |
| **Recorded by** | Middleware in `common/metrics.py` (automatic for all routes, skips `/metrics`) |

### 2.2 HTTP Request Count

| | |
|---|---|
| **Prometheus name** | `http_requests_total` |
| **Type** | Counter |
| **Labels** | `service="comms"`, `method`, `endpoint`, `status_code` |
| **Description** | Total request count (same metric name as adapters, differentiated by `service` label). |
| **Recorded by** | Same middleware as above |

### Metrics Endpoint

`GET /metrics` mounted via `setup_metrics(app, service_name="comms")`.
Same shared registry and middleware as adapters.

> **Note:** The comms app currently only records the shared HTTP metrics (request
> duration and count). No comms-specific business metrics (external service call
> durations, infra operation durations, etc.) are instrumented yet.

---

## 3. Unity — Custom Metrics

Source: `unity/conversation_manager/metrics.py`, `unity/conversation_manager/metrics_push.py`,
`unity/conversation_manager/assistant_jobs.py`, `unity/conversation_manager/domains/managers_utils.py`,
`unity/conversation_manager/main.py`, `entrypoint.sh`

### 3.1 Container Spin-Up Time (U1)

| | |
|---|---|
| **OTel name** | `unity_container_spinup_seconds` |
| **Type** | Histogram |
| **Unit** | seconds |
| **Labels** | — |
| **Description** | Time from container start (`entrypoint.sh`) to CommsManager start. |
| **Recorded by** | `main.py` → `run_conversation_manager()`, computed as `time.time()*1000 - CONTAINER_START_TIME_MS` |
| **Condition** | Only when `SESSION_DETAILS.assistant.id == UNASSIGNED_ASSISTANT_ID` (idle cloud containers) |

`CONTAINER_START_TIME_MS` is exported as an env var at the top of `entrypoint.sh`
(`date +%s%3N`).

### 3.2 Manager Init Total (U2)

| | |
|---|---|
| **OTel name** | `unity_manager_init_seconds` |
| **Type** | Histogram |
| **Unit** | seconds |
| **Labels** | — |
| **Description** | Total duration of `_init_managers()` (all managers combined). |
| **Recorded by** | `managers_utils.py` → end of `_init_managers()` |

### 3.3 Per-Manager Init (U3)

| | |
|---|---|
| **OTel name** | `unity_per_manager_init_seconds` |
| **Type** | Histogram |
| **Unit** | seconds |
| **Labels** | `manager` |
| **Description** | Init time per individual manager step. |
| **Recorded by** | `managers_utils.py` → after each manager init block |

Label values (7 steps):

| `manager` value | What it measures |
|---|---|
| `unity` | `unity.init()` call |
| `event_bus` | EventBus configuration |
| `contact_manager` | ContactManager initialization |
| `transcript_manager` | TranscriptManager initialization |
| `memory_manager` | MemoryManager initialization (only if enabled) |
| `conversation_manager_handle` | ConversationManagerHandle initialization |
| `actor` | Actor initialization |

### 3.4 Session Duration (U9)

| | |
|---|---|
| **OTel name** | `unity_session_duration_seconds` |
| **Type** | Histogram |
| **Unit** | seconds |
| **Labels** | — |
| **Description** | Total assistant session duration from startup to shutdown. |
| **Recorded by** | `assistant_jobs.py` → `mark_job_done()`, computed from `perf_counter` delta since `log_job_startup()` |

### 3.5 Running Job Count (X1)

| | |
|---|---|
| **OTel name** | `unity_running_job_count` |
| **Type** | Gauge |
| **Labels** | — |
| **Description** | Cluster-wide count of assistant jobs with `running==True` at the moment the metric is sampled. |
| **Recorded by** | `assistant_jobs.py` → `_record_running_job_count()` (queries `unify.get_logs()` with `filter="running == 'true'"`) |
| **Triggered at** | Immediately after `log_job_startup()` updates the record, and immediately after `mark_job_done()` clears the record |

### Metrics Lifecycle

| Function | File | When called |
|---|---|---|
| `init_metrics()` | `main.py` | On startup, only if `assistant.id == UNASSIGNED_ASSISTANT_ID` |
| `flush_metrics()` | `main.py` | During `main()` cleanup, before shutdown |
| `shutdown_metrics()` | `main.py` | Immediately after `flush_metrics()` |

**Guard conditions** (metrics are silently no-ops when any of these hold):
- `TEST` env var is set (unit tests)
- `GOOGLE_APPLICATION_CREDENTIALS` env var is not set (local dev)
- Assistant ID is not `UNASSIGNED_ASSISTANT_ID` (pre-specified assistant, not a cloud idle container)

---

## 4. GCP Built-In Metrics (Zero Setup)

These are collected automatically by GCP Cloud Monitoring. No code or deployment
changes are needed. They appear in Grafana via the Google Cloud Monitoring data source.

### 4.1 Cloud Run (Adapters + Comms App)

| GCP Metric | What It Shows |
|---|---|
| `run.googleapis.com/container/cpu/utilizations` | Per-instance CPU usage (0–1) |
| `run.googleapis.com/container/memory/utilizations` | Per-instance memory usage (0–1) |
| `run.googleapis.com/container/instance_count` | Running instance count (autoscaling) |
| `run.googleapis.com/container/startup_latencies` | Cold start duration per instance |
| `run.googleapis.com/request_count` | Requests bucketed by status code |
| `run.googleapis.com/request_latencies` | End-to-end request duration distribution |
| `run.googleapis.com/container/max_concurrent_requests` | Concurrency pressure per instance |
| `run.googleapis.com/container/network/sent_bytes_count` | Outbound network traffic |
| `run.googleapis.com/container/network/received_bytes_count` | Inbound network traffic |
| `run.googleapis.com/container/billable_instance_time` | Billable seconds consumed |

Filter by `service_name`: `unity-adapters` or `unity-comms-app`.

### 4.2 GKE (Unity Containers)

| GCP Metric | What It Shows |
|---|---|
| `kubernetes.io/container/cpu/core_usage_time` | CPU time consumed per pod |
| `kubernetes.io/container/memory/used_bytes` | Memory used per pod |
| `kubernetes.io/container/restart_count` | Container restarts (crash detection) |

Filter by `namespace_name` and pod name pattern `unity-*`.

### 4.3 Pub/Sub

| GCP Metric | What It Shows |
|---|---|
| `pubsub.googleapis.com/subscription/num_undelivered_messages` | Queue depth / backlog |
| `pubsub.googleapis.com/subscription/oldest_unacked_message_age` | How long messages are waiting |
| `pubsub.googleapis.com/topic/send_message_operation_count` | Traffic volume per topic |

---

## 5. PromQL Quick Reference

Example queries for the custom metrics. Use these in Grafana or GCP Metrics Explorer (PromQL mode).

### Adapters

```promql
# p95 request latency per endpoint (last 5 min)
histogram_quantile(0.95,
  rate(http_request_duration_seconds_bucket{service="adapters"}[5m])
)

# Error rate (5xx) per endpoint
sum(rate(http_requests_total{service="adapters", status_code=~"5.."}[5m])) by (endpoint)
/
sum(rate(http_requests_total{service="adapters"}[5m])) by (endpoint)

# p95 get_assistant latency (successful calls only)
histogram_quantile(0.95,
  rate(orchestra_get_assistant_duration_seconds_bucket{status="success"}[5m])
)

# Average build_webhook_context duration by channel (job started path)
rate(build_webhook_context_duration_seconds_sum{job_started="true", status="success"}[5m])
/
rate(build_webhook_context_duration_seconds_count{job_started="true", status="success"}[5m])
```

### Comms App

```promql
# p95 request latency per endpoint
histogram_quantile(0.95,
  rate(http_request_duration_seconds_bucket{service="comms"}[5m])
)
```

### Unity

Unity metrics land in GCP Cloud Monitoring as `custom.googleapis.com/opencensus/unity_*`
or `workload.googleapis.com/unity_*` (depending on the exporter version). Query via
the Google Cloud Monitoring data source in Grafana or via Metrics Explorer PromQL mode.

```promql
# Container spin-up time distribution
unity_container_spinup_seconds

# Per-manager init time
unity_per_manager_init_seconds{manager="actor"}

# Current running job count
unity_running_job_count

# Session duration distribution
unity_session_duration_seconds
```

---

## 6. File Inventory

### Communication Repository

| File | Role |
|---|---|
| `common/metrics.py` | Metric definitions, shared middleware, `/metrics` endpoint setup |
| `adapters/main.py` | Calls `setup_metrics(app, "adapters")` |
| `adapters/helpers.py` | Records `get_assistant`, `mark_job_running`, `build_webhook_context` durations |
| `communication/main.py` | Calls `setup_metrics(app, "comms")` |
| `scripts/setup_gmp_sidecar.py` | Adds GMP sidecar to Cloud Run deployments |

### Unity Repository

| File | Role |
|---|---|
| `entrypoint.sh` | Exports `CONTAINER_START_TIME_MS` |
| `unity/conversation_manager/metrics.py` | OTel metric instrument definitions (5 metrics) |
| `unity/conversation_manager/metrics_push.py` | `init_metrics()` / `flush_metrics()` / `shutdown_metrics()` lifecycle |
| `unity/conversation_manager/main.py` | Initializes metrics, records U1, calls flush/shutdown on exit |
| `unity/conversation_manager/domains/managers_utils.py` | Records U2 (total init) and U3 (per-manager init, 7 steps) |
| `unity/conversation_manager/assistant_jobs.py` | Records U9 (session duration) and X1 (running job count) |
| `scripts/grafana/` | Grafana deployment manifests (deployment, service, ingress, cert, datasource) |
