# Unity GKE Job Logs

View logs from Unity assistant jobs running on Google Kubernetes Engine.

## Quick Start

```bash
# 1. One-time auth setup
bash scripts/job_logs/setup_auth.sh

# 2. View logs for a job
uv run scripts/dev/job_logs/stream_logs.py --job <job_name>

# Example (staging is the default)
uv run scripts/dev/job_logs/stream_logs.py --job unity-2026-02-10-17-30-53-staging
```

That's it. The script figures out whether to stream or fetch based on the job's status.

## How It Works

`stream_logs.py` takes a GKE job name and namespace and does the right thing:

1. **Queries AssistantJobs** (via the Unify REST API) to check if the job is currently running, and prints session metadata (assistant name, user, medium, start time).
2. **If running** — the job has a live pod, so it prints all existing logs and then streams new ones in real-time using `kubectl logs -f` (press Ctrl+C to stop).
3. **If not running** — tries `kubectl logs` first (the pod may still exist), then falls back to fetching historical logs from GCP Cloud Logging via `gcloud logging read`.

## Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| `gcloud` | GCP authentication + Cloud Logging | `brew install --cask google-cloud-sdk` |
| `kubectl` | Log streaming from pods | `gcloud components install kubectl` |
| Python 3.10+ | Runs the script (uses the `unify` package) | Pre-installed on macOS/Linux |

### Environment Variable

```bash
# Add to ~/.zshrc or ~/.bashrc
export SHARED_UNIFY_KEY='your_key_here'
```

Ask a team member for the shared Unify API key. This is the same key used by `debug_logger.py` and the communication adapters to access the `AssistantJobs` project.

## Auth Setup

Run once (safe to re-run — skips completed steps):

```bash
bash ./debug_tools/setup_auth.sh
```

This walks you through:

1. Checking that `gcloud` and `kubectl` are installed
2. Authenticating your Google account (`gcloud auth login`)
3. Setting the active GCP project (`responsive-city-458413-a2`)
4. Fetching GKE cluster credentials for the `unity` cluster
5. Verifying cluster connectivity
6. Checking that `SHARED_UNIFY_KEY` is set

### GCP Permissions

Your Google account needs access to the `responsive-city-458413-a2` project. Ask a team admin to grant:

| Role | Purpose |
|------|---------|
| `roles/container.viewer` | Read pods and job resources via kubectl |
| `roles/logging.viewer` | Read historical logs from Cloud Logging |

## Finding the Job Name

Job names are tracked in the **AssistantJobs** project on the [Unify Console](https://console.unify.ai):

1. Open the **AssistantJobs** project
2. Go to the **startup_events** context
3. Filter by `assistant_id`, `user_name`, or timestamp
4. Copy the `job_name` field (e.g. `unity-2026-02-10-17-30-53-staging`)
5. Note the namespace the job runs in (e.g. `staging`, `production`)

## GCP Infrastructure Reference

| Setting | Value |
|---------|-------|
| GCP Project | `responsive-city-458413-a2` |
| GKE Cluster | `unity` |
| Region | `us-central1` |
| Namespaces | `staging`, `production`, etc. |

## Troubleshooting

### "gcloud: command not found"

Run `./setup_auth.sh` or install manually: `brew install --cask google-cloud-sdk`

### "kubectl: command not found"

`gcloud components install kubectl` or `brew install kubectl`

### "SHARED_UNIFY_KEY is not set"

Ask a team member for the key and add to your shell profile:

```bash
echo 'export SHARED_UNIFY_KEY="your_key"' >> ~/.zshrc && source ~/.zshrc
```

### "Failed to get cluster credentials"

Your GCP account needs project access. Ask a team admin for `container.viewer` and `logging.viewer` roles on project `responsive-city-458413-a2`.

### kubectl shows "No pods found" but job should be running

The AssistantJobs record might be stale. Check the job directly:

```bash
kubectl get jobs -n <namespace> | grep <job_name>
kubectl get pods -n <namespace> -l job-name=<job_name>
```

### gcloud logging returns empty results

- Cloud Logging has a retention period — very old logs may be unavailable.
- Double-check the job name (exact match required).

### "Permission denied" errors

Re-run `./setup_auth.sh` to refresh credentials.
