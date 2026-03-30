# Dev Scripts

Developer utilities for managing Unity jobs, assistants, and logs against the live staging/production infrastructure.

Most scripts that talk to the comms service require two environment variables:

```bash
export UNITY_COMMS_URL="https://..."
export ORCHESTRA_ADMIN_KEY="..."
```

Scripts that talk to the adapters service pick the correct URL automatically based on `--env` (override with `--adapters-url`).

---

## idle_job_refresh.py

Create fresh idle K8s jobs and clean up stale ones. Designed to run after a Unity Cloud Build completes.

- Creates two new idle jobs via the adapters `/scheduled/jobs/create` endpoint.
- Waits 30 s (configurable) for the jobs to register as idle.
- Calls the adapters `/scheduled/jobs/cleanup` endpoint.
- Lists all job names before, after creation, and after cleanup (disable with `--no-list-jobs`).

```bash
python scripts/dev/idle_job_refresh.py                       # staging (default)
python scripts/dev/idle_job_refresh.py --env production      # production
python scripts/dev/idle_job_refresh.py --env preview         # preview
python scripts/dev/idle_job_refresh.py --no-list-jobs        # skip job listing
python scripts/dev/idle_job_refresh.py --delay 45            # custom wait
```

## suspend_job.py

Suspend (stop) a running Unity K8s job by name via the comms `/infra/job/stop` endpoint.

```bash
python scripts/dev/suspend_job.py                                        # auto-detect staging
python scripts/dev/suspend_job.py --env production                       # auto-detect production
python scripts/dev/suspend_job.py unity-2026-02-25-12-00-00              # explicit job
python scripts/dev/suspend_job.py unity-2026-02-25-12-00-00 --namespace custom-ns
```

## local_assistant.py

Create or fetch a local assistant from production Orchestra and print the `export` lines needed to run Unity on your machine.

```bash
# Create / fetch by name:
python scripts/dev/local_assistant.py --api-key YOUR_KEY --name "Dev Assistant"

# Fetch by ID:
python scripts/dev/local_assistant.py --api-key YOUR_KEY --id 42

# Source into your shell:
source <(python scripts/dev/local_assistant.py --api-key YOUR_KEY --name "Dev")

# Write to a .env file:
python scripts/dev/local_assistant.py --api-key YOUR_KEY --name "Dev" > .env.local
```

## keep_pod_alive.sh

Keep a deployed Unity pod alive by sending periodic keepalive pings to its Pub/Sub topic. Prevents the inactivity timeout from shutting down the container while you're debugging or developing.

```bash
./scripts/dev/keep_pod_alive.sh 25                        # staging (default), ping every 30s
./scripts/dev/keep_pod_alive.sh 25 --env production       # production
./scripts/dev/keep_pod_alive.sh 25 --env preview          # preview
./scripts/dev/keep_pod_alive.sh 25 --interval 60          # custom interval
```

Requires `gcloud` CLI authenticated with access to the `responsive-city-458413-a2` project.

## job_logs/

Tooling for streaming and inspecting Unity K8s job logs. See [`job_logs/README.md`](job_logs/README.md) for setup and usage.
