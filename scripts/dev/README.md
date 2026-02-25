# Dev Scripts

Developer utilities for managing Unity jobs, assistants, and logs against the live staging/production infrastructure.

Most scripts that talk to the comms service require two environment variables:

```bash
export UNITY_COMMS_URL="https://..."
export ORCHESTRA_ADMIN_KEY="..."
```

Scripts that talk to the adapters service pick the correct URL automatically based on `--staging` (override with `--adapters-url`).

---

## idle_job_refresh.py

Create fresh idle K8s jobs and clean up stale ones. Designed to run after a Unity Cloud Build completes.

- Creates two new idle jobs via the adapters `/scheduled/jobs/create` endpoint.
- Waits 30 s (configurable) for the jobs to register as idle.
- Calls the adapters `/scheduled/jobs/cleanup` endpoint.
- Lists all job names before, after creation, and after cleanup (disable with `--no-list-jobs`).

```bash
python scripts/dev/idle_job_refresh.py                 # prod (default)
python scripts/dev/idle_job_refresh.py --staging       # staging
python scripts/dev/idle_job_refresh.py --no-list-jobs  # skip job listing
python scripts/dev/idle_job_refresh.py --delay 45      # custom wait
```

## suspend_job.py

Suspend (stop) a running Unity K8s job by name via the comms `/infra/job/stop` endpoint.

```bash
python scripts/dev/suspend_job.py unity-2026-02-25-12-00-00              # prod
python scripts/dev/suspend_job.py unity-2026-02-25-12-00-00 --staging    # staging
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

## job_logs/

Tooling for streaming and inspecting Unity K8s job logs. See [`job_logs/README.md`](job_logs/README.md) for setup and usage.
