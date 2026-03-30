# Deployment (Internal)

This directory contains deployment configurations and operational tooling for the
hosted Unify platform. If you're using Unity as an open-source project, you can
ignore this directory entirely — nothing here is needed for local development or
the sandbox.

## Contents

- `Dockerfile` / `entrypoint.sh` — Production container image
- `cloudbuild*.yaml` — GCP Cloud Build CI/CD pipelines
- `desktop/` — Virtual desktop container image (VNC/noVNC + browser automation)
- `guides/` — Internal infrastructure documentation (GKE, call recording, etc.)
- `scripts/dev/` — Developer utilities for managing live K8s jobs, assistants, and logs
- `scripts/job-watcher/` — Kopf operator for K8s job lifecycle management
- `scripts/kubernetes/` — Cluster setup, priority classes, job creation helpers
- `scripts/stress_test/` — Load testing and multi-assistant orchestration
- `scripts/visualizer/` — Log visualization tool (requires GCP access)
