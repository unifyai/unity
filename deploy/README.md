# Deployment (Internal)

This directory contains deployment configurations for the hosted Unify platform.
If you're using Unity as an open-source project, you can ignore this directory entirely.

## Contents

- `Dockerfile` / `entrypoint.sh` — Production container image
- `cloudbuild*.yaml` — GCP Cloud Build pipelines
- `desktop/` — Virtual desktop container image
- `scripts/` — K8s operators, cluster setup, infrastructure management
- `guides/` — Internal infrastructure documentation
