# Deployment

Docker image and CI/CD configuration for the Unity container.

## Contents

- `Dockerfile` — Production container image (Python, Node, system deps, agent-service, browser automation)
- `entrypoint.sh` — Container entrypoint (memory watchdog, display setup, app startup)
- `cloudbuild.yaml` / `cloudbuild-staging.yaml` / `cloudbuild-preview.yaml` — GCP Cloud Build pipelines
- `desktop/` — Virtual desktop stack (VNC/noVNC, audio devices, browser) for computer-use sessions

## Docker

Build the image locally:

```bash
docker build -f deploy/Dockerfile \
  --build-arg GITHUB_TOKEN=your-token \
  --build-arg UNIFY_KEY=your-key \
  -t unity .
```

The Dockerfile clones `unify` and `unillm` from GitHub at build time (they're not bundled in the image context). A `GITHUB_TOKEN` with repo read access is required.

## Cloud Build

The Cloud Build configs are triggered by pushes to `main`, `staging`, and `preview`. They build the base image, push to Artifact Registry, and trigger a downstream enterprise overlay build.

These configs use `${PROJECT_ID}` and other Cloud Build substitution variables — no credentials are hardcoded.
