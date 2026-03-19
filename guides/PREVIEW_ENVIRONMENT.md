# Preview Environment

## Purpose

The **preview** environment is a third deployment tier alongside production and staging. It runs the full Unity stack in isolation — its own GKE namespace, Cloud Run services, Pub/Sub topics, K8s job-watcher, and scheduled maintenance — so that large, in-progress features can be deployed and tested end-to-end without affecting staging or production assistants.

Staging tracks `main` closely (auto-synced via GitHub Actions) and is the primary pre-production environment. Preview diverges intentionally: it carries feature branches that aren't ready for staging yet, and its infrastructure is fully isolated so a broken preview deploy can never impact staging assistants.

## Architecture

Preview reuses the same GCP project (`responsive-city-458413-a2`) and GKE cluster (`unity`) as staging and production, but runs in a separate K8s namespace (`preview`) with dedicated Cloud Run services:

| Component | Production | Staging | Preview |
|---|---|---|---|
| **GKE Namespace** | `production` | `staging` | `preview` |
| **Unity Image** | `unity:latest` | `unity-staging:latest` | `unity-preview:latest` |
| **Adapters** | `unity-adapters` | `unity-adapters-staging` | `unity-adapters-preview` |
| **Comms App** | `unity-comms-app` | `unity-comms-app-staging` | `unity-comms-app-preview` |
| **Job Watcher** | `job-watcher` | `job-watcher-staging` | `job-watcher-preview` |
| **Pub/Sub Topic Suffix** | (none) | `-staging` | `-preview` |
| **Job Name Suffix** | (none) | `-staging` | `-preview` |
| **Image Hash Bucket** | `image_hash.txt` | `image_hash_staging.txt` | `image_hash_preview.txt` |

All three environments share the same Orchestra (staging backend at `api.staging.internal.saas.unify.ai`), GCP secrets, and Artifact Registry. The environment boundary is enforced by `DEPLOY_ENV` and its derived `ENV_SUFFIX`, which partition resource names at every layer.

## How Environment Detection Works

### Unity (this repo)

Unity uses a pydantic-settings `SETTINGS` singleton (`unity/settings.py`). The `DEPLOY_ENV` field defaults to `"production"` and accepts `"staging"` or `"preview"` via environment variable. The container's entrypoint script (`entrypoint.sh`) detects the environment from the job name suffix and exports `DEPLOY_ENV` accordingly. The derived property `SETTINGS.ENV_SUFFIX` returns `""`, `"-staging"`, or `"-preview"`.

### Communication repo

The communication services use a `DEPLOY_ENV` environment variable read at import time in each service's helpers module. The adapters and comms app each define:

```python
def _get_deploy_env() -> str:
    deploy_env = (os.getenv("DEPLOY_ENV") or "production").strip().lower()
    return deploy_env if deploy_env in {"production", "staging", "preview"} else "production"

DEPLOY_ENV = _get_deploy_env()
ENV_SUFFIX = "" if DEPLOY_ENV == "production" else f"-{DEPLOY_ENV}"
```

Cloud Build sets `DEPLOY_ENV=preview` when deploying the preview services (visible in the `--set-env-vars` flag in `adapters-preview.yaml` and `--update-env-vars` in `unity-comms-app-preview.yaml`).

## What Gets Deployed

Three Cloud Build configs drive preview deployments:

### 1. Unity Container (`cloudbuild-preview.yaml`)

Triggered on push to the `preview` branch of the unity repo. Steps:

1. Build `unity-preview` Docker image (with `--build-arg BRANCH=preview`)
2. Push SHA and latest tags to Artifact Registry
3. Build `job-watcher-preview` image
4. Deploy job-watcher to the `preview` K8s namespace via `kubectl apply`
5. Upload commit SHA to `gs://unity-image-hash/image_hash_preview.txt`
6. Refresh idle jobs: hit `unity-adapters-preview` to create fresh idle containers and clean up stale ones

### 2. Adapters (`cloudbuild/adapters-preview.yaml`)

Triggered on push to the `preview` branch of the communication repo. Steps:

1. Build and push `unity-adapters-preview` image
2. Deploy to Cloud Run with `DEPLOY_ENV=preview`
3. Create or update Cloud Scheduler jobs for preview (email watches, job create/cleanup, stale expire, Microsoft tokens, Teams watches, cert renewal) — all suffixed with `-preview`

### 3. Comms App (`cloudbuild/unity-comms-app-preview.yaml`)

Triggered on push to the `preview` branch of the communication repo. Steps:

1. Build and push `unity-comms-app-preview` image
2. Update the existing Cloud Run service with `DEPLOY_ENV=preview`

## Branching Model

Both the `unity` and `communication` repos follow the same branch layout:

```
main ──── staging ──── preview
```

- `main` is production. Pushes trigger production Cloud Build.
- `staging` tracks main (auto-synced by GitHub Actions on push to main). Pushes trigger staging Cloud Build.
- `preview` diverges from staging to carry in-progress features. Pushes trigger preview Cloud Build.

### Merging Staging into Preview

Because preview carries features that may have been reverted on staging (to keep staging stable), a naive `git merge staging` can **silently undo** preview-only code. Git interprets the staging reverts as intentional deletions.

The safe merge strategy is:

1. Create a temporary branch from `origin/staging`
2. `git revert --no-edit` each revert commit on staging (re-applying the features)
3. Merge the temporary branch into `preview`

This makes Git aware that both branches intend for the features to exist, producing real conflicts only where the code genuinely diverges.

## Infrastructure Setup

### Prerequisites

Before the first preview build succeeds, these one-time setup steps are needed:

1. **K8s namespace**: `kubectl create namespace preview`

2. **K8s ConfigMap and Secrets**: The following must exist in the `preview` namespace. Copy them from staging (stripping metadata) if setting up from scratch:
   - `unity-config` (ConfigMap) — contains `PROJECT_ID`, `VERTEXAI_LOCATION`, `VERTEXAI_PROJECT`
   - `unity-secrets` (Secret) — contains API keys (`ORCHESTRA_ADMIN_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.)
   - `comm-sa-key` (Secret) — GCP service account key used by Unity containers
   - `desktop-unify-wildcard-tls` (Secret) — TLS wildcard certificate

   ```bash
   # Example: copy a resource from staging to preview
   kubectl get secret unity-secrets -n staging -o json | \
     python3 -c "import sys,json; o=json.load(sys.stdin); o['metadata']={'name':o['metadata']['name'],'namespace':'preview','labels':o['metadata'].get('labels',{})}; json.dump(o,sys.stdout)" | \
     kubectl apply -f -
   ```

3. **Service account**: The `comm-sa` K8s ServiceAccount must exist in the `preview` namespace. Unity jobs and the job-watcher both reference it via `serviceAccountName: comm-sa`. It is **not** created by the deployment YAML (same as staging/production — it was created separately).

4. **Cloud Run services**: The adapters and comms app Cloud Run services (`unity-adapters-preview`, `unity-comms-app-preview`) must exist. The adapters build uses `gcloud run deploy` (creates if absent), but the comms build uses `gcloud run services update` (requires pre-existing service).

5. **Cloud Run IAM policy**: Each Cloud Run service needs unauthenticated access enabled so inter-service calls work. Without this, Cloud Run returns a 401 at the infrastructure level before the request ever reaches the application:

   ```bash
   gcloud run services add-iam-policy-binding unity-comms-app-preview \
     --region=us-central1 --member=allUsers --role=roles/run.invoker \
     --project=responsive-city-458413-a2

   gcloud run services add-iam-policy-binding unity-adapters-preview \
     --region=us-central1 --member=allUsers --role=roles/run.invoker \
     --project=responsive-city-458413-a2
   ```

   This is a one-time step — the binding persists across deployments. Staging and production already have this.

6. **Pub/Sub resources**: Preview-suffixed topics and subscriptions must exist for features that use Pub/Sub (e.g., `unity-pending-startups-preview` and `unity-pending-startups-preview-sub`). These are not auto-created by the application.

7. **Cloud Build triggers**: Create triggers in the GCP console for each cloudbuild YAML, filtered to the `preview` branch of the respective repo.

### Scheduled Jobs

The adapters preview build automatically creates Cloud Scheduler jobs (with update-or-create logic) for:

| Scheduler Job | Schedule | Endpoint |
|---|---|---|
| `email-watches-preview` | Daily | `/scheduled/email-watches` |
| `jobs-create-preview` | Hourly | `/scheduled/jobs/create?refresh=true` |
| `jobs-cleanup-preview` | :10 past each hour | `/scheduled/jobs/cleanup` |
| `stale-jobs-expire-preview` | Every 6 hours | `/scheduled/jobs/expire-stale` |
| `microsoft-tokens-preview` | Every 30 min | `/scheduled/microsoft-tokens` |
| `teams-watches-preview` | Every 30 min | `/scheduled/teams-watches` |
| `cert-renewal-preview` | 1st of month, 3 AM | `/scheduled/cert-renewal` |

## Differences from Staging

### Feature Branches

Preview carries features not yet on staging — for example, the K8s Lease-based atomic container assignment and pending-startup queue in the communication repo, and the K8s annotation polling startup flow in unity. These replace the older Pub/Sub competing-consumer startup mechanism.

### Environment Configuration

The communication repo's `preview` branch uses the `DEPLOY_ENV` / `ENV_SUFFIX` pattern directly (module-level constants), while the `staging` branch on the communication repo was refactored to use a centralized `Settings` class (`common/settings.py`). Both approaches produce equivalent behaviour — the difference is structural.

**Important caveat**: `common/settings.py` still exists on the preview branch and is imported by some modules (e.g., `communication/dependencies.py`). The `Settings` class determines staging/production via a `STAGING` env var (`os.environ.get("STAGING", "false")`), **not** `DEPLOY_ENV`. When `STAGING` is absent (as it is on preview — only `DEPLOY_ENV=preview` is set), `_is_staging()` returns `False` and URLs resolve to **production** defaults. Any code path still using `SETTINGS.orchestra_url` or `SETTINGS.comms_url` will get production URLs unless `ORCHESTRA_URL` / `UNITY_COMMS_URL` are explicitly set as env vars (which they are on the Cloud Run services). Modules that were migrated to the `DEPLOY_ENV` pattern import URLs directly and are unaffected.

### Isolation

Preview assistants use completely separate:
- Pub/Sub topics (`unity-{id}-preview`, `unity-startup-preview`)
- Pub/Sub subscriptions (`*-preview-sub`)
- K8s jobs (suffixed `-preview`, running in the `preview` namespace)
- Cloud Run services (separate URL endpoints)
- Cloud Scheduler jobs (separate schedules, can be paused independently)
- Idle job pool (independent target count and refresh cycle)

A preview assistant can never accidentally consume a staging idle container or vice versa, because they subscribe to different startup topics and run in different K8s namespaces.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Build fails at "deploy-job-watcher" with `namespaces "preview" not found` | K8s `preview` namespace doesn't exist | `kubectl create namespace preview` |
| Job-watcher pod in `CreateContainerConfigError` | `unity-config` ConfigMap or `unity-secrets` Secret missing in the `preview` namespace | Copy the ConfigMap/Secrets from staging (see Prerequisites) |
| Unity job pods stuck in `Pending` with `FailedMount` | `comm-sa-key` Secret missing in the `preview` namespace | Copy the secret from staging |
| Unity job pods fail with `error looking up service account preview/comm-sa` | `comm-sa` ServiceAccount missing in the `preview` namespace | `kubectl create serviceaccount comm-sa -n preview` |
| Comms app build fails at "Deploy Unity Comms App" | Cloud Run service doesn't exist yet (uses `update` not `deploy`) | Create the service first via `gcloud run deploy unity-comms-app-preview ...` |
| Comms app returns `ImportError` at startup | Preview-only functions missing from `helpers.py` after merge | Ensure preview-only features (e.g., `publish_pending_startup`) are present in the deployed code |
| Adapters get 401 calling comms app (crash with `JSONDecodeError`) | Cloud Run IAM policy not set — Cloud Run rejects at infra level | `gcloud run services add-iam-policy-binding unity-comms-app-preview --member=allUsers --role=roles/run.invoker ...` |
| `/pending/process` returns 500 with `NotFound: unity-pending-startups-preview-sub` | Pub/Sub topic/subscription not created for preview | Create the topic and subscription in GCP |
| Idle jobs not created after build | `unity-adapters-preview` Cloud Run service not reachable, or admin key missing | Check the adapters service is deployed and `ORCHESTRA_ADMIN_KEY` secret exists |
| Preview assistant not receiving messages | Pub/Sub topic not provisioned, or wrong `DEPLOY_ENV` | Verify the assistant's topic exists (`unity-{id}-preview`) and the adapters have `DEPLOY_ENV=preview` |
| Merge from staging silently drops preview features | Git treats staging reverts as intentional deletions | Use the "revert the reverts" merge strategy described above |
