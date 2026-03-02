# Grafana on GKE

**THESE ARE ONE-TIME DEPLOYMENTS. DO NOT RE-DEPLOY.**

Deploys Grafana instances to the `staging` and `production` namespaces on the
`unity` GKE cluster (project `responsive-city-458413-a2`). Each instance is
pre-configured with a Google Cloud Monitoring data source for viewing built-in
metrics from Cloud Run, GKE, and Pub/Sub.

| Environment | URL | Namespace | Manifests |
|-------------|-----|-----------|-----------|
| Staging | `https://grafana.staging.internal.saas.unify.ai` | `staging` | `staging/` |
| Production | `https://grafana.internal.saas.unify.ai` | `production` | `production/` |

---

## Architecture Overview

```
Browser
  │
  │  HTTPS (grafana.staging.internal.saas.unify.ai)
  ▼
Google Cloud HTTP(S) Load Balancer    ← created automatically by GKE Ingress
  │                                      (project: responsive-city-458413-a2)
  │  TLS terminated here via
  │  GKE ManagedCertificate
  ▼
Ingress (grafana-staging)
  │
  ▼
Service (grafana-staging, ClusterIP :80)
  │
  ▼
Pod (grafana-staging, Grafana 11.5.1 on :3000)
  │
  │  Google Cloud Monitoring API
  │  (authenticated via comm-sa service account key)
  ▼
GCP Cloud Monitoring
  ├── Cloud Run metrics (adapters, comms app)
  ├── GKE / Kubernetes metrics (unity jobs)
  └── Pub/Sub metrics
```

### Cross-project DNS

The GKE cluster lives in `responsive-city-458413-a2`, but the domain
`*.internal.saas.unify.ai` is managed in Cloud DNS in the `saas-368716` project.
The Ingress creates a load balancer with an external IP in the cluster's project,
and we add an A record in the other project's Cloud DNS pointing to that IP.

---

## Directory Structure

```
scripts/grafana/
├── README.md                        # this file
├── staging/                         # staging namespace manifests
│   ├── deploy.sh
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── managed-certificate.yaml
│   └── datasource-configmap.yaml
└── production/                      # production namespace manifests
    ├── deploy.sh
    ├── deployment.yaml
    ├── service.yaml
    ├── ingress.yaml
    ├── managed-certificate.yaml
    └── datasource-configmap.yaml
```

Each environment folder contains the same set of files:

| File | Description |
|------|-------------|
| `deploy.sh` | One-command deployment script (creates secrets, applies all manifests, waits for IP) |
| `deployment.yaml` | Grafana pod — image, env vars, volume mounts, probes, resource limits |
| `service.yaml` | ClusterIP Service exposing Grafana internally on port 80 |
| `ingress.yaml` | GKE Ingress with ManagedCertificate annotation for HTTPS |
| `managed-certificate.yaml` | GKE-native TLS certificate (auto-provisioned and auto-renewed by Google) |
| `datasource-configmap.yaml` | Auto-provisions the Google Cloud Monitoring data source on Grafana startup |

---

## Deployment

Both environments follow the same process. Replace `<env>` with `staging` or
`production` below.

### Prerequisites

1. **kubectl pointed at the unity cluster:**
   ```bash
   gcloud container clusters get-credentials unity \
     --region us-central1 \
     --project responsive-city-458413-a2
   ```

2. **`comm-sa-key` secret exists in the target namespace** (contains the GCP
   service account key used by Grafana to authenticate with Cloud Monitoring).

### Quick Start (deploy.sh)

```bash
cd scripts/grafana/<env>

# Random admin password (printed once — save it)
./deploy.sh

# Or specify a password
./deploy.sh --password <your-password>
```

The script will:
1. Create the admin password Kubernetes Secret
2. Apply the datasource ConfigMap
3. Apply the Deployment
4. Apply the Service
5. Apply the Ingress
6. Wait for the pod to be ready
7. Poll for the Ingress external IP (up to 5 minutes)
8. Print the `gcloud dns record-sets create` command for DNS setup

### After the script finishes

1. **Add the DNS A record** (the script prints the exact command):
   ```bash
   gcloud dns record-sets create <DOMAIN> \
     --zone=internal-saas \
     --type=A \
     --ttl=300 \
     --rrdatas=<EXTERNAL_IP> \
     --project=saas-368716
   ```

2. **Apply the ManagedCertificate for HTTPS:**
   ```bash
   kubectl apply -f managed-certificate.yaml
   kubectl apply -f ingress.yaml
   ```
   The certificate takes 10-15 minutes to provision. Monitor with:
   ```bash
   kubectl get managedcertificate <CERT_NAME> -n <env> --watch
   ```
   Once status shows `Active`, HTTPS is live.

3. **Verify the data source in Grafana:**
   - Log in at the environment URL
   - Go to **Connections > Data Sources > Google Cloud Monitoring**
   - Click **Save & Test** — it should succeed immediately (the data source
     uses `gce` authentication, which picks up the SA key mounted from the
     `comm-sa-key` secret at `/secrets/key.json` via `GOOGLE_APPLICATION_CREDENTIALS`)

---

## Troubleshooting

### Pod won't start — plugin not found

If you see `404: plugin not found` in pod logs, it means `GF_INSTALL_PLUGINS`
contains an invalid plugin ID. The Google Cloud Monitoring data source
(`stackdriver` type) is a **core built-in plugin** since Grafana 7.x — no
`GF_INSTALL_PLUGINS` env var is needed.

**Fix:** Remove the `GF_INSTALL_PLUGINS` env var from `deployment.yaml` and
re-apply:
```bash
kubectl apply -f deployment.yaml
```

### Ingress stuck with no IP — "secrets not found"

The GCE Ingress controller requires the TLS secret to already exist. If you use
a `cert-manager` annotation (`cert-manager.io/cluster-issuer`) but cert-manager
isn't running, the Ingress will fail to sync with:

```
Error syncing to GCP: error initializing translator env: secrets "grafana-tls" not found
```

**Fix:** Use a GKE `ManagedCertificate` instead of cert-manager. Remove any
`cert-manager.io/*` annotations and TLS block from the Ingress, and use the
`networking.gke.io/managed-certificates` annotation (as configured in the current
`ingress.yaml`).

### Ingress has IP but returns 502

The GCE load balancer health checks haven't passed yet. Check backend status:
```bash
kubectl describe ingress <INGRESS_NAME> -n <NAMESPACE> | grep backends
```

Wait until all backends show `HEALTHY` (can take 3-5 minutes after IP
assignment). If a backend stays `Unknown` or `UNHEALTHY`, check the pod:
```bash
kubectl logs -n <NAMESPACE> -l app=<APP_LABEL>
kubectl get pods -n <NAMESPACE> -l app=<APP_LABEL>
```

### ManagedCertificate stuck in Provisioning

Google needs to verify domain ownership via the DNS A record. Ensure:
1. The A record exists and resolves: `dig <DOMAIN> +short`
2. The resolved IP matches the Ingress IP: `kubectl get ingress <INGRESS_NAME> -n <NAMESPACE>`

If DNS is correct, just wait — provisioning can take up to 15 minutes. Check
status:
```bash
kubectl describe managedcertificate <CERT_NAME> -n <NAMESPACE>
```

### cert-manager webhook errors

If you see `failed calling webhook "webhook.cert-manager.io"`, cert-manager's
CRDs are installed but the controller pods are not running:
```bash
kubectl get pods -n cert-manager
```

If empty, cert-manager is not operational. Use `ManagedCertificate` instead.

---

## Updating the DNS Record

If the Ingress IP changes (e.g. after deleting and recreating the Ingress):

```bash
# Delete old record
gcloud dns record-sets delete <DOMAIN> \
  --zone=internal-saas \
  --type=A \
  --project=saas-368716

# Create new record
gcloud dns record-sets create <DOMAIN> \
  --zone=internal-saas \
  --type=A \
  --ttl=300 \
  --rrdatas=<NEW_IP> \
  --project=saas-368716
```

---

## Tearing Down

Replace resource names and namespace for the target environment (see the
[Kubernetes Resources Summary](#kubernetes-resources-summary) table).

```bash
kubectl delete ingress <INGRESS_NAME> -n <NAMESPACE>
kubectl delete svc <SERVICE_NAME> -n <NAMESPACE>
kubectl delete deployment <DEPLOYMENT_NAME> -n <NAMESPACE>
kubectl delete configmap <CONFIGMAP_NAME> -n <NAMESPACE>
kubectl delete secret <SECRET_NAME> -n <NAMESPACE>
kubectl delete managedcertificate <CERT_NAME> -n <NAMESPACE>

gcloud dns record-sets delete <DOMAIN> \
  --zone=internal-saas \
  --type=A \
  --project=saas-368716
```

Note: deleting the Ingress can take several minutes as GKE tears down the
associated HTTP(S) Load Balancer.

---

## Dashboard Queries

Since dashboards are lost on pod restart (`emptyDir` storage), this section
documents the PromQL queries used so they can be recreated. All queries below
use the **Google Managed Prometheus** data source.

### Live Assistants (Supply)

Snapshot of how many assistant jobs have `running==True` at any point in time.
This is a gauge — no `rate()` needed. Use `max()` because each container
independently reports the same global count; `sum()` would multiply it by the
number of reporting containers.

```promql
max({__name__="workload.googleapis.com/unity_running_job_count", monitored_resource="k8s_container"})
```

### Live Assistants (Demand)

Number of inbound requests that required starting a new container (i.e., no
container was already running for that assistant).

```promql
sum(increase(adapter_job_demand_total[5m]))
```

By channel:

```promql
sum by (channel) (increase(adapter_job_demand_total[5m]))
```

### Average Session Duration (minutes)

How long assistant sessions last, averaged over a 5-minute window.

```promql
sum(rate({__name__="workload.googleapis.com/unity_session_duration_seconds_sum", monitored_resource="k8s_container"}[5m]))
/
sum(rate({__name__="workload.googleapis.com/unity_session_duration_seconds_count", monitored_resource="k8s_container"}[5m]))
/ 60
```

### Average Container Spinup Time (seconds)

Time from container start to ConversationManager `main()` being called.

```promql
sum(rate({__name__="workload.googleapis.com/unity_container_spinup_seconds_sum", monitored_resource="k8s_container"}[5m]))
/
sum(rate({__name__="workload.googleapis.com/unity_container_spinup_seconds_count", monitored_resource="k8s_container"}[5m]))
```

### Average Build Context Duration (seconds)

End-to-end time from inbound adapter request to webhook context built.

```promql
sum(rate(build_webhook_context_duration_seconds_sum[5m]))
/
sum(rate(build_webhook_context_duration_seconds_count[5m]))
```

By channel:

```promql
sum by (channel) (rate(build_webhook_context_duration_seconds_sum[5m]))
/
sum by (channel) (rate(build_webhook_context_duration_seconds_count[5m]))
```

### Average Get Assistant Duration (seconds)

Time spent calling the Orchestra `/admin/assistant` endpoint to resolve
assistant details from a phone number, email, or ID.

```promql
sum(rate(orchestra_get_assistant_duration_seconds_sum{status="success"}[5m]))
/
sum(rate(orchestra_get_assistant_duration_seconds_count{status="success"}[5m]))
```

By lookup type (email, phone, id):

```promql
sum by (lookup_type) (rate(orchestra_get_assistant_duration_seconds_sum{status="success"}[5m]))
/
sum by (lookup_type) (rate(orchestra_get_assistant_duration_seconds_count{status="success"}[5m]))
```

### Average Manager Init Duration (seconds)

Total duration of `init_conv_manager()` — the full manager initialization
sequence when a container goes live.

```promql
sum(rate({__name__="workload.googleapis.com/unity_manager_init_seconds_sum", monitored_resource="k8s_container"}[5m]))
/
sum(rate({__name__="workload.googleapis.com/unity_manager_init_seconds_count", monitored_resource="k8s_container"}[5m]))
```

### Average Mark Job Running Duration (seconds)

Time spent marking an AssistantJob as running via Orchestra `/logs`.

```promql
sum(rate(mark_job_running_duration_seconds_sum{status="success"}[5m]))
/
sum(rate(mark_job_running_duration_seconds_count{status="success"}[5m]))
```

### Adapter Request Latency (seconds)

Per-route request latency for the adapters service. Useful for identifying
slow endpoints.

```promql
sum by (endpoint) (rate(http_request_duration_seconds_sum{service="adapters"}[5m]))
/
sum by (endpoint) (rate(http_request_duration_seconds_count{service="adapters"}[5m]))
```

### Comms Request Latency (seconds)

For the **comms** service:

```promql
sum by (endpoint) (rate(http_request_duration_seconds_sum{service="comms"}[5m]))
/
sum by (endpoint) (rate(http_request_duration_seconds_count{service="comms"}[5m]))
```

### Notes

- Replace `5m` with a different window size as needed, or create a dashboard
  variable named `interval` (type: Custom, values: `1m,2m,5m,10m,15m,30m,1h`)
  and use `[$interval]` in queries.
- Adapter metrics may not have the `workload.googleapis.com/` prefix when
  queried through the Prometheus data source — use the bare metric name.
- Unity metrics (pushed via Pushgateway to GKE) use the
  `workload.googleapis.com/` prefix with `monitored_resource="k8s_container"`.
- If `$__rate_interval` doesn't resolve (parse error on `$`), use a fixed
  duration or the dashboard variable approach above.

---

## Persistent Storage

The current deployment uses `emptyDir` for `/var/lib/grafana`, meaning
dashboards, users, and settings are lost if the pod restarts. For production
use, replace the `emptyDir` volume with a `PersistentVolumeClaim`:

```yaml
# In deployment.yaml, replace the grafana-storage volume:
volumes:
  - name: grafana-storage
    persistentVolumeClaim:
      claimName: grafana-staging-pvc

# And create a PVC:
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grafana-staging-pvc
  namespace: staging
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 10Gi
```

---

## Kubernetes Resources Summary

### Staging

| Resource | Name | Namespace |
|----------|------|-----------|
| Deployment | `grafana-staging` | `staging` |
| Service (ClusterIP) | `grafana-staging` | `staging` |
| Ingress | `grafana-staging` | `staging` |
| ConfigMap | `grafana-staging-datasource-provisioning` | `staging` |
| Secret | `grafana-staging-secrets` | `staging` |
| ManagedCertificate | `grafana-staging-cert` | `staging` |
| Secret (pre-existing) | `comm-sa-key` | `staging` |

### Production

| Resource | Name | Namespace |
|----------|------|-----------|
| Deployment | `grafana` | `production` |
| Service (ClusterIP) | `grafana` | `production` |
| Ingress | `grafana` | `production` |
| ConfigMap | `grafana-datasource-provisioning` | `production` |
| Secret | `grafana-secrets` | `production` |
| ManagedCertificate | `grafana-cert` | `production` |
| Secret (pre-existing) | `comm-sa-key` | `production` |

