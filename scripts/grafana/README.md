# Grafana on GKE — Staging Deployment

**THIS IS A ONE-TIME DEPLOYMENT. DO NOT RE-DEPLOY.**

Deploys a Grafana instance to the `staging` namespace on the `unity` GKE cluster
(project `responsive-city-458413-a2`). The instance is pre-configured with a
Google Cloud Monitoring data source for viewing built-in metrics from Cloud Run,
GKE, and Pub/Sub.

**Live URL:** `https://grafana.staging.internal.saas.unify.ai`

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

## Files

| File | Description |
|------|-------------|
| `deploy.sh` | One-command deployment script (creates secrets, applies all manifests, waits for IP) |
| `deployment.yaml` | Grafana pod — image, env vars, volume mounts, probes, resource limits |
| `service.yaml` | ClusterIP Service exposing Grafana internally on port 80 |
| `ingress.yaml` | GKE Ingress with ManagedCertificate annotation for HTTPS |
| `managed-certificate.yaml` | GKE-native TLS certificate (auto-provisioned and auto-renewed by Google) |
| `datasource-configmap.yaml` | Auto-provisions the Google Cloud Monitoring data source on Grafana startup |

---

## Quick Start (deploy.sh)

The script handles everything except DNS and the data source key upload.

### Prerequisites

1. **kubectl pointed at the unity cluster:**
   ```bash
   gcloud container clusters get-credentials unity \
     --region us-central1 \
     --project responsive-city-458413-a2
   ```

2. **`comm-sa-key` secret exists in the `staging` namespace** (contains the GCP
   service account key used by Grafana to authenticate with Cloud Monitoring).

### Run

```bash
# Random admin password (printed once — save it)
./deploy.sh

# Or specify a password
./deploy.sh --password <your-password>
```

The script will:
1. Create the `grafana-staging-secrets` Kubernetes Secret (admin password)
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
   gcloud dns record-sets create grafana.staging.internal.saas.unify.ai \
     --zone=internal-saas \
     --type=A \
     --ttl=300 \
     --rrdatas=<EXTERNAL_IP> \
     --project=saas-368716
   ```

2. **Apply the ManagedCertificate for HTTPS** (not included in deploy.sh):
   ```bash
   kubectl apply -f managed-certificate.yaml
   ```
   Then re-apply the Ingress so it picks up the annotation:
   ```bash
   kubectl apply -f ingress.yaml
   ```
   The certificate takes 10-15 minutes to provision. Monitor with:
   ```bash
   kubectl get managedcertificate grafana-staging-cert -n staging --watch
   ```
   Once status shows `Active`, HTTPS is live.

3. **Configure the data source in Grafana:**
   - Log in at `https://grafana.staging.internal.saas.unify.ai`
   - Go to **Connections > Data Sources > Google Cloud Monitoring**
   - Upload the service account key (the same key from `comm-sa-key`)
   - Set the default project to `responsive-city-458413-a2`
   - Click **Save & Test**

---

## Manual Step-by-Step Deployment

If you prefer to apply manifests individually rather than using `deploy.sh`:

### 1. Point kubectl at the cluster

```bash
gcloud container clusters get-credentials unity \
  --region us-central1 \
  --project responsive-city-458413-a2
```

### 2. Create the admin password secret

```bash
kubectl create secret generic grafana-staging-secrets \
  --namespace=staging \
  --from-literal=admin-password="<YOUR_PASSWORD>" \
  --dry-run=client -o yaml | kubectl apply -f -
```

### 3. Apply the ConfigMap

```bash
kubectl apply -f datasource-configmap.yaml
```

This provisions the Google Cloud Monitoring data source automatically on Grafana
startup. The data source uses JWT authentication and expects the service account
key to be mounted at `/secrets/key.json` (handled by the Deployment).

### 4. Apply the Deployment

```bash
kubectl apply -f deployment.yaml
```

Key configuration:
- **Image:** `grafana/grafana:11.5.1`
- **Service account:** `comm-sa` (Kubernetes SA)
- **SA key mount:** `/secrets/key.json` from the `comm-sa-key` Secret
- **Resources:** 250m-500m CPU, 256Mi-512Mi memory
- **Storage:** `emptyDir` (dashboards/settings are lost on pod restart — see
  [Persistent Storage](#persistent-storage) below)

Wait for the pod:
```bash
kubectl rollout status deployment/grafana-staging -n staging --timeout=120s
```

### 5. Apply the Service

```bash
kubectl apply -f service.yaml
```

ClusterIP service on port 80, forwarding to the pod's port 3000.

### 6. Apply the Ingress

```bash
kubectl apply -f ingress.yaml
```

This triggers GKE to create an HTTP(S) Load Balancer. Wait for the external IP:
```bash
kubectl get ingress grafana-staging -n staging --watch
```

The IP typically appears in 2-5 minutes. Check backend health:
```bash
kubectl describe ingress grafana-staging -n staging
```

Look for `ingress.kubernetes.io/backends` — all backends should show `HEALTHY`
(this can take an additional 2-3 minutes after the IP is assigned).

### 7. Add the DNS record

```bash
gcloud dns record-sets create grafana.staging.internal.saas.unify.ai \
  --zone=internal-saas \
  --type=A \
  --ttl=300 \
  --rrdatas=<EXTERNAL_IP> \
  --project=saas-368716
```

Verify DNS resolution:
```bash
dig grafana.staging.internal.saas.unify.ai +short
```

### 8. Enable HTTPS with ManagedCertificate

```bash
kubectl apply -f managed-certificate.yaml
kubectl apply -f ingress.yaml   # picks up the annotation
```

Monitor certificate provisioning (10-15 minutes):
```bash
kubectl get managedcertificate grafana-staging-cert -n staging --watch
```

Once `Active`, HTTPS is live. Verify:
```bash
curl -s https://grafana.staging.internal.saas.unify.ai/api/health
```

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
kubectl describe ingress grafana-staging -n staging | grep backends
```

Wait until all backends show `HEALTHY` (can take 3-5 minutes after IP
assignment). If a backend stays `Unknown` or `UNHEALTHY`, check the pod:
```bash
kubectl logs -n staging -l app=grafana-staging
kubectl get pods -n staging -l app=grafana-staging
```

### ManagedCertificate stuck in Provisioning

Google needs to verify domain ownership via the DNS A record. Ensure:
1. The A record exists and resolves: `dig grafana.staging.internal.saas.unify.ai +short`
2. The resolved IP matches the Ingress IP: `kubectl get ingress grafana-staging -n staging`

If DNS is correct, just wait — provisioning can take up to 15 minutes. Check
status:
```bash
kubectl describe managedcertificate grafana-staging-cert -n staging
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
gcloud dns record-sets delete grafana.staging.internal.saas.unify.ai \
  --zone=internal-saas \
  --type=A \
  --project=saas-368716

# Create new record
gcloud dns record-sets create grafana.staging.internal.saas.unify.ai \
  --zone=internal-saas \
  --type=A \
  --ttl=300 \
  --rrdatas=<NEW_IP> \
  --project=saas-368716
```

---

## Tearing Down

```bash
# Delete all Grafana resources
kubectl delete ingress grafana-staging -n staging
kubectl delete svc grafana-staging -n staging
kubectl delete deployment grafana-staging -n staging
kubectl delete configmap grafana-staging-datasource-provisioning -n staging
kubectl delete secret grafana-staging-secrets -n staging
kubectl delete managedcertificate grafana-staging-cert -n staging

# Delete the DNS record
gcloud dns record-sets delete grafana.staging.internal.saas.unify.ai \
  --zone=internal-saas \
  --type=A \
  --project=saas-368716
```

Note: deleting the Ingress can take several minutes as GKE tears down the
associated HTTP(S) Load Balancer.

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

| Resource | Name | Namespace |
|----------|------|-----------|
| Deployment | `grafana-staging` | `staging` |
| Service (ClusterIP) | `grafana-staging` | `staging` |
| Ingress | `grafana-staging` | `staging` |
| ConfigMap | `grafana-staging-datasource-provisioning` | `staging` |
| Secret | `grafana-staging-secrets` | `staging` |
| ManagedCertificate | `grafana-staging-cert` | `staging` |
| Secret (pre-existing) | `comm-sa-key` | `staging` |
