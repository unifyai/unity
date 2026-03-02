#!/bin/bash
set -e

# Deploy Grafana to the staging namespace on the unity GKE cluster.
#
# Prerequisites:
#   - kubectl configured to point at the unity cluster in responsive-city-458413-a2
#     (run: gcloud container clusters get-credentials unity --region us-central1 --project responsive-city-458413-a2)
#   - comm-sa-key secret already exists in staging namespace
#
# Usage:
#   ./deploy.sh                        # uses a random admin password
#   ./deploy.sh --password <password>  # uses the given admin password

NAMESPACE="staging"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ADMIN_PASSWORD=""
DNS_PROJECT="saas-368716"
DOMAIN="grafana.staging.internal.saas.unify.ai"

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --password)
      ADMIN_PASSWORD="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Generate a random password if none provided
if [ -z "$ADMIN_PASSWORD" ]; then
  ADMIN_PASSWORD=$(openssl rand -base64 16)
  echo "Generated admin password: $ADMIN_PASSWORD"
  echo "(save this — you won't see it again)"
fi

echo ""
echo "=== Deploying Grafana to namespace: $NAMESPACE ==="
echo ""

# 1. Create the admin password secret (replace if exists)
echo "[1/5] Creating grafana-staging-secrets..."
kubectl create secret generic grafana-staging-secrets \
  --namespace="$NAMESPACE" \
  --from-literal=admin-password="$ADMIN_PASSWORD" \
  --dry-run=client -o yaml | kubectl apply -f -

# 2. Apply the datasource provisioning ConfigMap
echo "[2/5] Applying datasource ConfigMap..."
kubectl apply -f "$SCRIPT_DIR/datasource-configmap.yaml"

# 3. Apply the Deployment
echo "[3/5] Applying Deployment..."
kubectl apply -f "$SCRIPT_DIR/deployment.yaml"

# 4. Apply the Service
echo "[4/5] Applying Service..."
kubectl apply -f "$SCRIPT_DIR/service.yaml"

# 5. Apply the Ingress
echo "[5/5] Applying Ingress..."
kubectl apply -f "$SCRIPT_DIR/ingress.yaml"

echo ""
echo "=== Deployment complete ==="
echo ""
echo "Waiting for pod to be ready..."
kubectl rollout status deployment/grafana-staging -n "$NAMESPACE" --timeout=120s

echo ""
echo "=== Waiting for Ingress external IP (this can take 2-5 minutes) ==="
echo ""

EXTERNAL_IP=""
ATTEMPTS=0
MAX_ATTEMPTS=30  # 5 minutes at 10s intervals

while [ -z "$EXTERNAL_IP" ] && [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
  EXTERNAL_IP=$(kubectl get ingress grafana-staging -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
  if [ -z "$EXTERNAL_IP" ]; then
    ATTEMPTS=$((ATTEMPTS + 1))
    echo "  Waiting for IP... (attempt $ATTEMPTS/$MAX_ATTEMPTS)"
    sleep 10
  fi
done

if [ -z "$EXTERNAL_IP" ]; then
  echo ""
  echo "WARNING: Ingress IP not assigned yet. Check manually with:"
  echo "  kubectl get ingress -n $NAMESPACE grafana-staging --watch"
  echo ""
  echo "Once you have the IP, add the DNS record:"
  echo "  gcloud dns record-sets create $DOMAIN \\"
  echo "    --zone=internal-saas \\"
  echo "    --type=A \\"
  echo "    --ttl=300 \\"
  echo "    --rrdatas=<EXTERNAL_IP> \\"
  echo "    --project=$DNS_PROJECT"
else
  echo ""
  echo "=== Ingress IP assigned: $EXTERNAL_IP ==="
  echo ""
  echo "Now add a DNS A record in the $DNS_PROJECT project Cloud DNS:"
  echo ""
  echo "  gcloud dns record-sets create $DOMAIN \\"
  echo "    --zone=internal-saas \\"
  echo "    --type=A \\"
  echo "    --ttl=300 \\"
  echo "    --rrdatas=$EXTERNAL_IP \\"
  echo "    --project=$DNS_PROJECT"
fi

echo ""
echo "Access Grafana at: http://$DOMAIN"
echo "  Username: admin"
echo "  Password: $ADMIN_PASSWORD"
echo ""
echo "Once logged in, go to Connections > Data Sources > Google Cloud Monitoring"
echo "and upload the service account key to complete the data source setup."
