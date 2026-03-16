#!/bin/bash
set -e

# Deploy the job-watcher to the staging namespace on the unity GKE cluster.
#
# Prerequisites:
#   - kubectl configured to point at the unity cluster in responsive-city-458413-a2
#     (run: gcloud container clusters get-credentials unity --region us-central1 --project responsive-city-458413-a2)
#   - unity-config configmap and unity-secrets secret exist in staging namespace
#   - Docker image already pushed (use --build to build & push first)
#   - GITHUB_TOKEN env var set (for cloning the private unify repo during build)
#
# Usage:
#   ./deploy.sh            # apply manifests only (image must already exist)
#   ./deploy.sh --build    # build, push, then apply

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PARENT_DIR/../.." && pwd)"
IMAGE="us-central1-docker.pkg.dev/responsive-city-458413-a2/unity/job-watcher:latest"

BUILD=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --build)
      BUILD=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ "$BUILD" = true ]; then
  if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_TOKEN env var is required for --build (used to clone private unify repo)"
    exit 1
  fi
  echo "=== Building Docker image ==="
  docker build -t "$IMAGE" -f "$PARENT_DIR/Dockerfile" \
    --build-arg GITHUB_TOKEN="$GITHUB_TOKEN" \
    --build-arg BRANCH=staging \
    "$REPO_ROOT"
  echo ""
  echo "=== Pushing to Artifact Registry ==="
  docker push "$IMAGE"
  echo ""
fi

echo "=== Deploying job-watcher to staging ==="
kubectl apply -f "$SCRIPT_DIR/deployment.yaml"

echo ""
echo "Waiting for pod to be ready..."
kubectl rollout status deployment/job-watcher -n staging --timeout=120s

echo ""
echo "=== Deployment complete ==="
echo ""
echo "Check logs with:"
echo "  kubectl logs -n staging -l app=job-watcher -f"
