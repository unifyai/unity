#!/usr/bin/env bash
#
# One-time provisioning of the discord-alerts Pub/Sub topic and push subscription.
# Production only — staging/preview do not send Discord alerts.
#
# Prerequisites:
#   - gcloud CLI authenticated with a project-owner or pubsub.admin account
#   - The adapters Cloud Run service must already be deployed
#
# Usage:
#   ./setup.sh

set -euo pipefail

PROJECT_ID="responsive-city-458413-a2"
REGION="us-central1"
TOPIC="discord-alerts"
SUBSCRIPTION="discord-alerts-push-sub"
ADAPTERS_SERVICE="unity-adapters"

# Service account used by Pub/Sub to invoke Cloud Run
PUSH_SA="cloud-run-pubsub-invoker@${PROJECT_ID}.iam.gserviceaccount.com"

# Resolve adapters URL
ADAPTERS_URL=$(gcloud run services describe "$ADAPTERS_SERVICE" \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --format='value(status.url)' 2>/dev/null || true)

if [ -z "$ADAPTERS_URL" ]; then
    echo "WARNING: Could not resolve URL for ${ADAPTERS_SERVICE}. Using convention."
    ADAPTERS_URL="https://${ADAPTERS_SERVICE}-ky4ja5fxna-uc.a.run.app"
fi

PUSH_ENDPOINT="${ADAPTERS_URL}/discord/alert"

echo "=== Provisioning Discord Alerts (production) ==="
echo "  Topic:         ${TOPIC}"
echo "  Subscription:  ${SUBSCRIPTION}"
echo "  Push endpoint: ${PUSH_ENDPOINT}"
echo ""

# Create topic (idempotent)
if gcloud pubsub topics describe "$TOPIC" --project "$PROJECT_ID" &>/dev/null; then
    echo "  Topic ${TOPIC} already exists, skipping."
else
    gcloud pubsub topics create "$TOPIC" --project "$PROJECT_ID"
    echo "  Created topic ${TOPIC}."
fi

# Create push subscription (idempotent)
if gcloud pubsub subscriptions describe "$SUBSCRIPTION" --project "$PROJECT_ID" &>/dev/null; then
    echo "  Subscription ${SUBSCRIPTION} already exists, updating push config."
    gcloud pubsub subscriptions update "$SUBSCRIPTION" \
        --project "$PROJECT_ID" \
        --push-endpoint="$PUSH_ENDPOINT" \
        --push-auth-service-account="$PUSH_SA"
else
    gcloud pubsub subscriptions create "$SUBSCRIPTION" \
        --project "$PROJECT_ID" \
        --topic="$TOPIC" \
        --push-endpoint="$PUSH_ENDPOINT" \
        --push-auth-service-account="$PUSH_SA" \
        --ack-deadline=30 \
        --min-retry-delay=10s \
        --max-retry-delay=600s
    echo "  Created subscription ${SUBSCRIPTION}."
fi

echo ""
echo "Done. Verify with:"
echo "  gcloud pubsub topics list --project ${PROJECT_ID} | grep discord-alerts"
echo "  gcloud pubsub subscriptions list --project ${PROJECT_ID} | grep discord-alerts"
