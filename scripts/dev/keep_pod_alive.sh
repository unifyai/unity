#!/usr/bin/env bash
#
# Periodically publish keepalive pings to a Unity pod's Pub/Sub topic,
# preventing the inactivity timeout from shutting down the container.
#
# Uses the same Ping(kind="keepalive") mechanism that idle containers use
# internally, routed through GCP Pub/Sub → comms_manager → event broker.
#
# Usage:
#   ./scripts/dev/keep_pod_alive.sh <assistant_id>                  # staging (default)
#   ./scripts/dev/keep_pod_alive.sh <assistant_id> --production     # production
#   ./scripts/dev/keep_pod_alive.sh <assistant_id> --interval 60    # custom interval
#
# Requires:
#   - gcloud CLI authenticated with access to the responsive-city-458413-a2 project

set -euo pipefail

GCP_PROJECT="responsive-city-458413-a2"
DEFAULT_INTERVAL=30

usage() {
    echo "Usage: $0 <assistant_id> [--production] [--interval SECONDS]"
    echo
    echo "Keep a Unity pod alive by sending periodic keepalive pings via Pub/Sub."
    echo
    echo "Arguments:"
    echo "  assistant_id          The assistant's numeric ID"
    echo
    echo "Options:"
    echo "  --production          Target the production environment (default: staging)"
    echo "  --interval SECONDS    Ping interval in seconds (default: ${DEFAULT_INTERVAL})"
    echo "  -h, --help            Show this help message"
    exit 1
}

# --- Parse arguments ---

ASSISTANT_ID=""
STAGING=true
INTERVAL=$DEFAULT_INTERVAL

while [[ $# -gt 0 ]]; do
    case "$1" in
        --production)
            STAGING=false
            shift
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            ;;
        *)
            if [[ -z "$ASSISTANT_ID" ]]; then
                ASSISTANT_ID="$1"
            else
                echo "Unexpected argument: $1" >&2
                usage
            fi
            shift
            ;;
    esac
done

if [[ -z "$ASSISTANT_ID" ]]; then
    echo "Error: assistant_id is required" >&2
    usage
fi

# --- Build topic name ---

TOPIC="unity-${ASSISTANT_ID}"
if [[ "$STAGING" == "true" ]]; then
    TOPIC="${TOPIC}-staging"
fi

PING_MESSAGE='{"thread":"ping","event":{}}'

echo "Keeping pod alive:"
echo "  Project:      ${GCP_PROJECT}"
echo "  Topic:        ${TOPIC}"
echo "  Environment:  $(if [[ "$STAGING" == "true" ]]; then echo staging; else echo production; fi)"
echo "  Interval:     ${INTERVAL}s"
echo
echo "Press Ctrl+C to stop."
echo

# --- Ping loop ---

while true; do
    if gcloud pubsub topics publish "$TOPIC" \
        --project="$GCP_PROJECT" \
        --message="$PING_MESSAGE" \
        --quiet 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] ping sent"
    else
        echo "[$(date '+%H:%M:%S')] ping failed" >&2
    fi
    sleep "$INTERVAL"
done
