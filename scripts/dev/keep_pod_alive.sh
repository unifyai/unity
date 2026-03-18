#!/usr/bin/env bash
#
# Periodically publish keepalive pings to a Unity pod's Pub/Sub topic,
# preventing the inactivity timeout from shutting down the container.
#
# Uses the same Ping(kind="keepalive") mechanism that idle containers use
# internally, routed through GCP Pub/Sub → comms_manager → event broker.
#
# Usage:
#   ./scripts/dev/keep_pod_alive.sh                                 # auto-detect latest staging pod
#   ./scripts/dev/keep_pod_alive.sh <assistant_id>                  # explicit assistant, staging
#   ./scripts/dev/keep_pod_alive.sh --env production                # auto-detect latest production pod
#   ./scripts/dev/keep_pod_alive.sh --env preview                   # auto-detect latest preview pod
#   ./scripts/dev/keep_pod_alive.sh <assistant_id> --env production # explicit assistant, production
#   ./scripts/dev/keep_pod_alive.sh <assistant_id> --interval 60    # custom interval
#
# Requires:
#   - gcloud CLI authenticated with access to the responsive-city-458413-a2 project
#   - When auto-detecting: UNIFY_KEY and SHARED_UNIFY_KEY env vars (or in .env)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"

GCP_PROJECT="responsive-city-458413-a2"
DEFAULT_INTERVAL=30

usage() {
    echo "Usage: $0 [assistant_id] [--env ENV] [--interval SECONDS]"
    echo
    echo "Keep a Unity pod alive by sending periodic keepalive pings via Pub/Sub."
    echo
    echo "Arguments:"
    echo "  assistant_id          The assistant's numeric ID (optional; auto-detected from"
    echo "                        your latest running pod when omitted)"
    echo
    echo "Options:"
    echo "  --env ENV             Target environment: production, staging, or preview (default: staging)"
    echo "  --interval SECONDS    Ping interval in seconds (default: ${DEFAULT_INTERVAL})"
    echo "  -h, --help            Show this help message"
    exit 1
}

# --- Parse arguments ---

ASSISTANT_ID=""
DEPLOY_ENV=staging
INTERVAL=$DEFAULT_INTERVAL

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)
            DEPLOY_ENV="$2"
            case "$DEPLOY_ENV" in
                production|staging|preview) ;;
                *) echo "Invalid --env value: $DEPLOY_ENV" >&2; usage ;;
            esac
            shift 2
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

# --- Auto-detect assistant_id if not provided ---

if [[ -z "$ASSISTANT_ID" ]]; then
    ASSISTANT_ID=$("$PYTHON" "${SCRIPT_DIR}/job_utils.py" assistant-id --env "$DEPLOY_ENV")
fi

# --- Build topic name ---

TOPIC="unity-${ASSISTANT_ID}"
if [[ "$DEPLOY_ENV" != "production" ]]; then
    TOPIC="${TOPIC}-${DEPLOY_ENV}"
fi

PING_MESSAGE='{"thread":"ping","event":{}}'

echo "Keeping pod alive:"
echo "  Project:      ${GCP_PROJECT}"
echo "  Topic:        ${TOPIC}"
echo "  Environment:  ${DEPLOY_ENV}"
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
