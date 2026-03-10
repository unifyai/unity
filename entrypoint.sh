#!/bin/bash

# Exit on any error
set -e

# Record container start time (milliseconds since epoch).
# Used by the Python process to compute container spin-up duration (U1).
export CONTAINER_START_TIME_MS=$(date +%s%3N)

# Global variables to track processes
MAIN_PID=""
AGENT_PID=""

stop_agent_service() {
    if [ ! -z "$AGENT_PID" ]; then
        echo "Stopping agent-service (PID: $AGENT_PID)..."
        kill -TERM $AGENT_PID 2>/dev/null || true
        wait $AGENT_PID 2>/dev/null || true
    else
        echo "Stopping agent-service..."
        pkill -f "ts-node" 2>/dev/null || true
    fi
}

# Signal handler: forward SIGTERM to the Python process and wait for it to
# finish its own shutdown sequence (which now includes the GCS log upload).
on_signal() {
    echo "$(date '+%Y-%m-%d %H:%M:%S.%3N') - [ENTRYPOINT] Received shutdown signal, cleaning up..."

    if [ ! -z "$MAIN_PID" ]; then
        echo "Stopping main application (PID: $MAIN_PID)..."
        kill -TERM $MAIN_PID 2>/dev/null || true
        wait $MAIN_PID 2>/dev/null || true
    fi

    stop_agent_service
    echo "Cleanup complete"
    exit 0
}

trap on_signal SIGTERM SIGINT

# Create log directories for file-based traces in background
mkdir -p /var/log/unity /var/log/unify /var/log/unillm &

# Announce where logs will be preserved after shutdown
if [ ! -z "$UNITY_CONVERSATION_JOB_NAME" ]; then
    _GCS_BUCKET="${GCS_LOG_BUCKET:-unity-pod-logs}"
    # Derive namespace from job name suffix
    case "$UNITY_CONVERSATION_JOB_NAME" in
        *-staging)    _NS="staging" ;;
        *-production) _NS="production" ;;
        *)            _NS="unknown" ;;
    esac
    _GCS_PATH="gs://${_GCS_BUCKET}/${_NS}/${UNITY_CONVERSATION_JOB_NAME}/"
    echo "═══════════════════════════════════════════════════════════"
    echo "  Pod logs will be uploaded on shutdown to:"
    echo "  ${_GCS_PATH}"
    echo "  (auto-deleted after 7 days)"
    echo "═══════════════════════════════════════════════════════════"
fi

# Seed the emptyDir-backed /tmp with pre-downloaded HuggingFace models in background.
# The Dockerfile bakes models into /opt/hf-cache (user-agnostic); at runtime
# HF_HOME=/tmp/huggingface redirects lookups to the writable emptyDir volume.
if [ -d /opt/hf-cache ] && [ ! -d /tmp/huggingface ]; then
    cp -r /opt/hf-cache /tmp/huggingface &
fi

# Start agent-service on port 3000 (for web automation via Magnitude)
echo "⬥ Starting agent-service..."
# Use pre-compiled JavaScript if available, otherwise fallback to ts-node
if [ -f "/app/agent-service/dist/index.js" ]; then
    cd /app/agent-service && node dist/index.js &
else
    cd /app/agent-service && npx ts-node src/index.ts &
fi
AGENT_PID=$!
cd /app
echo "⬥ Agent-service started with PID: $AGENT_PID"

# Start the main application
echo "⬥ Starting convo manager..."
python3 unity/conversation_manager/main.py &
MAIN_PID=$!
echo "⬥ Main application started with PID: $MAIN_PID"

# Wait for the main process to exit (inactivity timeout or other self-initiated shutdown).
# The SIGTERM path is handled by the on_signal trap above and never reaches here.
wait $MAIN_PID || true
stop_agent_service
