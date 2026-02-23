#!/bin/bash

# Exit on any error
set -e

# Record container start time (milliseconds since epoch).
# Used by the Python process to compute container spin-up duration (U1).
export CONTAINER_START_TIME_MS=$(date +%s%3N)

# Global variables to track processes
MAIN_PID=""
AGENT_PID=""

# Function to handle graceful shutdown
cleanup() {
    echo "$(date '+%Y-%m-%d %H:%M:%S.%3N') - [ENTRYPOINT] Received shutdown signal, cleaning up..."

    # Stop the main application
    if [ ! -z "$MAIN_PID" ]; then
        echo "Stopping main application (PID: $MAIN_PID)..."
        kill -TERM $MAIN_PID 2>/dev/null || true
        wait $MAIN_PID 2>/dev/null || true
    fi

    if [ ! -z "$AGENT_PID" ]; then
        echo "Stopping agent-service (PID: $AGENT_PID)..."
        kill -TERM $AGENT_PID 2>/dev/null || true
        wait $AGENT_PID 2>/dev/null || true
    else
        echo "Stopping agent-service..."
        pkill -f "ts-node" 2>/dev/null || true
    fi

    echo "Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Create log directories for file-based traces
mkdir -p /var/log/unity /var/log/unify /var/log/unillm

# Start agent-service on port 3000 (for web automation via Magnitude)
echo "⬥ Starting agent-service..."
cd /app/agent-service && npx ts-node src/index.ts &
AGENT_PID=$!
cd /app
echo "⬥ Agent-service started with PID: $AGENT_PID"

# Start the main application
echo "⬥ Starting convo manager..."
python3 unity/conversation_manager/main.py &
MAIN_PID=$!
echo "⬥ Main application started with PID: $MAIN_PID"

# Wait for main process
wait $MAIN_PID
