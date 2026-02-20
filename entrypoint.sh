#!/bin/bash

# Exit on any error
set -e

# Record container start time (milliseconds since epoch).
# Used by the Python process to compute container spin-up duration (U1).
export CONTAINER_START_TIME_MS=$(date +%s%3N)

# Global variables to track processes
REDIS_PID=""
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

    # Stop Redis
    if [ ! -z "$REDIS_PID" ]; then
        echo "Stopping Redis (PID: $REDIS_PID)..."
        kill -TERM $REDIS_PID 2>/dev/null || true
        wait $REDIS_PID 2>/dev/null || true
    else
        echo "Stopping Redis..."
        redis-cli shutdown 2>/dev/null || true
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

echo "Starting Redis server and services..."

# Clear any existing Redis data to avoid format compatibility issues
echo "Clearing existing Redis data..."
rm -f /app/dump.rdb /tmp/dump.rdb /var/lib/redis/dump.rdb 2>/dev/null || true

# Start Redis in the background and capture its PID
echo "Starting Redis server..."
redis-server --save "" --appendonly no &
REDIS_PID=$!
echo "Redis started with PID: $REDIS_PID"

# Start agent-service on port 3000 (for web automation via Magnitude)
echo "Starting agent-service..."
npx ts-node /app/agent-service/src/index.ts &
AGENT_PID=$!
echo "Agent-service started with PID: $AGENT_PID"

# Start the main application
echo "Starting convo manager..."
python3 unity/conversation_manager/main.py &
MAIN_PID=$!
echo "Main application started with PID: $MAIN_PID"

# Wait for main process
wait $MAIN_PID
