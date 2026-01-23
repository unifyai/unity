#!/bin/bash

# Exit on any error
set -e

# Source orchestra URL configuration (set during Docker build based on branch)
if [ -f /app/.env.orchestra ]; then
    source /app/.env.orchestra
fi

# Global variables to track processes
REDIS_PID=""
MAIN_PID=""
AGENT_PID=""
CSB_PID=""
DESKTOP_PID=""

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

    if [ ! -z "$DESKTOP_PID" ]; then
        echo "Stopping desktop (PID: $DESKTOP_PID)..."
        kill -TERM $DESKTOP_PID 2>/dev/null || true
        wait $DESKTOP_PID 2>/dev/null || true
    else
        echo "Stopping desktop..."
        pkill -f "desktop.sh" 2>/dev/null || true
    fi

    if [ ! -z "$AGENT_PID" ]; then
        echo "Stopping agent-service (PID: $AGENT_PID)..."
        kill -TERM $AGENT_PID 2>/dev/null || true
        wait $AGENT_PID 2>/dev/null || true
    else
        echo "Stopping agent-service..."
        pkill -f "ts-node" 2>/dev/null || true
    fi

    if [ ! -z "$CSB_PID" ]; then
        echo "Stopping codesandbox-service (PID: $CSB_PID)..."
        kill -TERM $CSB_PID 2>/dev/null || true
        wait $CSB_PID 2>/dev/null || true
    else
        echo "Stopping codesandbox-service..."
        pkill -f "codesandbox-service" 2>/dev/null || true
    fi

    echo "Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

echo "Starting Redis server and convo manager..."

# Clear any existing Redis data to avoid format compatibility issues
echo "Clearing existing Redis data..."
rm -f /app/dump.rdb /tmp/dump.rdb /var/lib/redis/dump.rdb 2>/dev/null || true

# Start Redis in the background and capture its PID
echo "Starting Redis server..."
redis-server --save "" --appendonly no &
REDIS_PID=$!
echo "Redis started with PID: $REDIS_PID"


# Virtual desktop and devices
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

mkdir -p /run/dbus
dbus-daemon --system --fork
eval "$(dbus-launch)"
export DBUS_SESSION_BUS_ADDRESS

# Start virtual device
pipewire &
pipewire-pulse &
wireplumber &
sleep 2

# 1. For capturing Meet participant audio
pactl load-module module-null-sink sink_name=meet_sink
pactl load-module module-remap-source master=meet_sink.monitor source_name=meet_mic

# 2. For agent TTS (only goes to Meet, not to agent itself)
pactl load-module module-null-sink sink_name=agent_sink
pactl load-module module-remap-source master=agent_sink.monitor source_name=agent_mic

pactl set-default-source meet_mic
pactl set-default-sink agent_sink

# Launch virtual desktop
bash /app/desktop/desktop.sh &
DESKTOP_PID=$!

# Start agent-service (ts-node)
npx ts-node /app/agent-service/src/index.ts &
AGENT_PID=$!

# Start codesandbox-service (ts-node)
npx ts-node /app/codesandbox-service/src/index.ts &
CSB_PID=$!

# echo "Starting virtual desktop and Magnitude server..."
# bash desktop/startup.sh &
# DESKTOP_PID=$!


# Start the main application in parallel
echo "Starting convo manager..."
uv run unity/conversation_manager/main.py &
MAIN_PID=$!
echo "Main application started with PID: $MAIN_PID"

# Wait for main processes
wait $MAIN_PID
