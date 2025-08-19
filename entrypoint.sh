#!/bin/bash

# Exit on any error
set -e

# Global variables to track processes
REDIS_PID=""
MAIN_PID=""

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
bash desktop.sh &
DESKTOP_PID=$!
DISPLAY=:99 xterm -fa 'Monospace' -fs 10 &

# Start the main application in parallel
echo "Starting convo manager..."
python start.py &
MAIN_PID=$!
echo "Main application started with PID: $MAIN_PID"

# Wait for main processes
wait $MAIN_PID
