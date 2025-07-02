#!/bin/bash

# Exit on any error
set -e

# Function to handle graceful shutdown
cleanup() {
    echo "Received shutdown signal, cleaning up..."
    
    # Stop the main application (uvicorn)
    if [ ! -z "$UVICORN_PID" ]; then
        echo "Stopping uvicorn (PID: $UVICORN_PID)..."
        kill -TERM $UVICORN_PID 2>/dev/null || true
        wait $UVICORN_PID 2>/dev/null || true
    fi
    
    # Stop Redis
    echo "Stopping Redis..."
    redis-cli shutdown 2>/dev/null || true
    
    echo "Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

echo "Starting Redis server..."

# Clear any existing Redis data to avoid format compatibility issues
echo "Clearing existing Redis data..."
rm -f /app/dump.rdb /tmp/dump.rdb /var/lib/redis/dump.rdb 2>/dev/null || true

# Start Redis in the background with specific configuration
redis-server --daemonize yes --save "" --appendonly no

# Wait a moment for Redis to start
echo "Waiting for Redis to start..."
sleep 2

# Check if Redis is running with retries
MAX_RETRIES=5
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if redis-cli ping > /dev/null 2>&1; then
        echo "Redis is running successfully"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "Redis not ready yet, retrying... ($RETRY_COUNT/$MAX_RETRIES)"
        sleep 2
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Error: Redis failed to start after $MAX_RETRIES attempts"
    exit 1
fi

echo "Starting wrapper app..."

# Start the main application in the background
python wrapper_app.py &
WRAPPER_APP_PID=$!

# Wait for the wrapper app process
wait $WRAPPER_APP_PID
