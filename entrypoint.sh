#!/bin/bash

# Exit on any error
set -e

# Global variables to track processes
REDIS_PID=""
MAIN_PID=""

# Function to handle graceful shutdown
cleanup() {
    echo "Received shutdown signal, cleaning up..."

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

    if [ ! -z "$BROWSER_PID" ]; then
        echo "Stopping browser (PID: $BROWSER_PID)..."
        kill -TERM $BROWSER_PID 2>/dev/null || true
        wait $BROWSER_PID 2>/dev/null || true
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






xdg-desktop-portal &
xdg-desktop-portal-gtk &  # or -gtk, depending on your compositor


# Set up for virtual audio
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

mkdir -p /run/dbus
dbus-daemon --system --fork
eval "$(dbus-launch)"
export DBUS_SESSION_BUS_ADDRESS

pipewire &
pipewire-pulse &
wireplumber &
sleep 2

# Create the virtual sink/mic
pactl load-module module-null-sink sink_name=virtual_sink sink_properties=device.description="Virtual_Sink"
pactl load-module module-remap-source master=virtual_sink.monitor source_name=virtual_mic
pactl set-default-source virtual_mic

bash device.sh &
BROWSER_PID=$!



# Start the main application in parallel
echo "Starting convo manager..."
python start.py &
MAIN_PID=$!
echo "Main application started with PID: $MAIN_PID"

# Wait for main processes
wait $MAIN_PID
