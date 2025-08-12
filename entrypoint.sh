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

    if [ ! -z "$BROWSER_PID" ]; then
        echo "Stopping browser (PID: $BROWSER_PID)..."
        kill -TERM $BROWSER_PID 2>/dev/null || true
        wait $BROWSER_PID 2>/dev/null || true
    fi
    # Stop the BrowserAgent (Node) service
    if [ ! -z "$NODE_PID" ]; then
        echo "Stopping BrowserAgent service (PID: $NODE_PID)..."
        kill -TERM $NODE_PID 2>/dev/null || true
        wait $NODE_PID 2>/dev/null || true
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

# Start the BrowserAgent (Node) service via ts-node
echo "Starting BrowserAgent service via ts-node..."
cd /app/agent-service
npm install
npx ts-node src/index.ts &
NODE_PID=$!
cd /app


# Set up for virtual audio
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

mkdir -p /run/dbus
dbus-daemon --system --fork
eval "$(dbus-launch)"
export DBUS_SESSION_BUS_ADDRESS


# Set up for live viewing browser
Xvfb :99 -screen 0 1920x1080x16 &
sleep 2

# Provide minimal Fluxbox init to suppress missing-key warnings
mkdir -p ~/.fluxbox
printf "# Minimal Fluxbox init\n" > ~/.fluxbox/init

# Start window manager, VNC server and noVNC proxy
fluxbox 2>/dev/null &
x11vnc -display :99 -nopw -forever -shared -bg -rfbport 5900 \
       -rfbportv6 0 -noxdamage -nowf -noxfixes -nodpms
websockify --web=/opt/novnc 6080 localhost:5900 &
BROWSER_PID=$!

/usr/libexec/xdg-desktop-portal &
/usr/libexec/xdg-desktop-portal-gtk &


# Create the virtual sink/mic
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

# Set up for remote browser/os
# bash device.sh &
# BROWSER_PID=$!


# Start the main application in parallel
echo "Starting convo manager..."
python start.py &
MAIN_PID=$!
echo "Main application started with PID: $MAIN_PID"

# Wait for main processes
wait $MAIN_PID
