#!/bin/bash
set -e

export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

# Graceful shutdown
DESKTOP_PID=""
DEVICE_PID=""
AGENT_PID=""

cleanup() {
  echo "[startup] Caught termination signal, shutting down..."
  if [ -n "$AGENT_PID" ]; then
    kill -TERM "$AGENT_PID" 2>/dev/null || true
    wait "$AGENT_PID" 2>/dev/null || true
  fi
  if [ -n "$DEVICE_PID" ]; then
    kill -TERM "$DEVICE_PID" 2>/dev/null || true
    wait "$DEVICE_PID" 2>/dev/null || true
  fi
  if [ -n "$DESKTOP_PID" ]; then
    kill -TERM "$DESKTOP_PID" 2>/dev/null || true
    wait "$DESKTOP_PID" 2>/dev/null || true
  fi
  exit 0
}

trap cleanup SIGTERM SIGINT

# Start DBus for portals if present
mkdir -p /run/dbus
dbus-daemon --system --fork
eval "$(dbus-launch)"
export DBUS_SESSION_BUS_ADDRESS

# Launch virtual desktop
bash /app/desktop/desktop.sh &
DESKTOP_PID=$!

# Start virtual device
bash /app/desktop/device.sh &
DEVICE_PID=$!

# Start agent-service (ts-node)
npx ts-node /app/agent-service/src/index.ts --headless --desktop &
AGENT_PID=$!

# Wait for agent-service to exit, then cleanup
wait "$AGENT_PID"
cleanup
