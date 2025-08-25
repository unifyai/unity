#!/bin/bash
set -e

export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

# Start DBus for portals if present
mkdir -p /run/dbus
dbus-daemon --system --fork
eval "$(dbus-launch)"
export DBUS_SESSION_BUS_ADDRESS

# Launch virtual desktop
bash /app/desktop.sh &

# Start virtual device
bash /app/device.sh &

# Start agent-service (ts-node)
exec npx ts-node /app/agent-service/src/index.ts
