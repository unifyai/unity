#!/bin/bash
set -e

export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

# Start DBus (required by portals and audio)
mkdir -p /run/dbus
dbus-daemon --system --fork
eval "$(dbus-launch)"
export DBUS_SESSION_BUS_ADDRESS

# Refresh apt cache for runtime package installs by the agent
apt-get update

exec /usr/bin/supervisord -n -c /app/desktop/supervisord.conf
