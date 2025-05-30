#!/bin/bash
set -e

export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

DBUS_ADDRESS=$(dbus-daemon --session --print-address --fork)
export DBUS_SESSION_BUS_ADDRESS=$DBUS_ADDRESS

pipewire &
pipewire-pulse &
wireplumber &
sleep 2

# Create the virtual sink/mic
pactl load-module module-null-sink sink_name=virtual_sink sink_properties=device.description="Virtual_Sink"
pactl load-module module-remap-source master=virtual_sink.monitor source_name=virtual_mic
pactl set-default-source virtual_mic

# Start X11 virtual display
Xvfb :99 -screen 0 1920x1080x16 &

# Wait for Xvfb to initialize
sleep 2

# Start window manager
fluxbox &

# Start VNC server
x11vnc -display :99 -nopw -forever -shared -bg -rfbport 5900

# Start noVNC websockify proxy (this will block and keep the container running)
websockify --web=/opt/novnc 6080 localhost:5900 &

python browseruse.py
