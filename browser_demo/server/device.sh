#!/bin/bash
set -e

apt-get update && apt-get install -y xdg-desktop-portal xdg-desktop-portal-gtk

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

# Set up for virtual camera


# Set up for live viewing browser
Xvfb :99 -screen 0 1920x1080x16 &
sleep 2

# Start window manager, VNC server and noVNC proxy
fluxbox &
x11vnc -display :99 -nopw -forever -shared -bg -rfbport 5900
websockify --web=/opt/novnc 6080 localhost:5900 &

# Start interactive browser agent
python browseruse.py
