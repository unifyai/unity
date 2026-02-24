#!/bin/bash
# Virtual display only (no VNC / noVNC web access)
# Uses TigerVNC as the X display server but does not expose VNC remotely.

# Dummy VNC password (required by Xtigervnc even when not used remotely)
mkdir -p /root/.vnc
echo "unused" | vncpasswd -f > /root/.vnc/passwd
chmod 600 /root/.vnc/passwd

# Start TigerVNC as display server (VNC port left at default but unexposed)
Xtigervnc :99 -geometry 1920x1080 -depth 24 \
    -rfbport 5900 -rfbauth /root/.vnc/passwd \
    -desktop "Unity Display" &
sleep 2

# Provide minimal Fluxbox init to suppress missing-key warnings
mkdir -p /root/.fluxbox
printf "# Minimal Fluxbox init\n" > /root/.fluxbox/init

# Start window manager on the virtual display
DISPLAY=:99 fluxbox 2>/dev/null &
DISPLAY=:99 xsetroot -cursor_name left_ptr

# Disable screen blanking
xset s off
xset -dpms
xset s noblank

/usr/libexec/xdg-desktop-portal &
/usr/libexec/xdg-desktop-portal-gtk &

mkdir -p /tmp/unify/assistant/install
apt-get update
DISPLAY=:99 xterm -fa 'Monospace' -fs 10
