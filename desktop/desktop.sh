#!/bin/bash
# Virtual desktop: TigerVNC (display + VNC server), XFCE4 DE, noVNC proxy

# Set VNC password (UNIFY_KEY shared with agent-service and developers)
mkdir -p /root/.vnc
echo "${UNIFY_KEY:-changeme}" | vncpasswd -f > /root/.vnc/passwd
chmod 600 /root/.vnc/passwd

# Start TigerVNC (combined X display server + VNC server in one process)
Xtigervnc :99 -geometry 1920x1080 -depth 24 \
    -rfbport 5900 -rfbauth /root/.vnc/passwd \
    -AlwaysShared -desktop "Unity Desktop" &
sleep 2

# XFCE desktop session
startxfce4 &

# Disable screen blanking and DPMS (prevents black screen over VNC)
xset s off
xset -dpms
xset s noblank

# Desktop portals
/usr/libexec/xdg-desktop-portal &
/usr/libexec/xdg-desktop-portal-gtk &

mkdir -p /tmp/unify/assistant/install

# noVNC proxy (foreground — keeps this process alive for supervisord)
exec websockify --web=/opt/novnc 6080 localhost:5900
