#!/bin/bash
# Virtual desktop: TigerVNC (display + VNC server), Fluxbox WM, noVNC proxy

# Set VNC password (UNIFY_KEY shared with agent-service and developers)
mkdir -p /root/.vnc
echo "${UNIFY_KEY:-changeme}" | vncpasswd -f > /root/.vnc/passwd
chmod 600 /root/.vnc/passwd

# Provide minimal Fluxbox init to suppress missing-key warnings
mkdir -p /root/.fluxbox
printf "# Minimal Fluxbox init\n" > /root/.fluxbox/init

# Start TigerVNC (combined X display server + VNC server in one process)
Xtigervnc :99 -geometry 1920x1080 -depth 24 \
    -rfbport 5900 -rfbauth /root/.vnc/passwd \
    -AlwaysShared -desktop "Unity Desktop" &
sleep 2

# Window manager
fluxbox 2>/dev/null &

# Disable screen blanking and DPMS (prevents black screen over VNC)
xset s off
xset -dpms
xset s noblank

# Desktop portals
/usr/libexec/xdg-desktop-portal &
/usr/libexec/xdg-desktop-portal-gtk &

# Terminal
mkdir -p /tmp/unify/assistant/install
xterm -fa 'Monospace' -fs 10 &

# noVNC proxy (foreground — keeps this process alive for supervisord)
exec websockify --web=/opt/novnc 6080 localhost:5900
