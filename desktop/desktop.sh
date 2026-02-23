# Set up for live viewing the desktop
Xvfb :99 -screen 0 1920x1080x16 &
sleep 2

# Provide minimal Fluxbox init to suppress missing-key warnings
mkdir -p ~/.fluxbox
printf "# Minimal Fluxbox init\n" > ~/.fluxbox/init

# Start window manager, VNC server and noVNC proxy
fluxbox 2>/dev/null &
# Set VNC password to UNIFY_KEY so the agent-service and developers can
# authenticate with the same credential via ?password= in the noVNC URL.
mkdir -p /root/.vnc
x11vnc -storepasswd "${UNIFY_KEY:-changeme}" /root/.vnc/passwd
chmod 600 /root/.vnc/passwd || true

# Start x11vnc with rfbauth
x11vnc -display :99 -rfbauth /root/.vnc/passwd -forever -shared -bg -rfbport 5900 \
       -rfbportv6 0 -noxdamage -nowf -nodpms

/usr/libexec/xdg-desktop-portal &
/usr/libexec/xdg-desktop-portal-gtk &

mkdir -p /tmp/unify/assistant/install
apt-get update
DISPLAY=:99 xterm -fa 'Monospace' -fs 10 &
websockify --web=/opt/novnc 6080 localhost:5900
