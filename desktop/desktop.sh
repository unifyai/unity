# Set up for live viewing browser
Xvfb :99 -screen 0 1920x1080x16 &
sleep 2

# Provide minimal Fluxbox init to suppress missing-key warnings
mkdir -p ~/.fluxbox
printf "# Minimal Fluxbox init\n" > ~/.fluxbox/init

# Start window manager, VNC server and noVNC proxy
fluxbox 2>/dev/null &
DISPLAY=:99 xsetroot -cursor_name left_ptr
x11vnc -display :99 -nopw -forever -shared -bg -rfbport 5900 \
       -rfbportv6 0 -noxdamage -nowf -nocursorshape -cursor arrow -nodpms

/usr/libexec/xdg-desktop-portal &
/usr/libexec/xdg-desktop-portal-gtk &

mkdir -p /tmp/unify/assistant/install
apt-get update
DISPLAY=:99 xterm -fa 'Monospace' -fs 10 &
websockify --web=/opt/novnc 6080 localhost:5900
