# Set up virtual display only (no VNC / noVNC)
Xvfb :99 -screen 0 1920x1080x16 &
sleep 2

# Provide minimal Fluxbox init to suppress missing-key warnings
mkdir -p ~/.fluxbox
printf "# Minimal Fluxbox init\n" > ~/.fluxbox/init

# Start window manager on the virtual display
DISPLAY=:99 fluxbox 2>/dev/null &
DISPLAY=:99 xsetroot -cursor_name left_ptr

/usr/libexec/xdg-desktop-portal &
/usr/libexec/xdg-desktop-portal-gtk &

mkdir -p /tmp/unify/assistant/install
apt-get update
DISPLAY=:99 xterm -fa 'Monospace' -fs 10
