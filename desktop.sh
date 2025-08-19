# Set up for virtual audio
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

mkdir -p /run/dbus
dbus-daemon --system --fork
eval "$(dbus-launch)"
export DBUS_SESSION_BUS_ADDRESS


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


# Create the virtual sink/mic
pipewire &
pipewire-pulse &
wireplumber &
sleep 2

# 1. For capturing Meet participant audio
pactl load-module module-null-sink sink_name=meet_sink
pactl load-module module-remap-source master=meet_sink.monitor source_name=meet_mic

# 2. For agent TTS (only goes to Meet, not to agent itself)
pactl load-module module-null-sink sink_name=agent_sink
pactl load-module module-remap-source master=agent_sink.monitor source_name=agent_mic

pactl set-default-source meet_mic
pactl set-default-sink agent_sink

websockify --web=/opt/novnc 6080 localhost:5900
