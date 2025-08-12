# Set up for live viewing browser
Xvfb :99 -screen 0 1920x1080x16 &
sleep 2

# Provide minimal Fluxbox init to suppress missing-key warnings
mkdir -p ~/.fluxbox
printf "# Minimal Fluxbox init\n" > ~/.fluxbox/init

# Start window manager, VNC server and noVNC proxy
fluxbox 2>/dev/null &
x11vnc -display :99 -nopw -forever -shared -bg -rfbport 5900 \
       -rfbportv6 0 -noxdamage -nowf -noxfixes -nodpms
websockify --web=/opt/novnc 6080 localhost:5900

## Set up for virtual camera (host.sh)
# sudo apt-get update && sudo apt-get install -y v4l2loopback-dkms v4l2loopback-utils v4l-utils linux-modules-extra-$(uname -r)

# sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="TestCam" exclusive_caps=1

# sudo usermod -aG video $USER
# newgrp video
