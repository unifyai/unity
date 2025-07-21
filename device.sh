# Set up for live viewing browser
Xvfb :99 -screen 0 1920x1080x16 &
sleep 2

# Start window manager, VNC server and noVNC proxy
fluxbox &
x11vnc -display :99 -nopw -forever -shared -bg -rfbport 5900
websockify --web=/opt/novnc 6080 localhost:5900
