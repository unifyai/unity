# Start X11 virtual display
Xvfb :99 -screen 0 1920x1080x16 &

# Wait for Xvfb to initialize
sleep 2

# Start window manager
fluxbox &

# Start VNC server
x11vnc -display :99 -nopw -forever -shared -bg -rfbport 5900

# Start noVNC websockify proxy (this will block and keep the container running)
websockify --web=/opt/novnc 6080 localhost:5900
