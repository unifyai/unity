FROM node:22-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99

WORKDIR /app

# System dependencies for virtual desktop/devices, web runtime, and native modules
RUN apt-get update && apt-get install -y \
    curl wget unzip git gnupg2 \
    xvfb x11vnc fluxbox xdotool wmctrl xterm dbus dbus-x11 websockify \
    xdg-desktop-portal xdg-desktop-portal-gtk \
    libnss3 libatk-bridge2.0-0 libgtk-3-0 libxss1 \
    libasound2 libxshmfence1 libxcomposite1 libxdamage1 \
    libxrandr2 libgbm1 libx11-xcb1 fonts-liberation xdg-utils \
    ffmpeg ca-certificates \
    pipewire pipewire-pulse wireplumber pulseaudio-utils alsa-utils \
    fuse3 libfuse2 squashfs-tools \
    build-essential python3 pkg-config libvips \
    && rm -rf /var/lib/apt/lists/*

# noVNC static files
RUN mkdir -p /opt/novnc && \
    wget https://github.com/novnc/noVNC/archive/refs/heads/master.zip && \
    unzip master.zip && \
    mv noVNC-master/* /opt/novnc && \
    rm -rf master.zip noVNC-master
COPY desktop/novnc/custom.html /opt/novnc/custom.html

# Desktop script
COPY desktop/ /app/desktop/
RUN chmod +x /app/desktop/desktop.sh
RUN chmod -x /app/desktop/device.sh
RUN chmod -x /app/desktop/startup.sh

# Magnitude packages (agent-service depends on magnitude-core + magnitude-extract
# via file: references that npm resolves as symlinks).
COPY magnitude/packages/magnitude-core /app/magnitude/packages/magnitude-core
COPY magnitude/packages/magnitude-extract /app/magnitude/packages/magnitude-extract
# Install runtime dependencies for each magnitude package directly, bypassing the
# workspace build system. This ensures platform-specific native modules (e.g.
# @boundaryml/baml) are resolved for Linux.
WORKDIR /app/magnitude/packages/magnitude-core
RUN npm install --ignore-scripts --omit=dev
WORKDIR /app/magnitude/packages/magnitude-extract
RUN npm install --ignore-scripts --omit=dev

# Agent service install
WORKDIR /app/agent-service
COPY agent-service/ ./
RUN npm ci

# Install browser runtimes using magnitude-core's patchright (Playwright fork)
# so the browser build matches what magnitude-core expects at runtime.
RUN cd /app/magnitude/packages/magnitude-core && npx patchright install --with-deps chromium

WORKDIR /app

# Install helper tools
COPY scripts/sandbox-dpkg .
RUN install -m 0755 sandbox-dpkg /usr/local/bin/sandbox-dpkg

# Startup script to run desktop and agent-service
EXPOSE 3000 6080 5900
CMD ["/bin/bash", "/app/desktop/startup.sh"]
