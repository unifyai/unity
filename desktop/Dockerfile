FROM node:20-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99

WORKDIR /app

# System dependencies for virtual desktop/devices, browser runtime, and native modules
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

# Desktop script
COPY desktop/ /app
RUN chmod +x /app/desktop.sh
RUN chmod -x /app/device.sh
RUN chmod -x /app/startup.sh

# Agent service install
WORKDIR /app/agent-service
COPY agent-service/ ./
RUN npm ci

# Install Playwright browsers and rebuild native modules (sharp)
RUN npx playwright@1.52.0 install --with-deps chromium

WORKDIR /app

# # Install helper tools
# RUN install -m 0755 scripts/sandbox-dpkg /usr/local/bin/sandbox-dpkg

# Startup script to run desktop and agent-service
EXPOSE 3000 6080 5900
CMD ["/bin/bash", "/app/startup.sh"]
