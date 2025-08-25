FROM node:20-bullseye-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99

WORKDIR /app

# System dependencies for virtual desktop/devices, browser runtime, and native modules
RUN apt-get update && apt-get install -y \
    curl wget unzip git gnupg2 \
    xvfb x11vnc fluxbox xdotool wmctrl xterm dbus dbus-x11 websockify \
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
COPY desktop.sh /app/desktop.sh
RUN chmod +x /app/desktop.sh

# Agent service install
WORKDIR /app/agent-service
COPY agent-service/package*.json ./
RUN npm ci --foreground-scripts
COPY agent-service/src ./src

# Install Playwright browsers and rebuild native modules (sharp)
RUN npx playwright install --with-deps chromium && \
    npm rebuild sharp --verbose || true

WORKDIR /app

# Startup script to run desktop and agent-service
COPY desktop/startup.sh /usr/local/bin/startup.sh
RUN chmod +x /usr/local/bin/startup.sh

EXPOSE 3000 6080 5900
CMD ["/usr/local/bin/startup.sh"]
