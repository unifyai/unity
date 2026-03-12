FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99

WORKDIR /app

# Node.js 22 via NodeSource (matches prod VM)
RUN apt-get update && apt-get install -y curl gnupg2 ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g npm@latest && \
    rm -rf /var/lib/apt/lists/*

# System dependencies for virtual desktop/devices, web runtime, and native modules
RUN apt-get update && apt-get install -y \
    curl wget git \
    tigervnc-standalone-server tigervnc-common \
    xfce4 xfce4-goodies xfce4-terminal \
    xdotool wmctrl xterm dbus dbus-x11 \
    x11-xserver-utils \
    xdg-desktop-portal xdg-desktop-portal-gtk \
    libnss3 libatk-bridge2.0-0 libgtk-3-0 libxss1 \
    libasound2 libxshmfence1 libxcomposite1 libxdamage1 \
    libxrandr2 libgbm1 libx11-xcb1 fonts-liberation xdg-utils \
    ffmpeg \
    pipewire pipewire-pulse wireplumber pulseaudio-utils alsa-utils \
    fuse3 libfuse2 squashfs-tools \
    build-essential python3 python3-pip pkg-config libvips-dev \
    supervisor \
    && pip3 install --no-cache-dir websockify \
    && rm -rf /var/lib/apt/lists/*

# noVNC static files
RUN git clone --depth 1 https://github.com/novnc/noVNC.git /opt/novnc && \
    rm -rf /opt/novnc/.git
COPY desktop/novnc/custom.html /opt/novnc/custom.html

# Desktop scripts + supervisord config
COPY desktop/ /app/desktop/
RUN chmod +x /app/desktop/desktop.sh /app/desktop/device.sh /app/desktop/startup.sh

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

# Demo sites (pure HTML/JS, served by agent-service on demand)
COPY demo-sites/ /app/demo-sites/

# Agent service install
WORKDIR /app/agent-service
COPY agent-service/ ./
RUN npm ci

# Install browser runtimes using magnitude-core's patchright (Playwright fork)
# so the browser build matches what magnitude-core expects at runtime.
RUN cd /app/magnitude/packages/magnitude-core && npx patchright install --with-deps chromium

# Configure Chromium as default browser so GUI apps and XFCE can find it
RUN CHROME_PATH=$(find /root/.cache/ms-playwright -name "chrome" -type f -executable 2>/dev/null | head -1) && \
    printf '#!/bin/bash\nexec "%s" --no-sandbox "$@"\n' "$CHROME_PATH" > /usr/local/bin/chromium-browser && \
    chmod +x /usr/local/bin/chromium-browser && \
    printf '[Desktop Entry]\nVersion=1.0\nName=Chromium Web Browser\nExec=/usr/local/bin/chromium-browser %%U\nTerminal=false\nType=Application\nIcon=web-browser\nCategories=Network;WebBrowser;\nMimeType=text/html;text/xml;application/xhtml+xml;x-scheme-handler/http;x-scheme-handler/https;\nStartupNotify=true\n' \
        > /usr/share/applications/chromium-browser.desktop && \
    update-alternatives --install /usr/bin/x-www-browser x-www-browser /usr/local/bin/chromium-browser 100 && \
    update-alternatives --set x-www-browser /usr/local/bin/chromium-browser && \
    update-desktop-database /usr/share/applications/ 2>/dev/null || true && \
    xdg-mime default chromium-browser.desktop x-scheme-handler/http 2>/dev/null || true && \
    xdg-mime default chromium-browser.desktop x-scheme-handler/https 2>/dev/null || true && \
    xdg-mime default chromium-browser.desktop text/html 2>/dev/null || true

# XFCE helper definition so exo-open recognizes Chromium as a browser option
RUN mkdir -p /usr/share/xfce4/helpers && \
    printf '[Desktop Entry]\nVersion=1.0\nType=X-XFCE-Helper\nName=Chromium Web Browser\nIcon=web-browser\nX-XFCE-Binaries=chromium-browser;/usr/local/bin/chromium-browser;\nX-XFCE-Category=WebBrowser\nX-XFCE-Commands=%%B;\nX-XFCE-CommandsWithParameter=%%B %%s;\n' \
        > /usr/share/xfce4/helpers/chromium-browser.desktop

# XFCE configuration: suppress first-run dialogs, disable screensaver/power mgmt
RUN mkdir -p /root/.config/xfce4/xfconf/xfce-perchannel-xml && \
    printf '<?xml version="1.0" encoding="UTF-8"?>\n<channel name="xfce4-power-manager" version="1.0">\n  <property name="xfce4-power-manager" type="empty">\n    <property name="dpms-enabled" type="bool" value="false"/>\n    <property name="blank-on-ac" type="int" value="0"/>\n    <property name="dpms-on-ac-sleep" type="uint" value="0"/>\n    <property name="dpms-on-ac-off" type="uint" value="0"/>\n  </property>\n</channel>\n' \
        > /root/.config/xfce4/xfconf/xfce-perchannel-xml/xfce4-power-manager.xml && \
    printf '<?xml version="1.0" encoding="UTF-8"?>\n<channel name="xfce4-screensaver" version="1.0">\n  <property name="saver" type="empty">\n    <property name="enabled" type="bool" value="false"/>\n  </property>\n  <property name="lock" type="empty">\n    <property name="enabled" type="bool" value="false"/>\n  </property>\n</channel>\n' \
        > /root/.config/xfce4/xfconf/xfce-perchannel-xml/xfce4-screensaver.xml && \
    printf 'WebBrowser=chromium-browser\nTerminalEmulator=xfce4-terminal\n' \
        > /root/.config/xfce4/helpers.rc

WORKDIR /app

# Prepare supervisor and magnitude log directories
RUN mkdir -p /var/log/supervisor /var/log/magnitude

# Install helper tools
COPY scripts/sandbox-dpkg .
RUN install -m 0755 sandbox-dpkg /usr/local/bin/sandbox-dpkg

# Startup script to run desktop and agent-service via supervisord
EXPOSE 3000 6080 5900
CMD ["/bin/bash", "/app/desktop/startup.sh"]
