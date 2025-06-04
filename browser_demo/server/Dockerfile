# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99


# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl wget unzip gnupg2 \
    # xdg-desktop-portal xdg-desktop-portal-wlr weston wayvnc \
    xvfb x11vnc fluxbox \
    libnss3 libatk-bridge2.0-0 libgtk-3-0 libxss1 \
    libasound2 libxshmfence1 libxcomposite1 libxdamage1 \
    libxrandr2 libgbm1 libx11-xcb1 fonts-liberation xdg-utils \
    ffmpeg git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Dependencies for virtual audio
RUN apt update && apt install -y \
    pipewire pipewire-audio pipewire-bin pipewire-pulse wireplumber \
    libpipewire-0.3-modules libportaudio2 portaudio19-dev \
    pulseaudio-utils alsa-utils alsa-oss alsa-tools \
    dbus dbus-x11 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Dependencies for virtual camera
RUN apt-get update && apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-pipewire \
    gstreamer1.0-libcamera \
    python3-gi \
    gir1.2-gtk-3.0 \
    libgirepository1.0-dev \
    libcairo2-dev \
    pkg-config \
    build-essential \
    cmake \
    libspa-0.2-modules \
    libcamera0.0.3 \
    libcamera-tools \
    gir1.2-gst-plugins-base-1.0 \
    gir1.2-gstreamer-1.0 \
    && rm -rf /var/lib/apt/lists/*

# Install Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    fonts-liberation fonts-noto-color-emoji fonts-noto-core fonts-noto-ui-core fonts-freefont-ttf \
    libx11-xcb1 libxcomposite1 libxcursor1 libxdamage1 libxi6 libxtst6 libnss3 libxrandr2 libasound2 \
    libpangocairo-1.0-0 libatk1.0-0 libcups2 libdrm2 libgbm1 libxshmfence1

# Download noVNC
RUN mkdir -p /opt/novnc && \
    wget https://github.com/novnc/noVNC/archive/refs/heads/master.zip && \
    unzip master.zip && \
    mv noVNC-master/* /opt/novnc && \
    rm -rf master.zip noVNC-master

# Copy the demo application code
COPY . .

# Start everything
RUN chmod +x device.sh

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["/device.sh"]
