# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Avoid tzdata interactive prompt
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl wget unzip gnupg2 xvfb x11vnc fluxbox \
    libnss3 libatk-bridge2.0-0 libgtk-3-0 libxss1 \
    libasound2 libxshmfence1 libxcomposite1 libxdamage1 \
    libxrandr2 libgbm1 libx11-xcb1 fonts-liberation xdg-utils \
    git ca-certificates \
    alsa-utils \
    kmod \
    v4l2loopback-dkms \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the demo application code
COPY . .

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

ENV OPENAI_API_KEY="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"

# Auto-load kernel modules (only works in privileged containers)
# RUN echo "snd-aloop" >> /etc/modules && \
#     echo "v4l2loopback" >> /etc/modules && \
#     echo "options v4l2loopback devices=1 video_nr=10 card_label=VirtualCam exclusive_caps=1" > /etc/modprobe.d/v4l2loopback.conf

# modprobe snd-aloop && \
#   modprobe v4l2loopback devices=1 video_nr=10 card_label=VirtualCam exclusive_caps=1 && \

CMD bash -c "\
  Xvfb :99 -screen 0 1920x1080x16 & \
  sleep 2 && \
  fluxbox & \
  x11vnc -display :99 -nopw -forever -shared -bg -rfbport 5900 && \
  websockify --web=/opt/novnc 6080 localhost:5900 & \
  python browseruse.py"
