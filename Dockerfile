# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Accept build argument for UNIFY_KEY
ARG UNIFY_KEY

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install system dependencies including tini and redis
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    portaudio19-dev \
    python3-dev \
    pkg-config \
    tini \
    git \
    redis-server \
    && rm -rf /var/lib/apt/lists/*


# Virtual devices and remote browser setup
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99

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

# Dependencies for virtual camera
# RUN apt-get update && apt-get install -y \
#     gstreamer1.0-tools \
#     gstreamer1.0-plugins-base \
#     gstreamer1.0-plugins-good \
#     gstreamer1.0-plugins-bad \
#     gstreamer1.0-plugins-ugly \
#     gstreamer1.0-libav \
#     gstreamer1.0-pipewire \
#     gstreamer1.0-libcamera \
#     python3-gi \
#     gir1.2-gtk-3.0 \
#     libgirepository1.0-dev \
#     libcairo2-dev \
#     pkg-config \
#     build-essential \
#     cmake \
#     v4l-utils \
#     libspa-0.2-modules \
#     libcamera-tools \
#     gir1.2-gst-plugins-base-1.0 \
#     gir1.2-gstreamer-1.0 \
#     && rm -rf /var/lib/apt/lists/*

# Install Node.js & npm for agent-service
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs

# Copy requirements file
COPY requirements.txt .

# Install PyTorch CPU-only first (smaller and faster for containers)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . /app

# After the other apt-get installs in Dockerfile
RUN apt-get update && apt-get install -y \
    libgtk-4-1 \
    libharfbuzz-icu0 \
    libenchant-2-2 \
    libsecret-1-0 \
    libhyphen0 \
    libmanette-0.2-0 \
    && rm -rf /var/lib/apt/lists/*

# Build agent-service
WORKDIR /app/agent-service
RUN npm ci
WORKDIR /app


# Set environment variables
ENV PYTHONPATH=/app
ENV UNIFY_KEY=${UNIFY_KEY}
RUN install -m 0755 /app/scripts/sandbox-dpkg /usr/local/bin/sandbox-dpkg

# Download the turn detector model files
# Set memory-efficient environment variables for model loading
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
RUN python unity/conversation_manager/call.py download-files
RUN playwright install

# Set runtime environment variables for memory optimization
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

# Expose the ports that the applications use
EXPOSE 8000 6379

# Use Tini as init system to handle signals properly
ENTRYPOINT ["/usr/bin/tini", "--"]

# Use bash to run the startup script
CMD ["/bin/bash", "/app/entrypoint.sh"]
