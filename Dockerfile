# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Accept build argument for UNIFY_KEY
ARG UNIFY_KEY
ARG GITHUB_TOKEN

# Set working directory
WORKDIR /app

# Virtual devices and remote browser setup
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99

# Install all system dependencies from shared script (full production set)
COPY scripts/install-system-deps.sh /tmp/
RUN chmod +x /tmp/install-system-deps.sh && /tmp/install-system-deps.sh && rm /tmp/install-system-deps.sh

# Set locale environment
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# noVNC static files
RUN mkdir -p /opt/novnc && \
    wget https://github.com/novnc/noVNC/archive/refs/heads/master.zip && \
    unzip master.zip && \
    mv noVNC-master/* /opt/novnc && \
    rm -rf master.zip noVNC-master
COPY desktop/novnc/custom.html /opt/novnc/custom.html

# Dependencies for virtual camera (currently disabled)
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

# Install uv using official installation script
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Configure git to use GITHUB_TOKEN for private repo authentication
RUN git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"

# Install PyTorch CPU-only first (smaller and faster for containers)
RUN uv pip install --system --no-cache torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies using uv (system-wide, no virtual environment)
RUN uv pip install --system --no-cache .

# Remove git credentials from config after install (security best practice)
RUN git config --global --unset url."https://${GITHUB_TOKEN}@github.com/".insteadOf

# Copy all application files
COPY . /app

# Ensure desktop scripts are executable
RUN chmod +x /app/desktop/desktop.sh /app/desktop/display.sh /app/desktop/device.sh /app/desktop/update_vnc_password.sh /app/desktop/startup.sh /app/entrypoint.sh || true

# Build agent-service
WORKDIR /app/agent-service
RUN npm ci
WORKDIR /app

# Build codesandbox-service
WORKDIR /app/codesandbox-service
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
RUN python unity/conversation_manager/medium_scripts/call.py download-files
RUN playwright install

# Set runtime environment variables for memory optimization
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

# Expose the ports that the applications use
EXPOSE 8000 6379 6080

# Use Tini as init system to handle signals properly
ENTRYPOINT ["/usr/bin/tini", "--"]

# Use bash to run the startup script
CMD ["/bin/bash", "/app/entrypoint.sh"]
