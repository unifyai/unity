#!/usr/bin/env bash
set -euo pipefail

# Install runtime dependencies used by linux.sh (x11vnc and websockify) and tools to fetch noVNC
apt-get update
apt-get install -y \
  x11vnc \
  websockify \
  wget \
  unzip \
  curl \
  ca-certificates \
  gnupg

mkdir -p /opt/novnc && \
    wget https://github.com/novnc/noVNC/archive/refs/heads/master.zip && \
    unzip master.zip && \
    mv noVNC-master/* /opt/novnc && \
    rm -rf master.zip noVNC-master

# Install Node.js 22.x (NodeSource) and project dependencies
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs

# Install Node deps for agent-service
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT/agent-service"

# Prefer clean, lockfile-resolved install if lockfile exists
if [ -f package-lock.json ]; then
  npm ci
else
  npm install
fi
