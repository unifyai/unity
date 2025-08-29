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

# Install global TypeScript runner and Node deps for agent-service
cd agent-service

# Prefer clean, lockfile-resolved install if lockfile exists
npm install -g ts-node typescript
if [ -f package-lock.json ]; then
  npm ci
else
  npm install
fi

# Add cloudflare gpg key
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null

# Add this repo to your apt repositories
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' | sudo tee /etc/apt/sources.list.d/cloudflared.list

# install cloudflared
sudo apt-get update && sudo apt-get install cloudflared
