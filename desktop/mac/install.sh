#!/usr/bin/env bash
set -euo pipefail

# Install websockify via Python pip (assumes Python is installed)
if command -v python3 >/dev/null 2>&1; then
PY=python3
elif command -v python >/dev/null 2>&1; then
PY=python
else
echo "Error: Python is not installed (python3/python)." >&2
exit 1
fi

echo "Installing/Upgrading websockify via pip..."
if ! $PY -m pip install --upgrade --user websockify 2>/dev/null; then
  sudo $PY -m pip install --upgrade websockify
fi

# Ensure websockify is reachable
if ! command -v websockify >/dev/null 2>&1; then
USER_BASE=$($PY -m site --user-base 2>/dev/null || echo "")
if [ -n "$USER_BASE" ] && [ -x "$USER_BASE/bin/websockify" ]; then
  echo "websockify installed at $USER_BASE/bin/websockify. Consider adding it to your PATH."
else
  echo "Warning: websockify not found on PATH. Ensure your pip bin directory is in PATH." >&2
fi
fi

# Fetch noVNC via git into /opt/novnc (assumes git is installed)
echo "Fetching noVNC into /opt/novnc..."
sudo mkdir -p /opt
if [ -d "/opt/novnc/.git" ]; then
  sudo git -C /opt/novnc fetch --depth=1 origin || true
  # Try common default branches
  if sudo git -C /opt/novnc rev-parse --verify origin/master >/dev/null 2>&1; then
    sudo git -C /opt/novnc reset --hard origin/master
  elif sudo git -C /opt/novnc rev-parse --verify origin/main >/dev/null 2>&1; then
    sudo git -C /opt/novnc reset --hard origin/main
  fi
else
  sudo rm -rf /opt/novnc
  sudo git clone --depth 1 https://github.com/novnc/noVNC.git /opt/novnc
fi

sudo chmod -R a+rX /opt/novnc

echo "mac install: websockify (pip) and noVNC (git) installed. Web root at /opt/novnc."

# Install Node-based tooling and agent-service dependencies (assumes Node/npm installed)
if ! command -v npm >/dev/null 2>&1; then
  echo "Error: npm not found on PATH. Please install Node.js (includes npm)." >&2
  exit 1
fi

echo "Installing global TypeScript tooling (ts-node, typescript)..."
npm install -g ts-node typescript

echo "Installing agent-service npm dependencies..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
pushd "$PROJECT_ROOT/agent-service" >/dev/null
if [ -f package-lock.json ]; then
  npm ci
  npx playwright@1.52.0 install --with-deps chromium
else
  npm install
  npx playwright@1.52.0 install --with-deps chromium
fi
popd >/dev/null
